from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
import uuid

from ..db import StateLedger
from ..db.state import utc_now
from .base import ExtractorProvider, StubExtractorProvider
from .cache import (
    canonical_selected_pages,
    extractor_cache_key,
    recommended_mineru_timeout_seconds,
    sha256_file,
    stable_options_hash,
)
from .mineru import MinerUAPIError
from .recovery import classify_extract_job
from .key_pool import ExtractorKeyPool


REUSABLE_EXTRACT_STATES = {"submitted", "running", "completed", "downloaded"}


@dataclass(frozen=True)
class ExtractionRequest:
    input_file: Path
    attachment_key: str | None = None
    pdf_sha256: str | None = None
    selected_pages: str = ""
    selected_page_count: int = 1
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExtractionResult:
    job: dict[str, Any]
    cache_hit: bool


class ExtractionManager:
    """Coordinates extraction state, cache keys, and provider calls.

    This class is deliberately small and synchronous for the first milestone.
    Later MinerU workers can call the same ledger methods from a single writer
    queue while submit/poll/download happen in worker threads or processes.
    """

    def __init__(
        self,
        *,
        ledger: StateLedger,
        cache_dir: str | Path,
        provider: ExtractorProvider | None = None,
        key_pool: ExtractorKeyPool | None = None,
        failure_cooldown_seconds: float = 60.0,
        poll_interval_seconds: float = 30.0,
        poll_timeout_seconds: float = 21600.0,
    ) -> None:
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.provider = provider or StubExtractorProvider()
        self.key_pool = key_pool or ExtractorKeyPool()
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.poll_timeout_seconds = poll_timeout_seconds

    def ensure_extraction(self, request: ExtractionRequest) -> ExtractionResult:
        input_file = Path(request.input_file)
        pdf_sha256 = request.pdf_sha256 or sha256_file(input_file)
        selected_pages = canonical_selected_pages(request.selected_pages)
        options = dict(request.options or {})
        if "timeout_seconds" not in options:
            options["timeout_seconds"] = recommended_mineru_timeout_seconds(request.selected_page_count)
        provider_options = dict(options)
        provider_options.setdefault("page_ranges", selected_pages)
        options_hash = stable_options_hash(options)
        endpoint_url = getattr(self.provider, "apply_upload_url", "") or getattr(self.provider, "endpoint", "")
        cache_key = extractor_cache_key(
            pdf_sha256=pdf_sha256,
            selected_pages=selected_pages,
            extractor_name=self.provider.name,
            extractor_version=self.provider.version,
            options_hash=options_hash,
            endpoint_url=endpoint_url,
        )

        existing = self.ledger.get_extract_job_by_cache_key(cache_key)
        if existing is not None and existing["state"] in REUSABLE_EXTRACT_STATES:
            return ExtractionResult(job=existing, cache_hit=True)

        api_key = self.key_pool.acquire_key()
        if api_key is None and self.key_pool.has_keys():
            raise RuntimeError("no extractor API key is currently available; all keys are busy or cooling down")
        api_key_alias = api_key.alias if api_key is not None else f"{self.provider.name}_stub"
        job_id = existing["job_id"] if existing is not None else str(uuid.uuid4())
        now = utc_now()
        job = self.ledger.upsert_extract_job(
            {
                "job_id": job_id,
                "attachment_key": request.attachment_key,
                "pdf_sha256": pdf_sha256,
                "selected_pages": selected_pages,
                "cache_key": cache_key,
                "provider": self.provider.name,
                "provider_version": self.provider.version,
                "options_hash": options_hash,
                "api_key_alias": api_key_alias,
                "state": "submitted",
                "local_stage": "submitted",
                "submitted_at": now,
                "payload": {
                    "input_file": str(input_file),
                    "options": options,
                    "key_alias": api_key_alias,
                },
            }
        )

        try:
            submitted = self.provider.submit(
                input_file,
                options_hash,
                options=provider_options,
                api_key=api_key.secret if api_key is not None else None,
            )
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="running" if submitted.state == "running" else submitted.state,
                local_stage="poll",
                external_job_id=submitted.external_job_id,
                last_poll_at=utc_now(),
            )

            # MinerU processing is async: submit returns immediately, and the
            # batch runs remotely for minutes. Poll in a loop until the job
            # completes, fails, or the timeout expires. The poll interval and
            # timeout are configurable on the ExtractionManager.
            poll_deadline = time.monotonic() + self.poll_timeout_seconds
            polled = self.provider.poll(
                submitted.external_job_id,
                api_key=api_key.secret if api_key is not None else None,
                options=options,
            )
            while polled.state not in ("completed", "failed"):
                if time.monotonic() >= poll_deadline:
                    raise MinerUAPIError(
                        f"MinerU batch {submitted.external_job_id} did not complete within "
                        f"{self.poll_timeout_seconds:.0f}s",
                        stage="poll-timeout",
                        batch_id=submitted.external_job_id,
                    )
                time.sleep(self.poll_interval_seconds)
                polled = self.provider.poll(
                    submitted.external_job_id,
                    api_key=api_key.secret if api_key is not None else None,
                    options=options,
                )
                self.ledger.set_extract_job_state(
                    job["job_id"],
                    state=polled.state,
                    local_stage="download" if polled.state == "completed" else "poll",
                    external_job_id=polled.external_job_id,
                    last_poll_at=utc_now(),
                )

            if polled.state == "failed":
                self.ledger.set_extract_job_state(
                    job["job_id"],
                    state="failed_retryable",
                    local_stage="error",
                    error_code="MinerUAPIError",
                    error_message=polled.message or "MinerU batch failed",
                )
                raise MinerUAPIError(
                    polled.message or "MinerU batch failed",
                    stage="extract",
                    batch_id=submitted.external_job_id,
                )

            # polled.state == "completed"
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="completed",
                local_stage="download",
                external_job_id=polled.external_job_id,
                last_poll_at=utc_now(),
            )

            artifact_dir = self.cache_dir / cache_key
            artifact = self.provider.download(
                polled.external_job_id,
                artifact_dir,
                api_key=api_key.secret if api_key is not None else None,
                source_pdf=input_file,
                options=options,
            )
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="downloaded",
                local_stage="downloaded",
                artifact_dir=artifact.artifact_dir,
                extract_dir=artifact.artifact_dir,
                manifest_path=artifact.manifest_path,
                payload={
                    "input_file": str(input_file),
                    "options": options,
                    "key_alias": api_key_alias,
                    "source_pdf": str(artifact.source_pdf),
                },
            )
            final_job = self.ledger.get_extract_job(job_id=job["job_id"]) or job
            return ExtractionResult(job=final_job, cache_hit=False)
        except Exception as exc:
            if api_key is not None:
                self.key_pool.mark_key_cooldown(
                    api_key.alias,
                    cooldown_seconds=self.failure_cooldown_seconds,
                )
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="failed_retryable",
                local_stage="error",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
        finally:
            if api_key is not None:
                self.key_pool.release_key(api_key.alias)

    def resume_extraction(self, job_id: str) -> ExtractionResult:
        """Resume one persisted extraction job from its recorded local stage.

        This method is explicit execution, not a status check. It never runs
        during diagnostics or recovery planning. For remote providers, callers
        must construct the manager with the intended provider/key pool first.
        """

        job = self.ledger.get_extract_job(job_id=job_id)
        if job is None:
            raise KeyError(f"extract job not found: {job_id}")
        recovery = classify_extract_job(job)
        if recovery.action == "skip":
            return ExtractionResult(job=job, cache_hit=True)
        if recovery.action == "manual_review":
            raise RuntimeError(f"extract job {job_id} requires manual review: {recovery.reason}")

        api_key = self.key_pool.acquire_key()
        if api_key is None and self.key_pool.has_keys():
            raise RuntimeError("no extractor API key is currently available; all keys are busy or cooling down")
        try:
            if recovery.action == "poll":
                return self._resume_poll(job, api_key_secret=api_key.secret if api_key else None)
            if recovery.action == "download":
                return self._resume_download(job, api_key_secret=api_key.secret if api_key else None)
            if recovery.action == "submit":
                return self._resume_submit(job, api_key_secret=api_key.secret if api_key else None)
            raise RuntimeError(f"unsupported extract recovery action: {recovery.action}")
        except Exception as exc:
            if api_key is not None:
                self.key_pool.mark_key_cooldown(
                    api_key.alias,
                    cooldown_seconds=self.failure_cooldown_seconds,
                )
            self.ledger.set_extract_job_state(
                job_id,
                state="failed_retryable",
                local_stage="error",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
        finally:
            if api_key is not None:
                self.key_pool.release_key(api_key.alias)

    def _resume_submit(self, job: dict[str, Any], *, api_key_secret: str | None) -> ExtractionResult:
        payload = job.get("payload") or {}
        input_file_text = payload.get("input_file")
        if not input_file_text:
            raise RuntimeError("cannot resubmit extract job without payload.input_file")
        input_file = Path(input_file_text)
        if not input_file.is_file():
            raise FileNotFoundError(f"extract input file no longer exists: {input_file}")
        options = dict(payload.get("options") or {})
        provider_options = dict(options)
        provider_options.setdefault("page_ranges", job.get("selected_pages") or "")
        submitted = self.provider.submit(
            input_file,
            str(job["options_hash"]),
            options=provider_options,
            api_key=api_key_secret,
        )
        self.ledger.set_extract_job_state(
            job["job_id"],
            state="running" if submitted.state == "running" else submitted.state,
            local_stage="poll",
            external_job_id=submitted.external_job_id,
            last_poll_at=utc_now(),
        )
        refreshed = self.ledger.get_extract_job(job_id=job["job_id"]) or job
        if submitted.state == "completed":
            return self._resume_download(refreshed, api_key_secret=api_key_secret)
        return ExtractionResult(job=refreshed, cache_hit=False)

    def _resume_poll(self, job: dict[str, Any], *, api_key_secret: str | None) -> ExtractionResult:
        external_job_id = job.get("external_job_id")
        if not external_job_id:
            raise RuntimeError("cannot poll extract job without external_job_id")
        options = (job.get("payload") or {}).get("options") or {}
        polled = self.provider.poll(str(external_job_id), api_key=api_key_secret, options=options)
        self.ledger.set_extract_job_state(
            job["job_id"],
            state=polled.state,
            local_stage="download" if polled.state == "completed" else "poll",
            external_job_id=polled.external_job_id,
            last_poll_at=utc_now(),
        )
        refreshed = self.ledger.get_extract_job(job_id=job["job_id"]) or job
        if polled.state == "completed":
            return self._resume_download(refreshed, api_key_secret=api_key_secret)
        return ExtractionResult(job=refreshed, cache_hit=False)

    def _resume_download(self, job: dict[str, Any], *, api_key_secret: str | None) -> ExtractionResult:
        external_job_id = job.get("external_job_id")
        if not external_job_id:
            raise RuntimeError("cannot download extract job without external_job_id")
        artifact_dir = self.cache_dir / str(job["cache_key"])
        payload = job.get("payload") or {}
        source_pdf_text = payload.get("source_pdf") or payload.get("input_file")
        source_pdf = Path(source_pdf_text) if source_pdf_text else None
        options = payload.get("options") or {}
        artifact = self.provider.download(
            str(external_job_id),
            artifact_dir,
            api_key=api_key_secret,
            source_pdf=source_pdf,
            options=options,
        )
        payload = {
            **payload,
            "source_pdf": str(artifact.source_pdf),
        }
        self.ledger.set_extract_job_state(
            job["job_id"],
            state="downloaded",
            local_stage="downloaded",
            artifact_dir=artifact.artifact_dir,
            extract_dir=artifact.artifact_dir,
            manifest_path=artifact.manifest_path,
            payload=payload,
        )
        return ExtractionResult(job=self.ledger.get_extract_job(job_id=job["job_id"]) or job, cache_hit=False)
