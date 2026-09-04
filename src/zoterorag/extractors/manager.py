from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
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
from .mineru import MinerUAPIError, PollTimeoutError
from .recovery import classify_extract_job
from .key_pool import ExtractorKeyPool


REUSABLE_EXTRACT_STATES = {"submitted", "running", "completed", "downloaded"}

# M2: per-job poll budgets (pages*6+30, persisted as options.timeout_seconds)
# are clamped into this band. The floor keeps jobs persisted with a tiny
# budget (e.g. an unknown page count falling back to the 1-page estimate of
# 36s) from timing out prematurely; the ceiling is the task lifetime cap so a
# single poll round never outlives the job itself.
MIN_POLL_BUDGET_SECONDS = 1800.0


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
        key_acquire_timeout_seconds: float = 60.0,
        job_max_lifetime_seconds: float = 21600.0,
    ) -> None:
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.provider = provider or StubExtractorProvider()
        self.key_pool = key_pool or ExtractorKeyPool()
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self.poll_interval_seconds = poll_interval_seconds
        # poll_timeout_seconds no longer drives poll deadlines directly; it is
        # the fallback budget for jobs persisted without options.timeout_seconds
        # (see _poll_budget_seconds).
        self.poll_timeout_seconds = poll_timeout_seconds
        self.key_acquire_timeout_seconds = key_acquire_timeout_seconds
        # 决策1: total task lifetime measured from submitted_at. Jobs exceeding
        # it are flipped to failed_manual_review instead of being polled or
        # resubmitted forever.
        self.job_max_lifetime_seconds = job_max_lifetime_seconds

    def _acquire_key_with_wait(self, timeout_seconds: float | None = None) -> ApiKeyRef | None:
        """Acquire an extractor key, waiting until one becomes available.

        Returns ``None`` immediately when the pool is empty (stub/no-key mode).
        Raises no exception here; callers decide whether a key is mandatory.
        """
        if not self.key_pool.has_keys():
            return None
        timeout = timeout_seconds if timeout_seconds is not None else self.key_acquire_timeout_seconds
        deadline = time.monotonic() + timeout
        while True:
            api_key = self.key_pool.acquire_key()
            if api_key is not None:
                return api_key
            wait = min(2.0, deadline - time.monotonic())
            if wait <= 0:
                return None
            time.sleep(wait)

    def _poll_budget_seconds(self, options: Mapping[str, Any]) -> float:
        """Effective poll budget for one round of polling.

        M2: the per-job budget persisted as ``options["timeout_seconds"]``
        (``pages*6+30`` at submission time) is now honored, clamped into
        [MIN_POLL_BUDGET_SECONDS, job_max_lifetime_seconds]. Legacy jobs
        persisted without a budget fall back to ``self.poll_timeout_seconds``.
        """
        raw = options.get("timeout_seconds")
        try:
            per_job = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            per_job = 0.0
        # `not (x > 0)` also rejects NaN, which would otherwise slip past the
        # clamp and turn poll_deadline into NaN — an infinite poll loop.
        if not (per_job > 0):
            per_job = float(self.poll_timeout_seconds)
        return min(max(per_job, MIN_POLL_BUDGET_SECONDS), float(self.job_max_lifetime_seconds))

    def _lifetime_exceeded(self, job: dict[str, Any]) -> bool:
        """Whether the job has outlived job_max_lifetime_seconds since submitted_at.

        Rows without a parseable submitted_at (legacy/hand-built) cannot be
        aged and are treated as not exceeded.
        """
        submitted_at = job.get("submitted_at")
        if not submitted_at:
            return False
        try:
            started = datetime.fromisoformat(str(submitted_at))
        except ValueError:
            return False
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - started).total_seconds()
        return age_seconds > float(self.job_max_lifetime_seconds)

    def _mark_job_max_lifetime(self, job: dict[str, Any]) -> None:
        """Flip a job past its lifetime cap to failed_manual_review (决策1)."""
        self.ledger.set_extract_job_state(
            job["job_id"],
            state="failed_manual_review",
            local_stage=job.get("local_stage") or "poll",
            error_code="job-max-lifetime",
            error_message=(
                f"exceeded job_max_lifetime_seconds={float(self.job_max_lifetime_seconds):.0f} "
                f"since submitted_at={job.get('submitted_at')}; manual review required"
            ),
        )

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
        if existing is not None and self._lifetime_exceeded(existing):
            # 决策1: a non-reusable job past its lifetime cap must not be
            # auto-resubmitted; upsert_extract_job COALESCEs submitted_at, so
            # the clock honestly measures from the first submission.
            self._mark_job_max_lifetime(existing)
            raise RuntimeError(
                f"extract job {existing['job_id']} exceeded "
                f"job_max_lifetime_seconds={float(self.job_max_lifetime_seconds):.0f}; "
                "marked failed_manual_review"
            )

        api_key = self._acquire_key_with_wait()
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
            # completes, fails, or the per-job poll budget (M2) expires.
            poll_budget = self._poll_budget_seconds(options)
            poll_deadline = time.monotonic() + poll_budget
            polled = self.provider.poll(
                submitted.external_job_id,
                api_key=api_key.secret if api_key is not None else None,
                options=options,
            )
            while polled.state not in ("completed", "failed"):
                if time.monotonic() >= poll_deadline:
                    raise PollTimeoutError(
                        f"MinerU batch {submitted.external_job_id} did not complete within "
                        f"{poll_budget:.0f}s; will resume polling the same batch next run",
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
        except PollTimeoutError as exc:
            # M2: a poll timeout is not a failure — the remote batch may still
            # be running. Stand down this round ("本轮摘牌"): keep state=running
            # and the external_job_id, do NOT cool down the key, do NOT
            # resubmit; the next ingest/resume round polls the same batch.
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="running",
                local_stage="poll",
                error_code="poll-timeout",
                error_message=str(exc),
                last_poll_at=utc_now(),
            )
            return ExtractionResult(
                job=self.ledger.get_extract_job(job_id=job["job_id"]) or job,
                cache_hit=False,
            )
        except Exception as exc:
            # Catch-all: any provider failure marks this job failed_retryable and
            # records the error without crashing the caller.
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
        if recovery.action in {"poll", "submit"} and self._lifetime_exceeded(job):
            # 决策1: enforce the total lifetime cap only on actions that would
            # keep remote work alive (poll/resubmit). "skip" (terminal) and
            # "download" (remote batch already completed; downloading is local
            # and free) must not be flipped to manual review.
            self._mark_job_max_lifetime(job)
            raise RuntimeError(
                f"extract job {job_id} exceeded "
                f"job_max_lifetime_seconds={float(self.job_max_lifetime_seconds):.0f}; "
                "marked failed_manual_review"
            )

        api_key = self._acquire_key_with_wait()
        if api_key is None and self.key_pool.has_keys():
            raise RuntimeError("no extractor API key is currently available; all keys are busy or cooling down")
        try:
            if recovery.action == "poll":
                secret = api_key.secret if api_key else None
                if job.get("state") == "failed_retryable":
                    return self._resume_poll_with_fallback(job, api_key_secret=secret)
                return self._resume_poll(job, api_key_secret=secret)
            if recovery.action == "download":
                return self._resume_download(job, api_key_secret=api_key.secret if api_key else None)
            if recovery.action == "submit":
                return self._resume_submit(job, api_key_secret=api_key.secret if api_key else None)
            raise RuntimeError(f"unsupported extract recovery action: {recovery.action}")
        except PollTimeoutError as exc:
            # M2: same "stand down this round" semantics as ensure_extraction —
            # keep state=running + external_job_id, no key cooldown, no
            # resubmit; the next round polls the same remote batch.
            self.ledger.set_extract_job_state(
                job_id,
                state="running",
                local_stage="poll",
                error_code="poll-timeout",
                error_message=str(exc),
                last_poll_at=utc_now(),
            )
            return ExtractionResult(
                job=self.ledger.get_extract_job(job_id=job_id) or job,
                cache_hit=False,
            )
        except Exception as exc:
            # Catch-all: any provider failure marks this job failed_retryable and
            # records the error without crashing the caller.
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
        if submitted.state == "running":
            return self._resume_poll(refreshed, api_key_secret=api_key_secret)
        if submitted.state == "completed":
            return self._resume_download(refreshed, api_key_secret=api_key_secret)
        raise MinerUAPIError(
            "resubmitted extract job failed immediately",
            stage="submit",
            batch_id=submitted.external_job_id,
        )

    def _resume_poll(self, job: dict[str, Any], *, api_key_secret: str | None) -> ExtractionResult:
        external_job_id = job.get("external_job_id")
        if not external_job_id:
            raise RuntimeError("cannot poll extract job without external_job_id")
        options = (job.get("payload") or {}).get("options") or {}

        # M2: honor the per-job budget persisted at submission time
        # (pages*6+30), clamped into [30min, job_max_lifetime_seconds]. The
        # floor keeps jobs persisted with a tiny budget (e.g. an unknown page
        # count falling back to a 1-page estimate) from being failed
        # prematurely on resume, which would trigger a costly full-PDF
        # resubmission; the cap keeps one poll round within the task lifetime.
        poll_timeout_seconds = self._poll_budget_seconds(options)
        poll_deadline = time.monotonic() + poll_timeout_seconds

        polled = self.provider.poll(str(external_job_id), api_key=api_key_secret, options=options)
        while polled.state not in ("completed", "failed"):
            if time.monotonic() >= poll_deadline:
                raise PollTimeoutError(
                    f"MinerU batch {external_job_id} did not complete within "
                    f"{poll_timeout_seconds:.0f}s; will resume polling the same batch next run",
                    batch_id=str(external_job_id),
                )
            time.sleep(self.poll_interval_seconds)
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

        # polled.state == "failed"
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
            batch_id=str(external_job_id),
        )

    def _resume_poll_with_fallback(
        self, job: dict[str, Any], *, api_key_secret: str | None
    ) -> ExtractionResult:
        """Poll a previously failed job; if the remote batch is gone or still
        failed, clear its external id and resubmit the PDF under a fresh job.
        """
        try:
            return self._resume_poll(job, api_key_secret=api_key_secret)
        except PollTimeoutError:
            # M2: a poll timeout must NOT trigger the resubmit fallback — the
            # remote batch may still be running, and resubmitting double-bills
            # the PDF. Propagate to the resume_extraction catch-all, which
            # records error_code="poll-timeout" and stands down this round.
            raise
        except MinerUAPIError:
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="submitted",
                local_stage="submitted",
                external_job_id=None,
                error_code=None,
                error_message=None,
            )
            return self._resume_submit(job, api_key_secret=api_key_secret)

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
