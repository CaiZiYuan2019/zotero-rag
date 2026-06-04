from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    ) -> None:
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.provider = provider or StubExtractorProvider()
        self.key_pool = key_pool or ExtractorKeyPool()

    def ensure_extraction(self, request: ExtractionRequest) -> ExtractionResult:
        input_file = Path(request.input_file)
        pdf_sha256 = request.pdf_sha256 or sha256_file(input_file)
        selected_pages = canonical_selected_pages(request.selected_pages)
        options = dict(request.options or {})
        if "timeout_seconds" not in options:
            options["timeout_seconds"] = recommended_mineru_timeout_seconds(request.selected_page_count)
        options_hash = stable_options_hash(options)
        cache_key = extractor_cache_key(
            pdf_sha256=pdf_sha256,
            selected_pages=selected_pages,
            extractor_name=self.provider.name,
            extractor_version=self.provider.version,
            options_hash=options_hash,
        )

        existing = self.ledger.get_extract_job_by_cache_key(cache_key)
        if existing is not None and existing["state"] in REUSABLE_EXTRACT_STATES:
            return ExtractionResult(job=existing, cache_hit=True)

        api_key = self.key_pool.next_key()
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
            submitted = self.provider.submit(input_file, options_hash)
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="running" if submitted.state == "running" else submitted.state,
                local_stage="poll",
                external_job_id=submitted.external_job_id,
                last_poll_at=utc_now(),
            )

            polled = self.provider.poll(submitted.external_job_id)
            self.ledger.set_extract_job_state(
                job["job_id"],
                state=polled.state,
                local_stage="download" if polled.state == "completed" else "poll",
                external_job_id=polled.external_job_id,
                last_poll_at=utc_now(),
            )

            if polled.state != "completed":
                return ExtractionResult(job=self.ledger.get_extract_job(job_id=job["job_id"]) or job, cache_hit=False)

            artifact_dir = self.cache_dir / cache_key
            artifact = self.provider.download(polled.external_job_id, artifact_dir)
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
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="failed_retryable",
                local_stage="error",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
