from __future__ import annotations

from collections import Counter
from typing import Any

from ..db import StateLedger
from ..extractors import build_extract_recovery_plan
from .ingest import create_ingest_plan


def build_progress_report(
    ledger: StateLedger,
    *,
    include_ingest_plan: bool = True,
    recent_limit: int = 10,
) -> dict[str, Any]:
    """Summarize long-running build progress from the local state ledger.

    This report is read-only. It does not refresh the Zotero shadow, submit
    MinerU jobs, or call embedding providers; it only aggregates persisted
    control-plane state so interrupted builds can be inspected before resume.
    """

    jobs = ledger.list_jobs(limit=None)
    attachments = ledger.list_attachments(limit=None)
    extract_jobs = ledger.list_extract_jobs(limit=None)
    batches = ledger.list_embedding_batches(limit=None)
    vector_indexes = ledger.list_vector_indexes()
    report = {
        "state": ledger.status_summary(),
        "jobs": summarize_jobs(jobs, recent_limit=recent_limit),
        "library": summarize_library(attachments),
        "extract": summarize_extract_jobs(extract_jobs, recent_limit=recent_limit),
        "extract_recovery": build_extract_recovery_plan(extract_jobs)["summary"],
        "normalize": summarize_normalized(ledger),
        "embedding": summarize_embedding(batches, vector_indexes, recent_limit=recent_limit),
        "eta": {"available": False, "reason": "workers_do_not_record_timing_estimates_yet"},
    }
    if include_ingest_plan:
        try:
            report["ingest_plan"] = create_ingest_plan(ledger)["summary"]
        except Exception as exc:
            report["ingest_plan"] = {"available": False, "error": str(exc)}
    return report


def summarize_jobs(jobs: list[dict[str, Any]], *, recent_limit: int) -> dict[str, Any]:
    by_kind_status = Counter(f"{job['kind']}:{job['status']}" for job in jobs)
    by_status = Counter(job["status"] for job in jobs)
    return {
        "total": len(jobs),
        "by_status": dict(sorted(by_status.items())),
        "by_kind_status": dict(sorted(by_kind_status.items())),
        "recent": jobs[:recent_limit],
    }


def summarize_library(attachments: list[dict[str, Any]]) -> dict[str, Any]:
    by_classification = Counter(item["classification"] for item in attachments)
    buildable = by_classification.get("included_auto", 0) + by_classification.get("included_manual", 0)
    return {
        "attachment_count": len(attachments),
        "buildable_count": buildable,
        "review_count": by_classification.get("needs_review", 0),
        "blocked_missing_file_count": by_classification.get("missing_file", 0),
        "metadata_only_count": by_classification.get("orphan_metadata_only", 0),
        "by_classification": dict(sorted(by_classification.items())),
    }


def summarize_extract_jobs(extract_jobs: list[dict[str, Any]], *, recent_limit: int) -> dict[str, Any]:
    by_state = Counter(job["state"] for job in extract_jobs)
    retryable = sum(1 for job in extract_jobs if job["state"] == "failed_retryable")
    manual_review = sum(1 for job in extract_jobs if job["state"] == "failed_manual_review")
    return {
        "job_count": len(extract_jobs),
        "by_state": dict(sorted(by_state.items())),
        "retryable_count": retryable,
        "manual_review_count": manual_review,
        "recent": extract_jobs[:recent_limit],
    }


def summarize_normalized(ledger: StateLedger) -> dict[str, Any]:
    artifacts = ledger.list_normalized_artifacts(limit=None)
    chunks = ledger.status_summary().get("chunks", {})
    return {
        "artifact_count": len(artifacts),
        "chunk_counts": chunks,
        "text_chunk_count": chunks.get("text", 0),
        "image_chunk_count": chunks.get("image", 0),
    }


def summarize_embedding(
    batches: list[dict[str, Any]],
    vector_indexes: list[dict[str, Any]],
    *,
    recent_limit: int,
) -> dict[str, Any]:
    by_status = Counter(batch["status"] for batch in batches)
    by_profile_status = Counter(f"{batch['profile_name']}:{batch['status']}" for batch in batches)
    return {
        "batch_count": len(batches),
        "batches_by_status": dict(sorted(by_status.items())),
        "batches_by_profile_status": dict(sorted(by_profile_status.items())),
        "vector_indexes": vector_indexes,
        "indexed_chunk_count": sum(int(index["chunk_count"]) for index in vector_indexes if index.get("active")),
        "recent_batches": batches[:recent_limit],
    }
