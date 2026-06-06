from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtractRecoveryItem:
    job_id: str
    attachment_key: str | None
    cache_key: str
    state: str
    local_stage: str
    action: str
    reason: str
    can_resume_without_resubmit: bool
    missing_paths: tuple[str, ...] = ()
    external_job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "attachment_key": self.attachment_key,
            "cache_key": self.cache_key,
            "state": self.state,
            "local_stage": self.local_stage,
            "action": self.action,
            "reason": self.reason,
            "can_resume_without_resubmit": self.can_resume_without_resubmit,
            "missing_paths": list(self.missing_paths),
            "external_job_id": self.external_job_id,
        }


def build_extract_recovery_plan(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    items = [classify_extract_job(job) for job in jobs]
    action_counts = Counter(item.action for item in items)
    resumable_count = sum(1 for item in items if item.can_resume_without_resubmit)
    return {
        "summary": {
            "job_count": len(items),
            "resumable_without_resubmit": resumable_count,
            "by_action": dict(sorted(action_counts.items())),
        },
        "items": [item.to_dict() for item in items],
    }


def classify_extract_job(job: dict[str, Any]) -> ExtractRecoveryItem:
    state = str(job.get("state") or "")
    local_stage = str(job.get("local_stage") or "")
    external_job_id = job.get("external_job_id")
    missing_paths = tuple(missing_declared_paths(job))

    if state in {"downloaded", "normalized"}:
        manifest_path = job.get("manifest_path")
        if manifest_path and path_exists(manifest_path):
            return recovery_item(job, "skip", "artifact_manifest_exists", True, missing_paths=missing_paths)
        if job.get("extract_dir") and path_exists(job["extract_dir"]):
            return recovery_item(job, "normalize", "extract_dir_exists_without_manifest", True, missing_paths=missing_paths)
        if job.get("zip_path") and path_exists(job["zip_path"]):
            return recovery_item(job, "extract_zip", "zip_exists_without_extract_manifest", True, missing_paths=missing_paths)
        return recovery_item(job, "manual_review", "downloaded_state_missing_local_artifacts", False, missing_paths=missing_paths)

    if state == "completed" or local_stage == "download":
        if external_job_id:
            return recovery_item(job, "download", "remote_completed_with_external_job_id", True, missing_paths=missing_paths)
        return recovery_item(job, "manual_review", "completed_state_missing_external_job_id", False, missing_paths=missing_paths)

    if state in {"submitted", "running"} or local_stage == "poll":
        if external_job_id:
            return recovery_item(job, "poll", "external_job_id_recorded", True, missing_paths=missing_paths)
        return recovery_item(job, "submit", "no_external_job_id_recorded", False, missing_paths=missing_paths)

    if state == "failed_retryable":
        if external_job_id:
            return recovery_item(job, "poll", "retryable_failure_with_external_job_id", True, missing_paths=missing_paths)
        return recovery_item(job, "submit", "retryable_failure_before_submit", False, missing_paths=missing_paths)

    if state == "failed_manual_review":
        return recovery_item(job, "manual_review", "manual_review_required", False, missing_paths=missing_paths)

    return recovery_item(job, "manual_review", f"unknown_extract_state:{state or 'empty'}", False, missing_paths=missing_paths)


def recovery_item(
    job: dict[str, Any],
    action: str,
    reason: str,
    can_resume_without_resubmit: bool,
    *,
    missing_paths: tuple[str, ...],
) -> ExtractRecoveryItem:
    return ExtractRecoveryItem(
        job_id=str(job["job_id"]),
        attachment_key=job.get("attachment_key"),
        cache_key=str(job.get("cache_key") or ""),
        state=str(job.get("state") or ""),
        local_stage=str(job.get("local_stage") or ""),
        action=action,
        reason=reason,
        can_resume_without_resubmit=can_resume_without_resubmit,
        missing_paths=missing_paths,
        external_job_id=job.get("external_job_id"),
    )


def missing_declared_paths(job: dict[str, Any]) -> list[str]:
    missing = []
    for field in ("zip_path", "extract_dir", "artifact_dir", "manifest_path"):
        value = job.get(field)
        if value and not path_exists(value):
            missing.append(field)
    return missing


def path_exists(value: str | Path) -> bool:
    return Path(value).exists()
