from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from ..db import JobEvent, StateLedger


IngestMode = Literal["incremental", "full"]
BUILDABLE_CLASSIFICATIONS = {"included_auto", "included_manual"}
DONE_EXTRACT_STATES = {"downloaded", "normalized", "completed"}


def create_ingest_plan(
    ledger: StateLedger,
    *,
    mode: IngestMode = "incremental",
    zotero_key: str | None = None,
    include_multimodal: bool = True,
) -> dict[str, Any]:
    """Build a resumable ingest plan from the local state ledger.

    This is deliberately non-executing. It does not call MinerU, does not call
    embedding providers, and does not read Zotero's live database. The plan is
    the durable contract that later workers will consume stage by stage.
    """

    if mode not in {"incremental", "full"}:
        raise ValueError("ingest mode must be 'incremental' or 'full'")

    attachments = select_candidate_attachments(ledger, zotero_key=zotero_key)
    artifacts_by_attachment = {
        artifact.get("attachment_key"): artifact
        for artifact in ledger.list_normalized_artifacts(limit=None)
        if artifact.get("attachment_key")
    }
    extract_jobs_by_attachment = group_extract_jobs_by_attachment(ledger)
    text_profile = default_profile_name(ledger, mode="text")
    multimodal_profile = default_profile_name(ledger, mode="multimodal") if include_multimodal else None

    documents = [
        plan_attachment(
            ledger,
            attachment=attachment,
            artifact=artifacts_by_attachment.get(attachment["attachment_key"]),
            extract_jobs=extract_jobs_by_attachment.get(attachment["attachment_key"], []),
            text_profile=text_profile,
            multimodal_profile=multimodal_profile,
            mode=mode,
        )
        for attachment in attachments
    ]
    summary = summarize_plan(documents)
    return {
        "mode": mode,
        "zotero_key": zotero_key,
        "include_multimodal": include_multimodal,
        "text_profile": text_profile,
        "multimodal_profile": multimodal_profile,
        "summary": summary,
        "documents": documents,
    }


def start_ingest_job(
    ledger: StateLedger,
    *,
    mode: IngestMode = "incremental",
    zotero_key: str | None = None,
    include_multimodal: bool = True,
    execute: bool = False,
) -> dict[str, Any]:
    """Create a persisted ingest job and checkpoint its document plan.

    Execution is intentionally disabled until real MinerU and qwen workers are
    added. This prevents accidental expensive API calls during control-plane
    testing while still making progress/resume state durable.
    """

    if execute:
        raise NotImplementedError("ingest execution workers are not implemented; start with execute=false")
    plan = create_ingest_plan(
        ledger,
        mode=mode,
        zotero_key=zotero_key,
        include_multimodal=include_multimodal,
    )
    job_id = ledger.create_job(
        "ingest",
        {
            "mode": mode,
            "zotero_key": zotero_key,
            "include_multimodal": include_multimodal,
            "execute": False,
            "summary": plan["summary"],
        },
    )
    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="plan",
            status="completed",
            message="ingest plan created without executing external providers",
            payload=plan["summary"],
        )
    )
    for document in plan["documents"]:
        ledger.checkpoint(
            document["attachment_key"],
            "ingest_plan",
            "planned",
            {"job_id": job_id, "document": document},
        )
    ledger.set_job_status(job_id, "planned")
    return {"job": ledger.get_job(job_id, include_events=True), "plan": plan}


def pause_ingest_job(ledger: StateLedger, job_id: str, *, reason: str = "manual pause") -> dict[str, Any]:
    return transition_ingest_job(ledger, job_id, status="paused", stage="pause", message=reason)


def resume_ingest_job(ledger: StateLedger, job_id: str, *, reason: str = "manual resume") -> dict[str, Any]:
    job = require_ingest_job(ledger, job_id)
    payload = dict(job.get("payload") or {})
    plan = create_ingest_plan(
        ledger,
        mode=payload.get("mode", "incremental"),
        zotero_key=payload.get("zotero_key"),
        include_multimodal=bool(payload.get("include_multimodal", True)),
    )
    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="resume",
            status="planned",
            message=reason,
            payload=plan["summary"],
        )
    )
    for document in plan["documents"]:
        ledger.checkpoint(
            document["attachment_key"],
            "ingest_plan",
            "planned",
            {"job_id": job_id, "document": document},
        )
    ledger.set_job_status(job_id, "planned")
    return {"job": ledger.get_job(job_id, include_events=True), "plan": plan}


def cancel_ingest_job(ledger: StateLedger, job_id: str, *, reason: str = "manual cancel") -> dict[str, Any]:
    return transition_ingest_job(ledger, job_id, status="cancelled", stage="cancel", message=reason)


def select_candidate_attachments(ledger: StateLedger, *, zotero_key: str | None) -> list[dict[str, Any]]:
    attachments = [
        item
        for item in ledger.list_attachments(limit=None)
        if item["classification"] in BUILDABLE_CLASSIFICATIONS
    ]
    if zotero_key is not None:
        attachments = [
            item
            for item in attachments
            if item["attachment_key"] == zotero_key or item.get("parent_key") == zotero_key
        ]
    return attachments


def plan_attachment(
    ledger: StateLedger,
    *,
    attachment: dict[str, Any],
    artifact: dict[str, Any] | None,
    extract_jobs: list[dict[str, Any]],
    text_profile: str | None,
    multimodal_profile: str | None,
    mode: IngestMode,
) -> dict[str, Any]:
    document_id = artifact["document_id"] if artifact is not None else attachment["attachment_key"]
    full_rebuild = mode == "full"
    extract_stage = stage_for_extract(attachment, extract_jobs, force=full_rebuild, artifact=artifact)
    normalize_stage = stage_for_artifact(artifact, force=full_rebuild)
    text_stage = stage_for_embedding(
        ledger,
        document_id=document_id,
        artifact=artifact,
        profile_name=text_profile,
        force=full_rebuild,
    )
    multimodal_stage = stage_for_embedding(
        ledger,
        document_id=document_id,
        artifact=artifact,
        profile_name=multimodal_profile,
        force=full_rebuild,
    )
    stages = [extract_stage, normalize_stage, text_stage]
    if multimodal_profile is not None:
        stages.append(multimodal_stage)
    return {
        "attachment_key": attachment["attachment_key"],
        "parent_key": attachment.get("parent_key"),
        "document_id": document_id,
        "title": attachment.get("title"),
        "classification": attachment["classification"],
        "source_quality": attachment.get("source_quality"),
        "file_exists": attachment.get("file_exists"),
        "scan_status": attachment.get("scan_status"),
        "normalized": artifact is not None,
        "stages": stages,
        "next_stage": next_stage_name(stages),
    }


def stage_for_extract(
    attachment: dict[str, Any],
    extract_jobs: list[dict[str, Any]],
    *,
    force: bool,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not attachment.get("file_exists"):
        return {"stage": "extract", "status": "blocked", "reason": "missing_file"}
    done = next((job for job in extract_jobs if job["state"] in DONE_EXTRACT_STATES), None)
    if done is not None and not force:
        return {"stage": "extract", "status": "done", "job_id": done["job_id"], "state": done["state"]}
    if artifact is not None and not force:
        return {"stage": "extract", "status": "done", "reason": "normalized_artifact_exists"}
    return {"stage": "extract", "status": "pending", "reason": "full_rebuild" if force and done else "no_cached_extract"}


def stage_for_artifact(artifact: dict[str, Any] | None, *, force: bool) -> dict[str, Any]:
    if artifact is not None and not force:
        return {
            "stage": "normalize",
            "status": "done",
            "document_id": artifact["document_id"],
            "chunk_count": artifact["chunk_count"],
            "image_count": artifact["image_count"],
        }
    return {"stage": "normalize", "status": "pending", "reason": "full_rebuild" if force and artifact else "no_artifact"}


def stage_for_embedding(
    ledger: StateLedger,
    *,
    document_id: str,
    artifact: dict[str, Any] | None,
    profile_name: str | None,
    force: bool,
) -> dict[str, Any]:
    if profile_name is None:
        return {"stage": "embed", "status": "skipped", "reason": "no_default_profile"}
    stage_name = f"embed:{profile_name}"
    if artifact is None:
        return {"stage": stage_name, "status": "blocked", "reason": "not_normalized"}
    checkpoint = ledger.get_checkpoint(document_id, stage_name)
    if checkpoint is not None and checkpoint["status"] == "indexed" and not force:
        return {"stage": stage_name, "status": "done", "checkpoint": checkpoint}
    return {"stage": stage_name, "status": "pending", "reason": "full_rebuild" if force and checkpoint else "not_indexed"}


def summarize_plan(documents: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts: Counter[str] = Counter()
    next_stage_counts: Counter[str] = Counter()
    for document in documents:
        next_stage_counts[document["next_stage"]] += 1
        for stage in document["stages"]:
            stage_counts[f"{stage['stage']}:{stage['status']}"] += 1
    return {
        "document_count": len(documents),
        "next_stages": dict(sorted(next_stage_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
    }


def next_stage_name(stages: list[dict[str, Any]]) -> str:
    for stage in stages:
        if stage["status"] in {"blocked", "pending"}:
            return f"{stage['stage']}:{stage['status']}"
    return "complete"


def group_extract_jobs_by_attachment(ledger: StateLedger) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in ledger.list_extract_jobs(limit=None):
        attachment_key = job.get("attachment_key")
        if attachment_key:
            grouped.setdefault(attachment_key, []).append(job)
    return grouped


def default_profile_name(ledger: StateLedger, *, mode: Literal["text", "multimodal"]) -> str | None:
    flag = "default_for_text" if mode == "text" else "default_for_multimodal"
    expected_modality = "text" if mode == "text" else "multimodal"
    for profile in ledger.list_embedding_profiles():
        if profile["enabled"] and profile["modality"] == expected_modality and profile[flag]:
            return str(profile["name"])
    return None


def transition_ingest_job(
    ledger: StateLedger,
    job_id: str,
    *,
    status: str,
    stage: str,
    message: str,
) -> dict[str, Any]:
    require_ingest_job(ledger, job_id)
    ledger.add_event(JobEvent(job_id=job_id, stage=stage, status=status, message=message))
    ledger.set_job_status(job_id, status)
    job = ledger.get_job(job_id, include_events=True)
    return {"job": job}


def require_ingest_job(ledger: StateLedger, job_id: str) -> dict[str, Any]:
    job = ledger.get_job(job_id, include_events=False)
    if job is None:
        raise KeyError(f"ingest job not found: {job_id}")
    if job["kind"] != "ingest":
        raise ValueError(f"job {job_id} has kind {job['kind']}, expected ingest")
    return job
