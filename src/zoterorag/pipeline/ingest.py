from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import traceback
from typing import Any, Literal

from ..db import JobEvent, StateLedger
from ..embeddings.profile import embedding_profile_hash
from ..embeddings.indexer import index_normalized_document
from ..normalize.markdown import normalize_markdown_document
from ..search.results import RerankNotSupportedError


IngestMode = Literal["incremental", "full"]
BUILDABLE_CLASSIFICATIONS = {"included_auto", "included_manual"}
DONE_EXTRACT_STATES = {"downloaded", "normalized"}


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
    text_profile = default_profile(ledger, mode="text")
    multimodal_profile = default_profile(ledger, mode="multimodal") if include_multimodal else None

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
        "text_profile": text_profile["name"] if text_profile else None,
        "multimodal_profile": multimodal_profile["name"] if multimodal_profile else None,
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
    # --- execution dependencies (required when execute=True) ---
    extract_manager: Any = None,
    extract_cache_dir: str | Path | None = None,
    normalized_dir: str | Path | None = None,
    vector_store_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create a persisted ingest job and optionally execute it.

    When execute=False (the default), only the plan is created and persisted.
    This is safe for control-plane testing and diagnostics.

    When execute=True, the function processes documents stage by stage:
    extract → normalize → embed(text) → embed(multimodal). Each stage
    checks checkpoints before re-executing so interrupted runs can resume.
    """

    if execute:
        if extract_manager is None or extract_cache_dir is None or normalized_dir is None or vector_store_dir is None:
            raise ValueError(
                "extract_manager, extract_cache_dir, normalized_dir, and vector_store_dir "
                "are required when execute=True"
            )

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
            "execute": execute,
            "summary": plan["summary"],
        },
    )
    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="plan",
            status="completed",
            message="ingest plan created"
            if not execute
            else "ingest plan created; executing stages",
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

    if not execute:
        ledger.set_job_status(job_id, "planned")
        return {"job": ledger.get_job(job_id, include_events=True), "plan": plan}

    # --- execution path ---
    ledger.set_job_status(job_id, "running")
    extract_cache_dir = Path(extract_cache_dir)
    normalized_dir = Path(normalized_dir)
    vector_store_dir = Path(vector_store_dir)

    executed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    total = len(plan["documents"])
    _progress(f"\n=== ingest {total} documents ===\n")

    for idx, document in enumerate(plan["documents"], start=1):
        if document["next_stage"] == "complete":
            skipped.append(document)
            continue

        title = str(document.get("title") or document["document_id"])[:80]
        try:
            _progress(f"[{idx}/{total}] {title}")
            result = _execute_document_stages(
                ledger=ledger,
                job_id=job_id,
                document=document,
                extract_manager=extract_manager,
                extract_cache_dir=extract_cache_dir,
                normalized_dir=normalized_dir,
                vector_store_dir=vector_store_dir,
            )
            executed.append(result)
            _progress("  -> done\n")
        except (FileNotFoundError, ValueError, RuntimeError, RerankNotSupportedError) as exc:
            failed.append({"document": document, "error": str(exc)})
            _progress(f"  -> FAILED: {exc}\n")
            ledger.add_event(
                JobEvent(
                    job_id=job_id,
                    stage="document_error",
                    status="failed",
                    message=f"{document['document_id']}: {exc}",
                    payload={"document": document, "error": str(exc)},
                )
            )
        except Exception as exc:
            # Catch-all: a single misbehaving document must not abort the job.
            error_text = traceback.format_exc()
            failed.append({"document": document, "error": str(exc), "traceback": error_text})
            _progress(f"  -> FAILED (unexpected): {exc}\n")
            ledger.add_event(
                JobEvent(
                    job_id=job_id,
                    stage="document_error",
                    status="failed",
                    message=f"{document['document_id']}: unexpected {exc.__class__.__name__}: {exc}",
                    payload={"document": document, "error": str(exc), "traceback": error_text},
                )
            )

    final_status = "completed" if not failed else "completed_with_errors"
    _progress(f"\n=== {final_status}: {len(executed)} ok, {len(skipped)} skipped, {len(failed)} failed ===\n")
    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="complete",
            status=final_status,
            message=f"indexed {len(executed)} documents; skipped {len(skipped)}; failed {len(failed)}",
            payload={"indexed": len(executed), "skipped": len(skipped), "failed": len(failed)},
        )
    )
    ledger.set_job_status(job_id, final_status)
    return {
        "job": ledger.get_job(job_id, include_events=True),
        "plan": plan,
        "executed": executed,
        "skipped": skipped,
        "failed": failed,
    }


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
    text_profile: dict[str, Any] | None,
    multimodal_profile: dict[str, Any] | None,
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
    completed_job = next((job for job in extract_jobs if job["state"] == "completed"), None)
    if completed_job is not None and not force:
        # ``completed`` means MinerU finished remotely, but the local artifact may
        # not have been downloaded yet. The execution path will resume the download
        # unless the caller explicitly requested a full rebuild.
        return {
            "stage": "extract",
            "status": "pending",
            "reason": "completed_artifact_not_downloaded",
            "job_id": completed_job["job_id"],
        }
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
    profile_name: dict[str, Any] | str | None,
    force: bool,
) -> dict[str, Any]:
    if profile_name is None:
        return {"stage": "embed", "status": "skipped", "reason": "no_default_profile"}
    profile = profile_name if isinstance(profile_name, dict) else profile_by_name(ledger, profile_name)
    if profile is None:
        return {"stage": f"embed:{profile_name}", "status": "skipped", "reason": "profile_not_found"}
    current_hash = embedding_profile_hash(profile)
    stage_name = f"embed:{profile['name']}"
    if artifact is None:
        return {"stage": stage_name, "status": "blocked", "reason": "not_normalized", "profile_hash": current_hash}
    checkpoint = ledger.get_checkpoint(document_id, stage_name)
    if checkpoint is not None and checkpoint["status"] == "indexed" and not force:
        checkpoint_hash = checkpoint.get("payload", {}).get("profile_hash")
        if checkpoint_hash == current_hash:
            return {
                "stage": stage_name,
                "status": "done",
                "checkpoint": checkpoint,
                "profile_hash": current_hash,
            }
        reason = "missing_profile_hash" if checkpoint_hash is None else "profile_changed"
        return {
            "stage": stage_name,
            "status": "pending",
            "reason": reason,
            "profile_hash": current_hash,
            "checkpoint_profile_hash": checkpoint_hash,
        }
    return {
        "stage": stage_name,
        "status": "pending",
        "reason": "full_rebuild" if force and checkpoint else "not_indexed",
        "profile_hash": current_hash,
    }


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
    profile = default_profile(ledger, mode=mode)
    return str(profile["name"]) if profile else None


def default_profile(ledger: StateLedger, *, mode: Literal["text", "multimodal"]) -> dict[str, Any] | None:
    flag = "default_for_text" if mode == "text" else "default_for_multimodal"
    expected_modality = "text" if mode == "text" else "multimodal"
    for profile in ledger.list_embedding_profiles():
        if profile["enabled"] and profile["modality"] == expected_modality and profile[flag]:
            return profile
    return None


def profile_by_name(ledger: StateLedger, profile_name: str) -> dict[str, Any] | None:
    for profile in ledger.list_embedding_profiles():
        if profile["name"] == profile_name:
            return profile
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


# ---------------------------------------------------------------------------
# Execution helpers — called by start_ingest_job when execute=True
# ---------------------------------------------------------------------------


def _progress(msg: str) -> None:
    """Write a human-readable progress line to stderr."""
    print(msg, file=sys.stderr, flush=True)


def _execute_document_stages(
    *,
    ledger: StateLedger,
    job_id: str,
    document: dict[str, Any],
    extract_manager: Any,
    extract_cache_dir: Path,
    normalized_dir: Path,
    vector_store_dir: Path,
) -> dict[str, Any]:
    """Execute pending stages for a single document, resuming from checkpoints."""

    attachment_key = document["attachment_key"]
    document_id = document["document_id"]

    for stage in document["stages"]:
        stage_name = stage["stage"]
        if stage["status"] not in {"pending", "blocked"}:
            continue

        if stage_name == "extract":
            _execute_extract_stage(
                ledger=ledger,
                job_id=job_id,
                attachment_key=attachment_key,
                document=document,
                extract_manager=extract_manager,
                extract_cache_dir=extract_cache_dir,
            )
        elif stage_name == "normalize":
            _execute_normalize_stage(
                ledger=ledger,
                job_id=job_id,
                attachment_key=attachment_key,
                document_id=document_id,
                document=document,
                normalized_dir=normalized_dir,
            )
        elif stage_name.startswith("embed:"):
            profile_name = stage_name[len("embed:"):]
            _execute_embed_stage(
                ledger=ledger,
                job_id=job_id,
                document_id=document_id,
                profile_name=profile_name,
                vector_store_dir=vector_store_dir,
            )

    return {
        "attachment_key": attachment_key,
        "document_id": document_id,
        "title": document.get("title"),
    }


def _execute_extract_stage(
    *,
    ledger: StateLedger,
    job_id: str,
    attachment_key: str,
    document: dict[str, Any],
    extract_manager: Any,
    extract_cache_dir: Path,
) -> None:
    """Run MinerU extraction for one attachment."""

    from ..extractors.manager import ExtractionRequest

    attachment = ledger.get_attachment(attachment_key)
    if attachment is None:
        raise ValueError(f"attachment not found in ledger: {attachment_key}")

    file_path = attachment.get("file_path")
    if not file_path or not Path(file_path).is_file():
        raise FileNotFoundError(f"attachment file not found: {file_path}")

    ledger.checkpoint(
        attachment_key,
        "extract",
        "running",
        {"job_id": job_id, "input_file": str(file_path)},
    )

    _progress("  extract: submitting...")
    request = ExtractionRequest(
        input_file=Path(file_path),
        attachment_key=attachment_key,
    )
    result = extract_manager.ensure_extraction(request)

    # ``completed`` means the remote batch finished but the local artifact has
    # not been downloaded yet. Resume the download instead of leaving downstream
    # normalize with no artifact on disk.
    if result.job["state"] == "completed" and result.job.get("local_stage") != "downloaded":
        _progress(f"  extract: resuming download for completed job {result.job['job_id']}")
        result = extract_manager.resume_extraction(result.job["job_id"])

    _progress(f"  extract: {result.job['state']} (cache_hit={result.cache_hit})")

    ledger.checkpoint(
        attachment_key,
        "extract",
        "done",
        {
            "job_id": job_id,
            "extract_job_id": result.job["job_id"],
            "state": result.job["state"],
            "cache_hit": result.cache_hit,
        },
    )

    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="extract",
            status="completed" if result.job["state"] in DONE_EXTRACT_STATES else result.job["state"],
            message=f"extract job {result.job['job_id']} state={result.job['state']}",
            payload={"job_id": result.job["job_id"], "state": result.job["state"], "cache_hit": result.cache_hit},
        )
    )


def _execute_normalize_stage(
    *,
    ledger: StateLedger,
    job_id: str,
    attachment_key: str,
    document_id: str,
    document: dict[str, Any],
    normalized_dir: Path,
) -> None:
    """Run offline markdown normalization for one document."""

    # Find the extract job that has a downloaded artifact
    extract_jobs = [
        j for j in ledger.list_extract_jobs(limit=None)
        if j.get("attachment_key") == attachment_key and j["state"] in DONE_EXTRACT_STATES
    ]
    if not extract_jobs:
        raise ValueError(f"no completed extract job for {attachment_key}; run extraction first")

    extract_job = extract_jobs[0]
    artifact_dir = extract_job.get("artifact_dir") or extract_job.get("extract_dir")
    if not artifact_dir:
        raise ValueError(f"extract job {extract_job['job_id']} has no artifact_dir")

    artifact_path = Path(artifact_dir)
    source_md = _find_source_markdown(artifact_path)
    if source_md is None:
        raise FileNotFoundError(f"no markdown file found in extract artifact: {artifact_path}")

    ledger.checkpoint(
        document_id,
        "normalize",
        "running",
        {"job_id": job_id, "extract_job_id": extract_job["job_id"]},
    )

    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="normalize",
            status="running",
            message=f"normalizing {document_id}",
        )
    )

    _progress(f"  normalize: {document_id}...")
    normalize_result = normalize_markdown_document(
        source_markdown=source_md,
        output_root=normalized_dir,
        document_id=document_id,
        attachment_key=attachment_key,
        extract_job_id=extract_job["job_id"],
    )
    _progress(f"  normalize: {normalize_result.ledger_artifact()['chunk_count']} chunks, {normalize_result.ledger_artifact()['image_count']} images")

    # Persist the normalized artifact and its chunks in a single transaction so
    # an interruption cannot leave an artifact without chunks or vice versa.
    ledger.upsert_artifact_and_chunks(
        normalize_result.ledger_artifact(),
        normalize_result.chunks,
    )
    ledger.checkpoint(
        document_id,
        "normalize",
        "normalized",
        {"chunk_count": normalize_result.ledger_artifact()["chunk_count"]},
    )

    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="normalize",
            status="completed",
            message=f"normalized {document_id}: {normalize_result.ledger_artifact()['chunk_count']} chunks",
            payload=normalize_result.ledger_artifact(),
        )
    )


def _execute_embed_stage(
    *,
    ledger: StateLedger,
    job_id: str,
    document_id: str,
    profile_name: str,
    vector_store_dir: Path,
) -> None:
    """Run embedding for one document under one profile."""

    _progress(f"  embed:{profile_name}: {document_id}...")
    ledger.checkpoint(
        document_id,
        f"embed:{profile_name}",
        "running",
        {"job_id": job_id},
    )
    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage=f"embed:{profile_name}",
            status="running",
            message=f"embedding {document_id} with {profile_name}",
        )
    )

    index_result = index_normalized_document(
        ledger=ledger,
        vector_store_dir=vector_store_dir,
        profile_name=profile_name,
        document_id=document_id,
        # Provider is auto-resolved from profile by _resolve_embedding_provider
    )
    _progress(f"  embed:{profile_name}: {index_result.indexed_chunks} chunks (reused={index_result.reused_existing})")

    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage=f"embed:{profile_name}",
            status="completed",
            message=f"indexed {index_result.indexed_chunks} chunks for {document_id}",
            payload=index_result.to_dict(),
        )
    )


def _find_source_markdown(artifact_dir: Path) -> Path | None:
    """Find the main markdown file in a MinerU extraction artifact directory.

    MinerU typically produces a file like ``full.md`` or ``<name>.md`` in the
    top level of the extracted archive. This helper locates it.
    """
    candidates = list(artifact_dir.glob("*.md"))
    if not candidates:
        # Check one level deep (some archives have a root folder)
        for child in artifact_dir.iterdir():
            if child.is_dir():
                candidates.extend(child.glob("*.md"))
    if not candidates:
        return None
    # Prefer full.md, then the largest .md file
    for candidate in candidates:
        if candidate.name.lower() in ("full.md", "output.md"):
            return candidate
    return max(candidates, key=lambda p: p.stat().st_size)
