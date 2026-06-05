from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..db import JobEvent, StateLedger
from ..embeddings import index_normalized_document
from ..embeddings.indexer import require_embedding_profile
from ..embeddings.profile import embedding_profile_hash


def create_reembed_plan(
    ledger: StateLedger,
    *,
    profile_name: str,
    document_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Plan vector-only rebuild work from normalized artifacts.

    The plan never calls embedding providers. It only reads normalized artifact
    and checkpoint state so a long rebuild can resume without repeating MinerU
    extraction or re-indexing chunks already built for the same profile hash.
    """

    profile = require_embedding_profile(ledger, profile_name)
    artifacts = ledger.list_normalized_artifacts(limit=None)
    if document_id is not None:
        artifacts = [artifact for artifact in artifacts if artifact["document_id"] == document_id]
    documents = [
        plan_normalized_document(ledger, artifact=artifact, profile=profile, force=force)
        for artifact in artifacts
    ]
    return {
        "profile_name": profile_name,
        "profile_hash": embedding_profile_hash(profile),
        "document_id": document_id,
        "force": force,
        "from_normalized": True,
        "summary": summarize_reembed_plan(documents),
        "documents": documents,
    }


def start_reembed_job(
    ledger: StateLedger,
    *,
    vector_store_dir: str | Path,
    profile_name: str,
    document_id: str | None = None,
    force: bool = False,
    execute: bool = False,
    allow_stub_provider: bool = False,
) -> dict[str, Any]:
    """Create a vector-only rebuild job and optionally execute local indexing.

    Real qwen/dashscope providers are intentionally not invoked here. Execution
    uses the existing local stub provider unless the profile itself is `stub` or
    the caller explicitly allows stub substitution for control-plane testing.
    """

    profile = require_embedding_profile(ledger, profile_name)
    if execute and profile["provider"] != "stub" and not allow_stub_provider:
        raise NotImplementedError(
            f"embedding provider {profile['provider']} is not implemented for local reembed execution; "
            "rerun with execute=false or allow_stub_provider=true"
        )

    plan = create_reembed_plan(
        ledger,
        profile_name=profile_name,
        document_id=document_id,
        force=force,
    )
    job_id = ledger.create_job(
        "reembed",
        {
            "profile_name": profile_name,
            "document_id": document_id,
            "force": force,
            "execute": execute,
            "from_normalized": True,
            "summary": plan["summary"],
        },
    )
    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="plan",
            status="completed",
            message="reembed plan created from normalized artifacts",
            payload=plan["summary"],
        )
    )
    for document in plan["documents"]:
        ledger.checkpoint(
            document["document_id"],
            f"reembed_plan:{profile_name}",
            "planned",
            {"job_id": job_id, "document": document},
        )

    if not execute:
        ledger.set_job_status(job_id, "planned")
        return {"job": ledger.get_job(job_id, include_events=True), "plan": plan}

    executed = []
    skipped = []
    ledger.set_job_status(job_id, "running")
    for document in plan["documents"]:
        if document["status"] != "pending":
            skipped.append(document)
            ledger.add_event(
                JobEvent(
                    job_id=job_id,
                    stage="embed",
                    status="skipped",
                    message=f"{document['document_id']} is {document['status']}",
                    payload=document,
                )
            )
            continue
        result = index_normalized_document(
            ledger=ledger,
            vector_store_dir=vector_store_dir,
            profile_name=profile_name,
            document_id=document["document_id"],
            allow_stub_provider=allow_stub_provider,
        )
        result_payload = result.to_dict()
        executed.append(result_payload)
        ledger.add_event(
            JobEvent(
                job_id=job_id,
                stage=f"embed:{profile_name}",
                status="indexed",
                message=f"indexed {result.indexed_chunks} chunks for {document['document_id']}",
                payload=result_payload,
            )
        )
    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="complete",
            status="completed",
            message=f"indexed {len(executed)} documents; skipped {len(skipped)}",
            payload={"indexed": len(executed), "skipped": len(skipped)},
        )
    )
    ledger.set_job_status(job_id, "completed")
    return {
        "job": ledger.get_job(job_id, include_events=True),
        "plan": plan,
        "indexed": executed,
        "skipped": skipped,
    }


def plan_normalized_document(
    ledger: StateLedger,
    *,
    artifact: dict[str, Any],
    profile: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    profile_hash = embedding_profile_hash(profile)
    profile_name = profile["name"]
    chunk_type = "text" if profile["modality"] == "text" else "image"
    chunks = ledger.list_chunks(artifact["document_id"], chunk_type=chunk_type)
    stage_name = f"embed:{profile_name}"
    checkpoint = ledger.get_checkpoint(artifact["document_id"], stage_name)
    status = "pending"
    reason = "not_indexed"
    checkpoint_hash = None
    if not chunks:
        status = "blocked"
        reason = f"no_{chunk_type}_chunks"
    elif checkpoint is not None and checkpoint["status"] == "indexed" and not force:
        checkpoint_hash = checkpoint.get("payload", {}).get("profile_hash")
        if checkpoint_hash == profile_hash:
            status = "done"
            reason = "up_to_date"
        else:
            reason = "missing_profile_hash" if checkpoint_hash is None else "profile_changed"
    elif checkpoint is not None and force:
        reason = "full_rebuild"
    return {
        "document_id": artifact["document_id"],
        "attachment_key": artifact.get("attachment_key"),
        "profile_name": profile_name,
        "profile_hash": profile_hash,
        "checkpoint_profile_hash": checkpoint_hash,
        "chunk_type": chunk_type,
        "chunk_count": len(chunks),
        "status": status,
        "reason": reason,
    }


def summarize_reembed_plan(documents: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(document["status"] for document in documents)
    by_reason = Counter(document["reason"] for document in documents)
    return {
        "document_count": len(documents),
        "status_counts": dict(sorted(by_status.items())),
        "reason_counts": dict(sorted(by_reason.items())),
        "pending_count": by_status.get("pending", 0),
        "blocked_count": by_status.get("blocked", 0),
        "done_count": by_status.get("done", 0),
    }
