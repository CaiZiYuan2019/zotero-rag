from __future__ import annotations

from collections import Counter
from pathlib import Path
import traceback
from typing import Any

from ..db import JobEvent, StateLedger
from ..embeddings import IndexResult, publish_staged_documents, stage_indexed_document
from ..embeddings.indexer import require_embedding_profile, vector_path_for_profile
from ..embeddings.profile import embedding_profile_hash
from ..index import open_vector_store
from ..search.results import RerankNotSupportedError
from .runner import acquire_runner_lock


def create_reembed_plan(
    ledger: StateLedger,
    *,
    profile_name: str,
    vector_store_dir: str | Path | None = None,
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
        plan_normalized_document(
            ledger,
            artifact=artifact,
            profile=profile,
            force=force,
            vector_store_dir=vector_store_dir,
        )
        for artifact in artifacts
    ]
    return {
        "profile_name": profile_name,
        "profile_hash": embedding_profile_hash(profile),
        "document_id": document_id,
        "vector_store_dir": str(vector_store_dir) if vector_store_dir is not None else None,
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
    embedding_env_path: str | Path = ".env",
) -> dict[str, Any]:
    """Create a vector-only rebuild job and optionally execute indexing.

    When execute=True, each pending document is re-embedded using the configured
    profile. The embedding provider is auto-resolved from the profile and
    environment by ``stage_indexed_document``. Set ``allow_stub_provider=True``
    to force stub embeddings even for non-stub profiles (useful for testing).

    Phase 3: documents are staged one by one and the whole run is published
    once via ``publish_staged_documents`` instead of publishing per document.
    """

    require_embedding_profile(ledger, profile_name)

    plan = create_reembed_plan(
        ledger,
        profile_name=profile_name,
        vector_store_dir=vector_store_dir,
        document_id=document_id,
        force=force,
    )
    payload = {
        "profile_name": profile_name,
        "document_id": document_id,
        "force": force,
        "execute": execute,
        "from_normalized": True,
        "summary": plan["summary"],
    }

    if not execute:
        job_id = ledger.create_job("reembed", payload)
        # M10: mark the plan-only job planned immediately after creation, so
        # the runner-lock sweep (which treats lingering "created" jobs as
        # dead-process leftovers) never mistakes this live creation for one.
        ledger.set_job_status(job_id, "planned")
        _record_reembed_plan(ledger, job_id=job_id, plan=plan, profile_name=profile_name)
        return {"job": ledger.get_job(job_id, include_events=True), "plan": plan}

    # --- execution path ---
    # M10: same shared runner lock as ingest — see start_ingest_job.
    with acquire_runner_lock(ledger):
        job_id = ledger.create_job_if_no_active("reembed", payload)
        _record_reembed_plan(ledger, job_id=job_id, plan=plan, profile_name=profile_name)

        executed = []
        skipped = []
        failed = []
        # Phase 3: staged (document, StagedDocument) pairs awaiting the
        # run-level publish after the loop.
        staged_docs: list[tuple[dict[str, Any], Any]] = []
        ledger.set_job_status(job_id, "running")
        try:
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
                try:
                    outcome = stage_indexed_document(
                        ledger=ledger,
                        vector_store_dir=vector_store_dir,
                        profile_name=profile_name,
                        document_id=document["document_id"],
                        allow_stub_provider=allow_stub_provider,
                        embedding_env_path=embedding_env_path,
                    )
                    if isinstance(outcome, IndexResult):
                        # Reuse fast path: already published; count it now.
                        result_payload = outcome.to_dict()
                        executed.append(result_payload)
                        ledger.add_event(
                            JobEvent(
                                job_id=job_id,
                                stage=f"embed:{profile_name}",
                                status="indexed",
                                message=f"indexed {outcome.indexed_chunks} chunks for {document['document_id']}",
                                payload=result_payload,
                            )
                        )
                    else:
                        # Staged only; executed/event deferred to the publish
                        # below, when the vectors are actually visible.
                        staged_docs.append((document, outcome))
                except (FileNotFoundError, KeyError, ValueError, RuntimeError, RerankNotSupportedError) as exc:
                    failed.append({"document": document, "error": str(exc)})
                    ledger.add_event(
                        JobEvent(
                            job_id=job_id,
                            stage=f"embed:{profile_name}",
                            status="failed",
                            message=f"{document['document_id']}: {exc}",
                            payload={"document": document, "error": str(exc)},
                        )
                    )
                except Exception as exc:
                    # Catch-all: a single misbehaving document must not abort the job.
                    error_text = traceback.format_exc()
                    failed.append({"document": document, "error": str(exc), "traceback": error_text})
                    ledger.add_event(
                        JobEvent(
                            job_id=job_id,
                            stage=f"embed:{profile_name}",
                            status="failed",
                            message=f"{document['document_id']}: unexpected {exc.__class__.__name__}: {exc}",
                            payload={"document": document, "error": str(exc), "traceback": error_text},
                        )
                    )

            # Phase 3: publish every staged document of this run in one go. A
            # publish failure marks all staged documents failed (their batch
            # rows stay "running", so the next run retries them) and the final
            # status naturally lands on completed_with_errors.
            if staged_docs:
                try:
                    publish_results = publish_staged_documents(
                        ledger=ledger,
                        staged=[outcome for _, outcome in staged_docs],
                    )
                except Exception as exc:
                    error_text = traceback.format_exc()
                    for document, _ in staged_docs:
                        failed.append({"document": document, "error": str(exc), "traceback": error_text})
                        ledger.add_event(
                            JobEvent(
                                job_id=job_id,
                                stage=f"embed:{profile_name}",
                                status="failed",
                                message=f"{document['document_id']}: publish failed: {exc}",
                                payload={"document": document, "error": str(exc), "traceback": error_text},
                            )
                        )
                else:
                    for (document, _), result in zip(staged_docs, publish_results):
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

            final_status = "completed" if not failed else "completed_with_errors"
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
                "indexed": executed,
                "skipped": skipped,
                "failed": failed,
            }

        except KeyboardInterrupt:
            ledger.add_event(
                JobEvent(
                    job_id=job_id,
                    stage="complete",
                    status="interrupted",
                    message=f"reembed interrupted by user; {len(executed)} ok, {len(skipped)} skipped, {len(failed)} failed",
                    payload={"indexed": len(executed), "skipped": len(skipped), "failed": len(failed)},
                )
            )
            ledger.set_job_status(job_id, "interrupted")
            raise
        except Exception:
            error_text = traceback.format_exc()
            ledger.add_event(
                JobEvent(
                    job_id=job_id,
                    stage="complete",
                    status="failed",
                    message=f"reembed failed unexpectedly; {len(executed)} ok, {len(skipped)} skipped, {len(failed)} failed",
                    payload={"indexed": len(executed), "skipped": len(skipped), "failed": len(failed), "traceback": error_text},
                )
            )
            ledger.set_job_status(job_id, "failed")
            raise


def _record_reembed_plan(
    ledger: StateLedger, *, job_id: str, plan: dict[str, Any], profile_name: str
) -> None:
    """Persist the plan event and per-document checkpoints for a new job."""
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


def plan_normalized_document(
    ledger: StateLedger,
    *,
    artifact: dict[str, Any],
    profile: dict[str, Any],
    force: bool,
    vector_store_dir: str | Path | None = None,
) -> dict[str, Any]:
    profile_hash = embedding_profile_hash(profile)
    profile_name = profile["name"]
    chunk_type = "text" if profile["modality"] == "text" else "image"
    chunks = ledger.list_chunks(artifact["document_id"], chunk_type=chunk_type)
    stage_name = f"embed:{profile_name}"
    checkpoint = ledger.get_checkpoint(artifact["document_id"], stage_name)
    active_vector = active_document_vector_state(
        profile=profile,
        document_id=artifact["document_id"],
        chunk_type=chunk_type,
        vector_store_dir=vector_store_dir,
    )
    active_vector_chunks = active_vector["chunk_count"]
    active_profile_hashes = active_vector["profile_hashes"]
    active_vectors_match_profile = (
        active_vector_chunks == len(chunks)
        and active_profile_hashes == [profile_hash]
    )
    status = "pending"
    reason = "not_indexed"
    checkpoint_hash = None
    if not chunks:
        status = "blocked"
        reason = f"no_{chunk_type}_chunks"
    elif checkpoint is not None and checkpoint["status"] == "indexed" and not force:
        checkpoint_hash = checkpoint.get("payload", {}).get("profile_hash")
        if checkpoint_hash == profile_hash and active_vectors_match_profile:
            status = "done"
            reason = "up_to_date"
        elif checkpoint_hash == profile_hash:
            reason = "checkpoint_without_active_vectors"
        else:
            reason = "missing_profile_hash" if checkpoint_hash is None else "profile_changed"
    elif active_vectors_match_profile and not force:
        status = "done"
        reason = "active_vectors_present"
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
        "active_vector_chunks": active_vector_chunks,
        "active_vector_profile_hashes": active_profile_hashes,
        "status": status,
        "reason": reason,
    }


def active_document_vector_state(
    *,
    profile: dict[str, Any],
    document_id: str,
    chunk_type: str,
    vector_store_dir: str | Path | None,
) -> dict[str, Any]:
    if vector_store_dir is None:
        return {"chunk_count": None, "profile_hashes": []}
    backend = profile.get("backend", "lancedb")
    vector_path = vector_path_for_profile(
        vector_store_dir, str(profile["name"]), backend=backend
    )
    if backend == "sqlite-local" and not vector_path.is_file():
        return {"chunk_count": None, "profile_hashes": []}
    if backend == "lancedb" and not vector_path.is_dir():
        return {"chunk_count": None, "profile_hashes": []}
    store = open_vector_store(
        vector_path,
        profile_name=str(profile["name"]),
        dimension=int(profile["dimension"]),
        backend=backend,
    )
    try:
        profile_hashes = sorted(
            value
            for value in store.document_metadata_values(document_id, "profile_hash", modality=chunk_type)
            if value
        )
        return {
            "chunk_count": store.document_counts(document_id, modality=chunk_type)["chunks"],
            "profile_hashes": profile_hashes,
        }
    finally:
        store.close()


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
