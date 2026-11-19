from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import traceback
from typing import Any, Literal

from ..db import JobEvent, StateLedger
from ..embeddings.profile import embedding_profile_hash
from ..embeddings.indexer import (
    IndexResult,
    index_normalized_document,
    publish_staged_documents,
    stage_indexed_document,
)
from ..normalize.markdown import normalize_markdown_document, sanitize_id
from ..normalize.merge_extract_segments import (
    merge_extract_segments,
    segment_artifact_dirs_from_jobs,
)
from ..pdf_pages import get_pdf_page_count
from ..search.results import RerankNotSupportedError
from ..extractors.segments import PageSegment, compute_page_segments
from ..extractors.cache import UNKNOWN_PAGE_COUNT_FALLBACK, sha256_file
from ..config.settings import ExtractConfig
from .runner import acquire_runner_lock


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
    extract_config: ExtractConfig | None = None,
    batch_publish: bool = True,
    embedding_env_path: str | Path = ".env",
) -> dict[str, Any]:
    """Create a persisted ingest job and optionally execute it.

    When execute=False (the default), only the plan is created and persisted.
    This is safe for control-plane testing and diagnostics.

    When execute=True, the function processes documents stage by stage:
    extract → normalize → embed(text) → embed(multimodal). Each stage
    checks checkpoints before re-executing so interrupted runs can resume.

    Phase 3: with batch_publish=True (the default), embed stages only stage
    vectors per document and the whole run is published once per profile at
    the end; batch_publish=False falls back to per-document publishing.
    """

    if execute:
        # Dependency validation stays before any lock/ledger mutation so a
        # pure argument error neither touches the lock file nor creates a job.
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
    payload = {
        "mode": mode,
        "zotero_key": zotero_key,
        "include_multimodal": include_multimodal,
        "execute": execute,
        "summary": plan["summary"],
    }

    if not execute:
        job_id = ledger.create_job("ingest", payload)
        # M10: mark the plan-only job planned immediately after creation, so
        # the runner-lock sweep (which treats lingering "created" jobs as
        # dead-process leftovers) never mistakes this live creation for one.
        ledger.set_job_status(job_id, "planned")
        _record_ingest_plan(ledger, job_id=job_id, plan=plan, execute=False)
        return {"job": ledger.get_job(job_id, include_events=True), "plan": plan}

    # --- execution path ---
    # M10: the shared runner lock is held for the whole run. Acquiring it
    # non-blockingly distinguishes a live runner (lock held -> refuse) from
    # dead-process leftovers (lock free -> swept to "interrupted" inside), and
    # create_job_if_no_active closes the check-then-create TOCTOU window.
    with acquire_runner_lock(ledger):
        job_id = ledger.create_job_if_no_active("ingest", payload)
        _record_ingest_plan(ledger, job_id=job_id, plan=plan, execute=True)
        ledger.set_job_status(job_id, "running")
        return _execute_ingest_documents(
            ledger,
            embedding_env_path=embedding_env_path,
            job_id=job_id,
            plan=plan,
            extract_manager=extract_manager,
            extract_cache_dir=extract_cache_dir,
            normalized_dir=normalized_dir,
            vector_store_dir=vector_store_dir,
            extract_config=extract_config,
            batch_publish=batch_publish,
        )


# M9: abort the whole run after this many consecutive failures of the same
# exception class (e.g. a revoked API key failing every document).
CIRCUIT_BREAKER_CONSECUTIVE_FAILURES = 5


def _track_circuit_breaker(
    exc: BaseException, *, kind: str | None, streak: int
) -> tuple[str | None, int, bool]:
    """Update the consecutive-same-class failure streak.

    Returns (kind, streak, tripped). A different exception class resets the
    streak to 1 — only an unbroken run of identical failures indicates a
    systemic problem worth aborting the job for.
    """
    exc_kind = exc.__class__.__name__
    if exc_kind == kind:
        streak += 1
    else:
        kind, streak = exc_kind, 1
    return kind, streak, streak >= CIRCUIT_BREAKER_CONSECUTIVE_FAILURES


def _execute_ingest_documents(
    ledger: StateLedger,
    *,
    job_id: str,
    plan: dict[str, Any],
    extract_manager: Any,
    extract_cache_dir: str | Path,
    normalized_dir: str | Path,
    vector_store_dir: str | Path,
    extract_config: ExtractConfig | None,
    batch_publish: bool = True,
    embedding_env_path: str | Path = ".env",
) -> dict[str, Any]:
    """Shared execution body of start_ingest_job and resume --execute (M7).

    Expects the job to already be in "running" state and the runner lock to
    be held by the caller. Writes the terminal status (completed /
    completed_with_errors / blocked / interrupted / failed) before
    returning/raising.

    Phase 3: with batch_publish=True, embed stages collect StagedDocuments
    into embed_collector and every group is published once per profile after
    the document loop; a document joins ``executed`` only after all its staged
    profiles published successfully.
    """
    extract_cache_dir = Path(extract_cache_dir)
    normalized_dir = Path(normalized_dir)
    vector_store_dir = Path(vector_store_dir)
    if extract_config is None:
        extract_config = ExtractConfig()

    executed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    # Phase 3 batch publish bookkeeping: embed stages append staged vectors to
    # embed_collector instead of publishing per document; documents with
    # staged-but-unpublished embeds wait in awaiting_publish until the
    # run-level publish after the loop confirms them (or fails them).
    embed_collector: list[dict[str, Any]] = []
    awaiting_publish: dict[str, dict[str, Any]] = {}
    failed_doc_ids: set[str] = set()
    # M9 circuit breaker: consecutive failures sharing the same exception
    # class almost always mean a systemic problem (e.g. a revoked API key
    # fails 100% of documents). Abort the run as "failed" instead of
    # burning through every document and reporting completed_with_errors.
    breaker_kind: str | None = None
    breaker_streak = 0
    breaker_tripped = False

    try:
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
                    embedding_env_path=embedding_env_path,
                    ledger=ledger,
                    job_id=job_id,
                    document=document,
                    extract_manager=extract_manager,
                    extract_cache_dir=extract_cache_dir,
                    normalized_dir=normalized_dir,
                    vector_store_dir=vector_store_dir,
                    extract_config=extract_config,
                    embed_collector=embed_collector if batch_publish else None,
                )
                if result.get("blocked"):
                    # Not a failure: remote extraction is still running and
                    # the next run resumes it. Intentionally resets the
                    # breaker streak — a slow remote is not a systemic fault.
                    blocked.append(result)
                    breaker_kind = None
                    breaker_streak = 0
                    _progress("  -> blocked (extract incomplete; will resume next run)\n")
                    ledger.add_event(
                        JobEvent(
                            job_id=job_id,
                            stage="document_blocked",
                            status="blocked",
                            message=f"{document['document_id']}: extract incomplete; will resume next run",
                            payload={"document_id": document["document_id"], "blocked_stage": result.get("blocked_stage")},
                        )
                    )
                else:
                    # Phase 3: with batch publish, embed stages only staged
                    # vectors; the document joins ``executed`` only after the
                    # run-level publish below confirms every staged profile.
                    if result.get("staged_embeds"):
                        awaiting_publish[document["document_id"]] = {
                            "document": document,
                            "result": result,
                        }
                    else:
                        executed.append(result)
                    breaker_kind = None
                    breaker_streak = 0
                    _progress("  -> done\n")
            except (FileNotFoundError, ValueError, RuntimeError, RerankNotSupportedError) as exc:
                failed.append({"document": document, "error": str(exc)})
                failed_doc_ids.add(document["document_id"])
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
                breaker_kind, breaker_streak, breaker_tripped = _track_circuit_breaker(
                    exc, kind=breaker_kind, streak=breaker_streak
                )
                if breaker_tripped:
                    break
            except Exception as exc:
                # Catch-all: a single misbehaving document must not abort the job.
                error_text = traceback.format_exc()
                failed.append({"document": document, "error": str(exc), "traceback": error_text})
                failed_doc_ids.add(document["document_id"])
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
                breaker_kind, breaker_streak, breaker_tripped = _track_circuit_breaker(
                    exc, kind=breaker_kind, streak=breaker_streak
                )
                if breaker_tripped:
                    break

        # Phase 3: batch publish. Runs after the document loop no matter how
        # the loop ended (including a circuit-breaker break), so staged
        # vectors are never abandoned without an accounting outcome. One
        # publish per profile group replaces the historical per-document
        # publish (O(N^2) copying). Per-group failures are isolated: a failed
        # group fails its documents, other groups still publish.
        publish_group_count = 0
        publish_failed_groups = 0
        if embed_collector:
            publish_outcomes = _publish_collected_embeds(
                ledger,
                job_id=job_id,
                embed_collector=embed_collector,
            )
            publish_group_count = publish_outcomes["group_count"]
            publish_failed_groups = publish_outcomes["failed_groups"]
            # Merge awaiting_publish into executed/failed. Documents already
            # counted as failed by the per-document exception path are skipped
            # here so they are never double-counted.
            for document_id, awaiting in awaiting_publish.items():
                if document_id in failed_doc_ids:
                    continue
                entry_outcomes = [
                    outcome
                    for index, outcome in enumerate(publish_outcomes["entries"])
                    if embed_collector[index]["document_id"] == document_id
                ]
                if entry_outcomes and all(entry_outcomes):
                    executed.append(awaiting["result"])
                else:
                    failed_profiles = sorted(
                        embed_collector[index]["profile_name"]
                        for index, outcome in enumerate(publish_outcomes["entries"])
                        if embed_collector[index]["document_id"] == document_id and not outcome
                    )
                    error = f"embed publish failed for profile(s): {', '.join(failed_profiles)}"
                    failed.append({"document": awaiting["document"], "error": error})
                    ledger.add_event(
                        JobEvent(
                            job_id=job_id,
                            stage="document_error",
                            status="failed",
                            message=f"{document_id}: {error}",
                            payload={"document": awaiting["document"], "error": error},
                        )
                    )

        if breaker_tripped:
            final_status = "failed"
            _progress(
                f"\n=== circuit breaker: aborting after {breaker_streak} consecutive "
                f"{breaker_kind} failures ===\n"
            )
            ledger.add_event(
                JobEvent(
                    job_id=job_id,
                    stage="circuit_breaker",
                    status="failed",
                    message=(
                        f"aborting after {breaker_streak} consecutive {breaker_kind} failures; "
                        "likely a systemic problem (check API keys/quota)"
                    ),
                    payload={
                        "error_kind": breaker_kind,
                        "consecutive_failures": breaker_streak,
                        "processed": len(executed) + len(blocked) + len(failed),
                        "remaining": total - len(executed) - len(blocked) - len(failed) - len(skipped),
                    },
                )
            )
        elif embed_collector and publish_group_count and publish_failed_groups == publish_group_count and not executed:
            # Phase 3: every publish group failed and no document completed
            # independently — publishing itself is systemically broken (same
            # spirit as the circuit breaker), so the run is "failed" rather
            # than "completed_with_errors". Partial group failures still land
            # on the failed-driven completed_with_errors path below.
            final_status = "failed"
            ledger.add_event(
                JobEvent(
                    job_id=job_id,
                    stage="embed_publish",
                    status="failed",
                    message=(
                        f"all {publish_group_count} embed publish group(s) failed; "
                        "treating the run as failed (systemic publish fault)"
                    ),
                    payload={"groups": publish_group_count, "failed_groups": publish_failed_groups},
                )
            )
        elif failed:
            final_status = "completed_with_errors"
        elif blocked:
            final_status = "blocked"
        else:
            final_status = "completed"
        _progress(
            f"\n=== {final_status}: {len(executed)} ok, {len(skipped)} skipped, "
            f"{len(blocked)} blocked, {len(failed)} failed ===\n"
        )
        ledger.add_event(
            JobEvent(
                job_id=job_id,
                stage="complete",
                status=final_status,
                message=(
                    f"indexed {len(executed)} documents; skipped {len(skipped)}; "
                    f"blocked {len(blocked)}; failed {len(failed)}"
                ),
                payload={
                    "indexed": len(executed),
                    "skipped": len(skipped),
                    "blocked": len(blocked),
                    "failed": len(failed),
                },
            )
        )
        ledger.set_job_status(job_id, final_status)
        return {
            "job": ledger.get_job(job_id, include_events=True),
            "plan": plan,
            "executed": executed,
            "skipped": skipped,
            "blocked": blocked,
            "failed": failed,
        }

    except KeyboardInterrupt:
        ledger.add_event(
            JobEvent(
                job_id=job_id,
                stage="complete",
                status="interrupted",
                message=f"ingest interrupted by user; {len(executed)} ok, {len(skipped)} skipped, {len(failed)} failed",
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
                message=f"ingest failed unexpectedly; {len(executed)} ok, {len(skipped)} skipped, {len(failed)} failed",
                payload={"indexed": len(executed), "skipped": len(skipped), "failed": len(failed), "traceback": error_text},
            )
        )
        ledger.set_job_status(job_id, "failed")
        raise


def _publish_collected_embeds(
    ledger: StateLedger,
    *,
    job_id: str,
    embed_collector: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish every staged embed of this run, one publish per profile (Phase 3).

    Returns {"group_count", "failed_groups", "entries"} where ``entries`` is a
    per-collector-entry boolean list (True = its profile group published).
    Group failures are caught here so one broken profile cannot abort the
    run; the caller folds the outcomes into executed/failed.
    """
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(embed_collector):
        groups.setdefault(entry["profile_name"], []).append(index)

    entries: list[bool] = [False] * len(embed_collector)
    failed_groups = 0
    for profile_name, indexes in groups.items():
        try:
            results = publish_staged_documents(
                ledger=ledger,
                staged=[embed_collector[index]["staged"] for index in indexes],
            )
        except Exception as exc:
            failed_groups += 1
            ledger.add_event(
                JobEvent(
                    job_id=job_id,
                    stage="embed_publish",
                    status="failed",
                    message=f"batch publish failed for {profile_name}: {exc}",
                    payload={
                        "profile_name": profile_name,
                        "document_ids": [embed_collector[index]["document_id"] for index in indexes],
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
            continue
        # The completed event deferred by _execute_embed_stage is written only
        # now, once the vectors are actually visible to search.
        for index, index_result in zip(indexes, results):
            entries[index] = True
            ledger.add_event(
                JobEvent(
                    job_id=job_id,
                    stage=f"embed:{profile_name}",
                    status="completed",
                    message=f"indexed {index_result.indexed_chunks} chunks for {embed_collector[index]['document_id']}",
                    payload=index_result.to_dict(),
                )
            )
    return {"group_count": len(groups), "failed_groups": failed_groups, "entries": entries}


def _record_ingest_plan(
    ledger: StateLedger, *, job_id: str, plan: dict[str, Any], execute: bool
) -> None:
    """Persist the plan event and per-document checkpoints for a new job."""
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


def pause_ingest_job(ledger: StateLedger, job_id: str, *, reason: str = "manual pause") -> dict[str, Any]:
    return transition_ingest_job(ledger, job_id, status="paused", stage="pause", message=reason)


# M7: states from which `resume --execute` may take over execution.
# "completed" and "cancelled" are terminal decisions; "created"/"running" are
# handled by the runner-lock sweep (a stale one becomes "interrupted" first).
# M9: "blocked" (remote extraction still running) is resumable — the next run
# continues polling the same remote batches (M2).
RESUMABLE_EXECUTION_STATES = {"planned", "paused", "interrupted", "failed", "completed_with_errors", "blocked"}


def resume_ingest_job(
    ledger: StateLedger,
    job_id: str,
    *,
    reason: str = "manual resume",
    execute: bool = False,
    # --- execution dependencies (required when execute=True) ---
    extract_manager: Any = None,
    extract_cache_dir: str | Path | None = None,
    normalized_dir: str | Path | None = None,
    vector_store_dir: str | Path | None = None,
    extract_config: ExtractConfig | None = None,
    batch_publish: bool = True,
    embedding_env_path: str | Path = ".env",
) -> dict[str, Any]:
    """Rebuild the plan for an existing ingest job; with execute=True, also run it.

    M7: previously resume only re-planned and returned, leaving the job in
    "planned" with no consumer — the crash-recovery chain was broken. The
    execute path takes over the existing job under the runner lock.

    Phase 3: batch_publish is forwarded to the execution body; see
    start_ingest_job.
    """
    job = require_ingest_job(ledger, job_id)
    payload = dict(job.get("payload") or {})

    if execute:
        # Pure argument validation before any lock/ledger mutation.
        if extract_manager is None or extract_cache_dir is None or normalized_dir is None or vector_store_dir is None:
            raise ValueError(
                "extract_manager, extract_cache_dir, normalized_dir, and vector_store_dir "
                "are required when execute=True"
            )

    if not execute:
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

    # M7/M10: acquire the runner lock first. Its sweep marks a stale
    # "running"/"created" leftover of this very job "interrupted" — which is
    # resumable — so the state check must run *after* the sweep, not before.
    with acquire_runner_lock(ledger):
        job = require_ingest_job(ledger, job_id)
        status = str(job["status"])
        if status not in RESUMABLE_EXECUTION_STATES:
            raise ValueError(
                f"ingest job {job_id} is {status}; only {sorted(RESUMABLE_EXECUTION_STATES)} "
                "can be resumed with --execute. Use `ingest start` for a new run."
            )
        # Defense in depth: the sweep inside acquire_runner_lock already
        # interrupted every other active job, so this must be None.
        active = ledger.find_active_job(exclude_job_id=job_id)
        if active is not None:
            raise RuntimeError(
                f"an active {active['kind']} job exists (job_id={active['job_id']}, "
                f"status={active['status']}); refusing to resume {job_id}"
            )
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
                status="running",
                message=f"{reason}; executing stages",
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
        ledger.set_job_status(job_id, "running")
        return _execute_ingest_documents(
            ledger,
            embedding_env_path=embedding_env_path,
            job_id=job_id,
            plan=plan,
            extract_manager=extract_manager,
            extract_cache_dir=extract_cache_dir,
            normalized_dir=normalized_dir,
            vector_store_dir=vector_store_dir,
            extract_config=extract_config,
            batch_publish=batch_publish,
        )


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
    extract_stage = stage_for_extract(
        attachment, extract_jobs, ledger=ledger, force=full_rebuild, artifact=artifact
    )
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


def _all_segment_jobs_done(segment_job_ids: list[str], ledger: StateLedger) -> tuple[bool, list[dict[str, Any]]]:
    """Return (all_done, jobs) for a list of segment job ids."""

    jobs: list[dict[str, Any]] = []
    for job_id in segment_job_ids:
        job = ledger.get_extract_job(job_id=job_id)
        if job is None:
            return False, []
        jobs.append(job)
        if job["state"] not in DONE_EXTRACT_STATES:
            return False, jobs
    return True, jobs


def stage_for_extract(
    attachment: dict[str, Any],
    extract_jobs: list[dict[str, Any]],
    *,
    ledger: StateLedger,
    force: bool,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not attachment.get("file_exists"):
        return {"stage": "extract", "status": "blocked", "reason": "missing_file"}

    # If this attachment was previously split into segments, the checkpoint
    # records the segment job ids. We must wait for all segments to finish
    # before declaring the extract stage done.
    checkpoint = ledger.get_checkpoint(attachment["attachment_key"], "extract")
    segment_job_ids = (checkpoint.get("payload") or {}).get("segment_job_ids") if checkpoint else None
    if segment_job_ids and not force:
        all_done, segment_jobs = _all_segment_jobs_done(segment_job_ids, ledger)
        merged_md = Path(checkpoint["payload"].get("merged_markdown") or "")
        if all_done and checkpoint["status"] == "done" and merged_md.is_file():
            return {
                "stage": "extract",
                "status": "done",
                "job_id": segment_jobs[0]["job_id"],
                "state": segment_jobs[0]["state"],
                "segment_job_ids": segment_job_ids,
                "merged_artifact_dir": (checkpoint.get("payload") or {}).get("merged_artifact_dir"),
            }
        failed = next(
            (job for job in segment_jobs if job["state"] == "failed_manual_review"),
            None,
        )
        if failed is not None:
            return {
                "stage": "extract",
                "status": "blocked",
                "reason": "segment_failed",
                "job_id": failed["job_id"],
                "segment_job_ids": segment_job_ids,
            }
        return {
            "stage": "extract",
            "status": "pending",
            "reason": "segments_in_progress",
            "segment_job_ids": segment_job_ids,
        }

    # A crash can leave downloaded segments in the ledger before the merge.
    # Recover the saved plan instead of treating a single segment as a document.
    if checkpoint and not force and (checkpoint.get("payload") or {}).get("segment_size"):
        return {"stage": "extract", "status": "pending", "reason": "segments_in_progress"}
    extract_jobs = [job for job in extract_jobs if not job.get("selected_pages")]
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
    if artifact is None and not force:
        active_job = next((job for job in extract_jobs if job["state"] in {"submitted", "running", "empty-result"}), None)
        if active_job:
            return {"stage": "extract", "status": "pending", "reason": "resume_in_progress", "job_id": active_job["job_id"]}
    if artifact is not None and not force:
        return {"stage": "extract", "status": "done", "reason": "normalized_artifact_exists"}
    return {"stage": "extract", "status": "pending", "reason": "full_rebuild" if force else "no_cached_extract"}


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
# Execution helpers — called by start_ingest_job / resume --execute (execute=True)
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
    extract_config: ExtractConfig,
    embed_collector: list[dict[str, Any]] | None = None,
    embedding_env_path: str | Path = ".env",
) -> dict[str, Any]:
    """Execute pending stages for a single document, resuming from checkpoints.

    M9: stage outcomes are propagated. When extraction ends "blocked" (remote
    batches still running), normalize/embed are short-circuited so an
    incomplete extraction is never normalized (silent content truncation),
    and the returned dict carries blocked=True for the main loop's accounting.

    Phase 3: when embed_collector is given, embed stages append staged vectors
    to it instead of publishing; the returned dict's "staged_embeds" counts
    how many entries this document added so the main loop can hold the
    document back until the run-level publish.
    """

    attachment_key = document["attachment_key"]
    document_id = document["document_id"]
    staged_before = len(embed_collector) if embed_collector is not None else 0

    for stage in document["stages"]:
        stage_name = stage["stage"]
        if stage["status"] not in {"pending", "blocked"}:
            continue

        if stage_name == "extract":
            outcome = _execute_extract_stage(
                ledger=ledger,
                job_id=job_id,
                attachment_key=attachment_key,
                document=document,
                extract_manager=extract_manager,
                extract_cache_dir=extract_cache_dir,
                extract_config=extract_config,
            )
            if outcome == "blocked":
                return {
                    "attachment_key": attachment_key,
                    "document_id": document_id,
                    "title": document.get("title"),
                    "blocked": True,
                    "blocked_stage": "extract",
                    # Extract blocked before any embed stage ran, so this is 0
                    # unless a future stage order changes that.
                    "staged_embeds": (len(embed_collector) - staged_before) if embed_collector is not None else 0,
                }
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
                embedding_env_path=embedding_env_path,
                ledger=ledger,
                job_id=job_id,
                document_id=document_id,
                profile_name=profile_name,
                vector_store_dir=vector_store_dir,
                embed_collector=embed_collector,
            )

    return {
        "attachment_key": attachment_key,
        "document_id": document_id,
        "title": document.get("title"),
        "staged_embeds": (len(embed_collector) - staged_before) if embed_collector is not None else 0,
    }


def _execute_extract_stage(
    *,
    ledger: StateLedger,
    job_id: str,
    attachment_key: str,
    document: dict[str, Any],
    extract_manager: Any,
    extract_cache_dir: Path,
    extract_config: ExtractConfig,
) -> str:
    """Run OCR extraction for one attachment, splitting large PDFs into segments.

    M9: returns "done" when a complete extraction artifact is checkpointed, or
    "blocked" when remote batches are still running and the checkpoint records
    the blocked state. Hard errors still raise (the caller counts them as
    document failures).
    """

    from ..extractors.manager import ExtractionRequest, ExtractionResult

    extract_stage = next((s for s in document["stages"] if s["stage"] == "extract"), None)
    if extract_stage is None or extract_stage["status"] not in {"pending", "blocked"}:
        # Unreachable via _execute_document_stages (it pre-filters the same
        # statuses); keep the exit explicit so the return contract has no
        # implicit None.
        return "done"

    attachment = ledger.get_attachment(attachment_key)
    if attachment is None:
        raise ValueError(f"attachment not found in ledger: {attachment_key}")

    file_path = attachment.get("file_path")
    if not file_path or not Path(file_path).is_file():
        raise FileNotFoundError(f"attachment file not found: {file_path}")

    # Carry over segment ids from the planning stage so that interrupted
    # split extractions can be resumed without relying on the checkpoint
    # that we are about to overwrite with the "running" status.
    segment_job_ids = extract_stage.get("segment_job_ids")
    is_full_rebuild = extract_stage.get("reason") == "full_rebuild"

    total_pages = get_pdf_page_count(file_path)
    previous = ledger.get_checkpoint(attachment_key, "extract") if not is_full_rebuild else None
    saved = (previous or {}).get("payload") or {}
    previous_jobs = [ledger.get_extract_job(job_id=key) for key in (segment_job_ids or [])]
    if any(job is None for job in previous_jobs):
        raise ValueError(f"missing segment job for {attachment_key}")
    previous_job = ledger.get_extract_job(job_id=extract_stage["job_id"]) if extract_stage.get("job_id") else None
    pdf_sha256 = sha256_file(file_path)
    if previous_job and previous_job["pdf_sha256"] == pdf_sha256 and not previous_job.get("selected_pages"):
        result = extract_manager.resume_extraction(previous_job["job_id"])
        return _checkpoint_single_extract_result(ledger=ledger, job_id=job_id, attachment_key=attachment_key, result=result)

    if saved.get("pdf_sha256", pdf_sha256) != pdf_sha256 or any(job["pdf_sha256"] != pdf_sha256 for job in previous_jobs):
        saved, previous_jobs = {}, []
    recorded_provider = saved.get("ocr_provider") or (previous_jobs[0]["provider"] if previous_jobs else None)
    if recorded_provider and recorded_provider != extract_manager.provider.name:
        if extract_manager.resume_manager_for is None:
            raise ValueError(f"extract job requires its original OCR provider {recorded_provider}")
        extract_manager = extract_manager.resume_manager_for(recorded_provider)
    deepseek = extract_manager.provider.name == "deepseek"
    segment_size = extract_config.deepseek_pages_per_request if deepseek else extract_config.max_pages_per_extraction
    if saved.get("segment_size"):
        segment_size = saved["segment_size"]
    elif previous_jobs:
        # Legacy checkpoints persisted contiguous ranges but no split size.
        start, end = min(previous_jobs, key=_segment_page_start)["selected_pages"].split("-")
        segment_size = int(end) - int(start) + 1
    should_split = bool(previous_jobs or saved.get("segment_size")) or (
        extract_config.split_large_pdfs and total_pages is not None and total_pages > segment_size
    )
    running_payload = {"job_id": job_id, "input_file": str(file_path)}
    if should_split:
        if total_pages is None:
            raise ValueError(f"cannot verify segment page coverage for {attachment_key}")
        running_payload.update(segment_size=segment_size, ocr_provider=extract_manager.provider.name, pdf_sha256=pdf_sha256)
        if previous_jobs:
            running_payload["segment_job_ids"] = [job["job_id"] for job in previous_jobs]
    ledger.checkpoint(attachment_key, "extract", "running", running_payload)

    if not should_split:
        _progress("  extract: submitting...")
        # selected_page_count feeds the persisted per-job poll budget
        # (pages*6+30, clamped to [30min, job_max_lifetime_seconds] at poll
        # time). Use the real count when known and a conservative upper guess
        # when PyMuPDF could not read the pages, so the budget reflects the
        # actual document size.
        request = ExtractionRequest(
            input_file=Path(file_path),
            attachment_key=attachment_key,
            pdf_sha256=pdf_sha256,
            selected_page_count=total_pages if total_pages is not None else UNKNOWN_PAGE_COUNT_FALLBACK,
        )
        result = extract_manager.ensure_extraction(request)
        result = _drive_extraction_to_completion(result, extract_manager)
        return _checkpoint_single_extract_result(
            ledger=ledger,
            job_id=job_id,
            attachment_key=attachment_key,
            result=result,
        )

    segments = compute_page_segments(total_pages, segment_size)
    _progress(f"  extract: splitting PDF into {len(segments)} segments (pages={total_pages})...")
    existing_segments = {job["selected_pages"]: job for job in previous_jobs}

    def submit_segment(segment: PageSegment) -> ExtractionResult:
        existing = existing_segments.get(segment.selected_pages)
        if existing:
            # Keep remote IDs through a retry. A cache-key resubmit here can
            # duplicate work for a remote batch with a transient poll failure.
            return ExtractionResult(job=existing, cache_hit=True)
        return extract_manager.submit_only(ExtractionRequest(
            input_file=Path(file_path), attachment_key=attachment_key, pdf_sha256=pdf_sha256,
            selected_pages=segment.selected_pages, selected_page_count=segment.page_count,
        ))

    try:
        workers = extract_config.deepseek_concurrency if deepseek else extract_config.harvest_workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # DeepSeek queues its local job here and sends the HTTP request in harvest.
            segment_results = [submit_segment(segment) for segment in segments]
            segment_results = list(pool.map(lambda result: _harvest_one(result, extract_manager), segment_results))
        for result in segment_results:
            _progress(f"  extract: segment state={result.job['state']}")
    except Exception:
        # Persist the segment jobs that were submitted before the failure so
        # that a later resume can pick up where we left off.
        # H3: map each planned segment back to its own ledger job via the same
        # cache key ensure_extraction persisted it under. The previous code
        # scooped up *all* extract jobs for this attachment, which (a) pulled
        # in legacy whole-document jobs from before the split era and (b)
        # returned them in ledger order rather than segment order — both
        # corrupt the persisted segment_job_ids that resume/merge rely on.
        segment_jobs = _find_segment_jobs(
            segments,
            ledger=ledger,
            extract_manager=extract_manager,
            file_path=file_path,
            attachment_key=attachment_key,
        )
        if segment_jobs:
            _checkpoint_segment_blocked(
                ledger=ledger,
                job_id=job_id,
                attachment_key=attachment_key,
                segment_jobs=segment_jobs,
            )
        raise

    segment_jobs = [r.job for r in segment_results]
    if not _all_jobs_downloaded(segment_jobs):
        _checkpoint_segment_blocked(
            ledger=ledger,
            job_id=job_id,
            attachment_key=attachment_key,
            segment_jobs=segment_jobs,
        )
        return "blocked"

    _merge_and_checkpoint_segments(
        ledger=ledger,
        job_id=job_id,
        attachment_key=attachment_key,
        segment_jobs=segment_jobs,
        extract_cache_dir=extract_cache_dir,
        merged_artifact_dir=extract_stage.get("merged_artifact_dir"),
        # H2: a full rebuild must re-merge unconditionally so it can repair a
        # truncated/stale merged_full.md left behind by a killed run.
        force=is_full_rebuild or bool(previous_jobs) or bool(saved.get("segment_size")),
    )
    return "done"


def _drive_extraction_to_completion(result: Any, extract_manager: Any) -> Any:
    """Drive a freshly submitted extraction result to downloaded or failed."""

    if result.job["state"] == "completed" and result.job.get("local_stage") != "downloaded":
        result = extract_manager.resume_extraction(result.job["job_id"])
    if result.job["state"] in {"submitted", "running"}:
        result = extract_manager.resume_extraction(result.job["job_id"])
    return result


def _harvest_one(result: Any, extract_manager: Any) -> Any:
    """Drive one freshly submitted segment result to downloaded (Phase 2).

    Completed jobs and jobs awaiting manual review pass through untouched.
    Jobs whose local_stage is still "poll" go through harvest(), whose first
    poll confirms the remote batch state even when submit() already reported
    "completed" (H1). Anything else — a missing external_job_id (crash
    between upsert and submit) or an already-confirmed completion — routes
    through resume_extraction's classify dispatch.
    """

    job = result.job
    if job["state"] in DONE_EXTRACT_STATES or job["state"] == "failed_manual_review":
        return result
    if job.get("external_job_id") and job.get("local_stage") == "poll":
        return extract_manager.harvest(job["job_id"])
    return extract_manager.resume_extraction(job["job_id"])


def _all_jobs_downloaded(jobs: list[dict[str, Any]]) -> bool:
    return all(job["state"] in DONE_EXTRACT_STATES for job in jobs)


def _merged_artifact_dir(extract_cache_dir: Path, attachment_key: str) -> Path:
    return extract_cache_dir / "merged" / sanitize_id(attachment_key)


def _segment_page_start(job: dict[str, Any]) -> int:
    """Sort key for segment jobs: first page of ``selected_pages``.

    "201-400" -> 201. Whole-document jobs (empty selected_pages) and
    unparseable values sort first (0) so they never silently move to the end.
    """

    raw = str(job.get("selected_pages") or "").strip()
    try:
        return int(raw.split("-", 1)[0])
    except ValueError:
        return 0


def _find_segment_jobs(
    segments: list[Any],
    *,
    ledger: StateLedger,
    extract_manager: Any,
    file_path: str | Path,
    attachment_key: str,
) -> list[dict[str, Any]]:
    """Look up the ledger job for each planned page segment, in segment order.

    H3: each segment's cache key is recomputed exactly as ensure_extraction
    derived it, so the lookup finds precisely this split run's jobs — legacy
    whole-document jobs for the same attachment have a different key and are
    excluded. Segments whose job was never persisted (the failure happened
    before upsert_extract_job) are skipped.

    """

    from ..extractors.manager import ExtractionRequest, extraction_cache_context

    pdf_sha256 = sha256_file(file_path)
    jobs: list[dict[str, Any]] = []
    for segment in segments:
        request = ExtractionRequest(
            input_file=Path(file_path),
            attachment_key=attachment_key,
            selected_pages=segment.selected_pages,
            selected_page_count=segment.page_count,
        )
        cache_key = extraction_cache_context(request, extract_manager.provider, pdf_sha256=pdf_sha256).cache_key
        job = ledger.get_extract_job_by_cache_key(cache_key)
        if job is not None:
            jobs.append(job)
    return jobs


def _merge_and_checkpoint_segments(
    *,
    ledger: StateLedger,
    job_id: str,
    attachment_key: str,
    segment_jobs: list[dict[str, Any]],
    extract_cache_dir: Path,
    merged_artifact_dir: str | Path | None = None,
    force: bool = False,
) -> None:
    """Merge segment artifacts and persist a done checkpoint."""

    # H3: merge strictly in page order. Callers may hand us jobs in ledger or
    # legacy-checkpoint order (e.g. a resumed run whose persisted
    # segment_job_ids predate ordered persistence), which would otherwise
    # scramble merged_full.md. sorted() is stable, so ties keep input order.
    segment_jobs = sorted(segment_jobs, key=_segment_page_start)
    segment_job_ids = [job["job_id"] for job in segment_jobs]
    merged_dir = Path(merged_artifact_dir) if merged_artifact_dir else _merged_artifact_dir(extract_cache_dir, attachment_key)

    # Skip re-merging if the merged artifact already exists and looks valid.
    # H2: force (full rebuild) re-merges unconditionally so a truncated or
    # stale merged_full.md left behind by a killed run can be repaired;
    # re-merging is local, cheap, and idempotent (atomic write since H2).
    merged_md = merged_dir / "merged_full.md"
    if force or not merged_md.is_file():
        segment_dirs = segment_artifact_dirs_from_jobs(segment_jobs)
        merge_extract_segments(segment_dirs, merged_dir)

    ledger.checkpoint(
        attachment_key,
        "extract",
        "done",
        {
            "job_id": job_id,
            "extract_job_id": segment_job_ids[0],
            "segment_job_ids": segment_job_ids,
            "merged_artifact_dir": str(merged_dir),
            "merged_markdown": str(merged_md),
        },
    )

    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="extract",
            status="completed",
            message=f"extract segments merged for {attachment_key}: {len(segment_jobs)} segments",
            payload={
                "job_id": job_id,
                "attachment_key": attachment_key,
                "segment_job_ids": segment_job_ids,
                "merged_artifact_dir": str(merged_dir),
            },
        )
    )


def _checkpoint_single_extract_result(
    *,
    ledger: StateLedger,
    job_id: str,
    attachment_key: str,
    result: Any,
) -> str:
    """Persist the extract checkpoint for a single-PDF extraction.

    M9: returns "blocked" when the extract job is not done yet (the caller
    short-circuits normalize/embed), "done" otherwise.
    """
    _progress(f"  extract: {result.job['state']} (cache_hit={result.cache_hit})")

    if result.job["state"] not in DONE_EXTRACT_STATES:
        ledger.checkpoint(
            attachment_key,
            "extract",
            "blocked",
            {
                "job_id": job_id,
                "extract_job_id": result.job["job_id"],
                "state": result.job["state"],
                "cache_hit": result.cache_hit,
                "reason": f"extract job still {result.job['state']}",
            },
        )
        ledger.add_event(
            JobEvent(
                job_id=job_id,
                stage="extract",
                status=result.job["state"],
                message=f"extract job {result.job['job_id']} still {result.job['state']}; will retry",
                payload={
                    "job_id": result.job["job_id"],
                    "state": result.job["state"],
                    "cache_hit": result.cache_hit,
                },
            )
        )
        return "blocked"

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
            payload={
                "job_id": result.job["job_id"],
                "state": result.job["state"],
                "cache_hit": result.cache_hit,
            },
        )
    )
    return "done"


def _checkpoint_segment_blocked(
    *,
    ledger: StateLedger,
    job_id: str,
    attachment_key: str,
    segment_jobs: list[dict[str, Any]],
) -> None:
    # H3: persist segment_job_ids in page order, derived from the same sorted
    # jobs list as the states array so the two can never drift apart.
    segment_jobs = sorted(segment_jobs, key=_segment_page_start)
    segment_job_ids = [job["job_id"] for job in segment_jobs]
    failed_states = {job["state"] for job in segment_jobs if job["state"] not in DONE_EXTRACT_STATES}
    ledger.checkpoint(
        attachment_key,
        "extract",
        "blocked",
        {
            "job_id": job_id,
            **((ledger.get_checkpoint(attachment_key, "extract") or {}).get("payload") or {}),
            "segment_job_ids": segment_job_ids,
            "states": [job["state"] for job in segment_jobs],
            "reason": f"segment jobs not all downloaded: {failed_states}",
        },
    )
    ledger.add_event(
        JobEvent(
            job_id=job_id,
            stage="extract",
            status="blocked",
            message=f"segment extraction blocked for {attachment_key}: {failed_states}",
            payload={
                "attachment_key": attachment_key,
                "segment_job_ids": segment_job_ids,
                "states": [job["state"] for job in segment_jobs],
            },
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

    # Find the extract job that has a downloaded artifact. Prefer the job id
    # recorded by the extract stage so that attachments sharing the same PDF
    # cache_key can reuse a single extract job without creating duplicate
    # records.
    extract_job: dict[str, Any] | None = None
    segment_job_ids: list[str] | None = None
    extract_checkpoint = ledger.get_checkpoint(attachment_key, "extract")
    if (
        extract_checkpoint is not None
        and extract_checkpoint.get("status") != "done"
        and extract_checkpoint.get("payload", {}).get("segment_job_ids")
    ):
        # M9 guard: a segmented extraction whose checkpoint never reached
        # "done" (crash between segment completion and merge) must NOT fall
        # through to the fallback below — it would pick any single downloaded
        # segment fragment and normalize it as the full text (silent content
        # truncation). Fail loudly instead; the extract stage re-runs and
        # completes the merge on the next ingest.
        raise ValueError(
            f"segmented extraction for {attachment_key} is incomplete "
            f"(extract checkpoint status={extract_checkpoint.get('status')}); "
            "run extraction first"
        )
    if extract_checkpoint is not None and extract_checkpoint.get("status") == "done":
        shared_job_id = extract_checkpoint.get("payload", {}).get("extract_job_id")
        if shared_job_id:
            candidate = ledger.get_extract_job(job_id=shared_job_id)
            if candidate is not None and candidate["state"] in DONE_EXTRACT_STATES:
                extract_job = candidate
        segment_job_ids = extract_checkpoint.get("payload", {}).get("segment_job_ids")

    if extract_job is None:
        extract_jobs = [
            j for j in ledger.list_extract_jobs(limit=None)
            if j.get("attachment_key") == attachment_key and not j.get("selected_pages") and j["state"] in DONE_EXTRACT_STATES
        ]
        if not extract_jobs:
            raise ValueError(f"no completed extract job for {attachment_key}; run extraction first")
        extract_job = extract_jobs[0]

    source_md: Path | None = None
    merged_artifact_dir = extract_checkpoint.get("payload", {}).get("merged_artifact_dir") if extract_checkpoint else None
    if merged_artifact_dir:
        source_md = Path(merged_artifact_dir) / "merged_full.md"
        if not source_md.is_file():
            raise FileNotFoundError(f"merged markdown not found: {source_md}")
    else:
        artifact_dir = extract_job.get("artifact_dir") or extract_job.get("extract_dir")
        if not artifact_dir:
            raise ValueError(f"extract job {extract_job['job_id']} has no artifact_dir")
        source_md = _find_source_markdown(Path(artifact_dir))
        if source_md is None:
            raise FileNotFoundError(f"no markdown file found in extract artifact: {artifact_dir}")

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
        segment_job_ids=segment_job_ids,
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
    embed_collector: list[dict[str, Any]] | None = None,
    embedding_env_path: str | Path = ".env",
) -> None:
    """Run embedding for one document under one profile.

    Phase 3: when embed_collector is None (batch_publish=False), the document
    is published immediately via index_normalized_document — the historical
    behavior. Otherwise vectors are only staged and appended to the collector;
    the "completed" event is deferred to the run-level publish, which is the
    earliest point the vectors are actually visible to search.
    """

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

    if embed_collector is None:
        index_result = index_normalized_document(
            embedding_env_path=embedding_env_path,
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
        return

    # Exceptions from staging propagate to the main loop's per-document
    # failure accounting, exactly as index_normalized_document's did.
    staged = stage_indexed_document(
        embedding_env_path=embedding_env_path,
        ledger=ledger,
        vector_store_dir=vector_store_dir,
        profile_name=profile_name,
        document_id=document_id,
        # Provider is auto-resolved from profile by _resolve_embedding_provider
    )
    if isinstance(staged, IndexResult):
        # Reuse fast path: already published, so the completed event is due
        # immediately (nothing was deferred).
        _progress(f"  embed:{profile_name}: {staged.indexed_chunks} chunks (reused={staged.reused_existing})")
        ledger.add_event(
            JobEvent(
                job_id=job_id,
                stage=f"embed:{profile_name}",
                status="completed",
                message=f"indexed {staged.indexed_chunks} chunks for {document_id}",
                payload=staged.to_dict(),
            )
        )
        return
    embed_collector.append(
        {"document_id": document_id, "profile_name": profile_name, "staged": staged}
    )
    _progress(f"  embed:{profile_name}: {len(staged.records)} chunks staged (publish deferred)")


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
