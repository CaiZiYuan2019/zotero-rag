from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import unittest
from typing import Any

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.embeddings.profile import embedding_profile_hash
from unittest.mock import Mock, patch

from zoterorag.config import ExtractConfig
from zoterorag.extractors.base import ExtractArtifact, ExtractJobState, StubExtractorProvider
from zoterorag.extractors.manager import ExtractionManager, ExtractionRequest, ExtractionResult, extraction_cache_context
from zoterorag.normalize import normalize_markdown_document
from zoterorag.pipeline import (
    cancel_ingest_job,
    create_ingest_plan,
    pause_ingest_job,
    resume_ingest_job,
    start_ingest_job,
)
from zoterorag.pipeline.ingest import (
    CIRCUIT_BREAKER_CONSECUTIVE_FAILURES,
    _execute_normalize_stage,
    _harvest_one,
    _merge_and_checkpoint_segments,
    _segment_page_start,
    _track_circuit_breaker,
)


class CircuitBreakerUnitTests(unittest.TestCase):
    def test_streak_grows_only_for_same_error_class(self) -> None:
        kind, streak, tripped = _track_circuit_breaker(ValueError("a"), kind=None, streak=0)
        self.assertEqual(("ValueError", 1, False), (kind, streak, tripped))
        kind, streak, tripped = _track_circuit_breaker(ValueError("b"), kind=kind, streak=streak)
        self.assertEqual(("ValueError", 2, False), (kind, streak, tripped))
        # A different class restarts the streak at 1.
        kind, streak, tripped = _track_circuit_breaker(FileNotFoundError("c"), kind=kind, streak=streak)
        self.assertEqual(("FileNotFoundError", 1, False), (kind, streak, tripped))

    def test_trips_at_threshold(self) -> None:
        kind, streak, tripped = None, 0, False
        for _ in range(CIRCUIT_BREAKER_CONSECUTIVE_FAILURES - 1):
            kind, streak, tripped = _track_circuit_breaker(RuntimeError("x"), kind=kind, streak=streak)
            self.assertFalse(tripped)
        kind, streak, tripped = _track_circuit_breaker(RuntimeError("x"), kind=kind, streak=streak)
        self.assertTrue(tripped)
        self.assertEqual(CIRCUIT_BREAKER_CONSECUTIVE_FAILURES, streak)


class SegmentOrderingUnitTests(unittest.TestCase):
    def test_segment_page_start_parses_range(self) -> None:
        self.assertEqual(1, _segment_page_start({"selected_pages": "1-200"}))
        self.assertEqual(201, _segment_page_start({"selected_pages": " 201-400 "}))

    def test_segment_page_start_defaults_to_zero(self) -> None:
        # Whole-document jobs and unparseable values sort first, never last.
        self.assertEqual(0, _segment_page_start({"selected_pages": ""}))
        self.assertEqual(0, _segment_page_start({"selected_pages": None}))
        self.assertEqual(0, _segment_page_start({}))
        self.assertEqual(0, _segment_page_start({"selected_pages": "abc"}))


class IngestPipelineTests(unittest.TestCase):
    def test_incremental_plan_uses_local_state_without_external_execution(self) -> None:
        with workspace_tmpdir("ingest-plan-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                seed_attachments(ledger)
                normalized = normalize_fixture(tmpdir, ledger)
                ledger.checkpoint(
                    normalized.document_id,
                    "embed:text-profile",
                    "indexed",
                    {"chunks": 1, "profile_hash": profile_hash(ledger, "text-profile")},
                )
                ledger.checkpoint(
                    normalized.document_id,
                    "embed:mm-profile",
                    "indexed",
                    {"chunks": 1, "profile_hash": profile_hash(ledger, "mm-profile")},
                )

                plan = create_ingest_plan(ledger)
                by_key = {doc["attachment_key"]: doc for doc in plan["documents"]}

                self.assertEqual(3, plan["summary"]["document_count"])
                self.assertNotIn("ATT_REVIEW", by_key)
                self.assertEqual("complete", by_key["ATT_DONE"]["next_stage"])
                self.assertEqual("extract:pending", by_key["ATT_NEW"]["next_stage"])
                self.assertEqual("extract:blocked", by_key["ATT_MISSING"]["next_stage"])
                self.assertEqual("text-profile", plan["text_profile"])
                self.assertEqual("mm-profile", plan["multimodal_profile"])
            finally:
                ledger.close()

    def test_profile_hash_mismatch_marks_embedding_pending(self) -> None:
        with workspace_tmpdir("ingest-profile-hash-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                ledger.upsert_attachments([build_attachment("ATT_DONE", title="Already Indexed")])
                normalized = normalize_fixture(tmpdir, ledger)
                ledger.checkpoint(
                    normalized.document_id,
                    "embed:text-profile",
                    "indexed",
                    {"chunks": 1, "profile_hash": "old-hash"},
                )

                plan = create_ingest_plan(ledger, include_multimodal=False)
                document = plan["documents"][0]
                embed_stage = next(stage for stage in document["stages"] if stage["stage"] == "embed:text-profile")

                self.assertEqual("embed:text-profile:pending", document["next_stage"])
                self.assertEqual("pending", embed_stage["status"])
                self.assertEqual("profile_changed", embed_stage["reason"])
                self.assertEqual("old-hash", embed_stage["checkpoint_profile_hash"])
                self.assertEqual(profile_hash(ledger, "text-profile"), embed_stage["profile_hash"])
            finally:
                ledger.close()

    def test_start_pause_resume_cancel_persist_job_events_and_checkpoints(self) -> None:
        with workspace_tmpdir("ingest-job-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                seed_attachments(ledger)

                result = start_ingest_job(ledger, zotero_key="ATT_NEW", include_multimodal=False)
                job_id = result["job"]["job_id"]

                self.assertEqual("planned", result["job"]["status"])
                self.assertEqual(1, result["plan"]["summary"]["document_count"])
                checkpoint = ledger.get_checkpoint("ATT_NEW", "ingest_plan")
                self.assertIsNotNone(checkpoint)
                self.assertEqual(job_id, checkpoint["payload"]["job_id"])

                paused = pause_ingest_job(ledger, job_id, reason="test pause")
                self.assertEqual("paused", paused["job"]["status"])

                resumed = resume_ingest_job(ledger, job_id, reason="test resume")
                self.assertEqual("planned", resumed["job"]["status"])
                self.assertEqual(1, resumed["plan"]["summary"]["document_count"])

                cancelled = cancel_ingest_job(ledger, job_id, reason="test cancel")
                self.assertEqual("cancelled", cancelled["job"]["status"])
                stages = [event["stage"] for event in cancelled["job"]["events"]]
                self.assertEqual(["plan", "pause", "resume", "cancel"], stages)
            finally:
                ledger.close()

    def test_execute_true_requires_dependencies(self) -> None:
        with workspace_tmpdir("ingest-execute-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                seed_attachments(ledger)
                # execute=True without extract_manager and paths raises ValueError.
                with self.assertRaises(ValueError):
                    start_ingest_job(ledger, execute=True)
                self.assertEqual([], ledger.list_jobs(kind="ingest"))
            finally:
                ledger.close()

    def test_execute_sweeps_stale_running_reembed(self) -> None:
        """M10: a running job while the runner lock is free is a dead-process
        leftover — it is auto-interrupted and the new ingest proceeds."""
        with workspace_tmpdir("ingest-sweep-reembed-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_SWEEP", title="Sweep Paper")
                ledger.upsert_attachments([attachment])
                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                stale_id = ledger.create_job("reembed", {"profile_name": "text-profile"})
                ledger.set_job_status(stale_id, "running")

                result = start_ingest_job(
                    ledger,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )
                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual("interrupted", ledger.get_job(stale_id)["status"])
            finally:
                ledger.close()

    def test_execute_blocked_when_runner_lock_held(self) -> None:
        """M10: a live runner holding the lock makes a second start fail fast."""
        from zoterorag.pipeline.runner import acquire_runner_lock

        with workspace_tmpdir("ingest-lock-held-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_LOCKED", title="Locked Paper")
                ledger.upsert_attachments([attachment])
                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                with acquire_runner_lock(ledger):
                    with self.assertRaises(RuntimeError) as ctx:
                        start_ingest_job(
                            ledger,
                            execute=True,
                            extract_manager=extract_manager,
                            extract_cache_dir=tmpdir / "extract_cache",
                            normalized_dir=tmpdir / "normalized",
                            vector_store_dir=tmpdir / "vectors",
                        )
                self.assertIn("runner process is active", str(ctx.exception))
                self.assertEqual([], ledger.list_jobs(kind="ingest"))
            finally:
                ledger.close()

    # ------------------------------------------------------------------
    # M7: resume --execute takes over a job and actually runs it
    # ------------------------------------------------------------------

    def _resume_execute(self, ledger, tmpdir, job_id, **overrides):
        kwargs = dict(
            execute=True,
            extract_manager=ExtractionManager(
                ledger=ledger,
                cache_dir=tmpdir / "extract_cache",
                provider=MarkdownStubExtractorProvider(),
            ),
            extract_cache_dir=tmpdir / "extract_cache",
            normalized_dir=tmpdir / "normalized",
            vector_store_dir=tmpdir / "vectors",
        )
        kwargs.update(overrides)
        return resume_ingest_job(ledger, job_id, **kwargs)

    def test_resume_execute_runs_paused_job(self) -> None:
        with workspace_tmpdir("ingest-resume-execute-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_RESUME", title="Resume Paper")
                ledger.upsert_attachments([attachment])
                result = start_ingest_job(ledger, zotero_key="ATT_RESUME", include_multimodal=False)
                job_id = result["job"]["job_id"]
                pause_ingest_job(ledger, job_id, reason="test pause")

                resumed = self._resume_execute(ledger, tmpdir, job_id)

                self.assertEqual("completed", resumed["job"]["status"])
                self.assertEqual(1, len(resumed["executed"]))
                self.assertEqual("done", ledger.get_checkpoint("ATT_RESUME", "extract")["status"])
                self.assertEqual("indexed", ledger.get_checkpoint("ATT_RESUME", "embed:text-profile")["status"])
                resume_events = [e for e in resumed["job"]["events"] if e["stage"] == "resume"]
                self.assertEqual("running", resume_events[-1]["status"])
            finally:
                ledger.close()

    def test_resume_execute_takes_over_stale_running_job(self) -> None:
        """M7+M10: the crash-recovery chain — sweep marks the dead job
        interrupted, then the same job_id is resumed and executed."""
        with workspace_tmpdir("ingest-resume-stale-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_CRASH", title="Crashed Paper")
                ledger.upsert_attachments([attachment])
                job_id = ledger.create_job(
                    "ingest",
                    {"mode": "incremental", "zotero_key": "ATT_CRASH", "include_multimodal": False},
                )
                ledger.set_job_status(job_id, "running")  # dead-process residue

                resumed = self._resume_execute(ledger, tmpdir, job_id)

                self.assertEqual(job_id, resumed["job"]["job_id"])
                self.assertEqual("completed", resumed["job"]["status"])
                self.assertEqual("indexed", ledger.get_checkpoint("ATT_CRASH", "embed:text-profile")["status"])
                stages = [e["stage"] for e in resumed["job"]["events"]]
                self.assertIn("lifecycle", stages)  # runner-lock sweep
                self.assertIn("resume", stages)
                lifecycle = next(e for e in resumed["job"]["events"] if e["stage"] == "lifecycle")
                self.assertEqual({"previous_status": "running"}, lifecycle["payload"])
            finally:
                ledger.close()

    def test_resume_execute_rejects_completed_job(self) -> None:
        with workspace_tmpdir("ingest-resume-completed-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_DONE_JOB", title="Done Paper")
                ledger.upsert_attachments([attachment])
                result = start_ingest_job(
                    ledger,
                    zotero_key="ATT_DONE_JOB",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=ExtractionManager(
                        ledger=ledger,
                        cache_dir=tmpdir / "extract_cache",
                        provider=MarkdownStubExtractorProvider(),
                    ),
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )
                job_id = result["job"]["job_id"]
                with self.assertRaises(ValueError) as ctx:
                    self._resume_execute(ledger, tmpdir, job_id)
                self.assertIn("is completed;", str(ctx.exception))
            finally:
                ledger.close()

    def test_resume_execute_from_completed_with_errors(self) -> None:
        """P2: completed_with_errors is resumable — failed documents are
        re-planned and executed."""
        with workspace_tmpdir("ingest-resume-cwe-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_CWE", title="Retry Paper")
                ledger.upsert_attachments([attachment])
                job_id = ledger.create_job(
                    "ingest",
                    {"mode": "incremental", "zotero_key": "ATT_CWE", "include_multimodal": False},
                )
                ledger.set_job_status(job_id, "completed_with_errors")

                resumed = self._resume_execute(ledger, tmpdir, job_id)

                self.assertEqual(job_id, resumed["job"]["job_id"])
                self.assertEqual("completed", resumed["job"]["status"])
                self.assertEqual("indexed", ledger.get_checkpoint("ATT_CWE", "embed:text-profile")["status"])
            finally:
                ledger.close()

    def test_resume_execute_requires_dependencies(self) -> None:
        with workspace_tmpdir("ingest-resume-deps-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                seed_attachments(ledger)
                result = start_ingest_job(ledger, zotero_key="ATT_NEW", include_multimodal=False)
                job_id = result["job"]["job_id"]
                with self.assertRaises(ValueError):
                    resume_ingest_job(ledger, job_id, execute=True)
                # No ledger mutation happened: still planned, no resume event.
                job = ledger.get_job(job_id)
                self.assertEqual("planned", job["status"])
                self.assertEqual([], [e for e in job["events"] if e["stage"] == "resume"])
            finally:
                ledger.close()

    def test_resume_execute_blocked_when_runner_lock_held(self) -> None:
        from zoterorag.pipeline.runner import acquire_runner_lock

        with workspace_tmpdir("ingest-resume-locked-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                seed_attachments(ledger)
                result = start_ingest_job(ledger, zotero_key="ATT_NEW", include_multimodal=False)
                job_id = result["job"]["job_id"]
                with acquire_runner_lock(ledger):
                    with self.assertRaises(RuntimeError):
                        self._resume_execute(ledger, tmpdir, job_id)
                self.assertEqual("planned", ledger.get_job(job_id)["status"])
            finally:
                ledger.close()

    def test_execute_ingest_end_to_end_with_stub(self) -> None:
        with workspace_tmpdir("ingest-execute-e2e-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_E2E", title="End-to-End Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                result = start_ingest_job(
                    ledger,
                    zotero_key="ATT_E2E",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )

                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual(1, len(result["executed"]))
                self.assertEqual(0, len(result["failed"]))

                # Checkpoints record stage progress.
                self.assertEqual("done", ledger.get_checkpoint("ATT_E2E", "extract")["status"])
                self.assertEqual("normalized", ledger.get_checkpoint("ATT_E2E", "normalize")["status"])
                self.assertEqual("indexed", ledger.get_checkpoint("ATT_E2E", "embed:text-profile")["status"])

                # Normalized artifact and chunks were persisted.
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_E2E"))
                self.assertGreater(len(ledger.list_chunks("ATT_E2E", chunk_type="text")), 0)
            finally:
                ledger.close()

    def test_keyboard_interrupt_marks_ingest_job_interrupted(self) -> None:
        """Ctrl+C during ingest execution must leave the job in 'interrupted' state."""
        with workspace_tmpdir("ingest-interrupt-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_INTERRUPT", title="Interrupt Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )

                with patch("zoterorag.pipeline.ingest._execute_document_stages", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        start_ingest_job(
                            ledger,
                            zotero_key="ATT_INTERRUPT",
                            include_multimodal=False,
                            execute=True,
                            extract_manager=extract_manager,
                            extract_cache_dir=tmpdir / "extract_cache",
                            normalized_dir=tmpdir / "normalized",
                            vector_store_dir=tmpdir / "vectors",
                        )

                jobs = ledger.list_jobs(kind="ingest")
                self.assertEqual(1, len(jobs))
                self.assertEqual("interrupted", jobs[0]["status"])

                events = ledger.list_job_events(jobs[0]["job_id"])
                complete_event = next((e for e in events if e["stage"] == "complete"), None)
                self.assertIsNotNone(complete_event)
                self.assertEqual("interrupted", complete_event["status"])
                self.assertIn("interrupted by user", complete_event["message"])
            finally:
                ledger.close()

    def test_completed_extract_state_triggers_download_recovery(self) -> None:
        with workspace_tmpdir("ingest-completed-recovery-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_RECOV", title="Recovery Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                # First, get to a downloaded state.
                request = ExtractionRequest(
                    input_file=Path(attachment["file_path"]),
                    attachment_key="ATT_RECOV",
                )
                first = extract_manager.ensure_extraction(request)
                self.assertEqual("downloaded", first.job["state"])

                # Simulate the artifact going missing while the ledger still says
                # ``completed`` (remote done, local artifact not downloaded).
                artifact_dir = first.job.get("artifact_dir")
                self.assertIsNotNone(artifact_dir)
                import shutil

                shutil.rmtree(artifact_dir)
                ledger.set_extract_job_state(
                    first.job["job_id"],
                    state="completed",
                    local_stage="download",
                    artifact_dir=None,
                    extract_dir=None,
                    manifest_path=None,
                )

                plan = create_ingest_plan(ledger, include_multimodal=False)
                self.assertEqual("extract:pending", plan["documents"][0]["next_stage"])
                self.assertEqual("completed_artifact_not_downloaded", plan["documents"][0]["stages"][0]["reason"])

                # Pin the page count so the ingest-side cache key matches the
                # job created above with the 1-page default request.
                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=1):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_RECOV",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                    )

                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual(1, len(result["executed"]))
                self.assertEqual("downloaded", ledger.get_extract_job(job_id=first.job["job_id"])["state"])
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_RECOV"))
            finally:
                ledger.close()

    def test_resume_ingest_job_skips_already_done_documents(self) -> None:
        with workspace_tmpdir("ingest-resume-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_RESUME", title="Resume Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                first = start_ingest_job(
                    ledger,
                    zotero_key="ATT_RESUME",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )
                self.assertEqual("completed", first["job"]["status"])

                resumed = resume_ingest_job(ledger, first["job"]["job_id"])
                self.assertEqual("planned", resumed["job"]["status"])
                self.assertEqual("complete", resumed["plan"]["documents"][0]["next_stage"])
            finally:
                ledger.close()

    def test_shared_pdf_cache_key_attachment_reuses_extract_artifact(self) -> None:
        """Two attachments pointing to the same PDF must share one extract job."""
        with workspace_tmpdir("ingest-shared-cache-key-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                source = tmpdir / "shared.pdf"
                source.write_bytes(b"shared pdf content")
                att1 = build_attachment("ATT_SHARED_1", title="Shared One")
                att1["file_path"] = str(source)
                att1["file_exists"] = True
                att2 = build_attachment("ATT_SHARED_2", title="Shared Two")
                att2["file_path"] = str(source)
                att2["file_exists"] = True
                ledger.upsert_attachments([att1, att2])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                first = start_ingest_job(
                    ledger,
                    zotero_key="ATT_SHARED_1",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )
                self.assertEqual("completed", first["job"]["status"])

                # The second attachment reuses the same cached artifact.
                second = start_ingest_job(
                    ledger,
                    zotero_key="ATT_SHARED_2",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )
                self.assertEqual("completed", second["job"]["status"])
                self.assertEqual(1, len(second["executed"]))
                self.assertEqual(0, len(second["failed"]))
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_SHARED_2"))

                # Only one extract job record should exist for the shared PDF.
                jobs = ledger.list_extract_jobs(limit=None)
                self.assertEqual(1, len(jobs))
            finally:
                ledger.close()

    def test_running_extract_job_is_resumed_and_completes(self) -> None:
        """An interrupted run that left an extract job in 'running' state can resume."""
        with workspace_tmpdir("ingest-running-resume-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_RUN", title="Running Paper")
                ledger.upsert_attachments([attachment])

                provider = TwoPhaseStubExtractorProvider()

                # Seed a running extract job as if a previous run was interrupted.
                # submitted_at must be fresh: the M2 lifetime cap flips jobs
                # older than job_max_lifetime_seconds (6h) to manual review.
                from zoterorag.db.state import utc_now
                from zoterorag.extractors.cache import (
                    extractor_cache_key,
                    sha256_file,
                    stable_options_hash,
                )

                pdf_sha256 = sha256_file(attachment["file_path"])
                # Match the default timeout that ensure_extraction adds for a
                # single-page document when no options are provided.
                options = {"timeout_seconds": 36}
                options_hash = stable_options_hash(options)
                cache_key = extractor_cache_key(
                    pdf_sha256=pdf_sha256,
                    selected_pages="",
                    extractor_name=provider.name,
                    extractor_version=provider.version,
                    options_hash=options_hash,
                    endpoint_url="",
                )
                ledger.upsert_extract_job(
                    {
                        "job_id": "running-job-id",
                        "attachment_key": "ATT_RUN",
                        "pdf_sha256": pdf_sha256,
                        "selected_pages": "",
                        "cache_key": cache_key,
                        "provider": provider.name,
                        "provider_version": provider.version,
                        "options_hash": options_hash,
                        "api_key_alias": "stub",
                        "external_job_id": "ext-running",
                        "state": "running",
                        "local_stage": "poll",
                        "submitted_at": utc_now(),
                        "last_poll_at": utc_now(),
                        "payload": {
                            "input_file": attachment["file_path"],
                            "options": options,
                        },
                    }
                )

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=provider,
                )
                # Pin the page count to 1 so the ingest-side cache key matches
                # the seeded running job (which used the 1-page timeout).
                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=1):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_RUN",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                    )

                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual(1, len(result["executed"]))
                self.assertEqual(0, len(result["failed"]))
                self.assertEqual("downloaded", ledger.get_extract_job(job_id="running-job-id")["state"])
                self.assertEqual("done", ledger.get_checkpoint("ATT_RUN", "extract")["status"])
                self.assertEqual("normalized", ledger.get_checkpoint("ATT_RUN", "normalize")["status"])
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_RUN"))
            finally:
                ledger.close()

    def test_split_large_pdf_into_segments(self) -> None:
        with workspace_tmpdir("ingest-split-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_SPLIT", title="Long Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                extract_config = ExtractConfig(split_large_pdfs=True, max_pages_per_extraction=200)

                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_SPLIT",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )

                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual(1, len(result["executed"]))
                self.assertEqual(0, len(result["failed"]))

                checkpoint = ledger.get_checkpoint("ATT_SPLIT", "extract")
                self.assertEqual("done", checkpoint["status"])
                segment_job_ids = checkpoint["payload"]["segment_job_ids"]
                self.assertEqual(2, len(segment_job_ids))

                extract_jobs = [
                    ledger.get_extract_job(job_id=job_id) for job_id in segment_job_ids
                ]
                self.assertTrue(all(job["state"] == "downloaded" for job in extract_jobs))

                artifact = ledger.get_normalized_artifact("ATT_SPLIT")
                self.assertIsNotNone(artifact)
                self.assertEqual(segment_job_ids, artifact["payload"]["segment_job_ids"])
                self.assertGreater(len(ledger.list_chunks("ATT_SPLIT", chunk_type="text")), 0)
            finally:
                ledger.close()

    def test_split_large_pdf_resumes_segments(self) -> None:
        with workspace_tmpdir("ingest-split-resume-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(
                    tmpdir, "ATT_SPLIT_RESUME", title="Long Paper Resume"
                )
                ledger.upsert_attachments([attachment])

                class HangingSegmentProvider(MarkdownStubExtractorProvider):
                    def submit(
                        self,
                        input_file: Path,
                        options_hash: str,
                        *,
                        options: dict[str, Any] | None = None,
                        api_key: str | None = None,
                    ) -> ExtractJobState:
                        page_ranges = (options or {}).get("page_ranges", "")
                        if page_ranges == "1-200":
                            return ExtractJobState(external_job_id="seg1", state="completed")
                        # Pretend segment 2 is accepted but still processing.
                        return ExtractJobState(external_job_id="seg2", state="completed")

                    def poll(
                        self,
                        external_job_id: str,
                        *,
                        api_key: str | None = None,
                        options: dict[str, Any] | None = None,
                    ) -> ExtractJobState:
                        if external_job_id == "seg1":
                            return ExtractJobState(external_job_id=external_job_id, state="completed")
                        # Segment 2 is not ready yet during the first run.
                        return ExtractJobState(
                            external_job_id=external_job_id,
                            state="failed",
                            message="simulated pending",
                        )

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=HangingSegmentProvider(),
                )
                extract_config = ExtractConfig(split_large_pdfs=True, max_pages_per_extraction=200)

                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_SPLIT_RESUME",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )

                # Segment 2 is still running, so the ingest completes with errors.
                self.assertEqual("completed_with_errors", result["job"]["status"])
                checkpoint = ledger.get_checkpoint("ATT_SPLIT_RESUME", "extract")
                self.assertEqual("blocked", checkpoint["status"])
                segment_job_ids = checkpoint["payload"]["segment_job_ids"]
                self.assertEqual(2, len(segment_job_ids))

                # Simulate segment 2 completing remotely.
                seg2_job = ledger.get_extract_job(job_id=segment_job_ids[1])
                ledger.set_extract_job_state(
                    seg2_job["job_id"],
                    state="completed",
                    local_stage="download",
                    external_job_id=seg2_job["external_job_id"],
                )

                # Resume with a normal provider that can download segment 2.
                normal_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result2 = start_ingest_job(
                        ledger,
                        zotero_key="ATT_SPLIT_RESUME",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=normal_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )

                self.assertEqual("completed", result2["job"]["status"])
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_SPLIT_RESUME"))
            finally:
                ledger.close()

    def test_split_failure_checkpoint_excludes_legacy_jobs_and_keeps_order(self) -> None:
        """H3: when a split run fails mid-loop, the blocked checkpoint must
        list exactly this run's segment jobs in page order. The previous
        implementation scooped up every extract job of the attachment
        (including pre-split whole-document jobs) in ledger order."""
        with workspace_tmpdir("ingest-split-order-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(
                    tmpdir, "ATT_SPLIT_ORDER", title="Long Paper Order"
                )
                ledger.upsert_attachments([attachment])

                class FailingSegmentProvider(MarkdownStubExtractorProvider):
                    def submit(
                        self,
                        input_file: Path,
                        options_hash: str,
                        *,
                        options: dict[str, Any] | None = None,
                        api_key: str | None = None,
                    ) -> ExtractJobState:
                        page_ranges = (options or {}).get("page_ranges", "")
                        if page_ranges == "1-200":
                            return ExtractJobState(external_job_id="seg1", state="completed")
                        return ExtractJobState(external_job_id="seg2", state="completed")

                    def poll(
                        self,
                        external_job_id: str,
                        *,
                        api_key: str | None = None,
                        options: dict[str, Any] | None = None,
                    ) -> ExtractJobState:
                        if external_job_id == "seg1":
                            return ExtractJobState(external_job_id=external_job_id, state="completed")
                        # Segment 2 fails remotely -> MinerUAPIError mid-loop.
                        return ExtractJobState(
                            external_job_id=external_job_id,
                            state="failed",
                            message="simulated remote failure",
                        )

                provider = FailingSegmentProvider()

                # Seed a legacy whole-document extract job from before the
                # split era, derived through the same cache-key helper so the
                # row is indistinguishable from a real one except for its
                # whole-document selected_pages.
                from zoterorag.db.state import utc_now

                legacy_request = ExtractionRequest(
                    input_file=Path(attachment["file_path"]),
                    attachment_key="ATT_SPLIT_ORDER",
                    selected_page_count=250,
                )
                legacy_ctx = extraction_cache_context(legacy_request, provider)
                ledger.upsert_extract_job(
                    {
                        "job_id": "legacy-whole-doc-job",
                        "attachment_key": "ATT_SPLIT_ORDER",
                        "pdf_sha256": legacy_ctx.pdf_sha256,
                        "selected_pages": "",
                        "cache_key": legacy_ctx.cache_key,
                        "provider": provider.name,
                        "provider_version": provider.version,
                        "options_hash": legacy_ctx.options_hash,
                        "api_key_alias": "stub",
                        "state": "failed_retryable",
                        "local_stage": "error",
                        "submitted_at": utc_now(),
                        "payload": {
                            "input_file": attachment["file_path"],
                            "options": legacy_ctx.options,
                        },
                    }
                )
                # Make the legacy row the most recently updated one so any
                # updated_at-ordered listing would surface it first.
                ledger.conn.execute(
                    "UPDATE extract_jobs SET updated_at = ? WHERE job_id = ?",
                    ("2999-01-01T00:00:00+00:00", "legacy-whole-doc-job"),
                )

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=provider,
                )
                extract_config = ExtractConfig(split_large_pdfs=True, max_pages_per_extraction=200)

                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_SPLIT_ORDER",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )

                self.assertEqual("completed_with_errors", result["job"]["status"])
                checkpoint = ledger.get_checkpoint("ATT_SPLIT_ORDER", "extract")
                self.assertEqual("blocked", checkpoint["status"])
                segment_job_ids = checkpoint["payload"]["segment_job_ids"]
                # Exactly this run's two segment jobs, in page order — the
                # legacy whole-document job must not be listed.
                self.assertEqual(2, len(segment_job_ids))
                self.assertNotIn("legacy-whole-doc-job", segment_job_ids)
                segment_jobs = [ledger.get_extract_job(job_id=job_id) for job_id in segment_job_ids]
                self.assertEqual(["1-200", "201-250"], [job["selected_pages"] for job in segment_jobs])
                # states must align with the ids (derived from the same list).
                self.assertEqual("downloaded", checkpoint["payload"]["states"][0])
                self.assertEqual("failed_retryable", checkpoint["payload"]["states"][1])
            finally:
                ledger.close()

    def test_merge_and_checkpoint_sorts_segments_by_page(self) -> None:
        """H3: _merge_and_checkpoint_segments merges in page order even when
        handed jobs in a scrambled order (e.g. resumed from a legacy
        checkpoint persisted in ledger order)."""
        with workspace_tmpdir("ingest-merge-order-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                attachment = build_attachment_with_file(tmpdir, "ATT_MERGE_ORDER", title="Merge Order")
                ledger.upsert_attachments([attachment])
                # job_events has a FK to the jobs table; create a real job row.
                pipeline_job_id = ledger.create_job_if_no_active("ingest", {"mode": "incremental"})

                seg1_dir = tmpdir / "seg1"
                seg1_dir.mkdir()
                (seg1_dir / "full.md").write_text("pages 1-200 content\n", encoding="utf-8")
                seg2_dir = tmpdir / "seg2"
                seg2_dir.mkdir()
                (seg2_dir / "full.md").write_text("pages 201-250 content\n", encoding="utf-8")

                scrambled_jobs = [
                    {
                        "job_id": "seg-job-2",
                        "selected_pages": "201-250",
                        "state": "downloaded",
                        "artifact_dir": str(seg2_dir),
                    },
                    {
                        "job_id": "seg-job-1",
                        "selected_pages": "1-200",
                        "state": "downloaded",
                        "artifact_dir": str(seg1_dir),
                    },
                ]
                _merge_and_checkpoint_segments(
                    ledger=ledger,
                    job_id=pipeline_job_id,
                    attachment_key="ATT_MERGE_ORDER",
                    segment_jobs=scrambled_jobs,
                    extract_cache_dir=tmpdir / "extract_cache",
                )

                checkpoint = ledger.get_checkpoint("ATT_MERGE_ORDER", "extract")
                self.assertEqual("done", checkpoint["status"])
                self.assertEqual(
                    ["seg-job-1", "seg-job-2"],
                    checkpoint["payload"]["segment_job_ids"],
                )
                merged = Path(checkpoint["payload"]["merged_markdown"]).read_text(encoding="utf-8")
                self.assertLess(merged.index("pages 1-200"), merged.index("pages 201-250"))
            finally:
                ledger.close()

    def test_merge_and_checkpoint_force_remerges_truncated_file(self) -> None:
        """H2: force=True re-merges even when merged_full.md already exists
        (e.g. truncated by a killed run); the default keeps exists-skip."""
        with workspace_tmpdir("ingest-merge-force-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                attachment = build_attachment_with_file(tmpdir, "ATT_MERGE_FORCE", title="Merge Force")
                ledger.upsert_attachments([attachment])
                # job_events has a FK to the jobs table; create a real job row.
                pipeline_job_id = ledger.create_job_if_no_active("ingest", {"mode": "incremental"})

                seg1_dir = tmpdir / "seg1"
                seg1_dir.mkdir()
                (seg1_dir / "full.md").write_text("pages 1-200 content\n", encoding="utf-8")
                jobs = [
                    {
                        "job_id": "seg-job-1",
                        "selected_pages": "1-200",
                        "state": "downloaded",
                        "artifact_dir": str(seg1_dir),
                    }
                ]

                merged_dir = tmpdir / "merged_out"
                merged_dir.mkdir()
                (merged_dir / "merged_full.md").write_text("TRUNCATED", encoding="utf-8")

                # Default: exists-skip keeps the truncated file untouched.
                _merge_and_checkpoint_segments(
                    ledger=ledger,
                    job_id=pipeline_job_id,
                    attachment_key="ATT_MERGE_FORCE",
                    segment_jobs=jobs,
                    extract_cache_dir=tmpdir / "extract_cache",
                    merged_artifact_dir=merged_dir,
                )
                self.assertEqual("TRUNCATED", (merged_dir / "merged_full.md").read_text(encoding="utf-8"))

                # force=True repairs it.
                _merge_and_checkpoint_segments(
                    ledger=ledger,
                    job_id=pipeline_job_id,
                    attachment_key="ATT_MERGE_FORCE",
                    segment_jobs=jobs,
                    extract_cache_dir=tmpdir / "extract_cache",
                    merged_artifact_dir=merged_dir,
                    force=True,
                )
                self.assertEqual(
                    "pages 1-200 content\n",
                    (merged_dir / "merged_full.md").read_text(encoding="utf-8"),
                )
            finally:
                ledger.close()

    def test_full_rebuild_repairs_truncated_merged_markdown(self) -> None:
        """H2: --mode full re-merges segment artifacts even when a (truncated)
        merged_full.md already exists; an incremental run leaves it alone."""
        with workspace_tmpdir("ingest-full-remerge-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_FULL_REMERGE", title="Full Remerge")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                extract_config = ExtractConfig(split_large_pdfs=True, max_pages_per_extraction=200)

                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_FULL_REMERGE",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )
                self.assertEqual("completed", result["job"]["status"])

                checkpoint = ledger.get_checkpoint("ATT_FULL_REMERGE", "extract")
                merged_path = Path(checkpoint["payload"]["merged_markdown"])
                healthy = merged_path.read_text(encoding="utf-8")
                self.assertIn("## Part 2", healthy)

                # Simulate a crash-truncated merged file.
                merged_path.write_text("TRUNCATED", encoding="utf-8")

                # Incremental re-run: extract stage already done, so the
                # truncated file is left untouched (exists-skip by design).
                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result2 = start_ingest_job(
                        ledger,
                        zotero_key="ATT_FULL_REMERGE",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )
                self.assertEqual("completed", result2["job"]["status"])
                self.assertEqual("TRUNCATED", merged_path.read_text(encoding="utf-8"))

                # Full rebuild: segment jobs cache-hit as downloaded, then the
                # force merge repairs the truncated file.
                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result3 = start_ingest_job(
                        ledger,
                        zotero_key="ATT_FULL_REMERGE",
                        include_multimodal=False,
                        mode="full",
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )
                self.assertEqual("completed", result3["job"]["status"])
                repaired = merged_path.read_text(encoding="utf-8")
                self.assertIn("# Demo Paper", repaired)
                self.assertIn("## Part 2", repaired)
            finally:
                ledger.close()

    def test_split_segments_blocked_while_remote_running_and_resume_finishes(self) -> None:
        """M9: a split PDF whose segments are still running remotely ends the
        run as blocked — never normalized from a fragment — and resume
        --execute finishes it once the remote batches complete."""
        with workspace_tmpdir("ingest-split-blocked-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_BLOCKED", title="Blocked Paper")
                ledger.upsert_attachments([attachment])

                class PendingSegmentProvider(MarkdownStubExtractorProvider):
                    def submit(
                        self,
                        input_file: Path,
                        options_hash: str,
                        *,
                        options: dict[str, Any] | None = None,
                        api_key: str | None = None,
                    ) -> ExtractJobState:
                        page_ranges = (options or {}).get("page_ranges", "")
                        if page_ranges == "1-200":
                            return ExtractJobState(external_job_id="seg1", state="completed")
                        return ExtractJobState(external_job_id="seg2", state="running")

                    def poll(
                        self,
                        external_job_id: str,
                        *,
                        api_key: str | None = None,
                        options: dict[str, Any] | None = None,
                    ) -> ExtractJobState:
                        if external_job_id == "seg1":
                            return ExtractJobState(external_job_id=external_job_id, state="completed")
                        # Segment 2 never completes within the (zeroed) budget.
                        return ExtractJobState(external_job_id=external_job_id, state="running")

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=PendingSegmentProvider(),
                    poll_interval_seconds=0.0,
                )
                extract_config = ExtractConfig(split_large_pdfs=True, max_pages_per_extraction=200)

                # A zero poll budget trips the M2 timeout on the first check;
                # the round stands down (no exception) and the job stays running.
                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    with patch.object(ExtractionManager, "_poll_budget_seconds", return_value=0.0):
                        result = start_ingest_job(
                            ledger,
                            zotero_key="ATT_BLOCKED",
                            include_multimodal=False,
                            execute=True,
                            extract_manager=extract_manager,
                            extract_cache_dir=tmpdir / "extract_cache",
                            normalized_dir=tmpdir / "normalized",
                            vector_store_dir=tmpdir / "vectors",
                            extract_config=extract_config,
                        )

                self.assertEqual("blocked", result["job"]["status"])
                self.assertEqual(1, len(result["blocked"]))
                self.assertEqual(0, len(result["failed"]))
                checkpoint = ledger.get_checkpoint("ATT_BLOCKED", "extract")
                self.assertEqual("blocked", checkpoint["status"])
                # Short-circuit: normalize/embed never ran on the incomplete
                # split, so no fragment could be normalized as full text.
                self.assertIsNone(ledger.get_checkpoint("ATT_BLOCKED", "normalize"))
                self.assertIsNone(ledger.get_normalized_artifact("ATT_BLOCKED"))

                segment_job_ids = checkpoint["payload"]["segment_job_ids"]
                seg2_job = ledger.get_extract_job(job_id=segment_job_ids[1])
                # M2: the batch id survived the timeout; the job is resumable.
                self.assertEqual("running", seg2_job["state"])
                self.assertEqual("seg2", seg2_job["external_job_id"])

                # Segment 2 completes remotely (completed, not downloaded, so
                # the resume path downloads and merges it for real).
                ledger.set_extract_job_state(
                    seg2_job["job_id"],
                    state="completed",
                    local_stage="download",
                    external_job_id=seg2_job["external_job_id"],
                )

                job_id = result["job"]["job_id"]
                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result2 = resume_ingest_job(
                        ledger,
                        job_id,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )

                self.assertEqual("completed", result2["job"]["status"])
                self.assertEqual("done", ledger.get_checkpoint("ATT_BLOCKED", "extract")["status"])
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_BLOCKED"))
            finally:
                ledger.close()

    def test_circuit_breaker_aborts_on_systemic_failure(self) -> None:
        """M9: a run of identical failures (here: every file missing) trips
        the breaker and ends the job as failed instead of grinding through
        all documents."""
        with workspace_tmpdir("ingest-breaker-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                for index in range(6):
                    ledger.upsert_attachments(
                        [
                            build_attachment(
                                f"ATT_BRK{index}",
                                title=f"Broken {index}",
                                file_exists=False,
                            )
                        ]
                    )

                result = start_ingest_job(
                    ledger,
                    include_multimodal=False,
                    execute=True,
                    extract_manager=ExtractionManager(
                        ledger=ledger,
                        cache_dir=tmpdir / "extract_cache",
                        provider=MarkdownStubExtractorProvider(),
                    ),
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                    extract_config=ExtractConfig(),
                )

                self.assertEqual("failed", result["job"]["status"])
                # The breaker tripped at the threshold; the 6th document was
                # never attempted.
                self.assertEqual(5, len(result["failed"]))
                self.assertEqual(0, len(result["executed"]))
                events = result["job"].get("events") or []
                self.assertTrue(any(event["stage"] == "circuit_breaker" for event in events))
            finally:
                ledger.close()

    def test_normalize_refuses_segment_fragment_for_incomplete_split(self) -> None:
        """M9 guard: a segmented extraction whose checkpoint never reached
        'done' must fail loudly instead of normalizing a single segment
        fragment as the full text."""
        with workspace_tmpdir("ingest-fragment-guard-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_FRAG", title="Fragment Paper")
                ledger.upsert_attachments([attachment])

                # One fully downloaded segment artifact exists on disk…
                provider = MarkdownStubExtractorProvider()
                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=provider,
                )
                seg_result = extract_manager.ensure_extraction(
                    ExtractionRequest(
                        input_file=Path(attachment["file_path"]),
                        attachment_key="ATT_FRAG",
                        selected_pages="1-200",
                        selected_page_count=200,
                    )
                )
                self.assertEqual("downloaded", seg_result.job["state"])
                # …but the extract checkpoint says the split is still running
                # (crash between segment completion and merge).
                ledger.checkpoint(
                    "ATT_FRAG",
                    "extract",
                    "running",
                    {"job_id": "old-job", "segment_job_ids": [seg_result.job["job_id"]]},
                )

                with self.assertRaises(ValueError) as ctx:
                    _execute_normalize_stage(
                        ledger=ledger,
                        job_id="job-x",
                        attachment_key="ATT_FRAG",
                        document_id="doc-frag",
                        document={},
                        normalized_dir=tmpdir / "normalized",
                    )
                self.assertIn("incomplete", str(ctx.exception))
            finally:
                ledger.close()

    def test_split_disabled_uses_single_job(self) -> None:
        with workspace_tmpdir("ingest-no-split-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_NOSPLIT", title="Long Paper No Split")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                extract_config = ExtractConfig(split_large_pdfs=False, max_pages_per_extraction=200)

                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_NOSPLIT",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )

                self.assertEqual("completed", result["job"]["status"])
                checkpoint = ledger.get_checkpoint("ATT_NOSPLIT", "extract")
                self.assertNotIn("segment_job_ids", checkpoint["payload"])
                jobs = [
                    job for job in ledger.list_extract_jobs(limit=None)
                    if job.get("attachment_key") == "ATT_NOSPLIT"
                ]
                self.assertEqual(1, len(jobs))
            finally:
                ledger.close()

    def test_single_job_passes_real_page_count_to_poll_budget(self) -> None:
        with workspace_tmpdir("ingest-nosplit-pages-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_PAGES", title="Known Pages Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                extract_config = ExtractConfig(split_large_pdfs=False, max_pages_per_extraction=200)

                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_PAGES",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )

                self.assertEqual("completed", result["job"]["status"])
                jobs = [
                    job for job in ledger.list_extract_jobs(limit=None)
                    if job.get("attachment_key") == "ATT_PAGES"
                ]
                self.assertEqual(1, len(jobs))
                options = (jobs[0].get("payload") or {}).get("options") or {}
                # 250 known pages -> 250*6+30, not the old 1-page 36s budget.
                self.assertEqual(250 * 6 + 30, options.get("timeout_seconds"))
            finally:
                ledger.close()

    def test_single_job_unknown_page_count_uses_conservative_fallback(self) -> None:
        with workspace_tmpdir("ingest-nosplit-unknown-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_NOPAGES", title="Unknown Pages Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                extract_config = ExtractConfig(split_large_pdfs=False, max_pages_per_extraction=200)

                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=None):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_NOPAGES",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )

                self.assertEqual("completed", result["job"]["status"])
                jobs = [
                    job for job in ledger.list_extract_jobs(limit=None)
                    if job.get("attachment_key") == "ATT_NOPAGES"
                ]
                self.assertEqual(1, len(jobs))
                options = (jobs[0].get("payload") or {}).get("options") or {}
                # Unknown page count -> conservative 500-page budget (3030s)
                # instead of the 36s budget that caused premature resume
                # failures and full-PDF resubmissions.
                self.assertEqual(500 * 6 + 30, options.get("timeout_seconds"))
            finally:
                ledger.close()


def seed_profiles(ledger: StateLedger) -> None:
    ledger.upsert_embedding_profiles(
        [
            EmbeddingProfile(
                name="text-profile",
                provider="stub",
                model="stub",
                dimension=8,
                modality="text",
                enabled=True,
                default_for_text=True,
                backend="sqlite-local",
            ),
            EmbeddingProfile(
                name="mm-profile",
                provider="stub",
                model="stub-mm",
                dimension=8,
                modality="multimodal",
                enabled=True,
                default_for_multimodal=True,
                backend="sqlite-local",
            ),
        ]
    )


def profile_hash(ledger: StateLedger, profile_name: str) -> str:
    profile = next(profile for profile in ledger.list_embedding_profiles() if profile["name"] == profile_name)
    return embedding_profile_hash(profile)


def seed_attachments(ledger: StateLedger) -> None:
    ledger.upsert_attachments(
        [
            build_attachment("ATT_DONE", title="Already Indexed"),
            build_attachment("ATT_NEW", title="Needs Work"),
            build_attachment("ATT_MISSING", title="Missing PDF", file_exists=False),
            build_attachment("ATT_REVIEW", title="Review PDF", classification="needs_review"),
        ]
    )


def normalize_fixture(tmpdir, ledger: StateLedger):
    source_dir = tmpdir / "mineru"
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "hash.png").write_bytes(b"image")
    markdown = source_dir / "full.md"
    markdown.write_text("# Already Indexed\n\nBody.\n\n![Fig](images/hash.png)\n", encoding="utf-8")
    normalized = normalize_markdown_document(
        source_markdown=markdown,
        output_root=tmpdir / "normalized",
        document_id="DOC_DONE",
        attachment_key="ATT_DONE",
    )
    ledger.upsert_normalized_artifact(normalized.ledger_artifact())
    ledger.replace_document_chunks(normalized.document_id, normalized.chunks)
    return normalized


def build_attachment(
    attachment_key: str,
    *,
    title: str,
    classification: str = "included_auto",
    file_exists: bool = True,
) -> dict[str, object]:
    return {
        "attachment_key": attachment_key,
        "parent_key": f"PARENT-{attachment_key}",
        "content_type": "application/pdf",
        "relative_path": f"storage:{attachment_key}.pdf",
        "title": title,
        "abstract": "",
        "date": "2024",
        "url": "",
        "classification": classification,
        "source_quality": "primary_candidate",
        "reasons": [],
        "file_path": f"C:/Zotero/storage/{attachment_key}.pdf",
        "file_exists": file_exists,
        "file_size": 100 if file_exists else None,
        "file_mtime": 1.0 if file_exists else None,
        "metadata": {},
    }


def build_attachment_with_file(
    tmpdir: Path,
    attachment_key: str,
    *,
    title: str,
    classification: str = "included_auto",
) -> dict[str, object]:
    """Build an attachment dict whose file_path points to an existing file."""

    file_path = tmpdir / "attachments" / f"{attachment_key}.pdf"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"fake pdf content")
    attachment = build_attachment(
        attachment_key,
        title=title,
        classification=classification,
        file_exists=True,
    )
    attachment["file_path"] = str(file_path)
    return attachment


class MarkdownStubExtractorProvider(StubExtractorProvider):
    """Stub extractor that produces a markdown file suitable for normalization."""

    def download(
        self,
        external_job_id: str,
        output_dir: Path,
        *,
        api_key: str | None = None,
        source_pdf: str | Path | None = None,
        options: dict[str, object] | None = None,
    ) -> ExtractArtifact:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown = output_dir / "full.md"
        markdown.write_text("# Demo Paper\n\nalpha beta gamma.\n", encoding="utf-8")
        manifest = output_dir / "manifest.json"
        source_pdf_path = Path(source_pdf) if source_pdf is not None else Path(external_job_id)
        manifest.write_text(
            json.dumps(
                {
                    "provider": self.name,
                    "provider_version": self.version,
                    "external_job_id": external_job_id,
                    "state": "downloaded",
                    "source_pdf": str(source_pdf_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return ExtractArtifact(
            source_pdf=source_pdf_path,
            artifact_dir=output_dir,
            manifest_path=manifest,
        )


class TwoPhaseStubExtractorProvider(MarkdownStubExtractorProvider):
    """Stub extractor that returns 'running' on the first poll, then 'completed'."""

    name = "two-phase-stub"
    version = "0"

    def __init__(self) -> None:
        self._poll_count = 0

    def submit(
        self,
        input_file: Path,
        options_hash: str,
        *,
        options: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> ExtractJobState:
        return ExtractJobState(external_job_id=self.fingerprint(input_file, options_hash), state="running")

    def poll(
        self,
        external_job_id: str,
        *,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractJobState:
        self._poll_count += 1
        if self._poll_count == 1:
            return ExtractJobState(external_job_id=external_job_id, state="running")
        return ExtractJobState(external_job_id=external_job_id, state="completed")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Phase 2: concurrent harvest of split PDF segments (§2.2 two-pass flow).
# ---------------------------------------------------------------------------

import threading  # noqa: E402

from zoterorag.extractors.key_pool import ApiKeyRef, ExtractorKeyPool  # noqa: E402


class OrderingRecordingProvider(MarkdownStubExtractorProvider):
    """Thread-safe stub that records submit/poll events in call order.

    submit returns 'running' (so segments stay in flight until harvested)
    and poll completes immediately on first call.
    """

    name = "ordering-stub"
    version = "0"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[tuple[str, str]] = []

    def submit(
        self,
        input_file: Path,
        options_hash: str,
        *,
        options: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> ExtractJobState:
        page_ranges = str((options or {}).get("page_ranges", ""))
        with self._lock:
            self.events.append(("submit", page_ranges))
        return ExtractJobState(external_job_id=f"batch-{page_ranges}", state="running")

    def poll(
        self,
        external_job_id: str,
        *,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractJobState:
        with self._lock:
            self.events.append(("poll", external_job_id))
        return ExtractJobState(external_job_id=external_job_id, state="completed")


class ConcurrentHarvestTests(unittest.TestCase):
    def test_resume_partial_download_keeps_saved_two_page_segments(self) -> None:
        class SegmentProvider(OrderingRecordingProvider):
            name = "deepseek"

            def download(self, external_job_id, output_dir, **kwargs):
                artifact = super().download(external_job_id, output_dir, **kwargs)
                (artifact.artifact_dir / "full.md").write_text(external_job_id + "\n", encoding="utf-8")
                self.events.append(("download", external_job_id))
                return artifact

        with workspace_tmpdir("ingest-saved-split-") as root, patch(
            "zoterorag.pipeline.ingest.get_pdf_page_count", return_value=5
        ):
            cache = root / "extract_cache"
            db = root / "state.sqlite"
            key = "ATT_FIVE_PAGES"
            paths = dict(extract_cache_dir=cache, normalized_dir=root / "normalized", vector_store_dir=root / "vectors")
            with closing(StateLedger(db)) as ledger:
                seed_profiles(ledger)
                ledger.upsert_attachments([build_attachment_with_file(root, key, title="Five pages")])
                manager = ExtractionManager(ledger=ledger, cache_dir=cache, provider=SegmentProvider())
                submit = manager.submit_only

                def interrupt_after_download(request):
                    submitted = submit(request)
                    manager.harvest(submitted.job["job_id"])
                    raise KeyboardInterrupt

                with patch.object(manager, "submit_only", side_effect=interrupt_after_download), self.assertRaises(KeyboardInterrupt):
                    start_ingest_job(
                        ledger, zotero_key=key, include_multimodal=False, execute=True, extract_manager=manager,
                        extract_config=ExtractConfig(deepseek_pages_per_request=2), **paths,
                    )
                checkpoint = ledger.get_checkpoint(key, "extract")
                self.assertEqual("running", checkpoint["status"])
                self.assertEqual(2, checkpoint["payload"]["segment_size"])
                self.assertNotIn("segment_job_ids", checkpoint["payload"])
                (first,) = ledger.list_extract_jobs()
                self.assertEqual(("1-2", "downloaded"), (first["selected_pages"], first["state"]))
                (ingest_job,) = ledger.list_jobs(kind="ingest")
                self.assertEqual("interrupted", ingest_job["status"])

            with closing(StateLedger(db)) as ledger:
                provider = SegmentProvider()
                manager = ExtractionManager(ledger=ledger, cache_dir=cache, provider=provider)
                result = resume_ingest_job(
                    ledger, ingest_job["job_id"], execute=True, extract_manager=manager,
                    extract_config=ExtractConfig(deepseek_pages_per_request=7), **paths,
                )
                self.assertEqual("completed", result["job"]["status"])
                self.assertCountEqual(provider.events, [
                    ("submit", "3-4"), ("poll", "batch-3-4"), ("download", "batch-3-4"),
                    ("submit", "5-5"), ("poll", "batch-5-5"), ("download", "batch-5-5"),
                ])
                done = ledger.get_checkpoint(key, "extract")
                self.assertEqual("done", done["status"])
                ids = done["payload"]["segment_job_ids"]
                self.assertEqual(first["job_id"], ids[0])
                pages = [ledger.get_extract_job(job_id=job_id)["selected_pages"] for job_id in ids]
                self.assertEqual(["1-2", "3-4", "5-5"], pages)
                merged = Path(done["payload"]["merged_markdown"]).read_text(encoding="utf-8")
                self.assertEqual(["batch-1-2", "batch-3-4", "batch-5-5"],
                                 [line for line in merged.splitlines() if line.startswith("batch-")])

    def test_harvest_manual_review_with_stale_poll_stage(self) -> None:
        result = ExtractionResult(job={"state": "failed_manual_review", "local_stage": "poll",
                                       "external_job_id": "batch", "job_id": "segment"}, cache_hit=True)
        manager = Mock(spec=ExtractionManager)
        self.assertIs(result, _harvest_one(result, manager))
        manager.harvest.assert_not_called()
        manager.resume_extraction.assert_not_called()

    def test_all_segments_submitted_before_any_poll_and_keys_released(self) -> None:
        with workspace_tmpdir("ingest-phase2-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                attachment = build_attachment_with_file(tmpdir, "ATT_PHASE2", title="Concurrent Paper")
                ledger.upsert_attachments([attachment])
                provider = OrderingRecordingProvider()
                pool = ExtractorKeyPool(
                    [
                        ApiKeyRef(alias="mineru_a", secret="s-a"),
                        ApiKeyRef(alias="mineru_b", secret="s-b"),
                    ]
                )
                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=provider,
                    key_pool=pool,
                    poll_interval_seconds=0.0,
                )
                extract_config = ExtractConfig(
                    split_large_pdfs=True,
                    max_pages_per_extraction=200,
                    harvest_workers=2,
                )
                with patch("zoterorag.pipeline.ingest.get_pdf_page_count", return_value=250):
                    result = start_ingest_job(
                        ledger,
                        zotero_key="ATT_PHASE2",
                        include_multimodal=False,
                        execute=True,
                        extract_manager=extract_manager,
                        extract_cache_dir=tmpdir / "extract_cache",
                        normalized_dir=tmpdir / "normalized",
                        vector_store_dir=tmpdir / "vectors",
                        extract_config=extract_config,
                    )

                self.assertEqual("completed", result["job"]["status"])
                # Pass 1 submits every segment in page order before pass 2
                # polls anything (segments-first submission, §2.2).
                kinds = [kind for kind, _ in provider.events]
                submits = [value for kind, value in provider.events if kind == "submit"]
                self.assertEqual(["1-200", "201-250"], submits)
                self.assertEqual(["submit", "submit"], kinds[:2])
                self.assertNotIn("poll", kinds[:2])
                # All keys released after the run (no in-flight leaks).
                self.assertTrue(all(key["in_flight"] == 0 for key in pool.list_public_keys()))
            finally:
                ledger.close()


# ---------------------------------------------------------------------------
# Phase 3: run-level batch publish of embed stages (ingest.batch_publish).
# ---------------------------------------------------------------------------

from zoterorag.pipeline import ingest as ingest_module  # noqa: E402


class BatchPublishIngestTests(unittest.TestCase):
    def _run_ingest(self, tmpdir: Path, ledger: StateLedger, *, batch_publish: bool = True):
        extract_manager = ExtractionManager(
            ledger=ledger,
            cache_dir=tmpdir / "extract_cache",
            provider=MarkdownStubExtractorProvider(),
        )
        return start_ingest_job(
            ledger,
            include_multimodal=True,
            execute=True,
            extract_manager=extract_manager,
            extract_cache_dir=tmpdir / "extract_cache",
            normalized_dir=tmpdir / "normalized",
            vector_store_dir=tmpdir / "vectors",
            batch_publish=batch_publish,
        )

    def _seed_two_attachments(self, tmpdir: Path, ledger: StateLedger) -> None:
        seed_profiles(ledger)
        for key in ("ATT_BP1", "ATT_BP2"):
            ledger.upsert_attachments([build_attachment_with_file(tmpdir, key, title=f"Paper {key}")])

    def test_batch_publish_publishes_once_per_profile(self) -> None:
        with workspace_tmpdir("ingest-batch-publish-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                self._seed_two_attachments(tmpdir, ledger)
                real_publish = ingest_module.publish_staged_documents
                with patch(
                    "zoterorag.pipeline.ingest.publish_staged_documents",
                    side_effect=real_publish,
                ) as publish_spy:
                    result = self._run_ingest(tmpdir, ledger)

                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual(2, len(result["executed"]))
                self.assertEqual(0, len(result["failed"]))
                # 2 documents x 2 profiles would be 4 per-document publishes;
                # batch publish does exactly one per profile.
                self.assertEqual(2, publish_spy.call_count)
                for key in ("ATT_BP1", "ATT_BP2"):
                    self.assertEqual("indexed", ledger.get_checkpoint(key, "embed:text-profile")["status"])
                    self.assertEqual("indexed", ledger.get_checkpoint(key, "embed:mm-profile")["status"])
            finally:
                ledger.close()

    def test_batch_publish_false_falls_back_to_per_document_publish(self) -> None:
        with workspace_tmpdir("ingest-batch-publish-off-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                self._seed_two_attachments(tmpdir, ledger)
                real_index = ingest_module.index_normalized_document
                with patch(
                    "zoterorag.pipeline.ingest.index_normalized_document",
                    side_effect=real_index,
                ) as index_spy, patch(
                    "zoterorag.pipeline.ingest.publish_staged_documents",
                ) as publish_spy:
                    result = self._run_ingest(tmpdir, ledger, batch_publish=False)

                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual(2, len(result["executed"]))
                # Legacy path: one index call per document per profile, and
                # the run-level batch publish never engages.
                self.assertEqual(4, index_spy.call_count)
                publish_spy.assert_not_called()
                for key in ("ATT_BP1", "ATT_BP2"):
                    self.assertEqual("indexed", ledger.get_checkpoint(key, "embed:text-profile")["status"])
                    self.assertEqual("indexed", ledger.get_checkpoint(key, "embed:mm-profile")["status"])
            finally:
                ledger.close()

    def test_partial_publish_failure_marks_only_failed_groups_documents(self) -> None:
        with workspace_tmpdir("ingest-batch-publish-partial-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                self._seed_two_attachments(tmpdir, ledger)
                real_publish = ingest_module.publish_staged_documents

                def failing_publish(*, ledger, staged):
                    if staged and staged[0].profile_name == "mm-profile":
                        raise RuntimeError("simulated mm publish failure")
                    return real_publish(ledger=ledger, staged=staged)

                with patch(
                    "zoterorag.pipeline.ingest.publish_staged_documents",
                    side_effect=failing_publish,
                ):
                    result = self._run_ingest(tmpdir, ledger)

                self.assertEqual("completed_with_errors", result["job"]["status"])
                # Both documents staged the mm profile, so both fail; the text
                # profile group published fine.
                self.assertEqual(2, len(result["failed"]))
                self.assertEqual(0, len(result["executed"]))
                for key in ("ATT_BP1", "ATT_BP2"):
                    # Text profile published; mm stays "running" so the next
                    # run retries it.
                    self.assertEqual("indexed", ledger.get_checkpoint(key, "embed:text-profile")["status"])
                    self.assertEqual("running", ledger.get_checkpoint(key, "embed:mm-profile")["status"])
                events = ledger.list_job_events(result["job"]["job_id"])
                publish_failures = [
                    event for event in events
                    if event["stage"] == "embed_publish" and event["status"] == "failed"
                ]
                self.assertEqual(1, len(publish_failures))
                self.assertEqual("mm-profile", publish_failures[0]["payload"]["profile_name"])
            finally:
                ledger.close()
