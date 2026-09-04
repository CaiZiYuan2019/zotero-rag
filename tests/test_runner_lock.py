from __future__ import annotations

import threading
import unittest

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.pipeline.runner import acquire_runner_lock


class CreateJobIfNoActiveTests(unittest.TestCase):
    """M10 ledger layer: atomic check-then-create in one BEGIN IMMEDIATE
    transaction."""

    def test_creates_job_when_nothing_active(self) -> None:
        with workspace_tmpdir("job-if-no-active-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                job_id = ledger.create_job_if_no_active("ingest", {"mode": "incremental"})
                self.assertEqual("created", ledger.get_job(job_id)["status"])
            finally:
                ledger.close()

    def test_blocks_on_created_and_running_of_either_kind(self) -> None:
        with workspace_tmpdir("job-if-no-active-block-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                for kind in ("ingest", "reembed"):
                    for status in ("created", "running"):
                        existing = ledger.create_job(kind, {})
                        if status != "created":
                            ledger.set_job_status(existing, status)
                        with self.assertRaises(RuntimeError) as ctx:
                            ledger.create_job_if_no_active("ingest", {})
                        self.assertIn(existing, str(ctx.exception))
                        # Clean up for the next iteration.
                        ledger.set_job_status(existing, "failed")
            finally:
                ledger.close()

    def test_ignores_terminal_planned_and_paused_jobs(self) -> None:
        with workspace_tmpdir("job-if-no-active-terminal-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                for status in (
                    "completed",
                    "completed_with_errors",
                    "blocked",
                    "failed",
                    "interrupted",
                    "cancelled",
                    "planned",
                    "paused",
                ):
                    job_id = ledger.create_job("ingest", {})
                    ledger.set_job_status(job_id, status)
                # Unrelated kinds are never considered.
                other = ledger.create_job("zotero_scan", {})
                ledger.set_job_status(other, "running")
                new_id = ledger.create_job_if_no_active("reembed", {})
                self.assertIsNotNone(new_id)
            finally:
                ledger.close()

    def test_concurrent_creation_single_winner_in_process(self) -> None:
        with workspace_tmpdir("job-if-no-active-threads-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                successes: list[str] = []
                failures: list[Exception] = []
                barrier = threading.Barrier(8)

                def attempt() -> None:
                    try:
                        barrier.wait(timeout=10)
                        successes.append(ledger.create_job_if_no_active("ingest", {}))
                    except RuntimeError as exc:
                        failures.append(exc)

                threads = [threading.Thread(target=attempt) for _ in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=30)
                self.assertEqual(1, len(successes))
                self.assertEqual(7, len(failures))
            finally:
                ledger.close()

    def test_concurrent_creation_across_connections(self) -> None:
        """Two ledgers on the same file approximate the two-process case."""
        with workspace_tmpdir("job-if-no-active-conns-") as tmpdir:
            first = StateLedger(tmpdir / "state.sqlite")
            second = StateLedger(tmpdir / "state.sqlite")
            try:
                first.create_job_if_no_active("ingest", {})
                with self.assertRaises(RuntimeError):
                    second.create_job_if_no_active("reembed", {})
            finally:
                first.close()
                second.close()


class RunnerLockTests(unittest.TestCase):
    """M10 file-lock layer: liveness detection and stale-run sweeping."""

    def test_free_lock_sweeps_stale_created_and_running_jobs(self) -> None:
        with workspace_tmpdir("runner-sweep-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                running = ledger.create_job("ingest", {})
                ledger.set_job_status(running, "running")
                created = ledger.create_job("reembed", {})

                with acquire_runner_lock(ledger):
                    # The sweep ran before the lock was yielded.
                    self.assertIsNone(ledger.find_active_job())
                    self.assertEqual("interrupted", ledger.get_job(running)["status"])
                    self.assertEqual("interrupted", ledger.get_job(created)["status"])

                events = ledger.list_job_events(running)
                sweep = next(e for e in events if e["stage"] == "lifecycle")
                self.assertIn("runner lock", sweep["message"])
                self.assertEqual({"previous_status": "running"}, sweep["payload"])
            finally:
                ledger.close()

    def test_held_lock_refuses_second_runner(self) -> None:
        with workspace_tmpdir("runner-held-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                with acquire_runner_lock(ledger):
                    with self.assertRaises(RuntimeError) as ctx:
                        with acquire_runner_lock(ledger):
                            pass
                    self.assertIn("runner process is active", str(ctx.exception))
            finally:
                ledger.close()

    def test_planned_and_paused_jobs_are_not_swept(self) -> None:
        with workspace_tmpdir("runner-no-sweep-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                planned = ledger.create_job("ingest", {})
                ledger.set_job_status(planned, "planned")
                paused = ledger.create_job("reembed", {})
                ledger.set_job_status(paused, "paused")
                with acquire_runner_lock(ledger):
                    pass
                self.assertEqual("planned", ledger.get_job(planned)["status"])
                self.assertEqual("paused", ledger.get_job(paused)["status"])
            finally:
                ledger.close()

    def test_lock_released_after_context_exit(self) -> None:
        with workspace_tmpdir("runner-release-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                with acquire_runner_lock(ledger):
                    pass
                # A second acquisition after release must succeed immediately.
                with acquire_runner_lock(ledger):
                    pass
            finally:
                ledger.close()
