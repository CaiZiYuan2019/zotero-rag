from __future__ import annotations

import threading
import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db.state import JobEvent, SCHEMA_VERSION, StateLedger


class StateLedgerTests(unittest.TestCase):
    def test_job_event_and_checkpoint_persist_across_reopen(self) -> None:
        with workspace_tmpdir("state-ledger-") as tmpdir:
            db_path = tmpdir / "state.sqlite"

            ledger = StateLedger(db_path)
            try:
                job_id = ledger.create_job("shadow-sync", payload={"source": "zotero"})
                ledger.set_job_status(job_id, "running")
                ledger.add_event(
                    JobEvent(
                        job_id=job_id,
                        stage="shadow-copy",
                        status="ok",
                        message="copied",
                        payload={"rows": 12},
                    )
                )
                ledger.checkpoint(
                    subject_id="item-123",
                    stage="embed-text",
                    status="done",
                    payload={"chunks": 3},
                )
            finally:
                ledger.close()

            reopened = StateLedger(db_path)
            try:
                checkpoint = reopened.get_checkpoint("item-123", "embed-text")
                self.assertIsNotNone(checkpoint)
                self.assertEqual("done", checkpoint["status"])
                self.assertEqual({"chunks": 3}, checkpoint["payload"])

                summary = reopened.status_summary()
                self.assertEqual(SCHEMA_VERSION, summary["schema_version"])
                self.assertEqual(1, summary["checkpoints"])
                self.assertEqual(1, summary["jobs"]["running"])

                job_row = reopened.conn.execute(
                    "SELECT kind, status, payload_json FROM pipeline_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                self.assertIsNotNone(job_row)
                self.assertEqual("shadow-sync", job_row["kind"])
                self.assertEqual("running", job_row["status"])

                event_row = reopened.conn.execute(
                    """
                    SELECT stage, status, message, payload_json
                    FROM job_events
                    WHERE job_id = ?
                    ORDER BY event_id
                    """,
                    (job_id,),
                ).fetchone()
                self.assertIsNotNone(event_row)
                self.assertEqual("shadow-copy", event_row["stage"])
                self.assertEqual("ok", event_row["status"])
                self.assertEqual("copied", event_row["message"])
                self.assertIn('"rows": 12', event_row["payload_json"])

                job = reopened.get_job(job_id)
                self.assertIsNotNone(job)
                self.assertEqual("shadow-sync", job["kind"])
                self.assertEqual({"source": "zotero"}, job["payload"])
                self.assertEqual(1, len(job["events"]))
                self.assertEqual({"rows": 12}, job["events"][0]["payload"])

                jobs = reopened.list_jobs(kind="shadow-sync", status="running")
                self.assertEqual(1, len(jobs))
                self.assertEqual(job_id, jobs[0]["job_id"])
            finally:
                reopened.close()

    def test_checkpoint_upsert_replaces_status_and_payload(self) -> None:
        with workspace_tmpdir("state-ledger-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.checkpoint("item-7", "parse", "pending", {"attempt": 1})
                first = ledger.get_checkpoint("item-7", "parse")
                self.assertEqual("pending", first["status"])
                self.assertEqual({"attempt": 1}, first["payload"])

                ledger.checkpoint("item-7", "parse", "done", {"attempt": 2, "pages": 8})
                second = ledger.get_checkpoint("item-7", "parse")
                self.assertEqual("done", second["status"])
                self.assertEqual({"attempt": 2, "pages": 8}, second["payload"])
            finally:
                ledger.close()

    def test_vector_index_registration_is_listable_for_control_plane(self) -> None:
        with workspace_tmpdir("state-ledger-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.register_vector_index(
                    profile_name="qwen3vl_cloud_2560_text",
                    backend="sqlite-local",
                    path=tmpdir / "vectors.sqlite",
                    document_count=2,
                    chunk_count=5,
                    active=True,
                )
                indexes = ledger.list_vector_indexes()

                self.assertEqual(1, len(indexes))
                self.assertEqual("qwen3vl_cloud_2560_text", indexes[0]["profile_name"])
                self.assertEqual("sqlite-local", indexes[0]["backend"])
                self.assertEqual(2, indexes[0]["document_count"])
                self.assertEqual(5, indexes[0]["chunk_count"])
                self.assertTrue(indexes[0]["active"])
            finally:
                ledger.close()

    def test_embedding_batch_records_are_upserted_and_filterable(self) -> None:
        with workspace_tmpdir("state-ledger-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                first = ledger.upsert_embedding_batch(
                    batch_hash="b" * 64,
                    profile_name="profile-a",
                    profile_hash="p" * 64,
                    document_id="doc-1",
                    chunk_type="text",
                    chunk_count=2,
                    status="running",
                    provider="stub",
                    model="stub",
                    payload={"input_ids": ["c1", "c2"]},
                )
                second = ledger.upsert_embedding_batch(
                    batch_hash="b" * 64,
                    profile_name="profile-a",
                    profile_hash="p" * 64,
                    document_id="doc-1",
                    chunk_type="text",
                    chunk_count=2,
                    status="completed",
                    provider="stub",
                    model="stub",
                    payload={"input_ids": ["c1", "c2"], "indexed_chunks": 2},
                )

                self.assertEqual("running", first["status"])
                self.assertEqual(first["created_at"], second["created_at"])
                self.assertEqual("completed", second["status"])
                self.assertEqual(2, second["payload"]["indexed_chunks"])
                self.assertEqual(1, len(ledger.list_embedding_batches(profile_name="profile-a")))
                self.assertEqual(1, len(ledger.list_embedding_batches(document_id="doc-1", status="completed")))
                self.assertEqual({"completed": 1}, ledger.status_summary()["embedding_batches"])
            finally:
                ledger.close()

    def test_activate_embedding_profile_sets_one_default_and_survives_bootstrap(self) -> None:
        with workspace_tmpdir("state-ledger-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                profiles = [
                    EmbeddingProfile(
                        name="text-a",
                        provider="stub",
                        model="stub-a",
                        dimension=8,
                        modality="text",
                        enabled=True,
                        default_for_text=True,
                    ),
                    EmbeddingProfile(
                        name="text-b",
                        provider="stub",
                        model="stub-b",
                        dimension=8,
                        modality="text",
                        enabled=True,
                    ),
                    EmbeddingProfile(
                        name="mm-a",
                        provider="stub",
                        model="stub-mm",
                        dimension=8,
                        modality="multimodal",
                        enabled=True,
                        default_for_multimodal=True,
                    ),
                ]
                ledger.upsert_embedding_profiles(profiles)
                activated = ledger.activate_embedding_profile("text-b", "text")
                self.assertTrue(activated["default_for_text"])

                by_name = {item["name"]: item for item in ledger.list_embedding_profiles()}
                self.assertFalse(by_name["text-a"]["default_for_text"])
                self.assertTrue(by_name["text-b"]["default_for_text"])
                self.assertTrue(by_name["mm-a"]["default_for_multimodal"])

                ledger.upsert_embedding_profiles(profiles)
                by_name = {item["name"]: item for item in ledger.list_embedding_profiles()}
                self.assertFalse(by_name["text-a"]["default_for_text"])
                self.assertTrue(by_name["text-b"]["default_for_text"])

                with self.assertRaises(ValueError):
                    ledger.activate_embedding_profile("mm-a", "text")
            finally:
                ledger.close()

    def test_concurrent_writes_do_not_corrupt_state(self) -> None:
        with workspace_tmpdir("state-ledger-concurrent-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                errors: list[Exception] = []
                threads: list[threading.Thread] = []

                def worker(thread_id: int) -> None:
                    try:
                        for i in range(20):
                            job_id = ledger.create_job("test", payload={"thread": thread_id, "i": i})
                            ledger.set_job_status(job_id, "completed")
                            ledger.checkpoint(f"subject-{thread_id}-{i}", "stage", "done", {"i": i})
                    except Exception as exc:  # pragma: no cover
                        errors.append(exc)

                for t in range(5):
                    thread = threading.Thread(target=worker, args=(t,))
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                self.assertEqual([], errors)
                summary = ledger.status_summary()
                self.assertEqual(100, summary["jobs"]["completed"])
                self.assertEqual(100, summary["checkpoints"])
            finally:
                ledger.close()
