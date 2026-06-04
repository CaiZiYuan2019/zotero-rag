from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
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
