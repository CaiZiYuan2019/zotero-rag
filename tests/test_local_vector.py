from __future__ import annotations

import json
import logging
import sqlite3
import unittest
from unittest.mock import MagicMock

from tests._support import workspace_tmpdir
from zoterorag.index.local_vector import LocalVectorStore, VectorRecord


class LocalVectorStoreTests(unittest.TestCase):
    def test_search_skips_corrupt_vector_json(self) -> None:
        with workspace_tmpdir("local-vector-corrupt-") as tmpdir:
            store = LocalVectorStore(tmpdir / "vectors.sqlite", profile_name="p1", dimension=3)
            try:
                store.upsert(
                    [
                        VectorRecord(
                            record_id="p1:c1",
                            document_id="DOC1",
                            chunk_id="c1",
                            vector=[1.0, 0.0, 0.0],
                            text="good",
                            modality="text",
                        ),
                        VectorRecord(
                            record_id="p1:c2",
                            document_id="DOC1",
                            chunk_id="c2",
                            vector=[0.0, 1.0, 0.0],
                            text="good",
                            modality="text",
                        ),
                    ]
                )
                # Corrupt the second record's vector_json.
                with store.conn:
                    store.conn.execute(
                        "UPDATE vectors SET vector_json = ? WHERE record_id = ?",
                        ("not valid json", "p1:c2"),
                    )

                with self.assertLogs("zoterorag.index.local_vector", level=logging.WARNING) as cm:
                    hits = store.search([1.0, 0.0, 0.0], top_k=10)

                self.assertEqual(1, len(hits))
                self.assertEqual("c1", hits[0]["chunk_id"])
                self.assertTrue(
                    any("p1:c2" in message and "corrupt" in message for message in cm.output),
                    cm.output,
                )
            finally:
                store.close()

    def test_search_skips_dimension_mismatch(self) -> None:
        with workspace_tmpdir("local-vector-dim-") as tmpdir:
            store = LocalVectorStore(tmpdir / "vectors.sqlite", profile_name="p1", dimension=3)
            try:
                store.upsert(
                    [
                        VectorRecord(
                            record_id="p1:c1",
                            document_id="DOC1",
                            chunk_id="c1",
                            vector=[1.0, 0.0, 0.0],
                            text="good",
                            modality="text",
                        ),
                        VectorRecord(
                            record_id="p1:c2",
                            document_id="DOC1",
                            chunk_id="c2",
                            vector=[0.0, 1.0, 0.0],
                            text="bad dims",
                            modality="text",
                        ),
                    ]
                )
                with store.conn:
                    store.conn.execute(
                        "UPDATE vectors SET vector_json = ? WHERE record_id = ?",
                        (json.dumps([1.0, 2.0]), "p1:c2"),
                    )

                with self.assertLogs("zoterorag.index.local_vector", level=logging.WARNING) as cm:
                    hits = store.search([1.0, 0.0, 0.0], top_k=10)

                self.assertEqual(1, len(hits))
                self.assertEqual("c1", hits[0]["chunk_id"])
                self.assertTrue(
                    any("dimension mismatch" in message for message in cm.output),
                    cm.output,
                )
            finally:
                store.close()

    def test_close_logs_checkpoint_failure(self) -> None:
        with workspace_tmpdir("local-vector-close-") as tmpdir:
            store = LocalVectorStore(tmpdir / "vectors.sqlite", profile_name="p1", dimension=3)
            try:
                # Replace the thread-safe connection with a fake that rejects
                # checkpoints. This tests the retry/logging path without needing
                # to mutate a read-only sqlite3.Connection method.
                fake_conn = MagicMock()
                fake_conn.execute.side_effect = sqlite3.Error("forced checkpoint failure")
                store.conn = fake_conn

                with self.assertLogs(
                    "zoterorag.index.local_vector", level=logging.ERROR
                ) as cm:
                    store.close()

                self.assertTrue(
                    any("failed after retries" in message for message in cm.output),
                    cm.output,
                )
                fake_conn.close.assert_called_once()
            finally:
                store.close()
