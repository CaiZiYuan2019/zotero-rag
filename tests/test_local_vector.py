from __future__ import annotations

import json
import logging
import multiprocessing
import sqlite3
import unittest
from unittest.mock import MagicMock

from tests._support import workspace_tmpdir
from zoterorag.index.local_vector import LocalVectorStore, VectorRecord


def _upsert_worker(path: str, profile_name: str, start: int, count: int) -> None:
    store = LocalVectorStore(path, profile_name=profile_name, dimension=3)
    try:
        records = [
            VectorRecord(
                record_id=f"r{i}",
                document_id=f"DOC{i}",
                chunk_id=f"c{i}",
                vector=[1.0, 0.0, 0.0],
                text=f"text {i}",
                modality="text",
            )
            for i in range(start, start + count)
        ]
        store.upsert(records)
    finally:
        store.close()


class LocalVectorStoreVersionCleanupTests(unittest.TestCase):
    def _record(self, doc: str, chunk: str, vector: list[float]) -> VectorRecord:
        return VectorRecord(
            record_id=f"p1:{chunk}",
            document_id=doc,
            chunk_id=chunk,
            vector=vector,
            text=f"text {chunk}",
            modality="text",
        )

    def test_publish_deletes_superseded_version(self) -> None:
        with workspace_tmpdir("local-vector-prune-") as tmpdir:
            store = LocalVectorStore(tmpdir / "vectors.sqlite", profile_name="p1", dimension=3)
            try:
                store.upsert([self._record("DOCA", "c1", [1.0, 0.0, 0.0])], index_version="v1")
                store.publish_version("v1")
                store.copy_active_records_to_version(index_version="v2", exclude_document_ids=[])
                store.upsert([self._record("DOCB", "c2", [0.0, 1.0, 0.0])], index_version="v2")
                store.publish_version("v2")

                self.assertEqual(["v2"], store.list_versions())
                hits = store.search([1.0, 0.0, 0.0], top_k=10)
                self.assertEqual({"c1", "c2"}, {hit["chunk_id"] for hit in hits})
                self.assertEqual(2, store.counts()["chunks"])
            finally:
                store.close()

    def test_publish_same_version_keeps_active_rows(self) -> None:
        with workspace_tmpdir("local-vector-republish-") as tmpdir:
            store = LocalVectorStore(tmpdir / "vectors.sqlite", profile_name="p1", dimension=3)
            try:
                store.upsert([self._record("DOCA", "c1", [1.0, 0.0, 0.0])], index_version="v1")
                store.publish_version("v1")
                # Crash-recovery re-embed republishes the same batch hash; the
                # guard must not delete the active version itself.
                store.publish_version("v1")

                self.assertEqual(["v1"], store.list_versions())
                self.assertEqual(1, store.counts()["chunks"])
            finally:
                store.close()

    def test_first_publish_clears_legacy_rows(self) -> None:
        with workspace_tmpdir("local-vector-legacy-") as tmpdir:
            store = LocalVectorStore(tmpdir / "vectors.sqlite", profile_name="p1", dimension=3)
            try:
                store.upsert([self._record("DOCA", "c1", [1.0, 0.0, 0.0])])
                store.copy_active_records_to_version(index_version="v1", exclude_document_ids=[])
                store.publish_version("v1")

                self.assertEqual(["v1"], store.list_versions())
                hits = store.search([1.0, 0.0, 0.0], top_k=10)
                self.assertEqual(["c1"], [hit["chunk_id"] for hit in hits])
            finally:
                store.close()

    def test_delete_version_refuses_active(self) -> None:
        with workspace_tmpdir("local-vector-del-active-") as tmpdir:
            store = LocalVectorStore(tmpdir / "vectors.sqlite", profile_name="p1", dimension=3)
            try:
                store.upsert([self._record("DOCA", "c1", [1.0, 0.0, 0.0])], index_version="v1")
                store.publish_version("v1")

                with self.assertRaises(ValueError):
                    store.delete_version("v1")
                self.assertEqual(1, store.counts()["chunks"])
            finally:
                store.close()

    def test_delete_version_removes_orphan(self) -> None:
        with workspace_tmpdir("local-vector-del-orphan-") as tmpdir:
            store = LocalVectorStore(tmpdir / "vectors.sqlite", profile_name="p1", dimension=3)
            try:
                store.upsert([self._record("DOCA", "c1", [1.0, 0.0, 0.0])], index_version="v1")
                store.publish_version("v1")
                # Simulate a crash-interrupted staged version left behind.
                store.upsert([self._record("DOCB", "c2", [0.0, 1.0, 0.0])], index_version="orphan")
                self.assertEqual(["orphan", "v1"], store.list_versions())

                self.assertEqual(1, store.delete_version("orphan"))
                self.assertEqual(["v1"], store.list_versions())
            finally:
                store.close()


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

    def test_concurrent_upsert_across_processes(self) -> None:
        with workspace_tmpdir("local-vector-cross-proc-") as tmpdir:
            path = tmpdir / "vectors.sqlite"
            p1 = multiprocessing.Process(
                target=_upsert_worker, args=(str(path), "p1", 0, 50)
            )
            p2 = multiprocessing.Process(
                target=_upsert_worker, args=(str(path), "p1", 50, 50)
            )
            p1.start()
            p2.start()
            p1.join(timeout=60)
            p2.join(timeout=60)
            self.assertEqual(0, p1.exitcode)
            self.assertEqual(0, p2.exitcode)
            store = LocalVectorStore(path, profile_name="p1", dimension=3)
            try:
                counts = store.counts()
                self.assertEqual(100, counts["chunks"])
                self.assertEqual(100, counts["documents"])
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
