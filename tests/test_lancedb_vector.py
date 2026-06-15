from __future__ import annotations

import unittest

from zoterorag.index.lancedb_vector import (
    _build_profile_name_predicate,
    _build_record_id_in_predicate,
    _record_to_row,
    _validate_predicate_literal,
)
from zoterorag.index.local_vector import VectorRecord


class LanceDBPredicateTests(unittest.TestCase):
    def test_validate_predicate_literal_accepts_safe_ids(self) -> None:
        self.assertEqual("abc-123_:.", _validate_predicate_literal("abc-123_:.", "record_id"))

    def test_validate_predicate_literal_rejects_quotes(self) -> None:
        with self.assertRaises(ValueError):
            _validate_predicate_literal("id' OR '1'='1", "record_id")

    def test_validate_predicate_literal_rejects_backslash(self) -> None:
        with self.assertRaises(ValueError):
            _validate_predicate_literal("id\\x", "record_id")

    def test_build_record_id_in_predicate(self) -> None:
        predicate = _build_record_id_in_predicate(["prof:chunk-1", "prof:chunk-2"])
        self.assertEqual("record_id IN ('prof:chunk-1', 'prof:chunk-2')", predicate)

    def test_build_record_id_in_predicate_rejects_unsafe_id(self) -> None:
        with self.assertRaises(ValueError):
            _build_record_id_in_predicate(["prof:chunk-1", "bad'id"])

    def test_build_profile_name_predicate(self) -> None:
        predicate = _build_profile_name_predicate("qwen3vl_cloud_2560_text")
        self.assertEqual("profile_name = 'qwen3vl_cloud_2560_text'", predicate)

    def test_build_profile_name_predicate_rejects_unsafe_name(self) -> None:
        with self.assertRaises(ValueError):
            _build_profile_name_predicate("profile' OR '1'='1")

    def test_record_to_row_includes_profile_name(self) -> None:
        record = VectorRecord(
            record_id="chunk-1",
            document_id="doc-1",
            chunk_id="chunk-1",
            vector=[0.1, 0.2],
            text="hello",
            modality="text",
            metadata={"key": "value"},
        )
        row = _record_to_row(record, "batch-1", "my-profile")
        self.assertEqual("my-profile", row["profile_name"])
        self.assertEqual("batch-1:chunk-1", row["record_id"])


class LanceDBVectorStoreOptionalTests(unittest.TestCase):
    """Integration tests that run only when lancedb is installed."""

    def setUp(self) -> None:
        try:
            import lancedb  # noqa: F401
        except ImportError:
            self.skipTest("lancedb is not installed")

    def test_store_writes_profile_name(self) -> None:
        from pathlib import Path
        from tests._support import workspace_tmpdir
        from zoterorag.index.lancedb_vector import LanceDBVectorStore

        with workspace_tmpdir("lancedb-store-") as tmpdir:
            store = LanceDBVectorStore(tmpdir, "test-profile", 2)
            record = VectorRecord(
                record_id="chunk-1",
                document_id="doc-1",
                chunk_id="chunk-1",
                vector=[0.1, 0.2],
                text="hello",
                modality="text",
                metadata={},
            )
            store.upsert([record], index_version="batch-1")
            store.publish_version("batch-1")
            counts = store.counts()
            self.assertEqual(1, counts["chunks"])


if __name__ == "__main__":
    unittest.main()
