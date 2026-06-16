from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from zoterorag.index.lancedb_vector import (
    LanceDBVectorStore,
    _build_profile_name_predicate,
    _build_record_id_in_predicate,
    _is_table_missing,
    _validate_predicate_literal,
)


class LanceDBExceptionHelpersTests(unittest.TestCase):
    def test_is_table_missing_detects_file_not_found(self) -> None:
        self.assertTrue(_is_table_missing(FileNotFoundError("table.lance")))

    def test_is_table_missing_detects_notfound_exception_type(self) -> None:
        class TableNotFoundError(Exception):
            pass

        self.assertTrue(_is_table_missing(TableNotFoundError("whatever")))

    def test_is_table_missing_detects_message_indicators(self) -> None:
        for msg in (
            "table not found",
            "Table does not exist",
            "No such table 'vectors'",
        ):
            self.assertTrue(_is_table_missing(RuntimeError(msg)), msg)

    def test_is_table_missing_rejects_generic_errors(self) -> None:
        self.assertFalse(_is_table_missing(RuntimeError("disk full")))
        self.assertFalse(_is_table_missing(ValueError("bad input")))


class LanceDBPredicateTests(unittest.TestCase):
    def test_safe_record_ids_predicate(self) -> None:
        predicate = _build_record_id_in_predicate(["p1:c1", "p1.c2"])
        self.assertEqual("record_id IN ('p1:c1', 'p1.c2')", predicate)

    def test_unsafe_record_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build_record_id_in_predicate(["p1'; DROP"])

    def test_profile_name_predicate(self) -> None:
        self.assertEqual(
            "profile_name = 'my-profile'",
            _build_profile_name_predicate("my-profile"),
        )

    def test_unsafe_profile_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _validate_predicate_literal("p'; --", "profile_name")


class LanceDBVectorStoreBehaviorTests(unittest.TestCase):
    def _mock_lancedb_module(self) -> MagicMock:
        lancedb = MagicMock()
        lancedb.connect.return_value = self.mock_db
        return lancedb

    def setUp(self) -> None:
        self.mock_db = MagicMock()

    def test_active_version_returns_legacy_when_meta_table_missing(self) -> None:
        self.mock_db.open_table.side_effect = RuntimeError("table not found")
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            self.assertEqual("legacy", store.active_version())

    def test_active_version_raises_on_severe_error(self) -> None:
        self.mock_db.open_table.side_effect = RuntimeError("disk full")
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            with self.assertRaises(RuntimeError):
                store.active_version()

    def test_search_returns_empty_when_active_table_missing(self) -> None:
        self.mock_db.open_table.side_effect = RuntimeError("table not found")
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            self.assertEqual([], store.search([1.0, 0.0, 0.0]))

    def test_search_raises_on_severe_open_error(self) -> None:
        self.mock_db.open_table.side_effect = RuntimeError("permission denied")
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            with self.assertRaises(RuntimeError):
                store.search([1.0, 0.0, 0.0])

    def test_delete_records_propagates_errors(self) -> None:
        table = MagicMock()
        table.delete.side_effect = RuntimeError("delete failed")
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            with self.assertRaises(RuntimeError):
                store._delete_records(table, ["p1:c1"])

    def test_upsert_ensure_table_raises_on_non_missing_error(self) -> None:
        self.mock_db.open_table.side_effect = RuntimeError("permission denied")
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            with self.assertRaises(RuntimeError):
                store._ensure_table("vectors_v1", [])

    def test_counts_raises_on_severe_error(self) -> None:
        self.mock_db.open_table.side_effect = RuntimeError("permission denied")
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            with self.assertRaises(RuntimeError):
                store.counts()

    def test_copy_active_returns_zero_when_active_table_missing(self) -> None:
        meta_table = MagicMock()
        mock_arrow = MagicMock()
        mock_arrow.to_pylist.return_value = [
            {"profile_name": "p1", "active_version": "v1"}
        ]
        meta_table.to_arrow.return_value = mock_arrow

        def open_table(name: str):
            if name == "vector_meta":
                return meta_table
            raise RuntimeError("table not found")

        self.mock_db.open_table.side_effect = open_table
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            self.assertEqual(0, store.copy_active_records_to_version(index_version="v2"))
