from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

from tests._support import workspace_tmpdir
from zoterorag.index.lancedb_vector import (
    LanceDBVectorStore,
    _build_profile_name_predicate,
    _build_record_id_in_predicate,
    _is_table_missing,
    _validate_predicate_literal,
    clear_lancedb_cache,
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
        clear_lancedb_cache()

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
            self.assertEqual(
                0, store.copy_active_records_to_version(index_version="v2")
            )


class LanceDBVersionCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_db = MagicMock()
        clear_lancedb_cache()
        self.meta_table = MagicMock()
        arrow = MagicMock()
        arrow.to_pylist.return_value = [{"profile_name": "p1", "active_version": "v1"}]
        self.meta_table.to_arrow.return_value = arrow
        self.mock_db.open_table.return_value = self.meta_table

    def _store(self, tmpdir) -> LanceDBVectorStore:
        lancedb = MagicMock()
        lancedb.connect.return_value = self.mock_db
        with patch("zoterorag.index.lancedb_vector._import_lancedb", return_value=lancedb):
            return LanceDBVectorStore(tmpdir / "vectors", profile_name="p1", dimension=3)

    def test_publish_drops_superseded_table(self) -> None:
        with workspace_tmpdir("lancedb-publish-drop-") as tmpdir:
            store = self._store(tmpdir)
            store._table_cache["vectors_v1"] = object()
            store.publish_version("v2")

            self.mock_db.drop_table.assert_called_once_with("vectors_v1")
            self.assertNotIn("vectors_v1", store._table_cache)
            self.meta_table.delete.assert_called_once()
            self.meta_table.add.assert_called_once()

    def test_publish_same_version_does_not_drop(self) -> None:
        with workspace_tmpdir("lancedb-publish-same-") as tmpdir:
            store = self._store(tmpdir)
            # Meta reports "v1" active; republishing "v1" must not drop it.
            store.publish_version("v1")

            self.mock_db.drop_table.assert_not_called()

    def test_publish_drop_failure_only_warns(self) -> None:
        self.mock_db.drop_table.side_effect = RuntimeError("disk full")
        with workspace_tmpdir("lancedb-publish-dropfail-") as tmpdir:
            store = self._store(tmpdir)
            with self.assertLogs("zoterorag.index.lancedb_vector", level=logging.WARNING) as cm:
                store.publish_version("v2")

            self.assertTrue(any("vectors_v1" in message for message in cm.output), cm.output)

    def test_publish_missing_superseded_table_is_noop(self) -> None:
        self.mock_db.drop_table.side_effect = FileNotFoundError("vectors_v1")
        with workspace_tmpdir("lancedb-publish-dropmissing-") as tmpdir:
            store = self._store(tmpdir)
            store.publish_version("v2")  # must not raise

    def test_delete_version_guards_active_and_drops_orphan(self) -> None:
        self.mock_db.table_names.return_value = ["vectors_v1", "vectors_orphan", "vector_meta"]
        with workspace_tmpdir("lancedb-delete-version-") as tmpdir:
            store = self._store(tmpdir)
            with self.assertRaises(ValueError):
                store.delete_version("v1")
            self.assertEqual(1, store.delete_version("orphan"))
            self.mock_db.drop_table.assert_called_once_with("vectors_orphan")
            self.assertEqual(0, store.delete_version("ghost"))

    def test_list_versions_filters_meta_and_unrelated_tables(self) -> None:
        self.mock_db.table_names.return_value = ["vectors_b", "vector_meta", "vectors_a", "snapshots"]
        with workspace_tmpdir("lancedb-list-versions-") as tmpdir:
            store = self._store(tmpdir)
            self.assertEqual(["a", "b"], store.list_versions())

    def _set_active_version(self, version: str) -> None:
        arrow = MagicMock()
        arrow.to_pylist.return_value = [{"profile_name": "p1", "active_version": version}]
        self.meta_table.to_arrow.return_value = arrow

    def test_version_is_active_with_digit_leading_version(self) -> None:
        # sha256 batch hashes start with a digit ~62% of the time; the
        # sanitized table name (v_1abc) must still match the active version.
        self._set_active_version("1abc")
        with workspace_tmpdir("lancedb-active-digit-") as tmpdir:
            store = self._store(tmpdir)
            self.assertTrue(store.version_is_active("1abc"))
            self.assertTrue(store.version_is_active("v_1abc"))
            self.assertFalse(store.version_is_active("v_9xyz"))

    def test_delete_version_guard_with_sanitized_active_name(self) -> None:
        self._set_active_version("1abc")
        self.mock_db.table_names.return_value = ["vectors_v_1abc", "vectors_orphan"]
        with workspace_tmpdir("lancedb-guard-digit-") as tmpdir:
            store = self._store(tmpdir)
            # list_versions() yields the table-derived name; the guard must
            # still recognize it as the active version and refuse.
            with self.assertRaises(ValueError):
                store.delete_version("v_1abc")
            self.mock_db.drop_table.assert_not_called()

    def test_prune_skip_logic_with_digit_leading_active(self) -> None:
        self._set_active_version("1abc")
        self.mock_db.table_names.return_value = ["vectors_v_1abc", "vectors_orphan"]
        with workspace_tmpdir("lancedb-prune-digit-") as tmpdir:
            store = self._store(tmpdir)
            for version in store.list_versions():
                if store.version_is_active(version):
                    continue
                store.delete_version(version)
            # Only the orphan table is dropped; the active table survives.
            self.mock_db.drop_table.assert_called_once_with("vectors_orphan")
