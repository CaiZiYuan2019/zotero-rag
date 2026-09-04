from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

from tests._support import workspace_tmpdir
from zoterorag.index.lancedb_vector import (
    LanceDBVectorStore,
    _build_record_id_in_predicate,
    _is_table_missing,
    _validate_predicate_literal,
    clear_lancedb_cache,
)
from zoterorag.index.local_vector import VectorRecord


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
        # L8 rule: fresh store (no tables at all) still defaults to legacy.
        self.mock_db.table_names.return_value = []
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            self.assertEqual("legacy", store.active_version())

    def test_active_version_returns_legacy_when_only_legacy_table(self) -> None:
        self.mock_db.open_table.side_effect = RuntimeError("table not found")
        # L8 rule: a pre-versioning store (only vectors_legacy) stays readable.
        self.mock_db.table_names.return_value = ["vectors_legacy"]
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            self.assertEqual("legacy", store.active_version())

    def test_active_version_raises_when_meta_lost_but_versioned_tables_exist(
        self,
    ) -> None:
        self.mock_db.open_table.side_effect = RuntimeError("table not found")
        # L8 rule: meta lost while versioned tables survive is corruption and
        # must fail loudly (the old fallback silently searched an empty
        # legacy table).
        self.mock_db.table_names.return_value = ["vectors_9f8a", "vectors_legacy"]
        with patch(
            "zoterorag.index.lancedb_vector._import_lancedb",
            return_value=self._mock_lancedb_module(),
        ):
            store = LanceDBVectorStore("/tmp/vectors", profile_name="p1", dimension=3)
            with self.assertRaises(RuntimeError) as ctx:
                store.active_version()
            self.assertIn("vectors verify --profile p1", str(ctx.exception))

    def test_active_version_returns_legacy_when_profile_row_missing(self) -> None:
        meta_table = MagicMock()
        arrow = MagicMock()
        # Meta exists but this profile was never published.
        arrow.to_pylist.return_value = []
        meta_table.to_arrow.return_value = arrow
        self.mock_db.open_table.return_value = meta_table
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

    def _assert_merge_insert_publish(self, version: str) -> None:
        """publish_version must flip the meta row via one atomic merge_insert
        (M5), never via the old delete+add pair with its zero-row window."""
        self.meta_table.merge_insert.assert_called_once_with("profile_name")
        builder = self.meta_table.merge_insert.return_value
        builder.when_matched_update_all.assert_called_once_with()
        builder.when_matched_update_all.return_value.when_not_matched_insert_all.assert_called_once_with()
        execute = (
            builder.when_matched_update_all.return_value
            .when_not_matched_insert_all.return_value.execute
        )
        execute.assert_called_once_with(
            [{"profile_name": "p1", "active_version": version}]
        )
        self.meta_table.delete.assert_not_called()
        self.meta_table.add.assert_not_called()

    def test_publish_drops_superseded_table(self) -> None:
        with workspace_tmpdir("lancedb-publish-drop-") as tmpdir:
            store = self._store(tmpdir)
            store._table_cache["vectors_v1"] = object()
            store.publish_version("v2")

            self._assert_merge_insert_publish("v2")
            self.mock_db.drop_table.assert_called_once_with("vectors_v1")
            self.assertNotIn("vectors_v1", store._table_cache)

    def test_publish_same_version_does_not_drop(self) -> None:
        with workspace_tmpdir("lancedb-publish-same-") as tmpdir:
            store = self._store(tmpdir)
            # Meta reports "v1" active; republishing "v1" must not drop it.
            store.publish_version("v1")

            self._assert_merge_insert_publish("v1")
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


class _FakeTable:
    """Minimal stateful LanceDB table double: add accumulates rows; delete is
    a tolerated no-op (upsert's delete-then-add hits it on empty tables).

    When ``owner`` is set, every mutating call asserts the owner's write
    locks are held (M6).
    """

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = list(rows or [])
        self.delete_calls: list[str] = []
        self.owner: LanceDBVectorStore | None = None

    def _check_lock(self) -> None:
        if self.owner is not None:
            assert self.owner._file_lock.is_locked, "write op without file lock"
            # RLock has no public is-locked probe; _is_owned() is CPython's
            # equivalent and these tests run on it.
            assert self.owner._write_lock._is_owned(), "write op without thread lock"

    def add(self, rows) -> None:
        self._check_lock()
        self.rows.extend(rows)

    def delete(self, predicate: str) -> None:
        self._check_lock()
        self.delete_calls.append(predicate)

    def merge_insert(self, key: str):
        self._check_lock()
        table = self

        class _Builder:
            def when_matched_update_all(self):
                return self

            def when_not_matched_insert_all(self):
                return self

            def execute(self, rows) -> None:
                for row in rows:
                    for i, existing in enumerate(table.rows):
                        if existing.get(key) == row.get(key):
                            table.rows[i] = dict(row)
                            break
                    else:
                        table.rows.append(dict(row))

        return _Builder()

    def to_arrow(self) -> MagicMock:
        arrow = MagicMock()
        arrow.to_pylist.return_value = list(self.rows)
        return arrow


class LanceDBStagedDedupeTests(unittest.TestCase):
    """H1: a staged table left over from a crashed run must be dropped and
    rebuilt, never appended to, when the same version is staged again."""

    def setUp(self) -> None:
        clear_lancedb_cache()
        self.tables: dict[str, _FakeTable] = {
            "vector_meta": _FakeTable(
                [{"profile_name": "p1", "active_version": "v1"}]
            ),
            "vectors_v1": _FakeTable(
                [
                    {
                        "record_id": "v1:r1",
                        "document_id": "docA",
                        "chunk_id": "c1",
                        "vector_json": "[1.0, 0.0, 0.0]",
                        "text": "alpha",
                        "modality": "text",
                        "metadata_json": "{}",
                    },
                    {
                        "record_id": "v1:r2",
                        "document_id": "docB",
                        "chunk_id": "c2",
                        "vector_json": "[0.0, 1.0, 0.0]",
                        "text": "beta",
                        "modality": "text",
                        "metadata_json": "{}",
                    },
                ]
            ),
        }
        self.mock_db = MagicMock()
        # Auto-vivifying open_table avoids the pyarrow-dependent create branch.
        self.mock_db.open_table.side_effect = lambda name: self.tables.setdefault(
            name, _FakeTable()
        )
        self.mock_db.table_names.side_effect = lambda **_: sorted(self.tables)
        self.mock_db.drop_table.side_effect = lambda name: self.tables.pop(name, None)

    def _store(self, tmpdir) -> LanceDBVectorStore:
        lancedb = MagicMock()
        lancedb.connect.return_value = self.mock_db
        with patch("zoterorag.index.lancedb_vector._import_lancedb", return_value=lancedb):
            return LanceDBVectorStore(tmpdir / "vectors", profile_name="p1", dimension=3)

    def test_leftover_staged_table_is_dropped_and_rebuilt(self) -> None:
        leftover = _FakeTable([{"record_id": "v2:stale", "document_id": "docOld"}])
        self.tables["vectors_v2"] = leftover
        with workspace_tmpdir("lancedb-h1-rebuild-") as tmpdir:
            store = self._store(tmpdir)
            # A stale cache handle must not let the old table survive either.
            store._table_cache["vectors_v2"] = leftover
            copied = store.copy_active_records_to_version(index_version="v2")

            self.assertEqual(2, copied)
            self.mock_db.drop_table.assert_called_once_with("vectors_v2")
            staged = self.tables["vectors_v2"]
            self.assertIsNot(staged, leftover)
            self.assertEqual(2, len(staged.rows))
            self.assertEqual(
                {"v2:r1", "v2:r2"}, {row["record_id"] for row in staged.rows}
            )
            self.assertIs(store._table_cache["vectors_v2"], staged)

    def test_restaging_same_version_does_not_double_rows(self) -> None:
        with workspace_tmpdir("lancedb-h1-nodup-") as tmpdir:
            store = self._store(tmpdir)
            # Simulate two crash-retries: stage, crash before publish, stage again.
            first = store.copy_active_records_to_version(index_version="v2")
            second = store.copy_active_records_to_version(index_version="v2")
            self.assertEqual(2, first)
            self.assertEqual(2, second)
            self.assertEqual(2, len(self.tables["vectors_v2"].rows))

    def test_empty_copy_still_drops_leftover_staged_table(self) -> None:
        self.tables["vectors_v2"] = _FakeTable([{"record_id": "v2:stale"}])
        with workspace_tmpdir("lancedb-h1-empty-") as tmpdir:
            store = self._store(tmpdir)
            copied = store.copy_active_records_to_version(
                index_version="v2", exclude_document_ids={"docA", "docB"}
            )
            self.assertEqual(0, copied)
            self.mock_db.drop_table.assert_called_once_with("vectors_v2")
            self.assertNotIn("vectors_v2", self.tables)

    def test_staging_active_version_is_noop(self) -> None:
        # Staging the already-active version must not append its rows back
        # onto itself (which would double every row).
        with workspace_tmpdir("lancedb-h1-active-") as tmpdir:
            store = self._store(tmpdir)
            copied = store.copy_active_records_to_version(index_version="v1")
            self.assertEqual(0, copied)
            self.mock_db.drop_table.assert_not_called()
            self.assertEqual(2, len(self.tables["vectors_v1"].rows))

    def test_leftover_staged_drop_failure_raises(self) -> None:
        self.tables["vectors_v2"] = _FakeTable([{"record_id": "v2:stale"}])
        self.mock_db.drop_table.side_effect = RuntimeError("disk full")
        with workspace_tmpdir("lancedb-h1-dropfail-") as tmpdir:
            store = self._store(tmpdir)
            with self.assertRaises(RuntimeError):
                store.copy_active_records_to_version(index_version="v2")

    def test_leftover_staged_drop_missing_is_tolerated(self) -> None:
        # The staged table vanished between listing and dropping (e.g. a
        # concurrent prune): the guard must proceed and rebuild normally.
        self.tables["vectors_v2"] = _FakeTable([{"record_id": "v2:stale"}])

        def drop_then_vanish(name: str) -> None:
            self.tables.pop(name, None)
            raise FileNotFoundError(name)

        self.mock_db.drop_table.side_effect = drop_then_vanish
        with workspace_tmpdir("lancedb-h1-dropmissing-") as tmpdir:
            store = self._store(tmpdir)
            copied = store.copy_active_records_to_version(index_version="v2")
            self.assertEqual(2, copied)
            self.assertEqual(2, len(self.tables["vectors_v2"].rows))

    def test_leftover_staged_drop_uses_sanitized_table_name(self) -> None:
        # Digit-leading batch hashes map to an irreversible v_-prefixed table
        # name; the guard must drop in table-name space.
        self.tables["vectors_v_1abc"] = _FakeTable([{"record_id": "1abc:stale"}])
        with workspace_tmpdir("lancedb-h1-digit-") as tmpdir:
            store = self._store(tmpdir)
            copied = store.copy_active_records_to_version(index_version="1abc")
            self.assertEqual(2, copied)
            self.mock_db.drop_table.assert_called_once_with("vectors_v_1abc")
            self.assertEqual(2, len(self.tables["vectors_v_1abc"].rows))

    def test_copy_creates_meta_table_up_front_and_self_heals(self) -> None:
        # L8: a pre-fix leftover ("staged table, no vector_meta") must not be
        # misread as corruption — the write path creates meta first, after
        # which the leftover is dropped and rebuilt as usual.
        del self.tables["vector_meta"]
        self.tables["vectors_v2"] = _FakeTable([{"record_id": "v2:stale"}])
        # No published version either: only the staged leftover survives.
        del self.tables["vectors_v1"]
        with workspace_tmpdir("lancedb-l8-selfheal-") as tmpdir:
            store = self._store(tmpdir)
            copied = store.copy_active_records_to_version(index_version="v2")
            self.assertEqual(0, copied)
            self.assertIn("vector_meta", self.tables)
            self.mock_db.drop_table.assert_called_once_with("vectors_v2")
            self.assertNotIn("vectors_v2", self.tables)

    def test_upsert_creates_meta_table_up_front(self) -> None:
        del self.tables["vector_meta"]
        record = VectorRecord(
            record_id="r1",
            document_id="docA",
            chunk_id="c1",
            vector=[1.0, 0.0, 0.0],
            text="alpha",
            modality="text",
            metadata={},
        )
        with workspace_tmpdir("lancedb-l8-upsert-meta-") as tmpdir:
            store = self._store(tmpdir)
            self.assertEqual(1, store.upsert([record], index_version="v2"))
            self.assertIn("vector_meta", self.tables)


class LanceDBWriteLockTests(LanceDBStagedDedupeTests):
    """M6: every write op runs under the per-profile file lock, and
    stage_and_publish holds it across the whole copy->upsert->publish
    sequence (verified via _FakeTable.owner lock assertions)."""

    def _locked_store(self, tmpdir) -> LanceDBVectorStore:
        store = self._store(tmpdir)
        for table in self.tables.values():
            table.owner = store
        return store

    def test_upsert_holds_write_locks(self) -> None:
        with workspace_tmpdir("lancedb-m6-upsert-") as tmpdir:
            store = self._locked_store(tmpdir)
            record = VectorRecord(
                record_id="r3",
                document_id="docC",
                chunk_id="c3",
                vector=[0.0, 0.0, 1.0],
                text="gamma",
            )
            self.assertEqual(1, store.upsert([record], index_version="v2"))
            self.assertFalse(store._file_lock.is_locked)

    def test_stage_and_publish_locked_sequence(self) -> None:
        with workspace_tmpdir("lancedb-m6-sequence-") as tmpdir:
            store = self._locked_store(tmpdir)
            record = VectorRecord(
                record_id="r3",
                document_id="docC",
                chunk_id="c3",
                vector=[0.0, 0.0, 1.0],
                text="gamma",
            )
            copied, indexed, counts = store.stage_and_publish(
                [record], index_version="v2", exclude_document_ids=["docC"]
            )
            # Carry-forward copied the two active rows; publish flipped meta
            # and dropped the superseded table — all under one lock.
            self.assertEqual(2, copied)
            self.assertEqual(1, indexed)
            self.assertEqual({"documents": 3, "chunks": 3}, counts)
            self.assertEqual(
                [{"profile_name": "p1", "active_version": "v2"}],
                self.tables["vector_meta"].rows,
            )
            self.assertNotIn("vectors_v1", self.tables)
            self.assertFalse(store._file_lock.is_locked)

    def test_stage_and_publish_accepts_iterators(self) -> None:
        with workspace_tmpdir("lancedb-m6-iter-") as tmpdir:
            store = self._locked_store(tmpdir)
            records = (
                VectorRecord(
                    record_id="r3",
                    document_id="docC",
                    chunk_id="c3",
                    vector=[0.0, 0.0, 1.0],
                    text="gamma",
                )
                for _ in range(1)
            )
            copied, indexed, counts = store.stage_and_publish(
                records, index_version="v2", exclude_document_ids=["docC"]
            )
            self.assertEqual(1, indexed)
            self.assertEqual(3, counts["chunks"])

    def test_stage_and_publish_releases_locks_on_error(self) -> None:
        with workspace_tmpdir("lancedb-m6-error-") as tmpdir:
            store = self._locked_store(tmpdir)
            bad = VectorRecord(
                record_id="rbad",
                document_id="docC",
                chunk_id="cbad",
                vector=[1.0],  # wrong dimension -> upsert raises mid-sequence
                text="bad",
            )
            with self.assertRaises(ValueError):
                store.stage_and_publish([bad], index_version="v2")
            self.assertFalse(store._file_lock.is_locked)
            # A clean retry after the failure must proceed normally.
            copied, indexed, _ = store.stage_and_publish([], index_version="v2")
            self.assertEqual((2, 0), (copied, indexed))
