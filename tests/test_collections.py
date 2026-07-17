from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from zoterorag.db.state import StateLedger
from zoterorag.search import get_collection_tree, list_collections, search_by_collection


def _attachment(key: str, title: str) -> dict[str, object]:
    return {
        "attachment_key": key,
        "parent_key": f"P{key}",
        "content_type": "application/pdf",
        "relative_path": f"storage:{key}.pdf",
        "title": title,
        "abstract": "abstract text",
        "date": "2020",
        "url": None,
        "classification": "included_auto",
        "source_quality": "primary_candidate",
        "reasons": [],
        "file_path": None,
        "file_exists": False,
        "file_size": None,
        "file_mtime": None,
        "metadata": {},
    }


def seed(ledger: StateLedger) -> None:
    ledger.upsert_attachments(
        [
            _attachment("ROOTDOC", "Root document about physics"),
            _attachment("CHILDDOC", "Child document about chemistry"),
            _attachment("OTHERDOC", "Unrelated document"),
        ]
    )
    ledger.upsert_collections(
        [
            {"collection_key": "ROOT", "collection_name": "Root", "parent_collection_key": None, "library_id": 1},
            {"collection_key": "CHILD", "collection_name": "Child", "parent_collection_key": "ROOT", "library_id": 1},
            {"collection_key": "SOLO", "collection_name": "Solo", "parent_collection_key": None, "library_id": 1},
        ]
    )
    ledger.replace_attachment_collections("ROOTDOC", ["ROOT"])
    ledger.replace_attachment_collections("CHILDDOC", ["CHILD"])
    ledger.replace_attachment_collections("OTHERDOC", ["SOLO"])


class CollectionsTests(unittest.TestCase):
    def test_list_collections_is_flat_with_parent_keys(self) -> None:
        with workspace_tmpdir("collections-") as root:
            ledger = StateLedger(root / "state.sqlite")
            try:
                seed(ledger)
                collections = list_collections(ledger)
                self.assertEqual(3, len(collections))
                by_key = {c["collection_key"]: c for c in collections}
                self.assertEqual("ROOT", by_key["CHILD"]["parent_collection_key"])
                self.assertIsNone(by_key["ROOT"]["parent_collection_key"])
            finally:
                ledger.close()

    def test_collection_tree_nests_children(self) -> None:
        with workspace_tmpdir("collections-") as root:
            ledger = StateLedger(root / "state.sqlite")
            try:
                seed(ledger)
                tree = get_collection_tree(ledger)
                roots = {node["collection_key"]: node for node in tree}
                self.assertIn("ROOT", roots)
                self.assertIn("SOLO", roots)
                self.assertEqual(1, len(roots["ROOT"]["children"]))
                self.assertEqual("CHILD", roots["ROOT"]["children"][0]["collection_key"])
            finally:
                ledger.close()

    def test_search_by_collection_excludes_children_by_default(self) -> None:
        with workspace_tmpdir("collections-") as root:
            ledger = StateLedger(root / "state.sqlite")
            try:
                seed(ledger)
                results = search_by_collection(ledger, "ROOT", query="document")
                keys = {r["chunk_id"] for r in results}
                self.assertEqual({"ROOTDOC"}, keys)
            finally:
                ledger.close()

    def test_search_by_collection_include_children(self) -> None:
        with workspace_tmpdir("collections-") as root:
            ledger = StateLedger(root / "state.sqlite")
            try:
                seed(ledger)
                results = search_by_collection(
                    ledger, "ROOT", query="document", include_children=True
                )
                keys = {r["chunk_id"] for r in results}
                self.assertEqual({"ROOTDOC", "CHILDDOC"}, keys)
            finally:
                ledger.close()

    def test_search_by_collection_empty_query_lists_partition(self) -> None:
        with workspace_tmpdir("collections-") as root:
            ledger = StateLedger(root / "state.sqlite")
            try:
                seed(ledger)
                results = search_by_collection(ledger, "CHILD", query=None)
                keys = {r["chunk_id"] for r in results}
                self.assertEqual({"CHILDDOC"}, keys)
            finally:
                ledger.close()

    def test_search_by_collection_unknown_collection_returns_empty(self) -> None:
        with workspace_tmpdir("collections-") as root:
            ledger = StateLedger(root / "state.sqlite")
            try:
                seed(ledger)
                self.assertEqual([], search_by_collection(ledger, "NOPE", query="document"))
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
