from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from zoterorag.db.state import StateLedger
from zoterorag.search import metadata_search


class MetadataSearchTests(unittest.TestCase):
    def test_metadata_search_returns_text_only_results(self) -> None:
        with workspace_tmpdir("metadata-search-") as root:
            ledger = StateLedger(root / "state.sqlite")
            try:
                ledger.upsert_attachments(
                    [
                        {
                            "attachment_key": "HFZNNCNH",
                            "parent_key": "6H4APXQD",
                            "content_type": "application/pdf",
                            "relative_path": "storage:Gerum - 2019 - pylustrator Code generation for reproducible figures for publication.pdf",
                            "title": "pylustrator: Code generation for reproducible figures for publication",
                            "abstract": "Code generation for reproducible publication figures.",
                            "date": "2019-10-01",
                            "url": "https://arxiv.org/abs/1910.00279v2",
                            "classification": "included_auto",
                            "source_quality": "primary_candidate",
                            "reasons": ["pdf_with_parent"],
                            "file_path": str(root / "storage" / "HFZNNCNH" / "paper.pdf"),
                            "file_exists": True,
                            "file_size": 123,
                            "file_mtime": 1.0,
                            "metadata": {},
                        },
                        {
                            "attachment_key": "OTHER",
                            "parent_key": "PARENT",
                            "content_type": "application/pdf",
                            "relative_path": "storage:other.pdf",
                            "title": "Unrelated paper",
                            "abstract": "Nothing relevant.",
                            "date": "2020",
                            "url": None,
                            "classification": "included_auto",
                            "source_quality": "primary_candidate",
                            "reasons": ["pdf_with_parent"],
                            "file_path": None,
                            "file_exists": False,
                            "file_size": None,
                            "file_mtime": None,
                            "metadata": {},
                        },
                    ]
                )

                results = metadata_search(ledger, "pylustrator reproducible", consumer="llm_text")

                self.assertEqual(1, len(results))
                self.assertEqual("HFZNNCNH", results[0]["chunk_id"])
                self.assertIn("pylustrator", results[0]["text"])
                self.assertNotIn("images", results[0])
                self.assertNotIn("base64", repr(results[0]))
                self.assertIsNone(results[0]["rerank_score"])
                self.assertEqual("included_auto", results[0]["metadata"]["classification"])
                with self.assertRaises(RuntimeError):
                    metadata_search(ledger, "pylustrator", rerank=True)
            finally:
                ledger.close()

    def test_metadata_search_can_filter_by_classification(self) -> None:
        with workspace_tmpdir("metadata-search-") as root:
            ledger = StateLedger(root / "state.sqlite")
            try:
                ledger.upsert_attachments(
                    [
                        {
                            "attachment_key": "REVIEW",
                            "parent_key": "PARENT",
                            "content_type": "application/pdf",
                            "relative_path": "storage:paper_zh-CN_dual.pdf",
                            "title": "Dual paper",
                            "abstract": "dual review",
                            "date": "2024",
                            "url": None,
                            "classification": "needs_review",
                            "source_quality": "suspected_translation_pdf",
                            "reasons": ["translation_marker:_zh-cn_dual.pdf"],
                            "file_path": None,
                            "file_exists": False,
                            "file_size": None,
                            "file_mtime": None,
                            "metadata": {},
                        }
                    ]
                )

                self.assertEqual(1, len(metadata_search(ledger, "dual", classification="needs_review")))
                self.assertEqual(0, len(metadata_search(ledger, "dual", classification="included_auto")))
            finally:
                ledger.close()


def _biblio_attachment(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "attachment_key": "BIB1",
        "parent_key": "PBIB1",
        "content_type": "application/pdf",
        "relative_path": "storage:bib1.pdf",
        "title": "Neural methods",
        "abstract": "Deep learning study",
        "date": "2021-05-01",
        "url": None,
        "classification": "included_auto",
        "source_quality": "primary_candidate",
        "reasons": [],
        "file_path": None,
        "file_exists": False,
        "file_size": None,
        "file_mtime": None,
        "authors": ["Ada Lovelace"],
        "collections": [{"collection_key": "C1", "collection_name": "AI", "parent_collection_key": None}],
        "tags": ["neural"],
        "item_type": "journalArticle",
        "publication_title": "Journal of AI",
        "doi": "10.1000/neural",
        "date_added": "2021-06-01",
        "date_modified": "2021-06-02",
        "metadata": {},
    }
    record.update(overrides)
    return record


class MetadataSearchFilterTests(unittest.TestCase):
    def _ledger(self, root):
        ledger = StateLedger(root / "state.sqlite")
        ledger.upsert_attachments(
            [
                _biblio_attachment(),
                _biblio_attachment(
                    attachment_key="BIB2",
                    parent_key="PBIB2",
                    relative_path="storage:bib2.pdf",
                    title="Symbolic reasoning",
                    abstract="Logic based methods",
                    date="2010",
                    authors=["Grace Hopper"],
                    tags=["symbolic"],
                    publication_title="Journal of Logic",
                    doi="10.1000/logic",
                    date_added="2010-01-01",
                    date_modified="2010-01-02",
                ),
            ]
        )
        return ledger

    def test_empty_query_without_filters_returns_empty(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                self.assertEqual([], metadata_search(ledger, ""))
                self.assertEqual([], metadata_search(ledger, "   "))
            finally:
                ledger.close()

    def test_empty_query_with_filter_uses_pure_filter_path(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                results = metadata_search(ledger, "", authors=["hopper"])
                self.assertEqual(1, len(results))
                self.assertEqual("BIB2", results[0]["chunk_id"])
            finally:
                ledger.close()

    def test_author_filter_is_substring_case_insensitive(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                results = metadata_search(ledger, "methods", authors=["lovelace"])
                self.assertEqual({"BIB1"}, {r["chunk_id"] for r in results})
            finally:
                ledger.close()

    def test_tag_filter_exact_match(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                self.assertEqual(
                    {"BIB1"}, {r["chunk_id"] for r in metadata_search(ledger, "", tags=["neural"])}
                )
                self.assertEqual([], metadata_search(ledger, "", tags=["neur"]))
            finally:
                ledger.close()

    def test_date_range_filter(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                results = metadata_search(ledger, "", date_from="2015", date_to="2025")
                self.assertEqual({"BIB1"}, {r["chunk_id"] for r in results})
            finally:
                ledger.close()

    def test_doi_and_publication_filters(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                self.assertEqual(
                    {"BIB2"}, {r["chunk_id"] for r in metadata_search(ledger, "", doi="logic")}
                )
                self.assertEqual(
                    {"BIB1"},
                    {r["chunk_id"] for r in metadata_search(ledger, "", publication_title="AI")},
                )
            finally:
                ledger.close()

    def test_attachment_keys_prefilter(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                results = metadata_search(ledger, "", attachment_keys=["BIB2"])
                self.assertEqual({"BIB2"}, {r["chunk_id"] for r in results})
            finally:
                ledger.close()

    def test_score_weights_hit_author_and_doi(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                by_author = metadata_search(ledger, "lovelace")
                self.assertEqual("BIB1", by_author[0]["chunk_id"])
                by_doi = metadata_search(ledger, "logic")
                self.assertEqual("BIB2", by_doi[0]["chunk_id"])
            finally:
                ledger.close()

    def test_bm25_sort_exposes_larger_is_better_score(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                results = metadata_search(ledger, "methods", sort_by="bm25")
                self.assertTrue(results)
                scores = [r["score"] for r in results]
                self.assertEqual(scores, sorted(scores, reverse=True))
            finally:
                ledger.close()

    def test_results_metadata_is_enriched_and_metadata_json_untouched(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                results = metadata_search(ledger, "neural methods")
                meta = results[0]["metadata"]
                self.assertEqual(["Ada Lovelace"], meta["authors"])
                self.assertEqual("Journal of AI", meta["publication_title"])
                self.assertEqual("10.1000/neural", meta["doi"])
                self.assertEqual(["neural"], meta["tags"])
                # Stored metadata_json stays empty.
                self.assertEqual({}, ledger.get_attachment("BIB1")["metadata"])
            finally:
                ledger.close()

    def test_invalid_sort_by_raises_value_error(self) -> None:
        with workspace_tmpdir("metadata-filter-") as root:
            ledger = self._ledger(root)
            try:
                with self.assertRaises(ValueError):
                    metadata_search(ledger, "methods", sort_by="bogus")
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
