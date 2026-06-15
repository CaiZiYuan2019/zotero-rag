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


if __name__ == "__main__":
    unittest.main()
