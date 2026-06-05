from __future__ import annotations

import unittest

from tests.test_zotero_scanner import build_scanner_fixture_db, write_storage_file
from tests._support import workspace_tmpdir
from zoterorag.db.state import StateLedger
from zoterorag.review import explain_attachment_review
from zoterorag.zotero.scanner import scan_shadow_to_ledger


class ReviewExplanationTests(unittest.TestCase):
    def test_explain_translation_review_entry_without_writing_zotero(self) -> None:
        with workspace_tmpdir("review-explain-") as root:
            shadow_db = root / "shadow.sqlite"
            storage_dir = root / "storage"
            build_scanner_fixture_db(shadow_db)
            write_storage_file(storage_dir, "ATTDUAL", "paper_zh-CN_dual.pdf", b"%PDF-1.4 dual\n")
            write_storage_file(storage_dir, "ATTIMM", "immersive-copy.pdf", b"%PDF-1.4 immersive\n")
            ledger = StateLedger(root / "state.sqlite")
            try:
                scan_shadow_to_ledger(shadow_db, storage_dir, ledger)

                dual = explain_attachment_review(ledger, "ATTDUAL")
                immersive = explain_attachment_review(ledger, "ATTIMM")

                self.assertEqual("needs_review", dual["classification"])
                self.assertIn("translation_marker:_zh-cn_dual.pdf", dual["reasons"])
                self.assertIn("strong_translation_markers_go_to_review_not_auto_exclusion", dual["policy_notes"])
                self.assertEqual("needs_review", immersive["classification"])
                self.assertIn("weak_translation_signal", immersive["source_quality"])
                self.assertIn("weak_translation_markers_like_immersive_only_request_review", immersive["policy_notes"])
                self.assertNotIn("excluded_auto", {dual["classification"], immersive["classification"]})
            finally:
                ledger.close()

    def test_explain_manual_rule_and_missing_key(self) -> None:
        with workspace_tmpdir("review-explain-") as root:
            shadow_db = root / "shadow.sqlite"
            storage_dir = root / "storage"
            build_scanner_fixture_db(shadow_db)
            write_storage_file(storage_dir, "ATTDUAL", "paper_zh-CN_dual.pdf", b"%PDF-1.4 dual\n")
            ledger = StateLedger(root / "state.sqlite")
            try:
                ledger.upsert_review_rule("ATTDUAL", "include", "accepted translated copy")
                scan_shadow_to_ledger(shadow_db, storage_dir, ledger)

                dual = explain_attachment_review(ledger, "ATTDUAL")

                self.assertEqual("included_manual", dual["classification"])
                self.assertEqual("include", dual["manual_rule"]["decision"])
                self.assertIn("manual_rule_overrides_automatic_classification_on_next_scan", dual["policy_notes"])
                self.assertIn("attachment_is_eligible_for_ingest_planning", dual["policy_notes"])
                with self.assertRaises(KeyError):
                    explain_attachment_review(ledger, "NO_SUCH_KEY")
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
