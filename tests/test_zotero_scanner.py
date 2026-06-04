from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import unittest

from tests._support import workspace_tmpdir
from zoterorag.db.state import StateLedger
from zoterorag.zotero.scanner import scan_shadow_to_ledger


FIELD_ROWS = [
    (1, "title"),
    (2, "abstractNote"),
    (3, "date"),
    (4, "url"),
]


def build_scanner_fixture_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE items (
                itemID INTEGER PRIMARY KEY,
                key TEXT NOT NULL
            );
            CREATE TABLE itemAttachments (
                itemID INTEGER PRIMARY KEY,
                parentItemID INTEGER,
                contentType TEXT,
                path TEXT
            );
            CREATE TABLE fields (
                fieldID INTEGER PRIMARY KEY,
                fieldName TEXT NOT NULL
            );
            CREATE TABLE itemData (
                itemID INTEGER NOT NULL,
                fieldID INTEGER NOT NULL,
                valueID INTEGER NOT NULL
            );
            CREATE TABLE itemDataValues (
                valueID INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        conn.executemany("INSERT INTO fields(fieldID, fieldName) VALUES (?, ?)", FIELD_ROWS)
        conn.executemany(
            "INSERT INTO items(itemID, key) VALUES (?, ?)",
            [
                (1, "PARENTNORM"),
                (2, "ATTNORM"),
                (3, "PARENTDUAL"),
                (4, "ATTDUAL"),
                (5, "PARENTIMM"),
                (6, "ATTIMM"),
                (7, "ATTORPH"),
                (8, "PARENTMISS"),
                (9, "ATTMISS"),
                (10, "PARENTNOTE"),
                (11, "ATTNOTE"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO itemAttachments(itemID, parentItemID, contentType, path)
            VALUES (?, ?, ?, ?)
            """,
            [
                (2, 1, "application/pdf", "storage:paper.pdf"),
                (4, 3, "application/pdf", "storage:paper_zh-CN_dual.pdf"),
                (6, 5, "application/pdf", "storage:immersive-copy.pdf"),
                (7, None, "application/pdf", "storage:orphan-paper.pdf"),
                (9, 8, "application/pdf", "storage:missing-paper.pdf"),
                (11, 10, "text/plain", "storage:notes.txt"),
            ],
        )
        conn.executemany(
            "INSERT INTO itemDataValues(valueID, value) VALUES (?, ?)",
            [
                (101, "Primary paper"),
                (102, "Dual language paper"),
                (103, "Immersive reader export"),
                (104, "Orphan attachment title"),
                (105, "Missing file paper"),
                (106, "Attachment note"),
                (107, "2024"),
                (108, "2023"),
                (109, "2022"),
                (110, "https://example.invalid/paper"),
                (111, "https://example.invalid/paper-dual"),
                (112, "https://example.invalid/reader"),
                (113, "https://example.invalid/orphan"),
                (114, "https://example.invalid/missing"),
            ],
        )
        conn.executemany(
            "INSERT INTO itemData(itemID, fieldID, valueID) VALUES (?, ?, ?)",
            [
                (1, 1, 101),
                (1, 3, 107),
                (1, 4, 110),
                (3, 1, 102),
                (3, 3, 108),
                (3, 4, 111),
                (5, 1, 103),
                (5, 3, 109),
                (5, 4, 112),
                (7, 1, 104),
                (7, 4, 113),
                (8, 1, 105),
                (8, 4, 114),
                (10, 1, 106),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def write_storage_file(storage_dir: Path, attachment_key: str, filename: str, contents: bytes) -> Path:
    path = storage_dir / attachment_key / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


class ZoteroScannerTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path]:
        shadow_db = root / "shadow.sqlite"
        storage_dir = root / "storage"
        build_scanner_fixture_db(shadow_db)
        write_storage_file(storage_dir, "ATTNORM", "paper.pdf", b"%PDF-1.4 normal\n")
        write_storage_file(storage_dir, "ATTDUAL", "paper_zh-CN_dual.pdf", b"%PDF-1.4 dual\n")
        write_storage_file(storage_dir, "ATTIMM", "immersive-copy.pdf", b"%PDF-1.4 immersive\n")
        write_storage_file(storage_dir, "ATTORPH", "orphan-paper.pdf", b"%PDF-1.4 orphan\n")
        write_storage_file(storage_dir, "ATTNOTE", "notes.txt", b"plain text note\n")
        return shadow_db, storage_dir

    def attachment_map(self, ledger: StateLedger) -> dict[str, dict[str, object]]:
        return {
            row["attachment_key"]: row
            for row in ledger.list_attachments(limit=100)
        }

    def test_scan_persists_expected_classifications_and_summary(self) -> None:
        with workspace_tmpdir("zotero-scan-") as root:
            shadow_db, storage_dir = self.build_fixture(root)
            ledger = StateLedger(root / "state.sqlite")
            try:
                report = scan_shadow_to_ledger(shadow_db, storage_dir, ledger)

                self.assertEqual(6, report.scanned)
                self.assertEqual(
                    {
                        "included_auto": 1,
                        "needs_review": 2,
                        "orphan_metadata_only": 1,
                        "missing_file": 1,
                        "report_only": 1,
                    },
                    report.summary,
                )

                attachments = self.attachment_map(ledger)
                self.assertEqual("included_auto", attachments["ATTNORM"]["classification"])
                self.assertEqual("needs_review", attachments["ATTDUAL"]["classification"])
                self.assertEqual("needs_review", attachments["ATTIMM"]["classification"])
                self.assertEqual("weak_translation_signal", attachments["ATTIMM"]["source_quality"])
                self.assertNotEqual("excluded_auto", attachments["ATTIMM"]["classification"])
                self.assertEqual("orphan_metadata_only", attachments["ATTORPH"]["classification"])
                self.assertEqual("missing_file", attachments["ATTMISS"]["classification"])
                self.assertEqual(False, attachments["ATTMISS"]["file_exists"])
                self.assertEqual("report_only", attachments["ATTNOTE"]["classification"])

                checkpoint = ledger.get_checkpoint("zotero_shadow", "scan_zotero")
                self.assertIsNotNone(checkpoint)
                self.assertEqual("completed", checkpoint["status"])
                self.assertEqual(
                    {"scanned": 6, "summary": report.summary},
                    checkpoint["payload"],
                )

                summary = ledger.status_summary()
                self.assertEqual(1, summary["jobs"]["completed"])
                self.assertEqual(1, summary["checkpoints"])
                self.assertEqual(report.summary, summary["attachments"])

                scan_report_row = ledger.conn.execute(
                    """
                    SELECT summary_json
                    FROM scan_reports
                    ORDER BY report_id DESC
                    LIMIT 1
                    """
                ).fetchone()
                self.assertIsNotNone(scan_report_row)
                self.assertEqual(
                    {"scanned": 6, "summary": report.summary},
                    json.loads(scan_report_row["summary_json"]),
                )
            finally:
                ledger.close()

    def test_manual_include_overrides_review_classification(self) -> None:
        with workspace_tmpdir("zotero-scan-") as root:
            shadow_db, storage_dir = self.build_fixture(root)
            ledger = StateLedger(root / "state.sqlite")
            try:
                ledger.upsert_review_rule("ATTDUAL", "include", "reviewed and accepted")

                report = scan_shadow_to_ledger(shadow_db, storage_dir, ledger)

                attachments = self.attachment_map(ledger)
                self.assertEqual("included_manual", attachments["ATTDUAL"]["classification"])
                self.assertEqual("manual_override", attachments["ATTDUAL"]["source_quality"])
                self.assertIn("manual_include:reviewed and accepted", attachments["ATTDUAL"]["reasons"])
                self.assertEqual(1, report.summary["included_manual"])
                self.assertNotIn("ATTDUAL", [row["attachment_key"] for row in ledger.list_attachments(classification="needs_review")])
            finally:
                ledger.close()

    def test_full_scan_marks_absent_previous_attachment_deleted(self) -> None:
        with workspace_tmpdir("zotero-scan-") as root:
            shadow_db, storage_dir = self.build_fixture(root)
            ledger = StateLedger(root / "state.sqlite")
            try:
                scan_shadow_to_ledger(shadow_db, storage_dir, ledger)

                conn = sqlite3.connect(shadow_db)
                try:
                    conn.execute("DELETE FROM itemAttachments WHERE itemID = 2")
                    conn.commit()
                finally:
                    conn.close()

                report = scan_shadow_to_ledger(shadow_db, storage_dir, ledger)

                attachments = self.attachment_map(ledger)
                self.assertEqual("deleted", attachments["ATTNORM"]["classification"])
                self.assertEqual(5, report.scanned)
                self.assertEqual(1, report.summary["deleted"])
            finally:
                ledger.close()

    def test_scan_status_tracks_new_unchanged_changed_and_deleted(self) -> None:
        with workspace_tmpdir("zotero-scan-") as root:
            shadow_db, storage_dir = self.build_fixture(root)
            ledger = StateLedger(root / "state.sqlite")
            try:
                scan_shadow_to_ledger(shadow_db, storage_dir, ledger)
                first = self.attachment_map(ledger)
                self.assertEqual("new", first["ATTNORM"]["scan_status"])

                scan_shadow_to_ledger(shadow_db, storage_dir, ledger)
                second = self.attachment_map(ledger)
                self.assertEqual("unchanged", second["ATTNORM"]["scan_status"])

                write_storage_file(storage_dir, "ATTNORM", "paper.pdf", b"%PDF-1.4 changed\n")
                scan_shadow_to_ledger(shadow_db, storage_dir, ledger)
                third = self.attachment_map(ledger)
                self.assertEqual("changed", third["ATTNORM"]["scan_status"])

                conn = sqlite3.connect(shadow_db)
                try:
                    conn.execute("DELETE FROM itemAttachments WHERE itemID = 2")
                    conn.commit()
                finally:
                    conn.close()

                scan_shadow_to_ledger(shadow_db, storage_dir, ledger)
                fourth = self.attachment_map(ledger)
                self.assertEqual("deleted", fourth["ATTNORM"]["scan_status"])
                self.assertEqual("deleted", fourth["ATTNORM"]["classification"])
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
