from __future__ import annotations

from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from tests._support import workspace_tmpdir
import zoterorag.zotero.shadow as shadow_module
from zoterorag.zotero.shadow import ZoteroShadow, create_shadow_copy


def build_minimal_zotero_db(path: Path) -> None:
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
        conn.executemany(
            "INSERT INTO fields(fieldID, fieldName) VALUES (?, ?)",
            [
                (1, "title"),
                (2, "abstractNote"),
                (3, "date"),
                (4, "url"),
            ],
        )
        conn.executemany(
            "INSERT INTO items(itemID, key) VALUES (?, ?)",
            [
                (1, "PARENT1"),
                (2, "ATTACH1"),
            ],
        )
        conn.execute(
            """
            INSERT INTO itemAttachments(itemID, parentItemID, contentType, path)
            VALUES (?, ?, ?, ?)
            """,
            (2, 1, "application/pdf", "storage:paper.pdf"),
        )
        conn.executemany(
            "INSERT INTO itemDataValues(valueID, value) VALUES (?, ?)",
            [
                (11, "Example Paper"),
                (12, "Useful abstract"),
                (13, "2024"),
                (14, "https://example.invalid/paper"),
            ],
        )
        conn.executemany(
            "INSERT INTO itemData(itemID, fieldID, valueID) VALUES (?, ?, ?)",
            [
                (1, 1, 11),
                (1, 2, 12),
                (1, 3, 13),
                (1, 4, 14),
            ],
        )
        conn.commit()
    finally:
        conn.close()


class ZoteroShadowTests(unittest.TestCase):
    def test_create_shadow_copy_reads_source_read_only_and_supports_queries(self) -> None:
        with workspace_tmpdir("zotero-shadow-") as root:
            source = root / "zotero.sqlite"
            shadow = root / "shadow" / "zotero.sqlite"
            build_minimal_zotero_db(source)

            target = create_shadow_copy(source, shadow)
            self.assertEqual(shadow.resolve(), target.resolve())
            self.assertTrue(target.exists())

            reader = ZoteroShadow(target)
            try:
                attachments = reader.list_attachments()
                self.assertEqual(1, len(attachments))
                self.assertEqual("ATTACH1", attachments[0].attachment_key)
                self.assertEqual("PARENT1", attachments[0].parent_key)
                self.assertEqual("storage:paper.pdf", attachments[0].relative_path)
                self.assertEqual("Example Paper", attachments[0].title)
                self.assertEqual(1, reader.pdf_count())
            finally:
                reader.close()

            with self.assertRaises(sqlite3.OperationalError):
                conn = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True)
                try:
                    conn.execute("INSERT INTO items(itemID, key) VALUES (99, 'NOPE')")
                finally:
                    conn.close()

    def test_create_shadow_copy_rejects_same_source_and_destination(self) -> None:
        with workspace_tmpdir("zotero-shadow-") as root:
            source = root / "zotero.sqlite"
            build_minimal_zotero_db(source)
            with self.assertRaises(ValueError):
                create_shadow_copy(source, source)

    def test_create_shadow_copy_retries_readonly_after_immutable_failure(self) -> None:
        with workspace_tmpdir("zotero-shadow-fallback-") as root:
            source = root / "zotero.sqlite"
            shadow = root / "shadow" / "zotero.sqlite"
            build_minimal_zotero_db(source)
            calls = []

            def fake_backup(source_path, target_path, *, immutable, timeout_seconds):
                calls.append((Path(source_path), Path(target_path), immutable, timeout_seconds))
                if immutable:
                    raise sqlite3.OperationalError("simulated immutable backup failure")
                Path(target_path).write_bytes(Path(source_path).read_bytes())

            with patch.object(shadow_module, "_backup_readonly_source", side_effect=fake_backup):
                target = create_shadow_copy(source, shadow, timeout_seconds=1.5)

            self.assertEqual(shadow.resolve(), target.resolve())
            self.assertEqual([True, False], [call[2] for call in calls])
            self.assertEqual([1.5, 1.5], [call[3] for call in calls])
            self.assertNotEqual(calls[0][1], calls[1][1])

            reader = ZoteroShadow(target)
            try:
                self.assertEqual(1, reader.pdf_count())
            finally:
                reader.close()
