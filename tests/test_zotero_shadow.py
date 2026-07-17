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
                key TEXT NOT NULL,
                itemTypeID INTEGER,
                dateAdded TEXT,
                dateModified TEXT,
                libraryID INTEGER
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
            CREATE TABLE itemTypes (
                itemTypeID INTEGER PRIMARY KEY,
                typeName TEXT NOT NULL
            );
            CREATE TABLE creators (
                creatorID INTEGER PRIMARY KEY,
                firstName TEXT,
                lastName TEXT,
                fieldMode INTEGER
            );
            CREATE TABLE creatorTypes (
                creatorTypeID INTEGER PRIMARY KEY,
                creatorType TEXT NOT NULL
            );
            CREATE TABLE itemCreators (
                itemID INTEGER NOT NULL,
                creatorID INTEGER NOT NULL,
                creatorTypeID INTEGER NOT NULL,
                orderIndex INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE collections (
                collectionID INTEGER PRIMARY KEY,
                collectionName TEXT NOT NULL,
                parentCollectionID INTEGER,
                libraryID INTEGER,
                key TEXT
            );
            CREATE TABLE collectionItems (
                collectionID INTEGER NOT NULL,
                itemID INTEGER NOT NULL,
                orderIndex INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE tags (
                tagID INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE itemTags (
                itemID INTEGER NOT NULL,
                tagID INTEGER NOT NULL,
                type INTEGER NOT NULL DEFAULT 0
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
                (5, "publicationTitle"),
                (6, "DOI"),
            ],
        )
        conn.executemany(
            "INSERT INTO itemTypes(itemTypeID, typeName) VALUES (?, ?)",
            [(1, "journalArticle"), (2, "attachment")],
        )
        conn.executemany(
            "INSERT INTO creatorTypes(creatorTypeID, creatorType) VALUES (?, ?)",
            [(1, "author"), (2, "editor")],
        )
        conn.executemany(
            "INSERT INTO creators(creatorID, firstName, lastName, fieldMode) VALUES (?, ?, ?, ?)",
            [
                (1, "Ada", "Lovelace", 0),
                (2, "", "OpenAI", 1),
                (3, "Grace", "Hopper", 0),
            ],
        )
        conn.executemany(
            "INSERT INTO items(itemID, key, itemTypeID, dateAdded, dateModified, libraryID) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "PARENT1", 1, "2024-01-01 00:00:00", "2024-02-02 00:00:00", 1),
                (2, "ATTACH1", 2, "2024-01-03 00:00:00", "2024-02-03 00:00:00", 1),
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
                (15, "Journal of Examples"),
                (16, "10.1234/example"),
            ],
        )
        conn.executemany(
            "INSERT INTO itemData(itemID, fieldID, valueID) VALUES (?, ?, ?)",
            [
                (1, 1, 11),
                (1, 2, 12),
                (1, 3, 13),
                (1, 4, 14),
                (1, 5, 15),
                (1, 6, 16),
            ],
        )
        # Two authors (one a single-field institution) plus one editor that must
        # be excluded from the authors list.
        conn.executemany(
            "INSERT INTO itemCreators(itemID, creatorID, creatorTypeID, orderIndex) VALUES (?, ?, ?, ?)",
            [
                (1, 1, 1, 0),
                (1, 2, 1, 1),
                (1, 3, 2, 2),
            ],
        )
        conn.executemany(
            "INSERT INTO collections(collectionID, collectionName, parentCollectionID, libraryID, key) VALUES (?, ?, ?, ?, ?)",
            [
                (1, "Root Collection", None, 1, "COLLROOT"),
                (2, "Child Collection", 1, 1, "COLLCHILD"),
            ],
        )
        conn.execute("INSERT INTO collectionItems(collectionID, itemID, orderIndex) VALUES (?, ?, ?)", (2, 1, 0))
        conn.executemany(
            "INSERT INTO tags(tagID, name) VALUES (?, ?)",
            [(1, "history"), (2, "computing")],
        )
        conn.executemany(
            "INSERT INTO itemTags(itemID, tagID, type) VALUES (?, ?, ?)",
            [(1, 1, 0), (1, 2, 0)],
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

                att = attachments[0]
                # Only creatorType 'author' is included; the editor is dropped.
                self.assertEqual(["Ada Lovelace", "OpenAI"], att.authors)
                self.assertEqual({"history", "computing"}, set(att.tags))
                self.assertEqual("journalArticle", att.item_type)
                self.assertEqual("Journal of Examples", att.publication_title)
                self.assertEqual("10.1234/example", att.doi)
                self.assertEqual("2024-01-01 00:00:00", att.date_added)
                self.assertEqual("2024-02-02 00:00:00", att.date_modified)
                self.assertEqual(1, len(att.collections))
                self.assertEqual("COLLCHILD", att.collections[0]["collection_key"])
                self.assertEqual("Child Collection", att.collections[0]["collection_name"])
                self.assertEqual("COLLROOT", att.collections[0]["parent_collection_key"])

                collections = reader.list_collections()
                self.assertEqual(2, len(collections))
                by_key = {c.collection_key: c for c in collections}
                self.assertIsNone(by_key["COLLROOT"].parent_collection_key)
                self.assertEqual("COLLROOT", by_key["COLLCHILD"].parent_collection_key)
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
