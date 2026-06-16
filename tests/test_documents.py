from __future__ import annotations

import json
import unittest

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.documents import get_document, list_documents
from zoterorag.normalize import normalize_markdown_document


class DocumentServiceTests(unittest.TestCase):
    def test_lists_normalized_and_metadata_only_documents(self) -> None:
        with workspace_tmpdir("documents-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_attachments(
                    [
                        build_attachment("ATT1", title="Indexed Paper"),
                        build_attachment("ATT2", title="Review-only Paper", classification="needs_review"),
                    ]
                )
                normalized = normalize_sample_document(tmpdir, ledger)

                documents = list_documents(ledger, limit=None)
                by_id = {item["document_id"]: item for item in documents}

                self.assertEqual("normalized", by_id[normalized.document_id]["document_kind"])
                self.assertEqual("Indexed Paper", by_id[normalized.document_id]["title"])
                self.assertEqual("metadata_only", by_id["ATT2"]["document_kind"])
                self.assertFalse(by_id["ATT2"]["normalized"])
                self.assertNotIn("attachment", by_id["ATT2"])

                normalized_only = list_documents(ledger, limit=None, include_metadata_only=False)
                self.assertEqual([normalized.document_id], [item["document_id"] for item in normalized_only])
            finally:
                ledger.close()

    def test_get_document_defaults_to_text_safe_chunks(self) -> None:
        with workspace_tmpdir("documents-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_attachments([build_attachment("ATT1", title="Paper With Figure")])
                normalized = normalize_sample_document(tmpdir, ledger)

                document = get_document(ledger, normalized.document_id, include_chunks=True)

                self.assertIsNotNone(document)
                serialized = json.dumps(document, ensure_ascii=False)
                self.assertNotIn("images/img001.png", serialized)
                self.assertNotIn("document_md", document["artifact"])
                self.assertTrue(document["chunks"])
                self.assertEqual({"text"}, {chunk["chunk_type"] for chunk in document["chunks"]})
                # Image lines are now represented by dedicated image chunks, not by
                # placeholders inside text chunks.
                self.assertNotIn("[Image: Figure A]", document["chunks"][0]["text"])

                image_document = get_document(
                    ledger, normalized.document_id, include_chunks=True, chunk_type="image"
                )
                self.assertEqual({"image"}, {chunk["chunk_type"] for chunk in image_document["chunks"]})
                self.assertIn("[Image: Figure A]", image_document["chunks"][0]["text"])
            finally:
                ledger.close()

    def test_manual_document_read_can_return_image_file_refs(self) -> None:
        with workspace_tmpdir("documents-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_attachments([build_attachment("ATT1", title="Paper With Figure")])
                normalized = normalize_sample_document(tmpdir, ledger)

                document = get_document(
                    ledger,
                    normalized.document_id,
                    include_chunks=True,
                    chunk_type="image",
                    consumer="manual",
                )

                self.assertIsNotNone(document)
                image_chunk = document["chunks"][0]
                self.assertEqual("image", image_chunk["chunk_type"])
                self.assertIn("images/img001.png", image_chunk["metadata"]["image_path"])
                self.assertTrue(image_chunk["images"][0]["file_ref"].endswith("images\\img001.png"))

                metadata_only = get_document(ledger, "ATT1", consumer="manual")
                self.assertIsNotNone(metadata_only)
                self.assertIn("attachment", metadata_only)
                self.assertEqual("ATT1", metadata_only["attachment"]["attachment_key"])
            finally:
                ledger.close()


def normalize_sample_document(tmpdir, ledger: StateLedger):
    source_dir = tmpdir / "mineru"
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "hash-a.png").write_bytes(b"fake-image")
    markdown = source_dir / "full.md"
    markdown.write_text(
        "# Paper With Figure\n\n"
        "Intro text before the figure.\n\n"
        "![Figure A](images/hash-a.png)\n\n"
        "More text after the figure.\n",
        encoding="utf-8",
    )
    normalized = normalize_markdown_document(
        source_markdown=markdown,
        output_root=tmpdir / "normalized",
        document_id="DOC1",
        attachment_key="ATT1",
    )
    ledger.upsert_normalized_artifact(normalized.ledger_artifact())
    ledger.replace_document_chunks(normalized.document_id, normalized.chunks)
    return normalized


def build_attachment(
    attachment_key: str,
    *,
    title: str,
    classification: str = "included_auto",
) -> dict[str, object]:
    return {
        "attachment_key": attachment_key,
        "parent_key": f"PARENT-{attachment_key}",
        "content_type": "application/pdf",
        "relative_path": f"storage:{attachment_key}.pdf",
        "title": title,
        "abstract": "Abstract text",
        "date": "2024",
        "url": "https://example.test",
        "classification": classification,
        "source_quality": "primary_candidate",
        "reasons": [],
        "file_path": f"C:/Zotero/storage/{attachment_key}.pdf",
        "file_exists": True,
        "file_size": 100,
        "file_mtime": 1.0,
        "metadata": {"authors": ["Ada"]},
    }


if __name__ == "__main__":
    unittest.main()
