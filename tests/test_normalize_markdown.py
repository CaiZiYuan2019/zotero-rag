from __future__ import annotations

import json
import unittest

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.normalize import normalize_markdown_document


class NormalizeMarkdownTests(unittest.TestCase):
    def test_normalize_rewrites_images_and_persists_chunks(self) -> None:
        with workspace_tmpdir("normalize-md-") as tmpdir:
            source_dir = tmpdir / "mineru"
            images_dir = source_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "hash-b.png").write_bytes(b"fake-image-b")
            (images_dir / "hash-a.jpg").write_bytes(b"fake-image-a")
            markdown = source_dir / "full.md"
            markdown.write_text(
                "# Paper Title\n\n"
                "Intro paragraph with enough text for a chunk.\n\n"
                "![Figure A](images/hash-b.png)\n\n"
                "## Methods\n\n"
                "Methods paragraph.\n"
                "<img src=\"images/hash-a.jpg\">\n",
                encoding="utf-8",
            )
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                result = normalize_markdown_document(
                    source_markdown=markdown,
                    output_root=tmpdir / "normalized",
                    document_id="DOC1",
                    attachment_key="ATT1",
                    extract_job_id="JOB1",
                )
                artifact = ledger.upsert_normalized_artifact(result.ledger_artifact())
                ledger.replace_document_chunks(result.document_id, result.chunks)

                rewritten = result.document_md.read_text(encoding="utf-8")
                image_manifest = json.loads(result.image_manifest.read_text(encoding="utf-8"))
                chunks = ledger.list_chunks("DOC1")

                self.assertIn("images/img001.png", rewritten)
                self.assertIn('src="images/img002.jpg"', rewritten)
                self.assertEqual("ATT1", artifact["attachment_key"])
                self.assertEqual(2, artifact["image_count"])
                self.assertEqual(2, len(image_manifest))
                self.assertEqual("hash-b.png", image_manifest[0]["original_name"])
                self.assertEqual("img001.png", image_manifest[0]["ordered_name"])
                self.assertEqual("Figure A", image_manifest[0]["alt_text"])
                self.assertTrue((result.artifact_dir / "images" / "img001.png").is_file())
                self.assertTrue((result.artifact_dir / "chunks.jsonl").is_file())
                self.assertGreaterEqual(len(chunks), 3)
                text_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "text"]
                self.assertNotIn("images/hash-b.png", text_chunks[0]["text"])
                self.assertNotIn("images/img001.png", text_chunks[0]["text"])
                self.assertIn("[Image: Figure A]", text_chunks[0]["text"])
                self.assertEqual({"image": 2, "text": len(chunks) - 2}, ledger.status_summary()["chunks"])
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
