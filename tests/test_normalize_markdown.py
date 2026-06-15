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
                self.assertTrue((result.artifact_dir / "embedding_images" / "img001.png").is_file())
                self.assertEqual(
                    "embedding_images/img001.png",
                    image_manifest[0]["embedding_policy"]["embedding_relative_path"],
                )
                self.assertTrue((result.artifact_dir / "chunks.jsonl").is_file())
                self.assertGreaterEqual(len(chunks), 3)
                text_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "text"]
                image_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "image"]
                self.assertNotIn("images/hash-b.png", text_chunks[0]["text"])
                self.assertNotIn("images/img001.png", text_chunks[0]["text"])
                self.assertIn("[Image: Figure A]", text_chunks[0]["text"])
                self.assertEqual("images/img001.png", image_chunks[0]["metadata"]["image_path"])
                self.assertEqual(
                    "embedding_images/img001.png",
                    image_chunks[0]["metadata"]["image_embedding_path"],
                )
                self.assertEqual({"image": 2, "text": len(chunks) - 2}, ledger.status_summary()["chunks"])
            finally:
                ledger.close()

    def test_image_manifest_records_dimensions_and_consecutive_runs(self) -> None:
        with workspace_tmpdir("normalize-md-") as tmpdir:
            source_dir = tmpdir / "mineru"
            images_dir = source_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "panel-a.png").write_bytes(png_header(width=2000, height=1000))
            (images_dir / "panel-b.png").write_bytes(png_header(width=800, height=600))
            (images_dir / "separate.png").write_bytes(png_header(width=40, height=40))
            markdown = source_dir / "full.md"
            markdown.write_text(
                "# Figures\n\n"
                "![Panel A](images/panel-a.png)\n"
                "![Panel B](images/panel-b.png)\n\n"
                "A paragraph separates this image from the panel run.\n\n"
                "![Separate](images/separate.png)\n",
                encoding="utf-8",
            )

            result = normalize_markdown_document(
                source_markdown=markdown,
                output_root=tmpdir / "normalized",
                document_id="DOC_RUN",
            )
            image_manifest = json.loads(result.image_manifest.read_text(encoding="utf-8"))
            image_chunks = [chunk for chunk in result.chunks if chunk["chunk_type"] == "image"]

            self.assertEqual(3, len(image_manifest))
            self.assertEqual(2000, image_manifest[0]["width"])
            self.assertEqual(1000, image_manifest[0]["height"])
            self.assertIn("exceeds_embedding_long_edge", image_manifest[0]["image_quality_flags"])
            self.assertEqual("pending_resize", image_manifest[0]["embedding_policy"]["status"])
            self.assertIsNone(image_manifest[0]["embedding_policy"]["embedding_relative_path"])
            self.assertEqual("ready_embedding_copy", image_manifest[1]["embedding_policy"]["status"])
            self.assertEqual(
                "embedding_images/img002.png",
                image_manifest[1]["embedding_policy"]["embedding_relative_path"],
            )
            self.assertIn("tiny_image", image_manifest[2]["image_quality_flags"])

            self.assertEqual("run:00001", image_manifest[0]["image_run_id"])
            self.assertEqual("run:00001", image_manifest[1]["image_run_id"])
            self.assertEqual("run:00002", image_manifest[2]["image_run_id"])
            self.assertEqual(2, image_manifest[0]["image_run_count"])
            self.assertEqual(1, image_manifest[0]["image_run_position"])
            self.assertEqual(2, image_manifest[1]["image_run_position"])
            self.assertEqual("DOC_RUN:run:00001", image_chunks[0]["metadata"]["image_run_id"])
            self.assertEqual(2, image_chunks[0]["metadata"]["image_run_count"])
            self.assertEqual("pending_resize", image_chunks[0]["metadata"]["image_embedding_status"])
            self.assertIsNone(image_chunks[0]["metadata"]["image_embedding_path"])

def png_header(*, width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\r"
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )


if __name__ == "__main__":
    unittest.main()
