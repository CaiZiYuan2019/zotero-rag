from __future__ import annotations

import builtins
import json
import unittest
from unittest.mock import patch

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.normalize import ImagePolicy, normalize_markdown_document


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
                # Image lines are now skipped from text chunks; images have their own
                # chunks with surrounding text context.
                self.assertNotIn("images/hash-b.png", text_chunks[0]["text"])
                self.assertNotIn("images/img001.png", text_chunks[0]["text"])
                self.assertNotIn("[Image: Figure A]", text_chunks[0]["text"])
                self.assertIn("[Image: Figure A]", image_chunks[0]["text"])
                # The image chunk should include some surrounding text context.
                self.assertIn("Intro paragraph", image_chunks[0]["text"])
                self.assertEqual("images/img001.png", image_chunks[0]["metadata"]["image_path"])
                self.assertEqual(
                    "embedding_images/img001.png",
                    image_chunks[0]["metadata"]["image_embedding_path"],
                )
                self.assertEqual({"image": 2, "text": len(chunks) - 2}, ledger.status_summary()["chunks"])
            finally:
                ledger.close()

    def test_short_sections_are_merged_until_target_token_threshold(self) -> None:
        with workspace_tmpdir("normalize-merge-") as tmpdir:
            source_dir = tmpdir / "mineru"
            source_dir.mkdir(parents=True)
            markdown = source_dir / "full.md"
            # Each "word" is ~5 characters, so ~250 words give ~1250 characters and
            # therefore ~312 tokens (estimate_tokens uses len // 4).
            words = [f"word{i}" for i in range(250)]
            short_para = " ".join(words)
            markdown.write_text(
                f"# Title\n\n{short_para}\n\n"
                f"## Section A\n\n{short_para}\n\n"
                f"## Section B\n\n{short_para}\n",
                encoding="utf-8",
            )
            result = normalize_markdown_document(
                source_markdown=markdown,
                output_root=tmpdir / "normalized",
                document_id="DOC_MERGE",
            )
            text_chunks = [c for c in result.chunks if c["chunk_type"] == "text"]
            # The three short sections are merged into a single chunk because each
            # section is below the 1000-token target threshold and the total is below
            # the 2200-token maximum.
            self.assertEqual(1, len(text_chunks))
            self.assertGreaterEqual(text_chunks[0]["metadata"]["token_estimate"], 900)

    def test_long_sections_are_still_split_at_max_token_threshold(self) -> None:
        with workspace_tmpdir("normalize-split-") as tmpdir:
            source_dir = tmpdir / "mineru"
            source_dir.mkdir(parents=True)
            markdown = source_dir / "full.md"
            # 2400 words of ~5 chars each -> ~14400 chars -> ~3600 tokens.
            # Split across multiple lines so that a flush leaves remaining content.
            words = [f"word{i}" for i in range(2400)]
            long_lines = [" ".join(words[i : i + 100]) for i in range(0, 2400, 100)]
            long_text = "\n".join(long_lines)
            markdown.write_text(
                f"# Title\n\n{long_text}\n",
                encoding="utf-8",
            )
            result = normalize_markdown_document(
                source_markdown=markdown,
                output_root=tmpdir / "normalized",
                document_id="DOC_SPLIT",
            )
            text_chunks = [c for c in result.chunks if c["chunk_type"] == "text"]
            # The long paragraph is split into at least two chunks at the 2200-token
            # maximum boundary.
            self.assertGreaterEqual(len(text_chunks), 2)

    def test_text_chunks_include_overlap_from_previous_chunk(self) -> None:
        with workspace_tmpdir("normalize-overlap-") as tmpdir:
            source_dir = tmpdir / "mineru"
            source_dir.mkdir(parents=True)
            markdown = source_dir / "full.md"
            # 500 words -> ~5500 chars -> ~1375 tokens per section.
            # Both sections are above CHUNK_TOKEN_TARGET but below CHUNK_TOKEN_MAX,
            # so each flushes at its heading. The overlap should carry the trailing
            # ~200 tokens from the first chunk into the second chunk.
            words_a = [f"sectionA{i}" for i in range(500)]
            words_b = [f"sectionB{i}" for i in range(500)]
            section_a = " ".join(words_a)
            section_b = " ".join(words_b)
            markdown.write_text(
                f"# Title\n\n{section_a}\n\n## Section B\n\n{section_b}\n",
                encoding="utf-8",
            )
            result = normalize_markdown_document(
                source_markdown=markdown,
                output_root=tmpdir / "normalized",
                document_id="DOC_OVERLAP",
            )
            text_chunks = [c for c in result.chunks if c["chunk_type"] == "text"]
            self.assertEqual(2, len(text_chunks))
            first = text_chunks[0]["text"]
            second = text_chunks[1]["text"]
            # The second chunk should start with some overlap from the first chunk.
            self.assertIn("sectionA", second)
            self.assertIn("sectionB", second)

    def test_image_chunk_includes_surrounding_text_context(self) -> None:
        with workspace_tmpdir("normalize-image-context-") as tmpdir:
            source_dir = tmpdir / "mineru"
            images_dir = source_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "fig.png").write_bytes(b"fake-image")
            markdown = source_dir / "full.md"
            markdown.write_text(
                "# Title\n\n"
                "Paragraph before the figure with enough words.\n\n"
                "![Figure 1](images/fig.png)\n\n"
                "Paragraph after the figure with enough words.\n",
                encoding="utf-8",
            )
            result = normalize_markdown_document(
                source_markdown=markdown,
                output_root=tmpdir / "normalized",
                document_id="DOC_IMG_CTX",
            )
            image_chunks = [c for c in result.chunks if c["chunk_type"] == "image"]
            self.assertEqual(1, len(image_chunks))
            text = image_chunks[0]["text"]
            self.assertIn("[Image: Figure 1]", text)
            self.assertIn("before", text)
            self.assertIn("after", text)

    def test_final_text_chunk_preserves_document_tail(self) -> None:
        with workspace_tmpdir("normalize-tail-") as tmpdir:
            source_dir = tmpdir / "mineru"
            source_dir.mkdir(parents=True)
            markdown = source_dir / "full.md"
            # 500 words -> ~5500 chars -> ~1375 tokens, above overlap threshold.
            words = [f"word{i}" for i in range(500)]
            long_para = " ".join(words)
            markdown.write_text(
                f"# Title\n\n{long_para} PRESERVE_THIS_TAIL_MARKER\n",
                encoding="utf-8",
            )
            result = normalize_markdown_document(
                source_markdown=markdown,
                output_root=tmpdir / "normalized",
                document_id="DOC_TAIL",
            )
            text_chunks = [c for c in result.chunks if c["chunk_type"] == "text"]
            combined = "\n".join(c["text"] for c in text_chunks)
            self.assertIn("PRESERVE_THIS_TAIL_MARKER", combined)

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

    def test_oversized_image_gets_resized_embedding_derivative_when_pillow_is_available(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        with workspace_tmpdir("normalize-md-resize-") as tmpdir:
            source_dir = tmpdir / "mineru"
            images_dir = source_dir / "images"
            images_dir.mkdir(parents=True)
            image_path = images_dir / "large.png"
            Image.new("RGB", (240, 120), color=(12, 34, 56)).save(image_path)
            markdown = source_dir / "full.md"
            markdown.write_text("# Figure\n\n![Large](images/large.png)\n", encoding="utf-8")

            result = normalize_markdown_document(
                source_markdown=markdown,
                output_root=tmpdir / "normalized",
                document_id="DOC_RESIZE",
                image_policy=ImagePolicy(embedding_max_long_edge=100),
            )
            image_manifest = json.loads(result.image_manifest.read_text(encoding="utf-8"))
            policy = image_manifest[0]["embedding_policy"]

            self.assertTrue((result.artifact_dir / "images" / "img001.png").is_file())
            self.assertTrue((result.artifact_dir / "embedding_images" / "img001.png").is_file())
            self.assertEqual("ready_resized", policy["status"])
            self.assertEqual("embedding_images/img001.png", policy["embedding_relative_path"])
            self.assertEqual(100, policy["embedding_width"])
            self.assertEqual(50, policy["embedding_height"])

    def test_oversized_image_reports_explicit_reason_when_pillow_is_missing(self) -> None:
        real_import = builtins.__import__

        def block_pil(name: str, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("Pillow is unavailable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_pil):
            with workspace_tmpdir("normalize-md-no-pillow-") as tmpdir:
                source_dir = tmpdir / "mineru"
                images_dir = source_dir / "images"
                images_dir.mkdir(parents=True)
                image_path = images_dir / "large.png"
                image_path.write_bytes(png_header(width=2000, height=1000))
                markdown = source_dir / "full.md"
                markdown.write_text("# Figure\n\n![Large](images/large.png)\n", encoding="utf-8")

                result = normalize_markdown_document(
                    source_markdown=markdown,
                    output_root=tmpdir / "normalized",
                    document_id="DOC_NO_PILLOW",
                )
                image_manifest = json.loads(result.image_manifest.read_text(encoding="utf-8"))
                policy = image_manifest[0]["embedding_policy"]

                self.assertEqual("pending_resize", policy["status"])
                self.assertIsNone(policy["embedding_relative_path"])
                self.assertIn("Pillow", policy["reason"])

    def test_corrupt_image_reports_resize_failure_reason(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        with workspace_tmpdir("normalize-md-corrupt-") as tmpdir:
            source_dir = tmpdir / "mineru"
            images_dir = source_dir / "images"
            images_dir.mkdir(parents=True)
            image_path = images_dir / "large.png"
            # A syntactically valid PNG header with huge dimensions triggers the
            # resize path, but the file body is truncated so Pillow fails.
            image_path.write_bytes(png_header(width=9000, height=9000))
            markdown = source_dir / "full.md"
            markdown.write_text("# Figure\n\n![Large](images/large.png)\n", encoding="utf-8")

            result = normalize_markdown_document(
                source_markdown=markdown,
                output_root=tmpdir / "normalized",
                document_id="DOC_CORRUPT",
            )
            image_manifest = json.loads(result.image_manifest.read_text(encoding="utf-8"))
            policy = image_manifest[0]["embedding_policy"]

            self.assertEqual("pending_resize", policy["status"])
            self.assertIsNone(policy["embedding_relative_path"])
            self.assertIn("resize failed", policy["reason"])

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
