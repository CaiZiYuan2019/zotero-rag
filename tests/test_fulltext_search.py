from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.normalize import normalize_markdown_document
from zoterorag.search import fulltext_search


class FulltextSearchTests(unittest.TestCase):
    def test_fulltext_search_reads_normalized_chunks_without_embeddings(self) -> None:
        with workspace_tmpdir("fulltext-search-") as tmpdir:
            source_dir = tmpdir / "mineru"
            images_dir = source_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "fig.png").write_bytes(b"fake-image")
            markdown = source_dir / "full.md"
            markdown.write_text(
                "# Methods\n\n"
                "reproducible figure workflow text\n\n"
                "![workflow figure](images/fig.png)\n",
                encoding="utf-8",
            )
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                normalized = normalize_markdown_document(
                    source_markdown=markdown,
                    output_root=tmpdir / "normalized",
                    document_id="DOC-FULLTEXT",
                )
                ledger.upsert_normalized_artifact(normalized.ledger_artifact())
                ledger.replace_document_chunks(normalized.document_id, normalized.chunks)

                text_hits = fulltext_search(
                    ledger,
                    "reproducible workflow",
                    chunk_type="text",
                    consumer="llm_text",
                )
                image_hits = fulltext_search(
                    ledger,
                    "workflow figure",
                    chunk_type="image",
                    consumer="manual",
                    image_return="file_ref",
                )

                self.assertEqual("DOC-FULLTEXT", text_hits[0]["document_id"])
                self.assertNotIn("images/fig.png", text_hits[0]["text"])
                self.assertNotIn("images/img001.png", text_hits[0]["text"])
                self.assertEqual("DOC-FULLTEXT", image_hits[0]["document_id"])
                self.assertEqual("images/img001.png", image_hits[0]["images"][0]["file_ref"])
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
