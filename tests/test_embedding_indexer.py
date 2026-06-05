from __future__ import annotations

import base64
import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.embeddings import index_normalized_document, search_vector_index
from zoterorag.embeddings.profile import embedding_profile_hash
from zoterorag.normalize import normalize_markdown_document


class EmbeddingIndexerTests(unittest.TestCase):
    def test_stub_indexer_indexes_text_and_multimodal_profiles_separately(self) -> None:
        with workspace_tmpdir("embedding-indexer-") as tmpdir:
            source_dir = tmpdir / "mineru"
            images_dir = source_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "fig.png").write_bytes(b"fake-image")
            markdown = source_dir / "full.md"
            markdown.write_text(
                "# Demo Paper\n\n"
                "alpha beta gamma text evidence\n\n"
                "![important figure](images/fig.png)\n",
                encoding="utf-8",
            )
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="stub_text",
                            provider="stub",
                            model="stub",
                            dimension=8,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                        ),
                        EmbeddingProfile(
                            name="stub_mm",
                            provider="stub",
                            model="stub",
                            dimension=8,
                            modality="multimodal",
                            enabled=True,
                            default_for_multimodal=True,
                        ),
                    ]
                )
                normalized = normalize_markdown_document(
                    source_markdown=markdown,
                    output_root=tmpdir / "normalized",
                    document_id="DOC1",
                    attachment_key="ATT1",
                )
                ledger.upsert_normalized_artifact(normalized.ledger_artifact())
                ledger.replace_document_chunks(normalized.document_id, normalized.chunks)

                text_result = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC1",
                )
                mm_result = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_mm",
                    document_id="DOC1",
                )
                text_hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    query="alpha beta gamma",
                    mode="text",
                    consumer="llm_text",
                )
                mm_text_only_hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_mm",
                    query="important figure",
                    mode="multimodal",
                    consumer="llm_text",
                    image_return="none",
                )
                mm_manual_hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_mm",
                    query="important figure",
                    mode="multimodal",
                    consumer="manual",
                    image_return="file_ref",
                )
                mm_base64_hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_mm",
                    query="important figure",
                    mode="multimodal",
                    consumer="llm_multimodal",
                    image_return="base64",
                    max_image_bytes=1024,
                )
                mm_base64_omitted_hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_mm",
                    query="important figure",
                    mode="multimodal",
                    consumer="llm_multimodal",
                    image_return="base64",
                    max_image_bytes=1,
                )
                mm_image_query_hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_mm",
                    query="important figure",
                    mode="multimodal",
                    consumer="llm_text",
                    image_return="none",
                    query_image_base64=base64.b64encode(b"query-image").decode("ascii"),
                    query_image_mime_type="image/png",
                )

                self.assertEqual(1, text_result.indexed_chunks)
                self.assertEqual(1, mm_result.indexed_chunks)
                self.assertEqual(64, len(text_result.embedding_batch_hash))
                self.assertEqual(
                    embedding_profile_hash(
                        next(profile for profile in ledger.list_embedding_profiles() if profile["name"] == "stub_text")
                    ),
                    text_result.profile_hash,
                )
                checkpoint = ledger.get_checkpoint("DOC1", "embed:stub_text")
                self.assertEqual(text_result.profile_hash, checkpoint["payload"]["profile_hash"])
                self.assertEqual(text_result.embedding_batch_hash, checkpoint["payload"]["embedding_batch_hash"])
                batches = ledger.list_embedding_batches(profile_name="stub_text", document_id="DOC1")
                self.assertEqual(1, len(batches))
                self.assertEqual("completed", batches[0]["status"])
                self.assertEqual(text_result.embedding_batch_hash, batches[0]["batch_hash"])
                self.assertEqual(
                    [chunk["chunk_id"] for chunk in ledger.list_chunks("DOC1", chunk_type="text")],
                    batches[0]["payload"]["input_ids"],
                )
                self.assertNotIn("alpha beta gamma", repr(batches[0]))
                self.assertEqual("stub_text", text_hits[0]["metadata"]["profile_name"])
                self.assertEqual(text_result.profile_hash, text_hits[0]["metadata"]["profile_hash"])
                self.assertIsNone(text_hits[0]["rerank_score"])
                self.assertNotIn("images", mm_text_only_hits[0])
                self.assertTrue(mm_text_only_hits[0]["has_images"])
                self.assertIn("images", mm_manual_hits[0])
                self.assertEqual("images/img001.png", mm_manual_hits[0]["images"][0]["file_ref"])
                self.assertNotIn("file_ref", mm_base64_hits[0]["images"][0])
                self.assertEqual(base64.b64encode(b"fake-image").decode("ascii"), mm_base64_hits[0]["images"][0]["base64"])
                self.assertEqual("image/png", mm_base64_hits[0]["images"][0]["mime_type"])
                self.assertIn("omitted_reason", mm_base64_omitted_hits[0]["images"][0])
                self.assertNotIn("base64", mm_base64_omitted_hits[0]["images"][0])
                self.assertTrue(mm_image_query_hits)
                with self.assertRaises(NotImplementedError):
                    search_vector_index(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="stub_text",
                        query="alpha beta gamma",
                        mode="text",
                        consumer="llm_text",
                        rerank=True,
                    )
                with self.assertRaises(ValueError):
                    search_vector_index(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="stub_text",
                        query="alpha beta gamma",
                        mode="text",
                        consumer="llm_text",
                        query_image_base64=base64.b64encode(b"query-image").decode("ascii"),
                    )
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
