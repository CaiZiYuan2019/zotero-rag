from __future__ import annotations

import base64
import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.embeddings import (
    EmbeddingInput,
    EmbeddingVector,
    StubEmbeddingProvider,
    index_normalized_document,
    search_vector_index,
)
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
                text_index = next(index for index in ledger.list_vector_indexes() if index["profile_name"] == "stub_text")
                self.assertEqual(text_result.embedding_batch_hash, text_index["active_version"])
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

    def test_direct_indexing_rejects_non_stub_profile_without_explicit_override(self) -> None:
        with workspace_tmpdir("embedding-indexer-guard-") as tmpdir:
            source_dir = tmpdir / "mineru"
            source_dir.mkdir(parents=True)
            markdown = source_dir / "full.md"
            markdown.write_text("# Demo Paper\n\nalpha beta gamma text evidence\n", encoding="utf-8")
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="qwen_text",
                            provider="dashscope",
                            model="qwen3-vl-embedding",
                            dimension=8,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                        )
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

                with self.assertRaises(NotImplementedError):
                    index_normalized_document(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="qwen_text",
                        document_id="DOC1",
                    )

                result = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="qwen_text",
                    document_id="DOC1",
                    allow_stub_provider=True,
                )
                self.assertEqual(1, result.indexed_chunks)
                batch = ledger.list_embedding_batches(profile_name="qwen_text", document_id="DOC1")[0]
                self.assertEqual("stub", batch["provider"])

                with self.assertRaises(NotImplementedError):
                    search_vector_index(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="qwen_text",
                        query="alpha beta",
                        mode="text",
                    )

                hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="qwen_text",
                    query="alpha beta",
                    mode="text",
                    provider=StubEmbeddingProvider(dimension=8),
                )
                self.assertTrue(hits)
            finally:
                ledger.close()

    def test_incremental_indexing_keeps_previous_documents_visible(self) -> None:
        with workspace_tmpdir("embedding-indexer-incremental-") as tmpdir:
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
                        )
                    ]
                )
                seed_markdown_document(tmpdir, ledger, document_id="DOC_ALPHA", text="alpha unique evidence")
                seed_markdown_document(tmpdir, ledger, document_id="DOC_BETA", text="beta unique evidence")

                index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC_ALPHA",
                )
                index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC_BETA",
                )

                alpha_hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    query="alpha unique evidence",
                    mode="text",
                    top_k=5,
                )
                indexed = next(index for index in ledger.list_vector_indexes() if index["profile_name"] == "stub_text")

                self.assertEqual(2, indexed["document_count"])
                self.assertEqual({"DOC_ALPHA", "DOC_BETA"}, {hit["document_id"] for hit in alpha_hits})
            finally:
                ledger.close()

    def test_indexing_reuses_completed_batch_without_provider_call(self) -> None:
        with workspace_tmpdir("embedding-indexer-cache-") as tmpdir:
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
                        )
                    ]
                )
                seed_markdown_document(tmpdir, ledger, document_id="DOC_CACHE", text="cacheable evidence")
                first_provider = CountingProvider(dimension=8)
                first = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC_CACHE",
                    provider=first_provider,
                )
                second_provider = FailingProvider(dimension=8)
                second = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC_CACHE",
                    provider=second_provider,
                )

                self.assertEqual(1, first_provider.calls)
                self.assertTrue(second.reused_existing)
                self.assertEqual(first.embedding_batch_hash, second.embedding_batch_hash)
                self.assertEqual(0, second_provider.calls)
                checkpoint = ledger.get_checkpoint("DOC_CACHE", "embed:stub_text")
                self.assertTrue(checkpoint["payload"]["reused_existing_embedding"])
                batches = ledger.list_embedding_batches(profile_name="stub_text", document_id="DOC_CACHE")
                self.assertEqual(1, len(batches))
                self.assertEqual("completed", batches[0]["status"])
            finally:
                ledger.close()

    def test_non_stub_profile_can_reuse_completed_batch_without_provider(self) -> None:
        with workspace_tmpdir("embedding-indexer-qwen-reuse-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="qwen_text",
                            provider="dashscope",
                            model="qwen3-vl-embedding",
                            dimension=8,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                        )
                    ]
                )
                seed_markdown_document(tmpdir, ledger, document_id="DOC_QWEN", text="qwen reuse evidence")
                first = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="qwen_text",
                    document_id="DOC_QWEN",
                    allow_stub_provider=True,
                )
                second = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="qwen_text",
                    document_id="DOC_QWEN",
                )

                self.assertTrue(second.reused_existing)
                self.assertEqual(first.embedding_batch_hash, second.embedding_batch_hash)
            finally:
                ledger.close()


def seed_markdown_document(tmpdir, ledger: StateLedger, *, document_id: str, text: str):
    source_dir = tmpdir / f"mineru-{document_id}"
    source_dir.mkdir(parents=True)
    markdown = source_dir / "full.md"
    markdown.write_text(f"# {document_id}\n\n{text}\n", encoding="utf-8")
    normalized = normalize_markdown_document(
        source_markdown=markdown,
        output_root=tmpdir / "normalized",
        document_id=document_id,
        attachment_key=f"ATT_{document_id}",
    )
    ledger.upsert_normalized_artifact(normalized.ledger_artifact())
    ledger.replace_document_chunks(normalized.document_id, normalized.chunks)
    return normalized


class CountingProvider:
    def __init__(self, dimension: int = 8) -> None:
        self.name = "counting"
        self.model = "counting"
        self.dimension = dimension
        self.calls = 0
        self._stub = StubEmbeddingProvider(dimension=dimension)

    def embed(self, inputs: list[EmbeddingInput]) -> list[EmbeddingVector]:
        self.calls += 1
        return self._stub.embed(inputs)


class FailingProvider:
    def __init__(self, dimension: int = 8) -> None:
        self.name = "failing"
        self.model = "failing"
        self.dimension = dimension
        self.calls = 0

    def embed(self, inputs: list[EmbeddingInput]) -> list[EmbeddingVector]:
        self.calls += 1
        raise AssertionError(
            "embedding provider should not be called when a completed batch is reusable"
        )


if __name__ == "__main__":
    unittest.main()
