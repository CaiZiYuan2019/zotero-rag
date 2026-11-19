from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3
from pathlib import Path
import unittest
from unittest.mock import patch

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.embeddings import (
    EmbeddingInput,
    EmbeddingVector,
    IndexResult,
    StagedDocument,
    StubEmbeddingProvider,
    index_normalized_document,
    publish_staged_documents,
    search_vector_index,
    stage_indexed_document,
)
from zoterorag.embeddings.indexer import vector_path_for_profile
from zoterorag.embeddings.profile import embedding_profile_hash
from zoterorag.index import open_vector_store
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
                            backend="sqlite-local",
                        ),
                        EmbeddingProfile(
                            name="stub_mm",
                            provider="stub",
                            model="stub",
                            dimension=8,
                            modality="multimodal",
                            enabled=True,
                            default_for_multimodal=True,
                            backend="sqlite-local",
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
                mm_provider = CountingProvider(dimension=8)
                mm_result = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_mm",
                    document_id="DOC1",
                    provider=mm_provider,
                )
                # An existing vector index should pick up current Zotero identity on read.
                ledger.upsert_attachments([
                    {"attachment_key": "ATT1", "parent_key": "PARENT1", "classification": "included_auto"}
                ])
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

                for hit in [text_hits[0], mm_manual_hits[0], mm_text_only_hits[0]]:
                    self.assertEqual("DOC1", hit["document_id"])
                    self.assertEqual("ATT1", hit["attachment_key"])
                    self.assertEqual("PARENT1", hit["parent_item_key"])
                    self.assertEqual("PARENT1", hit["source_id"])
                    self.assertEqual("PARENT1", hit["metadata"]["parent_key"])
                self.assertEqual(1, text_result.indexed_chunks)
                self.assertEqual(1, mm_result.indexed_chunks)
                self.assertEqual(1, mm_provider.calls)
                self.assertTrue(mm_provider.last_inputs[0].image_path.endswith("embedding_images\\img001.png"))
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
                # Phase 3: the published version is named by the run hash
                # (sha256 of the batch hash), not the batch hash itself.
                self.assertEqual(
                    hashlib.sha256(text_result.embedding_batch_hash.encode("utf-8")).hexdigest(),
                    text_index["active_version"],
                )
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
                with self.assertRaises(RuntimeError):
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

    def test_direct_indexing_resolves_provider_from_profile(self) -> None:
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
                            backend="sqlite-local",
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

                # Without allow_stub_provider, the resolver tries to build a
                # real dashscope provider from the environment. The test
                # profile uses an invalid dimension (8) so the build fails
                # before any network call. In production this would raise
                # RuntimeError when no API key is configured.
                with self.assertRaises((RuntimeError, ValueError)):
                    index_normalized_document(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="qwen_text",
                        document_id="DOC1",
                    )

                # allow_stub_provider=True forces stub even for non-stub profiles.
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

                # A stub-built fixture needs an explicitly selected stub for queries.
                hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="qwen_text",
                    query="alpha beta",
                    mode="text",
                    allow_stub_provider=True,
                )
                self.assertTrue(hits)
                with self.assertRaises((RuntimeError, ValueError)):
                    search_vector_index(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="qwen_text",
                        query="alpha beta",
                        mode="text",
                    )

                # Passing an explicit stub provider works.
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
                            backend="sqlite-local",
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
                indexed_beta = index_normalized_document(
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

                ledger.upsert_attachments([
                    {"attachment_key": "ATT_" + doc, "parent_key": "P", "classification": "included_auto"}
                    for doc in ("DOC_ALPHA", "DOC_BETA")
                ])
                # Reproduce an older index whose siblings share a parent ID on disk.
                conn = sqlite3.connect(indexed_beta.vector_path)
                try:
                    with conn:
                        conn.execute("UPDATE vectors SET document_id = 'P'")
                finally:
                    conn.close()

                legacy_hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    query="alpha unique evidence",
                    mode="text",
                )
                self.assertEqual({"P"}, {hit["source_id"] for hit in legacy_hits})
                self.assertEqual(
                    {("DOC_ALPHA", "ATT_DOC_ALPHA"), ("DOC_BETA", "ATT_DOC_BETA")},
                    {(hit["document_id"], hit["attachment_key"]) for hit in legacy_hits},
                )
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
                            backend="sqlite-local",
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
                            backend="sqlite-local",
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


    def test_resolve_registered_vector_store_uses_registry(self) -> None:
        with workspace_tmpdir("embedding-indexer-backend-") as tmpdir:
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
                            backend="sqlite-local",
                        )
                    ]
                )
                from zoterorag.embeddings.indexer import _resolve_registered_vector_store

                # Before registration, the helper falls back to the caller
                # supplied backend and path.
                backend, path = _resolve_registered_vector_store(
                    ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    fallback_backend="sqlite-local",
                )
                self.assertEqual("sqlite-local", backend)
                self.assertEqual(
                    tmpdir / "vectors" / "stub_text" / "vectors.sqlite", path
                )

                # After registration, the stored backend/path take precedence.
                ledger.register_vector_index(
                    profile_name="stub_text",
                    backend="lancedb",
                    path=tmpdir / "vectors" / "stub_text",
                    document_count=0,
                    chunk_count=0,
                    active=True,
                    active_version="legacy",
                )
                backend, path = _resolve_registered_vector_store(
                    ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    fallback_backend="sqlite-local",
                )
                self.assertEqual("lancedb", backend)
                self.assertEqual(tmpdir / "vectors" / "stub_text", path)
            finally:
                ledger.close()

    def test_api_stub_selection_uses_stub_for_non_stub_profile(self) -> None:
        with workspace_tmpdir("embedding-indexer-api-stub-") as tmpdir:
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
                            backend="sqlite-local",
                        )
                    ]
                )
                seed_markdown_document(tmpdir, ledger, document_id="DOC1", text="api stub evidence")
                index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="qwen_text",
                    document_id="DOC1",
                    allow_stub_provider=True,
                )

                # Explicit offline selection can still query this synthetic index.
                hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="qwen_text",
                    query="api stub evidence",
                    mode="text",
                    allow_stub_provider=True,
                )
                self.assertTrue(hits)

                # The default selects the real provider.
                with self.assertRaises((RuntimeError, ValueError)):
                    search_vector_index(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="qwen_text",
                        query="api stub evidence",
                        mode="text",
                    )
            finally:
                ledger.close()

    def test_vector_path_for_profile_respects_backend(self) -> None:
        from zoterorag.embeddings.indexer import vector_path_for_profile

        sqlite_path = vector_path_for_profile("/store", "p1", backend="sqlite-local")
        lancedb_path = vector_path_for_profile("/store", "p1", backend="lancedb")
        self.assertEqual(sqlite_path, Path("/store") / "p1" / "vectors.sqlite")
        self.assertEqual(lancedb_path, Path("/store") / "p1")

    def test_indexer_uses_profile_backend_default(self) -> None:
        with workspace_tmpdir("embedding-indexer-backend-default-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="backend_text",
                            provider="stub",
                            model="stub",
                            dimension=8,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                            backend="sqlite-local",
                        )
                    ]
                )
                seed_markdown_document(
                    tmpdir, ledger, document_id="DOC_BACKEND", text="backend evidence"
                )
                result = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="backend_text",
                    document_id="DOC_BACKEND",
                )
                index = next(
                    index
                    for index in ledger.list_vector_indexes()
                    if index["profile_name"] == "backend_text"
                )
                self.assertEqual("sqlite-local", index["backend"])
                self.assertEqual(
                    tmpdir / "vectors" / "backend_text" / "vectors.sqlite",
                    result.vector_path,
                )
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
        self.last_inputs: list[EmbeddingInput] = []
        self._stub = StubEmbeddingProvider(dimension=dimension)

    def embed(self, inputs: list[EmbeddingInput]) -> list[EmbeddingVector]:
        self.calls += 1
        self.last_inputs = list(inputs)
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


class VectorIndexPruneTests(unittest.TestCase):
    def _seed_store_and_ledger(self, tmpdir: Path, *, active_version: str = "v1", active_chunks: int = 1):
        from zoterorag.index.local_vector import LocalVectorStore, VectorRecord

        ledger = StateLedger(tmpdir / "state.sqlite")
        ledger.upsert_embedding_profiles(
            [
                EmbeddingProfile(
                    name="p_text",
                    provider="stub",
                    model="stub",
                    dimension=3,
                    modality="text",
                    enabled=True,
                    default_for_text=True,
                    backend="sqlite-local",
                ),
            ]
        )
        vector_path = tmpdir / "vectors" / "p_text.sqlite"
        store = LocalVectorStore(vector_path, profile_name="p_text", dimension=3)
        if active_chunks:
            store.upsert(
                [
                    VectorRecord(
                        record_id="p1:c1",
                        document_id="DOCA",
                        chunk_id="c1",
                        vector=[1.0, 0.0, 0.0],
                        text="alpha",
                        modality="text",
                    )
                ],
                index_version="v1",
            )
        store.publish_version("v1")
        # A crash-interrupted staged version left behind as an orphan.
        store.upsert(
            [
                VectorRecord(
                    record_id="p1:c2",
                    document_id="DOCB",
                    chunk_id="c2",
                    vector=[0.0, 1.0, 0.0],
                    text="beta",
                    modality="text",
                )
            ],
            index_version="orphan",
        )
        store.close()
        ledger.register_vector_index(
            profile_name="p_text",
            backend="sqlite-local",
            path=vector_path,
            document_count=active_chunks,
            chunk_count=active_chunks,
            active=True,
            active_version=active_version,
        )
        return ledger, vector_path

    def test_prune_removes_non_active_versions_and_vacuums(self) -> None:
        from zoterorag.embeddings.indexer import prune_vector_index_versions
        from zoterorag.index.local_vector import LocalVectorStore

        with workspace_tmpdir("vector-prune-") as tmpdir:
            ledger, vector_path = self._seed_store_and_ledger(tmpdir)
            try:
                result = prune_vector_index_versions(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="p_text",
                    vacuum=True,
                )
                self.assertEqual(["v1"], result["kept_versions"])
                self.assertEqual({"orphan": 1}, result["removed_versions"])
                self.assertTrue(result["vacuumed"])

                store = LocalVectorStore(vector_path, profile_name="p_text", dimension=3)
                try:
                    self.assertEqual(["v1"], store.list_versions())
                    self.assertEqual(1, store.counts()["chunks"])
                finally:
                    store.close()
            finally:
                ledger.close()

    def test_prune_aborts_on_active_version_mismatch(self) -> None:
        from zoterorag.embeddings.indexer import prune_vector_index_versions

        with workspace_tmpdir("vector-prune-mismatch-") as tmpdir:
            ledger, _ = self._seed_store_and_ledger(tmpdir, active_version="stale-registration")
            try:
                with self.assertRaises(ValueError) as ctx:
                    prune_vector_index_versions(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="p_text",
                    )
                self.assertIn("mismatch", str(ctx.exception))
            finally:
                ledger.close()

    def test_prune_aborts_when_active_version_empty(self) -> None:
        from zoterorag.embeddings.indexer import prune_vector_index_versions

        with workspace_tmpdir("vector-prune-empty-") as tmpdir:
            ledger, _ = self._seed_store_and_ledger(tmpdir, active_chunks=0)
            try:
                with self.assertRaises(ValueError) as ctx:
                    prune_vector_index_versions(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="p_text",
                    )
                self.assertIn("empty", str(ctx.exception))
            finally:
                ledger.close()

    def test_prune_aborts_with_running_embedding_batch(self) -> None:
        from zoterorag.embeddings.indexer import prune_vector_index_versions

        with workspace_tmpdir("vector-prune-running-") as tmpdir:
            ledger, _ = self._seed_store_and_ledger(tmpdir)
            try:
                ledger.upsert_embedding_batch(
                    batch_hash="running-batch",
                    profile_name="p_text",
                    profile_hash="hash",
                    document_id="DOCX",
                    chunk_type="text",
                    chunk_count=1,
                    status="running",
                    provider="stub",
                    model="stub",
                    payload={},
                )
                with self.assertRaises(ValueError) as ctx:
                    prune_vector_index_versions(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="p_text",
                    )
                self.assertIn("running", str(ctx.exception))
            finally:
                ledger.close()

    def test_prune_recycles_stale_running_batch(self) -> None:
        """M4: a running batch stale beyond the threshold is flipped to failed
        and no longer blocks the prune."""
        from zoterorag.embeddings.indexer import prune_vector_index_versions

        with workspace_tmpdir("vector-prune-stale-") as tmpdir:
            ledger, _ = self._seed_store_and_ledger(tmpdir)
            try:
                ledger.upsert_embedding_batch(
                    batch_hash="stale-batch",
                    profile_name="p_text",
                    profile_hash="hash",
                    document_id="DOCX",
                    chunk_type="text",
                    chunk_count=1,
                    status="running",
                    provider="stub",
                    model="stub",
                    payload={},
                )
                stale_ts = (
                    datetime.now(timezone.utc) - timedelta(hours=25)
                ).isoformat(timespec="seconds")
                ledger.conn.execute(
                    "UPDATE embedding_batches SET updated_at = ? WHERE batch_hash = ?",
                    (stale_ts, "stale-batch"),
                )

                result = prune_vector_index_versions(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="p_text",
                )

                self.assertEqual(1, result["recycled_stale_batches"])
                self.assertEqual("failed", ledger.get_embedding_batch("stale-batch")["status"])
                self.assertEqual({"orphan": 1}, result["removed_versions"])
            finally:
                ledger.close()

    def test_prune_stale_recycling_can_be_disabled(self) -> None:
        """stale_batch_seconds=None keeps the old abort behavior verbatim."""
        from zoterorag.embeddings.indexer import prune_vector_index_versions

        with workspace_tmpdir("vector-prune-nostale-") as tmpdir:
            ledger, _ = self._seed_store_and_ledger(tmpdir)
            try:
                ledger.upsert_embedding_batch(
                    batch_hash="stale-batch",
                    profile_name="p_text",
                    profile_hash="hash",
                    document_id="DOCX",
                    chunk_type="text",
                    chunk_count=1,
                    status="running",
                    provider="stub",
                    model="stub",
                    payload={},
                )
                stale_ts = (
                    datetime.now(timezone.utc) - timedelta(hours=25)
                ).isoformat(timespec="seconds")
                ledger.conn.execute(
                    "UPDATE embedding_batches SET updated_at = ? WHERE batch_hash = ?",
                    (stale_ts, "stale-batch"),
                )
                with self.assertRaises(ValueError) as ctx:
                    prune_vector_index_versions(
                        ledger=ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="p_text",
                        stale_batch_seconds=None,
                    )
                self.assertIn("running", str(ctx.exception))
            finally:
                ledger.close()


class BatchPublishTests(unittest.TestCase):
    """Phase 3: stage/publish split — one publish per run per profile."""

    def _make_ledger(self, tmpdir: Path, *, backend: str = "sqlite-local") -> StateLedger:
        ledger = StateLedger(tmpdir / "state.sqlite")
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
                    backend=backend,
                )
            ]
        )
        return ledger

    def _stage(self, ledger: StateLedger, tmpdir: Path, document_id: str):
        return stage_indexed_document(
            ledger=ledger,
            vector_store_dir=tmpdir / "vectors",
            profile_name="stub_text",
            document_id=document_id,
        )

    def _open_store(self, tmpdir: Path, *, backend: str = "sqlite-local"):
        return open_vector_store(
            vector_path_for_profile(tmpdir / "vectors", "stub_text", backend=backend),
            profile_name="stub_text",
            dimension=8,
            backend=backend,
        )

    def test_three_documents_stage_then_publish_once(self) -> None:
        with workspace_tmpdir("batch-publish-") as tmpdir:
            ledger = self._make_ledger(tmpdir)
            try:
                # A pre-existing published document the run-level publish must
                # carry forward (so copied == its chunk count).
                seed_markdown_document(tmpdir, ledger, document_id="DOC_SEED", text="seed carried evidence")
                index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC_SEED",
                )
                for doc_id in ("DOC_A", "DOC_B", "DOC_C"):
                    seed_markdown_document(tmpdir, ledger, document_id=doc_id, text=f"{doc_id} unique evidence")

                staged = [self._stage(ledger, tmpdir, doc_id) for doc_id in ("DOC_A", "DOC_B", "DOC_C")]
                self.assertTrue(all(isinstance(item, StagedDocument) for item in staged))
                # Staging alone must not complete batches, register indexes, or
                # write "indexed" checkpoints.
                for item in staged:
                    self.assertEqual("running", ledger.get_embedding_batch(item.batch_hash)["status"])
                    self.assertNotEqual(
                        "indexed",
                        (ledger.get_checkpoint(item.document_id, "embed:stub_text") or {}).get("status"),
                    )
                # Staging must not have touched the registered index: it still
                # points at the seed document's version.
                registered_before = next(
                    i for i in ledger.list_vector_indexes() if i["profile_name"] == "stub_text"
                )
                store = self._open_store(tmpdir)
                try:
                    pre_active = store.active_version()
                    pre_counts = store.counts()
                finally:
                    store.close()
                self.assertEqual(1, pre_counts["chunks"])
                self.assertEqual(pre_active, registered_before["active_version"])

                results = publish_staged_documents(ledger=ledger, staged=staged)

                self.assertEqual(["DOC_A", "DOC_B", "DOC_C"], [r.document_id for r in results])
                self.assertTrue(all(not r.reused_existing for r in results))
                self.assertEqual([1, 1, 1], [r.indexed_chunks for r in results])

                run_hash = hashlib.sha256(
                    "\n".join(sorted(item.batch_hash for item in staged)).encode("utf-8")
                ).hexdigest()
                store = self._open_store(tmpdir)
                try:
                    # Exactly one new version, now active; the pre-run active
                    # version was superseded and removed by publish.
                    self.assertEqual([run_hash], store.list_versions())
                    self.assertEqual(run_hash, store.active_version())
                    self.assertNotEqual(pre_active, run_hash)
                    counts = store.counts()
                finally:
                    store.close()
                self.assertEqual({"documents": 4, "chunks": 4}, counts)

                for item in staged:
                    batch = ledger.get_embedding_batch(item.batch_hash)
                    self.assertEqual("completed", batch["status"])
                    self.assertEqual(run_hash, batch["payload"]["index_version"])
                    # Run-level carry-forward: the seed document's 1 chunk.
                    self.assertEqual(1, batch["payload"]["carried_forward_chunks"])
                    checkpoint = ledger.get_checkpoint(item.document_id, "embed:stub_text")
                    self.assertEqual("indexed", checkpoint["status"])
                    self.assertEqual(run_hash, checkpoint["payload"]["index_version"])
                index = next(i for i in ledger.list_vector_indexes() if i["profile_name"] == "stub_text")
                self.assertEqual(run_hash, index["active_version"])
                self.assertEqual(4, index["document_count"])
                self.assertEqual(4, index["chunk_count"])
            finally:
                ledger.close()

    def test_crash_after_publish_reruns_without_duplicate_rows(self) -> None:
        with workspace_tmpdir("batch-publish-crash-") as tmpdir:
            ledger = self._make_ledger(tmpdir)
            try:
                for doc_id in ("DOC_X", "DOC_Y"):
                    seed_markdown_document(tmpdir, ledger, document_id=doc_id, text=f"{doc_id} crash evidence")
                staged = [self._stage(ledger, tmpdir, doc_id) for doc_id in ("DOC_X", "DOC_Y")]

                # Simulate a crash in the bookkeeping window: publish committed,
                # the first "completed" batch upsert fails.
                original_upsert = ledger.upsert_embedding_batch
                crash_state = {"raised": False}

                def crashing_upsert(*args, **kwargs):
                    if kwargs.get("status") == "completed" and not crash_state["raised"]:
                        crash_state["raised"] = True
                        raise RuntimeError("simulated crash after publish")
                    return original_upsert(*args, **kwargs)

                with patch.object(ledger, "upsert_embedding_batch", side_effect=crashing_upsert):
                    with self.assertRaises(RuntimeError):
                        publish_staged_documents(ledger=ledger, staged=staged)

                store = self._open_store(tmpdir)
                try:
                    counts_after_crash = store.counts()
                    versions_after_crash = store.list_versions()
                finally:
                    store.close()
                self.assertEqual({"documents": 2, "chunks": 2}, counts_after_crash)
                # Batch rows stayed "running": the reuse fast path must miss.
                for item in staged:
                    self.assertEqual("running", ledger.get_embedding_batch(item.batch_hash)["status"])

                # Re-run stage + publish; the deterministic run hash makes the
                # second publish land on the same version.
                restaged = [self._stage(ledger, tmpdir, doc_id) for doc_id in ("DOC_X", "DOC_Y")]
                results = publish_staged_documents(ledger=ledger, staged=restaged)

                store = self._open_store(tmpdir)
                try:
                    self.assertEqual(counts_after_crash, store.counts())
                    self.assertEqual(versions_after_crash, store.list_versions())
                    for doc_id in ("DOC_X", "DOC_Y"):
                        self.assertEqual(1, store.document_counts(doc_id, modality="text")["chunks"])
                finally:
                    store.close()
                self.assertEqual(2, len(results))
                for item in restaged:
                    self.assertEqual("completed", ledger.get_embedding_batch(item.batch_hash)["status"])
            finally:
                ledger.close()

    def test_reuse_fast_path_returns_result_without_publish(self) -> None:
        with workspace_tmpdir("batch-publish-reuse-") as tmpdir:
            ledger = self._make_ledger(tmpdir)
            try:
                seed_markdown_document(tmpdir, ledger, document_id="DOC_R", text="reuse evidence")
                first = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC_R",
                )
                store = self._open_store(tmpdir)
                try:
                    version_before = store.active_version()
                finally:
                    store.close()

                with patch("zoterorag.embeddings.indexer.publish_staged_documents") as publish_spy:
                    outcome = self._stage(ledger, tmpdir, "DOC_R")

                self.assertIsInstance(outcome, IndexResult)
                self.assertTrue(outcome.reused_existing)
                self.assertEqual(first.embedding_batch_hash, outcome.embedding_batch_hash)
                publish_spy.assert_not_called()
                store = self._open_store(tmpdir)
                try:
                    self.assertEqual(version_before, store.active_version())
                finally:
                    store.close()
            finally:
                ledger.close()

    def test_publish_sweeps_leftover_staged_versions(self) -> None:
        with workspace_tmpdir("batch-publish-sweep-") as tmpdir:
            ledger = self._make_ledger(tmpdir)
            try:
                seed_markdown_document(tmpdir, ledger, document_id="DOC_S1", text="sweep base evidence")
                index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC_S1",
                )
                # A leftover staged version from a crashed earlier run.
                from zoterorag.index.local_vector import VectorRecord

                store = self._open_store(tmpdir)
                try:
                    store.upsert(
                        [
                            VectorRecord(
                                record_id="stub_text:orphan-chunk",
                                document_id="DOC_ORPHAN",
                                chunk_id="orphan-chunk",
                                vector=[0.0] * 8,
                                text="orphan",
                                modality="text",
                            )
                        ],
                        index_version="orphan-version",
                    )
                    self.assertIn("orphan-version", store.list_versions())
                finally:
                    store.close()

                seed_markdown_document(tmpdir, ledger, document_id="DOC_S2", text="sweep new evidence")
                staged = [self._stage(ledger, tmpdir, "DOC_S2")]
                publish_staged_documents(ledger=ledger, staged=staged)

                store = self._open_store(tmpdir)
                try:
                    versions = store.list_versions()
                    self.assertNotIn("orphan-version", versions)
                    self.assertEqual([store.active_version()], versions)
                    # The pre-existing document was carried forward.
                    self.assertEqual({"documents": 2, "chunks": 2}, store.counts())
                finally:
                    store.close()
            finally:
                ledger.close()

    def test_single_document_wrapper_matches_split_flow(self) -> None:
        with workspace_tmpdir("batch-publish-wrapper-") as tmpdir:
            ledger = self._make_ledger(tmpdir)
            try:
                seed_markdown_document(tmpdir, ledger, document_id="DOC_W", text="wrapper evidence")
                result = index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC_W",
                )
                run_hash = hashlib.sha256(result.embedding_batch_hash.encode("utf-8")).hexdigest()
                self.assertFalse(result.reused_existing)
                self.assertEqual(1, result.indexed_chunks)
                checkpoint = ledger.get_checkpoint("DOC_W", "embed:stub_text")
                self.assertEqual("indexed", checkpoint["status"])
                self.assertEqual(result.embedding_batch_hash, checkpoint["payload"]["embedding_batch_hash"])
                self.assertEqual(run_hash, checkpoint["payload"]["index_version"])
                index = next(i for i in ledger.list_vector_indexes() if i["profile_name"] == "stub_text")
                self.assertEqual(run_hash, index["active_version"])
                store = self._open_store(tmpdir)
                try:
                    self.assertEqual(run_hash, store.active_version())
                    self.assertEqual({"documents": 1, "chunks": 1}, store.counts())
                finally:
                    store.close()
                hits = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    query="wrapper evidence",
                    mode="text",
                )
                self.assertEqual("DOC_W", hits[0]["document_id"])
            finally:
                ledger.close()

    def test_publish_orchestration_against_lancedb_profile_with_fake_store(self) -> None:
        """The lancedb backend itself is covered by tests/test_lancedb_vector.py;
        here a minimal fake store verifies the indexer orchestration (grouping,
        leftover sweep, single stage_and_publish, ledger bookkeeping) when the
        profile backend is lancedb."""
        with workspace_tmpdir("batch-publish-lancedb-") as tmpdir:
            ledger = self._make_ledger(tmpdir, backend="lancedb")
            try:
                for doc_id in ("DOC_L1", "DOC_L2"):
                    seed_markdown_document(tmpdir, ledger, document_id=doc_id, text=f"{doc_id} lancedb evidence")
                staged = [self._stage(ledger, tmpdir, doc_id) for doc_id in ("DOC_L1", "DOC_L2")]
                self.assertTrue(all(item.backend == "lancedb" for item in staged))

                class FakeStore:
                    def __init__(self) -> None:
                        self.active = "legacy"
                        self.deleted: list[str] = []
                        self.published_records = None
                        self.excluded: list[str] = []
                        self.closed = False

                    def list_versions(self):
                        return ["orphan-version"]

                    def version_is_active(self, version):
                        return version == self.active

                    def delete_version(self, version):
                        self.deleted.append(version)
                        return 1

                    def stage_and_publish(self, records, *, index_version, exclude_document_ids=()):
                        self.published_records = list(records)
                        self.excluded = list(exclude_document_ids)
                        self.active = index_version
                        return 0, len(self.published_records), {"documents": 2, "chunks": len(self.published_records)}

                    def close(self):
                        self.closed = True

                fake = FakeStore()
                with patch("zoterorag.embeddings.indexer.open_vector_store", return_value=fake):
                    results = publish_staged_documents(ledger=ledger, staged=staged)

                run_hash = hashlib.sha256(
                    "\n".join(sorted(item.batch_hash for item in staged)).encode("utf-8")
                ).hexdigest()
                self.assertEqual(["orphan-version"], fake.deleted)
                self.assertEqual(run_hash, fake.active)
                self.assertEqual(2, len(fake.published_records))
                self.assertEqual({"DOC_L1", "DOC_L2"}, set(fake.excluded))
                self.assertTrue(fake.closed)
                self.assertEqual(["DOC_L1", "DOC_L2"], [r.document_id for r in results])
                index = next(i for i in ledger.list_vector_indexes() if i["profile_name"] == "stub_text")
                self.assertEqual("lancedb", index["backend"])
                self.assertEqual(run_hash, index["active_version"])
                for item in staged:
                    self.assertEqual(run_hash, ledger.get_embedding_batch(item.batch_hash)["payload"]["index_version"])
            finally:
                ledger.close()
