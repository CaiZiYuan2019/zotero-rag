from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

from tests._support import workspace_tmpdir
from zoterorag.config import AppConfig, EmbeddingProfile, PathsConfig
from zoterorag.db import StateLedger
from zoterorag.embeddings import StubEmbeddingProvider, index_normalized_document
from zoterorag.mcp import (
    McpToolContext,
    zotero_rag_get_document,
    zotero_rag_list_models,
    zotero_rag_metadata_search,
    zotero_rag_search_multimodal,
    zotero_rag_search_text,
    zotero_rag_status,
)
from zoterorag.normalize import normalize_markdown_document


class McpToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Exercise Qwen request construction while keeping all queries offline.
        client_patch = patch("zoterorag.embeddings.qwen._load_requests_client")
        self.http = client_patch.start().return_value
        self.addCleanup(client_patch.stop)
        response = self.http.post.return_value
        response.status_code = 200
        response.json.return_value = {"output": {"embeddings": [{"embedding": [1.0] * 256}]}}

    def test_mcp_text_tools_are_safe_for_plain_text_llms(self) -> None:
        with workspace_tmpdir("mcp-tools-") as tmpdir:
            context = build_context(tmpdir)
            try:
                seed_library(context)

                metadata = zotero_rag_metadata_search(context, query="Figure Paper")
                text = zotero_rag_search_text(context, query="figure evidence", include_vector=True)
                rerank_rejected = zotero_rag_search_text(context, query="figure evidence", rerank=True)
                document = zotero_rag_get_document(context, document_id="DOC1", include_chunks=True)

                serialized = json.dumps(
                    {"metadata": metadata, "text": text, "document": document},
                    ensure_ascii=False,
                )
                self.assertNotIn("images/img001.png", serialized)
                self.assertNotIn("base64", serialized)
                self.assertEqual("llm_text", text["consumer"])
                self.assertEqual("none", text["image_return"])
                self.assertFalse(rerank_rejected["results"])
                self.assertIn("rerank is reserved", rerank_rejected["warnings"][0])
                self.assertTrue(text["results"])
                self.http.post.assert_called_once()
                request = self.http.post.call_args.kwargs
                self.assertEqual("Bearer fixture-key", request["headers"]["Authorization"])
                self.assertEqual("text query instruction", request["json"]["parameters"]["instruct"])
                self.assertEqual([{"text": "figure evidence"}], request["json"]["input"]["contents"])
                self.assertTrue(document["document"]["chunks"])
            finally:
                context.ledger.close()

    def test_mcp_multimodal_requires_explicit_image_return(self) -> None:
        with workspace_tmpdir("mcp-tools-") as tmpdir:
            context = build_context(tmpdir)
            try:
                seed_library(context)

                text_default = zotero_rag_search_multimodal(context, query_text="important figure")
                multimodal = zotero_rag_search_multimodal(
                    context,
                    query_text="important figure",
                    consumer="llm_multimodal",
                    image_return="file_ref",
                )
                multimodal_base64 = zotero_rag_search_multimodal(
                    context,
                    query_text="important figure",
                    consumer="llm_multimodal",
                    profile_name="fixture_mm",
                    query_image={"type": "base64", "value": "aW1n", "mime_type": "image/png"},
                    image_return="base64",
                    max_image_bytes=1024,
                )

                payload = self.http.post.call_args.kwargs["json"]
                self.assertEqual("multimodal query instruction", payload["parameters"]["instruct"])
                self.assertIn({"image": "data:image/png;base64,aW1n"}, payload["input"]["contents"])
                self.assertEqual("llm_text", text_default["consumer"])
                self.assertEqual("none", text_default["image_return"])
                text_default_json = json.dumps(text_default, ensure_ascii=False)
                self.assertNotIn("image_path", text_default_json)
                self.assertNotIn("file_ref", text_default_json)
                self.assertNotIn("base64", text_default_json)
                self.assertEqual("llm_multimodal", multimodal["consumer"])
                self.assertEqual("file_ref", multimodal["image_return"])
                self.assertIn("images", multimodal["results"][0])
                self.assertEqual("images/img001.png", multimodal["results"][0]["images"][0]["file_ref"])
                self.assertEqual(base64.b64encode(b"fake-image").decode("ascii"), multimodal_base64["results"][0]["images"][0]["base64"])
            finally:
                context.ledger.close()

    def test_qwen_failure_preserves_requested_nonvector_results(self) -> None:
        with workspace_tmpdir("mcp-tools-") as tmpdir:
            context = build_context(tmpdir)
            try:
                seed_library(context)
                self.http.post.return_value.status_code = 401
                self.http.post.return_value.json.return_value = {"message": "invalid credential"}
                result = zotero_rag_search_text(context, query="Figure Paper", include_metadata=True)
                self.assertEqual(["metadata"], [r["metadata"]["retrieval_source"] for r in result["results"]])
                self.assertIn("HTTP 401", result["warnings"][0])
                self.http.post.reset_mock()
                result = zotero_rag_search_text(context, query="Figure Paper", include_metadata=True, include_vector=False)
                self.assertEqual([], result["warnings"])
                self.http.post.assert_not_called()
            finally:
                context.ledger.close()

    def test_mcp_status_and_models_expose_control_state(self) -> None:
        with workspace_tmpdir("mcp-tools-") as tmpdir:
            context = build_context(tmpdir)
            try:
                seed_library(context)

                status = zotero_rag_status(context)
                models = zotero_rag_list_models(context)

                self.assertIn("library", status)
                self.assertIn("build", status)
                self.assertIn("profiles", status)
                self.assertEqual(2, len(models["models"]))
                self.assertEqual(2, len(models["vector_indexes"]))
            finally:
                context.ledger.close()


def build_context(tmpdir) -> McpToolContext:
    config = AppConfig(
        paths=PathsConfig(
            zotero_db=tmpdir / "zotero.sqlite",
            zotero_storage=tmpdir / "storage",
            data_dir=tmpdir / "data",
        ),
        embedding_profiles=(
            EmbeddingProfile(
                name="fixture_text",
                provider="dashscope",
                model="qwen3-vl-embedding",
                instruction_template="text query instruction",
                dimension=256,
                modality="text",
                enabled=True,
                default_for_text=True,
                backend="sqlite-local",
            ),
            EmbeddingProfile(
                name="fixture_mm",
                provider="dashscope",
                model="qwen3-vl-embedding",
                instruction_template="multimodal query instruction",
                dimension=256,
                modality="multimodal",
                enabled=True,
                default_for_multimodal=True,
                backend="sqlite-local",
            ),
        ),
    )
    config.ensure_runtime_dirs()
    ledger = StateLedger(config.paths.state_db)
    ledger.upsert_embedding_profiles(config.embedding_profiles)
    (tmpdir / "pyproject.toml").touch()
    (tmpdir / ".env").write_text("QWEN_API_KEY=fixture-key\n", encoding="utf-8")
    config_path = tmpdir / "config.toml"
    config_path.touch()
    return McpToolContext(config=config, ledger=ledger, config_path=config_path)


def seed_library(context: McpToolContext) -> None:
    ledger = context.ledger
    ledger.upsert_attachments(
        [
            {
                "attachment_key": "ATT1",
                "parent_key": "PARENT1",
                "content_type": "application/pdf",
                "relative_path": "storage:paper.pdf",
                "title": "Figure Paper",
                "abstract": "Abstract with figure evidence.",
                "date": "2025",
                "url": "https://example.test/paper",
                "classification": "included_auto",
                "source_quality": "primary_candidate",
                "reasons": [],
                "file_path": "C:/Zotero/storage/paper.pdf",
                "file_exists": True,
                "file_size": 100,
                "file_mtime": 1.0,
                "metadata": {},
            }
        ]
    )
    source_dir = context.config.paths.data_dir / "mineru-fixture"
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "hash.png").write_bytes(b"fake-image")
    markdown = source_dir / "full.md"
    markdown.write_text(
        "# Figure Paper\n\n"
        "Textual figure evidence.\n\n"
        "![important figure](images/hash.png)\n",
        encoding="utf-8",
    )
    normalized = normalize_markdown_document(
        source_markdown=markdown,
        output_root=context.config.paths.normalized_dir,
        document_id="DOC1",
        attachment_key="ATT1",
    )
    ledger.upsert_normalized_artifact(normalized.ledger_artifact())
    ledger.replace_document_chunks(normalized.document_id, normalized.chunks)
    index_normalized_document(
        ledger=ledger,
        vector_store_dir=context.config.paths.vector_store_dir,
        profile_name="fixture_text",
        document_id="DOC1",
        provider=StubEmbeddingProvider(dimension=256),
    )
    index_normalized_document(
        ledger=ledger,
        vector_store_dir=context.config.paths.vector_store_dir,
        profile_name="fixture_mm",
        document_id="DOC1",
        provider=StubEmbeddingProvider(dimension=256),
    )


if __name__ == "__main__":
    unittest.main()
