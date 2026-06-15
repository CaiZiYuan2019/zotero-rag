from __future__ import annotations

import json
import unittest

from tests._support import workspace_tmpdir
from zoterorag.providers import (
    build_mineru_provider,
    build_qwen_embedding_provider,
    mineru_urls_from_env,
    provider_readiness,
)


class ProviderRuntimeTests(unittest.TestCase):
    def test_provider_readiness_reports_configuration_without_secrets(self) -> None:
        with workspace_tmpdir("providers-") as tmpdir:
            env_path = tmpdir / ".env"
            env_path.write_text(
                "MINERU_KEY=mineru-secret\n"
                "BAILIAN_KEY=qwen-secret\n"
                "BAILIAN_URL=https://dashscope.example.test/embed\n",
                encoding="utf-8",
            )

            status = provider_readiness(env_path)
            serialized = json.dumps(status, ensure_ascii=False)

            self.assertTrue(status["mineru"]["configured"])
            self.assertEqual(1, status["mineru"]["key_count"])
            self.assertEqual("mineru_1", status["mineru"]["keys"][0]["alias"])
            self.assertTrue(status["qwen3vl_embedding"]["configured"])
            self.assertEqual("BAILIAN_KEY", status["qwen3vl_embedding"]["key_env_name"])
            # BAILIAN_URL is intentionally excluded from embedding endpoint
            # resolution (it points at the chat API, not embeddings).
            self.assertIsNone(status["qwen3vl_embedding"]["endpoint_env_name"])
            self.assertNotIn("mineru-secret", serialized)
            self.assertNotIn("qwen-secret", serialized)

    def test_build_qwen_provider_uses_profile_and_env_without_network(self) -> None:
        with workspace_tmpdir("providers-") as tmpdir:
            env_path = tmpdir / ".env"
            env_path.write_text(
                "BAILIAN_KEY=qwen-secret\n"
                "DASHSCOPE_MULTIMODAL_EMBEDDING_URL=https://dashscope.example.test/embed\n",
                encoding="utf-8",
            )

            provider = build_qwen_embedding_provider(
                {
                    "name": "qwen_text",
                    "provider": "dashscope",
                    "model": "qwen3-vl-embedding",
                    "dimension": 2560,
                    "modality": "text",
                    "instruction_template": "query instruction",
                    "profile": {"document_instruction": "document instruction"},
                },
                env_path,
                client=NoopClient(),
            )

            self.assertEqual("qwen3-vl-embedding", provider.model)
            self.assertEqual(2560, provider.dimension)
            self.assertEqual("https://dashscope.example.test/embed", provider.endpoint)
            self.assertEqual("query instruction", provider.query_instruction)
            self.assertEqual("document instruction", provider.document_instruction)

    def test_mineru_url_base_is_used_as_prefix_directly(self) -> None:
        # MINERU_URL is used as-is; caller is responsible for including
        # any path prefix like /api/v4/extract/task in the env var.
        urls = mineru_urls_from_env({"MINERU_URL": "https://mineru.example.test/api/v4/extract/task"})

        self.assertEqual(
            "https://mineru.example.test/api/v4/extract/task/file-urls/batch",
            urls["apply_upload_url"],
        )
        self.assertEqual(
            "https://mineru.example.test/api/v4/extract/task/extract-results/batch/{batch_id}",
            urls["batch_result_url"],
        )

    def test_build_mineru_provider_uses_env_endpoints_without_key_value(self) -> None:
        with workspace_tmpdir("providers-") as tmpdir:
            env_path = tmpdir / ".env"
            env_path.write_text("MINERU_URL=https://mineru.example.test/api/v4\n", encoding="utf-8")

            provider = build_mineru_provider(env_path, client=NoopClient())

            self.assertEqual("https://mineru.example.test/api/v4/file-urls/batch", provider.apply_upload_url)
            self.assertEqual(
                "https://mineru.example.test/api/v4/extract-results/batch/{batch_id}",
                provider.batch_result_url,
            )


class NoopClient:
    pass


if __name__ == "__main__":
    unittest.main()
