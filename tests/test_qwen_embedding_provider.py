from __future__ import annotations

import base64
import unittest

from tests._support import workspace_tmpdir
from zoterorag.embeddings import EmbeddingInput, Qwen3VLEmbeddingProvider, QwenEmbeddingError


class QwenEmbeddingProviderTests(unittest.TestCase):
    def test_text_query_payload_uses_query_instruction_and_dimension(self) -> None:
        client = FakeQwenClient(dimension=2560)
        provider = Qwen3VLEmbeddingProvider(api_key="sk-test-secret", client=client)

        vectors = provider.embed([EmbeddingInput(input_id="q1", text="figure reproducibility", role="query")])

        self.assertEqual("q1", vectors[0].input_id)
        self.assertEqual(2560, len(vectors[0].vector))
        request = client.posts[0]
        self.assertEqual("Bearer sk-test-secret", request["headers"]["Authorization"])
        self.assertEqual("qwen3-vl-embedding", request["json"]["model"])
        self.assertEqual(2560, request["json"]["parameters"]["dimension"])
        self.assertIn("query", request["json"]["parameters"]["instruct"].lower())
        self.assertEqual([{"text": "figure reproducibility"}], request["json"]["input"]["contents"])
        self.assertNotIn("enable_fusion", request["json"]["parameters"])

    def test_multimodal_payload_uses_image_data_uri_and_fusion(self) -> None:
        with workspace_tmpdir("qwen-provider-") as tmpdir:
            image_path = tmpdir / "figure.png"
            image_path.write_bytes(b"fake image bytes")
            client = FakeQwenClient(dimension=2560)
            provider = Qwen3VLEmbeddingProvider(api_key="sk-test-secret", client=client)

            provider.embed(
                [
                    EmbeddingInput(
                        input_id="img1",
                        text="Figure 1 caption",
                        image_path=str(image_path),
                        role="document",
                    )
                ]
            )

            payload = client.posts[0]["json"]
            contents = payload["input"]["contents"]
            self.assertEqual({"text": "Figure 1 caption"}, contents[0])
            self.assertTrue(contents[1]["image"].startswith("data:image/png;base64,"))
            self.assertEqual(
                base64.b64encode(b"fake image bytes").decode("ascii"),
                contents[1]["image"].split(",", 1)[1],
            )
            self.assertTrue(payload["parameters"]["enable_fusion"])
            self.assertIn("scientific paper", payload["parameters"]["instruct"])

    def test_rejects_too_large_image_before_network_request(self) -> None:
        with workspace_tmpdir("qwen-provider-") as tmpdir:
            image_path = tmpdir / "big.png"
            image_path.write_bytes(b"x" * 11)
            client = FakeQwenClient(dimension=2560)
            provider = Qwen3VLEmbeddingProvider(api_key="sk-test-secret", client=client, max_image_bytes=10)

            with self.assertRaises(QwenEmbeddingError):
                provider.embed([EmbeddingInput(input_id="img1", text="", image_path=str(image_path))])
            self.assertEqual([], client.posts)

    def test_rejects_dimension_mismatch(self) -> None:
        client = FakeQwenClient(dimension=3)
        provider = Qwen3VLEmbeddingProvider(api_key="sk-test-secret", client=client)

        with self.assertRaises(QwenEmbeddingError):
            provider.embed([EmbeddingInput(input_id="bad", text="hello")])


class FakeQwenResponse:
    def __init__(self, *, dimension: int, status_code: int = 200) -> None:
        self.status_code = status_code
        self.text = "{}"
        self.dimension = dimension

    def json(self):
        return {
            "output": {
                "embeddings": [
                    {
                        "embedding": [0.125] * self.dimension,
                    }
                ]
            }
        }


class FakeQwenClient:
    def __init__(self, *, dimension: int) -> None:
        self.dimension = dimension
        self.posts: list[dict] = []

    def post(self, url, *, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers or {}, "json": json or {}, "timeout": timeout})
        return FakeQwenResponse(dimension=self.dimension)


if __name__ == "__main__":
    unittest.main()
