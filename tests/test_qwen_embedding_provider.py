from __future__ import annotations

import base64
import json
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

    def test_retries_on_transient_http_status(self) -> None:
        client = FakeQwenClient(
            dimension=2560,
            responses=[
                FakeQwenResponse(dimension=2560, status_code=503),
                FakeQwenResponse(dimension=2560, status_code=429),
                FakeQwenResponse(dimension=2560, status_code=200),
            ],
        )
        provider = Qwen3VLEmbeddingProvider(
            api_key="sk-test-secret",
            client=client,
            retry_initial_seconds=0.0,
            retry_jitter=0.0,
        )

        vectors = provider.embed([EmbeddingInput(input_id="q1", text="hello", role="query")])

        self.assertEqual(3, len(client.posts))
        self.assertEqual("q1", vectors[0].input_id)
        self.assertEqual(2560, len(vectors[0].vector))

    def test_status_code_checked_before_json_parsing(self) -> None:
        client = FakeQwenClient(
            dimension=2560,
            responses=[
                FakeQwenResponse(
                    dimension=2560,
                    status_code=400,
                    text="<html>bad request</html>",
                    json_error=True,
                ),
            ],
        )
        provider = Qwen3VLEmbeddingProvider(api_key="sk-test-secret", client=client)

        with self.assertRaises(QwenEmbeddingError) as ctx:
            provider.embed([EmbeddingInput(input_id="q1", text="hello", role="query")])

        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertNotIn("valid JSON", str(ctx.exception))

    def test_retries_on_timeout_exception(self) -> None:
        class TimeoutOnFirstCall:
            def __init__(self, client: FakeQwenClient) -> None:
                self.client = client
                self.calls = 0

            def post(self, url, *, headers=None, json=None, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("connection timed out")
                return self.client.post(url, headers=headers, json=json, timeout=timeout)

        real_client = FakeQwenClient(dimension=2560)
        wrapper = TimeoutOnFirstCall(real_client)
        provider = Qwen3VLEmbeddingProvider(
            api_key="sk-test-secret",
            client=wrapper,
            retry_initial_seconds=0.0,
            retry_jitter=0.0,
        )

        vectors = provider.embed([EmbeddingInput(input_id="q1", text="hello", role="query")])

        self.assertEqual(2, wrapper.calls)
        self.assertEqual(1, len(real_client.posts))
        self.assertEqual(2560, len(vectors[0].vector))

    def test_missing_image_file_raises_qwen_error(self) -> None:
        provider = Qwen3VLEmbeddingProvider(api_key="sk-test-secret", client=FakeQwenClient(dimension=2560))

        with self.assertRaises(QwenEmbeddingError) as ctx:
            provider.embed(
                [EmbeddingInput(input_id="img1", text="", image_path="/nonexistent/path/figure.png")]
            )

        self.assertIn("not found", str(ctx.exception).lower())

    def test_image_path_outside_allowed_roots_is_rejected(self) -> None:
        with workspace_tmpdir("qwen-allowed-roots-") as tmpdir:
            allowed = tmpdir / "allowed"
            outside = tmpdir / "outside"
            allowed.mkdir()
            outside.mkdir()
            (allowed / "ok.png").write_bytes(b"ok image")
            (outside / "secret.png").write_bytes(b"secret image")

            provider = Qwen3VLEmbeddingProvider(api_key="sk-test-secret", client=FakeQwenClient(dimension=2560))

            allowed_input = EmbeddingInput(input_id="ok", text="", image_path=str(allowed / "ok.png"))
            provider.embed([allowed_input])

            outside_input = EmbeddingInput(input_id="leak", text="", image_path=str(outside / "secret.png"))
            with self.assertRaises(QwenEmbeddingError) as ctx:
                from zoterorag.embeddings.qwen import image_data_uri_for_input

                image_data_uri_for_input(
                    outside_input,
                    max_image_bytes=provider.max_image_bytes,
                    allowed_roots=[allowed],
                )
            self.assertIn("outside allowed roots", str(ctx.exception).lower())


class FakeQwenResponse:
    def __init__(
        self,
        *,
        dimension: int,
        status_code: int = 200,
        text: str = "{}",
        json_error: bool = False,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.dimension = dimension
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise json.JSONDecodeError("not valid JSON", "", 0)
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
    def __init__(self, *, dimension: int, responses: list[FakeQwenResponse] | None = None) -> None:
        self.dimension = dimension
        self._responses = list(responses) if responses is not None else None
        self.posts: list[dict] = []

    def post(self, url, *, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers or {}, "json": json or {}, "timeout": timeout})
        if self._responses is not None:
            return self._responses.pop(0)
        return FakeQwenResponse(dimension=self.dimension)


if __name__ == "__main__":
    unittest.main()
