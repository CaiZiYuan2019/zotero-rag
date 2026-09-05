from __future__ import annotations

import base64
import json
import threading
import time
import unittest
from unittest.mock import patch

from tests._support import workspace_tmpdir
from zoterorag.embeddings import EmbeddingInput, Qwen3VLEmbeddingProvider, QwenEmbeddingError
from zoterorag.embeddings.qwen import _TokenBucket
from zoterorag.providers import _validate_endpoint_url, build_qwen_embedding_provider


class EndpointValidationTests(unittest.TestCase):
    def test_accepts_allowed_dashscope_https_url(self) -> None:
        self.assertEqual(
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
            _validate_endpoint_url(
                "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
                ("dashscope.aliyuncs.com",),
            ),
        )

    def test_rejects_http_url(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_endpoint_url("http://dashscope.aliyuncs.com/api", ("dashscope.aliyuncs.com",))
        self.assertIn("https", str(ctx.exception))

    def test_rejects_private_ip(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_endpoint_url("https://192.168.1.1/api", ("dashscope.aliyuncs.com",))
        self.assertIn("IP address", str(ctx.exception))

    def test_rejects_wrong_hostname(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_endpoint_url("https://evil.example.test/api", ("dashscope.aliyuncs.com",))
        self.assertIn("hostname", str(ctx.exception))

    def test_build_qwen_provider_rejects_bad_env_url(self) -> None:
        with workspace_tmpdir("qwen-build-") as tmpdir:
            env_path = tmpdir / ".env"
            env_path.write_text(
                "DASHSCOPE_API_KEY=sk-test\nDASHSCOPE_MULTIMODAL_EMBEDDING_URL=https://evil.example.test\n",
                encoding="utf-8",
            )
            profile = {"model": "qwen3-vl-embedding", "dimension": 2560}
            with self.assertRaises(ValueError):
                build_qwen_embedding_provider(profile, env_path)


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


class ThreadSafeSlowClient:
    """Fake client that is safe under concurrent embed: tracks the in-flight
    call count and can simulate a fixed latency / failure status."""

    def __init__(self, *, dimension: int, delay_seconds: float = 0.0, status_code: int = 200) -> None:
        self.dimension = dimension
        self.delay_seconds = delay_seconds
        self.status_code = status_code
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.post_count = 0

    def post(self, url, *, headers=None, json=None, timeout=None):
        with self._lock:
            self._in_flight += 1
            self.post_count += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            return FakeQwenResponse(dimension=self.dimension, status_code=self.status_code, text="boom")
        finally:
            with self._lock:
                self._in_flight -= 1


class ConcurrentEmbeddingTests(unittest.TestCase):
    """Phase 1: embed_workers concurrency, ordering, and fail-fast semantics."""

    def test_concurrent_embed_preserves_order_and_overlaps(self) -> None:
        client = ThreadSafeSlowClient(dimension=2560, delay_seconds=0.05)
        provider = Qwen3VLEmbeddingProvider(api_key="sk-test", client=client, embed_workers=4)
        inputs = [
            EmbeddingInput(input_id=f"doc{i}", text=f"text {i}", role="document") for i in range(8)
        ]

        vectors = provider.embed(inputs)

        self.assertEqual([item.input_id for item in inputs], [v.input_id for v in vectors])
        self.assertGreater(client.max_in_flight, 1)
        self.assertEqual(8, client.post_count)

    def test_default_workers_stays_serial(self) -> None:
        """embed_workers=1 (the default) must keep the original serial path."""
        client = ThreadSafeSlowClient(dimension=2560, delay_seconds=0.02)
        provider = Qwen3VLEmbeddingProvider(api_key="sk-test", client=client)
        inputs = [
            EmbeddingInput(input_id=f"doc{i}", text=f"text {i}", role="document") for i in range(4)
        ]

        vectors = provider.embed(inputs)

        self.assertEqual([item.input_id for item in inputs], [v.input_id for v in vectors])
        self.assertEqual(1, client.max_in_flight)
        self.assertEqual(4, client.post_count)

    def test_failure_cancels_queued_inputs(self) -> None:
        """A systemic failure must not burn through every remaining input with
        retries: the first error cancels queued (not yet started) tasks."""
        # 400 is not in the transient set, so every started task fails after
        # exactly one post.
        client = ThreadSafeSlowClient(dimension=2560, delay_seconds=0.01, status_code=400)
        provider = Qwen3VLEmbeddingProvider(api_key="sk-test", client=client, embed_workers=2)
        inputs = [
            EmbeddingInput(input_id=f"doc{i}", text=f"text {i}", role="document") for i in range(50)
        ]

        with self.assertRaises(QwenEmbeddingError):
            provider.embed(inputs)

        self.assertLess(client.post_count, 50)

    def test_embed_workers_validation(self) -> None:
        for bad in (0, -1, True, 1.5, "4"):
            with self.assertRaises(ValueError):
                Qwen3VLEmbeddingProvider(api_key="sk-test", client=FakeQwenClient(dimension=2560), embed_workers=bad)
        with self.assertRaises(ValueError):
            Qwen3VLEmbeddingProvider(
                api_key="sk-test",
                client=FakeQwenClient(dimension=2560),
                rate_limit_per_second=-1.0,
            )
        with self.assertRaises(ValueError):
            Qwen3VLEmbeddingProvider(
                api_key="sk-test",
                client=FakeQwenClient(dimension=2560),
                rate_limit_per_second=True,
            )


class RateLimitTests(unittest.TestCase):
    """Phase 1: token-bucket pacing, billed per HTTP attempt (retries too)."""

    def _fake_clock(self):
        clock = [0.0]
        sleeps: list[float] = []

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock[0] += seconds

        return clock, sleeps, fake_sleep

    def test_token_bucket_capacity_and_refill(self) -> None:
        clock, sleeps, fake_sleep = self._fake_clock()
        with patch("zoterorag.embeddings.qwen.time") as mock_time:
            mock_time.monotonic.side_effect = lambda: clock[0]
            mock_time.sleep.side_effect = fake_sleep
            bucket = _TokenBucket(rate_per_second=2.0, capacity=2.0)
            bucket.acquire()  # instant (token 1 of 2)
            bucket.acquire()  # instant (token 2 of 2)
            bucket.acquire()  # waits 0.5s
            bucket.acquire()  # waits 0.5s again
        self.assertEqual([0.5, 0.5], sleeps)

    def test_rate_limit_paces_serial_requests(self) -> None:
        """workers=1 + rps=2: capacity is 1, so after the first request every
        attempt waits 0.5s — the serial path is paced too."""
        clock, sleeps, fake_sleep = self._fake_clock()
        client = FakeQwenClient(dimension=2560)
        with patch("zoterorag.embeddings.qwen.time") as mock_time:
            mock_time.monotonic.side_effect = lambda: clock[0]
            mock_time.sleep.side_effect = fake_sleep
            # Construct inside the patch so the bucket clock is the fake one.
            provider = Qwen3VLEmbeddingProvider(
                api_key="sk-test",
                client=client,
                embed_workers=1,
                rate_limit_per_second=2.0,
            )
            inputs = [
                EmbeddingInput(input_id=f"doc{i}", text=f"text {i}", role="document") for i in range(5)
            ]
            provider.embed(inputs)
        self.assertEqual(5, len(client.posts))
        self.assertEqual(4, len(sleeps))
        self.assertAlmostEqual(2.0, sum(sleeps))

    def test_retries_count_against_the_rate_limit(self) -> None:
        """Two 500s then a 200 = 3 HTTP attempts; each must acquire a token."""
        clock, sleeps, fake_sleep = self._fake_clock()
        responses = [
            FakeQwenResponse(dimension=2560, status_code=500),
            FakeQwenResponse(dimension=2560, status_code=500),
            FakeQwenResponse(dimension=2560, status_code=200),
        ]
        client = FakeQwenClient(dimension=2560, responses=responses)
        with patch("zoterorag.embeddings.qwen.time") as mock_time:
            mock_time.monotonic.side_effect = lambda: clock[0]
            mock_time.sleep.side_effect = fake_sleep
            provider = Qwen3VLEmbeddingProvider(
                api_key="sk-test",
                client=client,
                retry_initial_seconds=0.0,  # keep retry backoff out of the accounting
                rate_limit_per_second=1.0,
            )
            provider.embed([EmbeddingInput(input_id="doc0", text="text", role="document")])
        # 3 attempts at rps=1 with capacity 1: attempt 1 instant, 2 and 3
        # each wait 1.0s; the two zero-length backoff sleeps ride along.
        self.assertEqual(3, len(client.posts))
        positive_sleeps = [s for s in sleeps if s > 0]
        self.assertEqual([1.0, 1.0], positive_sleeps)

    def test_build_qwen_provider_passes_concurrency_settings(self) -> None:
        with workspace_tmpdir("qwen-build-") as tmpdir:
            env_path = tmpdir / ".env"
            env_path.write_text("DASHSCOPE_API_KEY=sk-test\n", encoding="utf-8")
            profile = {
                "model": "qwen3-vl-embedding",
                "dimension": 2560,
                "profile": {"embed_workers": 4, "rate_limit": {"requests_per_second": 10}},
            }
            provider = build_qwen_embedding_provider(profile, env_path)
            self.assertEqual(4, provider.embed_workers)
            self.assertIsNotNone(provider._rate_bucket)
            self.assertEqual(10.0, provider._rate_bucket.rate)

    def test_build_qwen_provider_defaults_to_serial_unlimited(self) -> None:
        with workspace_tmpdir("qwen-build-") as tmpdir:
            env_path = tmpdir / ".env"
            env_path.write_text("DASHSCOPE_API_KEY=sk-test\n", encoding="utf-8")
            provider = build_qwen_embedding_provider(
                {"model": "qwen3-vl-embedding", "dimension": 2560}, env_path
            )
            self.assertEqual(1, provider.embed_workers)
            self.assertIsNone(provider._rate_bucket)


if __name__ == "__main__":
    unittest.main()
