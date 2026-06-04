from __future__ import annotations

import json
import unittest

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.extractors import (
    ApiKeyRef,
    ExtractionManager,
    ExtractionRequest,
    ExtractorKeyPool,
    recommended_mineru_timeout_seconds,
)
from zoterorag.extractors.base import ExtractArtifact, ExtractJobState
from zoterorag.extractors.cache import extractor_cache_key, stable_options_hash


class ExtractionManagerTests(unittest.TestCase):
    def test_cache_key_is_stable_and_includes_provider_profile(self) -> None:
        options_hash = stable_options_hash({"model_version": "vlm", "enable_formula": True})
        first = extractor_cache_key(
            pdf_sha256="a" * 64,
            selected_pages="1-3",
            extractor_name="mineru",
            extractor_version="v4",
            options_hash=options_hash,
        )
        second = extractor_cache_key(
            pdf_sha256="a" * 64,
            selected_pages=[1, 2, 3],
            extractor_name="mineru",
            extractor_version="v4",
            options_hash=options_hash,
        )
        changed_model = extractor_cache_key(
            pdf_sha256="a" * 64,
            selected_pages="1-3",
            extractor_name="mineru",
            extractor_version="v5",
            options_hash=options_hash,
        )

        self.assertEqual(first, first)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, changed_model)

    def test_recommended_timeout_uses_conservative_page_formula(self) -> None:
        self.assertEqual(36, recommended_mineru_timeout_seconds(1))
        self.assertEqual(630, recommended_mineru_timeout_seconds(100))
        with self.assertRaises(ValueError):
            recommended_mineru_timeout_seconds(0)

    def test_stub_extraction_persists_job_and_reuses_cache(self) -> None:
        with workspace_tmpdir("extract-manager-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake test body")
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(ledger=ledger, cache_dir=tmpdir / "extract_cache")
                request = ExtractionRequest(
                    input_file=source,
                    attachment_key="ATTACH1",
                    selected_pages="1",
                    selected_page_count=1,
                    options={"model_version": "vlm"},
                )

                first = manager.ensure_extraction(request)
                second = manager.ensure_extraction(request)

                self.assertFalse(first.cache_hit)
                self.assertTrue(second.cache_hit)
                self.assertEqual(first.job["cache_key"], second.job["cache_key"])
                self.assertEqual("downloaded", second.job["state"])
                self.assertEqual("stub_stub", second.job["api_key_alias"])
                self.assertTrue((tmpdir / "extract_cache" / second.job["cache_key"] / "manifest.json").is_file())

                summary = ledger.status_summary()
                self.assertEqual({"downloaded": 1}, summary["extract_jobs"])
            finally:
                ledger.close()

    def test_env_key_pool_uses_aliases_without_leaking_secrets(self) -> None:
        with workspace_tmpdir("extract-key-pool-") as tmpdir:
            env_path = tmpdir / ".env"
            env_path.write_text(
                "MINERU_KEY=sk-test-secret-0001\n"
                "MINERU_API_KEY_SECOND=sk-test-secret-0002\n"
                "BAILIAN_KEY=not-for-mineru\n",
                encoding="utf-8",
            )
            pool = ExtractorKeyPool.from_env_file(env_path)
            first = pool.next_key()
            second = pool.next_key()
            public = json.dumps(pool.list_public_keys(), ensure_ascii=False)

            self.assertEqual("mineru_1", first.alias)
            self.assertEqual("mineru_second", second.alias)
            self.assertNotIn("sk-test-secret", public)
            self.assertIn("mineru_second", public)

    def test_key_pool_skips_busy_and_cooldown_keys(self) -> None:
        clock = FakeClock()
        pool = ExtractorKeyPool(
            [
                ApiKeyRef(alias="mineru_a", secret="secret-a"),
                ApiKeyRef(alias="mineru_b", secret="secret-b"),
            ],
            per_key_submit_concurrency=1,
            clock=clock.now,
        )

        first = pool.acquire_key()
        second = pool.acquire_key()
        third = pool.acquire_key()

        self.assertEqual("mineru_a", first.alias)
        self.assertEqual("mineru_b", second.alias)
        self.assertIsNone(third)

        pool.release_key("mineru_a", cooldown_seconds=10)
        self.assertIsNone(pool.acquire_key())

        clock.advance(10.1)
        available = pool.acquire_key()
        self.assertEqual("mineru_a", available.alias)
        public = json.dumps(pool.list_public_keys(), ensure_ascii=False)
        self.assertIn('"in_flight"', public)
        self.assertNotIn("secret-a", public)

    def test_extraction_failure_releases_key_and_sets_cooldown(self) -> None:
        with workspace_tmpdir("extract-manager-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake test body")
            clock = FakeClock()
            pool = ExtractorKeyPool(
                [ApiKeyRef(alias="mineru_a", secret="secret-a")],
                clock=clock.now,
            )
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=FailingSubmitProvider(),
                    key_pool=pool,
                    failure_cooldown_seconds=30,
                )
                with self.assertRaises(RuntimeError):
                    manager.ensure_extraction(ExtractionRequest(input_file=source, attachment_key="ATTACH1"))

                public = pool.list_public_keys()[0]
                self.assertEqual(0, public["in_flight"])
                self.assertGreater(public["cooldown_remaining_seconds"], 0)
                self.assertIsNone(pool.acquire_key())
                job = ledger.list_extract_jobs()[0]
                self.assertEqual("failed_retryable", job["state"])
                self.assertEqual("mineru_a", job["api_key_alias"])
            finally:
                ledger.close()

    def test_manager_refuses_submit_when_configured_keys_are_unavailable(self) -> None:
        with workspace_tmpdir("extract-manager-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake test body")
            pool = ExtractorKeyPool([ApiKeyRef(alias="mineru_a", secret="secret-a")])
            held = pool.acquire_key()
            self.assertEqual("mineru_a", held.alias)
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    key_pool=pool,
                )
                with self.assertRaises(RuntimeError):
                    manager.ensure_extraction(ExtractionRequest(input_file=source, attachment_key="ATTACH1"))
                self.assertEqual([], ledger.list_extract_jobs())
            finally:
                pool.release_key("mineru_a")
                ledger.close()

class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FailingSubmitProvider:
    name = "failing"
    version = "0"

    def fingerprint(self, input_file, options_hash: str) -> str:
        return "failing"

    def submit(self, input_file, options_hash: str) -> ExtractJobState:
        raise RuntimeError("submit failed")

    def poll(self, external_job_id: str) -> ExtractJobState:
        raise AssertionError("poll should not be called")

    def download(self, external_job_id: str, output_dir) -> ExtractArtifact:
        raise AssertionError("download should not be called")


if __name__ == "__main__":
    unittest.main()
