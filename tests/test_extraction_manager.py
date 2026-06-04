from __future__ import annotations

import json
import unittest

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.extractors import (
    ExtractionManager,
    ExtractionRequest,
    ExtractorKeyPool,
    recommended_mineru_timeout_seconds,
)
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


if __name__ == "__main__":
    unittest.main()
