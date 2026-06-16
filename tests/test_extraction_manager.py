from __future__ import annotations

import json
from pathlib import Path
import unittest
import uuid

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.extractors import (
    ApiKeyRef,
    ExtractionManager,
    ExtractionRequest,
    ExtractorKeyPool,
    MinerUProvider,
    recommended_mineru_timeout_seconds,
)
from zoterorag.extractors.base import ExtractArtifact, ExtractJobState
from zoterorag.extractors.cache import extractor_cache_key, stable_options_hash
from tests.test_mineru_provider import FakeMinerUClient, FlakyFakeMinerUClient, build_zip


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

    def test_api_key_ref_repr_and_str_redact_secret(self) -> None:
        key = ApiKeyRef(alias="mineru_a", secret="sk-test-secret-xyz")
        repr_text = repr(key)
        str_text = str(key)

        self.assertIn("mineru_a", repr_text)
        self.assertIn("mineru_a", str_text)
        self.assertNotIn("sk-test-secret-xyz", repr_text)
        self.assertNotIn("sk-test-secret-xyz", str_text)
        self.assertIn("<redacted>", repr_text)
        self.assertIn("<redacted>", str_text)

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

    def test_manager_waits_for_key_to_become_available(self) -> None:
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
                # With the only key held, a zero-timeout acquire returns None.
                self.assertIsNone(manager._acquire_key_with_wait(timeout_seconds=0.0))

                # Release the key and verify the manager can acquire it.
                pool.release_key("mineru_a")
                self.assertIsNotNone(manager._acquire_key_with_wait(timeout_seconds=1.0))
            finally:
                pool.release_key("mineru_a")
                ledger.close()

    def test_manager_still_raises_when_key_never_becomes_available(self) -> None:
        with workspace_tmpdir("extract-manager-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake test body")
            clock = FakeClock()
            pool = ExtractorKeyPool(
                [ApiKeyRef(alias="mineru_a", secret="secret-a")],
                clock=clock.now,
            )
            pool.acquire_key()
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    key_pool=pool,
                    key_acquire_timeout_seconds=0.1,
                )
                with self.assertRaises(RuntimeError):
                    manager.ensure_extraction(ExtractionRequest(input_file=source, attachment_key="ATTACH1"))
                self.assertEqual([], ledger.list_extract_jobs())
            finally:
                pool.release_key("mineru_a")
                ledger.close()

    def test_cache_key_includes_provider_endpoint(self) -> None:
        options_hash = stable_options_hash({"model_version": "vlm"})
        base = extractor_cache_key(
            pdf_sha256="a" * 64,
            selected_pages="1-3",
            extractor_name="mineru",
            extractor_version="v4",
            options_hash=options_hash,
        )
        with_endpoint = extractor_cache_key(
            pdf_sha256="a" * 64,
            selected_pages="1-3",
            extractor_name="mineru",
            extractor_version="v4",
            options_hash=options_hash,
            endpoint_url="https://mineru.example.test/api",
        )
        self.assertNotEqual(base, with_endpoint)

    def test_resume_download_uses_persisted_source_pdf_after_restart(self) -> None:
        with workspace_tmpdir("extract-resume-") as tmpdir:
            source = tmpdir / "original.pdf"
            source.write_bytes(b"%PDF-1.4 fake test body")
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                # First manager processes up to the completed/download stage.
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=RecordingProvider(),
                )
                first = manager.ensure_extraction(
                    ExtractionRequest(
                        input_file=source,
                        attachment_key="ATTACH1",
                        selected_pages="1",
                        selected_page_count=1,
                    )
                )
                job_id = first.job["job_id"]

                # Simulate a process restart: the in-memory provider is replaced
                # and the job is reset to the remote-completed state.
                ledger.set_extract_job_state(
                    job_id,
                    state="completed",
                    local_stage="download",
                    payload=first.job.get("payload") or {},
                )
                new_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=RecordingProvider(),
                )
                result = new_manager.resume_extraction(job_id)

                self.assertEqual("downloaded", result.job["state"])
                self.assertEqual(str(source), result.job["payload"]["source_pdf"])
            finally:
                ledger.close()

    def test_recommended_timeout_is_passed_to_mineru_provider(self) -> None:
        with workspace_tmpdir("extract-timeout-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            ledger = StateLedger(tmpdir / "state.sqlite")
            client = FakeMinerUClient(zip_bytes=build_zip({"full.md": "# Done\n"}))
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MinerUProvider(
                        client=client,
                        request_timeout_seconds=30,
                        transfer_timeout_seconds=30,
                    ),
                    key_pool=ExtractorKeyPool([ApiKeyRef(alias="mineru_a", secret="secret-a")]),
                )
                manager.ensure_extraction(
                    ExtractionRequest(
                        input_file=source,
                        attachment_key="ATTACH1",
                        selected_pages="1-2",
                        selected_page_count=2,
                    )
                )
                expected_timeout = 2 * 6 + 30  # recommended_mineru_timeout_seconds(2)
                self.assertEqual(expected_timeout, client.posts[0]["timeout"])
                self.assertEqual(expected_timeout, client.puts[0]["timeout"])
            finally:
                ledger.close()

    def test_manager_persists_desensitized_error_messages(self) -> None:
        with workspace_tmpdir("extract-error-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            ledger = StateLedger(tmpdir / "state.sqlite")
            client = FlakyFakeMinerUClient(zip_bytes=b"", post_failures=1, post_status=400)
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MinerUProvider(client=client, retry_initial_seconds=0.0, retry_jitter=0.0),
                    key_pool=ExtractorKeyPool([ApiKeyRef(alias="mineru_a", secret="secret-a")]),
                )
                with self.assertRaises(RuntimeError):
                    manager.ensure_extraction(ExtractionRequest(input_file=source, attachment_key="ATTACH1"))

                job = ledger.list_extract_jobs()[0]
                error_message = job["error_message"]
                self.assertIn("MinerU API request failed", error_message)
                self.assertNotIn("https://", error_message)
                self.assertNotIn("transient failure", error_message)
            finally:
                ledger.close()

    def test_dotenv_parser_handles_inline_comments_and_escaped_quotes(self) -> None:
        with workspace_tmpdir("dotenv-parser-") as tmpdir:
            env_path = tmpdir / ".env"
            env_path.write_text(
                'MINERU_KEY=sk-test-secret  # inline comment\n'
                'MINERU_API_KEY_SECOND="sk-second with spaces"\n'
                'MINERU_API_KEY_THIRD="say \\"hi\\""\n'
                'BAILIAN_KEY=plain#notcomment\n'
                '# full line comment\n'
                'EMPTY_VAR=\n',
                encoding="utf-8",
            )
            pool = ExtractorKeyPool.from_env_file(env_path)
            aliases = {key.alias: key.secret for key in pool._keys}
            self.assertEqual("sk-test-secret", aliases["mineru_1"])
            self.assertEqual("sk-second with spaces", aliases["mineru_second"])
            self.assertEqual('say "hi"', aliases["mineru_third"])
            # BAILIAN_KEY is not a MinerU key, but the parser must still preserve
            # the inline # character in the value.
            from zoterorag.extractors.key_pool import load_dotenv_values
            self.assertEqual("plain#notcomment", load_dotenv_values(env_path)["BAILIAN_KEY"])

class RecordingProvider:
    name = "recording"
    version = "1"

    def __init__(self) -> None:
        self.download_calls: list[dict[str, Any]] = []

    def fingerprint(self, input_file, options_hash: str) -> str:
        return "recording"

    def submit(self, input_file, options_hash: str, *, options=None, api_key=None) -> ExtractJobState:
        return ExtractJobState(external_job_id="batch-1", state="running")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        return ExtractJobState(external_job_id=external_job_id, state="completed")

    def download(
        self,
        external_job_id: str,
        output_dir,
        *,
        api_key=None,
        source_pdf=None,
        options=None,
    ) -> ExtractArtifact:
        self.download_calls.append({"external_job_id": external_job_id, "source_pdf": source_pdf, "options": options})
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "manifest.json"
        source = Path(source_pdf) if source_pdf is not None else Path(external_job_id)
        manifest.write_text(json.dumps({"source_pdf": str(source)}, ensure_ascii=False), encoding="utf-8")
        return ExtractArtifact(source_pdf=source, artifact_dir=output_dir, manifest_path=manifest)


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

    def submit(self, input_file, options_hash: str, *, options=None, api_key=None) -> ExtractJobState:
        raise RuntimeError("submit failed")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        raise AssertionError("poll should not be called")

    def download(self, external_job_id: str, output_dir, *, api_key=None, source_pdf=None, options=None) -> ExtractArtifact:
        raise AssertionError("download should not be called")


class ExtractionManagerResumeTests(unittest.TestCase):
    def test_resume_poll_completes_and_downloads_artifact(self) -> None:
        with workspace_tmpdir("extract-resume-poll-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=RecordingProvider(),
                )
                first = manager.ensure_extraction(
                    ExtractionRequest(input_file=source, attachment_key="ATT1")
                )
                job_id = first.job["job_id"]

                # Simulate a restart at the poll stage.
                ledger.set_extract_job_state(
                    job_id,
                    state="running",
                    local_stage="poll",
                    payload=first.job.get("payload") or {},
                )
                result = manager.resume_extraction(job_id)
                self.assertEqual("downloaded", result.job["state"])
                self.assertEqual(str(source), result.job["payload"]["source_pdf"])
            finally:
                ledger.close()

    def test_resume_download_failure_marks_retryable(self) -> None:
        with workspace_tmpdir("extract-resume-download-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=RecordingProvider(),
                )
                first = manager.ensure_extraction(
                    ExtractionRequest(input_file=source, attachment_key="ATT1")
                )
                job_id = first.job["job_id"]

                ledger.set_extract_job_state(
                    job_id,
                    state="completed",
                    local_stage="download",
                    payload=first.job.get("payload") or {},
                )
                failing_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=FailingDownloadProvider(),
                )
                with self.assertRaises(RuntimeError):
                    failing_manager.resume_extraction(job_id)

                job = ledger.get_extract_job(job_id=job_id)
                self.assertEqual("failed_retryable", job["state"])
                self.assertIn("download failed", job["error_message"].lower())
            finally:
                ledger.close()

    def test_resume_poll_failure_marks_retryable(self) -> None:
        with workspace_tmpdir("extract-resume-poll-fail-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=FailingPollProvider(),
                )
                job_id = str(uuid.uuid4())
                ledger.upsert_extract_job(
                    {
                        "job_id": job_id,
                        "attachment_key": "ATT1",
                        "pdf_sha256": "a" * 64,
                        "selected_pages": "",
                        "cache_key": "cache-1",
                        "provider": "recording",
                        "provider_version": "1",
                        "options_hash": "options",
                        "api_key_alias": "stub_stub",
                        "external_job_id": "batch-1",
                        "state": "running",
                        "local_stage": "poll",
                        "payload": {"input_file": str(source)},
                    }
                )
                with self.assertRaises(RuntimeError):
                    manager.resume_extraction(job_id)

                job = ledger.get_extract_job(job_id=job_id)
                self.assertEqual("failed_retryable", job["state"])
                self.assertIn("poll failed", job["error_message"].lower())
            finally:
                ledger.close()


class FailingDownloadProvider(RecordingProvider):
    name = "failing-download"
    version = "1"

    def download(
        self,
        external_job_id: str,
        output_dir,
        *,
        api_key=None,
        source_pdf=None,
        options=None,
    ) -> ExtractArtifact:
        raise RuntimeError("download failed")


class FailingPollProvider:
    name = "failing-poll"
    version = "1"

    def fingerprint(self, input_file, options_hash: str) -> str:
        return "failing-poll"

    def submit(
        self,
        input_file,
        options_hash: str,
        *,
        options=None,
        api_key=None,
    ) -> ExtractJobState:
        raise AssertionError("submit should not be called")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        raise RuntimeError("poll failed")

    def download(
        self,
        external_job_id: str,
        output_dir,
        *,
        api_key=None,
        source_pdf=None,
        options=None,
    ) -> ExtractArtifact:
        raise AssertionError("download should not be called")


if __name__ == "__main__":
    unittest.main()
