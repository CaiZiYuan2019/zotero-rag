from __future__ import annotations

from datetime import datetime, timedelta, timezone
import itertools
import json
from pathlib import Path
from typing import Any
import unittest
from unittest.mock import patch
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

    def test_env_key_pool_splits_comma_separated_keys(self) -> None:
        with workspace_tmpdir("extract-key-pool-comma-") as tmpdir:
            env_path = tmpdir / ".env"
            env_path.write_text(
                "MINERU_KEY=sk-test-secret-0001,sk-test-secret-0002\n"
                "MINERU_KEY_SECOND=sk-test-secret-0003\n",
                encoding="utf-8",
            )
            pool = ExtractorKeyPool.from_env_file(env_path)
            aliases = [key.alias for key in pool._keys]
            secrets = {key.secret for key in pool._keys}
            public = json.dumps(pool.list_public_keys(), ensure_ascii=False)

            self.assertEqual(["mineru_1", "mineru_2", "mineru_second"], aliases)
            self.assertEqual(
                {"sk-test-secret-0001", "sk-test-secret-0002", "sk-test-secret-0003"},
                secrets,
            )
            self.assertNotIn("sk-test-secret", public)

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

    def test_recommended_timeout_is_poll_budget_not_http_timeout(self) -> None:
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
                result = manager.ensure_extraction(
                    ExtractionRequest(
                        input_file=source,
                        attachment_key="ATTACH1",
                        selected_pages="1-2",
                        selected_page_count=2,
                    )
                )
                # MinerU progress is tracked by polling: individual HTTP calls
                # must fail fast with the provider defaults instead of waiting
                # for the whole per-job poll budget.
                self.assertEqual(30, client.posts[0]["timeout"])
                self.assertEqual(30, client.puts[0]["timeout"])
                # The recommended budget is still persisted as the poll budget.
                expected_budget = 2 * 6 + 30  # recommended_mineru_timeout_seconds(2)
                persisted_options = (result.job.get("payload") or {}).get("options") or {}
                self.assertEqual(expected_budget, persisted_options.get("timeout_seconds"))
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


class AlwaysRunningPollProvider:
    name = "always-running"
    version = "0"

    def fingerprint(self, input_file, options_hash: str) -> str:
        return "always-running"

    def submit(self, input_file, options_hash: str, *, options=None, api_key=None) -> ExtractJobState:
        raise AssertionError("submit should not be called")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        return ExtractJobState(external_job_id=external_job_id, state="running")

    def download(self, external_job_id, output_dir, *, api_key=None, source_pdf=None, options=None) -> ExtractArtifact:
        raise AssertionError("download should not be called")


class SlowBatchProvider:
    """Provider whose remote batch stays running for a configurable number of
    polls (None = forever). Records submit/poll calls so tests can assert that
    no resubmission happened and the same external id kept being polled."""

    name = "slow-batch"
    version = "0"

    def __init__(self, polls_until_done: int | None = None) -> None:
        self.polls_until_done = polls_until_done
        self.submit_calls: list[dict[str, Any]] = []
        self.poll_calls: list[str] = []

    def fingerprint(self, input_file, options_hash: str) -> str:
        return "slow-batch"

    def submit(self, input_file, options_hash: str, *, options=None, api_key=None) -> ExtractJobState:
        self.submit_calls.append({"options_hash": options_hash})
        return ExtractJobState(external_job_id="batch-slow-1", state="running")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        self.poll_calls.append(external_job_id)
        if self.polls_until_done is not None and len(self.poll_calls) > self.polls_until_done:
            return ExtractJobState(external_job_id=external_job_id, state="completed")
        return ExtractJobState(external_job_id=external_job_id, state="running")

    def download(self, external_job_id, output_dir, *, api_key=None, source_pdf=None, options=None) -> ExtractArtifact:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "manifest.json"
        source = Path(source_pdf) if source_pdf is not None else Path(external_job_id)
        manifest.write_text(json.dumps({"source_pdf": str(source)}, ensure_ascii=False), encoding="utf-8")
        return ExtractArtifact(source_pdf=source, artifact_dir=output_dir, manifest_path=manifest)


class ExtractionManagerResumeTests(unittest.TestCase):
    def test_resume_poll_timeout_stands_down_and_keeps_batch(self) -> None:
        """M2: a poll timeout is not a failure. The persisted 36s budget is
        clamped up to the 30min floor, and on timeout the job stays running
        with its external_job_id so the next round polls the same batch."""
        with workspace_tmpdir("extract-resume-deadline-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=AlwaysRunningPollProvider(),
                    poll_interval_seconds=0.0,
                )
                job_id = str(uuid.uuid4())
                ledger.upsert_extract_job(
                    {
                        "job_id": job_id,
                        "attachment_key": "ATT1",
                        "pdf_sha256": "a" * 64,
                        "selected_pages": "",
                        "cache_key": "cache-tiny-budget",
                        "provider": "always-running",
                        "provider_version": "0",
                        "options_hash": "options",
                        "api_key_alias": "stub_stub",
                        "external_job_id": "batch-tiny-budget",
                        "state": "running",
                        "local_stage": "poll",
                        "payload": {"options": {"timeout_seconds": 36}},
                    }
                )
                ticks = itertools.count(start=0.0, step=10000.0)
                with patch("time.monotonic", side_effect=lambda: next(ticks)):
                    # No exception escapes: the round simply stands down.
                    result = manager.resume_extraction(job_id)
                self.assertEqual("running", result.job["state"])
                job = ledger.get_extract_job(job_id=job_id)
                self.assertEqual("running", job["state"])
                self.assertEqual("poll", job["local_stage"])
                self.assertEqual("poll-timeout", job["error_code"])
                # The 36s persisted budget is clamped up to the 1800s floor.
                self.assertIn("1800s", job["error_message"])
                # The remote batch id survives so the next round resumes it.
                self.assertEqual("batch-tiny-budget", job["external_job_id"])
            finally:
                ledger.close()

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

    def test_failed_retryable_resume_poll_falls_back_to_resubmit(self) -> None:
        with workspace_tmpdir("extract-resume-fallback-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=FailingPollThenRecordingProvider(),
                )
                job_id = str(uuid.uuid4())
                ledger.upsert_extract_job(
                    {
                        "job_id": job_id,
                        "attachment_key": "ATT1",
                        "pdf_sha256": "a" * 64,
                        "selected_pages": "",
                        "cache_key": "cache-1",
                        "provider": "failing-poll-then-recording",
                        "provider_version": "1",
                        "options_hash": "options",
                        "api_key_alias": "stub_stub",
                        "external_job_id": "batch-old",
                        "state": "failed_retryable",
                        "local_stage": "error",
                        "payload": {"input_file": str(source)},
                    }
                )
                result = manager.resume_extraction(job_id)

                self.assertEqual("downloaded", result.job["state"])
                self.assertNotEqual("batch-old", result.job.get("external_job_id"))
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


class FailingPollThenRecordingProvider(RecordingProvider):
    name = "failing-poll-then-recording"
    version = "1"

    def __init__(self) -> None:
        super().__init__()
        self._poll_count = 0

    def submit(
        self,
        input_file,
        options_hash: str,
        *,
        options=None,
        api_key=None,
    ) -> ExtractJobState:
        return ExtractJobState(external_job_id="batch-new", state="running")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        self._poll_count += 1
        if self._poll_count == 1:
            return ExtractJobState(external_job_id=external_job_id, state="failed", message="remote gone")
        return ExtractJobState(external_job_id=external_job_id, state="completed")


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


class PollTimeoutSemanticsTests(unittest.TestCase):
    """M2: poll timeout = stand down this round, resume the same batch next
    round; plus the decision-1 total lifetime cap."""

    def test_poll_budget_clamped_into_band(self) -> None:
        with workspace_tmpdir("extract-budget-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(ledger=ledger, cache_dir=tmpdir / "extract_cache")
                # Tiny persisted budgets clamp up to the 30min floor.
                self.assertEqual(1800.0, manager._poll_budget_seconds({"timeout_seconds": 36}))
                # Huge budgets clamp down to the lifetime cap (default 6h).
                self.assertEqual(21600.0, manager._poll_budget_seconds({"timeout_seconds": 100000}))
                # Missing/garbage budgets fall back to poll_timeout_seconds.
                self.assertEqual(21600.0, manager._poll_budget_seconds({}))
                self.assertEqual(21600.0, manager._poll_budget_seconds({"timeout_seconds": "junk"}))
                capped = ExtractionManager(
                    ledger=ledger, cache_dir=tmpdir / "extract_cache", job_max_lifetime_seconds=5000
                )
                self.assertEqual(5000.0, capped._poll_budget_seconds({"timeout_seconds": 100000}))
                floored = ExtractionManager(
                    ledger=ledger, cache_dir=tmpdir / "extract_cache", poll_timeout_seconds=900
                )
                self.assertEqual(1800.0, floored._poll_budget_seconds({}))
            finally:
                ledger.close()

    def test_ensure_timeout_does_not_cooldown_key_or_fail_job(self) -> None:
        with workspace_tmpdir("extract-ensure-timeout-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            clock = FakeClock()
            pool = ExtractorKeyPool(
                [ApiKeyRef(alias="mineru_a", secret="secret-a")],
                clock=clock.now,
            )
            provider = SlowBatchProvider(polls_until_done=None)
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=provider,
                    key_pool=pool,
                    poll_interval_seconds=0.0,
                )
                ticks = itertools.count(start=0.0, step=10000.0)
                with patch("time.monotonic", side_effect=lambda: next(ticks)):
                    result = manager.ensure_extraction(
                        ExtractionRequest(input_file=source, attachment_key="ATT1")
                    )
                # Stand-down semantics: no exception, job stays running.
                self.assertEqual("running", result.job["state"])
                self.assertEqual("poll-timeout", result.job["error_code"])
                self.assertEqual("batch-slow-1", result.job["external_job_id"])
                # Exactly one submission; the key was released without cooldown.
                self.assertEqual(1, len(provider.submit_calls))
                public = pool.list_public_keys()[0]
                self.assertEqual(0, public["in_flight"])
                self.assertEqual(0, public["cooldown_remaining_seconds"])
                self.assertIsNotNone(pool.acquire_key())
            finally:
                ledger.close()

    def test_resume_after_timeout_continues_same_batch_and_clears_error(self) -> None:
        with workspace_tmpdir("extract-timeout-resume-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            provider = SlowBatchProvider(polls_until_done=3)
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=provider,
                    poll_interval_seconds=0.0,
                )
                ticks = itertools.count(start=0.0, step=10000.0)
                with patch("time.monotonic", side_effect=lambda: next(ticks)):
                    first = manager.ensure_extraction(
                        ExtractionRequest(input_file=source, attachment_key="ATT1")
                    )
                self.assertEqual("running", first.job["state"])

                # Next round (real clock, generous budget): the same external
                # batch is polled to completion; no resubmission happens.
                result = manager.resume_extraction(first.job["job_id"])
                self.assertEqual("downloaded", result.job["state"])
                self.assertEqual(1, len(provider.submit_calls))
                self.assertEqual({"batch-slow-1"}, set(provider.poll_calls))
                # The success path clears the poll-timeout error record.
                self.assertIsNone(result.job["error_code"])
                self.assertIsNone(result.job["error_message"])
            finally:
                ledger.close()

    def test_failed_retryable_timeout_does_not_trigger_resubmit(self) -> None:
        """The fallback path must re-raise PollTimeoutError instead of
        resubmitting; AlwaysRunningPollProvider.submit raises AssertionError,
        so any resubmit attempt fails the test loudly."""
        with workspace_tmpdir("extract-fallback-timeout-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=AlwaysRunningPollProvider(),
                    poll_interval_seconds=0.0,
                )
                job_id = str(uuid.uuid4())
                ledger.upsert_extract_job(
                    {
                        "job_id": job_id,
                        "attachment_key": "ATT1",
                        "pdf_sha256": "a" * 64,
                        "selected_pages": "",
                        "cache_key": "cache-fallback",
                        "provider": "always-running",
                        "provider_version": "0",
                        "options_hash": "options",
                        "api_key_alias": "stub_stub",
                        "external_job_id": "batch-failed",
                        "state": "failed_retryable",
                        "local_stage": "poll",
                        "payload": {"options": {"timeout_seconds": 36}},
                    }
                )
                ticks = itertools.count(start=0.0, step=10000.0)
                with patch("time.monotonic", side_effect=lambda: next(ticks)):
                    result = manager.resume_extraction(job_id)
                # The job is stood down (running), not resubmitted; the
                # external batch id is preserved for the next round.
                self.assertEqual("running", result.job["state"])
                job = ledger.get_extract_job(job_id=job_id)
                self.assertEqual("poll-timeout", job["error_code"])
                self.assertEqual("batch-failed", job["external_job_id"])
            finally:
                ledger.close()

    def test_resume_flips_overdue_job_to_manual_review(self) -> None:
        with workspace_tmpdir("extract-lifetime-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=AlwaysRunningPollProvider(),
                    poll_interval_seconds=0.0,
                )
                job_id = str(uuid.uuid4())
                old = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
                ledger.upsert_extract_job(
                    {
                        "job_id": job_id,
                        "attachment_key": "ATT1",
                        "pdf_sha256": "a" * 64,
                        "selected_pages": "",
                        "cache_key": "cache-overdue",
                        "provider": "always-running",
                        "provider_version": "0",
                        "options_hash": "options",
                        "api_key_alias": "stub_stub",
                        "external_job_id": "batch-overdue",
                        "state": "running",
                        "local_stage": "poll",
                        "submitted_at": old,
                        "payload": {"options": {"timeout_seconds": 36}},
                    }
                )
                with self.assertRaises(RuntimeError) as ctx:
                    manager.resume_extraction(job_id)
                self.assertIn("failed_manual_review", str(ctx.exception))
                job = ledger.get_extract_job(job_id=job_id)
                self.assertEqual("failed_manual_review", job["state"])
                self.assertEqual("job-max-lifetime", job["error_code"])
                # A second resume hits the classify manual_review branch and
                # keeps the state stable.
                with self.assertRaises(RuntimeError) as ctx2:
                    manager.resume_extraction(job_id)
                self.assertIn("manual review", str(ctx2.exception))
                self.assertEqual("failed_manual_review", ledger.get_extract_job(job_id=job_id)["state"])
            finally:
                ledger.close()

    def test_lifetime_cap_ignores_download_and_terminal_states(self) -> None:
        with workspace_tmpdir("extract-lifetime-skip-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            provider = SlowBatchProvider(polls_until_done=0)
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=provider,
                    poll_interval_seconds=0.0,
                )
                done = manager.ensure_extraction(
                    ExtractionRequest(input_file=source, attachment_key="ATT1")
                )
                self.assertEqual("downloaded", done.job["state"])
                job_id = done.job["job_id"]
                old = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
                ledger.conn.execute(
                    "UPDATE extract_jobs SET submitted_at = ? WHERE job_id = ?", (old, job_id)
                )
                # Terminal (downloaded) jobs must never be flipped by the cap.
                skipped = manager.resume_extraction(job_id)
                self.assertTrue(skipped.cache_hit)
                self.assertEqual("downloaded", ledger.get_extract_job(job_id=job_id)["state"])

                # A completed-but-not-downloaded job only needs a local
                # download; the cap must not waste the finished remote batch.
                ledger.set_extract_job_state(job_id, state="completed", local_stage="download")
                resumed = manager.resume_extraction(job_id)
                self.assertEqual("downloaded", resumed.job["state"])
            finally:
                ledger.close()

    def test_ensure_does_not_auto_resubmit_overdue_job(self) -> None:
        with workspace_tmpdir("extract-overdue-resubmit-") as tmpdir:
            source = tmpdir / "paper.pdf"
            source.write_bytes(b"%PDF-1.4 fake body")
            provider = SlowBatchProvider(polls_until_done=None)
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=provider,
                    poll_interval_seconds=0.0,
                )
                ticks = itertools.count(start=0.0, step=10000.0)
                with patch("time.monotonic", side_effect=lambda: next(ticks)):
                    first = manager.ensure_extraction(
                        ExtractionRequest(input_file=source, attachment_key="ATT1")
                    )
                job_id = first.job["job_id"]
                # Simulate an old, failed job: backdate the clock and mark it
                # failed_retryable (non-reusable -> ensure would resubmit).
                old = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
                ledger.set_extract_job_state(job_id, state="failed_retryable", local_stage="error")
                ledger.conn.execute(
                    "UPDATE extract_jobs SET submitted_at = ? WHERE job_id = ?", (old, job_id)
                )
                with self.assertRaises(RuntimeError) as ctx:
                    manager.ensure_extraction(
                        ExtractionRequest(input_file=source, attachment_key="ATT1")
                    )
                self.assertIn("failed_manual_review", str(ctx.exception))
                job = ledger.get_extract_job(job_id=job_id)
                self.assertEqual("failed_manual_review", job["state"])
                self.assertEqual("job-max-lifetime", job["error_code"])
                self.assertEqual(1, len(provider.submit_calls))
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Phase 2 (MODIFICATION_PLAN_20260904 §2.2): submit_only/poll_once/harvest
# primitives, graded error classification, and the 决策2 resubmit budget.
# ---------------------------------------------------------------------------

from zoterorag.extractors.mineru import MinerUAPIError  # noqa: E402


class StatusCodeFailingPollProvider:
    """Poll raises MinerUAPIError with a fixed HTTP status_code.

    fail_times=None raises forever; otherwise completes after that many
    failures. Records the api_key secret passed to each poll so tests can
    assert key affinity/fallback.
    """

    name = "status-failing"
    version = "0"

    def __init__(self, status_code: int, fail_times: int | None = None) -> None:
        self.status_code = status_code
        self.fail_times = fail_times
        self.poll_keys: list[str | None] = []
        self.submit_calls = 0

    def fingerprint(self, input_file, options_hash: str) -> str:
        return "status-failing"

    def submit(self, input_file, options_hash: str, *, options=None, api_key=None) -> ExtractJobState:
        self.submit_calls += 1
        return ExtractJobState(external_job_id="batch-sf", state="running")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        self.poll_keys.append(api_key)
        if self.fail_times is None or len(self.poll_keys) <= self.fail_times:
            raise MinerUAPIError(
                f"http {self.status_code}",
                stage="poll",
                batch_id=external_job_id,
                status_code=self.status_code,
            )
        return ExtractJobState(external_job_id=external_job_id, state="completed")

    def download(self, external_job_id, output_dir, *, api_key=None, source_pdf=None, options=None) -> ExtractArtifact:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "manifest.json"
        source = Path(source_pdf) if source_pdf is not None else Path(external_job_id)
        manifest.write_text(json.dumps({"source_pdf": str(source)}, ensure_ascii=False), encoding="utf-8")
        return ExtractArtifact(source_pdf=source, artifact_dir=output_dir, manifest_path=manifest)


class KeyAwarePollProvider:
    """Poll fails with 401 for one key's secret and completes for any other."""

    name = "key-aware"
    version = "0"

    def __init__(self, failing_secret: str) -> None:
        self.failing_secret = failing_secret
        self.poll_keys: list[str | None] = []

    def fingerprint(self, input_file, options_hash: str) -> str:
        return "key-aware"

    def submit(self, input_file, options_hash: str, *, options=None, api_key=None) -> ExtractJobState:
        return ExtractJobState(external_job_id="batch-ka", state="running")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        self.poll_keys.append(api_key)
        if api_key == self.failing_secret:
            raise MinerUAPIError("unauthorized", stage="poll", batch_id=external_job_id, status_code=401)
        return ExtractJobState(external_job_id=external_job_id, state="completed")

    def download(self, external_job_id, output_dir, *, api_key=None, source_pdf=None, options=None) -> ExtractArtifact:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "manifest.json"
        source = Path(source_pdf) if source_pdf is not None else Path(external_job_id)
        manifest.write_text(json.dumps({"source_pdf": str(source)}, ensure_ascii=False), encoding="utf-8")
        return ExtractArtifact(source_pdf=source, artifact_dir=output_dir, manifest_path=manifest)


class RemoteFailingPollProvider:
    """Poll reports the remote batch as truly failed (state='failed')."""

    name = "remote-failing"
    version = "0"

    def __init__(self) -> None:
        self.submit_calls = 0
        self.poll_calls = 0

    def fingerprint(self, input_file, options_hash: str) -> str:
        return "remote-failing"

    def submit(self, input_file, options_hash: str, *, options=None, api_key=None) -> ExtractJobState:
        self.submit_calls += 1
        return ExtractJobState(external_job_id=f"batch-rf-{self.submit_calls}", state="running")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        self.poll_calls += 1
        return ExtractJobState(external_job_id=external_job_id, state="failed", message="remote exploded")

    def download(self, external_job_id, output_dir, *, api_key=None, source_pdf=None, options=None) -> ExtractArtifact:
        raise AssertionError("download should not be called")


class SubmitCompletedPollFailedProvider:
    """submit() reports completed but the confirmation poll reports failed
    (H1: the first harvest poll must confirm even a 'completed' submit)."""

    name = "submit-completed"
    version = "0"

    def fingerprint(self, input_file, options_hash: str) -> str:
        return "submit-completed"

    def submit(self, input_file, options_hash: str, *, options=None, api_key=None) -> ExtractJobState:
        return ExtractJobState(external_job_id="batch-sc", state="completed")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        return ExtractJobState(external_job_id=external_job_id, state="failed", message="died after submit")

    def download(self, external_job_id, output_dir, *, api_key=None, source_pdf=None, options=None) -> ExtractArtifact:
        raise AssertionError("download should not be called")


def _phase2_manager(tmpdir: Path, provider: Any, secrets: list[str], **kwargs: Any):
    """Build ledger/pool/manager for Phase 2 tests (poll_interval=0 so
    harvest loops never really sleep)."""
    ledger = StateLedger(tmpdir / "state.sqlite")
    pool = ExtractorKeyPool(
        [ApiKeyRef(alias=f"mineru_{chr(97 + i)}", secret=secret) for i, secret in enumerate(secrets)],
        clock=FakeClock().now,
    )
    manager = ExtractionManager(
        ledger=ledger,
        cache_dir=tmpdir / "cache",
        provider=provider,
        key_pool=pool,
        poll_interval_seconds=0.0,
        **kwargs,
    )
    return ledger, pool, manager


class Phase2PrimitiveTests(unittest.TestCase):
    def test_submit_only_releases_key_immediately(self) -> None:
        with workspace_tmpdir("phase2-submit-only-") as tmpdir:
            provider = RecordingProvider()
            ledger, pool, manager = _phase2_manager(tmpdir, provider, ["s-a"])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                result = manager.submit_only(
                    ExtractionRequest(input_file=input_file, attachment_key="ATT1")
                )
                self.assertFalse(result.cache_hit)
                self.assertEqual("running", result.job["state"])
                self.assertEqual("poll", result.job["local_stage"])
                self.assertEqual("batch-1", result.job["external_job_id"])
                # Phase 2: the key is released right after the submit call,
                # not held across the poll loop.
                self.assertEqual(0, pool.list_public_keys()[0]["in_flight"])
                self.assertEqual(0.0, pool.cooldown_remaining("mineru_a"))
            finally:
                ledger.close()

    def test_ensure_confirms_submit_completed_with_poll(self) -> None:
        with workspace_tmpdir("phase2-h1-") as tmpdir:
            provider = SubmitCompletedPollFailedProvider()
            ledger, pool, manager = _phase2_manager(tmpdir, provider, [])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                with self.assertRaises(MinerUAPIError):
                    manager.ensure_extraction(
                        ExtractionRequest(input_file=input_file, attachment_key="ATT2")
                    )
                job = ledger.list_extract_jobs()[0]
                self.assertEqual("failed_retryable", job["state"])
                self.assertEqual("died after submit", job["error_message"])
            finally:
                ledger.close()

    def test_poll_once_returns_none_when_no_key_available(self) -> None:
        with workspace_tmpdir("phase2-nokey-") as tmpdir:
            provider = RecordingProvider()
            ledger, pool, manager = _phase2_manager(tmpdir, provider, ["s-a"])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                submitted = manager.submit_only(
                    ExtractionRequest(input_file=input_file, attachment_key="ATT3")
                )
                held = pool.acquire_key()
                self.assertIsNotNone(held)
                # The only key is checked out: poll_once must return None
                # (harvest then sleeps) instead of failing the job.
                self.assertIsNone(manager.poll_once(submitted.job["job_id"]))
                job = ledger.get_extract_job(job_id=submitted.job["job_id"])
                self.assertEqual("running", job["state"])
                pool.release_key(held.alias)
            finally:
                ledger.close()

    def test_poll_once_key_affinity_and_fallback(self) -> None:
        with workspace_tmpdir("phase2-affinity-") as tmpdir:
            provider = StatusCodeFailingPollProvider(503, fail_times=0)
            ledger, pool, manager = _phase2_manager(tmpdir, provider, ["s-a", "s-b"])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                submitted = manager.submit_only(
                    ExtractionRequest(input_file=input_file, attachment_key="ATT4")
                )
                job_id = submitted.job["job_id"]
                # submit grabbed mineru_a (round-robin first): the poll
                # prefers the same key (single-account key affinity).
                manager.poll_once(job_id)
                self.assertEqual(["s-a"], provider.poll_keys)
                # Preferred key busy -> fall back to any available key.
                held = pool.acquire_key(prefer_alias="mineru_a")
                self.assertEqual("mineru_a", held.alias)
                manager.poll_once(job_id)
                self.assertEqual(["s-a", "s-b"], provider.poll_keys)
                pool.release_key(held.alias)
            finally:
                ledger.close()

    def test_submit_pace_enforces_min_interval(self) -> None:
        with workspace_tmpdir("phase2-pace-") as tmpdir:
            provider = RecordingProvider()
            ledger, pool, manager = _phase2_manager(
                tmpdir, provider, [], submit_min_interval_seconds=5.0
            )
            try:
                clock = FakeClock()
                sleeps: list[float] = []

                def fake_sleep(seconds: float) -> None:
                    sleeps.append(seconds)
                    clock.advance(seconds)

                first = tmpdir / "a.pdf"
                first.write_bytes(b"%PDF-1.4")
                with patch("time.monotonic", clock.now), patch("time.sleep", fake_sleep):
                    manager.submit_only(ExtractionRequest(input_file=first, attachment_key="A"))
                    manager.submit_only(
                        ExtractionRequest(
                            input_file=first,
                            attachment_key="B",
                            selected_pages="1-2",
                            selected_page_count=2,
                        )
                    )
                # The second submit waits out the full 5s interval.
                self.assertEqual([5.0], sleeps)
            finally:
                ledger.close()

    def test_poll_rate_limiter_waits_when_window_full(self) -> None:
        with workspace_tmpdir("phase2-ratelimit-") as tmpdir:
            provider = RecordingProvider()
            ledger, pool, manager = _phase2_manager(tmpdir, provider, [], poll_max_per_minute=2)
            try:
                clock = FakeClock()
                sleeps: list[float] = []

                def fake_sleep(seconds: float) -> None:
                    sleeps.append(seconds)
                    clock.advance(seconds)

                with patch("time.monotonic", clock.now), patch("time.sleep", fake_sleep):
                    manager._poll_rate_wait()
                    manager._poll_rate_wait()
                    manager._poll_rate_wait()  # window full: must wait ~60s
                self.assertTrue(sleeps)
                self.assertAlmostEqual(60.0, sum(sleeps), delta=1.0)
            finally:
                ledger.close()


class Phase2ErrorClassificationTests(unittest.TestCase):
    def test_429_stands_down_and_cools_key(self) -> None:
        with workspace_tmpdir("phase2-429-") as tmpdir:
            provider = StatusCodeFailingPollProvider(429)
            ledger, pool, manager = _phase2_manager(tmpdir, provider, ["s-a"])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                # 429 stands the round down (no exception): job stays running
                # with the batch id, key cools for rate_limit_cooldown_seconds.
                result = manager.ensure_extraction(
                    ExtractionRequest(input_file=input_file, attachment_key="ATT5")
                )
                self.assertEqual("running", result.job["state"])
                self.assertEqual("rate-limited", result.job["error_code"])
                self.assertEqual("batch-sf", result.job["external_job_id"])
                self.assertEqual(180.0, pool.cooldown_remaining("mineru_a"))
            finally:
                ledger.close()

    def test_401_single_key_fails_job(self) -> None:
        with workspace_tmpdir("phase2-401-") as tmpdir:
            provider = StatusCodeFailingPollProvider(401)
            ledger, pool, manager = _phase2_manager(tmpdir, provider, ["s-a"])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                # The only key fails auth -> pool effectively disabled -> the
                # job fails (M9 surfaces it as a blocked document).
                with self.assertRaises(MinerUAPIError):
                    manager.ensure_extraction(
                        ExtractionRequest(input_file=input_file, attachment_key="ATT6")
                    )
                job = ledger.list_extract_jobs()[0]
                self.assertEqual("failed_retryable", job["state"])
                self.assertEqual("no-keys-available", job["error_code"])
                self.assertEqual(3600.0, pool.cooldown_remaining("mineru_a"))
            finally:
                ledger.close()

    def test_401_falls_back_to_other_key(self) -> None:
        with workspace_tmpdir("phase2-401-fallback-") as tmpdir:
            provider = KeyAwarePollProvider("s-a")
            ledger, pool, manager = _phase2_manager(tmpdir, provider, ["s-a", "s-b"])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                result = manager.ensure_extraction(
                    ExtractionRequest(input_file=input_file, attachment_key="ATT7")
                )
                self.assertEqual("downloaded", result.job["state"])
                # First poll used the submitting key (affinity) and got 401;
                # the retry fell back to the second key and completed.
                self.assertEqual(["s-a", "s-b"], provider.poll_keys)
                self.assertEqual(3600.0, pool.cooldown_remaining("mineru_a"))
                self.assertEqual(0.0, pool.cooldown_remaining("mineru_b"))
            finally:
                ledger.close()

    def test_503_retries_with_backoff_then_completes(self) -> None:
        with workspace_tmpdir("phase2-503-") as tmpdir:
            provider = StatusCodeFailingPollProvider(503, fail_times=2)
            ledger, pool, manager = _phase2_manager(tmpdir, provider, ["s-a"])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                result = manager.ensure_extraction(
                    ExtractionRequest(input_file=input_file, attachment_key="ATT8")
                )
                self.assertEqual("downloaded", result.job["state"])
                # 2 transient failures + the successful third poll, and the
                # key was never cooled (5xx is not the key's fault).
                self.assertEqual(3, len(provider.poll_keys))
                self.assertEqual(0.0, pool.cooldown_remaining("mineru_a"))
            finally:
                ledger.close()

    def test_unclassified_400_goes_catchall(self) -> None:
        with workspace_tmpdir("phase2-400-") as tmpdir:
            provider = StatusCodeFailingPollProvider(400)
            ledger, pool, manager = _phase2_manager(tmpdir, provider, ["s-a"])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                with self.assertRaises(MinerUAPIError):
                    manager.ensure_extraction(
                        ExtractionRequest(input_file=input_file, attachment_key="ATT9")
                    )
                job = ledger.list_extract_jobs()[0]
                self.assertEqual("failed_retryable", job["state"])
                self.assertEqual("MinerUAPIError", job["error_code"])
                # Unclassified errors keep the historical failure cooldown.
                self.assertEqual(60.0, pool.cooldown_remaining("mineru_a"))
            finally:
                ledger.close()

    def test_remote_batch_failure_does_not_cool_key(self) -> None:
        with workspace_tmpdir("phase2-remotefail-") as tmpdir:
            provider = RemoteFailingPollProvider()
            ledger, pool, manager = _phase2_manager(tmpdir, provider, ["s-a"])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                with self.assertRaises(MinerUAPIError):
                    manager.ensure_extraction(
                        ExtractionRequest(input_file=input_file, attachment_key="ATT10")
                    )
                job = ledger.list_extract_jobs()[0]
                self.assertEqual("failed_retryable", job["state"])
                self.assertEqual("remote exploded", job["error_message"])
                # §2.2 error table: a truly failed remote batch does not cool
                # the key (deliberate change from the old 60s catch-all).
                self.assertEqual(0.0, pool.cooldown_remaining("mineru_a"))
            finally:
                ledger.close()


class Phase2ResubmitBudgetTests(unittest.TestCase):
    def test_resume_resubmit_capped_and_counted(self) -> None:
        with workspace_tmpdir("phase2-resubmit-cap-") as tmpdir:
            provider = RemoteFailingPollProvider()
            ledger, pool, manager = _phase2_manager(tmpdir, provider, [])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                request = ExtractionRequest(input_file=input_file, attachment_key="ATT11")
                with self.assertRaises(MinerUAPIError):
                    manager.ensure_extraction(request)
                job = ledger.list_extract_jobs()[0]
                job_id = job["job_id"]
                # 决策2: each resume resubmits once (attempts 1..3), then the
                # next resume flips the job to failed_manual_review.
                for expected_attempts in (1, 2, 3):
                    with self.assertRaises(MinerUAPIError):
                        manager.resume_extraction(job_id)
                    job = ledger.get_extract_job(job_id=job_id)
                    self.assertEqual("failed_retryable", job["state"])
                    self.assertEqual(expected_attempts, job["payload"]["resubmit_attempts"])
                with self.assertRaises(RuntimeError):
                    manager.resume_extraction(job_id)
                job = ledger.get_extract_job(job_id=job_id)
                self.assertEqual("failed_manual_review", job["state"])
                self.assertEqual("max-resubmit-attempts", job["error_code"])
                # 1 initial submit + 3 resubmits; the capped round submitted
                # nothing new.
                self.assertEqual(4, provider.submit_calls)
            finally:
                ledger.close()

    def test_ensure_resubmit_carries_count_and_respects_cap(self) -> None:
        with workspace_tmpdir("phase2-ensure-cap-") as tmpdir:
            provider = RemoteFailingPollProvider()
            ledger, pool, manager = _phase2_manager(tmpdir, provider, [])
            try:
                input_file = tmpdir / "doc.pdf"
                input_file.write_bytes(b"%PDF-1.4")
                request = ExtractionRequest(input_file=input_file, attachment_key="ATT12")
                with self.assertRaises(MinerUAPIError):
                    manager.ensure_extraction(request)
                job = ledger.list_extract_jobs()[0]
                job_id = job["job_id"]
                self.assertEqual(0, job["payload"].get("resubmit_attempts", 0))
                # ensure-path resubmission carries the count in the payload
                # (upsert would otherwise reset it every round).
                with self.assertRaises(MinerUAPIError):
                    manager.ensure_extraction(request)
                job = ledger.get_extract_job(job_id=job_id)
                self.assertEqual(1, job["payload"]["resubmit_attempts"])
                # Push the counter to the cap: ensure must now refuse to
                # resubmit and flip the job to failed_manual_review.
                payload = dict(job["payload"])
                payload["resubmit_attempts"] = 3
                ledger.set_extract_job_state(
                    job_id,
                    state="failed_retryable",
                    local_stage="error",
                    payload=payload,
                )
                with self.assertRaises(RuntimeError):
                    manager.ensure_extraction(request)
                job = ledger.get_extract_job(job_id=job_id)
                self.assertEqual("failed_manual_review", job["state"])
                self.assertEqual("max-resubmit-attempts", job["error_code"])
                self.assertEqual(2, provider.submit_calls)
            finally:
                ledger.close()
