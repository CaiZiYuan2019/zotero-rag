from __future__ import annotations

import json
from pathlib import Path
import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.embeddings.profile import embedding_profile_hash
from zoterorag.extractors.base import ExtractArtifact, ExtractJobState, StubExtractorProvider
from zoterorag.extractors.manager import ExtractionManager, ExtractionRequest
from zoterorag.normalize import normalize_markdown_document
from zoterorag.pipeline import (
    cancel_ingest_job,
    create_ingest_plan,
    pause_ingest_job,
    resume_ingest_job,
    start_ingest_job,
)


class IngestPipelineTests(unittest.TestCase):
    def test_incremental_plan_uses_local_state_without_external_execution(self) -> None:
        with workspace_tmpdir("ingest-plan-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                seed_attachments(ledger)
                normalized = normalize_fixture(tmpdir, ledger)
                ledger.checkpoint(
                    normalized.document_id,
                    "embed:text-profile",
                    "indexed",
                    {"chunks": 1, "profile_hash": profile_hash(ledger, "text-profile")},
                )
                ledger.checkpoint(
                    normalized.document_id,
                    "embed:mm-profile",
                    "indexed",
                    {"chunks": 1, "profile_hash": profile_hash(ledger, "mm-profile")},
                )

                plan = create_ingest_plan(ledger)
                by_key = {doc["attachment_key"]: doc for doc in plan["documents"]}

                self.assertEqual(3, plan["summary"]["document_count"])
                self.assertNotIn("ATT_REVIEW", by_key)
                self.assertEqual("complete", by_key["ATT_DONE"]["next_stage"])
                self.assertEqual("extract:pending", by_key["ATT_NEW"]["next_stage"])
                self.assertEqual("extract:blocked", by_key["ATT_MISSING"]["next_stage"])
                self.assertEqual("text-profile", plan["text_profile"])
                self.assertEqual("mm-profile", plan["multimodal_profile"])
            finally:
                ledger.close()

    def test_profile_hash_mismatch_marks_embedding_pending(self) -> None:
        with workspace_tmpdir("ingest-profile-hash-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                ledger.upsert_attachments([build_attachment("ATT_DONE", title="Already Indexed")])
                normalized = normalize_fixture(tmpdir, ledger)
                ledger.checkpoint(
                    normalized.document_id,
                    "embed:text-profile",
                    "indexed",
                    {"chunks": 1, "profile_hash": "old-hash"},
                )

                plan = create_ingest_plan(ledger, include_multimodal=False)
                document = plan["documents"][0]
                embed_stage = next(stage for stage in document["stages"] if stage["stage"] == "embed:text-profile")

                self.assertEqual("embed:text-profile:pending", document["next_stage"])
                self.assertEqual("pending", embed_stage["status"])
                self.assertEqual("profile_changed", embed_stage["reason"])
                self.assertEqual("old-hash", embed_stage["checkpoint_profile_hash"])
                self.assertEqual(profile_hash(ledger, "text-profile"), embed_stage["profile_hash"])
            finally:
                ledger.close()

    def test_start_pause_resume_cancel_persist_job_events_and_checkpoints(self) -> None:
        with workspace_tmpdir("ingest-job-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                seed_attachments(ledger)

                result = start_ingest_job(ledger, zotero_key="ATT_NEW", include_multimodal=False)
                job_id = result["job"]["job_id"]

                self.assertEqual("planned", result["job"]["status"])
                self.assertEqual(1, result["plan"]["summary"]["document_count"])
                checkpoint = ledger.get_checkpoint("ATT_NEW", "ingest_plan")
                self.assertIsNotNone(checkpoint)
                self.assertEqual(job_id, checkpoint["payload"]["job_id"])

                paused = pause_ingest_job(ledger, job_id, reason="test pause")
                self.assertEqual("paused", paused["job"]["status"])

                resumed = resume_ingest_job(ledger, job_id, reason="test resume")
                self.assertEqual("planned", resumed["job"]["status"])
                self.assertEqual(1, resumed["plan"]["summary"]["document_count"])

                cancelled = cancel_ingest_job(ledger, job_id, reason="test cancel")
                self.assertEqual("cancelled", cancelled["job"]["status"])
                stages = [event["stage"] for event in cancelled["job"]["events"]]
                self.assertEqual(["plan", "pause", "resume", "cancel"], stages)
            finally:
                ledger.close()

    def test_execute_true_requires_dependencies(self) -> None:
        with workspace_tmpdir("ingest-execute-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                seed_attachments(ledger)
                # execute=True without extract_manager and paths raises ValueError.
                with self.assertRaises(ValueError):
                    start_ingest_job(ledger, execute=True)
                self.assertEqual([], ledger.list_jobs(kind="ingest"))
            finally:
                ledger.close()

    def test_execute_ingest_end_to_end_with_stub(self) -> None:
        with workspace_tmpdir("ingest-execute-e2e-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_E2E", title="End-to-End Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                result = start_ingest_job(
                    ledger,
                    zotero_key="ATT_E2E",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )

                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual(1, len(result["executed"]))
                self.assertEqual(0, len(result["failed"]))

                # Checkpoints record stage progress.
                self.assertEqual("done", ledger.get_checkpoint("ATT_E2E", "extract")["status"])
                self.assertEqual("normalized", ledger.get_checkpoint("ATT_E2E", "normalize")["status"])
                self.assertEqual("indexed", ledger.get_checkpoint("ATT_E2E", "embed:text-profile")["status"])

                # Normalized artifact and chunks were persisted.
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_E2E"))
                self.assertGreater(len(ledger.list_chunks("ATT_E2E", chunk_type="text")), 0)
            finally:
                ledger.close()

    def test_completed_extract_state_triggers_download_recovery(self) -> None:
        with workspace_tmpdir("ingest-completed-recovery-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_RECOV", title="Recovery Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                # First, get to a downloaded state.
                request = ExtractionRequest(
                    input_file=Path(attachment["file_path"]),
                    attachment_key="ATT_RECOV",
                )
                first = extract_manager.ensure_extraction(request)
                self.assertEqual("downloaded", first.job["state"])

                # Simulate the artifact going missing while the ledger still says
                # ``completed`` (remote done, local artifact not downloaded).
                artifact_dir = first.job.get("artifact_dir")
                self.assertIsNotNone(artifact_dir)
                import shutil

                shutil.rmtree(artifact_dir)
                ledger.set_extract_job_state(
                    first.job["job_id"],
                    state="completed",
                    local_stage="download",
                    artifact_dir=None,
                    extract_dir=None,
                    manifest_path=None,
                )

                plan = create_ingest_plan(ledger, include_multimodal=False)
                self.assertEqual("extract:pending", plan["documents"][0]["next_stage"])
                self.assertEqual("completed_artifact_not_downloaded", plan["documents"][0]["stages"][0]["reason"])

                result = start_ingest_job(
                    ledger,
                    zotero_key="ATT_RECOV",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )

                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual(1, len(result["executed"]))
                self.assertEqual("downloaded", ledger.get_extract_job(job_id=first.job["job_id"])["state"])
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_RECOV"))
            finally:
                ledger.close()

    def test_resume_ingest_job_skips_already_done_documents(self) -> None:
        with workspace_tmpdir("ingest-resume-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_RESUME", title="Resume Paper")
                ledger.upsert_attachments([attachment])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                first = start_ingest_job(
                    ledger,
                    zotero_key="ATT_RESUME",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )
                self.assertEqual("completed", first["job"]["status"])

                resumed = resume_ingest_job(ledger, first["job"]["job_id"])
                self.assertEqual("planned", resumed["job"]["status"])
                self.assertEqual("complete", resumed["plan"]["documents"][0]["next_stage"])
            finally:
                ledger.close()

    def test_shared_pdf_cache_key_attachment_reuses_extract_artifact(self) -> None:
        """Two attachments pointing to the same PDF must share one extract job."""
        with workspace_tmpdir("ingest-shared-cache-key-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                source = tmpdir / "shared.pdf"
                source.write_bytes(b"shared pdf content")
                att1 = build_attachment("ATT_SHARED_1", title="Shared One")
                att1["file_path"] = str(source)
                att1["file_exists"] = True
                att2 = build_attachment("ATT_SHARED_2", title="Shared Two")
                att2["file_path"] = str(source)
                att2["file_exists"] = True
                ledger.upsert_attachments([att1, att2])

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MarkdownStubExtractorProvider(),
                )
                first = start_ingest_job(
                    ledger,
                    zotero_key="ATT_SHARED_1",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )
                self.assertEqual("completed", first["job"]["status"])

                # The second attachment reuses the same cached artifact.
                second = start_ingest_job(
                    ledger,
                    zotero_key="ATT_SHARED_2",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )
                self.assertEqual("completed", second["job"]["status"])
                self.assertEqual(1, len(second["executed"]))
                self.assertEqual(0, len(second["failed"]))
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_SHARED_2"))

                # Only one extract job record should exist for the shared PDF.
                jobs = ledger.list_extract_jobs(limit=None)
                self.assertEqual(1, len(jobs))
            finally:
                ledger.close()

    def test_running_extract_job_is_resumed_and_completes(self) -> None:
        """An interrupted run that left an extract job in 'running' state can resume."""
        with workspace_tmpdir("ingest-running-resume-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                attachment = build_attachment_with_file(tmpdir, "ATT_RUN", title="Running Paper")
                ledger.upsert_attachments([attachment])

                provider = TwoPhaseStubExtractorProvider()

                # Seed a running extract job as if a previous run was interrupted.
                # Compute the same cache_key that ensure_extraction would use.
                from zoterorag.extractors.cache import (
                    extractor_cache_key,
                    sha256_file,
                    stable_options_hash,
                )

                pdf_sha256 = sha256_file(attachment["file_path"])
                # Match the default timeout that ensure_extraction adds for a
                # single-page document when no options are provided.
                options = {"timeout_seconds": 36}
                options_hash = stable_options_hash(options)
                cache_key = extractor_cache_key(
                    pdf_sha256=pdf_sha256,
                    selected_pages="",
                    extractor_name=provider.name,
                    extractor_version=provider.version,
                    options_hash=options_hash,
                    endpoint_url="",
                )
                ledger.upsert_extract_job(
                    {
                        "job_id": "running-job-id",
                        "attachment_key": "ATT_RUN",
                        "pdf_sha256": pdf_sha256,
                        "selected_pages": "",
                        "cache_key": cache_key,
                        "provider": provider.name,
                        "provider_version": provider.version,
                        "options_hash": options_hash,
                        "api_key_alias": "stub",
                        "external_job_id": "ext-running",
                        "state": "running",
                        "local_stage": "poll",
                        "submitted_at": "2026-01-01T00:00:00+00:00",
                        "last_poll_at": "2026-01-01T00:00:00+00:00",
                        "payload": {
                            "input_file": attachment["file_path"],
                            "options": options,
                        },
                    }
                )

                extract_manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=provider,
                )
                result = start_ingest_job(
                    ledger,
                    zotero_key="ATT_RUN",
                    include_multimodal=False,
                    execute=True,
                    extract_manager=extract_manager,
                    extract_cache_dir=tmpdir / "extract_cache",
                    normalized_dir=tmpdir / "normalized",
                    vector_store_dir=tmpdir / "vectors",
                )

                self.assertEqual("completed", result["job"]["status"])
                self.assertEqual(1, len(result["executed"]))
                self.assertEqual(0, len(result["failed"]))
                self.assertEqual("downloaded", ledger.get_extract_job(job_id="running-job-id")["state"])
                self.assertEqual("done", ledger.get_checkpoint("ATT_RUN", "extract")["status"])
                self.assertEqual("normalized", ledger.get_checkpoint("ATT_RUN", "normalize")["status"])
                self.assertIsNotNone(ledger.get_normalized_artifact("ATT_RUN"))
            finally:
                ledger.close()


def seed_profiles(ledger: StateLedger) -> None:
    ledger.upsert_embedding_profiles(
        [
            EmbeddingProfile(
                name="text-profile",
                provider="stub",
                model="stub",
                dimension=8,
                modality="text",
                enabled=True,
                default_for_text=True,
                backend="sqlite-local",
            ),
            EmbeddingProfile(
                name="mm-profile",
                provider="stub",
                model="stub-mm",
                dimension=8,
                modality="multimodal",
                enabled=True,
                default_for_multimodal=True,
                backend="sqlite-local",
            ),
        ]
    )


def profile_hash(ledger: StateLedger, profile_name: str) -> str:
    profile = next(profile for profile in ledger.list_embedding_profiles() if profile["name"] == profile_name)
    return embedding_profile_hash(profile)


def seed_attachments(ledger: StateLedger) -> None:
    ledger.upsert_attachments(
        [
            build_attachment("ATT_DONE", title="Already Indexed"),
            build_attachment("ATT_NEW", title="Needs Work"),
            build_attachment("ATT_MISSING", title="Missing PDF", file_exists=False),
            build_attachment("ATT_REVIEW", title="Review PDF", classification="needs_review"),
        ]
    )


def normalize_fixture(tmpdir, ledger: StateLedger):
    source_dir = tmpdir / "mineru"
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "hash.png").write_bytes(b"image")
    markdown = source_dir / "full.md"
    markdown.write_text("# Already Indexed\n\nBody.\n\n![Fig](images/hash.png)\n", encoding="utf-8")
    normalized = normalize_markdown_document(
        source_markdown=markdown,
        output_root=tmpdir / "normalized",
        document_id="DOC_DONE",
        attachment_key="ATT_DONE",
    )
    ledger.upsert_normalized_artifact(normalized.ledger_artifact())
    ledger.replace_document_chunks(normalized.document_id, normalized.chunks)
    return normalized


def build_attachment(
    attachment_key: str,
    *,
    title: str,
    classification: str = "included_auto",
    file_exists: bool = True,
) -> dict[str, object]:
    return {
        "attachment_key": attachment_key,
        "parent_key": f"PARENT-{attachment_key}",
        "content_type": "application/pdf",
        "relative_path": f"storage:{attachment_key}.pdf",
        "title": title,
        "abstract": "",
        "date": "2024",
        "url": "",
        "classification": classification,
        "source_quality": "primary_candidate",
        "reasons": [],
        "file_path": f"C:/Zotero/storage/{attachment_key}.pdf",
        "file_exists": file_exists,
        "file_size": 100 if file_exists else None,
        "file_mtime": 1.0 if file_exists else None,
        "metadata": {},
    }


def build_attachment_with_file(
    tmpdir: Path,
    attachment_key: str,
    *,
    title: str,
    classification: str = "included_auto",
) -> dict[str, object]:
    """Build an attachment dict whose file_path points to an existing file."""

    file_path = tmpdir / "attachments" / f"{attachment_key}.pdf"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"fake pdf content")
    attachment = build_attachment(
        attachment_key,
        title=title,
        classification=classification,
        file_exists=True,
    )
    attachment["file_path"] = str(file_path)
    return attachment


class MarkdownStubExtractorProvider(StubExtractorProvider):
    """Stub extractor that produces a markdown file suitable for normalization."""

    def download(
        self,
        external_job_id: str,
        output_dir: Path,
        *,
        api_key: str | None = None,
        source_pdf: str | Path | None = None,
        options: dict[str, object] | None = None,
    ) -> ExtractArtifact:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown = output_dir / "full.md"
        markdown.write_text("# Demo Paper\n\nalpha beta gamma.\n", encoding="utf-8")
        manifest = output_dir / "manifest.json"
        source_pdf_path = Path(source_pdf) if source_pdf is not None else Path(external_job_id)
        manifest.write_text(
            json.dumps(
                {
                    "provider": self.name,
                    "provider_version": self.version,
                    "external_job_id": external_job_id,
                    "state": "downloaded",
                    "source_pdf": str(source_pdf_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return ExtractArtifact(
            source_pdf=source_pdf_path,
            artifact_dir=output_dir,
            manifest_path=manifest,
        )


class TwoPhaseStubExtractorProvider(MarkdownStubExtractorProvider):
    """Stub extractor that returns 'running' on the first poll, then 'completed'."""

    name = "two-phase-stub"
    version = "0"

    def __init__(self) -> None:
        self._poll_count = 0

    def submit(
        self,
        input_file: Path,
        options_hash: str,
        *,
        options: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> ExtractJobState:
        return ExtractJobState(external_job_id=self.fingerprint(input_file, options_hash), state="running")

    def poll(
        self,
        external_job_id: str,
        *,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractJobState:
        self._poll_count += 1
        if self._poll_count == 1:
            return ExtractJobState(external_job_id=external_job_id, state="running")
        return ExtractJobState(external_job_id=external_job_id, state="completed")


if __name__ == "__main__":
    unittest.main()
