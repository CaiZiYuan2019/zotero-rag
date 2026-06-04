from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
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
                    {"chunks": 1},
                )
                ledger.checkpoint(
                    normalized.document_id,
                    "embed:mm-profile",
                    "indexed",
                    {"chunks": 1},
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

    def test_execute_true_is_explicitly_rejected_until_workers_exist(self) -> None:
        with workspace_tmpdir("ingest-execute-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                seed_attachments(ledger)
                with self.assertRaises(NotImplementedError):
                    start_ingest_job(ledger, execute=True)
                self.assertEqual([], ledger.list_jobs(kind="ingest"))
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
            ),
            EmbeddingProfile(
                name="mm-profile",
                provider="stub",
                model="stub-mm",
                dimension=8,
                modality="multimodal",
                enabled=True,
                default_for_multimodal=True,
            ),
        ]
    )


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


if __name__ == "__main__":
    unittest.main()
