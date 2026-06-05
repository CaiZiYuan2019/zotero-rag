from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.embeddings import index_normalized_document
from zoterorag.normalize import normalize_markdown_document
from zoterorag.pipeline import build_progress_report


class ProgressReportTests(unittest.TestCase):
    def test_progress_report_summarizes_local_build_state(self) -> None:
        with workspace_tmpdir("progress-report-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="stub_text",
                            provider="stub",
                            model="stub",
                            dimension=8,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                        )
                    ]
                )
                ledger.upsert_attachments(
                    [
                        {
                            "attachment_key": "ATT1",
                            "parent_key": "PARENT1",
                            "content_type": "application/pdf",
                            "relative_path": "storage:paper.pdf",
                            "title": "Progress Paper",
                            "abstract": "Progress test",
                            "date": "2026",
                            "url": None,
                            "classification": "included_auto",
                            "source_quality": "primary_candidate",
                            "reasons": ["pdf_with_parent"],
                            "file_path": str(tmpdir / "paper.pdf"),
                            "file_exists": True,
                            "file_size": 12,
                            "file_mtime": 1.0,
                            "metadata": {},
                        },
                        {
                            "attachment_key": "ATT_REVIEW",
                            "parent_key": "PARENT2",
                            "content_type": "application/pdf",
                            "relative_path": "storage:paper_zh-CN_dual.pdf",
                            "title": "Review Paper",
                            "abstract": "",
                            "date": "2026",
                            "url": None,
                            "classification": "needs_review",
                            "source_quality": "suspected_translation_pdf",
                            "reasons": ["translation_marker:_zh-cn_dual.pdf"],
                            "file_path": None,
                            "file_exists": False,
                            "file_size": None,
                            "file_mtime": None,
                            "metadata": {},
                        },
                    ]
                )
                source_dir = tmpdir / "mineru"
                images_dir = source_dir / "images"
                images_dir.mkdir(parents=True)
                (images_dir / "fig.png").write_bytes(b"fake-image")
                markdown = source_dir / "full.md"
                markdown.write_text("# Progress Paper\n\nalpha beta\n\n![figure](images/fig.png)\n", encoding="utf-8")
                normalized = normalize_markdown_document(
                    source_markdown=markdown,
                    output_root=tmpdir / "normalized",
                    document_id="DOC1",
                    attachment_key="ATT1",
                )
                ledger.upsert_normalized_artifact(normalized.ledger_artifact())
                ledger.replace_document_chunks(normalized.document_id, normalized.chunks)
                index_normalized_document(
                    ledger=ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    document_id="DOC1",
                )

                report = build_progress_report(ledger)

                self.assertEqual(2, report["library"]["attachment_count"])
                self.assertEqual(1, report["library"]["buildable_count"])
                self.assertEqual(1, report["library"]["review_count"])
                self.assertEqual(1, report["normalize"]["artifact_count"])
                self.assertEqual(1, report["embedding"]["batch_count"])
                self.assertEqual({"completed": 1}, report["embedding"]["batches_by_status"])
                self.assertEqual(1, report["embedding"]["indexed_chunk_count"])
                self.assertEqual(1, report["ingest_plan"]["document_count"])
                self.assertIn("complete", report["ingest_plan"]["next_stages"])
                self.assertFalse(report["eta"]["available"])
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
