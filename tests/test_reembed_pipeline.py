from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.normalize import normalize_markdown_document
from zoterorag.pipeline import create_reembed_plan, start_reembed_job


class ReembedPipelineTests(unittest.TestCase):
    def test_reembed_from_normalized_indexes_pending_and_skips_up_to_date(self) -> None:
        with workspace_tmpdir("reembed-pipeline-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                seed_profiles(ledger)
                normalized = seed_normalized_document(tmpdir, ledger)

                plan = create_reembed_plan(ledger, profile_name="stub_text")
                self.assertEqual({"pending": 1}, plan["summary"]["status_counts"])
                self.assertEqual("not_indexed", plan["documents"][0]["reason"])

                executed = start_reembed_job(
                    ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    execute=True,
                )
                self.assertEqual("completed", executed["job"]["status"])
                self.assertEqual(1, len(executed["indexed"]))
                checkpoint = ledger.get_checkpoint(normalized.document_id, "embed:stub_text")
                self.assertEqual("indexed", checkpoint["status"])
                self.assertEqual(plan["profile_hash"], checkpoint["payload"]["profile_hash"])

                second = start_reembed_job(
                    ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="stub_text",
                    execute=True,
                )
                self.assertEqual(0, len(second["indexed"]))
                self.assertEqual(1, len(second["skipped"]))
                self.assertEqual("done", second["plan"]["documents"][0]["status"])
                self.assertEqual("up_to_date", second["plan"]["documents"][0]["reason"])
            finally:
                ledger.close()

    def test_reembed_execute_rejects_non_stub_provider_without_explicit_override(self) -> None:
        with workspace_tmpdir("reembed-pipeline-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="qwen-text",
                            provider="dashscope",
                            model="qwen3-vl-embedding",
                            dimension=8,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                        )
                    ]
                )
                seed_normalized_document(tmpdir, ledger)

                with self.assertRaises(NotImplementedError):
                    start_reembed_job(
                        ledger,
                        vector_store_dir=tmpdir / "vectors",
                        profile_name="qwen-text",
                        execute=True,
                    )

                dry_run = start_reembed_job(
                    ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="qwen-text",
                    execute=False,
                )
                self.assertEqual("planned", dry_run["job"]["status"])

                executed = start_reembed_job(
                    ledger,
                    vector_store_dir=tmpdir / "vectors",
                    profile_name="qwen-text",
                    execute=True,
                    allow_stub_provider=True,
                )
                self.assertEqual("completed", executed["job"]["status"])
                self.assertEqual(1, len(executed["indexed"]))
            finally:
                ledger.close()


def seed_profiles(ledger: StateLedger) -> None:
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


def seed_normalized_document(tmpdir, ledger: StateLedger):
    source_dir = tmpdir / "mineru"
    images_dir = source_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "fig.png").write_bytes(b"fake-image")
    markdown = source_dir / "full.md"
    markdown.write_text(
        "# Demo Paper\n\n"
        "alpha beta gamma text evidence\n\n"
        "![important figure](images/fig.png)\n",
        encoding="utf-8",
    )
    normalized = normalize_markdown_document(
        source_markdown=markdown,
        output_root=tmpdir / "normalized",
        document_id="DOC1",
        attachment_key="ATT1",
    )
    ledger.upsert_normalized_artifact(normalized.ledger_artifact())
    ledger.replace_document_chunks(normalized.document_id, normalized.chunks)
    return normalized


if __name__ == "__main__":
    unittest.main()
