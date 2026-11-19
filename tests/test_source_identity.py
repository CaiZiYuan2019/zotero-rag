from __future__ import annotations

import unittest
from contextlib import closing

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.documents import get_document, list_documents
from zoterorag.search import fulltext_search, metadata_search


def artifact(document_id, attachment_key=None, **extra):
    return {
        "document_id": document_id,
        "attachment_key": attachment_key,
        "artifact_dir": "normalized",
        "document_md": "document.md",
        "chunks_path": "chunks.jsonl",
        "manifest_path": "manifest.json",
        "image_manifest": "images.json",
        "source_markdown": "full.md",
        "document_hash": document_id,
        "status": "normalized",
        **extra,
    }


class SourceIdentityTests(unittest.TestCase):
    def test_sibling_documents_round_trip_and_normalize_to_one_source(self):
        with workspace_tmpdir("identity-") as root, closing(StateLedger(root / "state.sqlite")) as ledger:
            ledger.upsert_attachments([
                {"attachment_key": key, "parent_key": "PARENT", "classification": "included_auto"}
                for key in ("ATT1", "ATT2", "META")
            ])
            # Explicit alias, old manifest-only link, and metadata-only sibling.
            ledger.upsert_normalized_artifact(artifact("DOC1", "ATT1"))
            ledger.upsert_normalized_artifact(artifact("DOC2", payload={"attachment_key": "ATT2"}))
            for doc in ("DOC1", "DOC2"):
                ledger.replace_document_chunks(doc, [{
                    "chunk_id": doc + "-chunk", "chunk_index": 0, "chunk_type": "text", "text": "evidence",
                }])

            expected = {"DOC1": "ATT1", "DOC2": "ATT2", "META": "META"}
            metadata = metadata_search(ledger, "", classification="included_auto")
            hits = fulltext_search(ledger, "evidence")
            documents = list_documents(ledger)
            self.assertEqual(set(expected), {r["document_id"] for r in metadata})
            self.assertEqual({"DOC1", "DOC2"}, {r["document_id"] for r in hits})
            self.assertCountEqual(expected, [r["document_id"] for r in documents])
            for result in metadata + hits + documents:
                doc = result["document_id"]
                attachment = expected[doc]
                detail = get_document(ledger, attachment, include_chunks=True)
                for row in [result, detail, *detail.get("chunks", [])]:
                    self.assertEqual((doc, attachment, "PARENT", "PARENT"), (
                        row["document_id"], row["attachment_key"], row["parent_item_key"], row["source_id"],
                    ))
                self.assertEqual(detail["document_id"], get_document(ledger, doc)["document_id"])
                if "metadata" in result:
                    self.assertEqual("PARENT", result["metadata"]["parent_key"])
                    self.assertEqual(attachment, result["metadata"]["attachment_key"])
            self.assertIsNone(get_document(ledger, "PARENT"))

    def test_multiple_artifacts_keep_exact_document_ids_and_select_a_stable_attachment_alias(self):
        with workspace_tmpdir("identity-") as root, closing(StateLedger(root / "state.sqlite")) as ledger:
            ledger.upsert_attachments([{"attachment_key": "ATT", "parent_key": "P", "classification": "included_auto"}])
            for key in ("ATT", "REVISION"):
                ledger.upsert_normalized_artifact(artifact(key, "ATT"))
            hit = metadata_search(ledger, "", classification="included_auto")[0]
            self.assertEqual("ATT", hit["document_id"])
            self.assertEqual("REVISION", get_document(ledger, "REVISION")["document_id"])

    def test_recovers_legacy_links_and_preserves_unlinked_ids(self):
        with workspace_tmpdir("identity-") as root, closing(StateLedger(root / "state.sqlite")) as ledger:
            ledger.upsert_attachments([
                {"attachment_key": "IMPLICIT", "classification": "included_auto"},
                {"attachment_key": "FROMJOB", "parent_key": "P", "classification": "included_auto"},
            ])
            ledger.upsert_extract_job({
                "job_id": "job", "attachment_key": "FROMJOB", "cache_key": "cache", "pdf_sha256": "hash",
                "provider": "test", "provider_version": "1", "options_hash": "", "state": "done", "local_stage": "done",
            })
            ledger.upsert_normalized_artifact(artifact("IMPLICIT"))
            ledger.upsert_normalized_artifact(artifact("JOBDOC", extract_job_id="job"))
            ledger.upsert_normalized_artifact(artifact("DETACHED", "UNSCANNED"))
            ledger.upsert_normalized_artifact(artifact("ORPHAN"))
            docs = {r["document_id"]: r for r in list_documents(ledger)}
            self.assertEqual({"IMPLICIT", "JOBDOC", "DETACHED", "ORPHAN"}, set(docs))
            self.assertEqual("IMPLICIT", docs["IMPLICIT"]["attachment_key"])
            self.assertIsNone(docs["IMPLICIT"]["parent_item_key"])
            self.assertIsNone(docs["ORPHAN"]["attachment_key"])
            self.assertEqual("JOBDOC", get_document(ledger, "P")["document_id"])
            self.assertEqual("DETACHED", get_document(ledger, "UNSCANNED")["document_id"])
