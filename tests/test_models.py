from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.embeddings import embedding_profile_hash
from zoterorag.models import list_embedding_model_catalog


class EmbeddingModelCatalogTests(unittest.TestCase):
    def test_catalog_joins_profiles_with_vector_index_state(self) -> None:
        with workspace_tmpdir("models-catalog-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            vector_path = tmpdir / "vectors" / "text-profile" / "vectors.sqlite"
            vector_path.parent.mkdir(parents=True)
            vector_path.touch()
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="text-profile",
                            provider="stub",
                            model="stub-text",
                            dimension=8,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                            query_role_mode="instruction",
                            document_role_mode="plain",
                            instruction_template="retrieve papers",
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
                ledger.register_vector_index(
                    profile_name="text-profile",
                            backend="sqlite-local",
                    path=vector_path,
                    document_count=3,
                    chunk_count=11,
                    active=True,
                    active_version="batch-live",
                )

                catalog = list_embedding_model_catalog(ledger)
                by_name = {item["name"]: item for item in catalog["models"]}

                self.assertEqual({"text": "text-profile", "multimodal": "mm-profile"}, catalog["defaults"])
                self.assertEqual(1, len(catalog["vector_indexes"]))
                self.assertEqual("ready", by_name["text-profile"]["index_status"])
                self.assertEqual(["text"], by_name["text-profile"]["query_modes"])
                self.assertEqual(11, by_name["text-profile"]["vector_index"]["chunk_count"])
                self.assertEqual("batch-live", by_name["text-profile"]["vector_index"]["active_version"])
                self.assertTrue(by_name["text-profile"]["vector_index"]["path_exists"])
                self.assertEqual("not_indexed", by_name["mm-profile"]["index_status"])
                self.assertEqual(["multimodal"], by_name["mm-profile"]["query_modes"])
                self.assertEqual(
                    embedding_profile_hash(next(p for p in ledger.list_embedding_profiles() if p["name"] == "text-profile")),
                    by_name["text-profile"]["profile_hash"],
                )
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
