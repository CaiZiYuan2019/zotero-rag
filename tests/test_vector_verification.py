from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.index import LocalVectorStore, VectorRecord, verify_vector_index


class VectorVerificationTests(unittest.TestCase):
    def test_verify_vector_index_checks_counts_and_dimensions(self) -> None:
        with workspace_tmpdir("vector-verify-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            vector_path = tmpdir / "vectors" / "profile-a" / "vectors.sqlite"
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="profile-a",
                            provider="stub",
                            model="stub",
                            dimension=3,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                        )
                    ]
                )
                store = LocalVectorStore(vector_path, profile_name="profile-a", dimension=3)
                try:
                    store.upsert(
                        [
                            VectorRecord(
                                record_id="r1",
                                document_id="d1",
                                chunk_id="c1",
                                vector=[1.0, 0.0, 0.0],
                                text="alpha",
                            )
                        ]
                    )
                finally:
                    store.close()
                ledger.register_vector_index(
                    profile_name="profile-a",
                    backend="sqlite-local",
                    path=vector_path,
                    document_count=1,
                    chunk_count=1,
                    active=True,
                )
                ok = verify_vector_index(ledger, "profile-a")
                self.assertTrue(ok.ok)
                self.assertEqual(1, ok.actual_chunks)

                ledger.register_vector_index(
                    profile_name="profile-a",
                    backend="sqlite-local",
                    path=vector_path,
                    document_count=9,
                    chunk_count=9,
                    active=True,
                )
                bad = verify_vector_index(ledger, "profile-a")
                self.assertFalse(bad.ok)
                self.assertIn("chunk_count_mismatch:9!=1", bad.errors)
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
