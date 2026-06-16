from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile
from zoterorag.db import StateLedger
from zoterorag.index.local_vector import LocalVectorStore, VectorRecord
from zoterorag.index.verification import verify_vector_index


class VectorVerificationTests(unittest.TestCase):
    def _seed_profile_and_index(self, ledger: StateLedger, tmpdir: Path) -> None:
        ledger.upsert_embedding_profiles(
            [
                EmbeddingProfile(
                    name="p1",
                    provider="stub",
                    model="stub",
                    dimension=3,
                    modality="text",
                    enabled=True,
                    default_for_text=True,
                )
            ]
        )
        store = LocalVectorStore(tmpdir / "vectors.sqlite", profile_name="p1", dimension=3)
        try:
            store.upsert(
                [
                    VectorRecord(
                        record_id="p1:c1",
                        document_id="DOC1",
                        chunk_id="c1",
                        vector=[1.0, 0.0, 0.0],
                        text="text",
                        modality="text",
                    ),
                    VectorRecord(
                        record_id="p1:c2",
                        document_id="DOC1",
                        chunk_id="c2",
                        vector=[0.0, 1.0, 0.0],
                        text="text2",
                        modality="text",
                    ),
                    VectorRecord(
                        record_id="p1:c3",
                        document_id="DOC2",
                        chunk_id="c3",
                        vector=[0.0, 0.0, 1.0],
                        text="text3",
                        modality="text",
                    ),
                ]
            )
            store.publish_version("legacy")
        finally:
            store.close()

    def test_verify_sqlite_happy_path(self) -> None:
        with workspace_tmpdir("verify-sqlite-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                self._seed_profile_and_index(ledger, tmpdir)
                ledger.register_vector_index(
                    profile_name="p1",
                            backend="sqlite-local",
                    path=tmpdir / "vectors.sqlite",
                    document_count=2,
                    chunk_count=3,
                    active=True,
                    active_version="legacy",
                )
                result = verify_vector_index(ledger, "p1")
                self.assertTrue(result.ok)
                self.assertEqual(2, result.actual_documents)
                self.assertEqual(3, result.actual_chunks)
            finally:
                ledger.close()

    def test_verify_sqlite_missing_file(self) -> None:
        with workspace_tmpdir("verify-missing-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="p1",
                            provider="stub",
                            model="stub",
                            dimension=3,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                        )
                    ]
                )
                ledger.register_vector_index(
                    profile_name="p1",
                            backend="sqlite-local",
                    path=tmpdir / "vectors.sqlite",
                    document_count=0,
                    chunk_count=0,
                    active=True,
                    active_version="legacy",
                )
                result = verify_vector_index(ledger, "p1")
                self.assertFalse(result.ok)
                self.assertTrue(
                    any("missing_vector_store" in error for error in result.errors)
                )
            finally:
                ledger.close()

    def test_verify_sqlite_corrupt_vector(self) -> None:
        with workspace_tmpdir("verify-corrupt-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                self._seed_profile_and_index(ledger, tmpdir)
                # Corrupt one vector's JSON.
                import sqlite3

                conn = sqlite3.connect(tmpdir / "vectors.sqlite")
                try:
                    conn.execute(
                        "UPDATE vectors SET vector_json = ? WHERE record_id = ?",
                        ("not json", "p1:c1"),
                    )
                    conn.commit()
                finally:
                    conn.close()

                ledger.register_vector_index(
                    profile_name="p1",
                            backend="sqlite-local",
                    path=tmpdir / "vectors.sqlite",
                    document_count=2,
                    chunk_count=3,
                    active=True,
                    active_version="legacy",
                )
                result = verify_vector_index(ledger, "p1")
                self.assertFalse(result.ok)
                self.assertTrue(
                    any("dimension_errors" in error for error in result.errors),
                    result.errors,
                )
            finally:
                ledger.close()

    def test_verify_lancedb_missing_directory(self) -> None:
        with workspace_tmpdir("verify-lancedb-missing-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="p1",
                            provider="stub",
                            model="stub",
                            dimension=3,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                            backend="sqlite-local",
                        )
                    ]
                )
                ledger.register_vector_index(
                    profile_name="p1",
                    backend="lancedb",
                    path=tmpdir / "lancedb_vectors",
                    document_count=0,
                    chunk_count=0,
                    active=True,
                    active_version="legacy",
                )
                result = verify_vector_index(ledger, "p1")
                self.assertFalse(result.ok)
                self.assertTrue(
                    any("missing_vector_store" in error for error in result.errors)
                )
            finally:
                ledger.close()


    def test_verify_lancedb_happy_path(self) -> None:
        with workspace_tmpdir("verify-lancedb-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="p1",
                            provider="stub",
                            model="stub",
                            dimension=3,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                            backend="sqlite-local",
                        )
                    ]
                )
                ledger.register_vector_index(
                    profile_name="p1",
                    backend="lancedb",
                    path=tmpdir / "lancedb_vectors",
                    document_count=2,
                    chunk_count=3,
                    active=True,
                    active_version="v1",
                )

                rows = [
                    {
                        "record_id": "p1:c1",
                        "profile_name": "p1",
                        "document_id": "DOC1",
                        "chunk_id": "c1",
                        "modality": "text",
                        "index_version": "v1",
                        "vector_json": json.dumps([1.0, 0.0, 0.0]),
                        "text": "t1",
                        "metadata_json": "{}",
                    },
                    {
                        "record_id": "p1:c2",
                        "profile_name": "p1",
                        "document_id": "DOC1",
                        "chunk_id": "c2",
                        "modality": "text",
                        "index_version": "v1",
                        "vector_json": json.dumps([0.0, 1.0, 0.0]),
                        "text": "t2",
                        "metadata_json": "{}",
                    },
                    {
                        "record_id": "p1:c3",
                        "profile_name": "p1",
                        "document_id": "DOC2",
                        "chunk_id": "c3",
                        "modality": "text",
                        "index_version": "v1",
                        "vector_json": json.dumps([0.0, 0.0, 1.0]),
                        "text": "t3",
                        "metadata_json": "{}",
                    },
                ]
                mock_table = MagicMock()
                mock_arrow = MagicMock()
                mock_arrow.to_pylist.return_value = rows
                mock_table.to_arrow.return_value = mock_arrow
                mock_db = MagicMock()
                mock_db.open_table.return_value = mock_table
                mock_lancedb = MagicMock()
                mock_lancedb.connect.return_value = mock_db

                (tmpdir / "lancedb_vectors").mkdir(parents=True, exist_ok=True)
                with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
                    result = verify_vector_index(ledger, "p1")

                self.assertTrue(result.ok)
                self.assertEqual(2, result.actual_documents)
                self.assertEqual(3, result.actual_chunks)
                mock_db.open_table.assert_called_once_with("vectors_v1")
            finally:
                ledger.close()

    def test_verify_lancedb_dimension_errors(self) -> None:
        with workspace_tmpdir("verify-lancedb-dim-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="p1",
                            provider="stub",
                            model="stub",
                            dimension=3,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                            backend="sqlite-local",
                        )
                    ]
                )
                ledger.register_vector_index(
                    profile_name="p1",
                    backend="lancedb",
                    path=tmpdir / "lancedb_vectors",
                    document_count=1,
                    chunk_count=2,
                    active=True,
                    active_version="v1",
                )

                rows = [
                    {
                        "record_id": "p1:c1",
                        "profile_name": "p1",
                        "document_id": "DOC1",
                        "chunk_id": "c1",
                        "modality": "text",
                        "index_version": "v1",
                        "vector_json": json.dumps([1.0, 0.0]),
                        "text": "t1",
                        "metadata_json": "{}",
                    },
                    {
                        "record_id": "p1:c2",
                        "profile_name": "p1",
                        "document_id": "DOC1",
                        "chunk_id": "c2",
                        "modality": "text",
                        "index_version": "v1",
                        "vector_json": "not json",
                        "text": "t2",
                        "metadata_json": "{}",
                    },
                ]
                mock_table = MagicMock()
                mock_arrow = MagicMock()
                mock_arrow.to_pylist.return_value = rows
                mock_table.to_arrow.return_value = mock_arrow
                mock_db = MagicMock()
                mock_db.open_table.return_value = mock_table
                mock_lancedb = MagicMock()
                mock_lancedb.connect.return_value = mock_db

                (tmpdir / "lancedb_vectors").mkdir(parents=True, exist_ok=True)
                with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
                    result = verify_vector_index(ledger, "p1")

                self.assertFalse(result.ok)
                self.assertTrue(
                    any("dimension_errors" in error for error in result.errors),
                    result.errors,
                )
            finally:
                ledger.close()
