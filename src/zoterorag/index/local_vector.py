from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    document_id: str
    chunk_id: str
    vector: list[float]
    text: str
    modality: str = "text"
    metadata: dict[str, Any] | None = None


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class LocalVectorStore:
    """Small SQLite vector store used for local tests and bootstrap.

    Production profiles can later point at LanceDB without changing search
    interfaces. This backend keeps the project runnable before optional vector
    dependencies are installed.
    """

    def __init__(self, path: str | Path, profile_name: str, dimension: int) -> None:
        self.path = Path(path)
        self.profile_name = profile_name
        self.dimension = dimension
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    def _migrate(self) -> None:
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    record_id TEXT PRIMARY KEY,
                    profile_name TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectors_profile ON vectors(profile_name, modality)"
            )

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        count = 0
        with self.conn:
            for record in records:
                if len(record.vector) != self.dimension:
                    raise ValueError(
                        f"record {record.record_id} has dimension {len(record.vector)}, expected {self.dimension}"
                    )
                self.conn.execute(
                    """
                    INSERT INTO vectors(
                        record_id, profile_name, document_id, chunk_id, modality,
                        vector_json, text, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(record_id) DO UPDATE SET
                        profile_name = excluded.profile_name,
                        document_id = excluded.document_id,
                        chunk_id = excluded.chunk_id,
                        modality = excluded.modality,
                        vector_json = excluded.vector_json,
                        text = excluded.text,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        record.record_id,
                        self.profile_name,
                        record.document_id,
                        record.chunk_id,
                        record.modality,
                        json.dumps(record.vector),
                        record.text,
                        json.dumps(record.metadata or {}, ensure_ascii=False, sort_keys=True),
                    ),
                )
                count += 1
        return count

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        modality: str | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"query vector has dimension {len(query_vector)}, expected {self.dimension}"
            )
        if modality is None:
            rows = self.conn.execute(
                "SELECT * FROM vectors WHERE profile_name = ?",
                (self.profile_name,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM vectors WHERE profile_name = ? AND modality = ?",
                (self.profile_name, modality),
            ).fetchall()

        scored = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            score = cosine_similarity(query_vector, vector)
            scored.append(
                {
                    "record_id": row["record_id"],
                    "document_id": row["document_id"],
                    "chunk_id": row["chunk_id"],
                    "modality": row["modality"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata_json"]),
                    "score": score,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def counts(self) -> dict[str, int]:
        row = self.conn.execute(
            """
            SELECT count(DISTINCT document_id) AS documents, count(*) AS chunks
            FROM vectors
            WHERE profile_name = ?
            """,
            (self.profile_name,),
        ).fetchone()
        return {"documents": int(row["documents"]), "chunks": int(row["chunks"])}

