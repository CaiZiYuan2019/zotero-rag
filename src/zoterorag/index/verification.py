from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any

from ..db import StateLedger


@dataclass(frozen=True)
class VectorIndexVerification:
    profile_name: str
    ok: bool
    path: Path | None
    expected_dimension: int | None
    registered_documents: int | None = None
    registered_chunks: int | None = None
    actual_documents: int | None = None
    actual_chunks: int | None = None
    dimension_errors: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "ok": self.ok,
            "path": str(self.path) if self.path is not None else None,
            "expected_dimension": self.expected_dimension,
            "registered_documents": self.registered_documents,
            "registered_chunks": self.registered_chunks,
            "actual_documents": self.actual_documents,
            "actual_chunks": self.actual_chunks,
            "dimension_errors": self.dimension_errors,
            "errors": list(self.errors),
        }


def verify_vector_index(ledger: StateLedger, profile_name: str) -> VectorIndexVerification:
    profiles = {profile["name"]: profile for profile in ledger.list_embedding_profiles()}
    indexes = {index["profile_name"]: index for index in ledger.list_vector_indexes()}
    errors: list[str] = []
    profile = profiles.get(profile_name)
    index = indexes.get(profile_name)
    if profile is None:
        return VectorIndexVerification(
            profile_name=profile_name,
            ok=False,
            path=None,
            expected_dimension=None,
            errors=[f"profile_not_found:{profile_name}"],
        )
    expected_dimension = int(profile["dimension"])
    if index is None:
        return VectorIndexVerification(
            profile_name=profile_name,
            ok=False,
            path=None,
            expected_dimension=expected_dimension,
            errors=[f"vector_index_not_registered:{profile_name}"],
        )

    path = Path(index["path"])
    registered_documents = int(index["document_count"])
    registered_chunks = int(index["chunk_count"])
    if not path.is_file():
        return VectorIndexVerification(
            profile_name=profile_name,
            ok=False,
            path=path,
            expected_dimension=expected_dimension,
            registered_documents=registered_documents,
            registered_chunks=registered_chunks,
            errors=[f"missing_vector_store:{path}"],
        )

    actual_documents = 0
    actual_chunks = 0
    dimension_errors = 0
    # Open read-only so verification cannot mutate vector stores. This matters
    # because verify commands are safe to run while ingest jobs are paused or
    # before taking a backup.
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vectors'"
        ).fetchone()
        if table is None:
            errors.append("missing_vectors_table")
        else:
            row = conn.execute(
                """
                SELECT count(DISTINCT document_id) AS documents, count(*) AS chunks
                FROM vectors
                WHERE profile_name = ?
                """,
                (profile_name,),
            ).fetchone()
            actual_documents = int(row["documents"])
            actual_chunks = int(row["chunks"])
            for vector_row in conn.execute(
                "SELECT record_id, vector_json FROM vectors WHERE profile_name = ?",
                (profile_name,),
            ):
                try:
                    vector = json.loads(vector_row["vector_json"])
                except json.JSONDecodeError:
                    errors.append(f"invalid_vector_json:{vector_row['record_id']}")
                    dimension_errors += 1
                    continue
                if not isinstance(vector, list) or len(vector) != expected_dimension:
                    dimension_errors += 1
            if registered_documents != actual_documents:
                errors.append(f"document_count_mismatch:{registered_documents}!={actual_documents}")
            if registered_chunks != actual_chunks:
                errors.append(f"chunk_count_mismatch:{registered_chunks}!={actual_chunks}")
            if dimension_errors:
                errors.append(f"dimension_errors:{dimension_errors}")
    finally:
        conn.close()

    return VectorIndexVerification(
        profile_name=profile_name,
        ok=not errors,
        path=path,
        expected_dimension=expected_dimension,
        registered_documents=registered_documents,
        registered_chunks=registered_chunks,
        actual_documents=actual_documents,
        actual_chunks=actual_chunks,
        dimension_errors=dimension_errors,
        errors=errors,
    )
