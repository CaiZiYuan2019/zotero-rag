from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from .local_vector import VectorRecord, cosine_similarity


logger = logging.getLogger(__name__)


class LanceDBVectorStore:
    """LanceDB-backed vector store implementing the same interface as
    :class:`LocalVectorStore`.

    Each profile stores vectors in a LanceDB database directory. Versioned
    publishing uses separate LanceDB tables (``vectors_{version}``) with an
    atomic ``vector_meta`` table to switch the active version.
    """

    def __init__(self, path: str | Path, profile_name: str, dimension: int) -> None:
        self.path = Path(path)
        self.profile_name = profile_name
        self.dimension = dimension
        self.path.mkdir(parents=True, exist_ok=True)

        lancedb = _import_lancedb()
        self._db = lancedb.connect(str(self.path))
        self._table_cache: dict[str, Any] = {}

    def close(self) -> None:
        self._table_cache.clear()
        # LanceDB handles cleanup automatically; no explicit close needed.

    def _delete_records(self, table: Any, record_ids: list[str]) -> None:
        """Delete records by record_id using a literal-safe predicate."""
        predicate = _build_record_id_in_predicate(record_ids)
        table.delete(predicate)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert(self, records: Iterable[VectorRecord], *, index_version: str = "legacy") -> int:
        table_name = _table_name(index_version)
        rows = [_record_to_row(record, index_version, self.profile_name) for record in records]
        if not rows:
            return 0
        for record in records:
            if len(record.vector) != self.dimension:
                raise ValueError(
                    f"record {record.record_id} has dimension {len(record.vector)}, "
                    f"expected {self.dimension}"
                )
        table = self._ensure_table(table_name, rows)
        # LanceDB merge_insert uses record_id as the primary merge key when
        # the schema declares it. For simplicity we delete-then-add within
        # the same version table.
        record_ids = [r["record_id"] for r in rows]
        self._delete_records(table, record_ids)
        table.add(rows)
        self._table_cache[table_name] = table
        return len(rows)

    def copy_active_records_to_version(
        self,
        *,
        index_version: str,
        exclude_document_ids: Iterable[str] = (),
    ) -> int:
        """Stage existing active records under a new version."""
        active_version = self.active_version()
        if not active_version:
            return 0
        try:
            active_table = self._db.open_table(_table_name(active_version))
        except Exception as exc:
            if _is_table_missing(exc):
                logger.warning(
                    "active table %s missing for profile %s, nothing to copy",
                    active_version,
                    self.profile_name,
                )
                return 0
            raise
        excluded = set(exclude_document_ids)
        all_rows = active_table.to_pandas().to_dict("records")
        rows_to_copy = [
            _record_to_row(
                VectorRecord(
                    record_id=_strip_version_prefix(
                        str(row.get("record_id", "")), active_version
                    ),
                    document_id=str(row.get("document_id", "")),
                    chunk_id=str(row.get("chunk_id", "")),
                    vector=_parse_vector_json(row.get("vector_json", "[]")),
                    text=str(row.get("text", "")),
                    modality=str(row.get("modality", "text")),
                    metadata=_parse_metadata_json(row.get("metadata_json", "{}")),
                ),
                index_version,
                self.profile_name,
            )
            for row in all_rows
            if row.get("document_id") not in excluded
        ]
        if not rows_to_copy:
            return 0
        table = self._ensure_table(_table_name(index_version), rows_to_copy)
        table.add(rows_to_copy)
        self._table_cache[_table_name(index_version)] = table
        return len(rows_to_copy)

    def publish_version(self, index_version: str) -> None:
        """Atomically switch to a completed version."""
        self._ensure_meta_table()
        meta_table = self._db.open_table("vector_meta")
        # Delete old row for this profile, then add the new one.
        predicate = _build_profile_name_predicate(self.profile_name)
        meta_table.delete(predicate)
        meta_table.add([{"profile_name": self.profile_name, "active_version": index_version}])

    def active_version(self) -> str:
        try:
            meta_table = self._db.open_table("vector_meta")
            rows = meta_table.to_pandas().to_dict("records")
        except Exception as exc:
            if _is_table_missing(exc):
                logger.warning(
                    "vector_meta table missing for profile %s, assuming legacy",
                    self.profile_name,
                )
                return "legacy"
            raise
        for row in rows:
            if row.get("profile_name") == self.profile_name:
                return str(row.get("active_version", "legacy"))
        return "legacy"

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

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
        active_version = self.active_version()
        table_name = _table_name(active_version)
        try:
            table = self._db.open_table(table_name)
        except Exception as exc:
            if _is_table_missing(exc):
                logger.warning(
                    "search table %s missing for profile %s",
                    table_name,
                    self.profile_name,
                )
                return []
            raise

        # LanceDB native vector search when a vector column is configured.
        # We fall back to brute-force cosine similarity for portability.
        all_rows = table.to_pandas().to_dict("records")

        scored = []
        for row in all_rows:
            vector = _parse_vector_json(row.get("vector_json", "[]"))
            if not vector:
                continue
            if modality is not None and row.get("modality") != modality:
                continue
            score = cosine_similarity(query_vector, vector)
            scored.append(
                {
                    "record_id": row.get("record_id", ""),
                    "document_id": row.get("document_id", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "modality": row.get("modality", "text"),
                    "text": row.get("text", ""),
                    "metadata": _parse_metadata_json(row.get("metadata_json", "{}")),
                    "score": score,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def counts(self, *, index_version: str | None = None) -> dict[str, int]:
        version = index_version or self.active_version()
        try:
            table = self._db.open_table(_table_name(version))
            rows = table.to_pandas().to_dict("records")
        except Exception as exc:
            if _is_table_missing(exc):
                logger.warning(
                    "counts table %s missing for profile %s",
                    _table_name(version),
                    self.profile_name,
                )
                return {"documents": 0, "chunks": 0}
            raise
        doc_ids = {r.get("document_id") for r in rows}
        return {"documents": len(doc_ids), "chunks": len(rows)}

    def document_counts(
        self,
        document_id: str,
        *,
        modality: str | None = None,
        index_version: str | None = None,
    ) -> dict[str, int]:
        version = index_version or self.active_version()
        try:
            table = self._db.open_table(_table_name(version))
            rows = table.to_pandas().to_dict("records")
        except Exception as exc:
            if _is_table_missing(exc):
                logger.warning(
                    "document_counts table %s missing for profile %s",
                    _table_name(version),
                    self.profile_name,
                )
                return {"chunks": 0}
            raise
        count = 0
        for row in rows:
            if row.get("document_id") != document_id:
                continue
            if modality is not None and row.get("modality") != modality:
                continue
            count += 1
        return {"chunks": count}

    def document_metadata_values(
        self,
        document_id: str,
        key: str,
        *,
        modality: str | None = None,
        index_version: str | None = None,
    ) -> set[Any]:
        version = index_version or self.active_version()
        try:
            table = self._db.open_table(_table_name(version))
            rows = table.to_pandas().to_dict("records")
        except Exception as exc:
            if _is_table_missing(exc):
                logger.warning(
                    "document_metadata_values table %s missing for profile %s",
                    _table_name(version),
                    self.profile_name,
                )
                return set()
            raise
        values: set[Any] = set()
        for row in rows:
            if row.get("document_id") != document_id:
                continue
            if modality is not None and row.get("modality") != modality:
                continue
            metadata = _parse_metadata_json(row.get("metadata_json", "{}"))
            if key in metadata:
                values.add(metadata[key])
        return values

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_table(self, table_name: str, sample_rows: list[dict[str, Any]]) -> Any:
        if table_name in self._table_cache:
            return self._table_cache[table_name]
        try:
            table = self._db.open_table(table_name)
        except Exception as exc:
            if not _is_table_missing(exc):
                raise
            import pyarrow as pa

            schema = pa.schema([
                pa.field("record_id", pa.string()),
                pa.field("profile_name", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("chunk_id", pa.string()),
                pa.field("modality", pa.string()),
                pa.field("index_version", pa.string()),
                pa.field("vector_json", pa.string()),
                pa.field("text", pa.string()),
                pa.field("metadata_json", pa.string()),
            ])
            # Create empty table with schema, then add data.
            table = self._db.create_table(table_name, schema=schema)
        self._table_cache[table_name] = table
        return table

    def _ensure_meta_table(self) -> Any:
        try:
            return self._db.open_table("vector_meta")
        except Exception as exc:
            if not _is_table_missing(exc):
                raise
            import pyarrow as pa

            schema = pa.schema([
                pa.field("profile_name", pa.string()),
                pa.field("active_version", pa.string()),
            ])
            table = self._db.create_table("vector_meta", schema=schema)
            return table


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _table_name(index_version: str) -> str:
    """LanceDB table names cannot start with a digit or contain hyphens."""
    safe = str(index_version).replace("-", "_").replace(".", "_")
    if safe and safe[0].isdigit():
        safe = "v_" + safe
    return f"vectors_{safe}" if safe else "vectors_legacy"


def _record_to_row(record: VectorRecord, index_version: str, profile_name: str) -> dict[str, Any]:
    stored_record_id = (
        record.record_id
        if index_version == "legacy"
        else f"{index_version}:{record.record_id}"
    )
    return {
        "record_id": stored_record_id,
        "profile_name": profile_name,
        "document_id": record.document_id,
        "chunk_id": record.chunk_id,
        "modality": record.modality,
        "index_version": index_version,
        "vector_json": json.dumps(record.vector),
        "text": record.text,
        "metadata_json": json.dumps(record.metadata or {}, ensure_ascii=False, sort_keys=True),
    }


def _parse_vector_json(raw: str | None) -> list[float]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return [float(v) for v in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _parse_metadata_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _strip_version_prefix(stored_record_id: str, index_version: str) -> str:
    if index_version and index_version != "legacy":
        prefix = f"{index_version}:"
        if stored_record_id.startswith(prefix):
            return stored_record_id[len(prefix):]
    return stored_record_id


def _import_lancedb():
    """Lazy-import lancedb so the store is optional at import time."""
    try:
        import lancedb
    except ImportError as exc:
        raise ImportError(
            "lancedb is required for LanceDBVectorStore; install with: pip install lancedb"
        ) from exc
    return lancedb


def _is_table_missing(exc: BaseException) -> bool:
    """Heuristic to detect "table does not exist" without importing lancedb."""
    if isinstance(exc, FileNotFoundError):
        return True
    name = type(exc).__name__.lower()
    if "notfound" in name or "missing" in name:
        return True
    msg = str(exc).lower()
    indicators = (
        "not found",
        "does not exist",
        "doesn't exist",
        "table not found",
        "no such table",
        "no table named",
        "table was not found",
    )
    return any(indicator in msg for indicator in indicators)


# ---------------------------------------------------------------------------
# Safe predicate builders
# ---------------------------------------------------------------------------

# Only allow characters that cannot break a LanceDB SQL literal or identifier.
# record_ids are of the form "<profile_name>:<chunk_id>" and may contain
# letters, digits, hyphens, underscores, colons, and dots.
_SAFE_LITERAL_RE = re.compile(r"^[A-Za-z0-9_.:\-]+$")


def _validate_predicate_literal(value: str, name: str) -> str:
    if not _SAFE_LITERAL_RE.match(value):
        raise ValueError(f"unsafe characters in {name}: {value!r}")
    return value


def _build_record_id_in_predicate(record_ids: list[str]) -> str:
    if not record_ids:
        raise ValueError("record_ids must not be empty")
    safe_ids = [_validate_predicate_literal(rid, "record_id") for rid in record_ids]
    literals = ", ".join(f"'{rid}'" for rid in safe_ids)
    return f"record_id IN ({literals})"


def _build_profile_name_predicate(profile_name: str) -> str:
    safe = _validate_predicate_literal(profile_name, "profile_name")
    return f"profile_name = '{safe}'"
