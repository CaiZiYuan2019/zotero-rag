from __future__ import annotations

import json
import logging
import re
import threading
import warnings
from pathlib import Path
from typing import Any, Iterable

from .local_vector import VectorRecord, cosine_similarity


logger = logging.getLogger(__name__)

# Module-level cache of ``lancedb.DB`` connections keyed by absolute path.
#
# LanceDB's synchronous ``connect()`` can deadlock when called from a worker
# thread while the main thread runs an asyncio event loop (e.g. the MCP
# stdio server). Caching the connection on the main thread and reusing it
# across threads avoids that initialization path without changing the public
# store interface.
_DB_CACHE: dict[str, Any] = {}
_DB_CACHE_LOCK = threading.Lock()


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
        cache_key = str(self.path.resolve())
        with _DB_CACHE_LOCK:
            db = _DB_CACHE.get(cache_key)
            if db is None:
                db = lancedb.connect(str(self.path))
                _DB_CACHE[cache_key] = db
        self._db = db
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

    def upsert(
        self, records: Iterable[VectorRecord], *, index_version: str = "legacy"
    ) -> int:
        table_name = _table_name(index_version)
        rows = [
            _record_to_row(record, index_version, self.profile_name)
            for record in records
        ]
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
        # H1: the staged version name is deterministic (content batch hash), so
        # a retry after a crash between staging and publish always reuses the
        # same name. This guard must run before every early return below:
        # appending carry-forward rows to a leftover staged table would double
        # them on every retry, and the leftover rows would survive the later
        # upsert's delete-then-add (which only deletes the new document's ids).
        staged_table_name = _table_name(index_version)
        if staged_table_name in _all_table_names(self._db):
            # Compare against the active snapshot already fetched above, in
            # table-name space (``_table_name`` sanitization is irreversible).
            if staged_table_name == _table_name(active_version):
                # Re-staging the already-active version (identical content
                # hash) would append the active rows back onto the active
                # table; there is nothing to stage.
                return 0
            # Drop strictly: a failed drop must abort the run rather than let
            # _ensure_table reuse the stale table (or a cached handle to it)
            # and silently duplicate rows.
            self._table_cache.pop(staged_table_name, None)
            try:
                self._db.drop_table(staged_table_name)
            except Exception as exc:
                if not _is_table_missing(exc):
                    raise
        if not active_version:
            return 0
        try:
            active_table = self._db.open_table(_table_name(active_version))
        except Exception as exc:
            # Broad because LanceDB raises varying exception types for missing tables.
            if _is_table_missing(exc):
                logger.warning(
                    "active table %s missing for profile %s, nothing to copy",
                    active_version,
                    self.profile_name,
                )
                return 0
            raise
        excluded = set(exclude_document_ids)
        all_rows = active_table.to_arrow().to_pylist()
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
        """Atomically switch to a completed version.

        The superseded version's table is dropped after the switch: old
        versions are never read once inactive, and keeping every staged
        ``vectors_{batch_hash}`` table made disk usage grow without bound.
        """
        previous = self.active_version()
        self._ensure_meta_table()
        meta_table = self._db.open_table("vector_meta")
        # Delete old row for this profile, then add the new one.
        predicate = _build_profile_name_predicate(self.profile_name)
        meta_table.delete(predicate)
        meta_table.add(
            [{"profile_name": self.profile_name, "active_version": index_version}]
        )
        # Guard: re-publishing the already-active version (e.g. a
        # crash-recovery re-embed with the same batch hash) must not drop
        # the active table itself.
        if previous != index_version:
            self._drop_version_table(previous)

    def _drop_version_table(self, index_version: str) -> None:
        """Drop a superseded version table; failures only leave an orphan."""
        table_name = _table_name(index_version)
        try:
            self._db.drop_table(table_name)
        except Exception as exc:
            if _is_table_missing(exc):
                return
            # Cleanup failure must not fail the indexing run; the orphan
            # table can be reclaimed later via `vectors prune`.
            logger.warning(
                "failed to drop superseded vector table %s: %s", table_name, exc
            )
            return
        self._table_cache.pop(table_name, None)

    def list_versions(self) -> list[str]:
        """Version names derived from existing ``vectors_*`` tables.

        Names are table-name derivatives: ``_table_name`` is not reversible
        (digit-leading versions get a ``v_`` prefix), so compare versions
        through :meth:`version_is_active` rather than raw string equality.
        """

        return sorted(
            name[len("vectors_") :]
            for name in _all_table_names(self._db)
            if name.startswith("vectors_")
        )

    def version_is_active(self, index_version: str) -> bool:
        """Whether `index_version` maps to the currently active table.

        Compares in table-name space because ``_table_name`` sanitizes
        version names irreversibly (e.g. digit-leading batch hashes).
        """

        return _table_name(index_version) == _table_name(self.active_version())

    def delete_version(self, index_version: str) -> int:
        """Drop the table of a non-active version (used by `vectors prune`)."""

        if self.version_is_active(index_version):
            raise ValueError(f"cannot delete the active version {index_version!r}")
        table_name = _table_name(index_version)
        if table_name not in _all_table_names(self._db):
            return 0
        self._db.drop_table(table_name)
        self._table_cache.pop(table_name, None)
        return 1

    def active_version(self) -> str:
        try:
            meta_table = self._db.open_table("vector_meta")
            rows = meta_table.to_arrow().to_pylist()
        except Exception as exc:
            # Broad because LanceDB raises varying exception types for missing tables.
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
            # Broad because LanceDB raises varying exception types for missing tables.
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
        all_rows = table.to_arrow().to_pylist()

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
            rows = table.to_arrow().to_pylist()
        except Exception as exc:
            # Broad because LanceDB raises varying exception types for missing tables.
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
            rows = table.to_arrow().to_pylist()
        except Exception as exc:
            # Broad because LanceDB raises varying exception types for missing tables.
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
            rows = table.to_arrow().to_pylist()
        except Exception as exc:
            # Broad because LanceDB raises varying exception types for missing tables.
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
            # Broad because LanceDB raises varying exception types for missing tables.
            if not _is_table_missing(exc):
                raise
            import pyarrow as pa

            schema = pa.schema(
                [
                    pa.field("record_id", pa.string()),
                    pa.field("profile_name", pa.string()),
                    pa.field("document_id", pa.string()),
                    pa.field("chunk_id", pa.string()),
                    pa.field("modality", pa.string()),
                    pa.field("index_version", pa.string()),
                    pa.field("vector_json", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("metadata_json", pa.string()),
                ]
            )
            # Create empty table with schema, then add data.
            table = self._db.create_table(table_name, schema=schema)
        self._table_cache[table_name] = table
        return table

    def _ensure_meta_table(self) -> Any:
        try:
            return self._db.open_table("vector_meta")
        except Exception as exc:
            # Broad because LanceDB raises varying exception types for missing tables.
            if not _is_table_missing(exc):
                raise
            import pyarrow as pa

            schema = pa.schema(
                [
                    pa.field("profile_name", pa.string()),
                    pa.field("active_version", pa.string()),
                ]
            )
            table = self._db.create_table("vector_meta", schema=schema)
            return table


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def warmup_lancedb_connection(path: str | Path) -> None:
    """Ensure a LanceDB connection exists in the module cache.

    Calling this from the main thread before any worker thread touches the
    store prevents ``lancedb.connect()`` from being invoked inside a worker
    thread while an asyncio event loop is running, which avoids deadlocks in
    the MCP stdio environment.
    """

    lancedb = _import_lancedb()
    resolved = Path(path).resolve()
    cache_key = str(resolved)
    with _DB_CACHE_LOCK:
        if cache_key in _DB_CACHE:
            return
        resolved.mkdir(parents=True, exist_ok=True)
        _DB_CACHE[cache_key] = lancedb.connect(str(resolved))


def clear_lancedb_cache() -> None:
    """Clear the module-level LanceDB connection cache.

    Intended for tests that mock ``lancedb.connect()`` and need a fresh cache
    for every test case.
    """

    with _DB_CACHE_LOCK:
        _DB_CACHE.clear()


def _table_name(index_version: str) -> str:
    """LanceDB table names cannot start with a digit or contain hyphens."""
    safe = str(index_version).replace("-", "_").replace(".", "_")
    if safe and safe[0].isdigit():
        safe = "v_" + safe
    return f"vectors_{safe}" if safe else "vectors_legacy"


def _all_table_names(db: Any) -> list[str]:
    """Fetch every table name; LanceDB paginates (``table_names`` defaults to
    ``limit=10``), so a single call silently truncates large databases.

    Pagination uses ``start_after`` semantics: the token is the last name of
    the previous page over the sorted table names. ``table_names`` is used
    instead of the newer ``list_tables`` because it exists across the whole
    supported lancedb range (>=0.18); its DeprecationWarning on >=0.33 is
    suppressed locally.
    """

    names: list[str] = []
    page_token: str | None = None
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                batch = list(db.table_names(page_token=page_token, limit=1000))
            except TypeError:
                # Older lancedb without pagination args returns all names.
                return list(db.table_names())
        if not batch:
            break
        names.extend(batch)
        if len(batch) < 1000:
            break
        page_token = batch[-1]
    return names


def _record_to_row(
    record: VectorRecord, index_version: str, profile_name: str
) -> dict[str, Any]:
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
        "metadata_json": json.dumps(
            record.metadata or {}, ensure_ascii=False, sort_keys=True
        ),
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
            return stored_record_id[len(prefix) :]
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
