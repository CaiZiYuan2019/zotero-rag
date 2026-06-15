from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable


logger = logging.getLogger(__name__)


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


def strip_stored_version_prefix(*, stored_record_id: str, index_version: str) -> str:
    if index_version and index_version != "legacy":
        prefix = f"{index_version}:"
        if stored_record_id.startswith(prefix):
            return stored_record_id[len(prefix) :]
    return stored_record_id


def open_vector_store(
    path: str | Path,
    *,
    profile_name: str,
    dimension: int,
    backend: str = "sqlite-local",
) -> "LocalVectorStore":
    """Open a vector store for *profile_name*, picking the right backend.

    ``backend`` values:
      - ``"sqlite-local"`` (default) — SQLite-backed :class:`LocalVectorStore`
      - ``"lancedb"`` — LanceDB-backed :class:`LanceDBVectorStore`
    """
    if backend == "lancedb":
        from .lancedb_vector import LanceDBVectorStore
        return LanceDBVectorStore(path, profile_name=profile_name, dimension=dimension)  # type: ignore[return-value]
    return LocalVectorStore(path, profile_name=profile_name, dimension=dimension)


class _ThreadSafeConnection:
    """Thread-safe proxy for a sqlite3 connection.

    LocalVectorStore may be accessed from FastAPI worker threads and background
    indexing workers. This wrapper serializes every method call and context
    manager entry/exit through an RLock.
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = conn
        self._lock = lock

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        return self._conn.__enter__()

    def __exit__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._conn.__exit__(*args, **kwargs)
        finally:
            self._lock.release()

    def __getattribute__(self, name: str) -> Any:
        if name in ("_conn", "_lock", "__enter__", "__exit__"):
            return object.__getattribute__(self, name)
        attr = getattr(object.__getattribute__(self, "_conn"), name)
        if callable(attr):
            lock = object.__getattribute__(self, "_lock")

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with lock:
                    return attr(*args, **kwargs)

            return wrapper
        return attr


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
        self._lock = threading.RLock()
        raw_conn = sqlite3.connect(self.path, check_same_thread=False)
        raw_conn.row_factory = sqlite3.Row
        self.conn: sqlite3.Connection = _ThreadSafeConnection(raw_conn, self._lock)
        self._migrate()

    def close(self) -> None:
        # Checkpoint before closing. A busy/interrupted checkpoint is retried so
        # that write-ahead log data is not silently left behind.
        for attempt in range(3):
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                break
            except sqlite3.Error as exc:
                if attempt < 2:
                    logger.warning(
                        "vector store checkpoint failed (attempt %d/%d): %s",
                        attempt + 1,
                        3,
                        exc,
                    )
                else:
                    logger.error("vector store checkpoint failed after retries: %s", exc)
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
                    index_version TEXT NOT NULL DEFAULT 'legacy',
                    vector_json TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            if "index_version" not in {
                row["name"] for row in self.conn.execute("PRAGMA table_info(vectors)").fetchall()
            }:
                self.conn.execute("ALTER TABLE vectors ADD COLUMN index_version TEXT NOT NULL DEFAULT 'legacy'")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_meta (
                    profile_name TEXT PRIMARY KEY,
                    active_version TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vectors_profile
                ON vectors(profile_name, modality, index_version)
                """
            )

    def upsert(self, records: Iterable[VectorRecord], *, index_version: str = "legacy") -> int:
        with self._lock:
            count = 0
            with self.conn:
                for record in records:
                    if len(record.vector) != self.dimension:
                        raise ValueError(
                            f"record {record.record_id} has dimension {len(record.vector)}, expected {self.dimension}"
                        )
                    # Older local stores use record_id as the primary key. Prefix
                    # non-legacy staged versions so rebuilding the same chunk cannot
                    # overwrite the currently published version before commit.
                    stored_record_id = (
                        record.record_id if index_version == "legacy" else f"{index_version}:{record.record_id}"
                    )
                    self.conn.execute(
                        """
                        INSERT INTO vectors(
                            record_id, profile_name, document_id, chunk_id, modality,
                            index_version, vector_json, text, metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(record_id) DO UPDATE SET
                            profile_name = excluded.profile_name,
                            document_id = excluded.document_id,
                            chunk_id = excluded.chunk_id,
                            modality = excluded.modality,
                            index_version = excluded.index_version,
                            vector_json = excluded.vector_json,
                            text = excluded.text,
                            metadata_json = excluded.metadata_json
                        """,
                        (
                            stored_record_id,
                            self.profile_name,
                            record.document_id,
                            record.chunk_id,
                            record.modality,
                            index_version,
                            json.dumps(record.vector),
                            record.text,
                            json.dumps(record.metadata or {}, ensure_ascii=False, sort_keys=True),
                        ),
                    )
                    count += 1
            return count

    def copy_active_records_to_version(
        self,
        *,
        index_version: str,
        exclude_document_ids: Iterable[str] = (),
    ) -> int:
        """Stage existing active records under a new version.

        Incremental indexing still publishes a complete active version. Before
        adding changed records for one document, copy all other active records
        into the new staged version so publishing does not hide previously
        indexed documents.

        This method is serialized with :attr:`_lock` so that concurrent writers
        do not read the same active version and then overwrite each other's
        staged records when they publish.
        """

        with self._lock:
            active_version = self.active_version()
            excluded = set(exclude_document_ids)
            rows = self.conn.execute(
                """
                SELECT record_id, document_id, chunk_id, modality, index_version,
                       vector_json, text, metadata_json
                FROM vectors
                WHERE profile_name = ? AND index_version = ?
                """,
                (self.profile_name, active_version),
            ).fetchall()
            count = 0
            with self.conn:
                for row in rows:
                    if row["document_id"] in excluded:
                        continue
                    logical_record_id = strip_stored_version_prefix(
                        stored_record_id=str(row["record_id"]),
                        index_version=str(row["index_version"]),
                    )
                    stored_record_id = (
                        logical_record_id
                        if index_version == "legacy"
                        else f"{index_version}:{logical_record_id}"
                    )
                    self.conn.execute(
                        """
                        INSERT INTO vectors(
                            record_id, profile_name, document_id, chunk_id, modality,
                            index_version, vector_json, text, metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(record_id) DO UPDATE SET
                            profile_name = excluded.profile_name,
                            document_id = excluded.document_id,
                            chunk_id = excluded.chunk_id,
                            modality = excluded.modality,
                            index_version = excluded.index_version,
                            vector_json = excluded.vector_json,
                            text = excluded.text,
                            metadata_json = excluded.metadata_json
                        """,
                        (
                            stored_record_id,
                            self.profile_name,
                            row["document_id"],
                            row["chunk_id"],
                            row["modality"],
                            index_version,
                            row["vector_json"],
                            row["text"],
                            row["metadata_json"],
                        ),
                    )
                    count += 1
            return count

    def publish_version(self, index_version: str) -> None:
        """Atomically switch searches for this profile to a completed version.

        Writers can stage a full rebuild under a fresh `index_version`; readers
        keep using the previous version until this metadata row is committed.
        """

        with self._lock:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO vector_meta(profile_name, active_version, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(profile_name) DO UPDATE SET
                        active_version = excluded.active_version,
                        updated_at = excluded.updated_at
                    """,
                    (self.profile_name, index_version),
                )

    def active_version(self) -> str:
        row = self.conn.execute(
            "SELECT active_version FROM vector_meta WHERE profile_name = ?",
            (self.profile_name,),
        ).fetchone()
        return str(row["active_version"]) if row is not None else "legacy"

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
        if modality is None:
            rows = self.conn.execute(
                "SELECT * FROM vectors WHERE profile_name = ? AND index_version = ?",
                (self.profile_name, active_version),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM vectors WHERE profile_name = ? AND modality = ? AND index_version = ?",
                (self.profile_name, modality, active_version),
            ).fetchall()

        scored = []
        for row in rows:
            record_id = row["record_id"]
            try:
                vector = json.loads(row["vector_json"])
            except json.JSONDecodeError as exc:
                logger.warning(
                    "skipping corrupt vector record %s: invalid vector_json: %s",
                    record_id,
                    exc,
                )
                continue
            try:
                score = cosine_similarity(query_vector, vector)
            except ValueError as exc:
                logger.warning(
                    "skipping corrupt vector record %s: dimension mismatch: %s",
                    record_id,
                    exc,
                )
                continue
            try:
                metadata = json.loads(row["metadata_json"])
            except json.JSONDecodeError:
                metadata = {}
            scored.append(
                {
                    "record_id": record_id,
                    "document_id": row["document_id"],
                    "chunk_id": row["chunk_id"],
                    "modality": row["modality"],
                    "text": row["text"],
                    "metadata": metadata,
                    "score": score,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def counts(self, *, index_version: str | None = None) -> dict[str, int]:
        version = index_version or self.active_version()
        row = self.conn.execute(
            """
            SELECT count(DISTINCT document_id) AS documents, count(*) AS chunks
            FROM vectors
            WHERE profile_name = ? AND index_version = ?
            """,
            (self.profile_name, version),
        ).fetchone()
        return {"documents": int(row["documents"]), "chunks": int(row["chunks"])}

    def document_counts(
        self,
        document_id: str,
        *,
        modality: str | None = None,
        index_version: str | None = None,
    ) -> dict[str, int]:
        version = index_version or self.active_version()
        if modality is None:
            row = self.conn.execute(
                """
                SELECT count(*) AS chunks
                FROM vectors
                WHERE profile_name = ? AND document_id = ? AND index_version = ?
                """,
                (self.profile_name, document_id, version),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT count(*) AS chunks
                FROM vectors
                WHERE profile_name = ? AND document_id = ? AND modality = ? AND index_version = ?
                """,
                (self.profile_name, document_id, modality, version),
            ).fetchone()
        return {"chunks": int(row["chunks"])}

    def document_metadata_values(
        self,
        document_id: str,
        key: str,
        *,
        modality: str | None = None,
        index_version: str | None = None,
    ) -> set[Any]:
        version = index_version or self.active_version()
        if modality is None:
            rows = self.conn.execute(
                """
                SELECT metadata_json
                FROM vectors
                WHERE profile_name = ? AND document_id = ? AND index_version = ?
                """,
                (self.profile_name, document_id, version),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT metadata_json
                FROM vectors
                WHERE profile_name = ? AND document_id = ? AND modality = ? AND index_version = ?
                """,
                (self.profile_name, document_id, modality, version),
            ).fetchall()
        return {json.loads(row["metadata_json"]).get(key) for row in rows}
