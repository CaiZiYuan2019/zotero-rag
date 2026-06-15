from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable, Iterable
import uuid


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class JobEvent:
    job_id: str
    stage: str
    status: str
    message: str = ""
    payload: dict[str, Any] | None = None
    created_at: str = ""


class _ThreadSafeConnection:
    """Thread-safe proxy for a sqlite3 connection.

    SQLite connections are not thread-safe. This wrapper serializes every
    method call and context-manager entry/exit through an RLock so that the
    ledger can be shared safely across FastAPI worker threads and background
    workers without ``check_same_thread`` races.
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


class StateLedger:
    """SQLite-backed state ledger.

    The implementation keeps writes explicit and transactional. Runtime workers
    should use this through a single writer queue in later pipeline stages; this
    class is the persistence primitive for that writer.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        raw_conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            timeout=30,
            # The connection is wrapped by _ThreadSafeConnection, which uses an
            # RLock to serialize access across threads. check_same_thread=False
            # is safe here because the wrapper prevents concurrent use.
            check_same_thread=False,
        )
        raw_conn.row_factory = sqlite3.Row
        self.conn: sqlite3.Connection = _ThreadSafeConnection(raw_conn, self._lock)
        self._configure()
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    def _configure(self) -> None:
        # Set the busy timeout before switching WAL mode. Startup code may have
        # multiple short-lived CLI/API processes opening the state DB at once;
        # waiting here avoids false "database is locked" failures during status
        # reads while still keeping SQLite as a single-writer ledger.
        self.conn.execute("PRAGMA busy_timeout=30000")
        # WAL lets readers inspect status while the single state writer records
        # progress. It does not make SQLite multi-writer; pipeline workers should
        # still funnel writes through one writer component.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def migrate(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                INSERT INTO schema_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES pipeline_jobs(job_id)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    subject_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(subject_id, stage)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_rules (
                    attachment_key TEXT PRIMARY KEY,
                    decision TEXT NOT NULL CHECK(decision IN ('include', 'exclude')),
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_profiles (
                    name TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    modality TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    default_for_text INTEGER NOT NULL,
                    default_for_multimodal INTEGER NOT NULL,
                    profile_json TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_indexes (
                    profile_name TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    path TEXT NOT NULL,
                    document_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1,
                    active_version TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_columns(
                "vector_indexes",
                {
                    "active_version": "TEXT NOT NULL DEFAULT ''",
                },
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_batches (
                    batch_hash TEXT PRIMARY KEY,
                    profile_name TEXT NOT NULL,
                    profile_hash TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            # Embedding workers can use this index to resume or audit a specific
            # document/profile without scanning all historical batch records.
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_batches_document_profile
                ON embedding_batches(document_id, profile_name, status)
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attachments (
                    attachment_key TEXT PRIMARY KEY,
                    parent_key TEXT,
                    content_type TEXT,
                    relative_path TEXT,
                    title TEXT,
                    abstract TEXT,
                    date_value TEXT,
                    url TEXT,
                    classification TEXT NOT NULL,
                    source_quality TEXT NOT NULL DEFAULT 'unknown',
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    file_path TEXT,
                    file_exists INTEGER NOT NULL DEFAULT 0,
                    file_size INTEGER,
                    file_mtime REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    content_fingerprint TEXT NOT NULL DEFAULT '',
                    scan_status TEXT NOT NULL DEFAULT 'new',
                    first_seen_at TEXT,
                    last_seen_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_columns(
                "attachments",
                {
                    "content_fingerprint": "TEXT NOT NULL DEFAULT ''",
                    "scan_status": "TEXT NOT NULL DEFAULT 'new'",
                    "first_seen_at": "TEXT",
                    "last_seen_at": "TEXT",
                },
            )
            # This is the durable review queue produced from the Zotero shadow.
            # It stores classification decisions and file facts only; no PDF
            # content, embedding vectors, or expensive extraction payloads live
            # in the state DB.
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_attachments_classification
                ON attachments(classification)
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backups (
                    backup_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS extract_jobs (
                    job_id TEXT PRIMARY KEY,
                    attachment_key TEXT,
                    pdf_sha256 TEXT NOT NULL,
                    selected_pages TEXT NOT NULL DEFAULT '',
                    cache_key TEXT NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    provider_version TEXT NOT NULL,
                    options_hash TEXT NOT NULL,
                    api_key_alias TEXT NOT NULL DEFAULT '',
                    external_job_id TEXT,
                    state TEXT NOT NULL,
                    local_stage TEXT NOT NULL,
                    zip_path TEXT,
                    extract_dir TEXT,
                    artifact_dir TEXT,
                    manifest_path TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    submitted_at TEXT,
                    last_poll_at TEXT,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            # MinerU and later extractor workers may run for hours. These
            # indexes keep resume/status lookups cheap without storing large API
            # payloads in SQLite.
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_extract_jobs_state
                ON extract_jobs(state, local_stage)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_extract_jobs_attachment
                ON extract_jobs(attachment_key)
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS normalized_artifacts (
                    document_id TEXT PRIMARY KEY,
                    attachment_key TEXT,
                    extract_job_id TEXT,
                    artifact_dir TEXT NOT NULL,
                    document_md TEXT NOT NULL,
                    image_manifest TEXT NOT NULL,
                    chunks_path TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    source_markdown TEXT NOT NULL,
                    status TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    image_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    heading_path_json TEXT NOT NULL DEFAULT '[]',
                    prev_chunk_id TEXT,
                    next_chunk_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document_type
                ON chunks(document_id, chunk_type, chunk_index)
                """
            )

    def _ensure_columns(self, table_name: str, columns: dict[str, str]) -> None:
        existing = {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for name, definition in columns.items():
            if name not in existing:
                self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {definition}")

    def create_job(self, kind: str, payload: dict[str, Any] | None = None) -> str:
        job_id = str(uuid.uuid4())
        now = utc_now()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO pipeline_jobs(job_id, kind, status, created_at, updated_at, payload_json)
                VALUES (?, ?, 'created', ?, ?, ?)
                """,
                (job_id, kind, now, now, payload_json),
            )
        return job_id

    def set_job_status(self, job_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE pipeline_jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (status, utc_now(), job_id),
            )

    def get_job(self, job_id: str, *, include_events: bool = True) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT job_id, kind, status, created_at, updated_at, payload_json
            FROM pipeline_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        job = decode_pipeline_job_row(row)
        if include_events:
            job["events"] = self.list_job_events(job_id)
        return job

    def list_jobs(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        limit: int | None = 50,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT job_id, kind, status, created_at, updated_at, payload_json
            FROM pipeline_jobs
        """
        where = []
        params: list[Any] = []
        if kind is not None:
            where.append("kind = ?")
            params.append(kind)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [decode_pipeline_job_row(row) for row in rows]

    def add_event(self, event: JobEvent) -> None:
        created_at = event.created_at or utc_now()
        payload_json = json.dumps(event.payload or {}, ensure_ascii=False, sort_keys=True, default=str)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO job_events(job_id, stage, status, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event.job_id, event.stage, event.status, event.message, payload_json, created_at),
            )
            self.conn.execute(
                "UPDATE pipeline_jobs SET updated_at = ? WHERE job_id = ?",
                (created_at, event.job_id),
            )

    def list_job_events(self, job_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT event_id, job_id, stage, status, message, payload_json, created_at
            FROM job_events
            WHERE job_id = ?
            ORDER BY event_id
        """
        params: list[Any] = [job_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [decode_job_event_row(row) for row in rows]

    def checkpoint(
        self,
        subject_id: str,
        stage: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO checkpoints(subject_id, stage, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(subject_id, stage) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (subject_id, stage, status, payload_json, utc_now()),
            )

    def get_checkpoint(self, subject_id: str, stage: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT subject_id, stage, status, payload_json, updated_at
            FROM checkpoints
            WHERE subject_id = ? AND stage = ?
            """,
            (subject_id, stage),
        ).fetchone()
        if row is None:
            return None
        return {
            "subject_id": row["subject_id"],
            "stage": row["stage"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            "updated_at": row["updated_at"],
        }

    def upsert_review_rule(self, attachment_key: str, decision: str, reason: str) -> None:
        if decision not in {"include", "exclude"}:
            raise ValueError("decision must be 'include' or 'exclude'")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO review_rules(attachment_key, decision, reason, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(attachment_key) DO UPDATE SET
                    decision = excluded.decision,
                    reason = excluded.reason,
                    created_at = excluded.created_at
                """,
                (attachment_key, decision, reason, utc_now()),
            )

    def list_review_rules(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT attachment_key, decision, reason, created_at FROM review_rules ORDER BY attachment_key"
        ).fetchall()
        return [dict(row) for row in rows]

    def review_rule_map(self) -> dict[str, dict[str, Any]]:
        return {row["attachment_key"]: row for row in self.list_review_rules()}

    def upsert_embedding_profiles(self, profiles: Iterable[Any]) -> None:
        with self.conn:
            for profile in profiles:
                data = profile.__dict__ if hasattr(profile, "__dict__") else dict(profile)
                existing = self.conn.execute(
                    """
                    SELECT default_for_text, default_for_multimodal
                    FROM embedding_profiles
                    WHERE name = ?
                    """,
                    (data["name"],),
                ).fetchone()
                default_for_text = (
                    bool(existing["default_for_text"])
                    if existing is not None
                    else bool(data.get("default_for_text", False))
                )
                default_for_multimodal = (
                    bool(existing["default_for_multimodal"])
                    if existing is not None
                    else bool(data.get("default_for_multimodal", False))
                )
                self.conn.execute(
                    """
                    INSERT INTO embedding_profiles(
                        name, provider, model, dimension, modality, enabled,
                        default_for_text, default_for_multimodal, profile_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        provider = excluded.provider,
                        model = excluded.model,
                        dimension = excluded.dimension,
                        modality = excluded.modality,
                        enabled = excluded.enabled,
                        default_for_text = excluded.default_for_text,
                        default_for_multimodal = excluded.default_for_multimodal,
                        profile_json = excluded.profile_json
                    """,
                    (
                        data["name"],
                        data["provider"],
                        data["model"],
                        int(data["dimension"]),
                        data["modality"],
                        int(bool(data.get("enabled", True))),
                        int(default_for_text),
                        int(default_for_multimodal),
                        json.dumps(data, ensure_ascii=False, sort_keys=True, default=str),
                    ),
                )

    def list_embedding_profiles(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT name, provider, model, dimension, modality, enabled,
                   default_for_text, default_for_multimodal, profile_json
            FROM embedding_profiles
            ORDER BY name
            """
        ).fetchall()
        profiles = []
        for row in rows:
            profile_json = json.loads(row["profile_json"])
            profile_json["enabled"] = bool(row["enabled"])
            profile_json["default_for_text"] = bool(row["default_for_text"])
            profile_json["default_for_multimodal"] = bool(row["default_for_multimodal"])
            profiles.append(
                {
                    **dict(row),
                    "enabled": bool(row["enabled"]),
                    "default_for_text": bool(row["default_for_text"]),
                    "default_for_multimodal": bool(row["default_for_multimodal"]),
                    "profile": profile_json,
                }
            )
        return profiles

    def activate_embedding_profile(self, profile_name: str, mode: str) -> dict[str, Any]:
        if mode not in {"text", "multimodal"}:
            raise ValueError("mode must be 'text' or 'multimodal'")
        expected_modality = "text" if mode == "text" else "multimodal"
        flag_column = "default_for_text" if mode == "text" else "default_for_multimodal"
        row = self.conn.execute(
            """
            SELECT name, modality, enabled
            FROM embedding_profiles
            WHERE name = ?
            """,
            (profile_name,),
        ).fetchone()
        if row is None:
            raise KeyError(f"embedding profile not found: {profile_name}")
        if row["modality"] != expected_modality:
            raise ValueError(f"profile {profile_name} has modality {row['modality']}, expected {expected_modality}")
        if not bool(row["enabled"]):
            raise ValueError(f"profile {profile_name} is disabled")

        with self.conn:
            # Exactly one default profile per query mode keeps search scoring
            # unambiguous. Cross-model score merging is intentionally avoided.
            self.conn.execute(
                f"UPDATE embedding_profiles SET {flag_column} = 0 WHERE modality = ?",
                (expected_modality,),
            )
            self.conn.execute(
                f"UPDATE embedding_profiles SET {flag_column} = 1 WHERE name = ?",
                (profile_name,),
            )
        for profile in self.list_embedding_profiles():
            if profile["name"] == profile_name:
                return profile
        raise KeyError(f"embedding profile not found after activation: {profile_name}")

    def register_vector_index(
        self,
        profile_name: str,
        backend: str,
        path: str | Path,
        document_count: int,
        chunk_count: int,
        active: bool = True,
        active_version: str = "",
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO vector_indexes(
                    profile_name, backend, path, document_count, chunk_count,
                    active, active_version, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_name) DO UPDATE SET
                    backend = excluded.backend,
                    path = excluded.path,
                    document_count = excluded.document_count,
                    chunk_count = excluded.chunk_count,
                    active = excluded.active,
                    active_version = excluded.active_version,
                    updated_at = excluded.updated_at
                """,
                (
                    profile_name,
                    backend,
                    str(path),
                    document_count,
                    chunk_count,
                    int(active),
                    active_version,
                    utc_now(),
                ),
            )

    def list_vector_indexes(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT profile_name, backend, path, document_count, chunk_count, active, active_version, updated_at
            FROM vector_indexes
            ORDER BY profile_name
            """
        ).fetchall()
        return [
            {
                **dict(row),
                "active": bool(row["active"]),
            }
            for row in rows
        ]

    def upsert_embedding_batch(
        self,
        *,
        batch_hash: str,
        profile_name: str,
        profile_hash: str,
        document_id: str,
        chunk_type: str,
        chunk_count: int,
        status: str,
        provider: str,
        model: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist resumable embedding batch progress.

        Only stable identifiers, counts, hashes, and small metadata are stored.
        Raw text, image bytes, and vectors stay out of SQLite so full backups and
        progress JSON cannot leak expensive provider payloads.
        """

        now = utc_now()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)
        existing = self.conn.execute(
            """
            SELECT created_at
            FROM embedding_batches
            WHERE batch_hash = ?
            """,
            (batch_hash,),
        ).fetchone()
        created_at = existing["created_at"] if existing is not None else now
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO embedding_batches(
                    batch_hash, profile_name, profile_hash, document_id,
                    chunk_type, chunk_count, status, provider, model,
                    created_at, updated_at, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_hash) DO UPDATE SET
                    profile_name = excluded.profile_name,
                    profile_hash = excluded.profile_hash,
                    document_id = excluded.document_id,
                    chunk_type = excluded.chunk_type,
                    chunk_count = excluded.chunk_count,
                    status = excluded.status,
                    provider = excluded.provider,
                    model = excluded.model,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    batch_hash,
                    profile_name,
                    profile_hash,
                    document_id,
                    chunk_type,
                    int(chunk_count),
                    status,
                    provider,
                    model,
                    created_at,
                    now,
                    payload_json,
                ),
            )
        batch = self.get_embedding_batch(batch_hash)
        if batch is None:
            raise KeyError(f"embedding batch not found after upsert: {batch_hash}")
        return batch

    def get_embedding_batch(self, batch_hash: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT batch_hash, profile_name, profile_hash, document_id,
                   chunk_type, chunk_count, status, provider, model,
                   created_at, updated_at, payload_json
            FROM embedding_batches
            WHERE batch_hash = ?
            """,
            (batch_hash,),
        ).fetchone()
        return decode_embedding_batch_row(row)

    def list_embedding_batches(
        self,
        *,
        profile_name: str | None = None,
        document_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT batch_hash, profile_name, profile_hash, document_id,
                   chunk_type, chunk_count, status, provider, model,
                   created_at, updated_at, payload_json
            FROM embedding_batches
        """
        where = []
        params: list[Any] = []
        if profile_name is not None:
            where.append("profile_name = ?")
            params.append(profile_name)
        if document_id is not None:
            where.append("document_id = ?")
            params.append(document_id)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [batch for row in rows if (batch := decode_embedding_batch_row(row)) is not None]

    def upsert_attachments(self, attachments: Iterable[dict[str, Any]]) -> int:
        count = 0
        with self.conn:
            for item in attachments:
                now = utc_now()
                fingerprint = attachment_fingerprint(item)
                existing = self.conn.execute(
                    """
                    SELECT content_fingerprint, first_seen_at, classification
                    FROM attachments
                    WHERE attachment_key = ?
                    """,
                    (item["attachment_key"],),
                ).fetchone()
                if existing is None or existing["classification"] == "deleted":
                    scan_status = "new"
                    first_seen_at = now
                elif existing["content_fingerprint"] != fingerprint:
                    scan_status = "changed"
                    first_seen_at = existing["first_seen_at"] or now
                else:
                    scan_status = "unchanged"
                    first_seen_at = existing["first_seen_at"] or now
                self.conn.execute(
                    """
                    INSERT INTO attachments(
                        attachment_key, parent_key, content_type, relative_path,
                        title, abstract, date_value, url, classification,
                        source_quality, reasons_json, file_path, file_exists,
                        file_size, file_mtime, metadata_json, content_fingerprint,
                        scan_status, first_seen_at, last_seen_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(attachment_key) DO UPDATE SET
                        parent_key = excluded.parent_key,
                        content_type = excluded.content_type,
                        relative_path = excluded.relative_path,
                        title = excluded.title,
                        abstract = excluded.abstract,
                        date_value = excluded.date_value,
                        url = excluded.url,
                        classification = excluded.classification,
                        source_quality = excluded.source_quality,
                        reasons_json = excluded.reasons_json,
                        file_path = excluded.file_path,
                        file_exists = excluded.file_exists,
                        file_size = excluded.file_size,
                        file_mtime = excluded.file_mtime,
                        metadata_json = excluded.metadata_json,
                        content_fingerprint = excluded.content_fingerprint,
                        scan_status = excluded.scan_status,
                        first_seen_at = COALESCE(attachments.first_seen_at, excluded.first_seen_at),
                        last_seen_at = excluded.last_seen_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        item["attachment_key"],
                        item.get("parent_key"),
                        item.get("content_type"),
                        item.get("relative_path"),
                        item.get("title"),
                        item.get("abstract"),
                        item.get("date"),
                        item.get("url"),
                        item["classification"],
                        item.get("source_quality", "unknown"),
                        json.dumps(item.get("reasons", []), ensure_ascii=False, sort_keys=True),
                        item.get("file_path"),
                        int(bool(item.get("file_exists", False))),
                        item.get("file_size"),
                        item.get("file_mtime"),
                        json.dumps(item.get("metadata", {}), ensure_ascii=False, sort_keys=True, default=str),
                        fingerprint,
                        scan_status,
                        first_seen_at,
                        now,
                        now,
                    ),
                )
                count += 1
        return count

    def list_attachments(
        self,
        classification: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT attachment_key, parent_key, content_type, relative_path,
                   title, abstract, date_value, url, classification,
                   source_quality, reasons_json, file_path, file_exists,
                   file_size, file_mtime, metadata_json, content_fingerprint,
                   scan_status, first_seen_at, last_seen_at, updated_at
            FROM attachments
        """
        params: list[Any] = []
        if classification is not None:
            query += " WHERE classification = ?"
            params.append(classification)
        query += " ORDER BY attachment_key"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [decode_attachment_row(row) for row in rows]

    def get_attachment(self, attachment_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT attachment_key, parent_key, content_type, relative_path,
                   title, abstract, date_value, url, classification,
                   source_quality, reasons_json, file_path, file_exists,
                   file_size, file_mtime, metadata_json, content_fingerprint,
                   scan_status, first_seen_at, last_seen_at, updated_at
            FROM attachments
            WHERE attachment_key = ?
            """,
            (attachment_key,),
        ).fetchone()
        return decode_attachment_row(row) if row is not None else None

    def search_attachments_metadata(
        self,
        query: str,
        *,
        classification: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Direct metadata search over scanned Zotero shadow results.

        This deliberately searches only the local state ledger populated from
        the shadow DB. It never reads Zotero's live database, and it never calls
        embedding services. The implementation is parameterized LIKE search for
        the bootstrap phase; the call shape is stable so an FTS5-backed version
        can replace it later without changing API/CLI consumers.
        """

        terms = [term.casefold() for term in query.split() if term.strip()]
        if not terms:
            return []

        where_parts = []
        params: list[Any] = []
        searchable_columns = (
            "attachment_key",
            "parent_key",
            "relative_path",
            "title",
            "abstract",
            "date_value",
            "url",
            "classification",
            "source_quality",
        )
        for term in terms:
            like = f"%{term}%"
            per_term = " OR ".join(f"lower(coalesce({column}, '')) LIKE ?" for column in searchable_columns)
            where_parts.append(f"({per_term})")
            params.extend([like] * len(searchable_columns))
        if classification is not None:
            where_parts.append("classification = ?")
            params.append(classification)

        query_sql = f"""
            SELECT attachment_key, parent_key, content_type, relative_path,
                   title, abstract, date_value, url, classification,
                   source_quality, reasons_json, file_path, file_exists,
                   file_size, file_mtime, metadata_json, content_fingerprint,
                   scan_status, first_seen_at, last_seen_at, updated_at
            FROM attachments
            WHERE {' AND '.join(where_parts)}
        """
        rows = self.conn.execute(query_sql, params).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            reasons_json = item.pop("reasons_json")
            metadata_json = item.pop("metadata_json")
            item["file_exists"] = bool(row["file_exists"])
            item["reasons"] = json.loads(reasons_json)
            item["metadata"] = json.loads(metadata_json)
            item["score"] = metadata_match_score(item, terms)
            results.append(item)
        results.sort(key=lambda item: (item["score"], item.get("title") or ""), reverse=True)
        return results[:limit]

    def add_scan_report(self, summary: dict[str, Any], job_id: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO scan_reports(job_id, summary_json, created_at)
                VALUES (?, ?, ?)
                """,
                (job_id, json.dumps(summary, ensure_ascii=False, sort_keys=True), utc_now()),
            )

    def add_backup_record(self, backup_id: str, mode: str, path: str | Path, manifest: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO backups(backup_id, mode, path, manifest_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(backup_id) DO UPDATE SET
                    mode = excluded.mode,
                    path = excluded.path,
                    manifest_json = excluded.manifest_json,
                    created_at = excluded.created_at
                """,
                (
                    backup_id,
                    mode,
                    str(path),
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True, default=str),
                    utc_now(),
                ),
            )

    def list_backups(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT backup_id, mode, path, manifest_json, created_at
            FROM backups
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [
            {
                "backup_id": row["backup_id"],
                "mode": row["mode"],
                "path": row["path"],
                "manifest": json.loads(row["manifest_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def upsert_extract_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Persist one extractor job without leaking API secrets.

        Callers may store an `api_key_alias` such as `mineru_1`, but never pass
        the raw API key here. This keeps exported state, progress JSON, and full
        backups safe to inspect or share.
        """

        now = utc_now()
        job_id = job.get("job_id") or str(uuid.uuid4())
        payload_json = json.dumps(job.get("payload", {}), ensure_ascii=False, sort_keys=True, default=str)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO extract_jobs(
                    job_id, attachment_key, pdf_sha256, selected_pages, cache_key,
                    provider, provider_version, options_hash, api_key_alias,
                    external_job_id, state, local_stage, zip_path, extract_dir,
                    artifact_dir, manifest_path, error_code, error_message,
                    retry_count, submitted_at, last_poll_at, updated_at, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    attachment_key = excluded.attachment_key,
                    pdf_sha256 = excluded.pdf_sha256,
                    selected_pages = excluded.selected_pages,
                    provider = excluded.provider,
                    provider_version = excluded.provider_version,
                    options_hash = excluded.options_hash,
                    api_key_alias = excluded.api_key_alias,
                    external_job_id = excluded.external_job_id,
                    state = excluded.state,
                    local_stage = excluded.local_stage,
                    zip_path = excluded.zip_path,
                    extract_dir = excluded.extract_dir,
                    artifact_dir = excluded.artifact_dir,
                    manifest_path = excluded.manifest_path,
                    error_code = excluded.error_code,
                    error_message = excluded.error_message,
                    retry_count = excluded.retry_count,
                    submitted_at = COALESCE(extract_jobs.submitted_at, excluded.submitted_at),
                    last_poll_at = excluded.last_poll_at,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    job_id,
                    job.get("attachment_key"),
                    job["pdf_sha256"],
                    job.get("selected_pages", ""),
                    job["cache_key"],
                    job["provider"],
                    job["provider_version"],
                    job["options_hash"],
                    job.get("api_key_alias", ""),
                    job.get("external_job_id"),
                    job["state"],
                    job["local_stage"],
                    job.get("zip_path"),
                    job.get("extract_dir"),
                    job.get("artifact_dir"),
                    job.get("manifest_path"),
                    job.get("error_code"),
                    job.get("error_message"),
                    int(job.get("retry_count", 0)),
                    job.get("submitted_at"),
                    job.get("last_poll_at"),
                    now,
                    payload_json,
                ),
            )
        return self.get_extract_job(job_id=job_id) or self.get_extract_job_by_cache_key(job["cache_key"])  # type: ignore[return-value]

    def get_extract_job(self, *, job_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT job_id, attachment_key, pdf_sha256, selected_pages, cache_key,
                   provider, provider_version, options_hash, api_key_alias,
                   external_job_id, state, local_stage, zip_path, extract_dir,
                   artifact_dir, manifest_path, error_code, error_message,
                   retry_count, submitted_at, last_poll_at, updated_at, payload_json
            FROM extract_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        return decode_extract_job_row(row)

    def get_extract_job_by_cache_key(self, cache_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT job_id, attachment_key, pdf_sha256, selected_pages, cache_key,
                   provider, provider_version, options_hash, api_key_alias,
                   external_job_id, state, local_stage, zip_path, extract_dir,
                   artifact_dir, manifest_path, error_code, error_message,
                   retry_count, submitted_at, last_poll_at, updated_at, payload_json
            FROM extract_jobs
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        return decode_extract_job_row(row)

    def list_extract_jobs(
        self,
        *,
        state: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT job_id, attachment_key, pdf_sha256, selected_pages, cache_key,
                   provider, provider_version, options_hash, api_key_alias,
                   external_job_id, state, local_stage, zip_path, extract_dir,
                   artifact_dir, manifest_path, error_code, error_message,
                   retry_count, submitted_at, last_poll_at, updated_at, payload_json
            FROM extract_jobs
        """
        params: list[Any] = []
        if state is not None:
            query += " WHERE state = ?"
            params.append(state)
        query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [job for row in rows if (job := decode_extract_job_row(row)) is not None]

    def set_extract_job_state(
        self,
        job_id: str,
        *,
        state: str,
        local_stage: str,
        external_job_id: str | None = None,
        zip_path: str | Path | None = None,
        extract_dir: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        last_poll_at: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        existing = self.get_extract_job(job_id=job_id)
        if existing is None:
            raise KeyError(f"extract job not found: {job_id}")
        payload_json = json.dumps(
            payload if payload is not None else existing.get("payload", {}),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        with self.conn:
            self.conn.execute(
                """
                UPDATE extract_jobs
                SET state = ?,
                    local_stage = ?,
                    external_job_id = COALESCE(?, external_job_id),
                    zip_path = COALESCE(?, zip_path),
                    extract_dir = COALESCE(?, extract_dir),
                    artifact_dir = COALESCE(?, artifact_dir),
                    manifest_path = COALESCE(?, manifest_path),
                    error_code = ?,
                    error_message = ?,
                    last_poll_at = COALESCE(?, last_poll_at),
                    updated_at = ?,
                    payload_json = ?
                WHERE job_id = ?
                """,
                (
                    state,
                    local_stage,
                    external_job_id,
                    str(zip_path) if zip_path is not None else None,
                    str(extract_dir) if extract_dir is not None else None,
                    str(artifact_dir) if artifact_dir is not None else None,
                    str(manifest_path) if manifest_path is not None else None,
                    error_code,
                    error_message,
                    last_poll_at,
                    utc_now(),
                    payload_json,
                    job_id,
                ),
            )

    def upsert_normalized_artifact(self, artifact: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        payload_json = json.dumps(artifact.get("payload", {}), ensure_ascii=False, sort_keys=True, default=str)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO normalized_artifacts(
                    document_id, attachment_key, extract_job_id, artifact_dir,
                    document_md, image_manifest, chunks_path, manifest_path,
                    source_markdown, status, document_hash, chunk_count,
                    image_count, updated_at, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    attachment_key = excluded.attachment_key,
                    extract_job_id = excluded.extract_job_id,
                    artifact_dir = excluded.artifact_dir,
                    document_md = excluded.document_md,
                    image_manifest = excluded.image_manifest,
                    chunks_path = excluded.chunks_path,
                    manifest_path = excluded.manifest_path,
                    source_markdown = excluded.source_markdown,
                    status = excluded.status,
                    document_hash = excluded.document_hash,
                    chunk_count = excluded.chunk_count,
                    image_count = excluded.image_count,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    artifact["document_id"],
                    artifact.get("attachment_key"),
                    artifact.get("extract_job_id"),
                    str(artifact["artifact_dir"]),
                    str(artifact["document_md"]),
                    str(artifact["image_manifest"]),
                    str(artifact["chunks_path"]),
                    str(artifact["manifest_path"]),
                    str(artifact["source_markdown"]),
                    artifact["status"],
                    artifact["document_hash"],
                    int(artifact.get("chunk_count", 0)),
                    int(artifact.get("image_count", 0)),
                    now,
                    payload_json,
                ),
            )
        return self.get_normalized_artifact(artifact["document_id"])  # type: ignore[return-value]

    def get_normalized_artifact(self, document_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT document_id, attachment_key, extract_job_id, artifact_dir,
                   document_md, image_manifest, chunks_path, manifest_path,
                   source_markdown, status, document_hash, chunk_count,
                   image_count, updated_at, payload_json
            FROM normalized_artifacts
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
        return decode_normalized_artifact_row(row)

    def list_normalized_artifacts(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT document_id, attachment_key, extract_job_id, artifact_dir,
                   document_md, image_manifest, chunks_path, manifest_path,
                   source_markdown, status, document_hash, chunk_count,
                   image_count, updated_at, payload_json
            FROM normalized_artifacts
            ORDER BY updated_at DESC
        """
        params: list[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [item for row in rows if (item := decode_normalized_artifact_row(row)) is not None]

    def replace_document_chunks(self, document_id: str, chunks: Iterable[dict[str, Any]]) -> int:
        now = utc_now()
        count = 0
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            for chunk in chunks:
                self.conn.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id, document_id, chunk_type, chunk_index, text,
                        heading_path_json, prev_chunk_id, next_chunk_id,
                        metadata_json, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["chunk_id"],
                        document_id,
                        chunk["chunk_type"],
                        int(chunk["chunk_index"]),
                        chunk.get("text", ""),
                        json.dumps(chunk.get("heading_path", []), ensure_ascii=False, sort_keys=True),
                        chunk.get("prev_chunk_id"),
                        chunk.get("next_chunk_id"),
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False, sort_keys=True, default=str),
                        now,
                    ),
                )
                count += 1
        return count

    def list_chunks(
        self,
        document_id: str,
        *,
        chunk_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT chunk_id, document_id, chunk_type, chunk_index, text,
                   heading_path_json, prev_chunk_id, next_chunk_id,
                   metadata_json, updated_at
            FROM chunks
            WHERE document_id = ?
        """
        params: list[Any] = [document_id]
        if chunk_type is not None:
            query += " AND chunk_type = ?"
            params.append(chunk_type)
        query += " ORDER BY chunk_index"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [decode_chunk_row(row) for row in rows]

    def search_chunks_fulltext(
        self,
        query: str,
        *,
        chunk_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        terms = [term.casefold() for term in query.split() if term.strip()]
        if not terms:
            return []
        where_parts = []
        params: list[Any] = []
        for term in terms:
            where_parts.append("lower(text) LIKE ?")
            params.append(f"%{term}%")
        if chunk_type is not None:
            where_parts.append("chunk_type = ?")
            params.append(chunk_type)
        sql = f"""
            SELECT chunk_id, document_id, chunk_type, chunk_index, text,
                   heading_path_json, prev_chunk_id, next_chunk_id,
                   metadata_json, updated_at
            FROM chunks
            WHERE {' AND '.join(where_parts)}
            ORDER BY document_id, chunk_index
        """
        rows = self.conn.execute(sql, params).fetchall()
        results = [decode_chunk_row(row) for row in rows]
        for item in results:
            item["score"] = chunk_match_score(item, terms)
        results.sort(key=lambda item: (item["score"], item["document_id"], -item["chunk_index"]), reverse=True)
        return results[:limit]

    def mark_absent_attachments_deleted(self, present_attachment_keys: Iterable[str]) -> int:
        keys = sorted(set(present_attachment_keys))
        now = utc_now()
        if not keys:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    UPDATE attachments
                    SET classification = 'deleted',
                        source_quality = 'deleted_from_zotero_shadow',
                        reasons_json = ?,
                        scan_status = 'deleted',
                        last_seen_at = ?,
                        updated_at = ?
                    WHERE classification != 'deleted'
                    """,
                    (json.dumps(["absent_from_full_shadow_scan"]), now, now),
                )
            return int(cursor.rowcount)

        placeholders = ",".join("?" for _ in keys)
        with self.conn:
            cursor = self.conn.execute(
                f"""
                UPDATE attachments
                SET classification = 'deleted',
                    source_quality = 'deleted_from_zotero_shadow',
                    reasons_json = ?,
                    scan_status = 'deleted',
                    last_seen_at = ?,
                    updated_at = ?
                WHERE attachment_key NOT IN ({placeholders})
                  AND classification != 'deleted'
                """,
                [json.dumps(["absent_from_full_shadow_scan"]), now, now, *keys],
            )
        return int(cursor.rowcount)

    def status_summary(self) -> dict[str, Any]:
        jobs = self.conn.execute(
            "SELECT status, count(*) AS n FROM pipeline_jobs GROUP BY status"
        ).fetchall()
        checkpoints = self.conn.execute("SELECT count(*) AS n FROM checkpoints").fetchone()
        profiles = self.conn.execute("SELECT count(*) AS n FROM embedding_profiles").fetchone()
        indexes = self.conn.execute("SELECT count(*) AS n FROM vector_indexes").fetchone()
        attachments = self.conn.execute("SELECT classification, count(*) AS n FROM attachments GROUP BY classification").fetchall()
        scan_statuses = self.conn.execute("SELECT scan_status, count(*) AS n FROM attachments GROUP BY scan_status").fetchall()
        backups = self.conn.execute("SELECT count(*) AS n FROM backups").fetchone()
        extract_jobs = self.conn.execute("SELECT state, count(*) AS n FROM extract_jobs GROUP BY state").fetchall()
        embedding_batches = self.conn.execute("SELECT status, count(*) AS n FROM embedding_batches GROUP BY status").fetchall()
        normalized = self.conn.execute("SELECT count(*) AS n FROM normalized_artifacts").fetchone()
        chunks = self.conn.execute("SELECT chunk_type, count(*) AS n FROM chunks GROUP BY chunk_type").fetchall()
        return {
            "schema_version": SCHEMA_VERSION,
            "jobs": {row["status"]: row["n"] for row in jobs},
            "checkpoints": checkpoints["n"],
            "embedding_profiles": profiles["n"],
            "vector_indexes": indexes["n"],
            "attachments": {row["classification"]: row["n"] for row in attachments},
            "scan_statuses": {row["scan_status"]: row["n"] for row in scan_statuses},
            "backups": backups["n"],
            "extract_jobs": {row["state"]: row["n"] for row in extract_jobs},
            "embedding_batches": {row["status"]: row["n"] for row in embedding_batches},
            "normalized_artifacts": normalized["n"],
            "chunks": {row["chunk_type"]: row["n"] for row in chunks},
        }


def attachment_fingerprint(item: dict[str, Any]) -> str:
    """Stable hash for deciding whether a scanned attachment changed.

    The hash excludes state timestamps and job bookkeeping. If this changes,
    later pipeline stages know the attachment's metadata, file facts, or review
    classification changed and can decide whether extraction/indexing is stale.
    """

    payload = {
        "parent_key": item.get("parent_key"),
        "content_type": item.get("content_type"),
        "relative_path": item.get("relative_path"),
        "title": item.get("title"),
        "abstract": item.get("abstract"),
        "date": item.get("date"),
        "url": item.get("url"),
        "classification": item.get("classification"),
        "source_quality": item.get("source_quality"),
        "reasons": item.get("reasons", []),
        "file_path": item.get("file_path"),
        "file_exists": bool(item.get("file_exists", False)),
        "file_size": item.get("file_size"),
        "file_mtime": item.get("file_mtime"),
        "metadata": item.get("metadata", {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_match_score(item: dict[str, Any], terms: list[str]) -> float:
    weighted_fields = (
        ("title", 5.0),
        ("abstract", 2.0),
        ("relative_path", 1.5),
        ("url", 1.0),
        ("date_value", 1.0),
        ("attachment_key", 1.0),
        ("parent_key", 1.0),
    )
    score = 0.0
    for field, weight in weighted_fields:
        value = str(item.get(field) or "").casefold()
        for term in terms:
            if term in value:
                score += weight
    return score


def chunk_match_score(item: dict[str, Any], terms: list[str]) -> float:
    text = str(item.get("text") or "").casefold()
    heading = " ".join(str(part) for part in item.get("heading_path", [])).casefold()
    score = 0.0
    for term in terms:
        score += text.count(term)
        if term in heading:
            score += 2.0
    if item.get("chunk_type") == "text":
        score += 0.25
    return score


def decode_extract_job_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    payload_json = item.pop("payload_json")
    item["payload"] = json.loads(payload_json)
    return item


def decode_embedding_batch_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    payload_json = item.pop("payload_json")
    item["payload"] = json.loads(payload_json)
    return item


def decode_pipeline_job_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    payload_json = item.pop("payload_json")
    item["payload"] = json.loads(payload_json)
    return item


def decode_job_event_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    payload_json = item.pop("payload_json")
    item["payload"] = json.loads(payload_json)
    return item


def decode_attachment_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    reasons_json = item.pop("reasons_json")
    metadata_json = item.pop("metadata_json")
    item["file_exists"] = bool(row["file_exists"])
    item["reasons"] = json.loads(reasons_json)
    item["metadata"] = json.loads(metadata_json)
    return item


def decode_normalized_artifact_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    payload_json = item.pop("payload_json")
    item["payload"] = json.loads(payload_json)
    return item


def decode_chunk_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["heading_path"] = json.loads(item.pop("heading_path_json"))
    item["metadata"] = json.loads(item.pop("metadata_json"))
    return item
