from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable
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


class StateLedger:
    """SQLite-backed state ledger.

    The implementation keeps writes explicit and transactional. Runtime workers
    should use this through a single writer queue in later pipeline stages; this
    class is the persistence primitive for that writer.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self._configure()
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    def _configure(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")

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
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create_job(self, kind: str, payload: dict[str, Any] | None = None) -> str:
        job_id = str(uuid.uuid4())
        now = utc_now()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
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

    def add_event(self, event: JobEvent) -> None:
        created_at = event.created_at or utc_now()
        payload_json = json.dumps(event.payload or {}, ensure_ascii=False, sort_keys=True)
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

    def checkpoint(
        self,
        subject_id: str,
        stage: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
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

    def upsert_embedding_profiles(self, profiles: Iterable[Any]) -> None:
        with self.conn:
            for profile in profiles:
                data = profile.__dict__ if hasattr(profile, "__dict__") else dict(profile)
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
                        int(bool(data.get("default_for_text", False))),
                        int(bool(data.get("default_for_multimodal", False))),
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
        return [
            {
                **dict(row),
                "enabled": bool(row["enabled"]),
                "default_for_text": bool(row["default_for_text"]),
                "default_for_multimodal": bool(row["default_for_multimodal"]),
                "profile": json.loads(row["profile_json"]),
            }
            for row in rows
        ]

    def register_vector_index(
        self,
        profile_name: str,
        backend: str,
        path: str | Path,
        document_count: int,
        chunk_count: int,
        active: bool = True,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO vector_indexes(profile_name, backend, path, document_count, chunk_count, active, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_name) DO UPDATE SET
                    backend = excluded.backend,
                    path = excluded.path,
                    document_count = excluded.document_count,
                    chunk_count = excluded.chunk_count,
                    active = excluded.active,
                    updated_at = excluded.updated_at
                """,
                (
                    profile_name,
                    backend,
                    str(path),
                    document_count,
                    chunk_count,
                    int(active),
                    utc_now(),
                ),
            )

    def status_summary(self) -> dict[str, Any]:
        jobs = self.conn.execute(
            "SELECT status, count(*) AS n FROM pipeline_jobs GROUP BY status"
        ).fetchall()
        checkpoints = self.conn.execute("SELECT count(*) AS n FROM checkpoints").fetchone()
        profiles = self.conn.execute("SELECT count(*) AS n FROM embedding_profiles").fetchone()
        indexes = self.conn.execute("SELECT count(*) AS n FROM vector_indexes").fetchone()
        return {
            "schema_version": SCHEMA_VERSION,
            "jobs": {row["status"]: row["n"] for row in jobs},
            "checkpoints": checkpoints["n"],
            "embedding_profiles": profiles["n"],
            "vector_indexes": indexes["n"],
        }

