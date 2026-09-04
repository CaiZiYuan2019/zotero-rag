from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Iterable
import uuid


logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1


# Ordered column list for the standalone attachments FTS5 index. The migration
# detects drift against this exact list and upsert/backfill INSERTs must stay in
# sync with it.
ATTACHMENTS_FTS_COLUMNS = (
    "attachment_key",
    "parent_key",
    "relative_path",
    "title",
    "abstract",
    "date_value",
    "url",
    "classification",
    "source_quality",
    "creators",
    "collections",
    "tags",
    "publication_title",
    "doi",
)

_ATTACHMENTS_FTS_INSERT_SQL = (
    "INSERT INTO attachments_fts(rowid, "
    + ", ".join(ATTACHMENTS_FTS_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in range(len(ATTACHMENTS_FTS_COLUMNS) + 1))
    + ")"
)

# Shared attachment SELECT projection so list/get/search all decode the same
# columns (including the bibliographic additions).
_ATTACHMENT_SELECT_COLUMNS = """attachment_key, parent_key, content_type, relative_path,
                   title, abstract, date_value, url, classification,
                   source_quality, reasons_json, file_path, file_exists,
                   file_size, file_mtime, metadata_json, content_fingerprint,
                   scan_status, first_seen_at, last_seen_at, updated_at,
                   creators_json, collections_json, tags_json, item_type,
                   publication_title, doi, date_added, date_modified,
                   date_year, date_normalized"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fts_columns_from_sql(sql: str) -> list[str]:
    """Extract FTS5 column names from a stored CREATE VIRTUAL TABLE statement.

    Only the plain column identifiers are returned; FTS5 options such as
    ``content=`` or ``tokenize=`` (which contain ``=``) are skipped.
    """

    if not sql:
        return []
    start = sql.find("(")
    end = sql.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return []
    inner = sql[start + 1 : end]
    columns: list[str] = []
    for part in inner.split(","):
        token = part.strip()
        if not token or "=" in token:
            continue
        token = token.split()[0].strip().strip('"').strip("`").strip("[").strip("]")
        if token:
            columns.append(token)
    return columns


def _expand_json_text(raw: str | None, *, key: str | None = None) -> str:
    """Expand a JSON array column into space-joined searchable text.

    ``key`` selects a field when the array holds objects (e.g. collections hold
    ``{"collection_name": ...}``); otherwise scalar string elements are joined.
    """

    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    parts: list[str] = []
    if isinstance(data, list):
        for element in data:
            if key is not None and isinstance(element, dict):
                value = element.get(key)
            else:
                value = element
            if value:
                parts.append(str(value))
    return " ".join(parts)


def _biblio_fts_text(
    creators_json: str | None,
    collections_json: str | None,
    tags_json: str | None,
) -> tuple[str, str, str]:
    creators_text = _expand_json_text(creators_json)
    collections_text = _expand_json_text(collections_json, key="collection_name")
    tags_text = _expand_json_text(tags_json)
    return creators_text, collections_text, tags_text


def normalize_date(value: str | None) -> tuple[int | None, str | None]:
    """Return ``(year, normalized)`` parsed loosely from a Zotero date string.

    Zotero stores dirty, human-entered dates. This best-effort parser extracts a
    4-digit year and, when possible, a ``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD``
    normalized form. On failure it returns ``(None, original_or_None)`` so the
    caller keeps the raw string for display while range filters simply skip it.
    """

    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None

    iso = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        year = int(iso.group(1))
        return year, f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"
    ym = re.match(r"^\s*(\d{4})-(\d{2})(?!\d)", text)
    if ym:
        return int(ym.group(1)), f"{ym.group(1)}-{ym.group(2)}"
    year_match = re.search(r"(\d{4})", text)
    if year_match:
        year = int(year_match.group(1))
        if 1000 <= year <= 9999:
            return year, str(year)
    return None, text or None


_METADATA_SORT_FIELDS = frozenset({"score", "bm25", "date", "date_added", "title"})
# Upper bound on rows materialized by the pure-filter / candidate load path so
# author/tag Python filtering never scans an unbounded table.
_METADATA_SCAN_CAP = 5000


@dataclass(frozen=True)
class _MetadataFilters:
    classification: str | None
    date_from: str | None
    date_to: str | None
    authors: list[str]
    collections: list[str]
    tags: list[str]
    item_type: str | None
    publication_title: str | None
    doi: str | None
    attachment_keys: list[str] | None

    def has_any(self) -> bool:
        return any(
            (
                self.classification is not None,
                self.date_from is not None,
                self.date_to is not None,
                bool(self.authors),
                bool(self.collections),
                bool(self.tags),
                self.item_type is not None,
                self.publication_title is not None,
                self.doi is not None,
                self.attachment_keys is not None,
            )
        )


def _date_bound(value: str, *, upper: bool) -> str:
    """Normalize a date filter bound for string comparison on date_normalized.

    Year-only ("2019") and year-month ("2019-05") upper bounds are extended to
    be inclusive of the whole period so ``date_to="2019"`` still matches
    ``2019-12-31``.
    """

    _, normalized = normalize_date(value)
    normalized = normalized or str(value).strip()
    if not upper:
        return normalized
    if re.fullmatch(r"\d{4}", normalized):
        return f"{normalized}-12-31"
    if re.fullmatch(r"\d{4}-\d{2}", normalized):
        return f"{normalized}-31"
    return normalized


def _passes_python_filters(item: dict[str, Any], filters: "_MetadataFilters") -> bool:
    """Apply author (substring) and tag (exact, case-insensitive) filters.

    These run on decoded structured lists rather than raw JSON strings so they
    cannot match JSON punctuation.
    """

    if filters.authors:
        creators = [str(name).casefold() for name in (item.get("creators") or [])]
        for wanted in filters.authors:
            needle = wanted.casefold()
            if not any(needle in name for name in creators):
                return False
    if filters.tags:
        tags = {str(tag).casefold() for tag in (item.get("tags") or [])}
        for wanted in filters.tags:
            if wanted.casefold() not in tags:
                return False
    return True


def _sort_metadata_results(
    results: list[dict[str, Any]], *, sort_by: str, sort_order: str
) -> None:
    reverse = sort_order == "desc"
    if sort_by in {"score", "bm25"}:
        results.sort(
            key=lambda item: (item.get("score", 0.0), str(item.get("title") or "").casefold()),
            reverse=reverse,
        )
    elif sort_by == "date":
        results.sort(key=lambda item: str(item.get("date_normalized") or ""), reverse=reverse)
    elif sort_by == "date_added":
        results.sort(key=lambda item: str(item.get("date_added") or ""), reverse=reverse)
    elif sort_by == "title":
        results.sort(key=lambda item: str(item.get("title") or "").casefold(), reverse=reverse)


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
        # Try to checkpoint any pending WAL frames before closing so that later
        # readers and backup tools see a consistent state without needing to
        # reopen the WAL. Failures are logged rather than swallowed.
        try:
            self._checkpoint_with_retry(retries=3, delay_seconds=0.5)
        except sqlite3.Error as exc:
            logger.error("state ledger checkpoint failed before close: %s", exc)
        try:
            self.conn.close()
        except sqlite3.Error as exc:
            logger.error("state ledger connection close failed: %s", exc)

    def _checkpoint_with_retry(self, retries: int, delay_seconds: float) -> None:
        last_exc: sqlite3.Error | None = None
        for attempt in range(retries):
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                return
            except sqlite3.Error as exc:
                last_exc = exc
                logger.warning(
                    "state ledger checkpoint attempt %d/%d failed: %s",
                    attempt + 1,
                    retries,
                    exc,
                )
                if attempt < retries - 1:
                    time.sleep(delay_seconds)
        if last_exc is not None:
            raise last_exc

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
            # Bibliographic fields added by the retrieval enhancement. They live
            # in dedicated columns (never in metadata_json) so biblio-only
            # changes cannot flip attachment_fingerprint()/scan_status.
            self._ensure_columns(
                "attachments",
                {
                    "creators_json": "TEXT NOT NULL DEFAULT '[]'",
                    "collections_json": "TEXT NOT NULL DEFAULT '[]'",
                    "tags_json": "TEXT NOT NULL DEFAULT '[]'",
                    "item_type": "TEXT",
                    "publication_title": "TEXT",
                    "doi": "TEXT",
                    "date_added": "TEXT",
                    "date_modified": "TEXT",
                    "date_year": "INTEGER",
                    "date_normalized": "TEXT",
                },
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS zotero_collections (
                    collection_key TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    parent_collection_key TEXT,
                    library_id INTEGER
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attachment_collections (
                    attachment_key TEXT NOT NULL,
                    collection_key TEXT NOT NULL,
                    PRIMARY KEY (attachment_key, collection_key)
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_attachment_collections_collection
                ON attachment_collections(collection_key)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_attachments_item_type
                ON attachments(item_type)
                """
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
            self._migrate_fts5()

    def _migrate_fts5(self) -> None:
        """Create and backfill standalone FTS5 indexes.

        FTS5 virtual tables store their own copy of the searchable text columns.
        They are maintained manually alongside the source tables so that writes
        remain explicit and schema migrations stay simple.

        If an older external-content-style FTS5 table exists (created with
        ``content='...'``), it is dropped and recreated as a standalone index.

        If FTS5 is unavailable in this SQLite build, the method logs a warning
        and returns so the application can still run using the LIKE fallback.
        """

        try:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
            self.conn.execute("DROP TABLE _fts5_probe")
        except sqlite3.Error as exc:
            logger.warning("FTS5 is not available in this SQLite build: %s", exc)
            return

        existing = {
            row["name"]: row["sql"] or ""
            for row in self.conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('attachments_fts', 'chunks_fts')"
            ).fetchall()
        }

        # attachments_fts: parse the stored CREATE SQL for its column set and
        # drop/recreate whenever it no longer matches the expected columns, is
        # an old external-content (content=) table, or is missing entirely.
        # Column detection intentionally parses sqlite_master.sql rather than
        # using PRAGMA table_info to avoid FTS5 hidden/rank columns.
        attach_sql = existing.get("attachments_fts")
        needs_rebuild = attach_sql is None
        if attach_sql is not None:
            if "content=" in attach_sql:
                needs_rebuild = True
            elif _fts_columns_from_sql(attach_sql) != list(ATTACHMENTS_FTS_COLUMNS):
                needs_rebuild = True
        if needs_rebuild and attach_sql is not None:
            self.conn.execute("DROP TABLE IF EXISTS attachments_fts")

        if "chunks_fts" in existing and "content=" in existing["chunks_fts"]:
            self.conn.execute("DROP TABLE chunks_fts")
            existing.pop("chunks_fts", None)

        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS attachments_fts USING fts5(
                {", ".join(ATTACHMENTS_FTS_COLUMNS)}
            )
            """
        )
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id, document_id, text
            )
            """
        )
        attachments_fts_count = self.conn.execute(
            "SELECT count(*) FROM attachments_fts"
        ).fetchone()[0]
        if attachments_fts_count == 0:
            self._backfill_attachments_fts()
        chunks_fts_count = self.conn.execute(
            "SELECT count(*) FROM chunks_fts"
        ).fetchone()[0]
        if chunks_fts_count == 0:
            self.conn.execute(
                """
                INSERT INTO chunks_fts(rowid, chunk_id, document_id, text)
                SELECT rowid, chunk_id, document_id, text FROM chunks
                """
            )

    def _backfill_attachments_fts(self) -> None:
        """Rebuild every attachments_fts row from the attachments table.

        JSON list columns (creators/collections/tags) are expanded into plain
        searchable text here so FTS5 can match author names, tag labels, and
        collection names without indexing raw JSON punctuation.
        """

        rows = self.conn.execute(
            """
            SELECT rowid, attachment_key, parent_key, relative_path, title,
                   abstract, date_value, url, classification, source_quality,
                   creators_json, collections_json, tags_json,
                   publication_title, doi
            FROM attachments
            """
        ).fetchall()
        for row in rows:
            creators_text, collections_text, tags_text = _biblio_fts_text(
                row["creators_json"], row["collections_json"], row["tags_json"]
            )
            self.conn.execute(
                _ATTACHMENTS_FTS_INSERT_SQL,
                (
                    row["rowid"],
                    row["attachment_key"],
                    row["parent_key"],
                    row["relative_path"],
                    row["title"],
                    row["abstract"],
                    row["date_value"],
                    row["url"],
                    row["classification"],
                    row["source_quality"],
                    creators_text,
                    collections_text,
                    tags_text,
                    row["publication_title"],
                    row["doi"],
                ),
            )

    def rebuild_attachments_fts(self, force: bool = False) -> int:
        """Drop and rebuild attachments_fts from scratch, returning row count.

        This only touches the FTS index and never triggers extraction or
        embedding. ``force`` drops the table first; otherwise it clears the
        existing rows before backfilling.
        """

        with self.conn:
            try:
                if force:
                    self.conn.execute("DROP TABLE IF EXISTS attachments_fts")
                    self.conn.execute(
                        f"""
                        CREATE VIRTUAL TABLE attachments_fts USING fts5(
                            {", ".join(ATTACHMENTS_FTS_COLUMNS)}
                        )
                        """
                    )
                else:
                    self.conn.execute("DELETE FROM attachments_fts")
                self._backfill_attachments_fts()
            except sqlite3.Error:
                logger.exception("attachments_fts rebuild failed")
                return 0
            return int(self.conn.execute("SELECT count(*) FROM attachments_fts").fetchone()[0])

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

    def find_active_job(
        self,
        *,
        kinds: Iterable[str] = ("ingest", "reembed"),
        statuses: Iterable[str] = ("created", "running"),
        exclude_job_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the newest active pipeline job of the given kinds, if any.

        "Active" means the job is between creation and a terminal state:
        ``created`` (the pre-running window, whose inclusion closes the
        check-then-create TOCTOU) or ``running``. ``planned``/``paused`` jobs
        are deliberately excluded so plan-only jobs and resumes are not
        blocked. Shared by :meth:`create_job_if_no_active`, the runner-lock
        stale-run sweep, and resume paths.
        """
        kinds = tuple(kinds)
        statuses = tuple(statuses)
        query = """
            SELECT job_id, kind, status, created_at, updated_at, payload_json
            FROM pipeline_jobs
            WHERE kind IN (%s) AND status IN (%s)
        """ % (
            ", ".join("?" for _ in kinds),
            ", ".join("?" for _ in statuses),
        )
        params: list[Any] = [*kinds, *statuses]
        if exclude_job_id is not None:
            query += " AND job_id != ?"
            params.append(exclude_job_id)
        query += " ORDER BY updated_at DESC LIMIT 1"
        row = self.conn.execute(query, params).fetchone()
        return decode_pipeline_job_row(row) if row is not None else None

    def create_job_if_no_active(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        active_kinds: Iterable[str] = ("ingest", "reembed"),
        active_statuses: Iterable[str] = ("created", "running"),
    ) -> str:
        """Atomically create a job only when no job of ``active_kinds`` is active.

        M10: the connection is in autocommit mode (``isolation_level=None``),
        so the check-then-insert only becomes a cross-process transaction with
        an explicit ``BEGIN IMMEDIATE``. ``with self.conn`` holds the shared
        RLock for the whole block (serializing in-process threads) and its
        ``__exit__`` rolls back safely on error, so a failed BEGIN cannot mask
        the original exception with a spurious rollback error.

        Note: while this transaction is open, do not call ledger methods that
        wrap themselves in ``with self.conn`` — the inner ``__exit__`` would
        commit the outer transaction early. ``find_active_job`` is safe (it
        only issues a bare SELECT).
        """
        kinds = tuple(active_kinds)
        statuses = tuple(active_statuses)
        job_id = str(uuid.uuid4())
        now = utc_now()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            active = self.find_active_job(kinds=kinds, statuses=statuses)
            if active is not None:
                raise RuntimeError(
                    f"an active {active['kind']} job exists "
                    f"(job_id={active['job_id']}, status={active['status']}). "
                    "Wait for it to finish or mark it failed before starting a new one."
                )
            self.conn.execute(
                """
                INSERT INTO pipeline_jobs(job_id, kind, status, created_at, updated_at, payload_json)
                VALUES (?, ?, 'created', ?, ?, ?)
                """,
                (job_id, kind, now, now, payload_json),
            )
            self.conn.execute("COMMIT")
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
                # Read existing defaults inside the same transaction so that a
                # concurrent activation cannot change the flags between read and
                # write. The whole batch is committed atomically.
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
            backend = profile_json.get("backend", "lancedb")
            profiles.append(
                {
                    **dict(row),
                    "enabled": bool(row["enabled"]),
                    "default_for_text": bool(row["default_for_text"]),
                    "default_for_multimodal": bool(row["default_for_multimodal"]),
                    "backend": backend,
                    "profile": profile_json,
                }
            )
        return profiles

    def activate_embedding_profile(self, profile_name: str, mode: str) -> dict[str, Any]:
        if mode not in {"text", "multimodal"}:
            raise ValueError("mode must be 'text' or 'multimodal'")
        expected_modality = "text" if mode == "text" else "multimodal"
        flag_column = "default_for_text" if mode == "text" else "default_for_multimodal"

        with self.conn:
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
        with self.conn:
            existing = self.conn.execute(
                """
                SELECT created_at
                FROM embedding_batches
                WHERE batch_hash = ?
                """,
                (batch_hash,),
            ).fetchone()
            created_at = existing["created_at"] if existing is not None else now
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
                creators_json = json.dumps(
                    item.get("authors", []), ensure_ascii=False, sort_keys=True, default=str
                )
                collections_json = json.dumps(
                    item.get("collections", []), ensure_ascii=False, sort_keys=True, default=str
                )
                tags_json = json.dumps(
                    item.get("tags", []), ensure_ascii=False, sort_keys=True, default=str
                )
                date_year, date_normalized = normalize_date(item.get("date"))
                self.conn.execute(
                    """
                    INSERT INTO attachments(
                        attachment_key, parent_key, content_type, relative_path,
                        title, abstract, date_value, url, classification,
                        source_quality, reasons_json, file_path, file_exists,
                        file_size, file_mtime, metadata_json, content_fingerprint,
                        scan_status, first_seen_at, last_seen_at, updated_at,
                        creators_json, collections_json, tags_json, item_type,
                        publication_title, doi, date_added, date_modified,
                        date_year, date_normalized
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        updated_at = excluded.updated_at,
                        creators_json = excluded.creators_json,
                        collections_json = excluded.collections_json,
                        tags_json = excluded.tags_json,
                        item_type = excluded.item_type,
                        publication_title = excluded.publication_title,
                        doi = excluded.doi,
                        date_added = excluded.date_added,
                        date_modified = excluded.date_modified,
                        date_year = excluded.date_year,
                        date_normalized = excluded.date_normalized
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
                        creators_json,
                        collections_json,
                        tags_json,
                        item.get("item_type"),
                        item.get("publication_title"),
                        item.get("doi"),
                        item.get("date_added"),
                        item.get("date_modified"),
                        date_year,
                        date_normalized,
                    ),
                )
                rowid_row = self.conn.execute(
                    "SELECT rowid FROM attachments WHERE attachment_key = ?",
                    (item["attachment_key"],),
                ).fetchone()
                self.conn.execute(
                    "DELETE FROM attachments_fts WHERE rowid = ?",
                    (rowid_row["rowid"],),
                )
                creators_text, collections_text, tags_text = _biblio_fts_text(
                    creators_json, collections_json, tags_json
                )
                self.conn.execute(
                    _ATTACHMENTS_FTS_INSERT_SQL,
                    (
                        rowid_row["rowid"],
                        item["attachment_key"],
                        item.get("parent_key"),
                        item.get("relative_path"),
                        item.get("title"),
                        item.get("abstract"),
                        item.get("date"),
                        item.get("url"),
                        item["classification"],
                        item.get("source_quality", "unknown"),
                        creators_text,
                        collections_text,
                        tags_text,
                        item.get("publication_title"),
                        item.get("doi"),
                    ),
                )
                count += 1
        return count

    def list_attachments(
        self,
        classification: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = f"""
            SELECT {_ATTACHMENT_SELECT_COLUMNS}
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
            f"""
            SELECT {_ATTACHMENT_SELECT_COLUMNS}
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
        date_from: str | None = None,
        date_to: str | None = None,
        authors: list[str] | None = None,
        collections: list[str] | None = None,
        tags: list[str] | None = None,
        item_type: str | None = None,
        publication_title: str | None = None,
        doi: str | None = None,
        attachment_keys: list[str] | None = None,
        sort_by: str = "score",
        sort_order: str = "desc",
        match_mode: str = "all",
    ) -> list[dict[str, Any]]:
        """Metadata search with a text path and a pure-filter path.

        * Text path (query has usable terms): FTS5 ``MATCH`` (LIKE fallback) to
          gather candidate keys, then structured filters are applied, then
          sorting/truncation. ``sort_by="bm25"`` is only honoured here.
        * Pure-filter path (empty query but at least one filter or
          ``attachment_keys``): structured SQL over ``attachments`` /
          ``attachment_collections`` without touching FTS.
        * Empty query and no filters returns ``[]`` (never scans the library).

        ``collections`` filters exclusively through ``attachment_collections``
        (values are collection keys). ``date_from`` / ``date_to`` compare against
        ``date_normalized``; dirty, unnormalized dates may be missed.
        """

        if sort_by not in _METADATA_SORT_FIELDS:
            raise ValueError(f"invalid sort_by: {sort_by}")
        if sort_order not in {"asc", "desc"}:
            raise ValueError(f"invalid sort_order: {sort_order}")

        from ..search.query_parser import build_fts_query, parse_query, scoring_terms

        plan = parse_query(query, match_mode=match_mode)
        terms = scoring_terms(plan)
        has_query = not plan.is_empty
        filters = _MetadataFilters(
            classification=classification,
            date_from=date_from,
            date_to=date_to,
            authors=[a for a in (authors or []) if a and a.strip()],
            collections=[c for c in (collections or []) if c and c.strip()],
            tags=[t for t in (tags or []) if t and t.strip()],
            item_type=item_type,
            publication_title=publication_title,
            doi=doi,
            attachment_keys=None if attachment_keys is None else list(attachment_keys),
        )
        has_filters = filters.has_any()

        if not has_query and not has_filters:
            return []

        bm25_scores: dict[str, float] = {}
        candidate_keys: list[str] | None = None

        if has_query:
            want_bm25 = sort_by == "bm25"
            candidate_keys, bm25_scores = self._fts_candidate_keys(
                build_fts_query(plan), terms, limit=limit, want_bm25=want_bm25
            )
            if not candidate_keys:
                # FTS failed (None) or returned no matches ([]). Fall back to a
                # LIKE candidate scan so substring recall matches the historic
                # behaviour; bm25 scoring is unavailable on this path.
                candidate_keys = self._like_candidate_keys(terms, filters, limit=limit)
                bm25_scores = {}
        else:
            # Pure-filter path cannot use bm25; fall back to date_added desc.
            if sort_by == "bm25":
                sort_by = "date_added"
                sort_order = "desc"

        rows = self._load_filtered_attachments(filters, candidate_keys=candidate_keys)
        results: list[dict[str, Any]] = []
        for row in rows:
            item = decode_attachment_row(row)
            if not _passes_python_filters(item, filters):
                continue
            if sort_by == "bm25":
                item["score"] = bm25_scores.get(item["attachment_key"], 0.0)
            else:
                item["score"] = metadata_match_score(item, terms)
            results.append(item)

        _sort_metadata_results(results, sort_by=sort_by, sort_order=sort_order)
        return results[:limit]

    def _fts_candidate_keys(
        self,
        fts_query: str,
        terms: list[str],
        *,
        limit: int,
        want_bm25: bool,
    ) -> tuple[list[str] | None, dict[str, float]]:
        """Return candidate keys from FTS (and bm25 scores) or ``None`` on error.

        bm25 is exposed larger-is-better by negating SQLite's ``bm25()`` (which
        returns smaller-is-better values).
        """

        if not fts_query:
            return [], {}
        try:
            if want_bm25:
                rows = self.conn.execute(
                    """
                    SELECT attachment_key, bm25(attachments_fts) AS rank
                    FROM attachments_fts
                    WHERE attachments_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, max(limit * 10, limit)),
                ).fetchall()
                keys = [row["attachment_key"] for row in rows]
                scores = {row["attachment_key"]: -float(row["rank"]) for row in rows}
                return keys, scores
            rows = self.conn.execute(
                """
                SELECT attachment_key
                FROM attachments_fts
                WHERE attachments_fts MATCH ?
                LIMIT ?
                """,
                (fts_query, max(limit * 10, limit)),
            ).fetchall()
            return [row["attachment_key"] for row in rows], {}
        except sqlite3.Error:
            logger.exception("FTS5 metadata search failed; falling back to LIKE")
            return None, {}

    def _like_candidate_keys(
        self,
        terms: list[str],
        filters: "_MetadataFilters",
        *,
        limit: int,
    ) -> list[str]:
        if not terms:
            return []
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
            "publication_title",
            "doi",
        )
        where_parts = []
        params: list[Any] = []
        for term in terms:
            like = f"%{term}%"
            per_term = " OR ".join(
                f"lower(coalesce({column}, '')) LIKE ?" for column in searchable_columns
            )
            where_parts.append(f"({per_term})")
            params.extend([like] * len(searchable_columns))
        sql = f"SELECT attachment_key FROM attachments WHERE {' AND '.join(where_parts)} LIMIT ?"
        params.append(max(limit * 20, limit))
        rows = self.conn.execute(sql, params).fetchall()
        return [row["attachment_key"] for row in rows]

    def _load_filtered_attachments(
        self,
        filters: "_MetadataFilters",
        *,
        candidate_keys: list[str] | None,
    ) -> list[sqlite3.Row]:
        where: list[str] = []
        params: list[Any] = []
        if candidate_keys is not None:
            if not candidate_keys:
                return []
            placeholders = ",".join("?" for _ in candidate_keys)
            where.append(f"attachment_key IN ({placeholders})")
            params.extend(candidate_keys)
        if filters.attachment_keys is not None:
            if not filters.attachment_keys:
                return []
            placeholders = ",".join("?" for _ in filters.attachment_keys)
            where.append(f"attachment_key IN ({placeholders})")
            params.extend(filters.attachment_keys)
        if filters.classification is not None:
            where.append("classification = ?")
            params.append(filters.classification)
        if filters.item_type is not None:
            where.append("lower(coalesce(item_type, '')) = ?")
            params.append(filters.item_type.casefold())
        if filters.publication_title is not None:
            where.append("lower(coalesce(publication_title, '')) LIKE ?")
            params.append(f"%{filters.publication_title.casefold()}%")
        if filters.doi is not None:
            where.append("lower(coalesce(doi, '')) LIKE ?")
            params.append(f"%{filters.doi.casefold()}%")
        if filters.date_from is not None:
            where.append("date_normalized IS NOT NULL AND date_normalized >= ?")
            params.append(_date_bound(filters.date_from, upper=False))
        if filters.date_to is not None:
            where.append("date_normalized IS NOT NULL AND date_normalized <= ?")
            params.append(_date_bound(filters.date_to, upper=True))
        if filters.collections:
            placeholders = ",".join("?" for _ in filters.collections)
            where.append(
                f"attachment_key IN (SELECT attachment_key FROM attachment_collections "
                f"WHERE collection_key IN ({placeholders}))"
            )
            params.extend(filters.collections)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        # Cap the scan so a pure-filter path cannot materialize the whole table
        # when Python-side author/tag filtering still needs to run afterwards.
        sql = f"SELECT {_ATTACHMENT_SELECT_COLUMNS} FROM attachments {where_sql} LIMIT ?"
        params.append(_METADATA_SCAN_CAP)
        return self.conn.execute(sql, params).fetchall()

    def upsert_collections(self, collections: Iterable[dict[str, Any]]) -> int:
        count = 0
        with self.conn:
            for collection in collections:
                library_id = collection.get("library_id")
                self.conn.execute(
                    """
                    INSERT INTO zotero_collections(
                        collection_key, collection_name, parent_collection_key, library_id
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(collection_key) DO UPDATE SET
                        collection_name = excluded.collection_name,
                        parent_collection_key = excluded.parent_collection_key,
                        library_id = excluded.library_id
                    """,
                    (
                        collection["collection_key"],
                        collection.get("collection_name") or "",
                        collection.get("parent_collection_key"),
                        int(library_id) if library_id is not None else None,
                    ),
                )
                count += 1
        return count

    def replace_attachment_collections(
        self, attachment_key: str, collection_keys: Iterable[str]
    ) -> None:
        keys = sorted({k for k in collection_keys if k})
        with self.conn:
            self.conn.execute(
                "DELETE FROM attachment_collections WHERE attachment_key = ?",
                (attachment_key,),
            )
            for collection_key in keys:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO attachment_collections(attachment_key, collection_key)
                    VALUES (?, ?)
                    """,
                    (attachment_key, collection_key),
                )

    def list_ledger_collections(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT collection_key, collection_name, parent_collection_key, library_id
            FROM zotero_collections
            ORDER BY collection_name, collection_key
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def attachment_keys_for_collections(self, collection_keys: Iterable[str]) -> list[str]:
        keys = sorted({k for k in collection_keys if k})
        if not keys:
            return []
        placeholders = ",".join("?" for _ in keys)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT attachment_key
            FROM attachment_collections
            WHERE collection_key IN ({placeholders})
            ORDER BY attachment_key
            """,
            keys,
        ).fetchall()
        return [row["attachment_key"] for row in rows]

    def prune_collections(self, present_collection_keys: Iterable[str]) -> int:
        keys = sorted({k for k in present_collection_keys if k})
        with self.conn:
            if not keys:
                cursor = self.conn.execute("DELETE FROM zotero_collections")
                return int(cursor.rowcount)
            placeholders = ",".join("?" for _ in keys)
            cursor = self.conn.execute(
                f"DELETE FROM zotero_collections WHERE collection_key NOT IN ({placeholders})",
                keys,
            )
            return int(cursor.rowcount)

    def prune_attachment_collections(self, present_attachment_keys: Iterable[str]) -> int:
        """Drop association rows for attachments no longer present in a full scan.

        ``mark_absent_attachments_deleted`` never touches attachment_collections,
        so the scanner must prune stale associations explicitly.
        """

        keys = sorted({k for k in present_attachment_keys if k})
        with self.conn:
            if not keys:
                cursor = self.conn.execute("DELETE FROM attachment_collections")
                return int(cursor.rowcount)
            placeholders = ",".join("?" for _ in keys)
            cursor = self.conn.execute(
                f"DELETE FROM attachment_collections WHERE attachment_key NOT IN ({placeholders})",
                keys,
            )
            return int(cursor.rowcount)

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
        with self.conn:
            existing = self.get_extract_job(job_id=job_id)
            if existing is None:
                raise KeyError(f"extract job not found: {job_id}")
            payload_json = json.dumps(
                payload if payload is not None else existing.get("payload", {}),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
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
            self.conn.execute(
                """
                DELETE FROM chunks_fts
                WHERE rowid IN (SELECT rowid FROM chunks WHERE document_id = ?)
                """,
                (document_id,),
            )
            self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            for chunk in chunks:
                cursor = self.conn.execute(
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
                self.conn.execute(
                    """
                    INSERT INTO chunks_fts(rowid, chunk_id, document_id, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cursor.lastrowid, chunk["chunk_id"], document_id, chunk.get("text", "")),
                )
                count += 1
        return count

    def upsert_artifact_and_chunks(
        self,
        artifact: dict[str, Any],
        chunks: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        """Persist a normalized artifact and its chunks in one transaction.

        An interrupted normalize stage cannot leave the state DB with an
        artifact but no chunks (or chunks pointing to a missing artifact). The
        caller is responsible for the on-disk normalized files; this method
        makes the ledger update atomic.
        """

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
            self.conn.execute(
                """
                DELETE FROM chunks_fts
                WHERE rowid IN (SELECT rowid FROM chunks WHERE document_id = ?)
                """,
                (artifact["document_id"],),
            )
            self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (artifact["document_id"],))
            for chunk in chunks:
                cursor = self.conn.execute(
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
                        artifact["document_id"],
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
                self.conn.execute(
                    """
                    INSERT INTO chunks_fts(rowid, chunk_id, document_id, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cursor.lastrowid, chunk["chunk_id"], artifact["document_id"], chunk.get("text", "")),
                )
        return self.get_normalized_artifact(artifact["document_id"])  # type: ignore[return-value]

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
        sort_by: str = "score",
        match_mode: str = "all",
    ) -> list[dict[str, Any]]:
        if sort_by not in {"score", "bm25"}:
            raise ValueError(f"invalid sort_by: {sort_by}")

        from ..search.query_parser import build_fts_query, parse_query, scoring_terms

        plan = parse_query(query, match_mode=match_mode)
        terms = scoring_terms(plan)
        if plan.is_empty:
            return []

        def _load_chunks_by_ids(chunk_ids: list[str]) -> dict[str, sqlite3.Row]:
            if not chunk_ids:
                return {}
            where = f"chunk_id IN ({','.join('?' for _ in chunk_ids)})"
            params: list[Any] = list(chunk_ids)
            if chunk_type is not None:
                where += " AND chunk_type = ?"
                params.append(chunk_type)
            sql = f"""
                SELECT chunk_id, document_id, chunk_type, chunk_index, text,
                       heading_path_json, prev_chunk_id, next_chunk_id,
                       metadata_json, updated_at
                FROM chunks
                WHERE {where}
                LIMIT ?
            """
            params.append(max(limit * 10, limit))
            return {row["chunk_id"]: row for row in self.conn.execute(sql, params).fetchall()}

        try:
            fts_query = build_fts_query(plan)
            if fts_query:
                if sort_by == "bm25":
                    fts_rows = self.conn.execute(
                        """
                        SELECT chunk_id, bm25(chunks_fts) AS rank
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, max(limit * 10, limit)),
                    ).fetchall()
                    bm25_scores = {row["chunk_id"]: -float(row["rank"]) for row in fts_rows}
                    chunk_ids = [row["chunk_id"] for row in fts_rows]
                else:
                    fts_rows = self.conn.execute(
                        """
                        SELECT chunk_id
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ?
                        LIMIT ?
                        """,
                        (fts_query, max(limit * 10, limit)),
                    ).fetchall()
                    bm25_scores = {}
                    chunk_ids = [row["chunk_id"] for row in fts_rows]
                if chunk_ids:
                    by_id = _load_chunks_by_ids(chunk_ids)
                    results = []
                    for chunk_id in chunk_ids:
                        row = by_id.get(chunk_id)
                        if row is None:
                            continue
                        item = decode_chunk_row(row)
                        if sort_by == "bm25":
                            item["score"] = bm25_scores.get(chunk_id, 0.0)
                        else:
                            item["score"] = chunk_match_score(item, terms)
                        results.append(item)
                    if sort_by == "bm25":
                        results.sort(key=lambda item: item["score"], reverse=True)
                    else:
                        results.sort(
                            key=lambda item: (item["score"], item["document_id"], -item["chunk_index"]),
                            reverse=True,
                        )
                    return results[:limit]
        except sqlite3.Error:
            logger.exception("FTS5 fulltext search failed; falling back to LIKE")

        # LIKE fallback (bm25 is unavailable without FTS; degrade to chunk score).
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
                self.conn.execute(
                    """
                    DELETE FROM attachments_fts
                    WHERE rowid IN (SELECT rowid FROM attachments WHERE classification = 'deleted')
                    """
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
            self.conn.execute(
                f"""
                DELETE FROM attachments_fts
                WHERE rowid IN (
                    SELECT rowid FROM attachments
                    WHERE attachment_key NOT IN ({placeholders})
                      AND classification = 'deleted'
                )
                """,
                list(keys),
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


def _build_fts5_query(terms: list[str]) -> str:
    """Build an FTS5 MATCH query that requires every term to appear.

    Terms are quoted and joined with AND. Empty or whitespace-only terms are
    dropped. This matches the existing LIKE behavior where each word in the
    query must be present somewhere in the searchable text.
    """

    escaped = [term.replace('"', '""') for term in terms if term.strip()]
    if not escaped:
        return ""
    return " AND ".join(f'"{term}"' for term in escaped)


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


def _creators_text(item: dict[str, Any]) -> str:
    creators = item.get("creators")
    if isinstance(creators, list):
        return " ".join(str(name) for name in creators if name)
    return ""


def _tags_text(item: dict[str, Any]) -> str:
    tags = item.get("tags")
    if isinstance(tags, list):
        return " ".join(str(tag) for tag in tags if tag)
    return ""


def _collections_text(item: dict[str, Any]) -> str:
    collections = item.get("collections")
    if isinstance(collections, list):
        names = []
        for element in collections:
            if isinstance(element, dict):
                name = element.get("collection_name")
                if name:
                    names.append(str(name))
            elif element:
                names.append(str(element))
        return " ".join(names)
    return ""


def metadata_match_score(item: dict[str, Any], terms: list[str]) -> float:
    # Scoring uses derived plain-text fields (never raw JSON strings) so author
    # names, tag labels, and collection names contribute without matching JSON
    # punctuation noise like ``["``.
    derived = {
        "creators_text": _creators_text(item),
        "tags_text": _tags_text(item),
        "collections_text": _collections_text(item),
    }
    weighted_fields = (
        ("title", 5.0),
        ("doi", 5.0),
        ("creators_text", 4.0),
        ("publication_title", 3.0),
        ("tags_text", 3.0),
        ("abstract", 2.0),
        ("collections_text", 1.5),
        ("relative_path", 1.5),
        ("url", 1.0),
        ("date_value", 1.0),
        ("attachment_key", 1.0),
        ("parent_key", 1.0),
    )
    score = 0.0
    for field, weight in weighted_fields:
        value = str(derived.get(field, item.get(field)) or "").casefold()
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
    # Expose bibliographic list columns as parsed Python lists. The raw *_json
    # columns are dropped from the returned dict to keep callers on structured
    # data instead of JSON strings.
    item["creators"] = _load_json_list(item.pop("creators_json", None))
    item["collections"] = _load_json_list(item.pop("collections_json", None))
    item["tags"] = _load_json_list(item.pop("tags_json", None))
    return item


def _load_json_list(raw: Any) -> list[Any]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return data if isinstance(data, list) else []


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
