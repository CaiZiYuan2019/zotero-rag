from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sqlite3
import time
import uuid


@dataclass(frozen=True)
class ZoteroAttachment:
    parent_key: str | None
    attachment_key: str
    content_type: str | None
    relative_path: str | None
    title: str | None
    abstract: str | None
    date: str | None
    url: str | None


class ShadowCopyTimeout(TimeoutError):
    """Raised when Zotero is too busy to produce a shadow copy promptly."""


def _readonly_sqlite_uri(path: Path, *, immutable: bool = False) -> str:
    uri = f"file:{path.as_posix()}?mode=ro"
    if immutable:
        uri += "&immutable=1"
    return uri


def create_shadow_copy(
    source_db: str | Path,
    shadow_db: str | Path,
    *,
    timeout_seconds: float = 30.0,
) -> Path:
    """Copy a Zotero database to a shadow database using a read-only source.

    This function never opens the source database in writable mode. It also
    refuses to run when source and destination resolve to the same path.
    """

    source = Path(source_db).expanduser().resolve()
    target = Path(shadow_db).expanduser().resolve()
    if source == target:
        raise ValueError("shadow destination must differ from the Zotero source database")
    if not source.is_file():
        raise FileNotFoundError(f"Zotero database not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")

    try:
        _backup_readonly_source(source, tmp, immutable=True, timeout_seconds=timeout_seconds)
    except (sqlite3.OperationalError, ShadowCopyTimeout):
        _cleanup_shadow_temp(tmp)
        # Some live Zotero databases keep rollback-journal state that cannot be
        # copied through an immutable connection. Retry with SQLite's read-only
        # mode so the source is still never opened for writes, but SQLite can
        # consult its journal sidecar while creating a consistent shadow.
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            _backup_readonly_source(source, tmp, immutable=False, timeout_seconds=timeout_seconds)
        except Exception:
            _cleanup_shadow_temp(tmp)
            raise

    shutil.move(str(tmp), str(target))
    return target


def _cleanup_shadow_temp(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-journal"), Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError:
            pass


def _backup_readonly_source(
    source: Path,
    target: Path,
    *,
    immutable: bool,
    timeout_seconds: float,
) -> None:
    src_conn = sqlite3.connect(_readonly_sqlite_uri(source, immutable=immutable), uri=True, timeout=1)
    started_at = time.monotonic()

    def check_timeout(status: int, remaining: int, total: int) -> None:
        if timeout_seconds > 0 and time.monotonic() - started_at > timeout_seconds:
            mode = "immutable" if immutable else "readonly"
            raise ShadowCopyTimeout(_timeout_message(timeout_seconds, mode))

    try:
        dst_conn = sqlite3.connect(target)
        try:
            # Copy in bounded chunks so active Zotero databases cannot leave the
            # CLI/API hanging indefinitely while SQLite waits on journal state.
            try:
                src_conn.backup(dst_conn, pages=256, progress=check_timeout, sleep=0.05)
            except sqlite3.OperationalError as exc:
                if timeout_seconds > 0 and time.monotonic() - started_at > timeout_seconds:
                    mode = "immutable" if immutable else "readonly"
                    raise ShadowCopyTimeout(_timeout_message(timeout_seconds, mode)) from exc
                raise
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def _timeout_message(timeout_seconds: float, mode: str) -> str:
    return (
        f"Zotero source stayed busy for more than {timeout_seconds:.1f}s during {mode} backup; "
        "close Zotero or retry with a larger shadow-copy timeout"
    )


class ZoteroShadow:
    """Read-only query helper over a copied Zotero shadow database."""

    def __init__(self, shadow_db: str | Path) -> None:
        self.shadow_db = Path(shadow_db)
        self.conn = sqlite3.connect(_readonly_sqlite_uri(self.shadow_db), uri=True)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        self.conn.close()

    def list_attachments(self, limit: int | None = None) -> list[ZoteroAttachment]:
        query = """
            SELECT
                p.key AS parent_key,
                a.key AS attachment_key,
                ia.contentType AS content_type,
                ia.path AS relative_path,
                titlev.value AS title,
                absv.value AS abstract,
                datev.value AS date,
                urlv.value AS url
            FROM itemAttachments ia
            JOIN items a ON ia.itemID = a.itemID
            LEFT JOIN items p ON ia.parentItemID = p.itemID
            LEFT JOIN itemData title ON COALESCE(p.itemID, a.itemID) = title.itemID
                AND title.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
            LEFT JOIN itemDataValues titlev ON title.valueID = titlev.valueID
            LEFT JOIN itemData abs ON COALESCE(p.itemID, a.itemID) = abs.itemID
                AND abs.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'abstractNote')
            LEFT JOIN itemDataValues absv ON abs.valueID = absv.valueID
            LEFT JOIN itemData dt ON COALESCE(p.itemID, a.itemID) = dt.itemID
                AND dt.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'date')
            LEFT JOIN itemDataValues datev ON dt.valueID = datev.valueID
            LEFT JOIN itemData url ON COALESCE(p.itemID, a.itemID) = url.itemID
                AND url.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'url')
            LEFT JOIN itemDataValues urlv ON url.valueID = urlv.valueID
            ORDER BY a.itemID
        """
        params: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        rows = self.conn.execute(query, params).fetchall()
        return [
            ZoteroAttachment(
                parent_key=row["parent_key"],
                attachment_key=row["attachment_key"],
                content_type=row["content_type"],
                relative_path=row["relative_path"],
                title=row["title"],
                abstract=row["abstract"],
                date=row["date"],
                url=row["url"],
            )
            for row in rows
        ]

    def pdf_count(self) -> int:
        row = self.conn.execute(
            "SELECT count(*) AS n FROM itemAttachments WHERE contentType = 'application/pdf'"
        ).fetchone()
        return int(row["n"])
