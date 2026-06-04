from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sqlite3


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


def _readonly_sqlite_uri(path: Path, *, immutable: bool = False) -> str:
    uri = f"file:{path.as_posix()}?mode=ro"
    if immutable:
        uri += "&immutable=1"
    return uri


def create_shadow_copy(source_db: str | Path, shadow_db: str | Path) -> Path:
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
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    # Zotero often leaves rollback-journal sidecars around. Opening the source
    # as immutable prevents SQLite from attempting lock or journal recovery
    # operations on the primary Zotero database. This is acceptable for a
    # point-in-time shadow scanner because the source is treated as immutable
    # during this copy operation and never written by this process.
    src_conn = sqlite3.connect(_readonly_sqlite_uri(source, immutable=True), uri=True, timeout=1)
    try:
        dst_conn = sqlite3.connect(tmp)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()

    shutil.move(str(tmp), str(target))
    return target


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
