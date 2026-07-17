from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import sqlite3
import time
import uuid


def _split_concat(value: str | None) -> list[str]:
    """Split a GROUP_CONCAT('|') aggregate into stripped, non-empty parts."""
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


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
    authors: list[str] = field(default_factory=list)
    collections: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    item_type: str | None = None
    publication_title: str | None = None
    doi: str | None = None
    date_added: str | None = None
    date_modified: str | None = None


@dataclass(frozen=True)
class ZoteroCollection:
    collection_key: str
    collection_name: str
    parent_collection_key: str | None
    library_id: int | None


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
        except (sqlite3.OperationalError, ShadowCopyTimeout, OSError):
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
        # Multi-valued fields (authors/collections/tags) are aggregated in
        # correlated subqueries so they cannot fan out the driving row into a
        # cartesian product. GROUP_CONCAT uses "|" because author names and
        # collection names routinely contain commas. Collections/tags/creators
        # attach to the parent item when present, falling back to the
        # attachment's own item id for orphan attachments.
        query = """
            SELECT
                p.key AS parent_key,
                a.key AS attachment_key,
                ia.contentType AS content_type,
                ia.path AS relative_path,
                titlev.value AS title,
                absv.value AS abstract,
                datev.value AS date,
                urlv.value AS url,
                pubv.value AS publication_title,
                doiv.value AS doi,
                it.typeName AS item_type,
                COALESCE(p.dateAdded, a.dateAdded) AS date_added,
                COALESCE(p.dateModified, a.dateModified) AS date_modified,
                (
                    SELECT GROUP_CONCAT(nm, '|') FROM (
                        SELECT TRIM(COALESCE(cr.firstName, '') || ' ' || COALESCE(cr.lastName, '')) AS nm
                        FROM itemCreators ic
                        JOIN creators cr ON ic.creatorID = cr.creatorID
                        JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
                        WHERE ic.itemID = COALESCE(p.itemID, a.itemID)
                          AND ct.creatorType = 'author'
                        ORDER BY ic.orderIndex
                    )
                ) AS authors,
                (
                    SELECT GROUP_CONCAT(ck, '|') FROM (
                        SELECT c.key AS ck
                        FROM collectionItems ci
                        JOIN collections c ON ci.collectionID = c.collectionID
                        WHERE ci.itemID = COALESCE(p.itemID, a.itemID)
                        ORDER BY c.key
                    )
                ) AS collection_keys,
                (
                    SELECT GROUP_CONCAT(cn, '|') FROM (
                        SELECT c.collectionName AS cn
                        FROM collectionItems ci
                        JOIN collections c ON ci.collectionID = c.collectionID
                        WHERE ci.itemID = COALESCE(p.itemID, a.itemID)
                        ORDER BY c.key
                    )
                ) AS collection_names,
                (
                    SELECT GROUP_CONCAT(pk, '|') FROM (
                        SELECT COALESCE(pc.key, '') AS pk
                        FROM collectionItems ci
                        JOIN collections c ON ci.collectionID = c.collectionID
                        LEFT JOIN collections pc ON c.parentCollectionID = pc.collectionID
                        WHERE ci.itemID = COALESCE(p.itemID, a.itemID)
                        ORDER BY c.key
                    )
                ) AS collection_parents,
                (
                    SELECT GROUP_CONCAT(tn, '|') FROM (
                        SELECT t.name AS tn
                        FROM itemTags itg
                        JOIN tags t ON itg.tagID = t.tagID
                        WHERE itg.itemID = COALESCE(p.itemID, a.itemID)
                        ORDER BY t.name
                    )
                ) AS tags
            FROM itemAttachments ia
            JOIN items a ON ia.itemID = a.itemID
            LEFT JOIN items p ON ia.parentItemID = p.itemID
            LEFT JOIN itemTypes it ON p.itemTypeID = it.itemTypeID
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
            LEFT JOIN itemData pub ON COALESCE(p.itemID, a.itemID) = pub.itemID
                AND pub.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'publicationTitle')
            LEFT JOIN itemDataValues pubv ON pub.valueID = pubv.valueID
            LEFT JOIN itemData doi ON COALESCE(p.itemID, a.itemID) = doi.itemID
                AND doi.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'DOI')
            LEFT JOIN itemDataValues doiv ON doi.valueID = doiv.valueID
            ORDER BY a.itemID
        """
        params: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_attachment(row) for row in rows]

    @staticmethod
    def _row_to_attachment(row: sqlite3.Row) -> ZoteroAttachment:
        keys = _split_concat(row["collection_keys"])
        names = _split_concat(row["collection_names"])
        # collection_parents keeps empty slots so it cannot be filtered with the
        # non-empty splitter; align it positionally with keys/names instead.
        raw_parents = (row["collection_parents"] or "").split("|")
        collections: list[dict[str, str]] = []
        for idx, ckey in enumerate(keys):
            name = names[idx] if idx < len(names) else ""
            parent = raw_parents[idx].strip() if idx < len(raw_parents) else ""
            collections.append(
                {
                    "collection_key": ckey,
                    "collection_name": name,
                    "parent_collection_key": parent or None,
                }
            )
        return ZoteroAttachment(
            parent_key=row["parent_key"],
            attachment_key=row["attachment_key"],
            content_type=row["content_type"],
            relative_path=row["relative_path"],
            title=row["title"],
            abstract=row["abstract"],
            date=row["date"],
            url=row["url"],
            authors=_split_concat(row["authors"]),
            collections=collections,
            tags=_split_concat(row["tags"]),
            item_type=row["item_type"],
            publication_title=row["publication_title"],
            doi=row["doi"],
            date_added=row["date_added"],
            date_modified=row["date_modified"],
        )

    def list_collections(self) -> list[ZoteroCollection]:
        """Return every Zotero collection definition for full ledger sync.

        Collection definitions are read once here rather than expanded on each
        attachment row so the shadow aggregate query stays cheap.
        """
        rows = self.conn.execute(
            """
            SELECT
                c.key AS collection_key,
                c.collectionName AS collection_name,
                pc.key AS parent_collection_key,
                c.libraryID AS library_id
            FROM collections c
            LEFT JOIN collections pc ON c.parentCollectionID = pc.collectionID
            ORDER BY c.key
            """
        ).fetchall()
        return [
            ZoteroCollection(
                collection_key=row["collection_key"],
                collection_name=row["collection_name"],
                parent_collection_key=row["parent_collection_key"],
                library_id=row["library_id"],
            )
            for row in rows
        ]

    def pdf_count(self) -> int:
        row = self.conn.execute(
            "SELECT count(*) AS n FROM itemAttachments WHERE contentType = 'application/pdf'"
        ).fetchone()
        return int(row["n"])
