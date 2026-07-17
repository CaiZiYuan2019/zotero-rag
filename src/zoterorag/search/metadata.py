from __future__ import annotations

from typing import Any

from ..db import StateLedger
from .results import ensure_rerank_disabled, sanitize_results_for_consumer


def metadata_search(
    ledger: StateLedger,
    query: str,
    *,
    classification: str | None = None,
    limit: int = 10,
    consumer: str = "llm_text",
    rerank: bool = False,
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
    """Search local Zotero metadata without vectors or external services.

    Supports title/author/date/DOI/journal/tag filtering plus collection
    filtering (via ``attachment_keys`` from the collections module). With an
    empty ``query`` but at least one filter, the pure-filter path lists the
    matching partition. ``sort_by`` accepts ``score`` | ``bm25`` | ``date`` |
    ``date_added`` | ``title``; ``bm25`` scores are exposed larger-is-better.
    """

    ensure_rerank_disabled(rerank)
    rows = ledger.search_attachments_metadata(
        query,
        classification=classification,
        limit=limit,
        date_from=date_from,
        date_to=date_to,
        authors=authors,
        collections=collections,
        tags=tags,
        item_type=item_type,
        publication_title=publication_title,
        doi=doi,
        attachment_keys=attachment_keys,
        sort_by=sort_by,
        sort_order=sort_order,
        match_mode=match_mode,
    )
    results = [
        {
            "document_id": row.get("parent_key") or row["attachment_key"],
            "chunk_id": row["attachment_key"],
            "title": row.get("title") or row.get("relative_path") or row["attachment_key"],
            "text": build_metadata_text(row),
            "score": row["score"],
            "metadata": {
                "attachment_key": row["attachment_key"],
                "parent_key": row.get("parent_key"),
                "content_type": row.get("content_type"),
                "classification": row.get("classification"),
                "source_quality": row.get("source_quality"),
                "date": row.get("date_value"),
                "url": row.get("url"),
                "file_exists": row.get("file_exists"),
                "scan_status": row.get("scan_status"),
                "authors": row.get("creators") or [],
                "collections": row.get("collections") or [],
                "tags": row.get("tags") or [],
                "item_type": row.get("item_type"),
                "publication_title": row.get("publication_title"),
                "doi": row.get("doi"),
                "date_added": row.get("date_added"),
                "date_modified": row.get("date_modified"),
            },
            "images": [],
        }
        for row in rows
    ]
    return sanitize_results_for_consumer(results, consumer=consumer, image_return="none")


def build_metadata_text(row: dict[str, Any]) -> str:
    parts = []
    for label, key in (
        ("Title", "title"),
        ("Authors", "_authors_text"),
        ("Date", "date_value"),
        ("Publication", "publication_title"),
        ("DOI", "doi"),
        ("URL", "url"),
        ("Tags", "_tags_text"),
        ("Collections", "_collections_text"),
        ("Abstract", "abstract"),
        ("Attachment", "relative_path"),
    ):
        if key == "_authors_text":
            value = ", ".join(str(name) for name in (row.get("creators") or []) if name)
        elif key == "_tags_text":
            value = ", ".join(str(tag) for tag in (row.get("tags") or []) if tag)
        elif key == "_collections_text":
            value = ", ".join(
                str(collection.get("collection_name") or "")
                for collection in (row.get("collections") or [])
                if isinstance(collection, dict) and collection.get("collection_name")
            )
        else:
            value = row.get(key)
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)
