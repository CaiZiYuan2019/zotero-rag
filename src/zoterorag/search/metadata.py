from __future__ import annotations

from typing import Any

from ..db import StateLedger
from .results import sanitize_results_for_consumer


def metadata_search(
    ledger: StateLedger,
    query: str,
    *,
    classification: str | None = None,
    limit: int = 10,
    consumer: str = "llm_text",
) -> list[dict[str, Any]]:
    """Search local Zotero metadata without vectors or external services."""

    rows = ledger.search_attachments_metadata(
        query,
        classification=classification,
        limit=limit,
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
        ("Date", "date_value"),
        ("URL", "url"),
        ("Abstract", "abstract"),
        ("Attachment", "relative_path"),
    ):
        value = row.get(key)
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)
