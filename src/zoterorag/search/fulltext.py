from __future__ import annotations

from typing import Any

from ..db import StateLedger
from .results import SearchResult, ensure_rerank_disabled, sanitize_results_for_consumer


def fulltext_search(
    ledger: StateLedger,
    query: str,
    *,
    chunk_type: str | None = None,
    limit: int = 10,
    consumer: str = "llm_text",
    image_return: str = "none",
    rerank: bool = False,
) -> list[dict[str, Any]]:
    """Search normalized Markdown chunks without embeddings or external APIs."""

    ensure_rerank_disabled(rerank)
    rows = ledger.search_chunks_fulltext(query, chunk_type=chunk_type, limit=limit)
    results = []
    for row in rows:
        images = []
        metadata = dict(row.get("metadata") or {})
        if row["chunk_type"] == "image":
            images.append(
                {
                    "image_id": row["chunk_id"],
                    "caption": row["text"],
                    "file_ref": metadata.get("image_path"),
                }
            )
        results.append(
            SearchResult(
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                title=row["document_id"],
                text=row["text"],
                score=float(row.get("score", 0.0)),
                metadata={
                    **metadata,
                    "chunk_type": row["chunk_type"],
                    "heading_path": row.get("heading_path", []),
                    "updated_at": row.get("updated_at"),
                },
                images=images,
            )
        )
    return sanitize_results_for_consumer(
        results,
        consumer=consumer,  # type: ignore[arg-type]
        image_return=image_return,  # type: ignore[arg-type]
    )
