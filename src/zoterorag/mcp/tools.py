from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..config import AppConfig
from ..db import StateLedger
from ..documents import get_document
from ..embeddings import search_vector_index
from ..runtime import config_as_public_dict
from ..search import fulltext_search, metadata_search


McpConsumer = Literal["llm_text", "llm_multimodal"]
ImageReturn = Literal["file_ref", "base64", "none"]


@dataclass(frozen=True)
class McpToolContext:
    config: AppConfig
    ledger: StateLedger


def zotero_rag_status(context: McpToolContext) -> dict[str, Any]:
    """Return runtime status safe for an external MCP caller."""

    return {
        "runtime": config_as_public_dict(context.config),
        "state": context.ledger.status_summary(),
    }


def zotero_rag_list_models(context: McpToolContext) -> dict[str, Any]:
    return {
        "models": context.ledger.list_embedding_profiles(),
        "vector_indexes": context.ledger.list_vector_indexes(),
    }


def zotero_rag_metadata_search(
    context: McpToolContext,
    *,
    query: str,
    classification: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    return {
        "results": metadata_search(
            context.ledger,
            query=query,
            classification=classification,
            limit=top_k,
            consumer="llm_text",
        ),
        "warnings": [],
    }


def zotero_rag_search_text(
    context: McpToolContext,
    *,
    query: str,
    profile_name: str | None = None,
    top_k: int = 10,
    include_metadata: bool = True,
    include_fulltext: bool = True,
    include_vector: bool = True,
) -> dict[str, Any]:
    """Text-only MCP search.

    The result is always safe for plain-text LLMs: no file paths, no base64, and
    no image payloads. Vector search failures are returned as warnings because a
    fresh library can still use direct metadata/fulltext before vectors exist.
    """

    warnings: list[str] = []
    results: list[dict[str, Any]] = []
    if include_metadata:
        results.extend(
            label_results(
                metadata_search(
                    context.ledger,
                    query=query,
                    limit=top_k,
                    consumer="llm_text",
                ),
                source="metadata",
            )
        )
    if include_fulltext:
        results.extend(
            label_results(
                fulltext_search(
                    context.ledger,
                    query=query,
                    chunk_type="text",
                    limit=top_k,
                    consumer="llm_text",
                    image_return="none",
                ),
                source="fulltext",
            )
        )
    if include_vector:
        try:
            results.extend(
                label_results(
                    search_vector_index(
                        ledger=context.ledger,
                        vector_store_dir=context.config.paths.vector_store_dir,
                        profile_name=profile_name,
                        query=query,
                        mode="text",
                        top_k=top_k,
                        consumer="llm_text",
                        image_return="none",
                    ),
                    source="text_vector",
                )
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            warnings.append(f"text_vector_unavailable:{exc}")
    return {
        "results": results[:top_k],
        "warnings": warnings,
        "consumer": "llm_text",
        "image_return": "none",
    }


def zotero_rag_search_multimodal(
    context: McpToolContext,
    *,
    query_text: str,
    profile_name: str | None = None,
    top_k: int = 10,
    consumer: McpConsumer = "llm_text",
    image_return: ImageReturn = "none",
) -> dict[str, Any]:
    """MCP multimodal search facade.

    `llm_text` remains the default and strips image payloads. A caller must
    explicitly request `consumer='llm_multimodal'` plus `file_ref` or `base64` to
    receive image handles.
    """

    consumer = normalize_mcp_consumer(consumer)
    image_return = normalize_image_return(image_return)
    if consumer == "llm_text":
        image_return = "none"
    try:
        results = search_vector_index(
            ledger=context.ledger,
            vector_store_dir=context.config.paths.vector_store_dir,
            profile_name=profile_name,
            query=query_text,
            mode="multimodal",
            top_k=top_k,
            consumer=consumer,
            image_return=image_return,
        )
        warnings: list[str] = []
    except (FileNotFoundError, KeyError, ValueError) as exc:
        results = []
        warnings = [f"multimodal_vector_unavailable:{exc}"]
    return {
        "results": label_results(results, source="multimodal_vector"),
        "warnings": warnings,
        "consumer": consumer,
        "image_return": image_return,
    }


def zotero_rag_get_document(
    context: McpToolContext,
    *,
    document_id: str,
    include_chunks: bool = True,
    chunk_type: str | None = None,
    limit: int | None = 20,
    consumer: McpConsumer = "llm_text",
) -> dict[str, Any]:
    consumer = normalize_mcp_consumer(consumer)
    document = get_document(
        context.ledger,
        document_id,
        include_chunks=include_chunks,
        chunk_type=chunk_type,
        limit=limit,
        consumer=consumer,
    )
    return {"document": document, "consumer": consumer}


def label_results(results: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    labeled = []
    for result in results:
        item = dict(result)
        metadata = dict(item.get("metadata") or {})
        metadata["retrieval_source"] = source
        item["metadata"] = metadata
        labeled.append(item)
    return labeled


def normalize_mcp_consumer(consumer: str) -> McpConsumer:
    if consumer not in {"llm_text", "llm_multimodal"}:
        raise ValueError("MCP consumer must be 'llm_text' or 'llm_multimodal'")
    return consumer  # type: ignore[return-value]


def normalize_image_return(image_return: str) -> ImageReturn:
    if image_return not in {"file_ref", "base64", "none"}:
        raise ValueError("image_return must be 'file_ref', 'base64', or 'none'")
    return image_return  # type: ignore[return-value]
