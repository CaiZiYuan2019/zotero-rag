from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..config import AppConfig
from ..db import StateLedger
from ..documents import get_document
from ..embeddings import search_vector_index
from ..models import list_embedding_model_catalog
from ..search import fulltext_search, metadata_search, normalize_query_image
from ..search.results import RerankNotSupportedError, ensure_rerank_disabled


McpConsumer = Literal["llm_text", "llm_multimodal"]
ImageReturn = Literal["file_ref", "base64", "none"]


@dataclass(frozen=True)
class McpToolContext:
    config: AppConfig
    ledger: StateLedger


def zotero_rag_status(context: McpToolContext) -> dict[str, Any]:
    """Return a concise build status summary safe for an external MCP caller."""

    summary = context.ledger.status_summary()
    attachments = summary.get("attachments", {})
    extract_jobs = summary.get("extract_jobs", {})
    batches = summary.get("embedding_batches", {})
    chunks = summary.get("chunks", {})

    # Profile-level searchable document counts.
    indexes = {idx["profile_name"]: idx for idx in context.ledger.list_vector_indexes()}
    text_idx = indexes.get("qwen3vl_cloud_2560_text", {})
    mm_idx = indexes.get("qwen3vl_cloud_2560_multimodal", {})

    return {
        "library": {
            "total_attachments": sum(attachments.values()),
            "buildable": attachments.get("included_auto", 0),
            "needs_review": attachments.get("needs_review", 0),
        },
        "build": {
            "extracted": extract_jobs.get("downloaded", 0),
            "normalized": summary.get("normalized_artifacts", 0),
            "text_indexed": text_idx.get("document_count", 0),
            "multimodal_indexed": mm_idx.get("document_count", 0),
            "text_chunks": chunks.get("text", 0),
            "image_chunks": chunks.get("image", 0),
            "failed_retryable": extract_jobs.get("failed_retryable", 0),
            "running_extract": extract_jobs.get("running", 0) + extract_jobs.get("submitted", 0),
            "completed_embedding_batches": batches.get("completed", 0),
        },
        "profiles": [p["name"] for p in context.ledger.list_embedding_profiles()],
    }


def zotero_rag_list_models(context: McpToolContext) -> dict[str, Any]:
    return list_embedding_model_catalog(context.ledger)


def zotero_rag_metadata_search(
    context: McpToolContext,
    *,
    query: str,
    classification: str | None = None,
    top_k: int = 10,
    rerank: bool = False,
) -> dict[str, Any]:
    try:
        ensure_rerank_disabled(rerank)
    except RerankNotSupportedError as exc:
        return {"results": [], "warnings": [str(exc)]}
    return {
        "results": metadata_search(
            context.ledger,
            query=query,
            classification=classification,
            limit=top_k,
            consumer="llm_text",
            rerank=False,
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
    rerank: bool = False,
) -> dict[str, Any]:
    """Text-only MCP search.

    The result is always safe for plain-text LLMs: no file paths, no base64, and
    no image payloads. Vector search failures are returned as warnings because a
    fresh library can still use direct metadata/fulltext before vectors exist.
    """

    warnings: list[str] = []
    try:
        ensure_rerank_disabled(rerank)
    except RerankNotSupportedError as exc:
        return {"results": [], "warnings": [str(exc)], "consumer": "llm_text", "image_return": "none"}
    results: list[dict[str, Any]] = []
    if include_metadata:
        results.extend(
            label_results(
                metadata_search(
                    context.ledger,
                    query=query,
                    limit=top_k,
                    consumer="llm_text",
                    rerank=False,
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
                    rerank=False,
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
                        rerank=False,
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
    query_image: dict[str, Any] | None = None,
    profile_name: str | None = None,
    top_k: int = 10,
    consumer: McpConsumer = "llm_text",
    image_return: ImageReturn = "none",
    max_images: int = 5,
    max_image_bytes: int = 256 * 1024,
    rerank: bool = False,
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
        ensure_rerank_disabled(rerank)
        normalized_image = normalize_query_image(
            query_image,
            allowed_roots=[context.config.paths.data_dir],
        )
        results = search_vector_index(
            ledger=context.ledger,
            vector_store_dir=context.config.paths.vector_store_dir,
            profile_name=profile_name,
            query=query_text,
            mode="multimodal",
            top_k=top_k,
            consumer=consumer,
            image_return=image_return,
            max_images=max_images,
            max_image_bytes=max_image_bytes,
            rerank=False,
            query_image_path=normalized_image.file_path if normalized_image else None,
            query_image_base64=normalized_image.base64_data if normalized_image else None,
            query_image_mime_type=normalized_image.mime_type if normalized_image else None,
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
