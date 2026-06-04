from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .db import StateLedger
from .search.results import strip_image_references_from_text


DocumentConsumer = Literal["manual", "llm_text", "llm_multimodal"]


def list_documents(
    ledger: StateLedger,
    *,
    limit: int | None = 50,
    include_metadata_only: bool = True,
) -> list[dict[str, Any]]:
    """List document-level records built from local state only.

    Normalized artifacts are the canonical indexed documents. Attachments that
    have been scanned but not normalized are exposed as metadata-only documents
    so the control plane can review and target them before expensive MinerU or
    embedding work starts.
    """

    attachments = {
        item["attachment_key"]: item
        for item in ledger.list_attachments(limit=None)
    }
    artifacts = ledger.list_normalized_artifacts(limit=None)
    documents = [
        build_normalized_document_summary(artifact, attachments.get(artifact.get("attachment_key") or ""))
        for artifact in artifacts
    ]

    if include_metadata_only:
        normalized_keys = {
            artifact.get("attachment_key")
            for artifact in artifacts
            if artifact.get("attachment_key")
        }
        for attachment in attachments.values():
            if attachment["attachment_key"] in normalized_keys:
                continue
            documents.append(build_metadata_only_document_summary(attachment))

    documents.sort(key=document_sort_key, reverse=True)
    return documents if limit is None else documents[:limit]


def get_document(
    ledger: StateLedger,
    document_id: str,
    *,
    include_chunks: bool = False,
    chunk_type: str | None = None,
    limit: int | None = None,
    consumer: DocumentConsumer = "llm_text",
) -> dict[str, Any] | None:
    """Return one document with optional chunks.

    `llm_text` is the safe default for MCP/LLM callers: it removes image file
    references and suppresses image chunks unless explicitly requested through a
    non-text consumer.
    """

    consumer = normalize_consumer(consumer)
    artifact = ledger.get_normalized_artifact(document_id)
    if artifact is not None:
        attachment = (
            ledger.get_attachment(artifact["attachment_key"])
            if artifact.get("attachment_key")
            else None
        )
        document = build_normalized_document_summary(artifact, attachment)
        document["artifact"] = artifact_for_consumer(artifact, consumer)
        if include_chunks:
            selected_chunk_type = "text" if consumer == "llm_text" and chunk_type is None else chunk_type
            chunks = ledger.list_chunks(document_id, chunk_type=selected_chunk_type, limit=limit)
            document["chunks"] = [
                chunk_for_consumer(chunk, artifact=artifact, consumer=consumer)
                for chunk in chunks
            ]
        return document

    attachment = ledger.get_attachment(document_id)
    if attachment is not None:
        document = build_metadata_only_document_summary(attachment)
        if consumer == "manual":
            document["attachment"] = attachment
        return document
    return None


def build_normalized_document_summary(
    artifact: dict[str, Any],
    attachment: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = attachment or {}
    return {
        "document_id": artifact["document_id"],
        "document_kind": "normalized",
        "attachment_key": artifact.get("attachment_key"),
        "parent_key": metadata.get("parent_key"),
        "title": metadata.get("title") or artifact["document_id"],
        "date": metadata.get("date_value"),
        "url": metadata.get("url"),
        "classification": metadata.get("classification"),
        "source_quality": metadata.get("source_quality"),
        "status": artifact["status"],
        "chunk_count": artifact["chunk_count"],
        "image_count": artifact["image_count"],
        "updated_at": artifact["updated_at"],
        "normalized": True,
    }


def build_metadata_only_document_summary(attachment: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": attachment["attachment_key"],
        "document_kind": "metadata_only",
        "attachment_key": attachment["attachment_key"],
        "parent_key": attachment.get("parent_key"),
        "title": attachment.get("title") or attachment.get("relative_path") or attachment["attachment_key"],
        "date": attachment.get("date_value"),
        "url": attachment.get("url"),
        "classification": attachment.get("classification"),
        "source_quality": attachment.get("source_quality"),
        "status": attachment.get("scan_status"),
        "chunk_count": 0,
        "image_count": 0,
        "updated_at": attachment["updated_at"],
        "normalized": False,
    }


def artifact_for_consumer(artifact: dict[str, Any], consumer: DocumentConsumer) -> dict[str, Any]:
    if consumer == "llm_text":
        return {
            "document_id": artifact["document_id"],
            "status": artifact["status"],
            "document_hash": artifact["document_hash"],
            "chunk_count": artifact["chunk_count"],
            "image_count": artifact["image_count"],
            "updated_at": artifact["updated_at"],
        }
    return dict(artifact)


def chunk_for_consumer(
    chunk: dict[str, Any],
    *,
    artifact: dict[str, Any],
    consumer: DocumentConsumer,
) -> dict[str, Any]:
    item = dict(chunk)
    item["text"] = strip_image_references_from_text(str(item.get("text", "")))
    metadata = dict(item.get("metadata") or {})
    if consumer == "llm_text":
        # Text-only document reads must not leak local image paths. Image chunks
        # can still expose captions/ids if requested explicitly, but not files.
        metadata.pop("image_path", None)
        metadata.pop("image_return", None)
        item["metadata"] = metadata
        return item

    if item.get("chunk_type") == "image" and metadata.get("image_path"):
        item["images"] = [image_payload_from_chunk(metadata, artifact)]
    item["metadata"] = metadata
    return item


def image_payload_from_chunk(metadata: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    image_path = metadata["image_path"]
    artifact_dir = Path(artifact["artifact_dir"])
    return {
        "image_id": metadata.get("image_run_id") or image_path,
        "file_ref": str(artifact_dir / image_path),
        "caption": metadata.get("caption"),
    }


def document_sort_key(document: dict[str, Any]) -> tuple[str, str]:
    return (str(document.get("updated_at") or ""), str(document.get("document_id") or ""))


def normalize_consumer(consumer: str) -> DocumentConsumer:
    if consumer not in {"manual", "llm_text", "llm_multimodal"}:
        raise ValueError(f"unknown document consumer: {consumer}")
    return consumer  # type: ignore[return-value]
