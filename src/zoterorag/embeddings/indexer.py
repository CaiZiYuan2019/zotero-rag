from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Literal

from ..db import StateLedger
from ..index.local_vector import LocalVectorStore, VectorRecord
from ..search.results import SearchResult, ensure_rerank_disabled, sanitize_results_for_consumer
from .base import EmbeddingInput, EmbeddingProvider, StubEmbeddingProvider
from .profile import embedding_profile_hash


SearchMode = Literal["text", "multimodal"]


@dataclass(frozen=True)
class IndexResult:
    profile_name: str
    document_id: str
    indexed_chunks: int
    vector_path: Path
    modality: str
    profile_hash: str
    embedding_batch_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "document_id": self.document_id,
            "indexed_chunks": self.indexed_chunks,
            "vector_path": str(self.vector_path),
            "modality": self.modality,
            "profile_hash": self.profile_hash,
            "embedding_batch_hash": self.embedding_batch_hash,
        }


def index_normalized_document(
    *,
    ledger: StateLedger,
    vector_store_dir: str | Path,
    profile_name: str,
    document_id: str,
    provider: EmbeddingProvider | None = None,
    allow_stub_provider: bool = False,
) -> IndexResult:
    profile = require_embedding_profile(ledger, profile_name)
    artifact = ledger.get_normalized_artifact(document_id)
    if artifact is None:
        raise KeyError(f"normalized artifact not found: {document_id}")

    profile_modality = str(profile["modality"])
    profile_hash = embedding_profile_hash(profile)
    chunk_type = "text" if profile_modality == "text" else "image"
    chunks = ledger.list_chunks(document_id, chunk_type=chunk_type)
    if provider is None and profile["provider"] != "stub" and not allow_stub_provider:
        raise NotImplementedError(
            f"embedding provider {profile['provider']} is not implemented for direct local indexing; "
            "use a real provider implementation or set allow_stub_provider=true for control-plane tests"
        )

    embedding_provider = provider or StubEmbeddingProvider(dimension=int(profile["dimension"]))
    if embedding_provider.dimension != int(profile["dimension"]):
        raise ValueError(
            f"provider dimension {embedding_provider.dimension} does not match profile {profile_name} "
            f"dimension {profile['dimension']}"
        )

    inputs = [embedding_input_for_chunk(chunk, artifact, role="document") for chunk in chunks]
    batch_hash = embedding_batch_hash(
        profile_hash=profile_hash,
        document_id=document_id,
        chunk_type=chunk_type,
        inputs=inputs,
    )
    ledger.upsert_embedding_batch(
        batch_hash=batch_hash,
        profile_name=profile_name,
        profile_hash=profile_hash,
        document_id=document_id,
        chunk_type=chunk_type,
        chunk_count=len(inputs),
        status="running",
        provider=embedding_provider.name,
        model=embedding_provider.model,
        payload={
            "input_ids": [item.input_id for item in inputs],
            "dimension": embedding_provider.dimension,
        },
    )
    vectors = {item.input_id: item.vector for item in embedding_provider.embed(inputs)}
    records = [
        VectorRecord(
            record_id=f"{profile_name}:{chunk['chunk_id']}",
            document_id=document_id,
            chunk_id=chunk["chunk_id"],
            vector=vectors[chunk["chunk_id"]],
            text=chunk["text"],
            modality=chunk_type,
            metadata={
                **chunk.get("metadata", {}),
                "heading_path": chunk.get("heading_path", []),
                "profile_name": profile_name,
                "profile_hash": profile_hash,
                "consumer_safe": chunk_type == "text",
            },
        )
        for chunk in chunks
    ]

    vector_path = vector_path_for_profile(vector_store_dir, profile_name)
    store = LocalVectorStore(vector_path, profile_name=profile_name, dimension=int(profile["dimension"]))
    try:
        # Stage records under the deterministic batch hash first. Search only
        # sees this rebuild after publish_version commits, so interrupted
        # embedding/index runs cannot expose a half-written vector set.
        indexed = store.upsert(records, index_version=batch_hash)
        store.publish_version(batch_hash)
        counts = store.counts(index_version=batch_hash)
    finally:
        store.close()

    ledger.upsert_embedding_batch(
        batch_hash=batch_hash,
        profile_name=profile_name,
        profile_hash=profile_hash,
        document_id=document_id,
        chunk_type=chunk_type,
        chunk_count=len(inputs),
        status="completed",
        provider=embedding_provider.name,
        model=embedding_provider.model,
        payload={
            "input_ids": [item.input_id for item in inputs],
            "dimension": embedding_provider.dimension,
            "indexed_chunks": indexed,
            "vector_path": str(vector_path),
        },
    )
    ledger.register_vector_index(
        profile_name=profile_name,
        backend="sqlite-local",
        path=vector_path,
        document_count=counts["documents"],
        chunk_count=counts["chunks"],
        active=bool(profile["enabled"]),
        active_version=batch_hash,
    )
    ledger.checkpoint(
        document_id,
        f"embed:{profile_name}",
        "indexed",
        {
            "chunks": indexed,
            "profile_modality": profile_modality,
            "chunk_type": chunk_type,
            "profile_hash": profile_hash,
            "embedding_batch_hash": batch_hash,
        },
    )
    return IndexResult(
        profile_name=profile_name,
        document_id=document_id,
        indexed_chunks=indexed,
        vector_path=vector_path,
        modality=chunk_type,
        profile_hash=profile_hash,
        embedding_batch_hash=batch_hash,
    )


def search_vector_index(
    *,
    ledger: StateLedger,
    vector_store_dir: str | Path,
    profile_name: str | None,
    query: str,
    mode: SearchMode,
    top_k: int = 10,
    consumer: str = "llm_text",
    image_return: str = "none",
    max_images: int = 5,
    max_image_bytes: int = 256 * 1024,
    rerank: bool = False,
    query_image_path: str | None = None,
    query_image_base64: str | None = None,
    query_image_mime_type: str | None = None,
    provider: EmbeddingProvider | None = None,
) -> list[dict[str, Any]]:
    profile = select_profile(ledger, profile_name=profile_name, mode=mode)
    profile_name = profile["name"]
    ensure_rerank_disabled(rerank)
    if mode == "text" and (query_image_path or query_image_base64):
        raise ValueError("text search does not accept query images")
    if provider is None and profile["provider"] != "stub":
        raise NotImplementedError(
            f"embedding provider {profile['provider']} is not implemented for direct local search; "
            "pass a real provider implementation for query embedding"
        )
    embedding_provider = provider or StubEmbeddingProvider(dimension=int(profile["dimension"]))
    if embedding_provider.dimension != int(profile["dimension"]):
        raise ValueError(
            f"provider dimension {embedding_provider.dimension} does not match profile {profile_name} "
            f"dimension {profile['dimension']}"
        )
    query_vector = embedding_provider.embed(
        [
            EmbeddingInput(
                input_id="query",
                text=query,
                image_path=query_image_path,
                image_base64=query_image_base64,
                image_mime_type=query_image_mime_type,
                role="query",
            )
        ]
    )[0].vector
    modality = "text" if mode == "text" else "image"
    store = LocalVectorStore(
        vector_path_for_profile(vector_store_dir, profile_name),
        profile_name=profile_name,
        dimension=int(profile["dimension"]),
    )
    try:
        raw_results = store.search(query_vector, top_k=top_k, modality=modality)
    finally:
        store.close()
    artifacts = {
        row["document_id"]: ledger.get_normalized_artifact(row["document_id"])
        for row in raw_results
        if row.get("document_id")
    }
    results = [search_result_from_vector_row(row, artifact=artifacts.get(row["document_id"])) for row in raw_results]
    return sanitize_results_for_consumer(
        results,
        consumer=consumer,  # type: ignore[arg-type]
        image_return=image_return,  # type: ignore[arg-type]
        max_images=max_images,
        max_image_bytes=max_image_bytes,
    )


def embedding_input_for_chunk(chunk: dict[str, Any], artifact: dict[str, Any], role: str) -> EmbeddingInput:
    metadata = chunk.get("metadata", {})
    image_path = None
    if chunk["chunk_type"] == "image" and metadata.get("image_path"):
        image_path = str(Path(artifact["artifact_dir"]) / metadata["image_path"])
    return EmbeddingInput(
        input_id=chunk["chunk_id"],
        text=chunk.get("text", ""),
        image_path=image_path,
        role=role,
    )


def embedding_batch_hash(
    *,
    profile_hash: str,
    document_id: str,
    chunk_type: str,
    inputs: list[EmbeddingInput],
) -> str:
    """Hash a batch without storing raw text, image bytes, or vectors."""

    payload = {
        "profile_hash": profile_hash,
        "document_id": document_id,
        "chunk_type": chunk_type,
        "inputs": [
            {
                "input_id": item.input_id,
                "text_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
                "image_path": item.image_path,
                "image_base64_sha256": (
                    hashlib.sha256(item.image_base64.encode("utf-8")).hexdigest()
                    if item.image_base64
                    else None
                ),
                "image_mime_type": item.image_mime_type,
                "role": item.role,
            }
            for item in inputs
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def search_result_from_vector_row(row: dict[str, Any], *, artifact: dict[str, Any] | None = None) -> SearchResult:
    metadata = dict(row.get("metadata") or {})
    images = []
    if row["modality"] == "image":
        image_path = metadata.get("image_path")
        image = {
            "image_id": row["chunk_id"],
            "caption": row["text"],
            "file_ref": image_path,
            "mime_type": mimetypes.guess_type(str(image_path or ""))[0],
        }
        image_abs_path = resolve_artifact_image_path(artifact, image_path)
        if image_abs_path is not None:
            image["image_abs_path"] = str(image_abs_path)
        images.append(image)
    return SearchResult(
        document_id=row["document_id"],
        chunk_id=row["chunk_id"],
        title=metadata.get("title") or row["document_id"],
        text=row["text"],
        score=float(row["score"]),
        metadata=metadata,
        images=images,
    )


def resolve_artifact_image_path(artifact: dict[str, Any] | None, image_path: Any) -> Path | None:
    if artifact is None or not image_path:
        return None
    artifact_dir = Path(str(artifact["artifact_dir"])).resolve()
    candidate = (artifact_dir / str(image_path)).resolve()
    try:
        candidate.relative_to(artifact_dir)
    except ValueError:
        return None
    return candidate


def require_embedding_profile(ledger: StateLedger, profile_name: str) -> dict[str, Any]:
    for profile in ledger.list_embedding_profiles():
        if profile["name"] == profile_name:
            return profile
    raise KeyError(f"embedding profile not found: {profile_name}")


def select_profile(ledger: StateLedger, *, profile_name: str | None, mode: SearchMode) -> dict[str, Any]:
    if profile_name:
        profile = require_embedding_profile(ledger, profile_name)
        expected = "text" if mode == "text" else "multimodal"
        if profile["modality"] != expected:
            raise ValueError(f"profile {profile_name} has modality {profile['modality']}, expected {expected}")
        return profile
    default_flag = "default_for_text" if mode == "text" else "default_for_multimodal"
    for profile in ledger.list_embedding_profiles():
        if profile["enabled"] and profile[default_flag]:
            return profile
    raise KeyError(f"no default {mode} embedding profile is configured")


def vector_path_for_profile(vector_store_dir: str | Path, profile_name: str) -> Path:
    return Path(vector_store_dir) / profile_name / "vectors.sqlite"
