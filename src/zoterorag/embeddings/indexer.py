from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..db import StateLedger
from ..index.local_vector import LocalVectorStore, VectorRecord
from ..search.results import SearchResult, sanitize_results_for_consumer
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "document_id": self.document_id,
            "indexed_chunks": self.indexed_chunks,
            "vector_path": str(self.vector_path),
            "modality": self.modality,
            "profile_hash": self.profile_hash,
        }


def index_normalized_document(
    *,
    ledger: StateLedger,
    vector_store_dir: str | Path,
    profile_name: str,
    document_id: str,
    provider: EmbeddingProvider | None = None,
) -> IndexResult:
    profile = require_embedding_profile(ledger, profile_name)
    artifact = ledger.get_normalized_artifact(document_id)
    if artifact is None:
        raise KeyError(f"normalized artifact not found: {document_id}")

    profile_modality = str(profile["modality"])
    profile_hash = embedding_profile_hash(profile)
    chunk_type = "text" if profile_modality == "text" else "image"
    chunks = ledger.list_chunks(document_id, chunk_type=chunk_type)
    embedding_provider = provider or StubEmbeddingProvider(dimension=int(profile["dimension"]))
    if embedding_provider.dimension != int(profile["dimension"]):
        raise ValueError(
            f"provider dimension {embedding_provider.dimension} does not match profile {profile_name} "
            f"dimension {profile['dimension']}"
        )

    inputs = [embedding_input_for_chunk(chunk, artifact, role="document") for chunk in chunks]
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
        indexed = store.upsert(records)
        counts = store.counts()
    finally:
        store.close()

    ledger.register_vector_index(
        profile_name=profile_name,
        backend="sqlite-local",
        path=vector_path,
        document_count=counts["documents"],
        chunk_count=counts["chunks"],
        active=bool(profile["enabled"]),
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
        },
    )
    return IndexResult(
        profile_name=profile_name,
        document_id=document_id,
        indexed_chunks=indexed,
        vector_path=vector_path,
        modality=chunk_type,
        profile_hash=profile_hash,
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
    query_image_path: str | None = None,
    query_image_base64: str | None = None,
    query_image_mime_type: str | None = None,
    provider: EmbeddingProvider | None = None,
) -> list[dict[str, Any]]:
    profile = select_profile(ledger, profile_name=profile_name, mode=mode)
    profile_name = profile["name"]
    if mode == "text" and (query_image_path or query_image_base64):
        raise ValueError("text search does not accept query images")
    embedding_provider = provider or StubEmbeddingProvider(dimension=int(profile["dimension"]))
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
    results = [search_result_from_vector_row(row) for row in raw_results]
    return sanitize_results_for_consumer(
        results,
        consumer=consumer,  # type: ignore[arg-type]
        image_return=image_return,  # type: ignore[arg-type]
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


def search_result_from_vector_row(row: dict[str, Any]) -> SearchResult:
    metadata = dict(row.get("metadata") or {})
    images = []
    if row["modality"] == "image":
        image_path = metadata.get("image_path")
        images.append(
            {
                "image_id": row["chunk_id"],
                "caption": row["text"],
                "file_ref": image_path,
            }
        )
    return SearchResult(
        document_id=row["document_id"],
        chunk_id=row["chunk_id"],
        title=metadata.get("title") or row["document_id"],
        text=row["text"],
        score=float(row["score"]),
        metadata=metadata,
        images=images,
    )


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
