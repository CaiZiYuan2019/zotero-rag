from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Literal

from ..db import StateLedger
from ..index.local_vector import open_vector_store, VectorRecord
from ..search.results import SearchResult, ensure_rerank_disabled, sanitize_results_for_consumer
from .base import EmbeddingInput, EmbeddingProvider, StubEmbeddingProvider
from .profile import embedding_profile_hash


SearchMode = Literal["text", "multimodal"]

# M4: prune recycles embedding batches stuck "running" longer than this.
# Keep in sync with the config default vectors.stale_embedding_batch_hours
# (24h) in config/settings.py.
DEFAULT_STALE_BATCH_SECONDS = 24.0 * 3600.0


@dataclass(frozen=True)
class IndexResult:
    profile_name: str
    document_id: str
    indexed_chunks: int
    vector_path: Path
    modality: str
    profile_hash: str
    embedding_batch_hash: str
    reused_existing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "document_id": self.document_id,
            "indexed_chunks": self.indexed_chunks,
            "vector_path": str(self.vector_path),
            "modality": self.modality,
            "profile_hash": self.profile_hash,
            "embedding_batch_hash": self.embedding_batch_hash,
            "reused_existing": self.reused_existing,
        }


@dataclass(frozen=True)
class StagedDocument:
    """Phase 3: carrier between the stage and publish halves of indexing.

    ``stage_indexed_document`` fills this with computed vectors without
    touching the vector store; ``publish_staged_documents`` later publishes
    a whole run of these in one ``stage_and_publish`` per profile store.
    """

    profile_name: str
    document_id: str
    chunk_type: str
    profile_modality: str
    profile_hash: str
    batch_hash: str
    records: list[VectorRecord]
    provider_name: str
    provider_model: str
    dimension: int
    backend: str
    vector_path: Path
    input_ids: list[str]
    profile_enabled: bool


def _resolve_embedding_provider(
    *,
    provider: EmbeddingProvider | None,
    profile: dict[str, Any],
    allow_stub_provider: bool,
) -> EmbeddingProvider:
    """Resolve an embedding provider, building real providers from env when needed.

    Callers that already have a concrete provider (CLI, API) should pass it
    directly. When no provider is given, this function tries to construct the
    correct implementation for the configured profile so that control-plane
    code (MCP tools, reembed workers) does not need to carry env-loading logic.
    """
    if provider is not None:
        return provider
    if profile["provider"] == "stub" or allow_stub_provider:
        return StubEmbeddingProvider(dimension=int(profile["dimension"]))
    # Lazy-import to avoid a circular dependency between embeddings and
    # the top-level providers module (which imports from embeddings).
    from ..providers import build_qwen_embedding_provider
    return build_qwen_embedding_provider(profile)


def stage_indexed_document(
    *,
    ledger: StateLedger,
    vector_store_dir: str | Path,
    profile_name: str,
    document_id: str,
    provider: EmbeddingProvider | None = None,
    allow_stub_provider: bool = False,
    vector_backend: str | None = None,
) -> IndexResult | StagedDocument:
    """Compute embeddings for one document without publishing them (Phase 3).

    This is the first half of the historical ``index_normalized_document``:
    profile/artifact resolution, the reuse fast path, provider embedding, and
    the "running" batch row. When the reuse fast path hits, an ``IndexResult``
    is returned directly (the document is already fully published); otherwise
    a ``StagedDocument`` carrying the computed vectors is returned for a later
    run-level ``publish_staged_documents`` call. The staging path never opens
    the store for writing, never completes the batch row, and never registers
    the index or writes the "indexed" checkpoint — that all happens at
    publish time.
    """
    profile = require_embedding_profile(ledger, profile_name)
    vector_backend = vector_backend or profile.get("backend", "lancedb")
    artifact = ledger.get_normalized_artifact(document_id)
    if artifact is None:
        raise KeyError(f"normalized artifact not found: {document_id}")

    profile_modality = str(profile["modality"])
    profile_hash = embedding_profile_hash(profile)
    chunk_type = "text" if profile_modality == "text" else "image"
    chunks = ledger.list_chunks(document_id, chunk_type=chunk_type)
    inputs = [embedding_input_for_chunk(chunk, artifact, role="document") for chunk in chunks]
    batch_hash = embedding_batch_hash(
        profile_hash=profile_hash,
        document_id=document_id,
        chunk_type=chunk_type,
        inputs=inputs,
    )
    vector_path = vector_path_for_profile(vector_store_dir, profile_name, backend=vector_backend)

    # For expensive providers, a completed batch plus matching active vectors is
    # enough proof to skip the provider entirely. This keeps accidental reruns
    # from repeating qwen calls after a long indexing job has already published.
    # Phase 3: this fast path is identical to the historical behavior — it
    # publishes nothing new, so it bypasses the stage/publish split.
    existing_batch = ledger.get_embedding_batch(batch_hash)
    if (
        existing_batch is not None
        and existing_batch["status"] == "completed"
        and active_vectors_match_embedding_batch(
            vector_path=vector_path,
            profile_name=profile_name,
            dimension=int(profile["dimension"]),
            document_id=document_id,
            chunk_type=chunk_type,
            batch_hash=batch_hash,
            expected_chunks=len(inputs),
            backend=vector_backend,
        )
    ):
        counts = vector_index_counts(vector_path, profile_name=profile_name, dimension=int(profile["dimension"]), backend=vector_backend)
        ledger.register_vector_index(
            profile_name=profile_name,
            backend=vector_backend,
            path=vector_path,
            document_count=counts["documents"],
            chunk_count=counts["chunks"],
            active=bool(profile["enabled"]),
            active_version=counts["active_version"],
        )
        ledger.checkpoint(
            document_id,
            f"embed:{profile_name}",
            "indexed",
            {
                "chunks": len(inputs),
                "carried_forward_chunks": 0,
                "profile_modality": profile_modality,
                "chunk_type": chunk_type,
                "profile_hash": profile_hash,
                "embedding_batch_hash": batch_hash,
                "reused_existing_embedding": True,
            },
        )
        return IndexResult(
            profile_name=profile_name,
            document_id=document_id,
            indexed_chunks=len(inputs),
            vector_path=vector_path,
            modality=chunk_type,
            profile_hash=profile_hash,
            embedding_batch_hash=batch_hash,
            reused_existing=True,
        )

    embedding_provider = _resolve_embedding_provider(
        provider=provider,
        profile=profile,
        allow_stub_provider=allow_stub_provider,
    )
    if embedding_provider.dimension != int(profile["dimension"]):
        raise ValueError(
            f"provider dimension {embedding_provider.dimension} does not match profile {profile_name} "
            f"dimension {profile['dimension']}"
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
                "embedding_batch_hash": batch_hash,
                "consumer_safe": chunk_type == "text",
            },
        )
        for chunk in chunks
    ]

    # Phase 3: stop before opening the store. Completing the batch row,
    # registering the vector index, and writing the "indexed" checkpoint are
    # deferred to publish_staged_documents so a whole run publishes once per
    # profile instead of once per document.
    return StagedDocument(
        profile_name=profile_name,
        document_id=document_id,
        chunk_type=chunk_type,
        profile_modality=profile_modality,
        profile_hash=profile_hash,
        batch_hash=batch_hash,
        records=records,
        provider_name=embedding_provider.name,
        provider_model=embedding_provider.model,
        dimension=embedding_provider.dimension,
        backend=vector_backend,
        vector_path=vector_path,
        input_ids=[item.input_id for item in inputs],
        profile_enabled=bool(profile["enabled"]),
    )


def publish_staged_documents(
    *,
    ledger: StateLedger,
    staged: Sequence[StagedDocument],
) -> list[IndexResult]:
    """Publish staged documents, one ``stage_and_publish`` per store (Phase 3).

    Documents are grouped by (profile_name, backend, vector_path); each group
    publishes every staged record in a single locked copy -> upsert -> publish
    sequence, replacing the historical per-document publish whose copy step
    made N documents O(N^2). Returns one ``IndexResult`` per input document,
    in input order (also across groups).
    """
    if not staged:
        return []

    groups: dict[tuple[str, str, Path], list[tuple[int, StagedDocument]]] = {}
    for position, document in enumerate(staged):
        key = (document.profile_name, document.backend, document.vector_path)
        groups.setdefault(key, []).append((position, document))

    results: dict[int, IndexResult] = {}
    for (profile_name, backend, vector_path), members in groups.items():
        documents = [document for _, document in members]
        # Phase 3: the published version is named by a per-run hash over the
        # group's batch hashes instead of a single document's batch_hash. The
        # 64-hex shape is unchanged so version-name handling elsewhere behaves
        # identically, and the hash is deterministic: re-running the same
        # batch recomputes the same name, so a crash between publish and
        # bookkeeping self-heals on the next run.
        run_hash = hashlib.sha256(
            "\n".join(sorted(document.batch_hash for document in documents)).encode("utf-8")
        ).hexdigest()
        store = open_vector_store(
            vector_path,
            profile_name=profile_name,
            dimension=documents[0].dimension,
            backend=backend,
        )
        try:
            # Sweep leftover staged versions before staging this run. The
            # run_hash version cannot exist yet, so no exemption is needed;
            # delete_version itself refuses to delete the active version; and
            # when a crashed run is retried with the same documents, the
            # leftover non-active same-named version is exactly what gets
            # removed here.
            for version in store.list_versions():
                if not store.version_is_active(version):
                    store.delete_version(version)
            records = [record for document in documents for record in document.records]
            # Stage records under the deterministic run hash first. Search only
            # sees this rebuild after publish_version commits, so interrupted
            # embedding/index runs cannot expose a half-written vector set.
            # Carrying forward the existing active records for other documents
            # keeps previously indexed evidence visible in the same profile.
            # M6: copy -> upsert -> publish runs as one cross-process locked
            # sequence; two writers must never interleave it.
            copied, indexed, counts = store.stage_and_publish(
                records,
                index_version=run_hash,
                exclude_document_ids=[document.document_id for document in documents],
            )
        finally:
            store.close()

        # Bookkeeping after publish. A crash inside this window leaves the
        # group's batch rows "running", so the reuse fast path does not hit on
        # the next run and the documents are re-embedded and re-published.
        # That re-publish produces no duplicate rows: the upsert is idempotent
        # per record_id and the copy step excludes this run's documents. Same
        # failure-window semantics as the historical single-document path, now
        # spanning the whole run.
        for document in documents:
            ledger.upsert_embedding_batch(
                batch_hash=document.batch_hash,
                profile_name=document.profile_name,
                profile_hash=document.profile_hash,
                document_id=document.document_id,
                chunk_type=document.chunk_type,
                chunk_count=len(document.input_ids),
                status="completed",
                provider=document.provider_name,
                model=document.provider_model,
                payload={
                    "input_ids": document.input_ids,
                    "dimension": document.dimension,
                    "indexed_chunks": len(document.records),
                    # Run-level copy count: one stage_and_publish carried every
                    # member of the group, so each document reports the same
                    # value.
                    "carried_forward_chunks": copied,
                    "vector_path": str(document.vector_path),
                    # Phase 3: links this batch row to the published version
                    # (named by run_hash, not batch_hash); consumed by
                    # diagnostics.vector_index_provenance.
                    "index_version": run_hash,
                },
            )
        ledger.register_vector_index(
            profile_name=profile_name,
            backend=backend,
            path=vector_path,
            document_count=counts["documents"],
            chunk_count=counts["chunks"],
            # All group members share the same profile, hence the same flag.
            active=documents[0].profile_enabled,
            active_version=run_hash,
        )
        for position, document in members:
            ledger.checkpoint(
                document.document_id,
                f"embed:{document.profile_name}",
                "indexed",
                {
                    "chunks": len(document.records),
                    "carried_forward_chunks": copied,
                    "profile_modality": document.profile_modality,
                    "chunk_type": document.chunk_type,
                    "profile_hash": document.profile_hash,
                    "embedding_batch_hash": document.batch_hash,
                    "index_version": run_hash,
                },
            )
            results[position] = IndexResult(
                profile_name=document.profile_name,
                document_id=document.document_id,
                indexed_chunks=len(document.records),
                vector_path=document.vector_path,
                modality=document.chunk_type,
                profile_hash=document.profile_hash,
                embedding_batch_hash=document.batch_hash,
                reused_existing=False,
            )
    return [results[position] for position in range(len(staged))]


def index_normalized_document(
    *,
    ledger: StateLedger,
    vector_store_dir: str | Path,
    profile_name: str,
    document_id: str,
    provider: EmbeddingProvider | None = None,
    allow_stub_provider: bool = False,
    vector_backend: str | None = None,
) -> IndexResult:
    """Index one normalized document into the profile's vector store.

    Phase 3: thin wrapper over the stage/publish split — stage the document,
    then immediately publish it as a one-document run. Behavior is unchanged
    except the published version name: it is now the run hash (sha256 of the
    single batch hash) instead of the batch hash itself.
    """
    staged = stage_indexed_document(
        ledger=ledger,
        vector_store_dir=vector_store_dir,
        profile_name=profile_name,
        document_id=document_id,
        provider=provider,
        allow_stub_provider=allow_stub_provider,
        vector_backend=vector_backend,
    )
    if isinstance(staged, IndexResult):
        # Reuse fast path: already completed and published; nothing to do.
        return staged
    return publish_staged_documents(ledger=ledger, staged=[staged])[0]


def active_vectors_match_embedding_batch(
    *,
    vector_path: Path,
    profile_name: str,
    dimension: int,
    document_id: str,
    chunk_type: str,
    batch_hash: str,
    expected_chunks: int,
    backend: str | None = None,
) -> bool:
    backend = backend or "lancedb"
    if expected_chunks == 0:
        return False
    if backend == "sqlite-local" and not vector_path.is_file():
        return False
    store = open_vector_store(vector_path, profile_name=profile_name, dimension=dimension, backend=backend)
    try:
        counts = store.document_counts(document_id, modality=chunk_type)
        batch_hashes = {
            value
            for value in store.document_metadata_values(document_id, "embedding_batch_hash", modality=chunk_type)
            if value
        }
        return counts["chunks"] == expected_chunks and batch_hashes == {batch_hash}
    finally:
        store.close()


def vector_index_counts(vector_path: Path, *, profile_name: str, dimension: int, backend: str | None = None) -> dict[str, Any]:
    backend = backend or "lancedb"
    store = open_vector_store(vector_path, profile_name=profile_name, dimension=dimension, backend=backend)
    try:
        active_version = store.active_version()
        return {
            **store.counts(index_version=active_version),
            "active_version": active_version,
        }
    finally:
        store.close()


def prune_vector_index_versions(
    *,
    ledger: StateLedger,
    vector_store_dir: str | Path,
    profile_name: str,
    vacuum: bool = False,
    stale_batch_seconds: float | None = DEFAULT_STALE_BATCH_SECONDS,
) -> dict[str, Any]:
    """Delete every non-active vector version for a profile (one-time reclaim).

    ``publish_version`` already deletes the directly superseded version, but
    versions orphaned by crashes (and everything written before that cleanup
    existed) linger. This removes them. Safety rails: the registered
    ``active_version`` must match the store's own metadata, and the active
    version must be non-empty, so a corrupt registration aborts the prune.
    Run this only while no indexing job is active for the profile: pruning
    concurrently with an in-flight index could delete the version being
    staged, so a running embedding batch aborts the prune.

    M4: batches stuck "running" by a crash are first recycled to "failed"
    when their updated_at is older than ``stale_batch_seconds`` (default 24h,
    matching the config default vectors.stale_embedding_batch_hours), so
    orphaned rows no longer block the prune forever. Pass None to disable
    the recycling. Only genuinely fresh running batches still abort.
    """

    profiles = {profile["name"]: profile for profile in ledger.list_embedding_profiles()}
    profile = profiles.get(profile_name)
    if profile is None:
        raise ValueError(f"profile_not_found:{profile_name}")
    recycled_stale = 0
    if stale_batch_seconds is not None:
        recycled_stale = ledger.mark_stale_embedding_batches_failed(
            profile_name, older_than_seconds=stale_batch_seconds
        )
    running_batches = ledger.list_embedding_batches(profile_name=profile_name, status="running")
    if running_batches:
        raise ValueError(
            f"{len(running_batches)} embedding batch(es) still running for {profile_name}; "
            "finish or recover them before pruning"
        )
    resolved_backend, vector_path = _resolve_registered_vector_store(
        ledger,
        vector_store_dir=vector_store_dir,
        profile_name=profile_name,
        fallback_backend=str(profile.get("backend", "lancedb")),
    )
    registered = next(
        (index for index in ledger.list_vector_indexes() if index["profile_name"] == profile_name),
        None,
    )
    store = open_vector_store(
        vector_path,
        profile_name=profile_name,
        dimension=int(profile["dimension"]),
        backend=resolved_backend,
    )
    try:
        active_version = store.active_version()
        registered_active = str((registered or {}).get("active_version") or "legacy")
        if registered is None:
            raise ValueError(f"vector_index_not_registered:{profile_name}")
        if registered_active != active_version:
            raise ValueError(
                f"active_version mismatch: ledger={registered_active!r} store={active_version!r}; "
                "run `vectors verify` before pruning"
            )
        counts = store.counts(index_version=active_version)
        if int(counts["chunks"]) <= 0:
            raise ValueError(f"active version {active_version!r} is empty; refusing to prune")
        versions = store.list_versions()
        removed: dict[str, int] = {}
        for version in versions:
            # Compare via the store: LanceDB table names are sanitized
            # irreversibly, so raw string equality can miss the active one.
            if store.version_is_active(version):
                continue
            removed[version] = store.delete_version(version)
        vacuumed = False
        if vacuum and hasattr(store, "vacuum"):
            store.vacuum()
            vacuumed = True
        return {
            "profile_name": profile_name,
            "backend": resolved_backend,
            "active_version": active_version,
            "kept_versions": [active_version],
            "removed_versions": removed,
            "recycled_stale_batches": recycled_stale,
            "vacuumed": vacuumed,
        }
    finally:
        store.close()


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
    vector_backend: str | None = None,
    allow_stub_provider: bool = True,
) -> list[dict[str, Any]]:
    profile = select_profile(ledger, profile_name=profile_name, mode=mode)
    profile_name = profile["name"]
    vector_backend = vector_backend or profile.get("backend", "lancedb")
    ensure_rerank_disabled(rerank)
    if mode == "text" and (query_image_path or query_image_base64):
        raise ValueError("text search does not accept query images")
    # Default to stub-safe resolution for search. Callers that want a real
    # provider must pass it explicitly; this matches the CLI/API default where
    # embedding_provider="stub" means "do not make unsolicited provider calls".
    embedding_provider = _resolve_embedding_provider(
        provider=provider,
        profile=profile,
        allow_stub_provider=allow_stub_provider,
    )
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
    resolved_backend, vector_path = _resolve_registered_vector_store(
        ledger,
        vector_store_dir=vector_store_dir,
        profile_name=profile_name,
        fallback_backend=vector_backend,
    )
    store = open_vector_store(
        vector_path,
        profile_name=profile_name,
        dimension=int(profile["dimension"]),
        backend=resolved_backend,
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
    if chunk["chunk_type"] == "image":
        image_path = metadata.get("image_embedding_path")
        if not image_path:
            status = metadata.get("image_embedding_status") or "missing"
            raise ValueError(
                f"image chunk {chunk['chunk_id']} is not ready for embedding: {status}"
            )
        image_path = str(Path(artifact["artifact_dir"]) / image_path)
        artifact_dir = Path(artifact["artifact_dir"]).resolve()
        resolved = Path(image_path).resolve()
        if not _is_relative_to(resolved, artifact_dir):
            raise ValueError(
                f"image chunk {chunk['chunk_id']} path resolves outside artifact directory: {image_path}"
            )
    return EmbeddingInput(
        input_id=chunk["chunk_id"],
        text=chunk.get("text", ""),
        image_path=image_path,
        role=role,
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


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


def vector_path_for_profile(
    vector_store_dir: str | Path,
    profile_name: str,
    backend: str | None = None,
) -> Path:
    """Return the storage path for a profile.

    LanceDB stores an entire database directory per profile, while the local
    SQLite backend stores a single file.
    """
    backend = backend or "lancedb"
    profile_dir = Path(vector_store_dir) / profile_name
    if backend == "lancedb":
        return profile_dir
    return profile_dir / "vectors.sqlite"


def _resolve_registered_vector_store(
    ledger: StateLedger,
    *,
    vector_store_dir: str | Path,
    profile_name: str,
    fallback_backend: str | None = None,
) -> tuple[str, Path]:
    """Use the backend and path registered in ``vector_indexes`` when available.

    Searching against a store whose backend was switched after indexing would
    otherwise open the wrong file type. The explicit ``fallback_backend`` is
    used only when the profile has not been registered yet.
    """
    fallback_backend = fallback_backend or "lancedb"
    for index in ledger.list_vector_indexes():
        if index["profile_name"] == profile_name:
            return str(index["backend"]), Path(str(index["path"]))
    return fallback_backend, vector_path_for_profile(
        vector_store_dir,
        profile_name,
        backend=fallback_backend,
    )
