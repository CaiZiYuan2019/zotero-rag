from __future__ import annotations

from pathlib import Path
from typing import Any

from .db import StateLedger
from .embeddings import embedding_profile_hash


def list_embedding_model_catalog(ledger: StateLedger) -> dict[str, Any]:
    """Return query-ready embedding profiles with their local index state.

    External callers need one catalog answer for model selection: profile
    semantics, default routing, and whether a vector index exists locally. This
    function deliberately reports one profile per entry because searches must
    use a single vector space and never merge scores across embedding models.
    """

    indexes = {item["profile_name"]: item for item in ledger.list_vector_indexes()}
    models = [
        describe_embedding_profile(profile, vector_index=indexes.get(profile["name"]))
        for profile in ledger.list_embedding_profiles()
    ]
    return {
        "models": models,
        "defaults": {
            "text": next((item["name"] for item in models if item["default_for_text"]), None),
            "multimodal": next((item["name"] for item in models if item["default_for_multimodal"]), None),
        },
        # Kept as a compatibility field for older MCP/API callers that already
        # consumed separate vector index records.
        "vector_indexes": list(indexes.values()),
    }


def describe_embedding_profile(
    profile: dict[str, Any],
    *,
    vector_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    embedded_profile = dict(profile.get("profile") or {})
    index_info = describe_vector_index(vector_index)
    return {
        "name": profile["name"],
        "provider": profile["provider"],
        "model": profile["model"],
        "dimension": int(profile["dimension"]),
        "modality": profile["modality"],
        "enabled": bool(profile["enabled"]),
        "default_for_text": bool(profile["default_for_text"]),
        "default_for_multimodal": bool(profile["default_for_multimodal"]),
        "query_role_mode": profile.get("query_role_mode", embedded_profile.get("query_role_mode")),
        "document_role_mode": profile.get("document_role_mode", embedded_profile.get("document_role_mode")),
        "instruction_template": profile.get(
            "instruction_template",
            embedded_profile.get("instruction_template", ""),
        ),
        "profile_hash": embedding_profile_hash(profile),
        "query_modes": query_modes_for_profile(profile),
        "index_status": model_index_status(profile, index_info),
        "vector_index": index_info,
    }


def describe_vector_index(vector_index: dict[str, Any] | None) -> dict[str, Any] | None:
    if vector_index is None:
        return None
    path = str(vector_index["path"])
    return {
        "profile_name": vector_index["profile_name"],
        "backend": vector_index["backend"],
        "path": path,
        "path_exists": Path(path).exists(),
        "document_count": int(vector_index["document_count"]),
        "chunk_count": int(vector_index["chunk_count"]),
        "active": bool(vector_index["active"]),
        "updated_at": vector_index["updated_at"],
    }


def query_modes_for_profile(profile: dict[str, Any]) -> list[str]:
    modality = profile["modality"]
    if modality == "text":
        return ["text"]
    if modality == "multimodal":
        return ["multimodal"]
    return []


def model_index_status(profile: dict[str, Any], index_info: dict[str, Any] | None) -> str:
    if not bool(profile["enabled"]):
        return "disabled"
    if index_info is None:
        return "not_indexed"
    if not bool(index_info["active"]):
        return "inactive"
    if int(index_info["chunk_count"]) == 0:
        return "empty"
    if not bool(index_info["path_exists"]):
        return "missing_files"
    return "ready"
