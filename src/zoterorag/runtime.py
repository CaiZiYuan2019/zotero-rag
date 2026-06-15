from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AppConfig, EmbeddingProfile, load_config
from .db import StateLedger
from .zotero import create_shadow_copy, scan_shadow_to_ledger


def vector_store_path_for_profile(vector_store_dir: str | Path, profile: EmbeddingProfile) -> Path:
    """Return the storage path for *profile* based on its backend.

    LanceDB stores an entire database directory per profile, while the local
    SQLite backend stores a single file.
    """
    profile_dir = Path(vector_store_dir) / profile.name
    if profile.backend == "lancedb":
        return profile_dir
    return profile_dir / "vectors.sqlite"


def initialize_runtime(config_path: str | Path = "config/config.toml") -> tuple[AppConfig, StateLedger]:
    config = load_config(config_path)
    config.ensure_runtime_dirs()
    ledger = StateLedger(config.paths.state_db)
    ledger.upsert_embedding_profiles(config.embedding_profiles)
    existing_indexes = {item["profile_name"]: item for item in ledger.list_vector_indexes()}
    for profile in config.embedding_profiles:
        vector_path = vector_store_path_for_profile(config.paths.vector_store_dir, profile)
        existing = existing_indexes.get(profile.name)
        ledger.register_vector_index(
            profile_name=profile.name,
            backend=profile.backend,
            path=vector_path,
            document_count=int(existing["document_count"]) if existing else 0,
            chunk_count=int(existing["chunk_count"]) if existing else 0,
            active=profile.enabled,
            active_version=str(existing.get("active_version") or "") if existing else "",
        )
    return config, ledger


def copy_zotero_shadow(
    config: AppConfig,
    ledger: StateLedger | None = None,
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    shadow_path = create_shadow_copy(
        config.paths.zotero_db,
        config.paths.shadow_db,
        timeout_seconds=timeout_seconds,
    )
    result = {
        "shadow_db": str(shadow_path),
        "source_db": str(config.paths.zotero_db),
        "timeout_seconds": timeout_seconds,
    }
    if ledger is not None:
        job_id = ledger.create_job("shadow_copy", result)
        ledger.checkpoint("zotero_shadow", "shadow_copy", "completed", result)
        ledger.set_job_status(job_id, "completed")
    return result


def scan_zotero_shadow(
    config: AppConfig,
    ledger: StateLedger,
    *,
    refresh_shadow: bool = True,
    limit: int | None = None,
    shadow_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    if refresh_shadow:
        copy_zotero_shadow(config, ledger, timeout_seconds=shadow_timeout_seconds)
    report = scan_shadow_to_ledger(
        shadow_db=config.paths.shadow_db,
        storage_dir=config.paths.zotero_storage,
        ledger=ledger,
        limit=limit,
    )
    return {
        "shadow_db": str(config.paths.shadow_db),
        "scanned": report.scanned,
        "summary": report.summary,
    }


def config_as_public_dict(config: AppConfig) -> dict[str, Any]:
    """Return config details safe for status output.

    Secrets and API keys are intentionally absent from AppConfig; this helper
    still avoids exposing anything beyond paths and model metadata.
    """

    return {
        "paths": {
            "zotero_db": str(config.paths.zotero_db),
            "zotero_storage": str(config.paths.zotero_storage),
            "data_dir": str(config.paths.data_dir),
            "state_db": str(config.paths.state_db),
            "shadow_db": str(config.paths.shadow_db),
            "vector_store_dir": str(config.paths.vector_store_dir),
            "extract_cache_dir": str(config.paths.extract_cache_dir),
            "normalized_dir": str(config.paths.normalized_dir),
            "embedding_cache_dir": str(config.paths.embedding_cache_dir),
        },
        "server": asdict(config.server),
        "embedding_profiles": [asdict(profile) for profile in config.embedding_profiles],
    }
