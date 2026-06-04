from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .db import StateLedger
from .zotero import create_shadow_copy, scan_shadow_to_ledger


def initialize_runtime(config_path: str | Path = "config/config.example.toml") -> tuple[AppConfig, StateLedger]:
    config = load_config(config_path)
    config.ensure_runtime_dirs()
    ledger = StateLedger(config.paths.state_db)
    ledger.upsert_embedding_profiles(config.embedding_profiles)
    for profile in config.embedding_profiles:
        vector_path = config.paths.vector_store_dir / profile.name / "vectors.sqlite"
        ledger.register_vector_index(
            profile_name=profile.name,
            backend="sqlite-local",
            path=vector_path,
            document_count=0,
            chunk_count=0,
            active=profile.enabled,
        )
    return config, ledger


def copy_zotero_shadow(config: AppConfig, ledger: StateLedger | None = None) -> dict[str, Any]:
    shadow_path = create_shadow_copy(config.paths.zotero_db, config.paths.shadow_db)
    result = {"shadow_db": str(shadow_path), "source_db": str(config.paths.zotero_db)}
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
) -> dict[str, Any]:
    if refresh_shadow:
        copy_zotero_shadow(config, ledger)
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
        },
        "server": asdict(config.server),
        "embedding_profiles": [asdict(profile) for profile in config.embedding_profiles],
    }
