from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

from .api.security import AccessDenied, verify_api_access
from .config import AppConfig
from .db import StateLedger
from .index import LocalVectorStore, VectorRecord, verify_vector_index
from .providers import provider_readiness
from .zotero import ZoteroShadow


def run_runtime_diagnostics(
    config: AppConfig,
    ledger: StateLedger,
    *,
    verify_vectors: bool = False,
    self_test_vector_store: bool = False,
) -> dict[str, Any]:
    """Run non-invasive readiness checks for the local control plane.

    Diagnostics intentionally do not refresh the Zotero shadow, do not open the
    live Zotero database with SQLite, and do not call MinerU or embedding
    providers. They are safe to run while Zotero is open.
    """

    checks = {
        "state": check_state_db(config, ledger),
        "zotero_source": check_zotero_source_paths(config),
        "shadow": check_existing_shadow(config),
        "vectors": check_vector_indexes(config, ledger, verify_vectors=verify_vectors),
        "vector_staging_self_test": (
            check_vector_staging_self_test(config)
            if self_test_vector_store
            else {
                "ok": True,
                "enabled": False,
                "status": "skipped",
                "note": "pass self_test_vector_store=true to create a temporary local vector index and verify staging",
            }
        ),
        "api_access": check_api_access(config),
        "providers": check_provider_configuration(config.paths.data_dir.parent / ".env"),
        "external_execution": {
            "ok": True,
            "mineru_executed": False,
            "embedding_executed": False,
            "note": "diagnostics never call external extraction or embedding providers",
        },
    }
    return {
        "ok": all(item.get("ok", False) for item in checks.values()),
        "checks": checks,
    }


def check_state_db(config: AppConfig, ledger: StateLedger) -> dict[str, Any]:
    state_db = config.paths.state_db
    try:
        summary = ledger.status_summary()
        journal_mode = ledger.conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = ledger.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    except Exception as exc:
        return {"ok": False, "path": str(state_db), "error": str(exc)}
    return {
        "ok": state_db.is_file() and str(journal_mode).casefold() == "wal",
        "path": str(state_db),
        "exists": state_db.is_file(),
        "journal_mode": str(journal_mode),
        "busy_timeout_ms": int(busy_timeout),
        "schema_version": summary.get("schema_version"),
        "summary": summary,
    }


def check_zotero_source_paths(config: AppConfig) -> dict[str, Any]:
    source = config.paths.zotero_db
    storage = config.paths.zotero_storage
    same_as_shadow = source.resolve() == config.paths.shadow_db.resolve()
    return {
        "ok": source.is_file() and not same_as_shadow,
        "source_db": str(source),
        "source_exists": source.is_file(),
        "storage_dir": str(storage),
        "storage_exists": storage.is_dir(),
        "same_as_shadow": same_as_shadow,
        "opened_source_db": False,
    }


def check_existing_shadow(config: AppConfig) -> dict[str, Any]:
    shadow_db = config.paths.shadow_db
    if not shadow_db.is_file():
        return {"ok": False, "path": str(shadow_db), "exists": False, "status": "not_created"}
    try:
        shadow = ZoteroShadow(shadow_db)
        try:
            pdf_count = shadow.pdf_count()
        finally:
            shadow.close()
    except Exception as exc:
        return {"ok": False, "path": str(shadow_db), "exists": True, "status": "unreadable", "error": str(exc)}
    return {"ok": True, "path": str(shadow_db), "exists": True, "status": "readable", "pdf_count": pdf_count}


def check_vector_indexes(
    config: AppConfig,
    ledger: StateLedger,
    *,
    verify_vectors: bool,
) -> dict[str, Any]:
    indexes = ledger.list_vector_indexes()
    result: dict[str, Any] = {
        "ok": True,
        "verify_vectors": verify_vectors,
        "vector_store_dir": str(config.paths.vector_store_dir),
        "indexes": [],
    }
    for index in indexes:
        path = Path(index["path"])
        item = {
            **index,
            "path_exists": path.is_file(),
            "provenance": vector_index_provenance(ledger, index),
        }
        if verify_vectors:
            verification = verify_vector_index(ledger, index["profile_name"]).to_dict()
            item["verification"] = verification
            result["ok"] = result["ok"] and bool(verification["ok"])
        result["indexes"].append(item)
    return result


def vector_index_provenance(ledger: StateLedger, index: dict[str, Any]) -> dict[str, Any]:
    batches = ledger.list_embedding_batches(
        profile_name=index["profile_name"],
        status="completed",
        limit=None,
    )
    active_version = str(index.get("active_version") or "")
    matching_active_version = [
        batch for batch in batches if active_version and batch["batch_hash"] == active_version
    ]
    if int(index.get("chunk_count") or 0) == 0:
        status = "empty"
    elif matching_active_version:
        status = "tracked_active_version"
    elif batches:
        status = "tracked_profile"
    else:
        status = "unattributed"
    return {
        "status": status,
        "completed_batch_count": len(batches),
        "active_version": active_version,
        "active_version_has_completed_batch": bool(matching_active_version),
    }


def check_vector_staging_self_test(config: AppConfig) -> dict[str, Any]:
    """Exercise local vector staging in a temporary store.

    This writes only under the OS temporary directory and removes the temporary
    directory on success/failure. It does not read Zotero, submit extraction
    jobs, or call embedding providers.
    """

    root = Path(tempfile.gettempdir()) / "zoterorag-diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    store: LocalVectorStore | None = None
    tmp = root / f"vector-staging-{uuid.uuid4().hex}"
    try:
        tmp.mkdir(parents=True, exist_ok=False)
        store = LocalVectorStore(tmp / "vectors.sqlite", profile_name="diagnostic", dimension=3)
        first = VectorRecord(
            record_id="record-a",
            document_id="doc-a",
            chunk_id="chunk-a",
            vector=[1.0, 0.0, 0.0],
            text="first staged version",
            modality="text",
        )
        store.upsert([first], index_version="batch-1")
        before_publish = store.search([1.0, 0.0, 0.0], top_k=1, modality="text")
        store.publish_version("batch-1")
        after_publish = store.search([1.0, 0.0, 0.0], top_k=1, modality="text")

        replacement = VectorRecord(
            record_id="record-a",
            document_id="doc-a",
            chunk_id="chunk-a",
            vector=[0.0, 1.0, 0.0],
            text="replacement staged version",
            modality="text",
        )
        store.upsert([replacement], index_version="batch-2")
        active_during_rebuild = store.search([1.0, 0.0, 0.0], top_k=1, modality="text")
        store.publish_version("batch-2")
        after_republish = store.search([0.0, 1.0, 0.0], top_k=1, modality="text")
        counts = store.counts()
        store.close()
        store = None

        ok = (
            before_publish == []
            and bool(after_publish)
            and after_publish[0]["text"] == "first staged version"
            and bool(active_during_rebuild)
            and active_during_rebuild[0]["text"] == "first staged version"
            and bool(after_republish)
            and after_republish[0]["text"] == "replacement staged version"
            and counts == {"documents": 1, "chunks": 1}
        )
        return {
            "ok": ok,
            "enabled": True,
            "status": "passed" if ok else "failed",
            "temp_root": str(root),
            "checks": {
                "staged_invisible_before_publish": before_publish == [],
                "published_visible": bool(after_publish)
                and after_publish[0]["text"] == "first staged version",
                "replacement_invisible_before_republish": bool(active_during_rebuild)
                and active_during_rebuild[0]["text"] == "first staged version",
                "replacement_visible_after_republish": bool(after_republish)
                and after_republish[0]["text"] == "replacement staged version",
                "counts_after_republish": counts,
            },
        }
    except Exception as exc:
        return {"ok": False, "enabled": True, "status": "error", "error": str(exc)}
    finally:
        if store is not None:
            store.close()
        shutil.rmtree(tmp, ignore_errors=True)


def check_api_access(config: AppConfig) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    try:
        verify_api_access(
            supplied_token=None,
            client_host="127.0.0.1",
            require_api_token=config.server.require_api_token,
        )
        checks["loopback_without_token_allowed"] = True
    except AccessDenied:
        checks["loopback_without_token_allowed"] = False

    try:
        verify_api_access(
            supplied_token=None,
            client_host="10.0.0.5",
            require_api_token=config.server.require_api_token,
        )
        checks["external_without_token_denied"] = False
    except AccessDenied:
        checks["external_without_token_denied"] = True

    return {
        "ok": checks["external_without_token_denied"],
        "require_api_token": config.server.require_api_token,
        **checks,
    }


def check_provider_configuration(env_path: str | Path = ".env") -> dict[str, Any]:
    env_path = Path(env_path)
    readiness = provider_readiness(env_path)
    return {
        "ok": env_path.is_file(),
        "env_path": str(env_path),
        "env_exists": env_path.is_file(),
        **readiness,
        "note": "provider diagnostics inspect configuration only and never call MinerU or qwen",
    }
