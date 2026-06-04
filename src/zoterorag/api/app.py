from __future__ import annotations

from pathlib import Path
from typing import Any

from ..backup import create_backup
from ..runtime import config_as_public_dict, copy_zotero_shadow, initialize_runtime, scan_zotero_shadow
from ..search import metadata_search
from .security import AccessDenied, verify_api_access


def create_app(config_path: str | Path = "config/config.example.toml") -> Any:
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Request
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("FastAPI is not installed. Install zotero-rag[api].") from exc

    config, ledger = initialize_runtime(config_path)
    app = FastAPI(title="ZoteroRAG", version="0.1.0")

    def require_access(
        request: Request,
        x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    ) -> None:
        try:
            verify_api_access(
                supplied_token=x_api_token,
                client_host=request.client.host if request.client else None,
                require_api_token=config.server.require_api_token,
            )
        except AccessDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status", dependencies=[Depends(require_access)])
    def status() -> dict[str, Any]:
        return {"runtime": config_as_public_dict(config), "state": ledger.status_summary()}

    @app.get("/models/embedding", dependencies=[Depends(require_access)])
    def list_embedding_models() -> dict[str, Any]:
        return {"models": ledger.list_embedding_profiles()}

    @app.get("/vectors", dependencies=[Depends(require_access)])
    def list_vector_indexes() -> dict[str, Any]:
        return {"indexes": ledger.list_vector_indexes()}

    @app.post("/scan", dependencies=[Depends(require_access)])
    def scan_shadow(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        return scan_zotero_shadow(
            config,
            ledger,
            refresh_shadow=bool(payload.get("refresh_shadow", True)),
            limit=payload.get("limit"),
        )

    @app.post("/review/include", dependencies=[Depends(require_access)])
    def include_attachment(payload: dict[str, Any]) -> dict[str, str]:
        key = str(payload["attachment_key"])
        reason = str(payload.get("reason", "manual include"))
        ledger.upsert_review_rule(key, "include", reason)
        return {"attachment_key": key, "decision": "include"}

    @app.post("/review/exclude", dependencies=[Depends(require_access)])
    def exclude_attachment(payload: dict[str, Any]) -> dict[str, str]:
        key = str(payload["attachment_key"])
        reason = str(payload.get("reason", "manual exclude"))
        ledger.upsert_review_rule(key, "exclude", reason)
        return {"attachment_key": key, "decision": "exclude"}

    @app.get("/review", dependencies=[Depends(require_access)])
    def review_rules() -> dict[str, Any]:
        return {
            "rules": ledger.list_review_rules(),
            "candidates": ledger.list_attachments(classification="needs_review"),
        }

    @app.get("/attachments", dependencies=[Depends(require_access)])
    def attachments(classification: str | None = None, limit: int | None = 100) -> dict[str, Any]:
        return {"attachments": ledger.list_attachments(classification=classification, limit=limit)}

    @app.post("/search/metadata", dependencies=[Depends(require_access)])
    def search_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": metadata_search(
                ledger,
                query=str(payload["query"]),
                classification=payload.get("classification"),
                limit=int(payload.get("limit", payload.get("top_k", 10))),
                consumer=str(payload.get("consumer", "llm_text")),
            )
        }

    @app.post("/backup/create", dependencies=[Depends(require_access)])
    def backup_create(payload: dict[str, Any]) -> dict[str, Any]:
        return create_backup(
            config,
            ledger,
            mode=str(payload.get("mode", "snapshot")),
            out_dir=payload["out"],
            config_path=config_path,
        ).to_dict()

    @app.get("/backup/list", dependencies=[Depends(require_access)])
    def backup_list() -> dict[str, Any]:
        return {"backups": ledger.list_backups()}

    @app.get("/extract/jobs", dependencies=[Depends(require_access)])
    def extract_jobs(state: str | None = None, limit: int | None = 50) -> dict[str, Any]:
        return {"jobs": ledger.list_extract_jobs(state=state, limit=limit)}

    return app
