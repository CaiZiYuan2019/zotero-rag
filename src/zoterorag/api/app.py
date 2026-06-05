from __future__ import annotations

from pathlib import Path
from typing import Any

from ..backup import create_backup, plan_restore_backup, resolve_backup_manifest
from ..documents import get_document as get_document_record
from ..documents import list_documents as list_document_records
from ..embeddings import index_normalized_document, search_vector_index
from ..index import verify_vector_index
from ..models import describe_embedding_profile, list_embedding_model_catalog
from ..pipeline import cancel_ingest_job, pause_ingest_job, resume_ingest_job, start_ingest_job
from ..review import explain_attachment_review
from ..runtime import config_as_public_dict, copy_zotero_shadow, initialize_runtime, scan_zotero_shadow
from ..search import fulltext_search, metadata_search, normalize_query_image
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
        return list_embedding_model_catalog(ledger)

    @app.post("/models/embedding/activate", dependencies=[Depends(require_access)])
    def activate_embedding_model(payload: dict[str, Any]) -> dict[str, Any]:
        activated = ledger.activate_embedding_profile(
            profile_name=str(payload["profile_name"]),
            mode=str(payload["mode"]),
        )
        return {"model": describe_embedding_profile(activated), **list_embedding_model_catalog(ledger)}

    @app.get("/vectors", dependencies=[Depends(require_access)])
    def list_vector_indexes() -> dict[str, Any]:
        return {"indexes": ledger.list_vector_indexes()}

    @app.get("/vectors/{profile_name}/verify", dependencies=[Depends(require_access)])
    def verify_vectors(profile_name: str) -> dict[str, Any]:
        return verify_vector_index(ledger, profile_name).to_dict()

    @app.get("/jobs", dependencies=[Depends(require_access)])
    def list_jobs(kind: str | None = None, status: str | None = None, limit: int | None = 50) -> dict[str, Any]:
        return {"jobs": ledger.list_jobs(kind=kind, status=status, limit=limit)}

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_access)])
    def get_job(job_id: str) -> dict[str, Any]:
        job = ledger.get_job(job_id, include_events=True)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        return {"job": job}

    @app.post("/scan", dependencies=[Depends(require_access)])
    def scan_shadow(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        return scan_zotero_shadow(
            config,
            ledger,
            refresh_shadow=bool(payload.get("refresh_shadow", True)),
            limit=payload.get("limit"),
        )

    @app.post("/ingest/start", dependencies=[Depends(require_access)])
    def ingest_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        try:
            return start_ingest_job(
                ledger,
                mode=str(payload.get("mode", "incremental")),  # type: ignore[arg-type]
                zotero_key=payload.get("zotero_key"),
                include_multimodal=bool(payload.get("include_multimodal", True)),
                execute=bool(payload.get("execute", False)),
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest/pause", dependencies=[Depends(require_access)])
    def ingest_pause(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return pause_ingest_job(
                ledger,
                str(payload["job_id"]),
                reason=str(payload.get("reason", "manual pause")),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/ingest/resume", dependencies=[Depends(require_access)])
    def ingest_resume(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return resume_ingest_job(
                ledger,
                str(payload["job_id"]),
                reason=str(payload.get("reason", "manual resume")),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/ingest/cancel", dependencies=[Depends(require_access)])
    def ingest_cancel(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return cancel_ingest_job(
                ledger,
                str(payload["job_id"]),
                reason=str(payload.get("reason", "manual cancel")),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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

    @app.get("/review/explain/{attachment_key}", dependencies=[Depends(require_access)])
    def review_explain(attachment_key: str) -> dict[str, Any]:
        try:
            return explain_attachment_review(ledger, attachment_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/attachments", dependencies=[Depends(require_access)])
    def attachments(classification: str | None = None, limit: int | None = 100) -> dict[str, Any]:
        return {"attachments": ledger.list_attachments(classification=classification, limit=limit)}

    @app.get("/documents", dependencies=[Depends(require_access)])
    def documents(limit: int | None = 50, include_metadata_only: bool = True) -> dict[str, Any]:
        return {
            "documents": list_document_records(
                ledger,
                limit=limit,
                include_metadata_only=include_metadata_only,
            )
        }

    @app.get("/documents/{document_id}", dependencies=[Depends(require_access)])
    def document(
        document_id: str,
        include_chunks: bool = False,
        chunk_type: str | None = None,
        limit: int | None = 20,
        consumer: str = "llm_text",
    ) -> dict[str, Any]:
        try:
            record = get_document_record(
                ledger,
                document_id,
                include_chunks=include_chunks,
                chunk_type=chunk_type,
                limit=limit,
                consumer=consumer,  # type: ignore[arg-type]
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
        return {"document": record}

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

    @app.post("/search/fulltext", dependencies=[Depends(require_access)])
    def search_fulltext(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": fulltext_search(
                ledger,
                query=str(payload["query"]),
                chunk_type=payload.get("chunk_type"),
                limit=int(payload.get("limit", payload.get("top_k", 10))),
                consumer=str(payload.get("consumer", "llm_text")),
                image_return=str(payload.get("image_return", "none")),
            )
        }

    @app.post("/search/text", dependencies=[Depends(require_access)])
    def search_text(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("query_image") is not None:
            raise HTTPException(status_code=400, detail="text search does not accept query_image")
        return {
            "results": search_vector_index(
                ledger=ledger,
                vector_store_dir=config.paths.vector_store_dir,
                profile_name=payload.get("profile_name"),
                query=str(payload["query"]),
                mode="text",
                top_k=int(payload.get("top_k", 10)),
                consumer=str(payload.get("consumer", "llm_text")),
                image_return="none",
            )
        }

    @app.post("/search/multimodal", dependencies=[Depends(require_access)])
    def search_multimodal(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            query_image = normalize_query_image(
                payload.get("query_image"),
                allowed_roots=[config.paths.data_dir],
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "results": search_vector_index(
                ledger=ledger,
                vector_store_dir=config.paths.vector_store_dir,
                profile_name=payload.get("profile_name"),
                query=str(payload.get("query_text", payload.get("query", ""))),
                mode="multimodal",
                top_k=int(payload.get("top_k", 10)),
                consumer=str(payload.get("consumer", "manual")),
                image_return=str(payload.get("image_return", "file_ref")),
                query_image_path=query_image.file_path if query_image else None,
                query_image_base64=query_image.base64_data if query_image else None,
                query_image_mime_type=query_image.mime_type if query_image else None,
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

    @app.post("/backup/restore-plan", dependencies=[Depends(require_access)])
    def backup_restore_plan(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            manifest_path = resolve_backup_manifest(ledger, str(payload["backup"]))
            return plan_restore_backup(config, manifest_path).to_dict()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/extract/jobs", dependencies=[Depends(require_access)])
    def extract_jobs(state: str | None = None, limit: int | None = 50) -> dict[str, Any]:
        return {"jobs": ledger.list_extract_jobs(state=state, limit=limit)}

    @app.get("/normalize/artifacts", dependencies=[Depends(require_access)])
    def normalized_artifacts(limit: int | None = 50) -> dict[str, Any]:
        return {"artifacts": ledger.list_normalized_artifacts(limit=limit)}

    @app.get("/normalize/chunks/{document_id}", dependencies=[Depends(require_access)])
    def normalized_chunks(
        document_id: str,
        chunk_type: str | None = None,
        limit: int | None = 20,
    ) -> dict[str, Any]:
        return {"chunks": ledger.list_chunks(document_id, chunk_type=chunk_type, limit=limit)}

    @app.post("/embed/index-normalized", dependencies=[Depends(require_access)])
    def embed_index_normalized(payload: dict[str, Any]) -> dict[str, Any]:
        return index_normalized_document(
            ledger=ledger,
            vector_store_dir=config.paths.vector_store_dir,
            profile_name=str(payload["profile_name"]),
            document_id=str(payload["document_id"]),
        ).to_dict()

    return app
