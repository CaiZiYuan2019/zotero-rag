from __future__ import annotations

from pathlib import Path
from typing import Any

from ..backup import create_backup, plan_restore_backup, resolve_backup_manifest
from ..diagnostics import run_runtime_diagnostics
from ..documents import get_document as get_document_record
from ..documents import list_documents as list_document_records
from ..embeddings import index_normalized_document, search_vector_index
from ..extractors import build_extract_recovery_plan
from ..index import verify_vector_index
from ..models import describe_embedding_profile, list_embedding_model_catalog
from ..pipeline import (
    build_progress_report,
    cancel_ingest_job,
    create_reembed_plan,
    pause_ingest_job,
    resume_ingest_job,
    start_ingest_job,
    start_reembed_job,
)
from ..providers import build_qwen_embedding_provider, provider_readiness
from ..review import explain_attachment_review
from ..runtime import config_as_public_dict, copy_zotero_shadow, initialize_runtime, scan_zotero_shadow
from ..search import fulltext_search, metadata_search, normalize_query_image
from .security import AccessDenied, verify_api_access


def create_app(config_path: str | Path = "config/config.example.toml") -> Any:
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Request
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("FastAPI is not installed. Install zotero-rag[api].") from exc

    # With postponed annotations enabled, FastAPI resolves the dependency
    # signature through module globals. Request is imported lazily to keep the
    # API extra optional, so expose it here before route functions are defined.
    globals()["Request"] = Request

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
        return {
            "runtime": config_as_public_dict(config),
            "state": ledger.status_summary(),
            "progress": build_progress_report(ledger),
        }

    @app.get("/diagnostics", dependencies=[Depends(require_access)])
    def diagnostics(verify_vectors: bool = False, self_test_vector_store: bool = False) -> dict[str, Any]:
        return run_runtime_diagnostics(
            config,
            ledger,
            verify_vectors=verify_vectors,
            self_test_vector_store=self_test_vector_store,
        )

    @app.get("/providers/status", dependencies=[Depends(require_access)])
    def providers_status() -> dict[str, Any]:
        return provider_readiness(".env")

    @app.get("/progress", dependencies=[Depends(require_access)])
    def progress(include_ingest_plan: bool = True, recent_limit: int = 10) -> dict[str, Any]:
        return build_progress_report(
            ledger,
            include_ingest_plan=include_ingest_plan,
            recent_limit=recent_limit,
        )

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
            shadow_timeout_seconds=float(payload.get("shadow_timeout_seconds", 30.0)),
        )

    @app.post("/ingest/start", dependencies=[Depends(require_access)])
    def ingest_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        try:
            execute = bool(payload.get("execute", False))
            kwargs: dict[str, Any] = dict(
                mode=str(payload.get("mode", "incremental")),  # type: ignore[arg-type]
                zotero_key=payload.get("zotero_key"),
                include_multimodal=bool(payload.get("include_multimodal", True)),
                execute=execute,
            )
            if execute:
                kwargs["extract_manager"] = _build_extract_manager(
                    ledger, config.paths.extract_cache_dir, str(payload.get("env", ".env"))
                )
                kwargs["extract_cache_dir"] = config.paths.extract_cache_dir
                kwargs["normalized_dir"] = config.paths.normalized_dir
                kwargs["vector_store_dir"] = config.paths.vector_store_dir
            return start_ingest_job(ledger, **kwargs)
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
        try:
            return {
                "results": metadata_search(
                    ledger,
                    query=str(payload["query"]),
                    classification=payload.get("classification"),
                    limit=int(payload.get("limit", payload.get("top_k", 10))),
                    consumer=str(payload.get("consumer", "llm_text")),
                    rerank=bool(payload.get("rerank", False)),
                )
            }
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/search/fulltext", dependencies=[Depends(require_access)])
    def search_fulltext(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return {
                "results": fulltext_search(
                    ledger,
                    query=str(payload["query"]),
                    chunk_type=payload.get("chunk_type"),
                    limit=int(payload.get("limit", payload.get("top_k", 10))),
                    consumer=str(payload.get("consumer", "llm_text")),
                    image_return=str(payload.get("image_return", "none")),
                    rerank=bool(payload.get("rerank", False)),
                )
            }
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/search/text", dependencies=[Depends(require_access)])
    def search_text(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("query_image") is not None:
            raise HTTPException(status_code=400, detail="text search does not accept query_image")
        try:
            profile = selected_embedding_profile(ledger, payload.get("profile_name"), mode="text")
            provider = explicit_embedding_provider(payload, profile)
            return {
                "results": search_vector_index(
                    ledger=ledger,
                    vector_store_dir=config.paths.vector_store_dir,
                    profile_name=profile["name"],
                    query=str(payload["query"]),
                    mode="text",
                    top_k=int(payload.get("top_k", 10)),
                    consumer=str(payload.get("consumer", "llm_text")),
                    image_return="none",
                    rerank=bool(payload.get("rerank", False)),
                    provider=provider,
                )
            }
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except (KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/search/multimodal", dependencies=[Depends(require_access)])
    def search_multimodal(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            query_image = normalize_query_image(
                payload.get("query_image"),
                allowed_roots=[config.paths.data_dir],
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            profile = selected_embedding_profile(ledger, payload.get("profile_name"), mode="multimodal")
            provider = explicit_embedding_provider(payload, profile)
            return {
                "results": search_vector_index(
                    ledger=ledger,
                    vector_store_dir=config.paths.vector_store_dir,
                    profile_name=profile["name"],
                    query=str(payload.get("query_text", payload.get("query", ""))),
                    mode="multimodal",
                    top_k=int(payload.get("top_k", 10)),
                    consumer=str(payload.get("consumer", "manual")),
                    image_return=str(payload.get("image_return", "file_ref")),
                    max_images=int(payload.get("max_images", 5)),
                    max_image_bytes=int(payload.get("max_image_bytes", 256 * 1024)),
                    rerank=bool(payload.get("rerank", False)),
                    query_image_path=query_image.file_path if query_image else None,
                    query_image_base64=query_image.base64_data if query_image else None,
                    query_image_mime_type=query_image.mime_type if query_image else None,
                    provider=provider,
                )
            }
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except (KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    @app.get("/extract/recovery-plan", dependencies=[Depends(require_access)])
    def extract_recovery_plan(state: str | None = None, limit: int | None = 50) -> dict[str, Any]:
        return build_extract_recovery_plan(ledger.list_extract_jobs(state=state, limit=limit))

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

    @app.get("/embed/batches", dependencies=[Depends(require_access)])
    def embed_batches(
        profile_name: str | None = None,
        document_id: str | None = None,
        status: str | None = None,
        limit: int | None = 50,
    ) -> dict[str, Any]:
        return {
            "batches": ledger.list_embedding_batches(
                profile_name=profile_name,
                document_id=document_id,
                status=status,
                limit=limit,
            )
        }

    @app.post("/embed/index-normalized", dependencies=[Depends(require_access)])
    def embed_index_normalized(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            profile = selected_embedding_profile(ledger, str(payload["profile_name"]), mode=None)
            provider = explicit_embedding_provider(payload, profile)
            return index_normalized_document(
                ledger=ledger,
                vector_store_dir=config.paths.vector_store_dir,
                profile_name=profile["name"],
                document_id=str(payload["document_id"]),
                provider=provider,
                allow_stub_provider=bool(payload.get("allow_stub_provider", False)),
            ).to_dict()
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except (KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/reembed/plan", dependencies=[Depends(require_access)])
    def reembed_plan(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            # This endpoint is intentionally read-only. It lets operators audit
            # long rebuild recovery state without creating a job or calling qwen.
            return create_reembed_plan(
                ledger,
                vector_store_dir=config.paths.vector_store_dir,
                profile_name=str(payload["profile_name"]),
                document_id=payload.get("document_id"),
                force=bool(payload.get("force", False)),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/reembed/from-normalized", dependencies=[Depends(require_access)])
    def reembed_from_normalized(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return start_reembed_job(
                ledger,
                vector_store_dir=config.paths.vector_store_dir,
                profile_name=str(payload["profile_name"]),
                document_id=payload.get("document_id"),
                force=bool(payload.get("force", False)),
                execute=bool(payload.get("execute", False)),
                allow_stub_provider=bool(payload.get("allow_stub_provider", False)),
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _build_extract_manager(ledger: Any, extract_cache_dir: Any, env_path: str) -> Any:
    """Build an ExtractionManager wired to real MinerU provider and key pool."""
    from ..providers import build_mineru_provider
    from ..extractors import ExtractorKeyPool, ExtractionManager

    key_pool = ExtractorKeyPool.from_env_file(env_path)
    provider = build_mineru_provider(env_path)
    return ExtractionManager(
        ledger=ledger,
        cache_dir=extract_cache_dir,
        provider=provider,
        key_pool=key_pool,
    )


def explicit_embedding_provider(payload: dict[str, Any], profile: dict[str, Any]) -> Any:
    provider_name = str(payload.get("embedding_provider", "stub"))
    if provider_name == "stub":
        return None
    if provider_name == "qwen3vl":
        return build_qwen_embedding_provider(profile, str(payload.get("env", ".env")))
    raise ValueError(f"unsupported embedding_provider: {provider_name}")


def selected_embedding_profile(ledger: Any, profile_name: str | None, *, mode: str | None) -> dict[str, Any]:
    profiles = ledger.list_embedding_profiles()
    if profile_name:
        for profile in profiles:
            if profile["name"] == profile_name:
                return profile
        raise KeyError(f"embedding profile not found: {profile_name}")
    if mode is None:
        raise ValueError("profile_name is required")
    expected_modality = "text" if mode == "text" else "multimodal"
    default_flag = "default_for_text" if mode == "text" else "default_for_multimodal"
    for profile in profiles:
        if profile["enabled"] and profile["modality"] == expected_modality and profile[default_flag]:
            return profile
    raise KeyError(f"no default {mode} embedding profile is configured")
