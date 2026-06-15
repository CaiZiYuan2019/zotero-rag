from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .backup import create_backup, plan_restore_backup, resolve_backup_manifest, restore_backup, verify_manifest_files
from .api.server import serve_api, validate_serve_access
from .diagnostics import run_runtime_diagnostics
from .documents import get_document, list_documents
from .embeddings import index_normalized_document, search_vector_index
from .extractors import (
    ExtractionManager,
    ExtractionRequest,
    ExtractorKeyPool,
    StubExtractorProvider,
    build_extract_recovery_plan,
)
from .index import verify_vector_index
from .models import describe_embedding_profile, list_embedding_model_catalog
from .normalize import normalize_markdown_document
from .pipeline import (
    build_progress_report,
    cancel_ingest_job,
    create_reembed_plan,
    pause_ingest_job,
    resume_ingest_job,
    start_ingest_job,
    start_reembed_job,
)
from .providers import build_qwen_embedding_provider, provider_readiness
from .review import explain_attachment_review
from .runtime import config_as_public_dict, copy_zotero_shadow, initialize_runtime, scan_zotero_shadow
from .search import fulltext_search, metadata_search, normalize_query_image
from .zotero import ZoteroShadow


def emit(data: object) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zoterorag")
    parser.add_argument("--config", default="config/config.example.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-state", help="Create runtime directories and initialize state.sqlite.")
    sub.add_parser("status", help="Show runtime and state status.")
    serve = sub.add_parser("serve", help="Start the FastAPI control server.")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--check", action="store_true", help="Only validate bind/auth settings without starting Uvicorn.")
    doctor = sub.add_parser("doctor", help="Run non-invasive local readiness diagnostics.")
    doctor.add_argument("--verify-vectors", action="store_true")
    doctor.add_argument(
        "--self-test-vector-store",
        action="store_true",
        help="Create a temporary local vector index to verify staged publish semantics.",
    )
    progress = sub.add_parser("progress", help="Show detailed local build progress without executing workers.")
    progress.add_argument("--no-ingest-plan", action="store_true")
    progress.add_argument("--recent-limit", type=int, default=10)
    shadow_copy = sub.add_parser("shadow-copy", help="Create a read-only Zotero shadow copy.")
    shadow_copy.add_argument("--timeout-seconds", type=float, default=30.0)
    scan = sub.add_parser("scan", help="Copy Zotero shadow and classify attachments into state.")
    scan.add_argument("--no-refresh-shadow", action="store_true", help="Scan the existing shadow DB without recopying Zotero.")
    scan.add_argument("--limit", type=int, default=None)
    scan.add_argument("--shadow-timeout-seconds", type=float, default=30.0)

    providers = sub.add_parser("providers", help="Inspect external provider configuration without executing it.")
    providers_sub = providers.add_subparsers(dest="providers_command", required=True)
    providers_status = providers_sub.add_parser("status")
    providers_status.add_argument("--env", default=".env")

    models = sub.add_parser("models", help="List configured embedding profiles.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list")
    models_activate = models_sub.add_parser("activate", help="Set the default embedding profile for one search mode.")
    models_activate.add_argument("--profile", required=True)
    models_activate.add_argument("--mode", required=True, choices=("text", "multimodal"))

    vectors = sub.add_parser("vectors", help="Inspect local vector index registrations.")
    vectors_sub = vectors.add_subparsers(dest="vectors_command", required=True)
    vectors_sub.add_parser("list")
    vectors_verify = vectors_sub.add_parser("verify")
    vectors_verify.add_argument("--profile", required=True)

    jobs = sub.add_parser("jobs", help="Inspect pipeline jobs and progress events.")
    jobs_sub = jobs.add_subparsers(dest="jobs_command", required=True)
    jobs_list = jobs_sub.add_parser("list")
    jobs_list.add_argument("--kind", default=None)
    jobs_list.add_argument("--status", default=None)
    jobs_list.add_argument("--limit", type=int, default=50)
    jobs_show = jobs_sub.add_parser("show")
    jobs_show.add_argument("job_id")

    ingest = sub.add_parser("ingest", help="Plan and control long-running ingest jobs.")
    ingest_sub = ingest.add_subparsers(dest="ingest_command", required=True)
    ingest_start = ingest_sub.add_parser("start", help="Create a non-executing ingest plan in the state ledger.")
    ingest_start.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    ingest_start.add_argument("--zotero-key", default=None)
    ingest_start.add_argument("--text-only", action="store_true")
    ingest_start.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for future workers. Currently rejected to avoid external API calls.",
    )
    ingest_pause = ingest_sub.add_parser("pause")
    ingest_pause.add_argument("job_id")
    ingest_pause.add_argument("--reason", default="manual pause")
    ingest_resume = ingest_sub.add_parser("resume")
    ingest_resume.add_argument("job_id")
    ingest_resume.add_argument("--reason", default="manual resume")
    ingest_cancel = ingest_sub.add_parser("cancel")
    ingest_cancel.add_argument("job_id")
    ingest_cancel.add_argument("--reason", default="manual cancel")

    review = sub.add_parser("review", help="Manage manual include/exclude rules.")
    review_sub = review.add_subparsers(dest="review_command", required=True)
    review_sub.add_parser("list")
    include = review_sub.add_parser("include")
    include.add_argument("--attachment-key", required=True)
    include.add_argument("--reason", default="manual include")
    exclude = review_sub.add_parser("exclude")
    exclude.add_argument("--attachment-key", required=True)
    exclude.add_argument("--reason", default="manual exclude")
    review_explain = review_sub.add_parser("explain")
    review_explain.add_argument("--attachment-key", required=True)

    attachments = sub.add_parser("attachments", help="List persisted attachment scan results.")
    attachments.add_argument("--classification", default=None)
    attachments.add_argument("--limit", type=int, default=50)

    documents = sub.add_parser("documents", help="Inspect document-level control records.")
    documents_sub = documents.add_subparsers(dest="documents_command", required=True)
    documents_list = documents_sub.add_parser("list")
    documents_list.add_argument("--limit", type=int, default=50)
    documents_list.add_argument("--normalized-only", action="store_true")
    documents_show = documents_sub.add_parser("show")
    documents_show.add_argument("document_id")
    documents_show.add_argument("--chunks", action="store_true")
    documents_show.add_argument("--chunk-type", default=None, choices=("text", "image"))
    documents_show.add_argument("--limit", type=int, default=20)
    documents_show.add_argument("--consumer", default="manual", choices=("manual", "llm_text", "llm_multimodal"))

    search_metadata = sub.add_parser("search-metadata", help="Search scanned Zotero metadata without embeddings.")
    search_metadata.add_argument("query")
    search_metadata.add_argument("--classification", default=None)
    search_metadata.add_argument("--limit", type=int, default=10)
    search_metadata.add_argument("--consumer", default="llm_text", choices=("manual", "llm_text", "llm_multimodal"))
    search_metadata.add_argument("--rerank", action="store_true")

    search_fulltext = sub.add_parser("search-fulltext", help="Search normalized fulltext chunks without embeddings.")
    search_fulltext.add_argument("query")
    search_fulltext.add_argument("--chunk-type", default=None, choices=("text", "image"))
    search_fulltext.add_argument("--limit", type=int, default=10)
    search_fulltext.add_argument("--consumer", default="llm_text", choices=("manual", "llm_text", "llm_multimodal"))
    search_fulltext.add_argument("--image-return", default="none", choices=("file_ref", "base64", "none"))
    search_fulltext.add_argument("--rerank", action="store_true")

    backup = sub.add_parser("backup", help="Create and inspect ZoteroRAG runtime backups.")
    backup_sub = backup.add_subparsers(dest="backup_command", required=True)
    backup_create = backup_sub.add_parser("create")
    backup_create.add_argument("--mode", choices=("snapshot", "full"), default="snapshot")
    backup_create.add_argument("--out", required=True)
    backup_sub.add_parser("list")
    backup_verify = backup_sub.add_parser("verify")
    backup_verify.add_argument("manifest")
    backup_restore = backup_sub.add_parser("restore")
    backup_restore.add_argument("backup", help="Backup id or backup_manifest.json path.")
    backup_restore.add_argument("--pre-restore-out", required=True)
    backup_restore.add_argument("--confirm", action="store_true")

    extract = sub.add_parser("extract", help="Manage PDF extraction jobs and cache state.")
    extract_sub = extract.add_subparsers(dest="extract_command", required=True)
    extract_dry = extract_sub.add_parser("dry-run", help="Create or reuse a non-network stub extraction job.")
    extract_dry.add_argument("--pdf", required=True, help="PDF path or any local file for stub-only tests.")
    extract_dry.add_argument("--attachment-key", default=None)
    extract_dry.add_argument("--sha256", default=None, help="Known PDF sha256. If omitted, the file is hashed.")
    extract_dry.add_argument("--pages", default="", help="Canonical selected page range used in the cache key.")
    extract_dry.add_argument(
        "--selected-page-count",
        type=int,
        default=1,
        help="Selected page count for timeout estimate; MinerU timeout defaults to pages*6+30.",
    )
    extract_dry.add_argument("--options-json", default="{}", help="Additional extractor options as JSON.")
    extract_dry.add_argument(
        "--env",
        default=".env",
        help="Optional env file for MinerU key aliases. Values are never printed or stored.",
    )
    extract_jobs = extract_sub.add_parser("jobs", help="List persisted extraction jobs.")
    extract_jobs.add_argument("--state", default=None)
    extract_jobs.add_argument("--limit", type=int, default=50)
    extract_recovery = extract_sub.add_parser("recovery-plan", help="Plan extraction resume actions without executing them.")
    extract_recovery.add_argument("--state", default=None)
    extract_recovery.add_argument("--limit", type=int, default=50)

    normalize = sub.add_parser("normalize", help="Normalize local Markdown extraction artifacts.")
    normalize_sub = normalize.add_subparsers(dest="normalize_command", required=True)
    normalize_markdown = normalize_sub.add_parser("markdown", help="Normalize a local MinerU-style full.md.")
    normalize_markdown.add_argument("--markdown", required=True)
    normalize_markdown.add_argument("--document-id", required=True)
    normalize_markdown.add_argument("--attachment-key", default=None)
    normalize_markdown.add_argument("--extract-job-id", default=None)
    normalize_sub.add_parser("list")
    normalize_chunks = normalize_sub.add_parser("chunks")
    normalize_chunks.add_argument("--document-id", required=True)
    normalize_chunks.add_argument("--chunk-type", default=None, choices=("text", "image"))
    normalize_chunks.add_argument("--limit", type=int, default=20)

    embed = sub.add_parser("embed", help="Offline embedding/index commands. Defaults to stub provider.")
    embed_sub = embed.add_subparsers(dest="embed_command", required=True)
    embed_index = embed_sub.add_parser("index-normalized", help="Index normalized chunks into a local vector store.")
    embed_index.add_argument("--document-id", required=True)
    embed_index.add_argument("--profile", required=True)
    embed_index.add_argument(
        "--allow-stub-provider",
        action="store_true",
        help="Allow local stub embeddings for non-stub profiles during control-plane tests.",
    )
    embed_index.add_argument(
        "--embedding-provider",
        choices=("stub", "qwen3vl"),
        default="stub",
        help="Provider used for vector creation. qwen3vl makes real API calls and must be selected explicitly.",
    )
    embed_index.add_argument("--env", default=".env", help="Env file containing qwen credentials when using qwen3vl.")
    embed_batches = embed_sub.add_parser("batches", help="List persisted embedding batch progress.")
    embed_batches.add_argument("--profile", default=None)
    embed_batches.add_argument("--document-id", default=None)
    embed_batches.add_argument("--status", default=None)
    embed_batches.add_argument("--limit", type=int, default=50)

    reembed = sub.add_parser("reembed", help="Plan or execute vector-only rebuilds from normalized artifacts.")
    reembed.add_argument("--profile", required=True)
    reembed.add_argument("--from-normalized", action="store_true", required=True)
    reembed.add_argument("--document-id", default=None)
    reembed.add_argument("--force", action="store_true")
    reembed.add_argument("--plan-only", action="store_true", help="Return a read-only rebuild plan without creating a job.")
    reembed.add_argument("--execute", action="store_true", help="Execute local stub indexing for pending documents.")
    reembed.add_argument(
        "--allow-stub-provider",
        action="store_true",
        help="Allow local stub embeddings for non-stub profiles during control-plane tests.",
    )

    search_vector = sub.add_parser("search-vector", help="Search local vector indexes.")
    search_vector.add_argument("query")
    search_vector.add_argument("--mode", choices=("text", "multimodal"), default="text")
    search_vector.add_argument("--profile", default=None)
    search_vector.add_argument("--top-k", type=int, default=10)
    search_vector.add_argument("--consumer", default="llm_text", choices=("manual", "llm_text", "llm_multimodal"))
    search_vector.add_argument("--image-return", default="none", choices=("file_ref", "base64", "none"))
    search_vector.add_argument("--rerank", action="store_true")
    search_vector.add_argument("--max-images", type=int, default=5)
    search_vector.add_argument("--max-image-bytes", type=int, default=256 * 1024)
    search_vector.add_argument("--query-image-file", default=None)
    search_vector.add_argument("--query-image-base64", default=None)
    search_vector.add_argument("--query-image-mime-type", default=None)
    search_vector.add_argument(
        "--embedding-provider",
        choices=("stub", "qwen3vl"),
        default="stub",
        help="Provider used for query embedding. qwen3vl makes real API calls and must be selected explicitly.",
    )
    search_vector.add_argument("--env", default=".env", help="Env file containing qwen credentials when using qwen3vl.")

    inspect = sub.add_parser("inspect-shadow", help="Read summary from an existing shadow DB.")
    inspect.add_argument("--limit", type=int, default=5)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, ledger = initialize_runtime(args.config)
    try:
        if args.command == "init-state":
            emit({"ok": True, "state": ledger.status_summary(), "runtime": config_as_public_dict(config)})
            return 0

        if args.command == "status":
            emit(
                {
                    "state": ledger.status_summary(),
                    "progress": build_progress_report(ledger),
                    "runtime": config_as_public_dict(config),
                }
            )
            return 0

        if args.command == "serve":
            preflight = validate_serve_access(config, host=args.host)
            if args.check:
                emit({"serve": {**preflight, "port": args.port or preflight["port"]}})
                return 0 if preflight["ok"] else 1
            if not preflight["ok"]:
                emit({"ok": False, "error": preflight["error"], "serve": preflight})
                return 1
            try:
                ledger.close()
                serve_api(args.config, host=args.host, port=args.port)
            except RuntimeError as exc:
                emit({"ok": False, "error": str(exc)})
                return 1
            return 0

        if args.command == "doctor":
            result = run_runtime_diagnostics(
                config,
                ledger,
                verify_vectors=args.verify_vectors,
                self_test_vector_store=args.self_test_vector_store,
            )
            emit(result)
            return 0 if result["ok"] else 1

        if args.command == "progress":
            emit(
                build_progress_report(
                    ledger,
                    include_ingest_plan=not args.no_ingest_plan,
                    recent_limit=args.recent_limit,
                )
            )
            return 0

        if args.command == "shadow-copy":
            emit(copy_zotero_shadow(config, ledger, timeout_seconds=args.timeout_seconds))
            return 0

        if args.command == "scan":
            emit(
                scan_zotero_shadow(
                    config,
                    ledger,
                    refresh_shadow=not args.no_refresh_shadow,
                    limit=args.limit,
                    shadow_timeout_seconds=args.shadow_timeout_seconds,
                )
            )
            return 0

        if args.command == "providers" and args.providers_command == "status":
            emit(provider_readiness(args.env))
            return 0

        if args.command == "models" and args.models_command == "list":
            emit(list_embedding_model_catalog(ledger))
            return 0
        if args.command == "models" and args.models_command == "activate":
            activated = ledger.activate_embedding_profile(args.profile, args.mode)
            emit({"model": describe_embedding_profile(activated), **list_embedding_model_catalog(ledger)})
            return 0

        if args.command == "vectors" and args.vectors_command == "list":
            emit({"indexes": ledger.list_vector_indexes()})
            return 0
        if args.command == "vectors" and args.vectors_command == "verify":
            result = verify_vector_index(ledger, args.profile)
            emit(result.to_dict())
            return 0 if result.ok else 1

        if args.command == "jobs":
            if args.jobs_command == "list":
                emit({"jobs": ledger.list_jobs(kind=args.kind, status=args.status, limit=args.limit)})
                return 0
            if args.jobs_command == "show":
                job = ledger.get_job(args.job_id, include_events=True)
                emit({"job": job})
                return 0 if job is not None else 1

        if args.command == "ingest":
            if args.ingest_command == "start":
                try:
                    kwargs: dict[str, Any] = dict(
                        mode=args.mode,
                        zotero_key=args.zotero_key,
                        include_multimodal=not args.text_only,
                        execute=args.execute,
                    )
                    if args.execute:
                        from .providers import build_mineru_provider
                        from .extractors import ExtractorKeyPool, ExtractionManager

                        key_pool = ExtractorKeyPool.from_env_file(args.env)
                        provider = build_mineru_provider(args.env)
                        kwargs["extract_manager"] = ExtractionManager(
                            ledger=ledger,
                            cache_dir=config.paths.extract_cache_dir,
                            provider=provider,
                            key_pool=key_pool,
                        )
                        kwargs["extract_cache_dir"] = config.paths.extract_cache_dir
                        kwargs["normalized_dir"] = config.paths.normalized_dir
                        kwargs["vector_store_dir"] = config.paths.vector_store_dir
                    emit(start_ingest_job(ledger, **kwargs))
                    return 0
                except NotImplementedError as exc:
                    emit({"ok": False, "error": str(exc)})
                    return 1
                except (FileNotFoundError, ValueError, RuntimeError) as exc:
                    emit({"ok": False, "error": str(exc)})
                    return 1
            if args.ingest_command == "pause":
                try:
                    emit(pause_ingest_job(ledger, args.job_id, reason=args.reason))
                    return 0
                except (KeyError, ValueError) as exc:
                    emit({"ok": False, "error": str(exc)})
                    return 1
            if args.ingest_command == "resume":
                try:
                    emit(resume_ingest_job(ledger, args.job_id, reason=args.reason))
                    return 0
                except (KeyError, ValueError) as exc:
                    emit({"ok": False, "error": str(exc)})
                    return 1
            if args.ingest_command == "cancel":
                try:
                    emit(cancel_ingest_job(ledger, args.job_id, reason=args.reason))
                    return 0
                except (KeyError, ValueError) as exc:
                    emit({"ok": False, "error": str(exc)})
                    return 1

        if args.command == "review":
            if args.review_command == "list":
                emit(
                    {
                        "rules": ledger.list_review_rules(),
                        "candidates": ledger.list_attachments(classification="needs_review"),
                    }
                )
                return 0
            if args.review_command == "include":
                ledger.upsert_review_rule(args.attachment_key, "include", args.reason)
                emit({"attachment_key": args.attachment_key, "decision": "include"})
                return 0
            if args.review_command == "exclude":
                ledger.upsert_review_rule(args.attachment_key, "exclude", args.reason)
                emit({"attachment_key": args.attachment_key, "decision": "exclude"})
                return 0
            if args.review_command == "explain":
                try:
                    emit(explain_attachment_review(ledger, args.attachment_key))
                    return 0
                except KeyError as exc:
                    emit({"ok": False, "error": str(exc)})
                    return 1

        if args.command == "attachments":
            emit(
                {
                    "attachments": ledger.list_attachments(
                        classification=args.classification,
                        limit=args.limit,
                    )
                }
            )
            return 0

        if args.command == "documents":
            if args.documents_command == "list":
                emit(
                    {
                        "documents": list_documents(
                            ledger,
                            limit=args.limit,
                            include_metadata_only=not args.normalized_only,
                        )
                    }
                )
                return 0
            if args.documents_command == "show":
                document = get_document(
                    ledger,
                    args.document_id,
                    include_chunks=args.chunks,
                    chunk_type=args.chunk_type,
                    limit=args.limit,
                    consumer=args.consumer,
                )
                emit({"document": document})
                return 0 if document is not None else 1

        if args.command == "search-metadata":
            try:
                results = metadata_search(
                    ledger,
                    query=args.query,
                    classification=args.classification,
                    limit=args.limit,
                    consumer=args.consumer,
                    rerank=args.rerank,
                )
            except NotImplementedError as exc:
                emit({"ok": False, "error": str(exc)})
                return 1
            emit({"results": results})
            return 0

        if args.command == "search-fulltext":
            try:
                results = fulltext_search(
                    ledger,
                    query=args.query,
                    chunk_type=args.chunk_type,
                    limit=args.limit,
                    consumer=args.consumer,
                    image_return=args.image_return,
                    rerank=args.rerank,
                )
            except NotImplementedError as exc:
                emit({"ok": False, "error": str(exc)})
                return 1
            emit({"results": results})
            return 0

        if args.command == "backup":
            if args.backup_command == "create":
                emit(
                    create_backup(
                        config,
                        ledger,
                        mode=args.mode,
                        out_dir=args.out,
                        config_path=args.config,
                    ).to_dict()
                )
                return 0
            if args.backup_command == "list":
                emit({"backups": ledger.list_backups()})
                return 0
            if args.backup_command == "verify":
                errors = verify_manifest_files(args.manifest)
                emit({"ok": not errors, "errors": errors})
                return 0 if not errors else 1
            if args.backup_command == "restore":
                manifest_path = resolve_backup_manifest(ledger, args.backup)
                if not args.confirm:
                    emit(plan_restore_backup(config, manifest_path).to_dict())
                    return 0
                emit(
                    restore_backup(
                        config,
                        ledger,
                        manifest_path=manifest_path,
                        pre_restore_out_dir=args.pre_restore_out,
                        config_path=args.config,
                        confirm=True,
                        close_ledger_before_apply=True,
                    ).to_dict()
                )
                return 0

        if args.command == "extract":
            if args.extract_command == "dry-run":
                options = json.loads(args.options_json)
                key_pool = ExtractorKeyPool.from_env_file(args.env)
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=config.paths.extract_cache_dir,
                    provider=StubExtractorProvider(),
                    key_pool=key_pool,
                )
                result = manager.ensure_extraction(
                    ExtractionRequest(
                        input_file=Path(args.pdf),
                        attachment_key=args.attachment_key,
                        pdf_sha256=args.sha256,
                        selected_pages=args.pages,
                        selected_page_count=args.selected_page_count,
                        options=options,
                    )
                )
                emit(
                    {
                        "cache_hit": result.cache_hit,
                        "job": result.job,
                        "available_key_aliases": key_pool.list_public_keys(),
                    }
                )
                return 0
            if args.extract_command == "jobs":
                emit({"jobs": ledger.list_extract_jobs(state=args.state, limit=args.limit)})
                return 0
            if args.extract_command == "recovery-plan":
                emit(build_extract_recovery_plan(ledger.list_extract_jobs(state=args.state, limit=args.limit)))
                return 0

        if args.command == "normalize":
            if args.normalize_command == "markdown":
                result = normalize_markdown_document(
                    source_markdown=args.markdown,
                    output_root=config.paths.normalized_dir,
                    document_id=args.document_id,
                    attachment_key=args.attachment_key,
                    extract_job_id=args.extract_job_id,
                )
                artifact = result.ledger_artifact()
                ledger.upsert_normalized_artifact(artifact)
                ledger.replace_document_chunks(result.document_id, result.chunks)
                emit(
                    {
                        "artifact": ledger.get_normalized_artifact(result.document_id),
                        "chunk_count": len(result.chunks),
                        "image_count": len(result.images),
                    }
                )
                return 0
            if args.normalize_command == "list":
                emit({"artifacts": ledger.list_normalized_artifacts()})
                return 0
            if args.normalize_command == "chunks":
                emit(
                    {
                        "chunks": ledger.list_chunks(
                            args.document_id,
                            chunk_type=args.chunk_type,
                            limit=args.limit,
                        )
                    }
                )
                return 0

        if args.command == "embed":
            if args.embed_command == "index-normalized":
                try:
                    provider = None
                    if args.embedding_provider == "qwen3vl":
                        profile = selected_embedding_profile(ledger, args.profile, mode=None)
                        provider = build_qwen_embedding_provider(profile, args.env)
                    result = index_normalized_document(
                        ledger=ledger,
                        vector_store_dir=config.paths.vector_store_dir,
                        profile_name=args.profile,
                        document_id=args.document_id,
                        provider=provider,
                        allow_stub_provider=args.allow_stub_provider,
                    )
                except (KeyError, NotImplementedError, RuntimeError, ValueError) as exc:
                    emit({"ok": False, "error": str(exc)})
                    return 1
                emit(result.to_dict())
                return 0
            if args.embed_command == "batches":
                emit(
                    {
                        "batches": ledger.list_embedding_batches(
                            profile_name=args.profile,
                            document_id=args.document_id,
                            status=args.status,
                            limit=args.limit,
                        )
                    }
                )
                return 0

        if args.command == "reembed":
            if not args.from_normalized:
                emit({"ok": False, "error": "only --from-normalized is supported"})
                return 1
            try:
                if args.plan_only:
                    if args.execute:
                        emit({"ok": False, "error": "--plan-only cannot be combined with --execute"})
                        return 1
                    # Read-only planning must not create jobs, checkpoints, or
                    # provider calls; it only inspects normalized/vector state.
                    emit(
                        create_reembed_plan(
                            ledger,
                            profile_name=args.profile,
                            vector_store_dir=config.paths.vector_store_dir,
                            document_id=args.document_id,
                            force=args.force,
                        )
                    )
                    return 0
                emit(
                    start_reembed_job(
                        ledger,
                        vector_store_dir=config.paths.vector_store_dir,
                        profile_name=args.profile,
                        document_id=args.document_id,
                        force=args.force,
                        execute=args.execute,
                        allow_stub_provider=args.allow_stub_provider,
                    )
                )
                return 0
            except (KeyError, NotImplementedError, ValueError) as exc:
                emit({"ok": False, "error": str(exc)})
                return 1

        if args.command == "search-vector":
            query_image = None
            if args.query_image_file or args.query_image_base64:
                if args.query_image_file and args.query_image_base64:
                    emit({"ok": False, "error": "use only one of --query-image-file or --query-image-base64"})
                    return 1
                try:
                    query_image = normalize_query_image(
                        {
                            "type": "file_path" if args.query_image_file else "base64",
                            "value": args.query_image_file or args.query_image_base64,
                            "mime_type": args.query_image_mime_type,
                        }
                    )
                except (FileNotFoundError, ValueError) as exc:
                    emit({"ok": False, "error": str(exc)})
                    return 1
            try:
                provider = None
                if args.embedding_provider == "qwen3vl":
                    profile = selected_embedding_profile(ledger, args.profile, mode=args.mode)
                    provider = build_qwen_embedding_provider(profile, args.env)
                elif args.embedding_provider == "stub":
                    from .embeddings import StubEmbeddingProvider
                    profile = selected_embedding_profile(ledger, args.profile, mode=args.mode)
                    provider = StubEmbeddingProvider(dimension=int(profile["dimension"]))
                results = search_vector_index(
                    ledger=ledger,
                    vector_store_dir=config.paths.vector_store_dir,
                    profile_name=args.profile,
                    query=args.query,
                    mode=args.mode,
                    top_k=args.top_k,
                    consumer=args.consumer,
                    image_return=args.image_return,
                    max_images=args.max_images,
                    max_image_bytes=args.max_image_bytes,
                    rerank=args.rerank,
                    query_image_path=query_image.file_path if query_image else None,
                    query_image_base64=query_image.base64_data if query_image else None,
                    query_image_mime_type=query_image.mime_type if query_image else None,
                    provider=provider,
                )
            except (FileNotFoundError, KeyError, NotImplementedError, RuntimeError, ValueError) as exc:
                emit({"ok": False, "error": str(exc)})
                return 1
            emit({"results": results})
            return 0

        if args.command == "inspect-shadow":
            shadow_path = Path(config.paths.shadow_db)
            shadow = ZoteroShadow(shadow_path)
            try:
                emit(
                    {
                        "pdf_count": shadow.pdf_count(),
                        "attachments": [item.__dict__ for item in shadow.list_attachments(args.limit)],
                    }
                )
            finally:
                shadow.close()
            return 0

    finally:
        ledger.close()

    print(f"Unhandled command: {args.command}", file=sys.stderr)
    return 2


def selected_embedding_profile(ledger, profile_name: str | None, *, mode: str | None) -> dict:
    profiles = ledger.list_embedding_profiles()
    if profile_name:
        for profile in profiles:
            if profile["name"] == profile_name:
                return profile
        raise KeyError(f"embedding profile not found: {profile_name}")
    if mode is None:
        raise ValueError("profile is required")
    expected_modality = "text" if mode == "text" else "multimodal"
    default_flag = "default_for_text" if mode == "text" else "default_for_multimodal"
    for profile in profiles:
        if profile["enabled"] and profile["modality"] == expected_modality and profile[default_flag]:
            return profile
    raise KeyError(f"no default {mode} embedding profile is configured")


if __name__ == "__main__":
    raise SystemExit(main())
