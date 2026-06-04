from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .backup import create_backup, verify_manifest_files
from .extractors import ExtractionManager, ExtractionRequest, ExtractorKeyPool, StubExtractorProvider
from .runtime import config_as_public_dict, copy_zotero_shadow, initialize_runtime, scan_zotero_shadow
from .search import metadata_search
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
    sub.add_parser("shadow-copy", help="Create a read-only Zotero shadow copy.")
    scan = sub.add_parser("scan", help="Copy Zotero shadow and classify attachments into state.")
    scan.add_argument("--no-refresh-shadow", action="store_true", help="Scan the existing shadow DB without recopying Zotero.")
    scan.add_argument("--limit", type=int, default=None)

    models = sub.add_parser("models", help="List configured embedding profiles.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list")

    vectors = sub.add_parser("vectors", help="Inspect local vector index registrations.")
    vectors_sub = vectors.add_subparsers(dest="vectors_command", required=True)
    vectors_sub.add_parser("list")

    review = sub.add_parser("review", help="Manage manual include/exclude rules.")
    review_sub = review.add_subparsers(dest="review_command", required=True)
    review_sub.add_parser("list")
    include = review_sub.add_parser("include")
    include.add_argument("--attachment-key", required=True)
    include.add_argument("--reason", default="manual include")
    exclude = review_sub.add_parser("exclude")
    exclude.add_argument("--attachment-key", required=True)
    exclude.add_argument("--reason", default="manual exclude")

    attachments = sub.add_parser("attachments", help="List persisted attachment scan results.")
    attachments.add_argument("--classification", default=None)
    attachments.add_argument("--limit", type=int, default=50)

    search_metadata = sub.add_parser("search-metadata", help="Search scanned Zotero metadata without embeddings.")
    search_metadata.add_argument("query")
    search_metadata.add_argument("--classification", default=None)
    search_metadata.add_argument("--limit", type=int, default=10)
    search_metadata.add_argument("--consumer", default="llm_text", choices=("manual", "llm_text", "llm_multimodal"))

    backup = sub.add_parser("backup", help="Create and inspect ZoteroRAG runtime backups.")
    backup_sub = backup.add_subparsers(dest="backup_command", required=True)
    backup_create = backup_sub.add_parser("create")
    backup_create.add_argument("--mode", choices=("snapshot", "full"), default="snapshot")
    backup_create.add_argument("--out", required=True)
    backup_sub.add_parser("list")
    backup_verify = backup_sub.add_parser("verify")
    backup_verify.add_argument("manifest")

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
            emit({"state": ledger.status_summary(), "runtime": config_as_public_dict(config)})
            return 0

        if args.command == "shadow-copy":
            emit(copy_zotero_shadow(config, ledger))
            return 0

        if args.command == "scan":
            emit(
                scan_zotero_shadow(
                    config,
                    ledger,
                    refresh_shadow=not args.no_refresh_shadow,
                    limit=args.limit,
                )
            )
            return 0

        if args.command == "models" and args.models_command == "list":
            emit({"models": ledger.list_embedding_profiles()})
            return 0

        if args.command == "vectors" and args.vectors_command == "list":
            emit({"indexes": ledger.list_vector_indexes()})
            return 0

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

        if args.command == "search-metadata":
            emit(
                {
                    "results": metadata_search(
                        ledger,
                        query=args.query,
                        classification=args.classification,
                        limit=args.limit,
                        consumer=args.consumer,
                    )
                }
            )
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


if __name__ == "__main__":
    raise SystemExit(main())
