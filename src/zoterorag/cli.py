from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .runtime import config_as_public_dict, copy_zotero_shadow, initialize_runtime
from .zotero import ZoteroShadow


def emit(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zoterorag")
    parser.add_argument("--config", default="config/config.example.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-state", help="Create runtime directories and initialize state.sqlite.")
    sub.add_parser("status", help="Show runtime and state status.")
    sub.add_parser("shadow-copy", help="Create a read-only Zotero shadow copy.")

    models = sub.add_parser("models", help="List configured embedding profiles.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list")

    review = sub.add_parser("review", help="Manage manual include/exclude rules.")
    review_sub = review.add_subparsers(dest="review_command", required=True)
    review_sub.add_parser("list")
    include = review_sub.add_parser("include")
    include.add_argument("--attachment-key", required=True)
    include.add_argument("--reason", default="manual include")
    exclude = review_sub.add_parser("exclude")
    exclude.add_argument("--attachment-key", required=True)
    exclude.add_argument("--reason", default="manual exclude")

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

        if args.command == "models" and args.models_command == "list":
            emit({"models": ledger.list_embedding_profiles()})
            return 0

        if args.command == "review":
            if args.review_command == "list":
                emit({"rules": ledger.list_review_rules()})
                return 0
            if args.review_command == "include":
                ledger.upsert_review_rule(args.attachment_key, "include", args.reason)
                emit({"attachment_key": args.attachment_key, "decision": "include"})
                return 0
            if args.review_command == "exclude":
                ledger.upsert_review_rule(args.attachment_key, "exclude", args.reason)
                emit({"attachment_key": args.attachment_key, "decision": "exclude"})
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

