"""MCP stdio server exposing ZoteroRAG search/retrieval tools.

Run with:

    python -m zoterorag.mcp

The server reads ``config/config.toml`` (or the file pointed to by
``ZOTERORAG_CONFIG``) and opens the configured state database.  It exposes the
same tool functions defined in ``zoterorag.mcp.tools`` through the MCP protocol.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool

from ..config import AppConfig
from ..db import StateLedger
from ..runtime import initialize_runtime
from .tools import (
    McpToolContext,
    zotero_rag_get_document,
    zotero_rag_list_models,
    zotero_rag_metadata_search,
    zotero_rag_search_multimodal,
    zotero_rag_search_text,
    zotero_rag_status,
)

SERVER_NAME = "zoterorag"
SERVER_VERSION = "0.1.0"

# JSON Schema for each exposed tool.  Keep these in sync with the signatures in
# ``zoterorag.mcp.tools``.
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "zotero_rag_status": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "zotero_rag_list_models": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "zotero_rag_metadata_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query over Zotero metadata."},
            "classification": {
                "type": "string",
                "description": "Optional classification filter.",
            },
            "top_k": {"type": "integer", "default": 10},
            "rerank": {"type": "boolean", "default": False},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "zotero_rag_search_text": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "profile_name": {"type": "string", "description": "Optional vector profile name."},
            "top_k": {"type": "integer", "default": 10},
            "include_metadata": {"type": "boolean", "default": True},
            "include_fulltext": {"type": "boolean", "default": True},
            "include_vector": {"type": "boolean", "default": True},
            "rerank": {"type": "boolean", "default": False},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "zotero_rag_search_multimodal": {
        "type": "object",
        "properties": {
            "query_text": {"type": "string", "description": "Text query."},
            "query_image": {
                "type": "object",
                "description": "Optional query image descriptor with type='file_path' or 'base64'.",
                "properties": {
                    "type": {"type": "string", "enum": ["file_path", "base64"]},
                    "value": {"type": "string"},
                    "mime_type": {"type": "string"},
                },
                "required": ["type", "value"],
                "additionalProperties": False,
            },
            "profile_name": {"type": "string", "description": "Optional vector profile name."},
            "top_k": {"type": "integer", "default": 10},
            "consumer": {
                "type": "string",
                "enum": ["llm_text", "llm_multimodal"],
                "default": "llm_text",
            },
            "image_return": {
                "type": "string",
                "enum": ["file_ref", "base64", "none"],
                "default": "none",
            },
            "max_images": {"type": "integer", "default": 5},
            "max_image_bytes": {"type": "integer", "default": 262144},
            "rerank": {"type": "boolean", "default": False},
        },
        "required": ["query_text"],
        "additionalProperties": False,
    },
    "zotero_rag_get_document": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "include_chunks": {"type": "boolean", "default": True},
            "chunk_type": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
            "consumer": {
                "type": "string",
                "enum": ["llm_text", "llm_multimodal"],
                "default": "llm_text",
            },
        },
        "required": ["document_id"],
        "additionalProperties": False,
    },
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "zotero_rag_status": "Return ZoteroRAG runtime status, state summary, and ingest progress.",
    "zotero_rag_list_models": "List available embedding model profiles.",
    "zotero_rag_metadata_search": "Search Zotero metadata only (no fulltext or vectors).",
    "zotero_rag_search_text": "Search across metadata, fulltext, and text vector indexes. Safe for plain-text LLMs.",
    "zotero_rag_search_multimodal": "Multimodal vector search. Use consumer='llm_multimodal' and image_return='file_ref' or 'base64' to receive images.",
    "zotero_rag_get_document": "Retrieve a single document with optional chunks.",
}

TOOL_FUNCTIONS: dict[str, Any] = {
    "zotero_rag_status": zotero_rag_status,
    "zotero_rag_list_models": zotero_rag_list_models,
    "zotero_rag_metadata_search": zotero_rag_metadata_search,
    "zotero_rag_search_text": zotero_rag_search_text,
    "zotero_rag_search_multimodal": zotero_rag_search_multimodal,
    "zotero_rag_get_document": zotero_rag_get_document,
}


def _load_runtime(config_path: str | None = None) -> tuple[AppConfig, StateLedger]:
    """Load config, ensure runtime dirs, and open the state ledger."""
    if config_path is None:
        config_path = os.environ.get("ZOTERORAG_CONFIG", "config/config.toml")
    return initialize_runtime(config_path)


def _build_mcp_context(config: AppConfig, ledger: StateLedger) -> McpToolContext:
    return McpToolContext(config=config, ledger=ledger)


def _make_tools() -> list[Tool]:
    return [
        Tool(
            name=name,
            description=TOOL_DESCRIPTIONS[name],
            inputSchema=schema,
        )
        for name, schema in TOOL_SCHEMAS.items()
    ]


async def serve(config: AppConfig, ledger: StateLedger) -> None:
    """Run the MCP stdio server."""
    server = Server(SERVER_NAME)
    context = _build_mcp_context(config, ledger)
    tools = _make_tools()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tools

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        func = TOOL_FUNCTIONS.get(name)
        if func is None:
            return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]
        try:
            result = func(context, **arguments)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
        except Exception as exc:  # pragma: no cover - keep server alive
            error_payload = {"error": type(exc).__name__, "message": str(exc)}
            return [TextContent(type="text", text=json.dumps(error_payload, ensure_ascii=False))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(
                notification_options=None,
                experimental_capabilities={},
            ),
        )


def main(config_path: str | None = None) -> None:
    """Entry point for ``python -m zoterorag.mcp``.

    Args:
        config_path: Optional TOML config path. Falls back to ``ZOTERORAG_CONFIG``
            env var, then ``config/config.toml``.
    """
    # Re-encode stdout before any operation so MCP JSON lines are always UTF-8.
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    if config_path is None:
        config_path = os.environ.get("ZOTERORAG_CONFIG")
    try:
        config, ledger = _load_runtime(config_path)
    except Exception as exc:
        print(f"[zoterorag_mcp] FATAL: failed to load runtime: {exc}", file=sys.stderr, flush=True)
        raise
    try:
        asyncio.run(serve(config, ledger))
    except Exception as exc:
        print(f"[zoterorag_mcp] FATAL: server crashed: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
