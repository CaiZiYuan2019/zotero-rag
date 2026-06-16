"""Tests for the MCP stdio server in ``zoterorag.mcp.server``."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

PYTHON = sys.executable


def _run_mcp_session(messages: list[dict[str, Any]], timeout: float = 10.0) -> list[dict[str, Any]]:
    """Start the MCP server, send ``messages`` as NDJSON, and return parsed responses."""
    proc = subprocess.Popen(
        [PYTHON, "-m", "zoterorag.mcp"],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _reader(stream, buf):
        try:
            for line in stream:
                buf.append(line)
        except Exception:
            pass

    threading.Thread(target=_reader, args=(proc.stdout, stdout_lines), daemon=True).start()
    threading.Thread(target=_reader, args=(proc.stderr, stderr_lines), daemon=True).start()

    try:
        for msg in messages:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            time.sleep(0.2)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(stdout_lines) < len(messages):
            time.sleep(0.1)

        if stderr_lines:
            raise AssertionError("MCP server wrote to stderr: " + "".join(stderr_lines))

        return [json.loads(line) for line in stdout_lines]
    finally:
        proc.kill()


def test_mcp_initialize() -> None:
    responses = _run_mcp_session([
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        },
    ])
    assert len(responses) == 1
    assert responses[0]["id"] == 1
    assert responses[0]["result"]["serverInfo"]["name"] == "zoterorag"


def test_mcp_list_tools() -> None:
    responses = _run_mcp_session([
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ])
    assert len(responses) == 2
    tool_names = {t["name"] for t in responses[1]["result"]["tools"]}
    assert tool_names == {
        "zotero_rag_status",
        "zotero_rag_list_models",
        "zotero_rag_metadata_search",
        "zotero_rag_search_text",
        "zotero_rag_search_multimodal",
        "zotero_rag_get_document",
    }


def test_mcp_call_status_tool() -> None:
    responses = _run_mcp_session([
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "zotero_rag_status", "arguments": {}},
        },
    ])
    assert len(responses) == 2
    content = responses[1]["result"]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    payload = json.loads(content[0]["text"])
    assert "runtime" in payload
    assert "state" in payload
    assert "progress" in payload


@pytest.mark.skipif(sys.platform == "win32", reason="stdio subprocess timing sensitive on Windows CI")
def test_mcp_call_unknown_tool_returns_error() -> None:
    responses = _run_mcp_session([
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "zotero_rag_nonexistent", "arguments": {}},
        },
    ])
    assert len(responses) == 2
    content = responses[1]["result"]["content"]
    payload = json.loads(content[0]["text"])
    assert "error" in payload
