"""Exercise the registered MCP tools over their normal stdio transport."""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest

mcp = pytest.importorskip("mcp")

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module", params=["module", "cli"])
async def server(request, anyio_backend, tmp_path_factory):
    root = tmp_path_factory.mktemp("mcp-project")
    (root / "pyproject.toml").touch()
    config = root / "config" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        '[paths]\nzotero_db="zotero.sqlite"\nzotero_storage="storage"\n'
        + "".join(
            f'\n[[embedding_profiles]]\nname="{mode}"\nmodality="{mode}"\n'
            'provider="dashscope"\nmodel="qwen3-vl-embedding"\ndimension=256\n'
            f'default_for_{mode}=true\nbackend="sqlite-local"\n'
            for mode in ("text", "multimodal")
        ),
        encoding="utf-8",
    )
    # A present env file without credentials distinguishes key loading from a missing file.
    (root / ".env").touch()
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    if request.param == "module":
        args = ["-m", "zoterorag.mcp"]
        env["ZOTERORAG_CONFIG"] = str(config)
    else:
        args = ["-m", "zoterorag", "--config", str(config), "mcp"]
    params = mcp.StdioServerParameters(
        command=sys.executable,
        args=args,
        cwd=str(tmp_path_factory.mktemp("mcp-client")),
        env=env,
    )
    async with mcp.stdio_client(params) as (reader, writer):
        async with mcp.ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=20)) as client:
            initialized = await client.initialize()
            yield client, initialized


async def test_mcp_initialize(server):
    _, initialized = server
    assert initialized.serverInfo.name == "zoterorag"


async def test_mcp_list_tools(server):
    client, _ = server
    response = await client.list_tools()
    assert {tool.name for tool in response.tools} == {
        "zotero_rag_status",
        "zotero_rag_list_models",
        "zotero_rag_list_collections",
        "zotero_rag_search_collection",
        "zotero_rag_metadata_search",
        "zotero_rag_search_text",
        "zotero_rag_search_multimodal",
        "zotero_rag_get_document",
    }


async def test_mcp_call_status_tool(server):
    client, _ = server
    response = await client.call_tool("zotero_rag_status", {})
    assert not response.isError
    payload = json.loads(response.content[0].text)
    assert "library" in payload
    assert "profiles" in payload
    assert isinstance(payload["build"]["text_indexed"], int)


@pytest.mark.parametrize(("mode", "query_arg"), [("text", "query"), ("multimodal", "query_text")])
async def test_mcp_vector_search_missing_key(server, mode, query_arg):
    client, _ = server
    response = await client.call_tool(f"zotero_rag_search_{mode}", {query_arg: "figure"})
    payload = json.loads(response.content[0].text)
    assert payload["results"] == []
    warning, = payload["warnings"]
    assert warning.startswith(f"{mode}_vector_unavailable:")
    assert "DASHSCOPE_API_KEY" in warning


async def test_mcp_call_unknown_tool_returns_error(server):
    client, _ = server
    response = await client.call_tool("zotero_rag_nonexistent", {})
    message = response.content[0].text
    assert response.isError or "error" in json.loads(message)
    assert "zotero_rag_nonexistent" in message
