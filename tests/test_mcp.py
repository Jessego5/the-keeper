"""Tier 2 — MCP tool integration (tools.py) against the reference filesystem server.

Covers discovery, reading, the sandbox security boundary, the read-only default,
and graceful errors. Skips entirely without node/npx + the mcp SDK.
"""
import shutil
import pytest
import tools

pytestmark = pytest.mark.integration

_HAVE = shutil.which("npx") is not None
try:
    import mcp  # noqa: F401
except ImportError:
    _HAVE = False
requires_node = pytest.mark.skipif(not _HAVE, reason="needs node/npx + mcp SDK")


def _spec(path, **extra):
    return {"name": "files", "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", str(path)],
            **extra}


def _read_tool(mgr):
    return next(t["function"]["name"] for t in mgr.openai_tools()
               if "read" in t["function"]["name"] and "file" in t["function"]["name"])


@requires_node
async def test_connect_discovers_tools(sandbox):
    mgr = tools.MCPManager([_spec(sandbox)])
    await mgr.connect()
    try:
        assert mgr.has_tools
        assert any("read" in t["function"]["name"] for t in mgr.openai_tools())
    finally:
        await mgr.aclose()


@requires_node
async def test_reads_a_sandbox_file(sandbox):
    mgr = tools.MCPManager([_spec(sandbox)])
    await mgr.connect()
    try:
        out = await mgr.call(_read_tool(mgr), {"path": "list.txt"})
        assert "gesso" in out and "Sam" in out
    finally:
        await mgr.aclose()


@requires_node
@pytest.mark.parametrize("escape", ["/etc/passwd", "../../../../etc/passwd"])
async def test_sandbox_escape_refused(sandbox, escape):
    mgr = tools.MCPManager([_spec(sandbox)])
    await mgr.connect()
    try:
        out = (await mgr.call(_read_tool(mgr), {"path": escape})).lower()
        assert any(w in out for w in ("denied", "not allow", "outside", "error"))
        assert "root:" not in out          # never actually leaked /etc/passwd
    finally:
        await mgr.aclose()


@requires_node
async def test_read_only_blocks_mutating_tools(sandbox):
    mgr = tools.MCPManager([_spec(sandbox)])            # read_only defaults True
    await mgr.connect()
    try:
        names = [t["function"]["name"] for t in mgr.openai_tools()]
        assert names and not any(tools._is_mutating(n) for n in names)
    finally:
        await mgr.aclose()


@requires_node
async def test_read_only_false_exposes_writes(sandbox):
    mgr = tools.MCPManager([_spec(sandbox, read_only=False)])
    await mgr.connect()
    try:
        names = [t["function"]["name"] for t in mgr.openai_tools()]
        assert any(tools._is_mutating(n) for n in names)
    finally:
        await mgr.aclose()


@requires_node
async def test_unknown_tool_is_graceful(sandbox):
    mgr = tools.MCPManager([_spec(sandbox)])
    await mgr.connect()
    try:
        out = await mgr.call("files__does_not_exist", {})
        assert "no such tool" in out.lower()
    finally:
        await mgr.aclose()


def test_missing_config_yields_no_tools(tmp_path):
    # No mcp.json -> zero tools, Keeper behaves as before (no crash).
    mgr = tools.MCPManager.from_config(tmp_path / "absent.json")
    assert mgr.openai_tools() == [] and not mgr.has_tools
