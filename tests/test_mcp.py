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


# --- portable server paths --- #

def test_keeper_root_expands_in_server_args(monkeypatch):
    """mcp.json used to hardcode /Users/<name>/rusty-companion, so the same config
    could not run in the container, where the app lives at /app. ${KEEPER_ROOT}
    keeps one config working in both."""
    monkeypatch.setattr(tools, "KEEPER_ROOT", "/app")
    out = tools._expand(["-y", "pkg", "${KEEPER_ROOT}/keeper_sandbox"])
    assert out == ["-y", "pkg", "/app/keeper_sandbox"]


def test_absolute_args_are_left_alone(monkeypatch):
    monkeypatch.setattr(tools, "KEEPER_ROOT", "/app")
    assert tools._expand(["/etc/hosts", 3]) == ["/etc/hosts", 3]


def test_shipped_config_uses_no_developer_path():
    """Guard against an absolute home path creeping back into the tracked config."""
    from pathlib import Path
    for name in ("mcp.json", "mcp.example.json"):
        p = Path(tools.__file__).resolve().parent / name
        if p.exists():
            assert "/Users/" not in p.read_text(), f"{name} hardcodes a home path"


# --- declared dependencies: a server must provide what it promised --- #

@requires_node
async def test_a_server_that_delivers_its_required_tools_is_healthy(sandbox):
    mgr = tools.MCPManager([_spec(sandbox, required=["list_directory"])])
    await mgr.connect()
    try:
        assert mgr.problems() == [], mgr.summary()
        assert mgr.health[0]["status"] == "ok"
    finally:
        await mgr.aclose()


@requires_node
async def test_a_vanished_tool_is_reported_not_silently_absent(sandbox):
    """Regression in kind: a tool renamed upstream, or a server that came up
    half-broken, used to leave no trace anywhere. The Keeper would simply never
    reach for it, and the gap read as reticence rather than as a fault."""
    mgr = tools.MCPManager([_spec(sandbox, required=["read_the_persons_mind"])])
    await mgr.connect()
    try:
        assert mgr.health[0]["status"] == "degraded"
        assert "read_the_persons_mind" in mgr.summary()
        assert mgr.has_tools          # the rest of the server still works
    finally:
        await mgr.aclose()


@requires_node
async def test_a_tool_our_own_filter_blocks_is_named_as_such(sandbox):
    """The confusing case: the tool exists, and OUR read-only default is what
    removed it. Saying only 'missing' would send you hunting the wrong server."""
    mgr = tools.MCPManager([_spec(sandbox, read_only=True,
                                  required=["write_file"])])
    await mgr.connect()
    try:
        assert mgr.health[0]["status"] == "degraded"
        assert "write_file" in mgr.summary()
        assert "blocked by this config" in mgr.summary()
        assert mgr.health[0]["filtered"] == ["write_file"]
    finally:
        await mgr.aclose()


@requires_node
async def test_strict_mode_refuses_a_degraded_server(sandbox, monkeypatch):
    monkeypatch.setattr(tools, "STRICT", True)
    mgr = tools.MCPManager([_spec(sandbox, required=["no_such_tool"])])
    with pytest.raises(RuntimeError, match="no_such_tool"):
        await mgr.connect()
    await mgr.aclose()
