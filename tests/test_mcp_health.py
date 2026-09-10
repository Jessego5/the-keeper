"""
These are the Tier 1 tests for MCP deadlines and health reporting (tools.py). No node,
no network.

Why these exist: an MCP server that hung, died, or quietly stopped offering a tool
used to leave exactly one line on stdout. Nothing reached the trace, /state, or the
dashboard, so a missing capability mid-demo looked like the Keeper choosing not to
reach for it. These pin the two halves of the fix: every call has a deadline, and
every server's condition is recorded rather than printed.
"""
import asyncio

import pytest
import tools

pytestmark = pytest.mark.unit


class _Hanging:
    """A server that accepts the call and never answers."""
    async def call_tool(self, name, arguments):
        await asyncio.sleep(30)


class _Fine:
    async def call_tool(self, name, arguments):
        class R:
            content = [type("B", (), {"text": "ok"})()]
        return R()


@pytest.fixture
def mgr():
    return tools.MCPManager([])


# --- deadlines --- #

async def test_a_hanging_tool_call_gives_up(mgr, monkeypatch):
    monkeypatch.setattr(tools, "CALL_TIMEOUT_S", 0.05)
    mgr._route["srv__slow"] = (_Hanging(), "slow")
    out = await mgr.call("srv__slow", {})
    assert "timed out" in out and "srv__slow" in out


async def test_the_timeout_does_not_punish_a_healthy_call(mgr, monkeypatch):
    monkeypatch.setattr(tools, "CALL_TIMEOUT_S", 5.0)
    mgr._route["srv__fast"] = (_Fine(), "fast")
    assert await mgr.call("srv__fast", {}) == "ok"


async def test_a_hanging_call_returns_rather_than_raising(mgr, monkeypatch):
    """The model has to be answered. A raise here would abort the person's turn."""
    monkeypatch.setattr(tools, "CALL_TIMEOUT_S", 0.05)
    mgr._route["srv__slow"] = (_Hanging(), "slow")
    out = await mgr.call("srv__slow", {})
    assert isinstance(out, str)


def test_timeouts_are_configurable_from_the_environment(monkeypatch):
    monkeypatch.setenv("KEEPER_MCP_TIMEOUT_S", "3.5")
    assert tools._env_float("KEEPER_MCP_TIMEOUT_S", 20.0) == 3.5


def test_a_junk_timeout_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("KEEPER_MCP_TIMEOUT_S", "soon")
    assert tools._env_float("KEEPER_MCP_TIMEOUT_S", 20.0) == 20.0


# --- health reporting --- #

def test_a_healthy_manager_reports_no_problems(mgr):
    mgr._record("files", "ok", tools=4, blocked=0, missing=[])
    assert mgr.problems() == [] and mgr.summary() == ""


def test_a_failed_server_is_recorded_with_its_reason(mgr):
    mgr._record("git", "failed", error="No such file or directory: uvx")
    assert [h["name"] for h in mgr.problems()] == ["git"]
    assert "git: failed" in mgr.summary() and "uvx" in mgr.summary()


def test_a_server_missing_a_required_tool_is_degraded(mgr):
    mgr._record("git", "degraded", tools=3, blocked=0, missing=["git_log"])
    assert mgr.problems()[0]["status"] == "degraded"
    assert "git: missing git_log" in mgr.summary()


def test_summary_names_every_unhealthy_server(mgr):
    mgr._record("files", "ok", tools=4, missing=[])
    mgr._record("git", "degraded", missing=["git_log"])
    mgr._record("fetch", "failed", error="boom")
    assert "files" not in mgr.summary()
    assert "git" in mgr.summary() and "fetch" in mgr.summary()


def test_strict_mode_refuses_to_start_degraded(mgr, monkeypatch):
    monkeypatch.setattr(tools, "STRICT", True)
    mgr._record("git", "degraded", missing=["git_log"])
    with pytest.raises(RuntimeError, match="git_log"):
        mgr._enforce()


def test_strict_mode_is_quiet_when_everything_is_healthy(mgr, monkeypatch):
    monkeypatch.setattr(tools, "STRICT", True)
    mgr._record("files", "ok", tools=4, missing=[])
    mgr._enforce()          # must not raise


async def test_a_missing_sdk_records_every_server(monkeypatch):
    """The Keeper still runs without the SDK, but 'no tools' must be legible as a
    fault rather than as a configuration the person chose."""
    import builtins
    real_import = builtins.__import__

    def no_mcp(name, *a, **kw):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("no mcp")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_mcp)
    mgr = tools.MCPManager([{"name": "files", "command": "npx"}])
    await mgr.connect()
    assert mgr.problems()[0]["name"] == "files"
    assert "SDK" in mgr.summary()


# --- a watched tool's arguments travel too --- #

def test_watch_args_expand_the_keeper_root(monkeypatch, tmp_path):
    """A watch usually points at a path (git_log wants its repo). Left literal,
    it would break the moment the app moved into the container."""
    monkeypatch.setattr(tools, "KEEPER_ROOT", "/app")
    cfg = tmp_path / "mcp.json"
    cfg.write_text('{"servers": [], "watch": [{"name": "repo",'
                   ' "tool": "git__git_log",'
                   ' "args": {"repo_path": "${KEEPER_ROOT}", "max_count": 5}}]}')
    mgr = tools.MCPManager.from_config(cfg)
    assert mgr.watches[0]["args"] == {"repo_path": "/app", "max_count": 5}


def test_a_config_with_no_watch_block_watches_nothing(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text('{"servers": []}')
    assert tools.MCPManager.from_config(cfg).watches == []


def test_an_unreadable_config_watches_nothing(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{not json")
    mgr = tools.MCPManager.from_config(cfg)
    assert mgr.watches == [] and mgr.specs == []
