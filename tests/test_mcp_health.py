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


# --- tools behind approval --- #

class _Failing:
    """A server that answers, but reports the call failed in the result."""
    async def call_tool(self, name, arguments):
        class R:
            content = [type("B", (), {"text": "401 unauthorized"})()]
            isError = True
        return R()


async def test_a_tool_behind_approval_cannot_be_reached_as_an_ordinary_one(mgr):
    """The separation proposals.py depends on: nothing that walks the ordinary
    tools (the chat path, a sub-agent's view) can run one that needs a yes."""
    mgr._proposable_route["mail__send"] = (_Fine(), "send")
    mgr._proposable.append({"name": "mail__send", "description": "",
                            "parameters": {}})
    assert mgr.openai_tools() == []
    assert "no such tool" in await mgr.call("mail__send", {})
    assert await mgr.call_proposable("mail__send", {}) == "ok"


async def test_an_approved_call_that_failed_says_so(mgr):
    """A model reading "401 unauthorized" understands it. A proposal reading it
    would have told the person their email went."""
    mgr._proposable_route["mail__send"] = (_Failing(), "send")
    assert (await mgr.call_proposable("mail__send", {})).startswith("(tool error")


async def test_an_ordinary_call_reads_as_it_always_did(mgr):
    mgr._route["mail__read"] = (_Failing(), "read")
    assert await mgr.call("mail__read", {}) == "401 unauthorized"


# --- a server that can act on someone's behalf (Google Workspace) --- #

@pytest.mark.parametrize("name", [
    "send_gmail_message", "draft_gmail_message", "manage_event",
    "manage_gmail_filter", "manage_out_of_office", "batch_modify_gmail_message_labels",
    "create_calendar",
])
def test_the_workspace_servers_verbs_count_as_mutating(name):
    """Regression, found by listing the real server's tools before wiring it in:
    send, draft and manage matched none of the filesystem-era verbs, so "send an
    email" would have passed the read-only filter as a read."""
    assert tools._is_mutating(name)


@pytest.mark.parametrize("name", [
    "search_gmail_messages", "get_gmail_message_content", "get_events",
    "list_calendars", "query_freebusy", "read_text_file", "list_directory",
    "get_current_time", "git_log", "fetch",
])
def test_reading_tools_still_read(name):
    assert not tools._is_mutating(name)


def test_env_references_expand_and_report_what_is_missing(monkeypatch):
    monkeypatch.setenv("KEEPER_GOOGLE_EMAIL", "me@example.com")
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    missing = set()
    out = tools._expand_env({"a": "${KEEPER_GOOGLE_EMAIL}",
                             "b": ["x-${NOT_SET_ANYWHERE}"]}, missing)
    assert out == {"a": "me@example.com", "b": ["x-"]}
    assert missing == {"NOT_SET_ANYWHERE"}


def test_a_fixed_argument_is_hidden_from_the_model():
    schema = {"type": "object",
              "properties": {"user_google_email": {}, "to": {}},
              "required": ["user_google_email", "to"]}
    out = tools._without(schema, {"user_google_email": "me@example.com"})
    assert list(out["properties"]) == ["to"] and out["required"] == ["to"]


class _Echo:
    """A server that answers with the arguments it was actually called with."""
    async def call_tool(self, name, arguments):
        import json as _json

        class R:
            content = [type("B", (), {"text": _json.dumps(arguments,
                                                          sort_keys=True)})()]
        return R()


async def test_the_config_chooses_the_account_not_the_model(mgr):
    """Which mailbox to act as is configuration. A model told by a fetched page
    to "use admin@..." must not be able to switch it."""
    mgr._route["google__search"] = (_Echo(), "search")
    mgr._fixed["google__search"] = {"user_google_email": "me@example.com"}
    out = await mgr.call("google__search", {"query": "x",
                                            "user_google_email": "admin@else.com"})
    assert '"user_google_email": "me@example.com"' in out


async def test_an_opt_in_server_is_off_not_broken(monkeypatch):
    """Copying mcp.example.json whole includes the google server. Without its
    credentials it is switched off and says how to turn it on; it must not read
    as a fault on the dashboard, or launch at all."""
    monkeypatch.delenv("KEEPER_GOOGLE_EMAIL", raising=False)
    mgr = tools.MCPManager([{"name": "google", "command": "definitely-not-run",
                             "requires_env": ["KEEPER_GOOGLE_EMAIL"]}])
    await mgr.connect()
    assert mgr.health == [{"name": "google", "status": "off",
                           "needs": ["KEEPER_GOOGLE_EMAIL"]}]
    assert mgr.problems() == [] and mgr.summary() == ""
