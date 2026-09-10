"""
These are the Tier 1 tests for sub-agents (subagents.py): routing, tool filtering,
delegation.
"""
import pytest
import subagents

pytestmark = pytest.mark.unit


# --- routing --- #

def test_route_picks_named_specialist():
    def gen(system, user):
        assert "researcher" in system    # the blurbs are in the router prompt
        return "researcher"
    assert subagents.route("look up watercolor brands", gen).name == "researcher"


def test_route_matches_name_in_prose():
    gen = lambda s, u: "I think the archivist is best here."
    assert subagents.route("find my old journal entry", gen).name == "archivist"


def test_route_defaults_when_unclear():
    gen = lambda s, u: "hmm, not sure"
    assert subagents.route("???", gen).name == subagents.DEFAULT_PROFILE


def test_route_survives_a_broken_model():
    def boom(s, u):
        raise RuntimeError("down")
    assert subagents.route("x", boom).name == subagents.DEFAULT_PROFILE


# --- filtered MCP view --- #

class _FakeMCP:
    def openai_tools(self):
        return [
            {"function": {"name": "search__search"}},
            {"function": {"name": "files__read_text_file"}},
            {"function": {"name": "git__git_log"}},
        ]
    async def call(self, name, args):
        return f"called {name}"


def test_filtered_mcp_exposes_only_allowed_servers():
    view = subagents._FilteredMCP(_FakeMCP(), ("search", "fetch"))
    names = [t["function"]["name"] for t in view.openai_tools()]
    assert names == ["search__search"]        # files/git hidden
    assert view.has_tools


async def test_filtered_mcp_calls_route_through():
    view = subagents._FilteredMCP(_FakeMCP(), ("search",))
    assert await view.call("search__search", {}) == "called search__search"


# --- run + delegate (compose.tool_reply stubbed) --- #

@pytest.fixture
def stub_tool_reply(monkeypatch):
    calls = {}
    async def fake(system, user, *, providers, model, max_rounds, **kw):
        calls["system"] = system
        calls["n_providers"] = len(providers)
        return "the researcher's finding"
    monkeypatch.setattr(subagents.compose, "tool_reply", fake)
    return calls


async def test_run_returns_result(stub_tool_reply):
    prof = subagents.PROFILES["researcher"]
    res = await subagents.run(prof, "find X", mcp=_FakeMCP())
    assert res.profile == "researcher"
    assert res.result == "the researcher's finding"
    assert stub_tool_reply["n_providers"] == 1        # researcher got the mcp view


async def test_delegate_routes_then_runs(stub_tool_reply):
    gen = lambda s, u: "researcher"
    res = await subagents.delegate("look up Y", gen, mcp=_FakeMCP())
    assert res.profile == "researcher" and res.result


# --- orchestrator: decompose -> parallel workers -> synthesize --- #

def test_decompose_splits_and_caps():
    gen = lambda s, u: "sub one\nsub two\nsub three\nsub four\nsub five"
    subs = subagents._decompose("big task", gen)
    assert subs[:2] == ["sub one", "sub two"]
    assert len(subs) == subagents.MAX_WORKERS      # capped


def test_decompose_single_is_one():
    assert subagents._decompose("simple", lambda s, u: "just the one") == ["just the one"]


@pytest.fixture
def stub_worker(monkeypatch):
    """Stub run() so each worker returns a tagged result without a real tool loop."""
    async def fake_run(profile, task, *, mcp=None, native=None, model="gpt-4o"):
        return subagents.SubAgentResult(profile=profile.name, task=task,
                                        result=f"did:{task}")
    monkeypatch.setattr(subagents, "run", fake_run)


async def test_orchestrate_fans_out_and_synthesizes(stub_worker):
    def gen(system, user):
        if "lead agent" in system and "Break a task" in system:      # decompose
            return "research the web part\ncompute the numbers part"
        if "Findings:" in user:                                       # synthesize
            return "SYNTHESIZED: both parts combined"
        return "researcher"                                          # route
    res = await subagents.orchestrate("do a two-part thing", gen)
    assert res.result == "SYNTHESIZED: both parts combined"
    assert len(res.workers) == 2                                     # two ran
    assert "did:research the web part" in [w.result for w in res.workers][0] \
        or any("did:" in w.result for w in res.workers)


async def test_orchestrate_single_subtask_degrades_to_one_worker(stub_worker):
    def gen(system, user):
        if "Break a task" in system:
            return "just one thing"          # single subtask -> no synthesis
        return "analyst"
    res = await subagents.orchestrate("one thing", gen)
    assert len(res.workers) == 1
    assert res.result == "did:just one thing"    # the worker's result, un-synthesized


async def test_orchestrate_who_lists_specialists(stub_worker):
    def gen(system, user):
        if "Break a task" in system:
            return "part a\npart b"
        if "Findings:" in user:
            return "combined"
        return "researcher"                  # both route to researcher
    res = await subagents.orchestrate("t", gen)
    assert res.who() == "researcher"         # deduped
