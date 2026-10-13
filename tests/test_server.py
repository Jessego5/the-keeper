"""
These are the Tier 2 tests for the running app (server.py): chat, state, and the
reach-out path.

Drives the real ASGI app (lifespan + background proactive loop) over httpx, with
the model forced to the offline stub and MCP disabled, so these run with no key,
no cost, and deterministically. The headline test is the reach-out: crank the
battery and assert a proactive line is delivered over SSE: the mechanism behind
the notifications.
"""
import asyncio
import json
import time

from datetime import datetime

import httpx
import pytest

import compose
import server
import tools

pytestmark = pytest.mark.integration


@pytest.fixture
async def client(monkeypatch, tmp_path):
    # No MCP (fast, deterministic) and force the offline stub generator.
    monkeypatch.setattr(tools.MCPManager, "from_config",
                        classmethod(lambda cls, *a, **k: tools.MCPManager([])))
    # Every unbidden line now opens a conversation if none is active and is written
    # to the log, so a run against the real memory_store/ would leave both behind.
    _isolate_delivery(monkeypatch, tmp_path)
    # The single-instance lock lives in memory_store/ too. Held by a Keeper
    # running on this machine, it would otherwise refuse the test its lifespan.
    import instance
    monkeypatch.setattr(instance, "LOCK_PATH", tmp_path / "keeper.lock")
    # backend/.env is loaded by compose.py on import, so without this the lifespan
    # builds the REAL embedder and mood classifier and the loop polls the real
    # KEEPER_FEEDS: these tests were fetching feeds and paying for embeddings on
    # the developer's key, against the developer's own memory, and a slow feed
    # stalled the loop past test_proactive_loop_reaches_out's deadline. Found as
    # a 1-in-4 flake. Offline means offline.
    monkeypatch.delenv("KEEPER_FEEDS", raising=False)
    # The same goes for every outward surface. Once a real bot token was in .env,
    # the lifespan built the real Telegram channel and started polling the real
    # bot: a run of this suite could message the developer's phone, and fought the
    # running Keeper for its updates (found when the suite slowed from 20s to 126s).
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DISCORD_WEBHOOK_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KEEPER_TELEGRAM_INBOX", "0")
    _isolate_stores(monkeypatch, tmp_path)
    async with server.app.router.lifespan_context(server.app):
        server.STATE.generate, server.STATE.fast = compose.stub_generator, None
        # The lifespan rebuilds delivery; keep it to the page alone.
        monkeypatch.setattr(server.STATE, "delivery", server.channels.Delivery(
            [server.channels.SSEChannel(server.STATE.listeners)]))
        # The lifespan builds these fresh; replace them after it has run.
        monkeypatch.setattr(server.STATE, "embed", None)
        monkeypatch.setattr(server.STATE, "llm_mood_signal", None)
        monkeypatch.setattr(server.STATE, "seen_sources",
                            sources.SeenStore(tmp_path / "seen.jsonl"))
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://localhost") as c:
            yield c


async def test_rejects_foreign_host(client):
    # DNS-rebinding defense: a request whose Host isn't localhost/127.0.0.1 (as a
    # malicious website's rebind would send) must be refused, not served.
    r = await client.get("/state", headers={"host": "evil.example.com"})
    assert r.status_code == 400, "foreign Host was served, DNS-rebinding open"


async def test_allows_localhost(client):
    assert (await client.get("/state", headers={"host": "localhost"})).status_code == 200
    assert (await client.get("/state", headers={"host": "127.0.0.1:8737"})).status_code == 200


async def test_chat_returns_a_reply(client):
    r = await client.post("/chat", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json()["reply"]


async def test_trace_records_a_turn(client):
    await client.post("/chat", json={"message": "hello there keeper"})
    r = await client.get("/trace")
    assert r.status_code == 200
    traces = r.json()
    assert traces, "a turn should have been traced"
    t = traces[0]
    assert t["cue"] == "hello there keeper"
    assert "memory_injected" in t and "tools_called" in t and "reply" in t
    assert t["path"] in ("tool", "compose")


async def test_a2a_agent_card_served(client):
    r = await client.get("/.well-known/agent.json")
    assert r.status_code == 200
    card = r.json()
    assert card["name"] == "The Keeper" and card["url"].endswith("/a2a")


async def test_a2a_message_send(client, monkeypatch):
    # stub the model-backed answer so the endpoint test is deterministic
    async def fake_answer(text):
        return f"kept: {text}"
    monkeypatch.setattr(server, "_answer_as_keeper", fake_answer)
    monkeypatch.setenv("KEEPER_A2A_TOKEN", "test-token")
    req = {"jsonrpc": "2.0", "id": "1", "method": "message/send",
           "params": {"message": {"role": "user",
                                  "parts": [{"kind": "text", "text": "who are you?"}]}}}
    r = await client.post("/a2a", json=req,
                          headers={"authorization": "Bearer test-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "1"
    import a2a
    assert a2a.text_of(body["result"]) == "kept: who are you?"


async def test_a2a_rejects_unknown_method(client, monkeypatch):
    monkeypatch.setenv("KEEPER_A2A_TOKEN", "test-token")
    r = await client.post("/a2a", json={"jsonrpc": "2.0", "id": "9",
                                        "method": "tasks/cancel", "params": {}},
                          headers={"authorization": "Bearer test-token"})
    assert r.json()["error"]["code"] == -32601


async def test_a2a_is_closed_without_a_token(client, monkeypatch):
    """Closed by default, because answering means recalling memory and this repo is
    about to be public. The refusal names the variable rather than just failing."""
    monkeypatch.delenv("KEEPER_A2A_TOKEN", raising=False)
    r = await client.post("/a2a", json={"jsonrpc": "2.0", "id": "1",
                                        "method": "message/send", "params": {}})
    err = r.json()["error"]
    assert err["code"] == -32001 and "KEEPER_A2A_TOKEN" in err["message"]


async def test_a2a_rejects_a_wrong_token(client, monkeypatch):
    monkeypatch.setenv("KEEPER_A2A_TOKEN", "test-token")
    r = await client.post("/a2a", json={"jsonrpc": "2.0", "id": "1",
                                        "method": "message/send", "params": {}},
                          headers={"authorization": "Bearer wrong"})
    assert r.json()["error"]["code"] == -32002


async def test_chat_survives_model_failure(client):
    # If the model call blows up (outage, rate limit), /chat must degrade to a
    # graceful in-voice line, not throw a raw 500 at the person.
    def boom(system, user):
        raise RuntimeError("model is down")
    server.STATE.generate = boom

    r = await client.post("/chat", json={"message": "are you there"})
    assert r.status_code == 200, f"got {r.status_code}, not a graceful reply"
    assert r.json()["reply"], "no reply text on failure"


async def test_state_shape(client):
    s = (await client.get("/state")).json()
    for key in ("energy", "base_score", "speak_probability", "facts_kept", "speed"):
        assert key in s


async def test_config_sets_speed(client):
    out = (await client.post("/config", json={"speed": 300})).json()
    assert out["speed"] == 300


def _isolate_stores(monkeypatch, tmp_path):
    """Every store the loop or a turn can read or write, in tmp_path."""
    import dial
    import jobs
    import memory as memory_mod
    import proposals
    import reminders
    import tasks
    monkeypatch.setattr(server.STATE, "store",
                        memory_mod.MemoryStore(tmp_path / "facts.jsonl"))
    monkeypatch.setattr(server.STATE, "goals", tasks.GoalStore(tmp_path / "g.jsonl"))
    monkeypatch.setattr(server.STATE, "reminders",
                        reminders.ReminderStore(tmp_path / "r.jsonl"))
    monkeypatch.setattr(server.STATE, "jobs", jobs.JobStore(tmp_path / "jobs.jsonl"))
    monkeypatch.setattr(server.STATE, "dial", dial.Dial(tmp_path / "dial.json"))
    monkeypatch.setattr(server.STATE, "proposals",
                        proposals.ProposalStore(tmp_path / "proposals.jsonl"))
    import ways
    monkeypatch.setattr(server.STATE, "ways", ways.WayStore(tmp_path / "ways.jsonl"))


def _isolate_delivery(monkeypatch, tmp_path):
    import sessions
    import unbidden
    monkeypatch.setattr(server.STATE, "sessions",
                        sessions.SessionStore(tmp_path / "sessions"))
    monkeypatch.setattr(server.STATE, "unbidden",
                        unbidden.UnbiddenLog(tmp_path / "unbidden.jsonl"))
    monkeypatch.setattr(server.STATE, "current_key", None)
    monkeypatch.setattr(server.STATE, "delivery",
                        server.channels.Delivery(
                            [server.channels.SSEChannel(server.STATE.listeners)]))


class _Req:
    """Just enough of a Request for /events: the one header it reads."""
    def __init__(self, last_event_id=None):
        self.headers = ({"last-event-id": str(last_event_id)}
                        if last_event_id is not None else {})


async def _frames(response, n, timeout=2):
    """The first n SSE frames off the real /events generator. ASGITransport buffers
    an infinite stream, so this reads the body iterator directly."""
    it = response.body_iterator
    out = []
    while len(out) < n:
        frame = await asyncio.wait_for(it.__anext__(), timeout)
        if not frame.startswith(":"):
            out.append(frame)
    await it.aclose()
    return out


async def test_an_unbidden_line_is_kept_even_with_no_conversation_open(
        monkeypatch, tmp_path):
    """Regression: a line spoken after "begin again", or on a first run, went to
    the open page and nowhere else, so a reload had nothing to show for it."""
    _isolate_delivery(monkeypatch, tmp_path)
    await server._deliver_proactive("the water stirs", "tidal",
                                    why=server.unbidden.Why.ROUTINE)
    key = server.STATE.current_key
    assert key is not None, "no conversation was opened to keep the line in"
    msgs = server.STATE.sessions.get(key).messages
    assert [(m.content, m.kind) for m in msgs] == [("the water stirs", "proactive")]
    assert server.STATE.unbidden.lines[-1].why == "routine"


async def test_the_day_counts_outreach_but_not_kept_promises(monkeypatch, tmp_path):
    """The split the four copies of this bookkeeping used to make by accident."""
    _isolate_delivery(monkeypatch, tmp_path)
    monkeypatch.setattr(server.STATE, "proactive_at", server.deque(maxlen=200))
    await server._deliver_proactive("restless", why=server.unbidden.Why.RESTLESS)
    await server._deliver_proactive("a promise", why=server.unbidden.Why.REMINDER,
                                    counts_toward_day=False)
    assert server.STATE.proactive_today() == 1


async def test_a_reconnecting_page_is_told_what_it_missed(monkeypatch, tmp_path):
    """The point of the log. A page that last saw line 1 reconnects and gets lines
    2 and 3, each with its id, so its next reconnect can pick up from there."""
    _isolate_delivery(monkeypatch, tmp_path)
    for text in ("one", "two", "three"):
        await server._deliver_proactive(text)
    resp = await server.events(_Req(last_event_id=1))
    frames = await _frames(resp, 2)
    assert [json.loads(f.split("data: ", 1)[1])["content"] for f in frames] == [
        "two", "three"]
    assert frames[0].startswith("id: 2\n")


async def test_a_fresh_page_gets_no_replay(monkeypatch, tmp_path):
    """No Last-Event-ID means a new page, which loads the conversation itself.
    Replaying to it would draw every line twice."""
    _isolate_delivery(monkeypatch, tmp_path)
    await server._deliver_proactive("already in the conversation")
    resp = await server.events(_Req())
    it = resp.body_iterator
    assert (await it.__anext__()).startswith(":")      # the greeting, then nothing
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(it.__anext__(), 0.2)
    await it.aclose()


async def test_a_line_said_during_the_replay_is_not_sent_twice(
        monkeypatch, tmp_path):
    """A line is recorded first and pushed after, so one recorded just before a
    page reconnects is both in its replay and, a moment later, in its queue. It
    must arrive once."""
    _isolate_delivery(monkeypatch, tmp_path)
    line = server.STATE.unbidden.record("caught in between")   # logged, not pushed
    resp = await server.events(_Req(last_event_id=0))
    await server.STATE.delivery.push(                           # ...the push lands
        "assistant", line.text, "proactive",
        meta={"id": line.seq, "session": None})
    await server._deliver_proactive("after")
    frames = await _frames(resp, 2)
    contents = [json.loads(f.split("data: ", 1)[1])["content"] for f in frames]
    assert contents == ["caught in between", "after"]


async def test_config_turns_the_dial(client, monkeypatch, tmp_path):
    import dial
    monkeypatch.setattr(server.STATE, "dial", dial.Dial(tmp_path / "dial.json"))
    out = (await client.post("/config", json={"level": "quiet"})).json()
    assert out["level"] == "quiet"
    s = (await client.get("/state")).json()
    assert s["level"] == "quiet" and s["daily_max"] == 4
    bad = (await client.post("/config", json={"level": "whisper"})).json()
    assert bad["level"] == "quiet" and "unknown level" in bad["error"]


async def test_off_keeps_a_promise_and_starts_nothing(client, monkeypatch, tmp_path):
    """The real loop, restless and sped up, at off. A due reminder must still come
    back; nothing of the Keeper's own may follow it."""
    import dial
    import reminders
    d = dial.Dial(tmp_path / "dial.json")
    d.set("off")
    monkeypatch.setattr(server.STATE, "dial", d)
    monkeypatch.setattr(server.STATE, "reminders",
                        reminders.ReminderStore(tmp_path / "r.jsonl"))
    server.STATE.reminders.add("water the plants", time.time() - 1)
    server.STATE.last_user_at = time.time() - 100_000   # as restless as it gets
    server.STATE.last_proactive_at = None
    server.STATE.config.respect_lock = False
    server.STATE.config.use_presence = False
    q: asyncio.Queue = asyncio.Queue()
    server.STATE.listeners.add(q)
    try:
        await client.post("/config", json={"speed": 900, "cooldown_min": 0})
        first = json.loads(await asyncio.wait_for(q.get(), 10))
        assert "water the plants" in first["content"]
        assert first["why"] == "reminder"
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q.get(), 3)
    finally:
        server.STATE.listeners.discard(q)
        await client.post("/config", json={"speed": 1, "cooldown_min": 600})


async def test_push_delivers_to_listener():
    # The delivery seam: a proactive line fans out to every registered listener
    # (this is what /events subscribes to internally).
    q: asyncio.Queue = asyncio.Queue()
    server.STATE.listeners.add(q)
    try:
        await server._push("assistant", "the tide returns", "proactive")
        payload = json.loads(await asyncio.wait_for(q.get(), 2))
        assert payload["kind"] == "proactive"
        assert payload["content"] == "the tide returns"
    finally:
        server.STATE.listeners.discard(q)


async def test_proactive_loop_reaches_out(client):
    # End-to-end reach-out: the real background loop, made restless and sped up,
    # must deliver a proactive line to a subscribed listener. This is the
    # mechanism behind the notifications. (Delivery over the /events HTTP stream
    # itself is verified manually with curl; ASGITransport buffers infinite SSE.)
    server.STATE.last_user_at = time.time() - 100_000   # long silence -> restless
    server.STATE.last_proactive_at = None               # hasn't just reached out
    server.STATE.config.respect_lock = False
    server.STATE.config.use_presence = False
    q: asyncio.Queue = asyncio.Queue()
    server.STATE.listeners.add(q)
    try:
        await client.post("/config", json={"speed": 900, "cooldown_min": 0})
        payload = json.loads(await asyncio.wait_for(q.get(), 25))
        assert payload["kind"] == "proactive"
        assert payload["content"]
    finally:
        server.STATE.listeners.discard(q)


# --- notifier: text that must survive AppleScript --- #

def test_applescript_string_cannot_escape_its_quotes():
    """Regression: the osascript fallback escaped double quotes but not backslashes.
    AppleScript treats "\\" as an escape, so a message ending in one escaped the
    CLOSING quote and let the string run on into the script. The text is model
    written, and the model reads fetched web pages, so it is not fully trusted."""
    import notifier
    assert "\\" not in notifier._as_string("all is well\\")
    assert '"' not in notifier._as_string('he said "hi"')
    assert "\n" not in notifier._as_string("line one\nline two")
    assert notifier._as_string("plain message") == "plain message"


# --- the register chain degrades in the right order --- #

def _chain(msg, *, keyword, llm, local):
    """Reproduce server.py's layering so the ORDER is pinned, not just the parts.

    The model decides when it is available; keyword + anchors are the offline
    path, reached only when there is no model or the call fails.
    """
    if llm is not None:
        try:
            return llm(msg)                  # None here is a decision: neutral
        except Exception:                    # noqa: BLE001
            pass
    signal = keyword(msg)
    if signal is None and local is not None:
        signal = local(msg)
    return signal


def test_the_model_outranks_the_lexicon():
    """The keyword layer used to go first, justified as high precision. Measured on
    real messages it fires 9 times in 32 and is right 5 of those: it met "go look
    into watercolor vs gouache" as grief, because "ache" sits inside "gouache".
    First in the chain it had veto over a classifier three times more accurate."""
    assert _chain("go look into gouache", keyword=lambda m: "frozen",
                  llm=lambda m: None, local=lambda m: "tidal") is None


def test_the_model_is_used_when_present():
    assert _chain("i just stare at the ceiling", keyword=lambda m: None,
                  llm=lambda m: "frozen", local=lambda m: "tidal") == "frozen"


def test_a_failing_model_falls_back_to_the_local_classifier():
    """A 429 or an outage must not cost register detection entirely: the local
    Model2Vec path is why this still works with no key and no network."""
    def boom(m): raise RuntimeError("429")
    assert _chain("x", keyword=lambda m: None, llm=boom,
                  local=lambda m: "tidal") == "tidal"


def test_with_no_key_at_all_the_local_classifier_carries_it():
    assert _chain("x", keyword=lambda m: None, llm=None,
                  local=lambda m: "frozen") == "frozen"


def test_all_layers_abstaining_means_inherit():
    """None reaches the caller, which leaves STATE.current_register alone."""
    assert _chain("what's on my list", keyword=lambda m: None,
                  llm=lambda m: None, local=lambda m: None) is None


def test_the_models_abstention_is_a_decision_not_a_gap():
    """Regression, caught live: the model answering None means it read the message
    as NEUTRAL. Falling through to the weaker classifier on None put "go look into
    watercolor vs gouache" back to frozen: the exact error the model was brought
    in to fix. Only an unavailable model may fall back."""
    assert _chain("go look into watercolor vs gouache", keyword=lambda m: None,
                  llm=lambda m: None, local=lambda m: "frozen") is None


# --- watching an MCP tool for something new --- #

import memory
import sources

_GIT_LOG = """commit 4f2a1b9c
Author: Jess <jess@example.com>
Date:   Thu Sep 4 19:02:11 2026

    finished stretching the big canvas
"""


class _FakeMCP:
    """Stands in for a connected manager. Only what _poll_sources touches."""

    def __init__(self, text=_GIT_LOG, watches=None):
        self.text = text
        self.watches = watches if watches is not None else [
            {"name": "repo", "tool": "git__git_log", "args": {"max_count": 5}}]
        self.calls = []

    async def call(self, tool, arguments):
        self.calls.append((tool, arguments))
        return self.text


@pytest.fixture
def polling(monkeypatch, tmp_path):
    """STATE wired for a poll: one fact known, nothing seen yet, no feeds."""
    st = server.STATE
    store = memory.MemoryStore(tmp_path / "facts.jsonl")
    store.add("Is a painter who stopped in March.", kind="fact")
    monkeypatch.setattr(st, "store", store)
    monkeypatch.setattr(st, "seen_sources", sources.SeenStore(tmp_path / "seen.jsonl"))
    monkeypatch.setattr(st, "last_poll_at", None)
    monkeypatch.setattr(st, "pending_item", None)
    monkeypatch.setenv("KEEPER_FEEDS", "")
    return st


async def test_a_watched_tool_can_produce_a_pending_item(polling, monkeypatch):
    """The feature in one line: with no feeds configured at all, something the
    Keeper NOTICED through its own tools can still become what it speaks about."""
    monkeypatch.setattr(server.relevance, "score_item",
                        lambda *a, **k: (0.9, "they paint"))
    fake = _FakeMCP()
    monkeypatch.setattr(polling, "mcp", fake)
    await server._poll_sources()
    assert fake.calls == [("git__git_log", {"max_count": 5})]
    assert polling.pending_item is not None
    assert polling.pending_item.title == "finished stretching the big canvas"
    assert polling.pending_item.source == "repo"


async def test_a_watched_item_that_is_irrelevant_is_not_kept(polling, monkeypatch):
    """Same gate as a feed item: noticing is not a reason to speak."""
    monkeypatch.setattr(server.relevance, "score_item", lambda *a, **k: (0.05, ""))
    monkeypatch.setattr(polling, "mcp", _FakeMCP())
    await server._poll_sources()
    assert polling.pending_item is None
    assert polling.last_scan, "the scan should still be visible on the dashboard"


async def test_a_broken_tool_contributes_nothing(polling, monkeypatch):
    """mcp.call answers with a parenthetical instead of raising. That string must
    never become an item the Keeper reports as news."""
    monkeypatch.setattr(server.relevance, "score_item",
                        lambda *a, **k: (0.99, "would speak"))
    monkeypatch.setattr(polling, "mcp",
                        _FakeMCP(text="(tool timed out after 20s: git__git_log)"))
    await server._poll_sources()
    assert polling.pending_item is None


async def test_a_watched_item_is_delivered_only_once(polling, monkeypatch):
    """The same commit is in the log every poll; only the first is new."""
    monkeypatch.setattr(server.relevance, "score_item",
                        lambda *a, **k: (0.9, "they paint"))
    monkeypatch.setattr(polling, "mcp", _FakeMCP())
    await server._poll_sources()
    first = polling.pending_item
    assert first is not None
    polling.seen_sources.mark(first)          # as delivery does
    polling.pending_item = None
    polling.last_poll_at = None
    await server._poll_sources()
    assert polling.pending_item is None


async def test_no_feeds_and_no_watches_does_nothing(polling, monkeypatch):
    monkeypatch.setattr(polling, "mcp", _FakeMCP(watches=[]))
    await server._poll_sources()
    assert polling.pending_item is None
    assert polling.last_poll_at is None, "the poll interval should not be burned"


# --- a score is a claim about memory as it stands now --- #

async def test_a_memory_change_rescores_without_refetching(polling, monkeypatch):
    """The bug this exists for, seen live while capturing screenshots. The first
    poll landed mid-conversation, when "Stopped painting in March." was the only
    fact, and the judge correctly scored a painting show near zero for someone who
    had stopped painting. The reversal arrived seconds later, but the all-silent
    scan was already stamped and stood for the next fifteen minutes."""
    fake = _FakeMCP()
    monkeypatch.setattr(polling, "mcp", fake)
    scores = iter([(0.05, ""), (0.9, "they paint")])
    monkeypatch.setattr(server.relevance, "score_item",
                        lambda *a, **k: next(scores))

    await server._poll_sources()                 # polls mid-arc, judges it silent
    assert polling.pending_item is None
    assert len(fake.calls) == 1

    polling.store.add("Started painting again.", kind="fact")   # the reversal lands
    await server._poll_sources()

    assert polling.pending_item is not None, "a changed memory did not re-score"
    assert len(fake.calls) == 1, "it went back to the network to re-score"


async def test_an_unchanged_memory_does_not_rescore(polling, monkeypatch):
    """The other direction: without this, every tick would re-judge the same
    items against the same facts and pay for a model call each time."""
    monkeypatch.setattr(polling, "mcp", _FakeMCP())
    calls = []
    monkeypatch.setattr(server.relevance, "score_item",
                        lambda *a, **k: (calls.append(1), (0.05, ""))[1])
    await server._poll_sources()
    first = len(calls)
    await server._poll_sources()
    assert len(calls) == first, "re-scored with nothing in memory changed"


async def test_a_superseded_fact_counts_as_a_change(polling, monkeypatch):
    """The fingerprint has to catch supersession, not just addition: the active
    count can stay flat while what is known changes completely."""
    before = server._memory_fingerprint()
    old = polling.store.add("Lives in Portland.", kind="fact")
    mid = server._memory_fingerprint()
    assert mid != before
    new = polling.store.add("Lives in Chicago.", kind="fact")
    new.supersedes = old.id
    old.valid_until = 1.0
    assert server._memory_fingerprint() != mid


async def test_the_fetch_interval_still_holds(polling, monkeypatch):
    """Re-scoring must not become a way to hammer the feeds."""
    fake = _FakeMCP()
    monkeypatch.setattr(polling, "mcp", fake)
    monkeypatch.setattr(server.relevance, "score_item", lambda *a, **k: (0.05, ""))
    await server._poll_sources()
    for i in range(3):
        polling.store.add(f"Fact number {i}.", kind="fact")
        await server._poll_sources()
    assert len(fake.calls) == 1, f"refetched {len(fake.calls)} times inside the interval"


# --- proposals: what it would do, and the yes it waits on --- #

class _FakeActions:
    """An MCP manager with one tool behind approval, recording what actually ran."""
    def __init__(self):
        self.ran = []

    def proposable(self):
        return [{"name": "mail__send", "description": "Send an email.",
                 "parameters": {"type": "object",
                                "properties": {"to": {"type": "string"},
                                               "body": {"type": "string"}},
                                "required": ["to", "body"]}}]

    async def call_proposable(self, name, args):
        self.ran.append((name, dict(args)))
        return "sent"

    has_tools = False
    watches: list = []

    def openai_tools(self):
        return []

    health: list = []

    def summary(self):
        return ""


@pytest.fixture
def actions(monkeypatch, tmp_path):
    import proposals
    _isolate_delivery(monkeypatch, tmp_path)
    fake = _FakeActions()
    monkeypatch.setattr(server.STATE, "mcp", fake)
    monkeypatch.setattr(server.STATE, "proposals",
                        proposals.ProposalStore(tmp_path / "proposals.jsonl"))
    return fake


async def _propose(to="sam@example.com", body="I can't make Thursday."):
    proposer = server._proposer("chat")
    await proposer.call("mail__send", {"to": to, "body": body})
    return server.STATE.proposals.items[-1]


async def test_a_proposal_is_shown_and_kept_without_running(actions):
    q: asyncio.Queue = asyncio.Queue()
    server.STATE.listeners.add(q)
    try:
        p = await _propose()
        payload = json.loads(q.get_nowait())
    finally:
        server.STATE.listeners.discard(q)
    assert actions.ran == []
    assert payload["kind"] == "proposal" and payload["proposal"]["id"] == p.id
    assert "to: sam@example.com" in payload["proposal"]["rendered"]
    (msg,) = server.STATE.sessions.get(server.STATE.current_key).messages
    assert (msg.kind, msg.ref) == ("proposal", p.id)


async def test_asking_permission_spends_none_of_the_outreach_budget(actions):
    """A proposal from the chat path lands mid-conversation. Counting it as the
    Keeper reaching out would silence a restless line for no reason, and spend
    a day's ceiling on paperwork."""
    server.STATE.last_proactive_at = None
    before = len(server.STATE.proactive_at)
    await _propose()
    assert server.STATE.last_proactive_at is None
    assert len(server.STATE.proactive_at) == before


async def test_approving_runs_exactly_what_was_shown_once(actions):
    p = await _propose()
    assert await server._decide(p.id, True, "web") == "Done."
    assert await server._decide(p.id, True, "telegram") == "Already done."
    assert actions.ran == [("mail__send", {"to": "sam@example.com",
                                           "body": "I can't make Thursday."})]
    assert p.status == "done" and p.decided_via == "web" and p.result == "sent"


async def test_declining_runs_nothing(actions):
    p = await _propose()
    assert "Nothing was done" in await server._decide(p.id, False, "telegram")
    assert await server._decide(p.id, True, "web") == "Already declined."
    assert actions.ran == []


async def test_a_decision_is_announced_to_the_page(actions):
    p = await _propose()
    q: asyncio.Queue = asyncio.Queue()
    server.STATE.listeners.add(q)
    try:
        await server._decide(p.id, True, "telegram")
        update = json.loads(q.get_nowait())
    finally:
        server.STATE.listeners.discard(q)
    assert update["kind"] == "proposal_update"
    assert update["proposal"]["status"] == "done"
    assert "Approved in telegram" in update["content"]


async def test_a_replayed_proposal_comes_back_as_it_now_stands(actions):
    """Decided from the phone while the page was away. The replay must not hand
    the page live buttons for something already done."""
    p = await _propose()
    await server._decide(p.id, False, "telegram")
    resp = await server.events(_Req(last_event_id=0))
    (frame,) = await _frames(resp, 1)
    replayed = json.loads(frame.split("data: ", 1)[1])
    assert replayed["kind"] == "proposal"
    assert replayed["proposal"]["status"] == "declined"


async def test_the_dial_at_off_still_asks(actions, monkeypatch, tmp_path):
    """A proposal is a question waiting on them. Holding it back at off would
    leave a decision they cannot see, ageing toward its expiry."""
    import dial
    d = dial.Dial(tmp_path / "dial.json")
    d.set("off")
    monkeypatch.setattr(server.STATE, "dial", d)
    q: asyncio.Queue = asyncio.Queue()
    server.STATE.listeners.add(q)
    try:
        await _propose()
        assert json.loads(q.get_nowait())["kind"] == "proposal"
    finally:
        server.STATE.listeners.discard(q)


async def test_decide_endpoint_refuses_another_site(client, actions):
    """A plain cross-site form POST carries Host: localhost and passes the Host
    check. Origin is what it cannot forge."""
    p = await _propose()
    r = await client.post("/proposals/decide",
                          json={"id": p.id, "decision": "approve"},
                          headers={"origin": "https://evil.example.com"})
    assert r.json()["ok"] is False and actions.ran == []
    ok = await client.post("/proposals/decide",
                           json={"id": p.id, "decision": "approve"},
                           headers={"origin": "http://localhost:8790"})
    assert ok.json()["proposal"]["status"] == "done"
    listed = (await client.get("/proposals")).json()
    assert listed["pending"] == [] and listed["recent"][0]["id"] == p.id


async def test_no_proposer_without_a_tool_behind_approval(monkeypatch):
    monkeypatch.setattr(server.STATE, "mcp", tools.MCPManager([]))
    assert server._proposer("chat") is None


async def test_the_person_can_be_asked_but_a_peer_agent_cannot(actions,
                                                                monkeypatch):
    """Chat gets the proposer; A2A does not. A queue anyone can fill trains
    people to tap yes."""
    import proposals
    seen = {}

    async def fake_tool_reply(system, user, *, providers=None, **kw):
        seen.setdefault("calls", []).append(providers or [])
        return "ok"

    monkeypatch.setattr(server.compose, "tool_reply", fake_tool_reply)
    monkeypatch.setattr(server.compose, "revoice", lambda text, *a, **k: text)
    monkeypatch.setattr(server.memory, "recall", lambda *a, **k: "")
    monkeypatch.setattr(server, "_distill_async", _noop_distill)
    await server.chat(server.ChatIn(message="email sam that i can't do thursday"))
    await server._answer_as_keeper("send an email for me")
    chat_providers, peer_providers = seen["calls"]
    assert any(isinstance(p, proposals.Proposer) for p in chat_providers)
    assert not any(isinstance(p, proposals.Proposer) for p in peer_providers)


async def _noop_distill(*a, **k):
    return None


# --- a turn sees the last few exchanges --- #

async def test_a_turn_sees_the_recent_thread_and_not_yesterday(monkeypatch, tmp_path):
    _isolate_delivery(monkeypatch, tmp_path)
    s = server.STATE.sessions.new()
    now = time.time()
    server.STATE.sessions.append(s.key, "user", "from yesterday", now - 86_400)
    for i in range(12):
        server.STATE.sessions.append(s.key, "user", f"m{i}", now - 60 + i)
    server.STATE.sessions.append(s.key, "assistant", "Sam wrote back.", now - 5,
                                 kind="proactive")
    turns = server._recent_turns(s.key, now)
    assert len(turns) == server.RECENT_TURNS
    assert "from yesterday" not in [t["content"] for t in turns]
    assert turns[-1] == {"role": "assistant", "content": "Sam wrote back."}


async def test_chat_hands_the_model_the_thread(monkeypatch, tmp_path):
    """Regression: every message used to arrive alone, so "what did he say?" had
    nothing to refer to. The second turn must see the first."""
    _isolate_delivery(monkeypatch, tmp_path)
    seen = []

    async def fake_tool_reply(system, user, *, prior=None, **kw):
        seen.append([m["content"] for m in prior or []])
        return "kept"

    monkeypatch.setattr(server.compose, "tool_reply", fake_tool_reply)
    monkeypatch.setattr(server.compose, "revoice", lambda text, *a, **k: text)
    monkeypatch.setattr(server.memory, "recall", lambda *a, **k: "")
    monkeypatch.setattr(server, "_distill_async", _noop_distill)
    await server.chat(server.ChatIn(message="sam asked about thursday"))
    await server.chat(server.ChatIn(message="what did he ask?"))
    assert seen == [[], ["sam asked about thursday", "kept"]]


# --- the phone's own thread --- #

class _PhoneChannel:
    name = "telegram"

    def wants(self, kind):
        return kind in ("proactive", "proposal", "proposal_update")

    async def deliver(self, *a, **k):
        return None


async def test_the_phone_talks_in_its_own_thread(monkeypatch, tmp_path):
    """Switching conversations on the page must never move where the phone is
    talking, so the phone has one fixed thread of its own."""
    _isolate_delivery(monkeypatch, tmp_path)

    async def fake_tool_reply(system, user, **kw):
        return "kept"

    monkeypatch.setattr(server.compose, "tool_reply", fake_tool_reply)
    monkeypatch.setattr(server.compose, "revoice", lambda text, *a, **k: text)
    monkeypatch.setattr(server.memory, "recall", lambda *a, **k: "")
    monkeypatch.setattr(server, "_distill_async", _noop_distill)
    web = server.STATE.sessions.new().key
    server.STATE.current_key = web
    assert await server._converse_from_phone("hello from the bus") == "kept"
    phone = server.STATE.sessions.get(server.PHONE_KEY)
    assert [m.content for m in phone.messages] == ["hello from the bus", "kept"]
    assert server.STATE.sessions.get(web).messages == []
    assert server.STATE.current_key == web


async def test_a_line_sent_to_the_phone_is_kept_in_its_thread(monkeypatch, tmp_path):
    _isolate_delivery(monkeypatch, tmp_path)
    server.STATE.delivery.channels.append(_PhoneChannel())
    server.STATE.current_key = server.STATE.sessions.new().key
    await server._deliver_proactive("Sam wrote back.", why=server.unbidden.Why.NOTICED)
    phone = server.STATE.sessions.get(server.PHONE_KEY)
    assert [(m.content, m.kind) for m in phone.messages] == [
        ("Sam wrote back.", "proactive")]


def test_the_page_never_resumes_into_the_phones_thread(tmp_path):
    import sessions
    store = sessions.SessionStore(tmp_path)
    web = store.new()
    store.append(web.key, "user", "earlier, on the page", 1.0)
    store.get_or_create(server.PHONE_KEY, title="telegram")
    store.append(server.PHONE_KEY, "user", "later, on the phone")
    assert store.most_recent_key(exclude=server.PHONE_KEY) == web.key


# --- scheduled jobs --- #

@pytest.fixture
def scheduled(monkeypatch, tmp_path):
    import dial
    import jobs
    import proposals
    import reminders
    import tasks
    _isolate_delivery(monkeypatch, tmp_path)
    monkeypatch.setattr(server.STATE, "jobs", jobs.JobStore(tmp_path / "jobs.jsonl"))
    monkeypatch.setattr(server.STATE, "dial", dial.Dial(tmp_path / "dial.json"))
    monkeypatch.setattr(server.STATE, "proposals",
                        proposals.ProposalStore(tmp_path / "proposals.jsonl"))
    monkeypatch.setattr(server.STATE, "reminders",
                        reminders.ReminderStore(tmp_path / "r.jsonl"))
    monkeypatch.setattr(server.STATE, "goals", tasks.GoalStore(tmp_path / "g.jsonl"))
    monkeypatch.setattr(server.STATE, "mcp", tools.MCPManager([]))
    monkeypatch.setattr(server.compose, "revoice", lambda text, *a, **k: text)
    return server.STATE.jobs


async def test_a_job_comes_back_with_its_result_even_at_off(scheduled, monkeypatch):
    """They scheduled it, so it is theirs: the dial at off holds the Keeper's own
    initiative, not work it was asked to do."""
    server.STATE.dial.set("off")

    async def fake_orchestrate(task, generate, **kw):
        return server.subagents.OrchestrationResult(
            task=task, result="The gallery posted dates: Nov 3 to Nov 20.")

    monkeypatch.setattr(server.subagents, "orchestrate", fake_orchestrate)
    job = scheduled.add("check the gallery's site", time.time() - 1)
    q: asyncio.Queue = asyncio.Queue()
    server.STATE.listeners.add(q)
    try:
        await server._run_job(job)
        payload = json.loads(q.get_nowait())
    finally:
        server.STATE.listeners.discard(q)
    assert payload["why"] == "job" and "Nov 3 to Nov 20" in payload["content"]
    assert job.last_status == "done" and job.runs == 1


async def test_a_job_that_finds_nothing_says_nothing(scheduled, monkeypatch):
    async def empty(task, generate, **kw):
        return server.subagents.OrchestrationResult(task=task, result="")

    monkeypatch.setattr(server.subagents, "orchestrate", empty)
    job = scheduled.add("check the gallery's site", time.time() - 1)
    await server._run_job(job)
    assert job.last_status == "failed" and server.STATE.unbidden.lines == []


def test_the_briefs_ledger_is_read_from_the_stores_not_written(scheduled):
    """Commitments are listed by code: a model summarising "you promised to call
    the dentist at three" can summarise it wrong."""
    now = time.time()
    server.STATE.reminders.add("call the dentist", now + 60)
    server.STATE.proposals.add("mail__send", {"to": "sam@example.com"})
    server.STATE.goals.add("get back to painting", ["[person] set out the paints"])
    ledger = "\n".join(server._brief_ledger(now))
    assert "Waiting on your yes: 1 (send (mail))" in ledger
    assert "call the dentist" in ledger
    assert "get back to painting (0/1), next: set out the paints" in ledger


async def test_an_empty_brief_is_not_sent(scheduled):
    """No Google, nothing held, nothing waiting: a brief of nothing is noise."""
    job = scheduled.add("the morning brief", time.time() - 1, kind="brief")
    await server._run_job(job)
    assert server.STATE.unbidden.lines == []


async def test_the_loop_runs_a_due_job(client, scheduled, monkeypatch):
    ran = asyncio.Event()

    async def fake_run_job(job):
        ran.set()

    monkeypatch.setattr(server, "_run_job", fake_run_job)
    scheduled.add("check the gallery's site", time.time() - 1)
    server.STATE.wake.set()
    await asyncio.wait_for(ran.wait(), 10)


# --- watching the inbox and calendar --- #

_INBOX = """Found 1 messages matching 'in:inbox is:unread':

📧 MESSAGES:
  1. Message ID: 18c2a9f0
     Subject: The gallery wants your paintings
     From: Riverside Gallery <hello@riverside.example>
     Date: Sat, 10 Oct 2026 18:02:11 -0500
     Web Link: https://mail.google.com/mail/u/0/#all/18c2a9f0
     Thread ID: 18c2a9f0
     Thread Link: https://mail.google.com/mail/u/0/#all/18c2a9f0

💡 USAGE:
  • Pass the Message IDs **as a list** to get_gmail_messages_content_batch()
"""


async def test_a_message_that_bears_on_a_kept_fact_becomes_worth_saying(
        polling, monkeypatch):
    """The inbox watched like any source: read by its own parser, scored against
    memory, and only the message that touches something kept is held."""
    monkeypatch.setattr(server.relevance, "score_item",
                        lambda item, *a, **k: (0.9, "they paint")
                        if "gallery" in item.title.lower() else (0.0, ""))
    fake = _FakeMCP(text=_INBOX, watches=[{
        "name": "inbox", "tool": "google__search_gmail_messages",
        "args": {"query": "in:inbox is:unread", "include_headers": True},
        "parser": "gmail"}])
    monkeypatch.setattr(polling, "mcp", fake)
    await server._poll_sources()
    assert polling.pending_item.title == (
        "The gallery wants your paintings (from Riverside Gallery "
        "<hello@riverside.example>)")
    assert [r["title"] for r in polling.last_scan] == [polling.pending_item.title]


async def test_a_calendar_watch_asks_about_now_not_the_day_it_was_written(
        polling, monkeypatch):
    monkeypatch.setattr(server.relevance, "score_item", lambda *a, **k: (0.0, ""))
    fake = _FakeMCP(text="", watches=[{
        "name": "calendar", "tool": "google__get_events",
        "args": {"time_min": "${NOW}", "time_max": "${NOW+1d}"},
        "parser": "calendar"}])
    monkeypatch.setattr(polling, "mcp", fake)
    await server._poll_sources()
    (tool, args), = fake.calls
    assert "${" not in args["time_min"] and "${" not in args["time_max"]
    assert args["time_min"][:4].isdigit() and args["time_min"] < args["time_max"]


# --- ways: learned, then applied to work --- #

async def test_a_learned_way_reaches_the_chat_turn(monkeypatch, tmp_path):
    import ways
    _isolate_delivery(monkeypatch, tmp_path)
    monkeypatch.setattr(server.STATE, "ways", ways.WayStore(tmp_path / "w.jsonl"))
    server.STATE.ways.add("emails to Sam", "sign off as J")
    systems = []

    async def fake_tool_reply(system, user, steer="", **kw):
        systems.append(steer)
        return "drafted"

    monkeypatch.setattr(server.compose, "tool_reply", fake_tool_reply)
    monkeypatch.setattr(server.compose, "revoice", lambda text, *a, **k: text)
    monkeypatch.setattr(server.memory, "recall", lambda *a, **k: "")
    monkeypatch.setattr(server, "_distill_async", _noop_distill)
    await server.chat(server.ChatIn(message="email Sam that thursday is off"))
    assert "emails to Sam: sign off as J" in systems[0]
    assert server.STATE.traces[0]["ways_applied"] == ["emails to Sam: sign off as J"]


async def test_a_learned_way_reaches_a_specialists_work(scheduled, monkeypatch,
                                                         tmp_path):
    """Before ways, a job's or a goal's draft ignored everything they had taught."""
    import ways
    monkeypatch.setattr(server.STATE, "ways", ways.WayStore(tmp_path / "w.jsonl"))
    server.STATE.ways.add("gallery", "check their Instagram, not the website")
    tasks_seen = []

    async def fake_orchestrate(task, generate, **kw):
        tasks_seen.append(task)
        return server.subagents.OrchestrationResult(task=task, result="")

    monkeypatch.setattr(server.subagents, "orchestrate", fake_orchestrate)
    await server._run_job(scheduled.add("check the gallery's autumn dates",
                                        time.time() - 1))
    assert "check their Instagram, not the website" in tasks_seen[0]


async def test_an_edited_approval_runs_the_edit_and_learns_from_it(
        actions, monkeypatch, tmp_path):
    import ways
    monkeypatch.setattr(server.STATE, "ways", ways.WayStore(tmp_path / "w.jsonl"))
    monkeypatch.setattr(server.STATE, "generate",
                        lambda s, u: "ABOUT: emails to Sam | HOW: sign off as J")
    p = await _propose(body="Can't do Thursday.\n\nBest,\nJess")
    note = await server._decide(p.id, True, "web",
                                edited={"body": "Can't do Thursday.\n\nJ"})
    assert note == "Done."
    assert actions.ran == [("mail__send", {"to": "sam@example.com",
                                           "body": "Can't do Thursday.\n\nJ"})]
    await asyncio.sleep(0.2)                   # the learner runs off the path
    assert p.learned == "emails to Sam: sign off as J"
    assert [w.line() for w in server.STATE.ways.active()] == [p.learned]


async def test_a_refused_edit_runs_nothing(actions):
    p = await _propose()
    note = await server._decide(p.id, True, "web", edited={"bcc": "x@else.com"})
    assert note.startswith("Not approved") and actions.ran == []
    assert p.status == "pending"


async def test_the_thread_knows_what_became_of_a_proposal(actions, monkeypatch):
    """Regression from the live demo: after a write failed, "what did you put in
    the note?" was answered as though the note existed, because the thread only
    held the question."""
    async def refuse(name, args):
        return "(tool error: Access denied - path outside allowed directories)"

    monkeypatch.setattr(actions, "call_proposable", refuse)
    p = await _propose()
    await server._decide(p.id, True, "web")
    key = server.STATE.current_key
    # The Keeper's own reply lands after the question, as it does in a real turn.
    server.STATE.sessions.append(key, "assistant", "It is waiting on your yes.",
                                 p.created + 1)
    p.decided_at = p.created + 2
    turns = [t["content"] for t in server._recent_turns(key, p.created + 3)]
    assert "Waiting on your yes" in turns[0]
    assert turns[1] == "It is waiting on your yes."
    assert "Approved, but it failed" in turns[-1] and "Access denied" in turns[-1]


async def test_an_edit_teaches_even_when_the_tool_then_fails(actions, monkeypatch,
                                                              tmp_path):
    import ways
    monkeypatch.setattr(server.STATE, "ways", ways.WayStore(tmp_path / "w.jsonl"))
    monkeypatch.setattr(server.STATE, "generate",
                        lambda s, u: "ABOUT: notes to Sam | HOW: all lowercase")

    async def refuse(name, args):
        return "(tool error: refused)"

    monkeypatch.setattr(actions, "call_proposable", refuse)
    p = await _propose(body="Can't do Thursday.")
    await server._decide(p.id, True, "web", edited={"body": "can't do thursday."})
    await asyncio.sleep(0.2)
    assert p.status == "failed" and p.learned == "notes to Sam: all lowercase"



async def test_the_suite_never_reaches_a_real_surface(client):
    """The guard above, pinned: whatever is in .env, a test run delivers to the
    page and nowhere else, and listens to no bot."""
    assert server.STATE.delivery.names() == ["web"]
    assert server.STATE.inbox is None


# --- the dashboard's week ahead --- #

class _FakeCalendar(_FakeActions):
    def __init__(self, text):
        super().__init__()
        self.text, self.calls = text, 0

    def openai_tools(self):
        return [{"function": {"name": "google__get_events"}}]

    async def call(self, name, args):
        self.calls += 1
        return self.text


async def test_the_week_ahead_is_read_once_and_kept_a_while(monkeypatch):
    cal = _FakeCalendar('- "Call the gallery" (Starts: 2026-10-13T15:00:00-05:00, '
                        'Ends: 2026-10-13T15:30:00-05:00) ID: e9 | Link: https://x/e9')
    monkeypatch.setattr(server.STATE, "mcp", cal)
    monkeypatch.setitem(server._upcoming, "at", 0.0)
    first = await server.upcoming()
    await server.upcoming()
    assert cal.calls == 1, "the dashboard's refresh must not reach Google each time"
    assert first["connected"] and first["events"][0]["title"] == "Call the gallery"


async def test_a_refused_calendar_is_not_an_empty_week(monkeypatch):
    monkeypatch.setattr(server.STATE, "mcp",
                        _FakeCalendar("(tool error: token expired)"))
    monkeypatch.setitem(server._upcoming, "at", 0.0)
    out = await server.upcoming()
    assert out == {"events": [], "connected": False}



def test_today_is_said_next_to_the_request(monkeypatch, tmp_path):
    """The note names the weekday, the date and the year, so "this week" cannot be
    read against the model's training year."""
    note = server._now_note(datetime(2026, 10, 12, 9, 30).timestamp())
    assert "Monday 12 October 2026, 09:30" in note and "this week" in note


# --- the bundled interface font --- #

async def test_the_bundled_font_is_served(client):
    r = await client.get("/fonts/Fraunces.ttf")
    assert r.status_code == 200 and r.headers["content-type"] == "font/ttf"


async def test_the_font_route_serves_nothing_else(client):
    """A fixed list, not a folder: no path a page could ask for reaches another
    file."""
    for name in ("OFL.txt", "..%2Fserver.py", "keeper.png", "Fraunces.TTF"):
        assert (await client.get(f"/fonts/{name}")).status_code == 404
