"""Tier 2 — the running app (server.py): chat, state, and the reach-out path.

Drives the real ASGI app (lifespan + background proactive loop) over httpx, with
the model forced to the offline stub and MCP disabled, so these run with no key,
no cost, and deterministically. The headline test is the reach-out: crank the
battery and assert a proactive line is delivered over SSE — the mechanism behind
the notifications.
"""
import asyncio
import json
import time

import httpx
import pytest

import compose
import server
import tools

pytestmark = pytest.mark.integration


@pytest.fixture
async def client(monkeypatch):
    # No MCP (fast, deterministic) and force the offline stub generator.
    monkeypatch.setattr(tools.MCPManager, "from_config",
                        classmethod(lambda cls, *a, **k: tools.MCPManager([])))
    async with server.app.router.lifespan_context(server.app):
        server.STATE.generate, server.STATE.fast = compose.stub_generator, None
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://localhost") as c:
            yield c


async def test_rejects_foreign_host(client):
    # DNS-rebinding defense: a request whose Host isn't localhost/127.0.0.1 (as a
    # malicious website's rebind would send) must be refused, not served.
    r = await client.get("/state", headers={"host": "evil.example.com"})
    assert r.status_code == 400, "foreign Host was served — DNS-rebinding open"


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
    req = {"jsonrpc": "2.0", "id": "1", "method": "message/send",
           "params": {"message": {"role": "user",
                                  "parts": [{"kind": "text", "text": "who are you?"}]}}}
    r = await client.post("/a2a", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "1"
    import a2a
    assert a2a.text_of(body["result"]) == "kept: who are you?"


async def test_a2a_rejects_unknown_method(client):
    r = await client.post("/a2a", json={"jsonrpc": "2.0", "id": "9",
                                        "method": "tasks/cancel", "params": {}})
    assert r.json()["error"]["code"] == -32601


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
