"""
These are the Tier 1 tests for the Almanac, a separate A2A agent the Keeper can consult.

It shares no code with backend/a2a.py on purpose: a consult that works is then two
independent implementations agreeing about the wire format, which is the only
thing that actually demonstrates a protocol. A module talking to itself
demonstrates that it agrees with itself.
"""
import pytest
from fastapi.testclient import TestClient

from scripts.almanac_agent import ENTRIES, app

pytestmark = pytest.mark.unit
client = TestClient(app)


def _send(text):
    return client.post("/a2a", json={
        "jsonrpc": "2.0", "id": "1", "method": "message/send",
        "params": {"message": {"role": "user",
                               "parts": [{"kind": "text", "text": text}]}}}).json()


def test_the_card_is_discoverable_and_names_a_different_agent():
    card = client.get("/.well-known/agent.json").json()
    assert card["name"] == "The Almanac"
    assert card["name"] != "The Keeper"
    assert card["url"].endswith("/a2a")


def test_the_card_advertises_no_auth():
    """It holds nothing private, so it asks for nothing. The Keeper's own endpoint
    requires a token precisely because answering there means reading a person's
    memory. Two agents, two honest postures."""
    assert "securitySchemes" not in client.get("/.well-known/agent.json").json()


def test_it_answers_from_its_own_reference():
    body = _send("what is gouache?")
    text = body["result"]["parts"][0]["text"]
    assert "opaque" in text and body["id"] == "1"


def test_it_knows_things_the_keeper_does_not():
    """The point of a peer: consulting it has to be worth doing."""
    assert {"gouache", "gesso", "ultramarine"} <= set(ENTRIES)


def test_an_unknown_question_offers_what_it_does_hold():
    text = _send("what is the airspeed of a swallow?")["result"]["parts"][0]["text"]
    assert "keeps entries on" in text and "gouache" in text


def test_an_unsupported_method_is_refused():
    r = client.post("/a2a", json={"jsonrpc": "2.0", "id": "9",
                                  "method": "tasks/cancel", "params": {}}).json()
    assert r["error"]["code"] == -32601


def test_an_empty_message_is_refused():
    r = _send("")
    assert r["error"]["code"] == -32602


def test_malformed_json_does_not_crash_it():
    r = client.post("/a2a", content=b"{not json",
                    headers={"content-type": "application/json"}).json()
    assert r["error"]["code"] == -32700


def test_the_reply_is_shaped_for_the_keepers_reader():
    """backend/a2a.text_of has to be able to read this, or the two do not actually
    interoperate."""
    import a2a
    assert "opaque" in a2a.text_of(_send("gouache")["result"])
