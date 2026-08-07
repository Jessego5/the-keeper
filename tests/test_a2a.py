"""Tier 1 — the A2A protocol helpers (a2a.py). Pure, no network."""
import pytest
import a2a

pytestmark = pytest.mark.unit


def test_agent_card_shape():
    card = a2a.build_agent_card("http://localhost:8790/")
    assert card["name"] == "The Keeper"
    assert card["url"] == "http://localhost:8790/a2a"      # trailing slash trimmed
    assert card["protocolVersion"] == a2a.PROTOCOL_VERSION
    assert {s["id"] for s in card["skills"]} >= {"companion", "research", "analyze"}
    assert card["capabilities"]["streaming"] is False


def test_text_of_joins_text_parts():
    msg = {"parts": [{"kind": "text", "text": "hello"},
                     {"kind": "text", "text": "there"},
                     {"kind": "data", "text": "IGNORED"}]}
    assert a2a.text_of(msg) == "hello\nthere"


def test_text_of_empty():
    assert a2a.text_of({}) == "" and a2a.text_of({"parts": []}) == ""


def test_make_message_structure():
    m = a2a.make_message("the tide turns", role="agent")
    assert m["role"] == "agent" and m["kind"] == "message"
    assert m["parts"][0] == {"kind": "text", "text": "the tide turns"}
    assert m["messageId"]                                  # has an id


def test_rpc_result_and_error_envelopes():
    r = a2a.rpc_result("42", a2a.make_message("hi"))
    assert r["jsonrpc"] == "2.0" and r["id"] == "42" and "result" in r
    e = a2a.rpc_error("42", -32601, "nope")
    assert e["error"] == {"code": -32601, "message": "nope"}


def test_round_trips_through_envelope():
    # what the server builds, the client can read back
    reply = a2a.rpc_result("1", a2a.make_message("Canson and Strathmore.", "agent"))
    assert a2a.text_of(reply["result"]) == "Canson and Strathmore."
