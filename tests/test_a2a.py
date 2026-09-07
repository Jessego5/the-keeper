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


# --- outbound safety: consult_peer takes a URL the MODEL chose --- #

def test_a_public_peer_is_reachable():
    assert a2a.check_peer_url("https://example.com") == "https://example.com"


@pytest.mark.parametrize("url,what", [
    ("http://169.254.169.254/latest/meta-data/", "cloud metadata"),
    ("http://127.0.0.1:5432", "a local database"),
    ("http://localhost:8790", "loopback by name"),
    ("http://192.168.1.10", "the LAN"),
    ("http://[::1]:8790", "ipv6 loopback"),
    ("http://10.0.0.5", "private range"),
])
def test_non_public_addresses_are_refused(url, what):
    """The Keeper reads web pages, so a prompt injection in one can name a URL for
    it to consult. Unchecked that is a READABLE SSRF: the response comes back into
    the conversation. These are what such an injection reaches for."""
    with pytest.raises(a2a.PeerBlocked):
        a2a.check_peer_url(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com",
                                 "gopher://x", "", "not a url"])
def test_only_http_schemes_are_allowed(url):
    with pytest.raises(a2a.PeerBlocked):
        a2a.check_peer_url(url)


def test_the_operator_can_permit_a_specific_origin(monkeypatch):
    """Loopback is how the demo consults the Keeper itself, so it has to be
    permitted deliberately rather than by default."""
    monkeypatch.setenv(a2a.ALLOW_ENV, "http://localhost:8790")
    assert a2a.check_peer_url("http://localhost:8790")


def test_permission_does_not_leak_to_other_local_ports(monkeypatch):
    """Allowing one origin must not open the rest of the machine."""
    monkeypatch.setenv(a2a.ALLOW_ENV, "http://localhost:8790")
    with pytest.raises(a2a.PeerBlocked):
        a2a.check_peer_url("http://127.0.0.1:5432")


def test_a_hostile_card_cannot_redirect_to_an_internal_endpoint():
    """send_message takes its endpoint from the PEER'S OWN card, so a permitted
    public peer could otherwise hand back an internal address to post to."""
    with pytest.raises(a2a.PeerBlocked):
        a2a.check_peer_url("http://169.254.169.254/a2a")
