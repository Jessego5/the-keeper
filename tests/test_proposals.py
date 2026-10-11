"""
These are the Tier 1 tests for proposals (proposals.py): what the person is shown,
that proposing never runs anything, and the two rules the module docstring says are
easy to erode. No network, no key.
"""
import json

import pytest

import proposals

pytestmark = pytest.mark.unit

SEND = proposals.Action(
    name="gmail__send_message",
    description="Send an email.",
    parameters={"type": "object",
                "properties": {"to": {"type": "string"},
                               "subject": {"type": "string"},
                               "body": {"type": "string"}},
                "required": ["to", "subject", "body"]})


@pytest.fixture
def store(tmp_path):
    return proposals.ProposalStore(tmp_path / "proposals.jsonl")


# --- what the person is shown --- #

def test_every_argument_is_shown_verbatim():
    """Rule 1. The approval shows the payload that will run, not a description."""
    text = proposals.render("gmail__send_message", {
        "to": "sam@example.com", "subject": "Thursday",
        "body": "Sam,\n\nI can't make Thursday. Friday?\n\nJ"})
    assert "send_message (gmail)" in text
    assert "to: sam@example.com" in text and "subject: Thursday" in text
    for line in ("Sam,", "I can't make Thursday. Friday?", "J"):
        assert line in text


def test_nothing_is_cut_short_however_long():
    body = "word " * 2000
    assert body.strip() in proposals.render("x__y", {"body": body}).replace(
        "\n    ", "\n")


def test_schema_order_wins_over_the_order_the_model_wrote():
    text = proposals.render("gmail__send_message",
                            {"body": "hi", "to": "a@b.c", "subject": "s"},
                            schema=SEND.parameters)
    assert text.index("to:") < text.index("subject:") < text.index("body:")


def test_structured_arguments_are_shown_as_json():
    text = proposals.render("cal__create_event",
                            {"attendees": ["a@b.c", "d@e.f"]})
    assert '["a@b.c", "d@e.f"]' in text


# --- proposing runs nothing --- #

async def test_proposing_records_and_asks_but_never_runs(store):
    asked = []

    async def notify(p):
        asked.append(p)

    proposer = proposals.Proposer(store, [SEND], notify)
    out = await proposer.call("gmail__send_message",
                              {"to": "sam@example.com", "subject": "s", "body": "b"})
    assert "NOT done" in out and "Do not say it is done" in out
    assert [p.status for p in store.items] == ["pending"]
    assert asked and asked[0].args["to"] == "sam@example.com"


async def test_an_incomplete_proposal_is_refused_before_anyone_is_asked(store):
    """Asking someone to approve a payload the tool will reject wastes the one
    decision they were asked to make."""
    asked = []

    async def notify(p):
        asked.append(p)

    out = await proposals.Proposer(store, [SEND], notify).call(
        "gmail__send_message", {"to": "sam@example.com"})
    assert "not proposed" in out and "subject" in out
    assert store.items == [] and asked == []


def test_the_model_is_told_it_proposes():
    proposer = proposals.Proposer(None, [SEND], None)
    (tool,) = proposer.openai_tools()
    assert tool["function"]["name"] == "gmail__send_message"
    assert tool["function"]["description"].startswith("PROPOSES")


async def test_the_args_kept_are_a_copy(store):
    """What the person approves must be what runs. A caller mutating the dict it
    passed after proposing must not change the stored payload."""
    args = {"to": "sam@example.com", "subject": "s", "body": "b"}
    p = store.add("gmail__send_message", args)
    args["to"] = "someone-else@example.com"
    assert p.args["to"] == "sam@example.com"


# --- at most once --- #

def test_approved_but_unfinished_is_failed_at_startup_not_retried(tmp_path):
    """Rule 2. The process died between the yes and the answer. Whether the email
    went is unknown; sending it again is the one answer that is certainly wrong."""
    path = tmp_path / "proposals.jsonl"
    store = proposals.ProposalStore(path)
    p = store.add("gmail__send_message", {"to": "a@b.c"})
    store.mark_approved(p, via="web")
    reopened = proposals.ProposalStore(path)
    (q,) = reopened.items
    assert q.status == "failed" and "not run again" in q.result
    # and that verdict is saved, not recomputed each boot
    assert json.loads(path.read_text().splitlines()[0])["status"] == "failed"


def test_a_failed_tool_is_never_reported_done(store):
    p = store.add("x__y", {})
    store.mark_approved(p, via="web")
    store.finish(p, "(tool error: 401 unauthorized)")
    assert p.status == "failed"
    q = store.add("x__y", {})
    store.mark_approved(q, via="web")
    store.finish(q, "Message sent. id 18c2")
    assert q.status == "done"


# --- expiry --- #

def test_an_unanswered_proposal_expires(store):
    p = store.add("x__y", {}, now=1000.0)
    assert store.pending(now=1000.0 + proposals.TTL_S - 1) == [p]
    assert store.pending(now=1000.0 + proposals.TTL_S + 1) == []
    assert p.status == "expired"


def test_a_decided_proposal_never_expires(store):
    p = store.add("x__y", {}, now=1000.0)
    store.decline(p, via="telegram", now=1001.0)
    store.expire_stale(now=1000.0 + proposals.TTL_S * 2)
    assert p.status == "declined" and p.decided_via == "telegram"


def test_public_carries_the_rendering(store):
    p = store.add("gmail__send_message", {"to": "a@b.c"})
    assert "to: a@b.c" in p.public()["rendered"]


async def test_the_approval_shows_the_account_it_will_act_as(store):
    """Fixed arguments are part of what will run, so rule 1 covers them: they are
    stored with the rest and shown, and the model cannot override them."""
    async def notify(p):
        return None

    send = proposals.Action(SEND.name, SEND.description, SEND.parameters,
                            fixed={"user_google_email": "me@example.com"})
    await proposals.Proposer(store, [send], notify).call(
        SEND.name, {"to": "sam@example.com", "subject": "s", "body": "b",
                    "user_google_email": "admin@else.com"})
    (p,) = store.items
    assert p.args["user_google_email"] == "me@example.com"
    assert "user_google_email: me@example.com" in p.public()["rendered"]
