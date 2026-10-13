"""
These are the Tier 3 tests for ways: the real model deciding what an edit teaches,
when a sentence is a standing instruction, and whether a learned way is followed.

Each fails quietly in the direction that matters. A learner that turns a corrected
date into a habit makes every later draft wrong in a new way. A model that keeps
"sign this one J" as a rule signs everything J forever. And a way handed to a draft
that does not follow it is a dashboard entry and nothing more.
"""
import os
import re
from datetime import datetime

import pytest

import compose
import native_tools
import persona
import proposals
import reminders
import server
import ways

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

TRIALS = 3


@pytest.fixture(scope="module")
def gen():
    g, fast = compose.make_generator()
    assert fast is not None, "expected a live backend with a key"
    return fast


def _body(sign):
    return f"Hi Sam,\n\nI can't make Thursday, but Saturday works.\n\n{sign}"


def test_a_changed_sign_off_is_a_habit(gen, tmp_path):
    for n in range(TRIALS):
        store = ways.WayStore(tmp_path / f"a{n}.jsonl")
        w = ways.learn_from_edit("mail__send", {"body": _body("Best regards,\nJordan")},
                                 {"body": _body("J")}, gen, store)
        assert w is not None, f"trial {n}: learned nothing"
        # How the task is named varies ("emails to Sam", "sign-offs"); what must
        # hold is that the way says to sign J.
        how = w.how.lower()
        assert re.search(r"\bj\b", how) or "initial" in how, f"trial {n}: {w}"
        assert "content shows" not in w.about, f"trial {n}: copied the template"


def test_a_corrected_day_is_not_a_habit(gen, tmp_path):
    for n in range(TRIALS):
        store = ways.WayStore(tmp_path / f"b{n}.jsonl")
        before = {"body": _body("J")}
        after = {"body": _body("J").replace("Saturday", "Sunday")}
        w = ways.learn_from_edit("mail__send", before, after, gen, store)
        assert w is None, f"trial {n}: made a habit of a one-off fix: {w}"


async def _turn(tmp_path, n, text, providers_extra=()):
    store = ways.WayStore(tmp_path / f"w{n}.jsonl")
    tools = native_tools.NativeTools(reminders.ReminderStore(tmp_path / f"r{n}.jsonl"),
                                     ways=store)
    ctx = f"current time {datetime.now().isoformat(timespec='minutes')}"
    system = (persona.build_system_prompt("passive", "tidal", memory="", context=ctx)
              + "\n\n---\n\n" + server.TOOL_ADDENDUM)
    await compose.tool_reply(system, text, providers=[tools, *providers_extra],
                             model=server.TOOL_MODEL, max_rounds=4,
                             expect_one_of=server._expected_tools(text))
    return store


async def test_from_now_on_is_kept(tmp_path):
    for n in range(TRIALS):
        store = await _turn(tmp_path, n, "from now on, sign my emails just 'J'")
        assert [w.how for w in store.active()], f"trial {n}: kept nothing"


async def test_this_one_is_not(tmp_path):
    for n in range(TRIALS):
        store = await _turn(tmp_path, 100 + n,
                            "sign this one 'J', just this once. it's for sam")
        assert store.active() == [], f"trial {n}: kept {store.active()}"


async def test_a_learned_way_is_followed_in_a_draft(tmp_path):
    send = proposals.Action(
        "mail__send_message", "Send an email.",
        {"type": "object", "properties": {"to": {"type": "string"},
                                          "subject": {"type": "string"},
                                          "body": {"type": "string"}},
         "required": ["to", "subject", "body"]})
    for n in range(TRIALS):
        pstore = proposals.ProposalStore(tmp_path / f"p{n}.jsonl")

        async def notify(p):
            return None

        wstore = ways.WayStore(tmp_path / f"x{n}.jsonl")
        wstore.add("emails to Sam", "sign off with just 'J', never the full name")
        system = (persona.build_system_prompt("passive", "tidal", memory="",
                                              context="")
                  + "\n\n---\n\n" + server.TOOL_ADDENDUM + "\n\n"
                  + ways.as_guidance(wstore.for_task("email Sam")))
        await compose.tool_reply(
            system, "email sam@example.com that thursday is off but saturday works",
            providers=[proposals.Proposer(pstore, [send], notify)],
            model=server.TOOL_MODEL, max_rounds=4)
        assert pstore.items, f"trial {n}: proposed nothing"
        p = pstore.items[-1]
        tail = p.args["body"].strip().splitlines()[-1].strip()
        assert tail.rstrip(".") == "J", f"trial {n}: signed off {tail!r}"
