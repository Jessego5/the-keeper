"""
These are the Tier 1 tests for ways (ways.py): what is kept, what replaces what,
which work a way is handed to, and what an edit may teach. The learner's model is a
fake here; tests/evals/test_ways.py is its real-model twin.
"""
import pytest

import native_tools
import proposals
import reminders
import ways

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    return ways.WayStore(tmp_path / "ways.jsonl")


def test_a_way_is_kept_and_survives_a_restart(tmp_path):
    ways.WayStore(tmp_path / "w.jsonl").add("emails to Sam", "sign off as J")
    (w,) = ways.WayStore(tmp_path / "w.jsonl").active()
    assert w.line() == "emails to Sam: sign off as J" and w.source == "said"


def test_saying_it_twice_keeps_one(store):
    a = store.add("any email", "keep it under five sentences")
    b = store.add("Any email", "Keep it under five sentences")
    assert a is b and len(store.active()) == 1


def test_a_newer_way_replaces_the_one_it_contradicts(store):
    """'sign as J' then 'sign as Jess' leaves one rule, and the history."""
    old = store.add("emails to Sam", "sign off as J")
    new = store.add("emails to Sam", "sign off as Jess")
    assert store.active() == [new]
    assert old.active is False and old.replaced_by == new.id


def test_different_ways_about_the_same_thing_both_stand(store):
    store.add("emails to Sam", "sign off as J")
    store.add("emails to Sam", "keep it short")
    assert len(store.active()) == 2


def test_a_way_reaches_the_work_it_is_about(store):
    sam = store.add("emails to Sam", "sign off as J")
    anymail = store.add("any email", "no exclamation marks")
    store.add("calendar events", "never before 10am")
    found = store.for_task("email Sam that I can't do Thursday")
    assert found[0] is sam and anymail in found
    assert all("calendar" not in w.about for w in found)


def test_writing_and_write_are_the_same_word(store):
    """Regression from the live demo: the stemmer left "writing" as "writ" and
    "write" as "write", so a way about writing never reached a note being written."""
    w = store.add("any writing", "sign with 'J'")
    assert store.for_task("write a short note to sam") == [w]


def test_a_way_matches_on_what_it_is_about_not_on_its_wording(store):
    """'keep it short' says nothing about which tasks it governs."""
    store.add("emails to Sam", "keep it short")
    assert store.for_task("keep this note short") == []


def test_a_forgotten_way_is_not_applied(store):
    store.add("emails to Sam", "sign off as J")
    assert store.forget("sign off").how == "sign off as J"
    assert store.for_task("email Sam") == []


def test_guidance_says_the_request_wins(store):
    """A way is a default, not an order: 'sign this one with my full name' must
    beat 'sign off as J'."""
    text = ways.as_guidance([store.add("emails to Sam", "sign off as J")])
    assert "emails to Sam: sign off as J" in text and "the request wins" in text


# --- learning from an edit --- #

BEFORE = {"to": "sam@example.com", "body": "Hi Sam,\n\nCan't do Thursday.\n\nBest,\nJess"}
AFTER = {"to": "sam@example.com", "body": "Hi Sam,\n\nCan't do Thursday.\n\nJ"}


def test_an_edit_that_shows_a_habit_becomes_a_way(store):
    seen = {}

    def gen(system, user):
        seen["user"] = user
        return "ABOUT: emails to Sam | HOW: sign off as J, not 'Best, Jess'"

    w = ways.learn_from_edit("mail__send", BEFORE, AFTER, gen, store)
    assert w.line() == "emails to Sam: sign off as J, not 'Best, Jess'"
    assert w.source == "edit" and "Best,\nJess" in w.evidence
    assert "Field body" in seen["user"] and "Field to" not in seen["user"]


def test_a_one_off_fix_teaches_nothing(store):
    """Most edits are a corrected date or a typo, and should be."""
    assert ways.learn_from_edit("mail__send", BEFORE, AFTER,
                                lambda s, u: "NONE", store) is None
    assert store.active() == []


@pytest.mark.parametrize("verdict", [
    "", "Sure! They like short emails.", "ABOUT: emails |", "HOW: x | ABOUT: y",
])
def test_an_answer_out_of_shape_teaches_nothing(store, verdict):
    assert ways.parse_learned(verdict, store) is None


def test_a_failing_learner_does_not_matter(store):
    def boom(system, user):
        raise RuntimeError("429")
    assert ways.learn_from_edit("x__y", BEFORE, AFTER, boom, store) is None


def test_an_unchanged_draft_is_not_sent_to_the_learner(store):
    def never(system, user):
        raise AssertionError("asked about an edit that did not happen")
    assert ways.learn_from_edit("x__y", BEFORE, dict(BEFORE), never, store) is None


# --- editing a proposal before approving it --- #

@pytest.fixture
def pstore(tmp_path):
    return proposals.ProposalStore(tmp_path / "p.jsonl")


def test_an_edit_is_applied_and_the_draft_kept(pstore):
    p = pstore.add("mail__send", BEFORE)
    assert pstore.apply_edit(p, {"body": AFTER["body"]}) is None
    assert p.args == AFTER and p.original_args == BEFORE
    assert "\n    J" in p.public()["rendered"]


def test_an_edit_cannot_add_a_field(pstore):
    p = pstore.add("mail__send", BEFORE)
    assert "cannot add" in pstore.apply_edit(p, {"bcc": "someone@else.com"})
    assert p.args == BEFORE


def test_an_edit_cannot_switch_the_account(pstore):
    """The config chooses who it acts as, and an edit is no way around that."""
    p = pstore.add("mail__send", {**BEFORE, "user_google_email": "me@example.com"},
                   fixed_keys=["user_google_email"])
    out = pstore.apply_edit(p, {"user_google_email": "admin@else.com"})
    assert "not editable" in out and p.args["user_google_email"] == "me@example.com"


def test_an_unchanged_edit_leaves_no_draft_to_learn_from(pstore):
    p = pstore.add("mail__send", BEFORE)
    assert pstore.apply_edit(p, dict(BEFORE)) is None and p.original_args is None


# --- the tools --- #

async def test_ways_are_taught_and_forgotten_by_name(tmp_path, store):
    t = native_tools.NativeTools(reminders.ReminderStore(tmp_path / "r.jsonl"),
                                 ways=store)
    out = await t.call("keep_way", {"about": "any email", "how": "no emoji"})
    assert out == "kept: any email: no emoji"
    assert "any email: no emoji (learned from said" in await t.call("list_ways", {})
    assert (await t.call("forget_way", {"key": "emoji"})).startswith("forgotten")


def test_a_sub_agent_cannot_teach_itself(tmp_path):
    names = {d["function"]["name"] for d in native_tools.NativeTools(
        reminders.ReminderStore(tmp_path / "r.jsonl")).openai_tools()}
    assert not names & {"keep_way", "list_ways", "forget_way"}
