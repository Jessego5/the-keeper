"""Tier 1 — reminders store + native action tools. No key needed."""
import time
import pytest
import native_tools
import reminders

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    return reminders.ReminderStore(tmp_path / "reminders.jsonl")


def test_add_and_pending_sorted(store):
    now = time.time()
    store.add("later thing", now + 200)
    store.add("sooner thing", now + 100)
    pend = store.pending()
    assert [r.text for r in pend] == ["sooner thing", "later thing"]


def test_due_returns_only_past_undelivered(store):
    now = time.time()
    past = store.add("past due", now - 10)
    store.add("future", now + 999)
    due = store.due(now)
    assert [r.id for r in due] == [past.id]


def test_mark_delivered_then_not_due(store):
    now = time.time()
    r = store.add("x", now - 10)
    store.mark_delivered(r.id)
    assert store.due(now) == []


def test_complete_by_text_and_persists(tmp_path):
    p = tmp_path / "r.jsonl"
    s1 = reminders.ReminderStore(p)
    s1.add("call the dentist", time.time() + 100)
    done = s1.complete("dentist")
    assert done is not None and done.done
    s2 = reminders.ReminderStore(p)                       # reload
    assert s2.pending() == []                             # completed -> not pending


# --- native tools (async) --- #

async def test_native_tools_expose_three(store):
    nt = native_tools.NativeTools(store)
    names = {t["function"]["name"] for t in nt.openai_tools()}
    assert names == {"remind_me", "list_reminders", "complete_reminder"}


async def test_remind_me_stores_with_iso(store):
    nt = native_tools.NativeTools(store)
    out = await nt.call("remind_me", {"text": "water the plants",
                                      "due_iso": "2027-01-01T09:00:00"})
    assert "kept" in out and len(store.pending()) == 1


async def test_bad_iso_is_graceful(store):
    nt = native_tools.NativeTools(store)
    out = await nt.call("remind_me", {"text": "x", "due_iso": "not a date"})
    assert "could not read" in out and store.pending() == []


async def test_list_and_complete(store):
    nt = native_tools.NativeTools(store)
    await nt.call("remind_me", {"text": "reply to Sam",
                                "due_iso": "2027-01-01T09:00:00"})
    assert "Sam" in await nt.call("list_reminders", {})
    assert "done" in await nt.call("complete_reminder", {"key": "Sam"})
