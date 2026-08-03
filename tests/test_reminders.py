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


# --- recurrence --- #

@pytest.mark.parametrize("phrase,canon", [
    ("daily", "daily"), ("every day", "daily"), ("Weekly", "weekly"),
    ("weekdays", "weekdays"), ("every 3 hours", "every 3h"),
    ("every 30 minutes", "every 30m"), ("hourly", "every 1h"),
    ("someday", None), ("", None), (None, None),
])
def test_normalize_repeat(phrase, canon):
    assert reminders.normalize_repeat(phrase) == canon


def test_next_occurrence_interval_skips_catchup():
    now = 1_000_000.0
    due = now - 10 * 3600            # 10 hours in the past, hourly
    nxt = reminders.next_occurrence(due, "every 1h", now)
    assert now < nxt <= now + 3600  # jumps to the next FUTURE slot, no storm


def test_next_occurrence_weekdays_skips_weekend():
    from datetime import datetime
    # a Friday noon -> next occurrence must be Monday (weekday < 5)
    fri = datetime(2026, 8, 7, 12, 0).timestamp()          # 2026-08-07 is a Friday
    nxt = reminders.next_occurrence(fri, "weekdays", fri + 60)
    assert datetime.fromtimestamp(nxt).weekday() == 0      # Monday


def test_one_shot_returns_none():
    assert reminders.next_occurrence(time.time(), None, time.time()) is None


def test_recurring_rearms_on_delivery(store):
    now = time.time()
    r = store.add("water the plants", now - 5, repeat="daily")
    assert store.due(now)                       # due now
    store.mark_delivered(r.id, now=now)
    assert store.due(now) == []                 # not due again immediately
    again = store.items[0]
    assert again.repeat == "daily" and again.fired_count == 1
    assert again.due_at > now and not again.delivered   # re-armed for tomorrow


async def test_remind_me_recurring(store):
    nt = native_tools.NativeTools(store)
    out = await nt.call("remind_me", {"text": "water the plants",
                                      "due_iso": "2027-01-01T09:00:00",
                                      "repeat": "daily"})
    assert "daily" in out
    assert store.pending()[0].repeat == "daily"
