"""
These are the Tier 1 tests for scheduled jobs (jobs.py) and the tools that set them
up: the two rules about time, and that only the person's own turn can schedule.
No network, no key.
"""
from datetime import datetime

import pytest

import jobs
import native_tools
import reminders

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    return jobs.JobStore(tmp_path / "jobs.jsonl")


def _at(h, m=0, day=12):
    return datetime(2026, 10, day, h, m).timestamp()


def test_a_due_job_is_rearmed_before_it_runs(store):
    """Rule 1. If the work dies mid-run, the next occurrence is already set and
    this one is not retried in a loop."""
    j = store.add("check the gallery's dates", _at(8), repeat="daily", now=_at(7))
    assert store.due(now=_at(8, 1)) == [j]
    assert store.claim(j, now=_at(8, 1)) is True
    assert j.next_at == _at(8, day=13)
    assert store.due(now=_at(8, 2)) == []


def test_a_badly_late_run_is_skipped_not_run(store):
    """Rule 2. A morning brief composed at nine at night describes a day that is
    over. It is skipped, and the next morning's is still on."""
    j = store.add("brief", _at(8), repeat="daily", kind="brief", now=_at(7))
    assert store.claim(j, now=_at(21)) is False
    assert j.last_status == "skipped" and j.next_at == _at(8, day=13)


def test_a_one_shot_job_runs_once(store):
    j = store.add("look into flights to Lisbon", _at(9), now=_at(8))
    assert store.claim(j, now=_at(9)) is True
    assert not j.enabled and store.due(now=_at(10)) == []


def test_every_weekday_at_8_said_at_noon_starts_tomorrow(store):
    """The roll native_tools does for recurring reminders, for the same reason:
    stored as given, today's already-past 8am would fire at once."""
    j = store.add("brief", _at(8), repeat="weekdays", kind="brief", now=_at(12))
    assert j.next_at == _at(8, day=13)


def test_a_job_survives_a_restart(tmp_path):
    jobs.JobStore(tmp_path / "jobs.jsonl").add("x", _at(8), repeat="daily",
                                               now=_at(7))
    (j,) = jobs.JobStore(tmp_path / "jobs.jsonl").active()
    assert j.task == "x" and j.repeat == "daily"


def test_cancel_by_a_word(store):
    store.add("check the gallery's dates", _at(8), repeat="weekly", now=_at(7))
    assert store.cancel("gallery").task == "check the gallery's dates"
    assert store.active() == [] and store.cancel("gallery") is None


def test_has_brief_ignores_cancelled_ones(store):
    store.add("the morning brief", _at(8), repeat="daily", kind="brief", now=_at(7))
    assert store.has_brief()
    store.cancel("brief")
    assert not store.has_brief()


# --- the tools --- #

def _tools(tmp_path, store=None):
    return native_tools.NativeTools(
        reminders.ReminderStore(tmp_path / "r.jsonl"), jobs=store)


async def test_asking_for_a_brief_schedules_one(tmp_path, store):
    t = _tools(tmp_path, store)
    out = await t.call("schedule_job", {"task": "", "kind": "brief",
                                        "first_iso": "2099-01-05T08:00:00",
                                        "repeat": "weekdays"})
    assert out.startswith("scheduled:") and "then weekdays" in out
    (j,) = store.active()
    assert (j.kind, j.task) == ("brief", "the morning brief")


async def test_a_job_needs_a_time_it_can_read(tmp_path, store):
    out = await _tools(tmp_path, store).call(
        "schedule_job", {"task": "x", "first_iso": "tomorrow morning"})
    assert "ISO" in out and store.active() == []


async def test_list_and_cancel(tmp_path, store):
    t = _tools(tmp_path, store)
    assert await t.call("list_jobs", {}) == "running nothing on a schedule."
    await t.call("schedule_job", {"task": "check the gallery", "repeat": "weekly",
                                  "first_iso": "2099-01-05T08:00:00"})
    assert "check the gallery" in await t.call("list_jobs", {})
    assert (await t.call("cancel_job", {"key": "gallery"})).startswith("stopped")


def test_only_the_persons_own_turn_can_schedule(tmp_path):
    """Sub-agents and A2A peers build NativeTools without a job store."""
    names = {d["function"]["name"] for d in _tools(tmp_path).openai_tools()}
    assert not names & {"schedule_job", "list_jobs", "cancel_job"}
