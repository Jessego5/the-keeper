"""Tier 1 — goal store + planner (tasks.py, planner.py). No key, no clock."""
import pytest
import planner
import tasks

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    return tasks.GoalStore(tmp_path / "goals.jsonl")


# --- GoalStore --- #

def test_add_and_next_step(store):
    g = store.add("get back to painting", ["set out the paints", "make one mark"])
    assert g.next_step().text == "set out the paints"
    assert g.progress() == (0, 2)


def test_advance_completes_steps_then_goal(store):
    g = store.add("reach Sam", ["draft a message", "send it"])
    s1 = store.advance(g, note="drafted it")
    assert s1.text == "draft a message" and s1.done and s1.note == "drafted it"
    assert g.status == "active" and g.progress() == (1, 2)
    store.advance(g, note="sent")
    assert g.status == "done" and g.next_step() is None    # auto-completed


def test_advance_schedules_next_check(store):
    now = 1_000_000.0
    g = store.add("tidy the studio", ["clear the table", "sweep"])
    store.advance(g, interval_s=3600, now=now)
    assert g.next_check_at == now + 3600


def test_due_returns_overdue_active_goal(store):
    now = 1_000_000.0
    a = store.add("A", ["a1"])
    b = store.add("B", ["b1"])
    a.next_check_at = now - 100     # overdue
    b.next_check_at = now + 100     # not yet
    store._save()
    due = store.due(now)
    assert due is not None and due.id == a.id


def test_due_skips_finished_goals(store):
    now = 1_000_000.0
    g = store.add("one-step", ["do it"])
    store.advance(g, now=now)       # completes the only step -> goal done
    assert store.due(now + 10_000) is None


def test_get_by_id_and_title(store):
    g = store.add("call the gallery about the show", ["find the number"])
    assert store.get(g.id) is g
    assert store.get("gallery") is g       # substring of title
    assert store.get("nonexistent") is None


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "goals.jsonl"
    s1 = tasks.GoalStore(p)
    g = s1.add("goal", ["step one", "step two"])
    s1.advance(g, note="did one")
    s2 = tasks.GoalStore(p)                 # reload from disk
    assert len(s2.goals) == 1
    g2 = s2.goals[0]
    assert g2.progress() == (1, 2) and g2.steps[0].note == "did one"


def test_complete_by_title(store):
    store.add("a lingering goal", ["x"])
    done = store.complete("lingering")
    assert done is not None and done.status == "done"


# --- planner --- #

def test_plan_parses_and_caps():
    def gen(system, user):
        return ("- Set out the paints\n"
                "- Make one small mark\n"
                "3. Step three\n4. Step four\n5. Step five\n6. Step six\n7. Seven")
    steps = planner.plan("get back to painting", gen)
    assert steps[0] == "Set out the paints"
    assert steps[1] == "Make one small mark"
    assert len(steps) == 5                  # capped at _MAX_STEPS


def test_plan_empty_goal():
    assert planner.plan("", lambda s, u: "whatever") == []


def test_advance_prompt_includes_goal_and_step():
    system, user = planner.advance_prompt("reach Sam", "send a short message")
    assert "reach Sam" in user and "send a short message" in user
    assert "companion" in system.lower()


def test_reads_as_done():
    assert planner.reads_as_done("yeah I already did that")
    assert not planner.reads_as_done("not yet, maybe tomorrow")


# --- goal tools via NativeTools (async) --- #

import native_tools
import reminders as reminders_mod


def _plan_gen(system, user):
    return "Set out the paints\nMake one small mark"


@pytest.fixture
def nt(tmp_path):
    rem = reminders_mod.ReminderStore(tmp_path / "r.jsonl")
    goals = tasks.GoalStore(tmp_path / "g.jsonl")
    return native_tools.NativeTools(rem, goals=goals, planner_generate=_plan_gen)


async def test_set_goal_plans_and_stores(nt):
    out = await nt.call("set_goal", {"title": "get back to painting"})
    assert "taken on" in out and "Set out the paints" in out
    assert len(nt.goals.active()) == 1
    assert nt.goals.active()[0].progress() == (0, 2)


async def test_list_goals_shows_progress(nt):
    await nt.call("set_goal", {"title": "reach Sam"})
    out = await nt.call("list_goals", {})
    assert "reach Sam" in out and "0/2" in out


async def test_complete_goal(nt):
    await nt.call("set_goal", {"title": "tidy the studio"})
    out = await nt.call("complete_goal", {"key": "studio"})
    assert "set down" in out
    assert nt.goals.active() == []


async def test_goal_tools_graceful_without_store(tmp_path):
    # a NativeTools with reminders only (no goal store) must not crash on goal calls
    bare = native_tools.NativeTools(
        reminders_mod.ReminderStore(tmp_path / "r.jsonl"))
    assert "cannot take on goals" in await bare.call("set_goal", {"title": "x"})
    assert "not holding" in await bare.call("list_goals", {})
