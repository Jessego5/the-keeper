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


def test_steps_default_to_person(store):
    g = store.add("goal", ["do a thing"])
    assert g.steps[0].actor == "person"


def test_add_parses_actor_labels(store):
    g = store.add("reconnect with Sam", [
        "[keeper] Look up when we last spoke",
        "[person] Call Sam",
        "[me] Draft a short message",      # 'me' (the Keeper) -> keeper
        "[you] Send it",                   # 'you' (the person) -> person
    ])
    actors = [(s.actor, s.text) for s in g.steps]
    assert actors == [
        ("keeper", "Look up when we last spoke"),
        ("person", "Call Sam"),
        ("keeper", "Draft a short message"),
        ("person", "Send it"),
    ]
    assert "[" not in g.steps[0].text        # the label never leaks into the text


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


def test_get_matches_a_shared_word_stem(store):
    # Regression: the model keys off the PERSON's words, not the goal's. "i set out
    # my paints" arrives as key="paints" against "get back to painting"; a plain
    # substring match missed it and the step was silently never advanced.
    g = store.add("get back to painting", ["set out the paints"])
    assert store.get("paints") is g
    assert store.get("painted") is g


def test_get_ignores_filler_words(store):
    # a stem match must not let "get back to ..." be found by its scaffolding
    store.add("get back to painting", ["x"])
    assert store.get("back") is None
    assert store.get("get") is None


def test_get_prefers_an_active_goal_over_a_finished_one(store):
    store.add("painting", ["x"])
    store.complete("painting")
    current = store.add("start painting again", ["y"])
    assert store.get("paints") is current


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


def _reflect_gen(system, user):
    """A model that first drafts a bad (too-big) plan, and on critique says so, then
    re-plans into small steps."""
    if "review a short plan" in system:                 # critique pass
        return ("Break the first step down"
                if "one giant step" in user.lower() else "GOOD")
    if "fix it:" in user:                               # re-plan with feedback
        return "Tiny step one\nTiny step two\nTiny step three"
    return "One giant step to do everything"            # first draft


def test_plan_reflect_revises_a_weak_plan():
    steps = planner.plan("get organized", _reflect_gen, reflect=True)
    assert steps == ["Tiny step one", "Tiny step two", "Tiny step three"]


def test_plan_reflect_keeps_a_good_plan():
    def good_gen(system, user):
        return "GOOD" if "review a short plan" in system else "Step one\nStep two"
    steps = planner.plan("simple goal", good_gen, reflect=True)
    assert steps == ["Step one", "Step two"]            # critique said GOOD, no revise


def test_critique_plan_reports_empty():
    assert "empty" in planner.critique_plan("g", [], lambda s, u: "GOOD")


def test_plan_without_reflect_is_single_pass():
    calls = []
    def gen(system, user):
        calls.append(system)
        return "a\nb"
    planner.plan("g", gen, reflect=False)
    assert len(calls) == 1                              # no critique/revise calls


def test_advance_prompt_includes_goal_and_step():
    system, user = planner.advance_prompt("reach Sam", "send a short message")
    assert "reach Sam" in user and "send a short message" in user
    assert "companion" in system.lower()


def test_reads_as_done():
    assert planner.reads_as_done("yeah I already did that")
    assert not planner.reads_as_done("not yet, maybe tomorrow")


# --- goal tools via NativeTools (async) --- #

import journal as journal_mod
import native_tools
import reminders as reminders_mod


def _plan_gen(system, user):
    return "Set out the paints\nMake one small mark"


@pytest.fixture
def nt(tmp_path):
    rem = reminders_mod.ReminderStore(tmp_path / "r.jsonl")
    goals = tasks.GoalStore(tmp_path / "g.jsonl")
    jrnl = journal_mod.Journal(tmp_path / "kept.jsonl")
    return native_tools.NativeTools(rem, goals=goals, planner_generate=_plan_gen,
                                    journal=jrnl)


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


async def test_advance_goal_marks_step_and_moves_on(nt):
    await nt.call("set_goal", {"title": "get back to painting"})  # 2 steps (fake plan)
    out = await nt.call("advance_goal", {"key": "painting", "note": "did it"})
    assert "marked done" in out and "1/2" in out and "next:" in out
    g = nt.goals.active()[0]
    assert g.steps[0].done and g.steps[0].note == "did it"


async def test_advance_goal_completes_on_last_step(nt):
    await nt.call("set_goal", {"title": "reach Sam"})       # 2 steps
    await nt.call("advance_goal", {"key": "reach"})
    out = await nt.call("advance_goal", {"key": "reach"})   # last step
    assert "completes" in out
    assert nt.goals.active() == []                          # goal finished


async def test_advance_goal_no_match(nt):
    assert "no matching goal" in await nt.call("advance_goal", {"key": "nope"})


async def test_advance_goal_matches_the_persons_own_words(nt):
    # Regression, seen live: "i set out my paints" reached advance_goal as
    # key="paints" against the goal "get back to painting" and returned
    # "no matching goal." — the goal never advanced and nothing surfaced the miss.
    await nt.call("set_goal", {"title": "get back to painting"})
    out = await nt.call("advance_goal", {"key": "paints", "note": "set them out"})
    assert "no matching goal" not in out
    assert "marked done" in out
    assert nt.goals.active()[0].steps[0].done


async def test_goal_tools_graceful_without_store(tmp_path):
    # a NativeTools with reminders only (no goal store) must not crash on goal calls
    bare = native_tools.NativeTools(
        reminders_mod.ReminderStore(tmp_path / "r.jsonl"))
    assert "cannot take on goals" in await bare.call("set_goal", {"title": "x"})
    assert "not holding" in await bare.call("list_goals", {})
