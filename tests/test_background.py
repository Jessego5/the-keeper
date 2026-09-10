"""
These are the Tier 1 tests for background task registry + spawn_task tool (async
delegation).
"""
import pytest
import background
import native_tools
import reminders as reminders_mod

pytestmark = pytest.mark.unit


def test_add_is_running():
    bt = background.BackgroundTasks()
    t = bt.add("research watercolor paper")
    assert t.status == "running" and t.id
    assert [x.id for x in bt.running()] == [t.id]


def test_finish_moves_out_of_running():
    bt = background.BackgroundTasks()
    t = bt.add("dig into something")
    bt.finish(t.id, "here's what I found", ok=True)
    assert bt.tasks[t.id].status == "done"
    assert bt.tasks[t.id].result == "here's what I found"
    assert bt.running() == []


def test_finish_failure():
    bt = background.BackgroundTasks()
    t = bt.add("x")
    bt.finish(t.id, "(failed)", ok=False)
    assert bt.tasks[t.id].status == "failed"


# --- the spawn_task tool wires to a spawner callback --- #

@pytest.fixture
def nt_with_spawner(tmp_path):
    started = {}
    async def spawner(task):
        started["task"] = task
        return f"on it: {task}"
    nt = native_tools.NativeTools(
        reminders_mod.ReminderStore(tmp_path / "r.jsonl"),
        spawner=spawner)
    return nt, started


async def test_spawn_task_tool_calls_spawner(nt_with_spawner):
    nt, started = nt_with_spawner
    out = await nt.call("spawn_task", {"task": "look into cheap watercolor sets"})
    assert "on it" in out
    assert started["task"] == "look into cheap watercolor sets"


async def test_spawn_task_present_only_with_spawner(tmp_path):
    with_s = native_tools.NativeTools(
        reminders_mod.ReminderStore(tmp_path / "a.jsonl"), spawner=lambda t: None)
    without = native_tools.NativeTools(
        reminders_mod.ReminderStore(tmp_path / "b.jsonl"))
    names_with = {t["function"]["name"] for t in with_s.openai_tools()}
    names_without = {t["function"]["name"] for t in without.openai_tools()}
    assert "spawn_task" in names_with
    assert "spawn_task" not in names_without      # off without a spawner


async def test_spawn_task_graceful_without_spawner(tmp_path):
    nt = native_tools.NativeTools(reminders_mod.ReminderStore(tmp_path / "r.jsonl"))
    # no spawner -> the tool isn't offered, and calling it anyway is graceful
    assert "cannot work in the background" in await nt.call("spawn_task", {"task": "x"})
