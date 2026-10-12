"""
These are the Tier 3 tests for scheduling: does the real model reach for a JOB when
asked for recurring work, and for the right one?

Two confusions are likely and both are silent. A brief set up with remind_me comes
back every morning as the words "the morning brief", with no brief in it. And a
time written as "08:00" in the wrong day or zone fires when nobody asked. The unit
tier drives the tools directly and can see neither.
"""
import os
from datetime import datetime

import pytest

import compose
import jobs
import native_tools
import persona
import reminders
import server

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

TRIALS = 3


async def _ask(tmp_path, n, text):
    store = jobs.JobStore(tmp_path / f"j{n}.jsonl")
    rem = reminders.ReminderStore(tmp_path / f"r{n}.jsonl")
    tools = native_tools.NativeTools(rem, jobs=store)
    ctx = f"current time {datetime.now().isoformat(timespec='minutes')}"
    system = (persona.build_system_prompt("passive", "tidal", memory="", context=ctx)
              + "\n\n---\n\n" + server.TOOL_ADDENDUM)
    reply = await compose.tool_reply(system, text, providers=[tools],
                                     model=server.TOOL_MODEL, max_rounds=4,
                                     expect_one_of=server._expected_tools(text))
    return store, rem, reply


async def test_a_morning_brief_is_a_brief_job(tmp_path):
    for n in range(TRIALS):
        store, rem, reply = await _ask(tmp_path, n, "brief me every weekday at 8am")
        assert rem.pending() == [], f"trial {n}: set a reminder instead of a job"
        assert len(store.active()) == 1, f"trial {n}: no job; it said {reply!r}"
        (job,) = store.active()
        assert job.kind == "brief" and job.repeat == "weekdays", f"trial {n}: {job}"
        when = datetime.fromtimestamp(job.next_at)
        assert (when.hour, when.minute) == (8, 0) and when.weekday() < 5


async def test_recurring_work_is_a_task_job(tmp_path):
    for n in range(TRIALS):
        store, _, reply = await _ask(tmp_path, 100 + n,
                                     "every sunday morning, check whether the "
                                     "riverside gallery has posted its autumn dates")
        assert len(store.active()) == 1, f"trial {n}: no job; it said {reply!r}"
        (job,) = store.active()
        assert job.kind == "task" and job.repeat == "weekly", f"trial {n}: {job}"
        assert "gallery" in job.task.lower()
