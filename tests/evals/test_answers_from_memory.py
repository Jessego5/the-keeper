"""
These are the Tier 3 tests for answering from what it keeps: with a fact recalled
and the journal and goal tools on offer, does the real model answer from the fact?

The unit tier proves the recalled lines are said next to the request
(tests/test_server.py). It cannot prove the model uses them. Without that note it
answered "how is my painting going?" two times in three by offering to read the
journal, although memory held the change and no tool was called.
"""
import os

import pytest

import compose
import journal
import native_tools
import persona
import reminders
import server
import tasks

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

TRIALS = 3

MEM = ("- Started painting again.\n\nwhat has changed (the tide turned, was, then "
       "became):\n- was: Stopped painting in March.  →  now: Started painting "
       "again.")


async def test_a_question_memory_answers_is_answered_from_memory(tmp_path):
    system = (persona.build_system_prompt("passive", "turn", memory=MEM, context="")
              + "\n\n---\n\n" + server.TOOL_ADDENDUM)
    steer = "\n\n".join((server._now_note(), server._memory_note(MEM)))
    for n in range(TRIALS):
        tools = native_tools.NativeTools(
            reminders.ReminderStore(tmp_path / f"r{n}.jsonl"),
            goals=tasks.GoalStore(tmp_path / f"g{n}.jsonl"),
            journal=journal.Journal(tmp_path / f"j{n}.jsonl"))
        reply = await compose.tool_reply(system, "how is my painting going?",
                                         providers=[tools], model=server.TOOL_MODEL,
                                         steer=steer)
        low = reply.lower()
        assert "journal" not in low and "notes" not in low, f"trial {n}: {reply!r}"
        assert any(w in low for w in ("again", "back", "return", "resum")), \
            f"trial {n}: {reply!r}"
