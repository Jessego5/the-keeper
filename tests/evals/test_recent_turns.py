"""
These are the Tier 3 tests for the conversation window: does the real model answer
from the last few turns when the question only makes sense because of them?

The unit tier proves the turns are handed over (tests/test_server.py). It cannot
prove the model reads them rather than answering the bare message, which here would
mean inventing a time, the exact failure test_never_invents_events was written for.
"""
import os

import pytest

import compose
import persona
import server

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

TRIALS = 3

PRIOR = [
    {"role": "assistant", "content": "Sam wrote back about Thursday. He can't do "
                                     "Thursday, but Saturday at 7 works for him."},
]


async def test_a_follow_up_is_answered_from_the_thread():
    system = (persona.build_system_prompt("passive", "tidal", memory="", context="")
              + "\n\n---\n\n" + server.TOOL_ADDENDUM)
    for n in range(TRIALS):
        reply = await compose.tool_reply(system, "what time did he say?",
                                         providers=[], model=server.TOOL_MODEL,
                                         prior=PRIOR)
        assert "7" in reply or "seven" in reply.lower(), f"trial {n}: {reply!r}"
