"""planner.py — the Planning pattern. Decompose a goal into steps the Keeper can work.

Given something the person wants help moving toward, produce a short ordered list of
small, concrete steps. Kept deliberately minimal: 2-5 steps, each a single doable
thing, phrased as the companion helping — because the proactive loop will walk them
one at a time, taking a step or checking in.

The model is injected (compose.Generator), so planning is testable offline with a
stub and never hard-depends on a network call.
"""

from __future__ import annotations

import re
from typing import Callable

Generator = Callable[[str, str], str]

_PLAN_SYSTEM = """You help a companion break a person's goal into a SHORT plan it can \
help them move through over days. Given the goal, output 2 to 5 small, concrete steps \
— each ONE doable thing, in order, phrased plainly (e.g. "Set out the paints where \
they can see them", "Send Sam a short message"). No sub-lists, no numbering, no \
preamble. Prefer gentle first steps. If the goal is already a single clear action, \
one step is fine. Output only the steps, one per line."""

_MAX_STEPS = 5


def plan(goal: str, generate: Generator) -> list[str]:
    """Decompose `goal` into an ordered list of concrete steps (2-5, capped)."""
    goal = (goal or "").strip()
    if not goal:
        return []
    raw = (generate(_PLAN_SYSTEM, goal) or "").strip()
    steps: list[str] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*0123456789.) ").strip()
        if line:
            steps.append(line)
    return steps[:_MAX_STEPS]


_ADVANCE_SYSTEM = """You are a companion helping a person through one step of a plan. \
You are given the overall goal and the specific next step. Decide, in one short line, \
how to move it forward NOW — either something you can do or look up for them, or a \
gentle check-in that invites them to take it. Speak to them directly, in one or two \
sentences, warm and unhurried. Do not list the whole plan; just this step."""


def advance_prompt(goal_title: str, step_text: str) -> tuple:
    """Build the (system, user) prompt for working a single step. Returned as a pair
    so the caller can route it through compose (for the Keeper's voice) or a plain
    generate."""
    user = f"Goal: {goal_title}\nNext step: {step_text}"
    return _ADVANCE_SYSTEM, user


_DONE_RE = re.compile(r"\b(done|finished|completed|already did|took care)\b", re.I)


def reads_as_done(reply: str) -> bool:
    """Light heuristic: did the person's note indicate the step is complete? Used
    when a step is advanced with a recorded result."""
    return bool(_DONE_RE.search(reply or ""))
