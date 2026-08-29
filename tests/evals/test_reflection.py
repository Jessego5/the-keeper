"""Tier 3 — reflection, on the real model. Run explicitly:

    .venv/bin/pytest tests/evals -m eval

Why this exists: the insight prompt's one-shot example was "She keeps circling back
to what she left unfinished", and the model copied the gender. Every insight it wrote
about the person used "he"/"his" with nothing in the conversation establishing it.
Those are stored as kind=insight facts and fed back through recall, so an invented
fact about the person compounded every turn. The Tier 1 test guards the prompt text;
this one drives the live model, which is where the copying actually happened.
"""
import os
import re

import pytest

import compose
import drift

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

TRIALS = 4
GENDERED = re.compile(r"\b(he|she|his|her|him|hers)\b", re.I)

# Deliberately gender-free, and about a family — the context that drew it out live.
FACTS = """- Is a painter.
- Has a brother named Sam.
- The person's mother is in the hospital.
- Started painting again.
- The gallery show is in July."""

QUESTIONS = ["What is this person moving through right now?",
             "What does this person need most?"]


@pytest.fixture(scope="module")
def gen():
    g, fast = compose.make_generator()
    assert fast is not None, "expected a live backend with a key"
    return fast or g


def test_insights_never_invent_a_gender(gen):
    offenders = []
    for question in QUESTIONS:
        for _ in range(TRIALS):
            out = (gen(drift._INSIGHT_SYSTEM,
                       f"Question: {question}\n\nWhat you keep:\n{FACTS}") or "").strip()
            if GENDERED.search(out):
                offenders.append(out)
    assert not offenders, f"invented a gender in {len(offenders)} insight(s): {offenders}"


def test_insights_still_say_something(gen):
    """Guard the other direction — the gender ban must not scare it into NONE."""
    outs = [(gen(drift._INSIGHT_SYSTEM,
                 f"Question: {QUESTIONS[0]}\n\nWhat you keep:\n{FACTS}") or "").strip()
            for _ in range(TRIALS)]
    empty = [o for o in outs if not o or o.upper().strip(".!") == "NONE"]
    assert not empty, f"refused to synthesize: {outs}"
