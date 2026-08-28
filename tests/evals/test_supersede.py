"""Tier 3 — the supersession judge, on the real model. Run explicitly:

    .venv/bin/pytest tests/evals -m eval

Why this exists: `test_memory.py` drives supersession through `_supersede_judge`, a
fake that returns a perfectly-spelled "SUPERSEDES". The real judge does not. It
answered "SUPERSCEDES" in roughly one call in three, which the old exact-substring
check read as DISTINCT — so "i stopped painting in march" then "actually i started
painting again" left BOTH facts active, `changes_tracked` stuck at 0, and recall
feeding the Keeper a contradiction it then recited flat:
"You have started painting again. You stopped painting in March."

These drive the live judge and assert on what the parser makes of its real output.
"""
import os
import pytest

import compose
import memory

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

TRIALS = 6

# The pair from FEATURES.md's "temporal / change-aware (the tide)" line, in the
# words distillation actually produces for it.
OLD_FACT = "Stopped painting in March."
NEW_FACT = "Has started painting again."


@pytest.fixture(scope="module")
def judge():
    g, fast = compose.make_generator()
    assert fast is not None, "expected a live backend with a key"
    return fast or g


def _verdict(judge, old: str, new: str) -> str:
    return judge(memory._SUPERSEDE_SYSTEM, f"OLD: {old}\nNEW: {new}") or ""


def test_a_real_change_is_read_as_superseding_every_time(judge):
    """However the judge spells it. Failing here means the parser has drifted back
    to being literal about a word the model does not spell reliably."""
    verdicts = [_verdict(judge, OLD_FACT, NEW_FACT) for _ in range(TRIALS)]
    missed = [v for v in verdicts if not memory._reads_as_supersedes(v)]
    assert not missed, f"parsed as DISTINCT: {missed} (all verdicts: {verdicts})"


def test_two_unrelated_facts_are_not_read_as_superseding(judge):
    """The other direction — the loosened parser must not turn every verdict into a
    supersession and start erasing facts that are still true."""
    verdicts = [_verdict(judge, "Has a brother, Sam.", "Works as a nurse.")
                for _ in range(TRIALS)]
    wrong = [v for v in verdicts if memory._reads_as_supersedes(v)]
    assert not wrong, f"parsed as SUPERSEDES: {wrong} (all verdicts: {verdicts})"
