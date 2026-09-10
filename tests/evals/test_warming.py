"""
These are the Tier 3 tests for the warming judge, on the real model. Run explicitly:

    .venv/bin/pytest tests/evals -m eval

`turn` is the Keeper's rarest register: the ice going out, and it is now earned
only by a recorded reversal INTO something better. That direction comes from a
free-text verdict, which this session has repeatedly shown to be the thing that
drifts: the supersede judge produced THREE spellings of its own answer. So the
contract pinned here is the model's output, not the parser.
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

TRIALS = 3


@pytest.fixture(scope="module")
def judge():
    g, fast = compose.make_generator()
    assert fast is not None, "expected a live backend with a key"
    return fast or g


def _verdict(judge, old: str, new: str) -> str:
    return judge(memory._WARMING_SYSTEM, f"OLD: {old}\nNEW: {new}") or ""


@pytest.mark.parametrize("old,new", [
    ("Stopped painting in March.", "Started painting again."),
    ("Hasn't spoken to their brother since spring.", "Called their brother."),
    ("Was signed off work with burnout.", "Went back to work part time."),
])
def test_a_return_reads_as_warmer(judge, old, new):
    for _ in range(TRIALS):
        v = _verdict(judge, old, new)
        assert memory._reads_as_warmer(v), f"{v!r} for {old!r} -> {new!r}"


@pytest.mark.parametrize("old,new", [
    ("Started painting again.", "Stopped painting once more."),
    ("Their mother was recovering.", "Their mother is back in hospital."),
])
def test_a_loss_never_reads_as_warmer(judge, old, new):
    """The direction that must never be wrong. A false WARMER would meet a loss
    with the ice-going-out register: the worst tonal failure the Keeper has."""
    for _ in range(TRIALS):
        v = _verdict(judge, old, new)
        assert not memory._reads_as_warmer(v), f"{v!r} for {old!r} -> {new!r}"


def test_a_directionless_update_is_not_a_warming(judge):
    """Most supersessions are neither: a changed detail should not spend the
    rarest register."""
    misread = [v for _ in range(TRIALS)
               if memory._reads_as_warmer(
                   v := _verdict(judge, "Lives in Leeds.", "Lives in Bristol."))]
    assert not misread, misread
