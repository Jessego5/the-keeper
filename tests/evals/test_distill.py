"""Tier 3 — distillation evals. Needs OPENAI_API_KEY; run explicitly:

    .venv/bin/pytest tests/evals -m eval

Why these exist: every Tier 1 memory test injects the fact directly
(`store.add("Stopped painting in March.", ...)`) or hands distill a FAKE generator
that returns a pre-tagged line. Nothing exercised the real prompt on real words, and
that gap hid a live bug — "i stopped painting in march", the opening line of the
documented agent arc in FEATURES.md, distilled to NONE every single time. Nothing
was stored, so recall stayed empty and the temporal-change flow ("the tide") had no
old fact to supersede.

Non-deterministic, so these assert rates over a small sample. The pairing matters as
much as either half: the prompt has to keep a life-change stated in passing WITHOUT
turning the store into a diary of moods.
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

TRIALS = 4

# The Keeper's own voice, in the same transcript as the fact. Distill is told never
# to capture the companion's metaphors — it must not over-apply that and drop the
# person's fact along with them.
POETIC_REPLY = (
    "The tide recedes. Paints rest still, colors held beneath the surface. "
    "The shore knows the shifts; the paints will be drawn out once more.")


@pytest.fixture(scope="module")
def gen():
    g, fast = compose.make_generator()
    assert fast is not None, "expected a live backend with a key"
    return fast or g


def _distill(gen, *messages) -> list[str]:
    """Run the real prompt with no store; return the fact texts that survived."""
    convo = [{"role": role, "content": content} for role, content in messages]
    return [f.text for f in memory.distill(convo, gen)]


def test_a_dropped_practice_is_kept(gen):
    """The seed of the documented arc. Was NONE 12/12 before the prompt named a
    life-change as durable rather than a passing mood."""
    runs = [_distill(gen, ("user", "i stopped painting in march"))
            for _ in range(TRIALS)]
    assert all(runs), f"expected a fact every run, got {runs}"
    assert all(any("paint" in t.lower() for t in r) for r in runs), runs


def test_the_keepers_own_poetry_does_not_suppress_the_fact(gen):
    """The fact survives sharing a transcript with the Keeper's water-imagery, and
    none of that imagery is stored as if it were a fact about the person."""
    runs = [_distill(gen,
                     ("user", "i stopped painting in march"),
                     ("assistant", POETIC_REPLY))
            for _ in range(TRIALS)]
    assert all(runs), f"expected a fact every run, got {runs}"
    for texts in runs:
        assert any("paint" in t.lower() for t in texts), texts
        assert not any(memory._is_junk_fact(t) for t in texts), texts
        assert not any("tide" in t.lower() or "shore" in t.lower() for t in texts), texts


def test_a_passing_mood_is_still_skipped(gen):
    """The other side of the fix: how they feel right now is not durable. If this
    starts failing, the prompt has been loosened too far and the store will fill
    with weather."""
    for message in ("i'm tired today", "ugh, long day", "good morning"):
        runs = [_distill(gen, ("user", message)) for _ in range(TRIALS)]
        assert not any(runs), f"{message!r} should distill to nothing, got {runs}"


def test_a_plain_durable_fact_still_lands(gen):
    """Control — the case that always worked, guarding against a regression in the
    opposite direction."""
    runs = [_distill(gen, ("user", "i'm a painter and i have a brother named Sam"))
            for _ in range(TRIALS)]
    assert all(runs), runs
    for texts in runs:
        joined = " ".join(texts).lower()
        assert "sam" in joined and "paint" in joined, texts
