"""Tier 1 — the register (mood) classifier's structure. No key, no model download:
these drive mood.classify with a tiny fake embedder, so they assert the RULES
(floor, margin, neutral-as-no-signal) rather than any model's accuracy. Accuracy
lives in backend/mood_bench.py against evals/register_dataset.json.
"""
import pytest

import mood

pytestmark = pytest.mark.unit


def _fake_embed(texts):
    """One dimension per register plus one for neutral; a text scores on whichever
    marker word it contains, so cosine is fully predictable."""
    out = []
    for t in texts:
        low = t.lower()
        out.append([1.0 if m in low else 0.0
                    for m in ("cold", "moving", "shifting", "lookup")])
    return out


ANCHORS = {"frozen": ["cold"], "tidal": ["moving"],
           "turn": ["shifting"], mood.NEUTRAL: ["lookup"]}


@pytest.fixture
def av():
    return {r: _fake_embed(sents) for r, sents in ANCHORS.items()}


def test_a_clear_register_is_returned(av):
    assert mood.classify("everything is cold", _fake_embed, av) == "frozen"
    assert mood.classify("things are moving", _fake_embed, av) == "tidal"


def test_neutral_reads_as_no_signal(av):
    """The fix: nearest-centroid must place every message SOMEWHERE, so without a
    neutral class a plain tool request was forced into an emotional register —
    confidently, with a wide margin, so no floor or margin could reject it. Winning
    the neutral class means no signal, and the caller inherits continuity."""
    assert mood.classify("do a lookup for me", _fake_embed, av) is None


def test_below_the_floor_is_no_signal(av):
    assert mood.classify("nothing matches here", _fake_embed, av) is None


def test_an_ambiguous_tie_is_no_signal(av):
    # hits "cold" and "moving" equally -> margin not met -> abstain
    assert mood.classify("cold and moving", _fake_embed, av) is None


def test_empty_message_is_no_signal(av):
    assert mood.classify("   ", _fake_embed, av) is None


def test_neutral_anchors_stay_request_shaped():
    """Regression on the anchor WORDING. A first pass used bare queries ("what time
    is it") and first-person lines ("what have i been working on lately"); those
    swallowed real implicit mood, which is also phrased as mundane first-person
    activity ("i just stare at the ceiling for hours") — implicit accuracy fell from
    38% to 25% on the held-out split. Neutral anchors must stay addressed to the
    Keeper, not describe the person's own life."""
    for anchor in mood.ANCHORS[mood.NEUTRAL]:
        assert not anchor.lower().startswith("i "), anchor
        assert " i " not in f" {anchor.lower()} ", anchor
