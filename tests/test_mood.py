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


# --- the LLM register read --- #

def test_llm_signal_reads_a_named_register():
    assert mood.llm_signal("x", lambda s, u: "frozen") == "frozen"
    assert mood.llm_signal("x", lambda s, u: "tidal") == "tidal"
    assert mood.llm_signal("x", lambda s, u: "turn") == "turn"


def test_llm_signal_tolerates_the_model_padding_its_answer():
    """The prompt asks for one lowercase word; models do not always comply."""
    assert mood.llm_signal("x", lambda s, u: "Frozen.") == "frozen"
    assert mood.llm_signal("x", lambda s, u: "  tidal\n") == "tidal"


def test_neutral_and_nonsense_both_mean_no_signal():
    """None is the contract shared with register_signal and the anchor classifier:
    no signal, so the caller inherits rather than resetting the register."""
    for answer in ("neutral", "", "   ", "I'm not sure", "banana"):
        # bind per iteration — a bare closure over `answer` is a late-binding trap
        assert mood.llm_signal("x", lambda s, u, a=answer: a) is None


def test_the_prompt_names_every_register_and_biases_to_neutral():
    """Guard on the prompt itself: most real messages carry no feeling, and an
    earlier version of this system spent its rarest register on "hello"."""
    p = mood._LLM_SYSTEM.lower()
    for name in ("frozen", "tidal", "turn", "neutral"):
        assert name in p
    assert "most messages are neutral" in p


def test_make_llm_mood_signal_matches_register_signal_shape():
    """It has to drop into the same chain, so it must take a message and return
    a register or None — exactly what voice_eval.register_signal does."""
    sig = mood.make_llm_mood_signal(lambda s, u: "frozen")
    assert sig("anything") == "frozen"
    assert mood.make_llm_mood_signal(lambda s, u: "neutral")("anything") is None
