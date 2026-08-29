"""Tier 3 — re-voicing long content, on the real model. Run explicitly:

    .venv/bin/pytest tests/evals -m eval

Why this exists: the fast model was hard-capped at 128 max_tokens. That is plenty for
what it was first used for — a one-word SUPERSEDES/DISTINCT verdict, a rerank list,
the voice rubric — but background research is BOTH synthesized and re-voiced through
that same callable. So a multi-paragraph answer was delivered cut off mid-sentence at
roughly 640 characters, with no error anywhere. Nothing in the unit suite could see
it: the cap lives inside the OpenAI generator, which the offline tests never build.
"""
import os

import pytest

import compose

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

# The shape of a real orchestration result — long enough that a 128-token cap cuts it.
LONG_FINDING = (
    "Watercolor is a transparent paint that relies on the white of the paper for its "
    "lightness, and it is best suited to loose, layered washes, landscapes and florals. "
    "Gouache is an opaque water-based paint with a matte finish; it can be reactivated "
    "with water once dry, and it is favoured for illustration, poster work and any "
    "piece where flat, vivid colour matters. A beginner watercolor set typically costs "
    "about twenty dollars, while a comparable gouache set runs closer to thirty-five. "
    "Both use the same brushes and paper weights, so a painter moving between them "
    "needs no second kit. The main practical difference is that watercolor mistakes "
    "are hard to correct, whereas gouache can be painted over opaquely."
)
TERMINAL = (".", "!", "?", '"', "'", ")", "—")


@pytest.fixture(scope="module")
def fast():
    g, f = compose.make_generator()
    assert f is not None, "expected a live backend with a key"
    return f


def test_a_long_answer_is_not_truncated(fast):
    """The regression: delivered at ~640 chars, ending mid-sentence."""
    out = compose.revoice(LONG_FINDING, "tidal", generate=fast).strip()
    assert out, "revoice returned nothing"
    assert out[-1] in TERMINAL, f"ends mid-sentence ({len(out)} chars): ...{out[-120:]!r}"


def test_a_long_answer_keeps_its_facts(fast):
    """Re-voicing must preserve the content, not just avoid the cliff."""
    out = compose.revoice(LONG_FINDING, "tidal", generate=fast).strip().lower()
    assert "gouache" in out and "watercolor" in out


def test_a_short_answer_stays_short(fast):
    """Raising the cap is an upper bound, not a target — a one-liner must not bloat."""
    out = compose.revoice("It is 14 degrees outside.", "tidal", generate=fast).strip()
    assert len(out) < 400, f"short answer ballooned to {len(out)} chars: {out!r}"
