"""Tier 3 — the relevance gate, against the REAL embedder. Run explicitly:

    .venv/bin/pytest tests/evals -m eval

The gate is an absolute cosine floor (memory._REL_FLOOR), and absolute thresholds
drift when the embedding model changes. It has to do two opposite jobs at once:

  * let a genuine semantic hit through even with NO shared word
    ("what do i do for work" -> "Is a painter.")
  * recall NOTHING for a vague request that matches nothing
    ("help me write things down" -> the model must not recite an unrelated memory)

At 0.25 it was failing the first job. The usable band measured only ~0.138-0.199,
so the margin either side of the floor is thin by nature — which is exactly why
both directions are pinned here rather than left to a comment.
"""
import os

import pytest

import embedder as embedder_mod
import memory

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

FACTS = [("Has a brother, Sam.", "identity"),
         ("Is a painter.", "identity"),
         ("Turns 30 next month.", "identity")]


@pytest.fixture(scope="module")
def emb():
    return embedder_mod.openai_embedder()


@pytest.fixture
def store(tmp_path, emb):
    s = memory.MemoryStore(tmp_path / "facts.jsonl")
    for text, kind in FACTS:
        s.add(text, kind, embed=emb)
    return s


@pytest.mark.parametrize("cue,expected", [
    ("tell me about my sibling", "brother"),   # no shared word with the fact
    ("what do i do for work", "painter"),      # no shared word either
])
def test_a_real_semantic_hit_survives_the_gate(store, emb, cue, expected):
    out = memory.recall(store, cue, k=1, embed=emb)
    assert expected in out, f"{cue!r} recalled {out!r}"


@pytest.mark.parametrize("cue", [
    "help me write things down",
    "what's the weather like",
])
def test_a_vague_cue_recalls_nothing(store, emb, cue):
    """The gate's other job: no recitation of an unrelated memory."""
    out = memory.recall(store, cue, k=1, embed=emb)
    assert not out.strip(), f"{cue!r} should recall nothing, got {out!r}"


def test_an_ambient_fact_ignores_the_gate(tmp_path, emb):
    """Importance >= 8 is deliberately exempt — a life-defining fact stays in mind
    whatever the cue. Lowering the floor must not have changed that."""
    s = memory.MemoryStore(tmp_path / "facts.jsonl")
    s.add("Their mother is in the hospital.", "state", embed=emb, importance=9.0)
    out = memory.recall(s, "what's the weather like", k=1, embed=emb)
    assert "hospital" in out
