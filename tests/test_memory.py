"""Tier 1 — the drawers (memory.py): recall, dedup, persistence."""
import pytest
import memory

pytestmark = pytest.mark.unit


def test_add_and_recall_relevant(store):
    store.add("Has a brother, Sam.", "identity")
    store.add("Is a painter.", "identity")
    out = memory.recall(store, "tell me about my brother", k=2)
    assert "Sam" in out


def test_dedup_bumps_instead_of_duplicating(store):
    store.add("Feels stuck lately, can't get started.", "state")
    dup = store.add("Feels stuck, can't get started these days.", "state")
    assert len(store.facts) == 1
    assert dup.mentions == 2


def test_distinct_facts_not_merged(store):
    store.add("Has a brother, Sam; not spoken since spring.", "identity")
    store.add("Has a brother named Sam.", "identity")   # low overlap -> distinct
    assert len(store.facts) == 2


def test_empty_cue_returns_present_facts(store):
    store.add("A.", "event")
    store.add("B.", "event")
    assert memory.recall(store, "", k=5).count("-") == 2


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "facts.jsonl"
    s1 = memory.MemoryStore(p)
    s1.add("Turns 30 next month.", "identity")
    s2 = memory.MemoryStore(p)                            # reload from disk
    assert len(s2.facts) == 1
    assert s2.facts[0].text == "Turns 30 next month."


def test_empty_store_recall_is_empty(store):
    assert memory.recall(store, "anything") == ""


# --- semantic layer (deterministic fake embedder, no key needed) --- #

_TOPICS = ["brother", "painter", "age"]


def _fake_embed(texts):
    """One-hot topic vectors, with query synonyms mapped to the same topic — so
    'sibling'~brother and 'work'~painter share a dimension without sharing words."""
    out = []
    for t in texts:
        low = t.lower()
        v = [0.0] * len(_TOPICS)
        for i, kw in enumerate(_TOPICS):
            if kw in low:
                v[i] = 1.0
        if "sibling" in low:
            v[0] = 1.0
        if "work" in low:
            v[1] = 1.0
        if "old" in low:
            v[2] = 1.0
        out.append(v)
    return out


def test_semantic_recall_ranks_by_cosine(store):
    # REGRESSION/feature: recall by MEANING, not shared words.
    store.add("Has a brother named Sam.", "identity", embed=_fake_embed)
    store.add("Is a painter.", "identity", embed=_fake_embed)
    store.add("Turns 30 next month.", "identity", embed=_fake_embed)
    assert "brother" in memory.recall(store, "tell me about my sibling", k=1,
                                      embed=_fake_embed)
    assert "painter" in memory.recall(store, "what is my work", k=1,
                                      embed=_fake_embed)


def test_semantic_dedup_merges_reworded_fact(store):
    store.add("Has a brother named Sam.", "identity", embed=_fake_embed)
    dup = store.add("Her brother is named Sam.", "identity", embed=_fake_embed)
    assert len(store.facts) == 1 and dup.mentions == 2  # same topic vector -> merge
