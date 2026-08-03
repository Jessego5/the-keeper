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


# --- Generative Agents retrieval: recency + importance + relevance --- #

def test_importance_parsed_from_distill(store):
    def gen(system, user):
        return "[state|9] Lost her job in March.\n[preference|2] Likes oat milk."
    facts = memory.distill([{"role": "user", "content": "..."}], gen, store)
    by_text = {f.text: f.importance for f in facts}
    assert by_text["Lost her job in March."] == 9.0
    assert by_text["Likes oat milk."] == 2.0


def test_distill_strips_stray_and_multiple_tags(store):
    # REGRESSION: the model emits unknown/multiple tags; none may leak into the text.
    def gen(system, user):
        return ("[emotion] Feels stuck.\n"
                "[activity] [state] Stopped painting in March.\n"
                "[relationship|7] Has a brother, Sam.")
    facts = memory.distill([{"role": "user", "content": "..."}], gen, store)
    texts = {f.text for f in facts}
    assert "Feels stuck." in texts                       # [emotion] dropped, not leaked
    assert "Stopped painting in March." in texts         # both tags stripped
    assert "Has a brother, Sam." in texts
    assert not any("[" in t for t in texts)              # nothing bracketed survives
    kinds = {f.text: f.kind for f in facts}
    assert kinds["Stopped painting in March."] == "state"   # first known tag wins
    assert kinds["Feels stuck."] == "event"                 # unknown -> default
    imp = {f.text: f.importance for f in facts}
    assert imp["Has a brother, Sam."] == 7.0                # importance read past tag


def test_importance_breaks_recency_tie(store):
    # Same recency (added together), so importance decides the ranking.
    store.add("Ran out of oat milk.", "state", importance=1)
    store.add("Her mother died last week.", "state", importance=10)
    top = memory.rank_facts(store.facts, cue="", k=1)
    assert "mother" in top[0].text


def test_recency_decays_ranking(store):
    import time
    store.add("Older, equally important.", "state", importance=5)
    store.add("Newer, equally important.", "state", importance=5)
    store.facts[0].last_seen = time.time() - 10 * 86400   # age the first one 10 days
    top = memory.rank_facts(store.facts, cue="", k=1)
    assert "Newer" in top[0].text


def test_dedup_keeps_higher_importance(store):
    store.add("Feels stuck lately.", "state", importance=3)
    dup = store.add("Feels stuck these days.", "state", importance=8)
    assert dup.importance == 8.0        # resurfacing raised the poignancy


def test_insights_rendered_under_own_heading(store):
    store.add("Has a brother, Sam.", "identity", importance=6)
    store.add("Keeps returning to what she left unfinished.", "insight", importance=7)
    out = memory.recall(store, "", k=5)
    assert "come to understand" in out          # insight framed as the Keeper's read
    assert "your own read" in out
    # the told fact is not under the insight heading
    assert out.index("Sam") < out.index("come to understand")


# --- MemGPT consolidation: compress the low-value tail under memory pressure --- #

def _merge_gen(system, user):
    return "Ran errands around town over several mundane days."


def test_consolidate_noop_under_budget(store):
    store.add("A single small fact.", "event")
    assert memory.consolidate(store, _merge_gen, budget=60) == []


def test_consolidate_merges_similar_tail_and_archives(store):
    # three low-importance, similar errands + one important recent fact
    store.add("Went to the store for milk.", "event", importance=1)
    store.add("Went to the store for bread.", "event", importance=1)
    store.add("Went to the store for eggs.", "event", importance=1)
    store.add("Her mother is in the hospital.", "state", importance=10)
    made = memory.consolidate(store, _merge_gen, budget=3)   # over by 1 -> compress
    assert made, "should have produced a consolidated fact"
    texts = [f.text for f in store.facts]
    assert "Her mother is in the hospital." in texts          # important fact kept
    # the errands were archived out of the active set
    assert store.archive_path.exists()
    assert sum("store for" in t for t in texts) < 3


def test_consolidate_leaves_insights_untouched(store):
    for i in range(4):
        store.add(f"Minor errand number {i}.", "event", importance=1)
    store.add("She keeps circling back to what she left unfinished.",
              "insight", importance=8)
    memory.consolidate(store, _merge_gen, budget=2)
    assert any(f.kind == "insight" for f in store.facts)      # insight never archived
