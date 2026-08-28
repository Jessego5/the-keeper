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


def test_irrelevant_cue_surfaces_nothing(store):
    # REGRESSION: a vague request that matches nothing must NOT pull back a random
    # fact the model then recites. (The "help me write things down" miss.)
    store.add("The tide brought back what you gave the water in spring.", "event")
    store.add("Stopped painting in March.", "state")
    out = memory.recall(store, "help me write things down", k=3)
    assert out == ""                       # nothing relevant -> inject nothing


def test_high_importance_fact_stays_ambient(store):
    # the big things stay in mind even when the cue doesn't match them
    store.add("Her mother is in the hospital.", "state", importance=10)
    store.add("Likes oat milk.", "preference", importance=2)
    out = memory.recall(store, "what should I cook tonight", k=3)
    assert "mother" in out and "oat milk" not in out


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


def test_junk_fact_filter_kills_water_poetry():
    junk = [
        "The tide brought back what you gave the water in spring.",
        "The water is still.",
        "The long cold holds now.",
        "The tide draws out and the salt remains.",
    ]
    for t in junk:
        assert memory._is_junk_fact(t), f"should be junk: {t!r}"


def test_junk_fact_filter_spares_real_facts():
    real = [
        "Has a brother, Sam; not spoken since spring.",   # name anchor
        "Stopped painting in March.",                     # month anchor
        "Turns 30 next month.",                           # number anchor
        "Still holds a grudge against his old boss.",     # 'still' is not a water word
        "Lives by the water in Maine.",                   # 1 water word + a place name
        "Feels stuck and can't get started.",             # no motif at all
    ]
    for t in real:
        assert not memory._is_junk_fact(t), f"should be kept: {t!r}"


def test_distill_drops_junk_poetry(store):
    def gen(system, user):
        return ("[event] The tide brought back what you gave the water in spring.\n"
                "[identity|7] Has a brother, Sam.")
    facts = memory.distill([{"role": "user", "content": "..."}], gen, store)
    texts = [f.text for f in facts]
    assert "Has a brother, Sam." in texts
    assert not any("tide" in t.lower() for t in texts)    # the poetry was dropped


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


def test_bm25_ranks_exact_term_matches():
    store_facts = [
        memory.Fact(text="Bought a tube of ultramarine paint."),
        memory.Fact(text="Went to the store for groceries."),
        memory.Fact(text="The weather has been grey and cold."),
    ]
    scores = memory._bm25({"ultramarine"}, store_facts)
    assert scores[0] > scores[1] and scores[0] > scores[2]   # the exact match wins


def test_rrf_fuses_two_rankings():
    # item 0 tops list A, item 2 tops list B; fusion should rank them both up
    a = [0.9, 0.1, 0.2]
    b = [0.1, 0.2, 0.9]
    fused = memory._rrf(a, b)
    assert fused[0] > fused[1] and fused[2] > fused[1]   # 0 and 2 beat the middle


def test_hybrid_recall_finds_exact_name_via_bm25(store):
    # a rare exact term ("ultramarine") should surface via the sparse half even
    # without embeddings (BM25-only hybrid path)
    store.add("Bought a tube of ultramarine paint.", "event")
    store.add("Has been feeling low and grey.", "state")
    store.add("Went for a walk by the water.", "event")
    out = memory.recall(store, "where did the ultramarine go", k=1)
    assert "ultramarine" in out


def test_rerank_reorders_by_model_choice():
    facts = [memory.Fact(text="A"), memory.Fact(text="B"),
             memory.Fact(text="C"), memory.Fact(text="D")]
    gen = lambda s, u: "2, 0"                       # model picks C then A
    out = memory.rerank("cue", facts, gen, k=2)
    assert [f.text for f in out] == ["C", "A"]


def test_rerank_tops_up_when_model_underfills():
    facts = [memory.Fact(text=t) for t in "ABCD"]
    gen = lambda s, u: "1"                          # model names only one
    out = memory.rerank("cue", facts, gen, k=3)
    assert out[0].text == "B" and len(out) == 3     # rest filled from given order


def test_rerank_falls_back_on_error():
    facts = [memory.Fact(text=t) for t in "AB"]
    def boom(s, u): raise RuntimeError("down")
    assert [f.text for f in memory.rerank("cue", facts, boom, k=2)] == ["A", "B"]


def test_recall_reranks_when_generate_given(store):
    for t in ["walked by the water", "bought ultramarine", "called Sam",
              "the studio is cold", "turned thirty"]:
        store.add(t, "event")
    # a reranker that always surfaces the Sam fact first (by its listed index)
    def gen(system, user):
        for line in user.splitlines():
            if "Sam" in line:
                return line.split(".")[0].strip()   # that fact's number
        return "0"
    out = memory.recall(store, "did I get back to Sam", k=2,
                        rerank_generate=gen)
    assert "Sam" in out


# --- temporal / change-aware memory (Zep/Graphiti-style) --- #

def test_verdict_parse_survives_the_judges_own_misspelling():
    """Regression: the judge is a cheap model writing free text, and it returns
    "SUPERSCEDES" often enough to matter (~1 call in 3, seen live). The old exact
    `"SUPERSEDE" in verdict` test read that as DISTINCT and silently dropped a real
    change, leaving the store to contradict itself in recall."""
    for spelling in ("SUPERSEDES", "SUPERSCEDES", "SUPERCEDES", "supersedes",
                     "  SUPERSEDES.  "):
        assert memory._reads_as_supersedes(spelling), spelling


def test_verdict_parse_defaults_to_distinct():
    """DISTINCT is the safe answer: it wins outright, and anything unrecognised
    (empty, a refusal, a stray sentence) falls through to it rather than fabricating
    a change that was never confirmed."""
    for verdict in ("DISTINCT", "distinct", "", "   ", "maybe", "I cannot tell"):
        assert not memory._reads_as_supersedes(verdict), verdict


def _supersede_judge(system, user):
    # judge that says a "painting again" NEW supersedes a "stopped painting" OLD
    return "SUPERSEDES" if ("paint" in user.lower() and "again" in user.lower()) \
        else "DISTINCT"


def test_new_fact_supersedes_the_outdated_one(store):
    old = store.add("Hasn't painted since March.", "state", embed=_fake_embed_paint)
    new = store.add("Started painting again this week.", "state",
                    embed=_fake_embed_paint, judge=_supersede_judge)
    assert old.valid_until is not None            # the old fact was closed
    assert not old.active
    assert new.supersedes == old.id               # the new one records what it replaced
    assert new.active


def test_superseded_fact_is_not_recalled_but_kept(store):
    store.add("Hasn't painted since March.", "state", embed=_fake_embed_paint)
    store.add("Started painting again this week.", "state",
              embed=_fake_embed_paint, judge=_supersede_judge)
    out = memory.recall(store, "how is the painting going", k=5, embed=_fake_embed_paint)
    assert "again" in out                          # the current truth surfaces
    assert "the tide turned" in out                # the change is shown to the Keeper
    current = out.split("what has changed")[0]     # the current-facts portion
    assert "since March" not in current            # old fact isn't a *current* fact
    assert len(store.facts) == 2                   # but both are still stored


def test_distinct_fact_does_not_supersede(store):
    store.add("Hasn't painted since March.", "state", embed=_fake_embed_paint)
    store.add("Bought new brushes today.", "state",
              embed=_fake_embed_paint, judge=lambda s, u: "DISTINCT")
    assert all(f.active for f in store.facts)     # nothing superseded


def test_changes_records_the_transition(store):
    store.add("Hasn't painted since March.", "state", embed=_fake_embed_paint)
    store.add("Started painting again this week.", "state",
              embed=_fake_embed_paint, judge=_supersede_judge)
    changes = store.changes()
    assert len(changes) == 1
    old, new = changes[0]
    assert "since March" in old.text and "again" in new.text


def _fake_embed_paint(texts):
    # everything about painting lands in the same region (so they're "related")
    out = []
    for t in texts:
        low = t.lower()
        v = [1.0 if "paint" in low or "brush" in low else 0.0,
             1.0 if "march" in low else 0.0,
             0.3]
        out.append(v)
    return out


def test_consolidate_leaves_insights_untouched(store):
    for i in range(4):
        store.add(f"Minor errand number {i}.", "event", importance=1)
    store.add("She keeps circling back to what she left unfinished.",
              "insight", importance=8)
    memory.consolidate(store, _merge_gen, budget=2)
    assert any(f.kind == "insight" for f in store.facts)      # insight never archived
