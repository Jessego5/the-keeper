"""
These are the Tier 3 tests for the relevance gate against the real judge. Needs
OPENAI_API_KEY:

    .venv/bin/pytest tests/evals/test_relevance.py -m eval

Why this exists. relevance.py decides what the outside world may interrupt someone
for, and it was covered ONLY by fake judges: `generate=lambda s, u: "8 - Is a
painter."` hands back the answer the test wants, so every unit test passed no
matter what the prompt did. That is the gap this repo has been bitten by before,
and it bit again here: the judge was quietly copying the EXAMPLE out of its own
prompt onto every item in a scan, birds and filmmakers included, naming a fact the
store did not hold. Nothing in the suite could see it. It was found by hand.

Measured before these thresholds were written (3 trials per item, real judge):

    rich store (identity + events)   relevant mean 0.86 (min 0.80)
                                     irrelevant mean 0.00 (max 0.00)
    thin store (events only)         relevant mean 0.86 (min 0.80)
                                     irrelevant mean 0.03 (max 0.20)

So the bars below sit well clear of what was observed. They are here to catch a
collapse, not to pin a decimal: a prompt edit that costs the gate its separation
should fail loudly rather than show up in a demo.
"""
import os
import statistics

import pytest

import compose
import embedder
import memory
import relevance
import sources

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

TRIALS = 2

# Deliberately WITHOUT a bare "Is a painter." line: that is the example inside the
# judge's own prompt, and a store that happens to hold it cannot tell a real
# attribution from the echo that this eval exists to catch.
KEPT = ["Stopped painting in March, then started again.",
        "Has a brother named Sam.",
        "The gallery show is in July."]

RELEVANT = [
    ("Denizens of a Crowded City Populate Erin Milez's Dense Paintings",
     "Dozens of figures crowd the painter's canvases, layered in oil and gouache."),
    ("A Painter's Guide to Stretching Your Own Canvas",
     "Cotton duck, staples and a bottle of gesso are all it takes."),
]
IRRELEVANT = [
    ("Avian Beauties and Rarities Top the 2026 Audubon Photography Awards",
     "Herons, owls and a rare albatross in flight."),
    ("Quantum Error Correction Reaches a New Milestone",
     "Researchers report a logical qubit below threshold."),
    ("A New Sourdough Bakery Opens on the Harbour Road",
     "Long queues for the rye and the cardamom buns."),
]


@pytest.fixture(scope="module")
def judge():
    gen, fast = compose.make_generator()
    assert fast is not None, "expected a live backend with a key"
    return fast or gen


@pytest.fixture(scope="module")
def embed():
    return embedder.make_embedder()


@pytest.fixture(scope="module")
def store(tmp_path_factory, embed):
    s = memory.MemoryStore(tmp_path_factory.mktemp("rel") / "facts.jsonl")
    for text in KEPT:
        s.add(text, kind="fact", embed=embed)
    return s


def _score(item_tuple, store, judge, embed):
    title, body = item_tuple
    item = sources.SourceItem(source="feed", title=title, body=body)
    return relevance.score_item(item, store, generate=judge, embed=embed)


def _sweep(items, store, judge, embed):
    return [_score(i, store, judge, embed) for i in items for _ in range(TRIALS)]


def test_something_they_care_about_clears_the_mention_bar(store, judge, embed):
    """The accepting half. Observed min 0.80 against a 0.45 bar."""
    results = _sweep(RELEVANT, store, judge, embed)
    scores = [s for s, _ in results]
    assert all(s >= relevance.MENTION for s in scores), scores


def test_the_rest_of_the_world_stays_below_it(store, judge, embed):
    """The rejecting half, which is the one that makes this a companion rather
    than a feed reader. Observed max 0.20 against a 0.45 bar."""
    results = _sweep(IRRELEVANT, store, judge, embed)
    scores = [s for s, _ in results]
    assert not any(s >= relevance.MENTION for s in scores), scores


def test_the_gate_actually_separates(store, judge, embed):
    """One number for the whole point of the file. Observed +0.82 to +0.86."""
    rel = [s for s, _ in _sweep(RELEVANT, store, judge, embed)]
    irr = [s for s, _ in _sweep(IRRELEVANT, store, judge, embed)]
    gap = statistics.mean(rel) - statistics.mean(irr)
    assert gap >= 0.40, f"separation collapsed to {gap:.2f} (rel {rel}, irr {irr})"


def test_it_never_names_a_fact_the_store_does_not_hold(store, judge, embed):
    """Regression, seen live. The judge copied "Is a painter." out of its own
    prompt onto every row of a scan; the store held no such line. `because` is
    what shows the person WHY it thought this mattered, so a fact they never gave
    it is a plausible lie, and this store deliberately does not contain the
    example that was being echoed."""
    for score, because in _sweep(RELEVANT + IRRELEVANT, store, judge, embed):
        if score >= relevance.MENTION:
            assert because in KEPT, f"attributed to something unkept: {because!r}"
        else:
            assert because == "", f"a silent item named a reason: {because!r}"


def test_an_empty_store_can_never_be_interrupted_for(judge, embed, tmp_path):
    """Nothing known means nothing can matter, without asking the judge at all."""
    empty = memory.MemoryStore(tmp_path / "empty.jsonl")
    assert _score(RELEVANT[0], empty, judge, embed) == (0.0, "")
