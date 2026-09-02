"""Tier 1 — drift / idle self-reflection (drift.py). No key: stub generator."""
import re
import time
import pytest
import drift
import memory

pytestmark = pytest.mark.unit


def _stub(system, user):
    return "They have been quiet. I keep what they gave me."


@pytest.fixture
def store(tmp_path):
    s = memory.MemoryStore(tmp_path / "f.jsonl")
    s.add("Is a painter; stopped in March.", "identity")
    return s


@pytest.fixture
def log(tmp_path):
    return drift.ReflectionLog(tmp_path / "reflections.jsonl")


def test_reflect_produces_and_logs(store, log):
    r = drift.reflect(store, _stub, log)
    assert r.text and len(log.items) == 1
    assert log.latest().text == r.text


def test_reflection_log_persists(tmp_path):
    p = tmp_path / "reflections.jsonl"
    l1 = drift.ReflectionLog(p)
    l1.add(drift.Reflection(text="a quiet note"))
    l2 = drift.ReflectionLog(p)                     # reload from disk
    assert len(l2.items) == 1 and l2.items[0].text == "a quiet note"


def test_maybe_drift_respects_interval(store, log):
    cfg = drift.DriftConfig(min_interval_min=180)
    now = time.time()
    # just drifted 10 min ago -> should NOT drift again
    assert drift.maybe_drift(store, _stub, log, last_drift_at=now - 600,
                             config=cfg, now=now) is None
    # last drift 4h ago -> should drift
    assert drift.maybe_drift(store, _stub, log, last_drift_at=now - 4 * 3600,
                             config=cfg, now=now) is not None


def test_disabled_never_drifts(store, log):
    cfg = drift.DriftConfig(enabled=False)
    assert drift.maybe_drift(store, _stub, log, last_drift_at=None, config=cfg) is None


def test_no_drift_on_empty_store(tmp_path, log):
    # REGRESSION: an empty store must NOT make the idle mind invent a person.
    empty = memory.MemoryStore(tmp_path / "empty.jsonl")
    def boom(system, user):
        raise AssertionError("model was called on an empty store")
    assert drift.maybe_drift(empty, boom, log, last_drift_at=None) is None
    assert log.items == []                       # nothing logged either


def test_reflect_empty_store_does_not_generate(tmp_path):
    empty = memory.MemoryStore(tmp_path / "e.jsonl")
    def boom(system, user):
        raise AssertionError("model called with empty memory")
    r = drift.reflect(empty, boom)               # must not call the model
    assert "Nothing kept yet" in r.text


# --- Generative Agents reflection synthesis --- #

def _synth_gen(system, user):
    """Routes on the prompt: salient questions, then a grounded insight."""
    if "salient high-level questions" in system:
        return "What is she avoiding?\nWhat does she keep returning to?"
    if "write ONE insight" in system:
        return "[8] She keeps circling back to what she left unfinished."
    return "a quiet note"


@pytest.fixture
def rich_store(tmp_path):
    s = memory.MemoryStore(tmp_path / "f.jsonl")
    s.add("Is a painter; stopped in March.", "identity")
    s.add("Left a canvas unfinished.", "event")
    s.add("Has a brother, Sam; not spoken since spring.", "identity")
    return s


def test_synthesize_creates_retrievable_insights(rich_store, log):
    insights = drift.synthesize(rich_store, _synth_gen, log)
    assert insights, "should produce at least one insight"
    assert all(f.kind == "insight" for f in insights)
    # stored back into the fact stream (retrievable), and logged for the dashboard
    assert any(f.kind == "insight" for f in rich_store.facts)
    assert log.latest().text == insights[-1].text


def test_synthesize_dedups_identical_insights(rich_store):
    # both questions yield the same insight text -> collapses to one
    insights = drift.synthesize(rich_store, _synth_gen)
    assert len(insights) == 1


def test_synthesize_skips_when_too_few_facts(tmp_path):
    s = memory.MemoryStore(tmp_path / "f.jsonl")
    s.add("Only one fact so far.", "event")
    assert drift.synthesize(s, _synth_gen) == []


def test_insight_recalled_as_own_read(rich_store):
    drift.synthesize(rich_store, _synth_gen)
    out = memory.recall(rich_store, "what is she avoiding", k=6)
    assert "come to understand" in out         # rendered as the Keeper's conclusion


def test_maybe_drift_synthesizes_when_rich(rich_store, log):
    import time
    r = drift.maybe_drift(rich_store, _synth_gen, log,
                          last_drift_at=time.time() - 4 * 3600)
    assert r is not None
    assert any(f.kind == "insight" for f in rich_store.facts)


# --- the insight prompt must not seed a gender --- #

def test_insight_prompt_carries_no_gendered_example():
    """Regression: the prompt's one-shot example read "She keeps circling back to what
    she left unfinished", and the model copied the gender — every insight it wrote
    called the person "he". Those land in facts.jsonl as kind=insight and feed back
    through recall, so an invented fact about the person compounds every turn."""
    prompt = drift._INSIGHT_SYSTEM
    # "never he, she, his or her" is the instruction naming them — strip the ban line
    instruction = prompt[prompt.find("The person's gender is NOT known"):]
    leaked = re.findall(r"\b(he|she|his|her|him|hers)\b",
                        prompt.replace(instruction, ""), re.I)
    assert not leaked, f"gendered wording outside the ban: {leaked}"
    assert "they" in prompt.lower()
    assert "gender is NOT known" in prompt


# --- reflection must not store the Keeper's own water-poetry as an insight --- #

def test_synthesize_drops_water_poetry_insights(tmp_path):
    """Regression: distill filters the Keeper's own motif lines out of facts, but
    reflection reaches the store by a different door and skipped that guard — so
    "The tide brought back what you gave the water in spring" was landing as
    kind=insight, the voice stored as a read of the person and recalled back to it."""
    store = memory.MemoryStore(tmp_path / "facts.jsonl")
    for t in ("Has a brother, Sam.", "Is a painter.", "Turns 30 next month."):
        store.add(t, "identity")

    def poetic(system, user):
        if "questions" in system.lower() or "ask the" in system.lower():
            return "What is he carrying?"
        return "[7] The tide brought back what you gave the water in spring."

    made = drift.synthesize(store, poetic)
    assert made == []
    assert [f for f in store.facts if f.kind == "insight"] == []


def test_synthesize_keeps_a_plain_insight(tmp_path):
    """The filter must not eat legitimate insights that merely mention water once."""
    store = memory.MemoryStore(tmp_path / "facts.jsonl")
    for t in ("Has a brother, Sam.", "Is a painter.", "Turns 30 next month."):
        store.add(t, "identity")

    def plain(system, user):
        if "questions" in system.lower() or "ask the" in system.lower():
            return "What are they moving through?"
        return "[7] They keep circling back to what they left unfinished."

    made = drift.synthesize(store, plain)
    assert len(made) == 1 and "circling back" in made[0].text
