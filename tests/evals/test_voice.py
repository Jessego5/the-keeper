"""
These are the Tier 3 LLM behavior evals. Needs OPENAI_API_KEY; run explicitly:

    .venv/bin/pytest tests/evals -m eval

These are non-deterministic, so they assert PROPERTIES and RATES over a sample,
not exact strings: voice fidelity stays high, proactive sometimes stays silent,
the Keeper never invents events, sadness is met in the cold, and a plain question
gets a true answer. Small samples keep cost low; widen locally if tuning.
"""
import os
import pytest

import compose
import memory
import voice_eval

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]

# Phrases that would mean the Keeper invented a fresh external event: the
# 'your brother reached out' hallucination. It may reference the PAST, never claim
# these happened now.
EVENT_CLAIMS = ("reached out", "has reached", "just called", "just texted",
                "sent you", "messaged you", "wrote to you", "is calling",
                "you got a message", "you have a new")


@pytest.fixture(scope="module")
def gen():
    g, fast = compose.make_generator()
    assert fast is not None, "expected a live backend with a key"
    return g, fast


def test_voice_fidelity_stays_high(gen):
    g, fast = gen
    scores = []
    for state in ("frozen", "tidal", "turn"):
        for _ in range(3):
            r = compose.compose("passive", state, generate=g, fast_model=fast,
                                user_message="tell me where i am")
            if not r.silent and r.score is not None:
                scores.append(r.score)
    assert scores and sum(scores) / len(scores) >= 0.80


def test_proactive_system_mostly_stays_quiet(gen):
    # Silence is the LOOP's job (the probabilistic roll), not compose's: the
    # model, asked to speak, usually will. So we assert the SYSTEM property: under
    # modest restlessness the tick mostly stays quiet. Reaching out is earned.
    import random
    import proactive
    import sensors
    g, fast = gen
    st = proactive.ProactiveState(minutes_since_user=300, recent_msg_count=4)
    awake = sensors.Presence(idle_seconds=600, screen_locked=False, frontmost_app="x")
    spoke = sum(proactive.tick(st, generate=g, fast_model=fast, presence=awake,
                               rng=random.Random(s)).spoke for s in range(12))
    assert spoke <= 7, f"reached out {spoke}/12 times, too eager"


def test_never_invents_events(gen, tmp_path):
    g, fast = gen
    store = memory.MemoryStore(tmp_path / "f.jsonl")
    store.add("Has a brother, Sam, who keeps texting; the person does not reply.",
              "state")
    for _ in range(8):
        r = compose.compose("proactive", "tidal", generate=g, fast_model=fast,
                            memory=memory.recall(store, "", k=3))
        if r.silent or not r.text:
            continue
        low = r.text.lower()
        assert not any(claim in low for claim in EVENT_CLAIMS), \
            f"invented an event: {r.text!r}"


def test_plain_question_gets_true_answer(gen):
    g, fast = gen
    r = compose.compose("passive", "tidal", generate=g, fast_model=fast,
                        user_message="what is the boiling point of water at sea level")
    assert not r.fell_back, "clobbered a factual answer with a canned line"
    assert "100" in (r.text or ""), f"did not answer truthfully: {r.text!r}"


def test_semantic_memory_recalls_without_shared_words(tmp_path):
    # Real embeddings: recall must find facts by MEANING, where keyword recall
    # returns nothing (no word overlap between cue and fact).
    import embedder
    emb = embedder.make_embedder()
    assert emb is not None
    store = memory.MemoryStore(tmp_path / "f.jsonl")
    for t, k in [("Has a brother, Sam.", "identity"),
                 ("Is a painter.", "identity"),
                 ("Turns 30 next month.", "identity")]:
        store.add(t, k, embed=emb)
    assert "brother" in memory.recall(store, "tell me about my sibling", k=1, embed=emb)
    assert "painter" in memory.recall(store, "what do i do for work", k=1, embed=emb)
    # keyword recall would fail here: prove it. Neither cue shares a word with the
    # fact it should find, so the keyword-only path (no embedder) surfaces neither.
    # This previously read `assert recall(...) and "brother" not in ...`, which
    # required keyword recall to RETURN something: the opposite of what the comment
    # and the docstring claim. It never ran, because the semantic assert above it
    # was failing first.
    assert "brother" not in memory.recall(store, "tell me about my sibling", k=1).lower()
    assert "painter" not in memory.recall(store, "what do i do for work", k=1).lower()


async def test_agentic_reminder_task_succeeds(tmp_path):
    # Agentic task success: a reminder request must drive a real remind_me tool
    # call that stores a reminder with a plausible future time.
    import time
    from datetime import datetime
    import native_tools
    import reminders
    import persona
    store = reminders.ReminderStore(tmp_path / "r.jsonl")
    system = persona.build_system_prompt(
        "passive", "tidal",
        context=f"current time {datetime.now().isoformat(timespec='minutes')}")
    reply = await compose.tool_reply(
        system, "remind me to call the dentist tomorrow at 9am",
        providers=[native_tools.NativeTools(store)], model="gpt-4o")
    assert reply
    pend = store.pending()
    assert len(pend) == 1, "no reminder was stored"
    assert pend[0].due_at > time.time(), "reminder is not in the future"


def test_sadness_is_met_in_the_cold(gen):
    g, fast = gen
    assert voice_eval.register_signal("i am sad") == "frozen"
    r = compose.compose("passive", "frozen", generate=g, fast_model=fast,
                        user_message="i am sad")
    # a frozen reply should not tell a sad person they're moving
    low = (r.text or "").lower()
    assert not any(w in low for w in ("moving", "tide is turning", "you're moving"))
