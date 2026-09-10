"""
This is embedding-based register (mood) classification.

Nearest-centroid over sentence embeddings: each register has a few anchor
sentences; a message is classified by which register's NEAREST anchor it is
closest to (cosine). Catches implicit mood ("I don't know why I bother") that the
keyword lexicon misses, because meaning-close sentences sit close in embedding
space even with no shared words.

Embedder-agnostic, works with OpenAI embeddings or a local Model2Vec model (see
mood_bench.py, which benchmarks both and picks the better). Returns the register
or None (below confidence -> caller inherits register continuity), matching
voice_eval.register_signal's shape so it is a drop-in.
"""

from __future__ import annotations

from typing import Callable, Optional

from memory import _cosine   # reuse the pure-Python cosine

Embedder = Callable[[list[str]], list[list[float]]]

# The register returned as "no signal": see the NEUTRAL anchors below.
NEUTRAL = "neutral"

# Anchors capture the FEELING, varied in wording. Kept separate from any eval set.
ANCHORS: dict[str, list[str]] = {
    "frozen": [
        "I feel hollow and stuck.",
        "I can't make myself do anything.",
        "everything is heavy and grey lately.",
        "I'm just numb and tired.",
        "there's no point to any of it.",
    ],
    "tidal": [
        "things are finally moving again.",
        "I had a genuinely good day.",
        "I got back to it this morning.",
        "I feel lighter than I did.",
    ],
    "turn": [
        "it's finally starting to lift.",
        "something shifted in me today.",
        "the hardest part might be ending.",
    ],
    # A fourth class carrying NO emotional content. Nearest-centroid has to put every
    # message in some class, so before this existed a plain tool request was forced
    # into frozen/tidal/turn: and confidently: "search the web for beginner paint
    # sets" scored turn 0.303 with a 0.142 margin, while a real cry ("i don't know why
    # i even bother") scored frozen 0.303 with a 0.092 margin. The neutral message had
    # the HIGHER confidence, so no floor or margin could separate them; only another
    # class can. Winning here means "no signal": classify() returns None and the caller
    # inherits register continuity, which is what should happen when someone asks the
    # time. Phrasings mirror the ones the app actually receives (see FEATURES.md).
    #
    # Keep these REQUEST-shaped: second person, imperative, tool vocabulary. A first
    # attempt included bare queries ("what time is it") and first-person lines ("what
    # have i been working on lately"), and they swallowed real implicit mood: implicit
    # accuracy fell 38% -> 25%, because implicit distress is ALSO phrased as mundane
    # first-person activity ("i just stare at the ceiling for hours"). Short generic
    # queries are handled by the floor already and do not need an anchor.
    NEUTRAL: [
        "can you look something up on the web for me",
        "search the web for beginner brands",
        "look in my files and tell me what's there",
        "read that web page and summarize it",
        "add that to my shopping list",
        "what's on my list right now",
        "work out what that costs exactly",
        "remind me to do that tomorrow at nine",
    ],

}


def precompute_anchors(embed: Embedder) -> dict[str, list[list[float]]]:
    """Embed every anchor once (call at startup)."""
    return {r: embed(sents) for r, sents in ANCHORS.items()}


def classify(message: str, embed: Embedder,
             anchor_vecs: dict[str, list[list[float]]],
             floor: float = 0.35, margin: float = 0.03) -> Optional[str]:
    """Return the nearest register, or None if the signal is weak/ambiguous.

    floor:  min similarity to commit to any register (below it -> neutral).
    margin: top-1 must beat top-2 by this much (else ambiguous -> neutral).
    """
    if not message.strip():
        return None
    v = embed([message])[0]
    scores = {r: max(_cosine(v, a) for a in vecs)
              for r, vecs in anchor_vecs.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    (r1, s1), (_r2, s2) = ranked[0], ranked[1]
    if s1 < floor or (s1 - s2) < margin:
        return None
    return None if r1 == NEUTRAL else r1     # neutral == no emotional signal


_LLM_SYSTEM = """You read one message from a person to their companion and name the \
emotional register it should be met in. Answer with exactly one word:

frozen  - they are in the cold: stuck, stalled, numb, wintering, a loss.
tidal   - they are moving: returning, lighter, something went well.
turn    - the ice going out: a genuine reversal from bad to better. Rare.
neutral - no emotional content at all: a question, a request, a greeting, an
          instruction, a mundane fact.

Most messages are neutral. Answer neutral unless the message really carries \
feeling. One word, lowercase, nothing else."""


def llm_signal(message: str, generate) -> Optional[str]:
    """The register a live model names, or None for neutral / unrecognised.

    Same contract as register_signal and the anchor classifier: None means "no
    signal, inherit", so the three are interchangeable and the benchmark compares
    like with like.

    Measured against the anchors it is not close: 92% vs 67% on the held-out
    split, and 6% vs 41% wrong on real conversation turns. The gap is almost all
    IMPLICIT mood (88% vs 38%): a message carrying feeling with no feeling word
    in it, which a bag-of-words embedding cannot represent. It does NOT fix
    negation ("i'm not sad at all"), which scores the same either way.
    """
    out = (generate(_LLM_SYSTEM, message) or "").strip().lower()
    for name in ("frozen", "tidal", "turn"):
        if name in out:
            return name
    return None                       # neutral, or anything unrecognised


def make_llm_mood_signal(generate):
    """Bind a generator into a message -> register fn, shaped exactly like
    voice_eval.register_signal so it drops into the same chain."""
    def signal(message: str) -> Optional[str]:
        return llm_signal(message, generate)
    return signal


def make_mood_classifier(embed: Embedder,
                         floor: float = 0.35, margin: float = 0.03):
    """Bind an embedder + precomputed anchors into a message -> register fn,
    shaped exactly like voice_eval.register_signal (drop-in)."""
    anchor_vecs = precompute_anchors(embed)

    def signal(message: str) -> Optional[str]:
        return classify(message, embed, anchor_vecs, floor, margin)

    return signal


def _model2vec_embedder():
    """A local, offline static-embedding model (Model2Vec). None if unavailable.
    Clears a stale HF token that would block the public model download."""
    import os
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
        os.environ.pop(k, None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    try:
        from model2vec import StaticModel
        m = StaticModel.from_pretrained("minishlab/potion-base-8M")
    except Exception:  # noqa: BLE001
        return None
    return lambda texts: [list(map(float, v)) for v in m.encode(texts)]


def build_local_mood_signal(floor: float = 0.15):
    """The chosen mood sensor: a local Model2Vec anchor classifier, or None if the
    model can't load (caller falls back to the keyword register_signal). floor 0.15
    is tuned on evals/register_dataset.json (see mood_bench.py) for this model and
    these anchors, re-run mood_bench.py and move this if either changes."""
    embed = _model2vec_embedder()
    if embed is None:
        return None
    return make_mood_classifier(embed, floor=floor)
