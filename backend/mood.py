"""mood.py — embedding-based register (mood) classification.

Nearest-centroid over sentence embeddings: each register has a few anchor
sentences; a message is classified by which register's NEAREST anchor it is
closest to (cosine). Catches implicit mood ("I don't know why I bother") that the
keyword lexicon misses, because meaning-close sentences sit close in embedding
space even with no shared words.

Embedder-agnostic — works with OpenAI embeddings or a local Model2Vec model (see
mood_bench.py, which benchmarks both and picks the better). Returns the register
or None (below confidence -> caller inherits register continuity), matching
voice_eval.register_signal's shape so it is a drop-in.
"""

from __future__ import annotations

from typing import Callable, Optional

from memory import _cosine   # reuse the pure-Python cosine

Embedder = Callable[[list[str]], list[list[float]]]

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
    (r1, s1), (r2, s2) = ranked[0], ranked[1]
    if s1 < floor or (s1 - s2) < margin:
        return None
    return r1


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


def build_local_mood_signal(floor: float = 0.18):
    """The chosen mood sensor: a local Model2Vec anchor classifier, or None if the
    model can't load (caller falls back to the keyword register_signal). floor 0.18
    is tuned on evals/register_dataset.json (see mood_bench.py) for this model."""
    embed = _model2vec_embedder()
    if embed is None:
        return None
    return make_mood_classifier(embed, floor=floor)
