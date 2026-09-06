"""mood_bench.py — pick the mood sensor with data, not vibes.

Benchmarks three register classifiers on a hand-labeled dataset
(evals/register_dataset.json) with a proper TRAIN/TEST split:

  keyword          voice_eval.register_signal (the current baseline)
  openai-anchor    mood.classify over OpenAI embeddings
  model2vec-anchor mood.classify over a local Model2Vec static model

Methodology: the anchor classifiers' confidence floor is tuned on TRAIN only;
accuracy is reported on the held-out TEST split, broken down by tag (explicit /
implicit / negation / neutral) so you can see WHERE embeddings beat keywords.
Latency per classify is measured too. Run:
  .venv/bin/python backend/mood_bench.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import mood
import voice_eval
from memory import _cosine

DATA = json.loads((Path(__file__).resolve().parent.parent /
                   "evals" / "register_dataset.json").read_text())
TRAIN, TEST = DATA["train"], DATA["test"]
TAGS = DATA["tags"]
FLOOR_GRID = [round(0.15 + 0.03 * i, 2) for i in range(15)]


def _gold(register: str) -> Optional[str]:
    return None if register == "neutral" else register


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _by_tag(cases, preds) -> dict[str, float]:
    out = {}
    for tag in TAGS:
        idx = [i for i, c in enumerate(cases) if c["tag"] == tag]
        if idx:
            out[tag] = _mean(preds[i] == _gold(cases[i]["register"]) for i in idx)
    return out


# ── anchor-classifier scoring (embed once, threshold cheaply) ──

def _scores(embed, anchors, texts):
    vecs = embed(texts)                                   # one batched call
    return [{r: max(_cosine(v, a) for a in av) for r, av in anchors.items()}
            for v in vecs]


def _predict(scores: dict, floor: float, margin: float = 0.03) -> Optional[str]:
    (r1, s1), (_, s2) = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    if s1 < floor or (s1 - s2) < margin:
        return None
    return None if r1 == mood.NEUTRAL else r1   # mirrors mood.classify


def tune_floor(embed, anchors) -> float:
    scores = _scores(embed, anchors, [c["text"] for c in TRAIN])
    best_acc, best_floor = 0.0, FLOOR_GRID[0]
    for f in FLOOR_GRID:
        acc = _mean(_predict(sc, f) == _gold(c["register"])
                    for sc, c in zip(scores, TRAIN, strict=True))
        if acc > best_acc:
            best_acc, best_floor = acc, f
    return best_floor


def eval_embed(embed, floor: float):
    anchors = mood.precompute_anchors(embed)
    scores = _scores(embed, anchors, [c["text"] for c in TEST])
    preds = [_predict(sc, floor) for sc in scores]
    acc = _mean(p == _gold(c["register"]) for p, c in zip(preds, TEST, strict=True))
    return acc, _by_tag(TEST, preds)


def eval_keyword():
    preds = [voice_eval.register_signal(c["text"]) for c in TEST]
    acc = _mean(p == _gold(c["register"]) for p, c in zip(preds, TEST, strict=True))
    return acc, _by_tag(TEST, preds)


# --- the LLM classifier (implementation lives in mood.py, so this measures
# exactly what runs) --- #

def eval_llm(generate):
    preds = [mood.llm_signal(c["text"], generate) for c in TEST]
    acc = _mean(p == _gold(c["register"]) for p, c in zip(preds, TEST, strict=True))
    return acc, _by_tag(TEST, preds)


def avg_llm_latency_ms(generate, n: int = 3) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        mood.llm_signal("how are you feeling today, really", generate)
    return (time.perf_counter() - t0) / n * 1000


def avg_latency_ms(embed, n: int = 5) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        embed(["how are you feeling today, really"])
    return (time.perf_counter() - t0) / n * 1000


def _openai_embedder():
    import embedder
    return embedder.make_embedder()


def _model2vec_embedder():
    import os
    # A stale HF token in the env breaks even public-model downloads; clear it.
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
        os.environ.pop(k, None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    try:
        from model2vec import StaticModel
        m = StaticModel.from_pretrained("minishlab/potion-base-8M")
    except Exception:  # noqa: BLE001 - not installed / download failed
        return None
    return lambda texts: [list(map(float, v)) for v in m.encode(texts)]


def run() -> None:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")

    rows = []  # (name, test_acc, by_tag, floor, latency_ms)
    kacc, ktag = eval_keyword()
    rows.append(("keyword", kacc, ktag, None, None))

    for name, embed in [("openai-anchor", _openai_embedder()),
                        ("model2vec-anchor", _model2vec_embedder())]:
        if embed is None:
            rows.append((name, None, None, None, None))
            continue
        floor = tune_floor(embed, mood.precompute_anchors(embed))
        acc, tag = eval_embed(embed, floor)
        rows.append((name, acc, tag, floor, avg_latency_ms(embed)))

    # The fourth option: ask a model instead of measuring distance in an embedding
    # space. No floor to tune — it either names a register or it does not.
    try:
        import compose
        _, fast = compose.make_generator()
    except Exception:  # noqa: BLE001 - no key, no network
        fast = None
    if fast is None:
        rows.append(("llm-classifier", None, None, None, None))
    else:
        acc, tag = eval_llm(fast)
        rows.append(("llm-classifier", acc, tag, None, avg_llm_latency_ms(fast)))

    print("\n" + "=" * 73)
    print(f"  MOOD SENSOR BENCHMARK   train={len(TRAIN)}  test={len(TEST)} (held out)")
    print("=" * 73)
    hdr = (f"  {'method':17} {'TEST':>5} {'implicit':>9} {'negation':>9} "
           f"{'neutral':>8} {'floor':>6} {'latency':>9}")
    print(hdr)
    print("  " + "-" * 69)
    for name, acc, tag, floor, lat in rows:
        if acc is None:
            print(f"  {name:17} {'skipped'}")
            continue
        def pc(d, k): return f"{d.get(k, 0)*100:.0f}%" if d and k in d else "  —"
        lt = f"{lat:.0f}ms" if lat else "  —"
        # The tuned floor is printed because it is the number that DRIFTS: it is
        # chosen here on TRAIN, and copied by hand into mood.build_local_mood_signal.
        # Those two silently disagreed once already; showing it makes that visible.
        fl = f"{floor:.2f}" if floor else "   —"
        print(f"  {name:17} {acc*100:>4.0f}% {pc(tag,'implicit'):>9} "
              f"{pc(tag,'negation'):>9} {pc(tag,'neutral'):>8} {fl:>6} {lt:>9}")
    print("=" * 73)

    scored = [(n, a, lat) for n, a, _, _, lat in rows if a is not None]
    win = max(scored, key=lambda x: (x[1], -(x[2] or 0)))
    print(f"  winner (test acc, then latency): {win[0]}  —  {win[1]*100:.0f}%\n")


if __name__ == "__main__":
    run()
