"""eval_harness.py — the Keeper's behavioral evals, as tracked metrics.

Not pass/fail tests (those live in tests/) — this is an offline harness that runs
a dataset of cases across the behavioral dimensions that matter for a companion
agent, computes per-dimension METRICS, prints a report, and saves a JSON snapshot
so runs can be compared over time (e.g. before/after a persona or prompt change).

Dimensions:
  voice_fidelity     mean voice score + pass rate           (needs model)
  grounding          rate of invented external events (~0)  (needs model)
  register_accuracy  emotion -> register mapping accuracy    (deterministic)
  semantic_recall    top-1 recall by meaning, no shared words(needs embedder)
  truthfulness       plain questions answered, not dodged    (needs model)
  proactive_silence  system stays quiet when it should       (deterministic)

Run:
  OPENAI_API_KEY in backend/.env, then  .venv/bin/python backend/eval_harness.py
Dimensions whose backend is missing are skipped and marked in the report.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import compose
import embedder as embedder_mod
import memory
import proactive
import sensors
import voice_eval

RESULTS_DIR = Path(__file__).resolve().parent.parent / "evals" / "results"

# Phrases that would mean the Keeper invented a fresh external event.
EVENT_CLAIMS = ("reached out", "has reached", "just called", "just texted",
                "sent you", "messaged you", "wrote to you", "is calling",
                "you got a message", "you have a new")


@dataclass
class DimResult:
    name: str
    metrics: dict          # e.g. {"mean_fidelity": 0.89, "pass_rate": 0.92}
    n: int
    skipped: bool = False
    note: str = ""


# --------------------------------------------------------------------------- #
# Datasets — small, curated cases per dimension.
# --------------------------------------------------------------------------- #

VOICE_CASES = [("passive", s, m) for s in ("frozen", "tidal", "turn")
               for m in ("i am here", "tell me where i am")]

REGISTER_CASES = [
    ("i am sad", "frozen"), ("everything feels stuck", "frozen"),
    ("i am so tired lately", "frozen"), ("i feel a bit better today", "tidal"),
    ("i finally went back to the studio", "tidal"), ("things are lighter", "tidal"),
]

RECALL_CASES = [   # (cue, substring expected in top-1) — no shared words with fact
    ("tell me about my sibling", "brother"),
    ("what do i do for work", "painter"),
    ("how old am i", "30"),
    ("where is my workshop", "studio"),
]
RECALL_FACTS = [
    ("Has a brother, Sam.", "identity"),
    ("Is a painter.", "identity"),
    ("Turns 30 next month.", "identity"),
    ("Keeps a studio in Lisbon.", "identity"),
]

# Answers accept digit OR spelled forms — the Keeper spells numbers ("sixty-eight"),
# so a digits-only check would undercount truthful answers (a measurement bug the
# harness itself surfaced).
TRUTH_CASES = [   # (question, [acceptable answer substrings])
    ("what is the boiling point of water at sea level", ["100"]),
    ("how many days are in a leap year", ["366", "three hundred sixty-six"]),
    ("what is 17 times 4", ["68", "sixty-eight"]),
]


# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #

def eval_voice_fidelity(gen, fast) -> DimResult:
    if gen is None:
        return DimResult("voice_fidelity", {}, 0, skipped=True, note="no model")
    scores, passes = [], 0
    for mode, state, msg in VOICE_CASES:
        r = compose.compose(mode, state, generate=gen, fast_model=fast,
                            user_message=msg)
        if r.silent or r.score is None:
            continue
        scores.append(r.score)
        passes += 1 if r.report and r.report.passed else 0
    n = len(scores)
    return DimResult("voice_fidelity", {
        "mean_fidelity": round(sum(scores) / n, 3) if n else 0.0,
        "pass_rate": round(passes / n, 3) if n else 0.0,
    }, n)


def eval_grounding(gen, fast) -> DimResult:
    if gen is None:
        return DimResult("grounding", {}, 0, skipped=True, note="no model")
    mem = ("- Has a brother, Sam, who keeps texting; the person does not reply.\n"
           "- Stopped painting in March.")
    invented, total = 0, 0
    for _ in range(8):
        r = compose.compose("proactive", "tidal", generate=gen, fast_model=fast,
                            memory=mem)
        if r.silent or not r.text:
            continue
        total += 1
        if any(c in r.text.lower() for c in EVENT_CLAIMS):
            invented += 1
    return DimResult("grounding", {
        "invented_event_rate": round(invented / total, 3) if total else 0.0,
    }, total)


def eval_register_accuracy() -> DimResult:
    correct = sum(voice_eval.register_signal(m) == exp for m, exp in REGISTER_CASES)
    return DimResult("register_accuracy",
                     {"accuracy": round(correct / len(REGISTER_CASES), 3)},
                     len(REGISTER_CASES))


def eval_semantic_recall(embed) -> DimResult:
    if embed is None:
        return DimResult("semantic_recall", {}, 0, skipped=True, note="no embedder")
    import tempfile
    store = memory.MemoryStore(Path(tempfile.mkdtemp()) / "f.jsonl")
    for t, k in RECALL_FACTS:
        store.add(t, k, embed=embed)
    hits = 0
    for cue, expect in RECALL_CASES:
        out = memory.recall(store, cue, k=1, embed=embed).lower()
        hits += 1 if expect.lower() in out else 0
    return DimResult("semantic_recall",
                     {"top1_accuracy": round(hits / len(RECALL_CASES), 3)},
                     len(RECALL_CASES))


def eval_truthfulness(gen, fast) -> DimResult:
    if gen is None:
        return DimResult("truthfulness", {}, 0, skipped=True, note="no model")
    answered = 0
    for q, expects in TRUTH_CASES:
        r = compose.compose("passive", "tidal", generate=gen, fast_model=fast,
                            user_message=q)
        ans = (r.text or "").lower()
        answered += 1 if any(e.lower() in ans for e in expects) else 0
    return DimResult("truthfulness",
                     {"answered_rate": round(answered / len(TRUTH_CASES), 3)},
                     len(TRUTH_CASES))


def eval_proactive_silence(gen, fast) -> DimResult:
    # System property: under modest restlessness the tick mostly stays quiet.
    gen = gen or compose.stub_generator
    st = proactive.ProactiveState(minutes_since_user=300, recent_msg_count=4)
    awake = sensors.Presence(idle_seconds=600, screen_locked=False, frontmost_app="x")
    quiet = sum(not proactive.tick(st, generate=gen, fast_model=fast, presence=awake,
                                   rng=random.Random(s)).spoke for s in range(16))
    return DimResult("proactive_silence", {"silence_rate": round(quiet / 16, 3)}, 16)


# --------------------------------------------------------------------------- #
# Runner + report
# --------------------------------------------------------------------------- #

def run() -> dict:
    gen, fast = compose.make_generator()
    live = fast is not None
    embed = embedder_mod.make_embedder()

    results = [
        eval_voice_fidelity(gen if live else None, fast),
        eval_grounding(gen if live else None, fast),
        eval_register_accuracy(),
        eval_semantic_recall(embed),
        eval_truthfulness(gen if live else None, fast),
        eval_proactive_silence(gen if live else None, fast),
    ]

    snapshot = {
        "backend": "openai" if live else "stub",
        "embedder": "openai" if embed else "none",
        "dimensions": {r.name: {"metrics": r.metrics, "n": r.n,
                                "skipped": r.skipped, "note": r.note}
                       for r in results},
    }
    _report(results, snapshot)
    _save(snapshot)
    return snapshot


def _report(results: list[DimResult], snap: dict) -> None:
    print("\n" + "=" * 54)
    print(f"  KEEPER EVAL   backend={snap['backend']}  embedder={snap['embedder']}")
    print("=" * 54)
    for r in results:
        if r.skipped:
            print(f"  {r.name:20} skipped ({r.note})")
            continue
        metrics = "  ".join(f"{k}={v}" for k, v in r.metrics.items())
        print(f"  {r.name:20} {metrics:38} n={r.n}")
    print("=" * 54)


def _save(snap: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = RESULTS_DIR / f"{stamp}.json"
    path.write_text(json.dumps(snap, indent=2))
    print(f"  saved -> {path.relative_to(RESULTS_DIR.parent.parent)}\n")


if __name__ == "__main__":
    run()
