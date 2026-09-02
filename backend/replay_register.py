"""replay_register.py — turn real conversations into register-dataset candidates.

evals/register_dataset.json is hand-written and small (45 train / 24 test), which
means every classifier decision is judged at about 4% per case — too coarse to
settle anything. It also missed a whole SHAPE of message: it had no search /
fetch / file / run_python phrasings, so the neutral tag read 100% while the live
Keeper was labelling "look up watercolor brands" as frozen.

Hand-writing more cases repeats that blind spot, because you invent the messages
you already think of. Replaying what was actually said does not. This reads the
persisted sessions, runs each real user message through BOTH register layers, and
writes the ones worth labelling — the model's guess included, so the human job is
to correct a label rather than compose an example.

    python backend/replay_register.py                  # summarise
    python backend/replay_register.py --out cand.json  # write candidates

Nothing is added to the dataset automatically. The predictions come from the
system under test, so promoting them unread would be marking its own homework.

PRIVACY. The output is derived from memory_store/, which .gitignore excludes
because conversation content is private. Writing candidates into the repo would
route real user messages around that rule, so evals/register_candidates.json is
itself gitignored and --out warns if you aim it somewhere tracked. Only the
reviewed evals/register_dataset.json belongs in git, and reviewing is the moment
to notice anything that should not be committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mood
import voice_eval

SESSIONS = Path(__file__).resolve().parent.parent / "memory_store" / "sessions"
DATASET = Path(__file__).resolve().parent.parent / "evals" / "register_dataset.json"


def user_messages(sessions_dir: Path = SESSIONS) -> list[str]:
    """Every distinct user turn ever said, oldest first, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for path in sorted(sessions_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for m in data.get("messages", []):
            if m.get("role") != "user":
                continue
            text = (m.get("content") or "").strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
    return out


def already_labelled(dataset: Path = DATASET) -> set[str]:
    if not dataset.exists():
        return set()
    d = json.loads(dataset.read_text())
    return {c["text"].strip().lower()
            for split in ("train", "test") for c in d.get(split, [])}


def classify_both(messages: list[str]) -> list[dict]:
    """Run both layers over each message and say why it is worth a human look."""
    signal = mood.build_local_mood_signal()
    rows = []
    for text in messages:
        keyword = voice_eval.register_signal(text)
        embedded = signal(text) if signal else None
        # The live system takes the keyword answer when there is one, else the
        # classifier's, else it inherits the previous register.
        effective = keyword or embedded
        if keyword and embedded and keyword != embedded:
            why = "layers disagree"            # the most informative cases
        elif keyword is None and embedded is not None:
            why = "classifier only"            # implicit mood, or a false positive
        elif effective is None:
            why = "no signal (inherits)"
        else:
            why = "layers agree"
        rows.append({"text": text, "keyword": keyword, "model2vec": embedded,
                     "predicted": effective, "why": why})
    return rows


def _has_labels(out: Path) -> bool:
    """True if `out` already holds human work worth not destroying."""
    if not out.exists():
        return False
    try:
        rows = json.loads(out.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return any(isinstance(r, dict) and r.get("register") for r in rows)


def _warn_if_tracked(out: Path) -> None:
    """Say something if the candidates are about to land somewhere git tracks.

    These messages come from real conversations; memory_store/ is ignored for that
    reason. A silent write into a tracked path is how private content ends up in a
    remote, so this is loud rather than clever.
    """
    import subprocess
    try:
        r = subprocess.run(["git", "check-ignore", "-q", str(out.resolve())],
                           cwd=Path(__file__).resolve().parent.parent,
                           capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return
    if r.returncode != 0:                      # 0 means "ignored"
        print(f"\n  WARNING: {out} is NOT gitignored, and these lines are real "
              f"conversation.\n           Add it to .gitignore before committing.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="write candidates as JSON")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an --out file that already carries labels")
    ap.add_argument("--all", action="store_true",
                    help="include messages already in the dataset")
    args = ap.parse_args()

    msgs = user_messages()
    known = already_labelled()
    fresh = [m for m in msgs if args.all or m.strip().lower() not in known]
    rows = classify_both(fresh)

    by_why: dict[str, int] = {}
    for r in rows:
        by_why[r["why"]] = by_why.get(r["why"], 0) + 1

    print(f"  user turns on record : {len(msgs)}")
    print(f"  already in dataset   : {len(msgs) - len(fresh)}")
    print(f"  candidates           : {len(fresh)}")
    for why, n in sorted(by_why.items(), key=lambda kv: -kv[1]):
        print(f"    {why:22} {n}")

    print("\n  worth labelling first (layers disagree, or classifier-only):")
    interesting = [r for r in rows if r["why"] in ("layers disagree",
                                                   "classifier only")]
    for r in interesting[:12]:
        print(f"    kw={str(r['keyword']):7} m2v={str(r['model2vec']):7} "
              f"{r['text'][:52]!r}")

    if args.out:
        if not args.force and _has_labels(args.out):
            raise SystemExit(
                f"  refusing to overwrite {args.out}: it already carries labels.\n"
                f"  Labelling is the expensive part — merge it into the dataset "
                f"first, or pass --force to discard it.")
        _warn_if_tracked(args.out)
        # Dataset shape, with `register` left EMPTY: a human sets it. `predicted`
        # is the system's own guess, kept only so the reviewer can agree quickly.
        cands = [{"text": r["text"], "register": "", "tag": "",
                  "predicted": r["predicted"], "why": r["why"]} for r in rows]
        args.out.write_text(json.dumps(cands, indent=2, ensure_ascii=False) + "\n")
        print(f"\n  wrote {len(cands)} candidates -> {args.out}")
        print("  set `register` (frozen|tidal|turn|neutral) and `tag` "
              "(explicit|implicit|negation|neutral), then merge into "
              "evals/register_dataset.json")


if __name__ == "__main__":
    main()
