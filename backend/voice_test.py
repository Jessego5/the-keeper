"""voice_test.py — does a real model actually sound like the Keeper?

Generates a spread of lines across every mode and water state through the real
compose loop, prints each with its score, and reports the aggregate that no single
line can tell you: mean fidelity, pass rate, hard-fail rate, how often it needed a
retry or a fallback, how often proactive chose silence, and — corpus-wide — how
often it stated hope outright (which the voice wants kept rare).

Run:  OPENAI_API_KEY in backend/.env, then  `.venv/bin/python backend/voice_test.py`
Falls back to the offline stub if no key, so it always runs (stub numbers are not
meaningful — they just prove the harness).
"""

from __future__ import annotations

from dataclasses import dataclass

import compose

# A spread of realistic situations. Each is (mode, water_state, user_message, memory).
CASES = [
    ("passive", "frozen", "i can't get started on anything lately", ""),
    ("passive", "frozen", "everything feels stuck", "- lost the job in winter"),
    ("passive", "tidal", "had a decent day actually", ""),
    ("passive", "tidal", "i went back to the studio today", "- painter; stopped in spring"),
    ("passive", "turn", "i think things might finally be shifting", "- long hard year"),
    ("proactive", "frozen", "", "- silent for days; left a note about being tired"),
    ("proactive", "frozen", "", "- winter has been long for them"),
    ("proactive", "tidal", "", "- left a project unfinished in spring"),
    ("proactive", "tidal", "", "- mentioned a brother, Sam, twice"),
    ("proactive", "turn", "", "- last talk, the first light one in months"),
]

REPEATS = 2  # per case, to see variance and the silence/retry rates


@dataclass
class Row:
    mode: str
    state: str
    result: compose.ComposeResult


def run() -> None:
    generate, fast = compose.make_generator()
    live = fast is not None
    print(f"backend: {'OPENAI (live)' if live else 'STUB (numbers not meaningful)'}")
    print(f"cases: {len(CASES)} x {REPEATS} = {len(CASES) * REPEATS} lines\n")

    rows: list[Row] = []
    for mode, state, user, memory in CASES:
        for _ in range(REPEATS):
            r = compose.compose(
                mode, state, generate=generate, user_message=user,
                memory=memory, fast_model=fast)
            rows.append(Row(mode, state, r))
            print(f"  {mode:9} {state:6} {r}")

    _report(rows, fast)


def _report(rows: list[Row], fast) -> None:
    sent = [r for r in rows if not r.result.silent]
    scored = [r for r in sent if r.result.score is not None]
    silent = [r for r in rows if r.result.silent]
    proactive = [r for r in rows if r.mode == "proactive"]
    fell_back = [r for r in sent if r.result.fell_back]
    retried = [r for r in sent if r.result.attempts > 1]

    # overt-hope rate needs the report; re-score sent lines if we have it.
    overt = sum(
        1 for r in sent
        if r.result.report and r.result.report.overt_hope)
    hard = sum(
        1 for r in sent
        if r.result.report and r.result.report.hard_fail)

    def pct(n: int, d: int) -> str:
        return f"{(100 * n / d):.0f}%" if d else "n/a"

    mean = sum(r.result.score for r in scored) / len(scored) if scored else 0.0

    print("\n" + "=" * 52)
    print("VOICE REPORT")
    print("=" * 52)
    print(f"  lines generated      {len(rows)}")
    print(f"  sent / silent        {len(sent)} / {len(silent)}")
    print(f"  mean fidelity        {mean:.2f}   (sent, scored lines)")
    print(f"  passed first try     {pct(len(sent) - len(retried), len(sent))}")
    print(f"  needed a retry       {pct(len(retried), len(sent))}")
    print(f"  fell back to safe    {pct(len(fell_back), len(sent))}")
    print(f"  hard-failed          {pct(hard, len(sent))}")
    print(f"  proactive silence    {pct(len(silent), len(proactive))}   (of proactive)")
    print(f"  overt-hope rate      {pct(overt, len(sent))}   (keep this LOW)")
    if fast is None:
        print("\n  note: semantic layer OFF (no fast model) — scores are")
        print("  deterministic-only. Set OPENAI_API_KEY for the full rubric.")


if __name__ == "__main__":
    run()
