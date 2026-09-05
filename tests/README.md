# Tests

A tiered suite. The rule: **every bug we find becomes a permanent test here**, so
it can't silently come back.

## Tiers

| Tier | Marker | What | Needs |
|---|---|---|---|
| 1 · Unit | `unit` | pure logic: energy math, voice scoring, memory recall/dedup, register + continuity, proactive gating | nothing |
| 2 · Integration | `integration` | real subsystems: OS sensors, MCP (connect/security/read-only), the reach-out→SSE delivery path | node/npx + mcp SDK (auto-skips if absent); macOS for live sensor probes |
| 3 · Evals | `eval` | LLM behavior: voice fidelity, no invented events, register-appropriateness, truthful facts | `OPENAI_API_KEY` (excluded by default) |

Tier 1/2 assert **exact** behavior. Tier 3 asserts **properties and rates** over a
sample (you can't assert an exact LLM line, but you can assert "0 invented events
in 8 proactive lines", "sad → frozen", "a plain question contains the real answer").

## Running

```bash
.venv/bin/pytest                      # unit + integration (evals excluded)
.venv/bin/pytest -m unit              # fast logic only (~0.3s)
.venv/bin/pytest -m integration       # subsystems (spins up MCP servers)
.venv/bin/pytest tests/evals -m eval  # LLM evals (needs OPENAI_API_KEY, ~1min, costs a little)
```

## The workflow for a change

1. `pytest -m unit` — start green.
2. Make the change.
3. **Write or update a test that captures the behavior** (a regression if it was a bug).
4. Run the relevant tier.
5. For persona/prompt/voice changes, run `tests/evals -m eval`.

## What's covered (regressions from real bugs)

- `test_voice_eval.py::test_invented_event_reportage_fails` — "your brother reached out"
- `test_register.py::test_continuity_holds_frozen_through_neutral_followup` — "what should i do" after sadness
- `test_evals ... test_plain_question_gets_true_answer` — facts survive the voice gate
- `test_mcp.py::test_sandbox_escape_refused` / `test_read_only_blocks_mutating_tools` — MCP safety
- `test_server.py::test_proactive_loop_reaches_out` — the battery actually delivers a line

## Every fake has a real-model twin

Three bugs this week lived where a Tier 1 test replaced the model with a fake that
returned output the real model does not produce — distill, the supersede judge,
and the goal matcher. The machinery was fine each time; the boundary was wrong,
and the suite stayed green.

So every behavioural fake in the unit tier is now paired with a Tier 3 eval that
drives the real thing:

| Faked in Tier 1 | Real-model twin |
|---|---|
| `distill` generator | `evals/test_distill.py` |
| `_supersede_judge` | `evals/test_supersede.py` |
| drift `_stub` | `evals/test_reflection.py` |
| `revoice` generator | `evals/test_revoice.py` |
| embedder (`_fake_embed`) | `evals/test_recall_gate.py` |
| `rerank`, `route`, `critique_plan`, `plan`, `_decompose` | `evals/test_model_seams.py` |

Adding a fake generator to a Tier 1 test means adding its twin here too —
otherwise the assertion is about output you invented.

## CI

| Workflow | When | Runs |
|---|---|---|
| `.github/workflows/ci.yml` | every push + PR to main | tiers 1 + 2, plus a collect-only check that the eval tier still imports |
| `.github/workflows/evals.yml` | **manual only** (`workflow_dispatch`) | tier 3 against real models, then `mood_bench.py` |

Two deliberate details. `-W error` is set in `pyproject.toml`, so a warning from
our own code fails the run (one narrow `ResourceWarning` ignore covers a
huggingface file-handle leak we do not own). And the eval job **fails if
`OPENAI_API_KEY` is missing** rather than skipping: every eval is `skipif` on that
key, so without the guard a run with no key would pass zero tests — silence that
looks exactly like success, which is the failure this tier exists to catch.

The eval tier is **not scheduled**: that would require handing GitHub a copy of
the API key, and the decision was to keep the credential on the machine that owns
it. So nothing runs these unless you do. Run them by hand before touching a
prompt, a threshold, or a parser:

```bash
.venv/bin/pytest tests/evals -m eval
```

CI installs `requirements.lock`, not `requirements.txt`. Several constants here
are calibrated against specific model versions (`memory._REL_FLOOR`, the mood
floor), so an unpinned upgrade decalibrates them silently.

## Notes

- Delivery over the live `/events` HTTP stream is verified manually with `curl`;
  ASGITransport buffers an infinite SSE stream, so the automated test asserts the
  loop→listener delivery seam directly instead.
- Silence is the loop's job (the probabilistic roll), unit-tested in
  `test_proactive.py`; the eval only checks the *system* isn't over-eager.
