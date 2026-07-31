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

## Notes

- Delivery over the live `/events` HTTP stream is verified manually with `curl`;
  ASGITransport buffers an infinite SSE stream, so the automated test asserts the
  loop→listener delivery seam directly instead.
- Silence is the loop's job (the probabilistic roll), unit-tested in
  `test_proactive.py`; the eval only checks the *system* isn't over-eager.
