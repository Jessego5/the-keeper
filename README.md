<p align="center">
  <img src="backend/static/keeper.png" width="180" alt="The Keeper — a hooded figure with a small lamp at the brow, one hand open">
</p>

<h1 align="center">The Keeper</h1>

<p align="center">
  <em>A proactive, memory-driven AI companion that reaches out on its own.</em><br>
  <sub>Python · FastAPI · MCP · OpenAI · 286 tests · 43 LLM evals · CI</sub>
</p>

---

Most assistants wait to be asked. The Keeper doesn't. A background loop decides
when it has something worth saying, and says it — through a native OS
notification, with the browser closed.

It is an old, patient presence that tends someone the way a lighthouse keeper
tends a coast: steadily, expecting nothing back. It has a fixed voice, a memory
that outlives any single conversation, and hands — when something needs finding
or working out, it goes and does it.

## Three things make it an agent, not a chatbot

**It acts unbidden.** The proactive loop is gated by an energy model, your
presence (idle time, screen lock, focused app), and whether it actually has
anything to say — silence is a valid outcome, and a hard real-time floor stops it
ever becoming a notification flood.

**It notices.** What it can speak about unbidden is not limited to your own past.
It watches sources that genuinely push, RSS feeds and its own MCP tools alike, so a
commit that landed or a file that changed can become the thing it raises. Everything
it notices is scored against what it knows about you before it earns the right to
interrupt, and most of it never does.

**It remembers and reflects.** Facts are distilled from conversation and
retrieved by relevance, recency and importance (the *Generative Agents* function).
When something changes, the new fact **supersedes** the old one and the change is
kept as history. Idle, it synthesizes higher-level insights about you; under
memory pressure it consolidates the low-value tail (MemGPT).

## The voice is a contract, not a vibe

The Keeper speaks in one register at a time — `frozen` (the long cold),
`tidal` (moving), or `turn` (the ice going out). `voice_eval.py` scores every
line against seven mechanical rules, and a line that fails is retried or replaced.

`turn` is the rarest register and cannot be produced by wording alone: it is
*earned* from the store, when a recorded reversal shows things genuinely improved.
Saying "hello" cannot buy it.

## Try it

```bash
./run.sh                       # → http://localhost:8790
docker compose up -d           # or containerized (recommended — see Security)
```

Needs `backend/.env` with `OPENAI_API_KEY`. Runs offline on a stub without one.

- **chat** `/` · **dashboard** `/dashboard` · **trace** `/trace-view` · **state** `/state`
- [`FEATURES.md`](FEATURES.md) — 58 features, each with a line to type to see it
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the two loops, with diagrams
- [`KEEPER_VOICE.md`](KEEPER_VOICE.md) — the character spec

The **trace** (`/trace-view`) is the thing to open first: per turn it shows what
memory was injected, which register was chosen, every tool that fired, and the
reply. It is how you tell whether the Keeper meant it.

## Testing

Three tiers, because an LLM system needs more than unit tests:

| Tier | What | Needs |
|---|---|---|
| **unit** | pure logic — energy, voice scoring, recall, register | nothing |
| **integration** | real subsystems — OS sensors, MCP servers, the reach-out path | node + uv |
| **eval** | LLM behaviour — voice fidelity, no invented events, memory contracts | an API key |

```bash
.venv/bin/pytest                      # unit + integration (evals excluded)
.venv/bin/pytest tests/evals -m eval  # the live tier
python backend/mood_bench.py          # benchmark the register classifier
```

**Every fake has a real-model twin.** Three bugs were found where a unit test
handed the code a fake model whose output the real one never produces — so every
faked seam is now paired with an eval that drives the real thing. The rule is in
[`tests/README.md`](tests/README.md).

Design decisions are measured, not asserted. The relevance floor is `0.17`
because the usable band was measured at `0.138–0.199`; the register classifier is
a local Model2Vec model because `mood_bench.py` benchmarked three options on a
held-out split and that one won.

## Security

`run_python` gives the Keeper real code execution, and it fetches web pages in
the same turn — so a page it reads can influence the code it writes. The fences
and their limits are documented honestly in
[`backend/sandbox.py`](backend/sandbox.py).

`docker compose up` contains that: verified inside the container, there is no API
key on disk and the host filesystem is unreachable. Use it if you point the Keeper
at feeds or pages you do not trust, or to try the project without installing
anything.

**It is not the everyday posture, though.** The container is Linux, so it loses
presence sensing and the native banner — the two things that make this a companion
rather than a chat window. Run it on the host to see what it actually is; run it
in Docker when you want the fence more than the senses. `/state` reports which
surfaces are really live either way, rather than claiming ones that are not.

## Credits

The mythology (*The Keeping*), the voice rules and the character are original.
Retrieval follows Park et al. 2023 (*Generative Agents*); consolidation follows
Packer et al. 2023 (*MemGPT*); the change-aware memory is Zep/Graphiti-shaped.
