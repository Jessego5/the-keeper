<p align="center">
  <img src="backend/static/keeper.png" width="180" alt="The Keeper, a hooded figure with a small lamp at the brow, one hand open">
</p>

<h1 align="center">The Keeper</h1>

<p align="center">
  <em>A proactive, memory-driven AI companion that reaches out on its own.</em><br>
  <sub>Python · FastAPI · MCP · A2A · OpenAI · 440 tests · 60 LLM evals · CI</sub>
</p>

---

Most assistants wait to be asked. The Keeper does not. A background loop decides
when it has something worth saying, and says it, through a native OS notification,
with the browser closed.

It is an old, patient presence that tends someone the way a lighthouse keeper
tends a coast: steadily, expecting nothing back. It has a fixed voice, a memory
that outlives any single conversation, and hands. When something needs finding or
working out, it goes and does it.

## Three things make it an agent, not a chatbot

**It acts unbidden.** The proactive loop is gated by an energy model, your
presence (idle time, screen lock, focused app), and whether it actually has
anything to say. Silence is a valid outcome, and a hard real-time floor stops it
ever becoming a notification flood. Simulated over 7 days and 20 seeds, the
cooldown accounts for 78% of the silences and the daily ceiling for none of them,
which is measured rather than assumed ([`proactive_bench.py`](backend/proactive_bench.py)).

**It notices.** What it can raise unbidden is not limited to your own past. It
watches sources that genuinely push, RSS feeds and its own MCP tools alike, so a
commit that landed or a file that changed can become the thing it mentions.
Everything it notices is scored against what it knows about you before it earns
the right to interrupt, and most of it never does. It will not tell you about
your own commits, because you wrote them.

**It remembers and reflects.** Facts are distilled from conversation and
retrieved by relevance, recency and importance (the *Generative Agents* function).
When something changes, the new fact **supersedes** the old one and the change is
kept as history. Idle, it synthesizes higher-level insights about you; under
memory pressure it consolidates the low-value tail (MemGPT).

## It has hands, and it is not alone

**Tools, when asked.** The passive path reaches for MCP servers you configure
(files, fetch, search, time, git) plus native tools of its own: reminders, a
journal, and `run_python` for anything that has to be worked out rather than
guessed. The proactive loop stays sealed and never touches the world unbidden.
A server that fails to start, or comes up missing a tool it declared, shows as
degraded in `/state` rather than the Keeper quietly never reaching for it.

**Goals it tends over days.** A goal is decomposed into small steps, each labelled
with who acts. The Keeper executes its own steps with its tools and returns to
yours on its own schedule, which is what separates tending something from
answering about it.

**Other agents.** It speaks [A2A](https://google.github.io/A2A/): it publishes an
Agent Card at `/.well-known/agent.json` and consults peers over JSON-RPC.
[`scripts/almanac_agent.py`](scripts/almanac_agent.py) is a second, genuinely
separate agent to talk to. Outbound calls resolve the host and refuse anything
that is not a global IP, so a peer URL cannot be used to reach your own network,
and the Keeper's own endpoint requires a bearer token.

## The voice is a contract, not a vibe

The Keeper speaks in one register at a time: `frozen` (the long cold), `tidal`
(moving), or `turn` (the ice going out). [`voice_eval.py`](backend/voice_eval.py)
scores every line against eleven rules, six mechanical (length, hedging,
sentiment, motif) and five judged by a fast model (flat declarative, calm
foreknowledge, ceremony). A line that fails is retried or replaced.

`turn` is the rarest register and cannot be produced by wording alone. It is
*earned* from the store, when a recorded reversal shows things genuinely
improved. Saying "hello" cannot buy it, and neither can saying "everything is
finally turning around for me" if nothing the store remembers changed.

## Try it

```bash
./run.sh                       # http://localhost:8790
docker compose up -d           # or containerized (see Security)

# optional, for the agent-to-agent demo
.venv/bin/python -m uvicorn scripts.almanac_agent:app --port 8791
```

Needs `backend/.env` with `OPENAI_API_KEY`. Runs offline on a stub without one.

- **chat** `/` · **dashboard** `/dashboard` · **trace** `/trace-view` · **state** `/state`
- [`FEATURES.md`](FEATURES.md), 58 features, each with a line to type to see it
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), the two loops, with diagrams
- [`docs/DEMO.md`](docs/DEMO.md), a seven-act walkthrough
- [`KEEPER_VOICE.md`](KEEPER_VOICE.md), the character spec

The **trace** (`/trace-view`) is the thing to open first: per turn it shows what
memory was injected, which register was chosen and why, every tool that fired,
and the reply. It is how you tell whether the Keeper meant it.

## Testing

Three tiers, because an LLM system needs more than unit tests:

| Tier | What | Needs |
|---|---|---|
| **unit** | pure logic: energy, voice scoring, recall, register, gating | nothing |
| **integration** | real subsystems: OS sensors, MCP servers, the reach-out path | node + uv |
| **eval** | LLM behaviour: voice fidelity, no invented events, memory contracts | an API key |

```bash
.venv/bin/pytest                      # unit + integration (evals excluded)
.venv/bin/pytest tests/evals -m eval  # the live tier
python backend/mood_bench.py          # benchmark the register classifier
python backend/proactive_bench.py     # benchmark the interruption policy
```

**Every fake has a real-model twin.** Several bugs were found where a unit test
handed the code a fake model whose output the real one never produces, so every
faked seam is now paired with an eval that drives the real thing. The rule is in
[`tests/README.md`](tests/README.md).

Design decisions are measured, not asserted:

- The register classifier is the **LLM classifier**, because `mood_bench.py`
  scored four methods on a 100-case hand-labelled dataset with a held-out test
  split, and it won at 97% against 74% for the keyword baseline. It falls back
  to keywords, then to a local Model2Vec model, when the API is unavailable.
- The supersede floor is `0.30` because a judge asked about 7 real updates and 7
  unrelated pairs was right 14 times out of 14, while the previous floor of
  `0.45` never put three of them to it at all.
- Nearly a third of the register dataset is **replayed from real sessions**
  rather than invented, because hand-written cases only hold the messages you
  already thought of. That alone dropped one classifier's neutral accuracy from
  100% to 75%, which is exactly the blind spot it existed to expose.

## Security

`run_python` gives the Keeper real code execution, and it fetches web pages in
the same turn, so a page it reads can influence the code it writes. The fences
and their limits are documented honestly in
[`backend/sandbox.py`](backend/sandbox.py).

`docker compose up` contains that: verified inside the container, there is no API
key on disk and the host filesystem is unreachable. Use it if you point the Keeper
at feeds or pages you do not trust, or to try the project without installing
anything.

**It is not the everyday posture, though.** The container is Linux, so it loses
presence sensing and the native banner, the two things that make this a companion
rather than a chat window. Run it on the host to see what it actually is; run it
in Docker when you want the fence more than the senses. `/state` reports which
surfaces are really live either way, rather than claiming ones that are not.

## Credits

The mythology (*The Keeping*), the voice rules and the character are original.
Retrieval follows Park et al. 2023 (*Generative Agents*); consolidation follows
Packer et al. 2023 (*MemGPT*); the change-aware memory is Zep/Graphiti-shaped.
The knowledge-update eval borrows its shape from LongMemEval. Tooling is
[MCP](https://modelcontextprotocol.io); agent interop is
[A2A](https://google.github.io/A2A/).
