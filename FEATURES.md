 # The Keeper: every feature, and how to try it

A complete, checkable inventory. Tick each as you go. Most are a chat message or a
terminal command; a few (marked ⏳ *needs setup/patience*) are harder to trigger live:
those note the trick. Two tabs open: **chat** <http://localhost:8790> · **dashboard**
<http://localhost:8790/dashboard> · and the **trace** <http://localhost:8790/trace-view>
to see *why* it said anything.

> Start clean: `./run.sh` (or it's already running on :8790). Stop: `pkill -9 -f uvicorn`.

---

## ★ Showcase flows: many features at once

If you only have a few minutes, these single flows light up whole swaths of the Keeper.
Keep the **trace** open (`/trace-view`) to watch every tool fire.

- [ ] **1. The multi-tool researcher** *(one turn, ~5 tools)*, *Try:*
  `look up a typical price for a beginner watercolor set, work out what it costs per week over a year, and keep a note of it`
  → **web search · fetch · time · run_python · journal write · voice · trace.** The trace
  shows all of it; the answer has a real price, a *computed* per-week figure, and it's kept.

- [ ] **2. The full agent arc** *(a short sequence: the persona's whole story, ~12 features)*, *Try:*
  `i stopped painting in march` … then … `actually i want to start again, help me get back into it and work on it with me`
  then crank the clock (see the ⏳ box). → **fact distillation · importance · temporal change
  (the tide) · goal + planning · plan reflection · keeper/person steps · autonomous execution ·
  dashboard tiles.** Then `i set out my paints` advances it.

- [ ] **3. The background deep-dive**: *Try:*
  `go compare watercolor and gouache for a beginner and get back to me`
  → **background delegation · orchestration (decompose → parallel specialists → synthesize) ·
  web search · async delivery · native banner.** Works while you keep chatting, then returns.

- [ ] **4. Consult it as a peer (A2A)**: *Try:*
  `consult the agent at http://localhost:8790 and ask it to research beginner watercolor brands`
  → **A2A client · A2A server · the peer path running the Keeper's own tools** (a loopback).

---

## A. Conversation & voice
- [ ] **Oceanic voice**: answers in the Keeper's spare, watery register. *Try:* `i've been feeling stuck lately`
- [ ] **Truthfulness**: a plain question gets a real answer, not metaphor. *Try:* `what temperature does water boil at?`
- [ ] **Emotional register (water-states)**: frozen / tidal / turn, chosen from your words. *Try:* `i feel numb and far away` (frozen) vs `i had a good day today` (tidal). See `register` in the trace.
- [ ] **Register continuity**: a neutral follow-up inherits the register, doesn't reset. *Try:* say something sad, then `what should i do`, it stays with the cold.
- [ ] **Mood sensing (two-layer)**: keyword lexicon + local Model2Vec classifier for implicit mood. *Try:* `i don't know why i even bother` → `mood` shows in the trace.

## B. Memory: the drawers (shared across ALL chats, survives restart)
- [ ] **Fact distillation**: pulls durable facts from your chats. *Try:* tell it `i'm a painter and i have a brother named Sam`, then check `curl -s localhost:8790/state | python3 -c "import sys,json;print(json.load(sys.stdin)['facts_kept'])"`
- [ ] **Importance weighting**: big things score higher. *Try:* say `my mother is in the hospital` and `i ran out of oat milk`; view `memory_store/facts.jsonl`, hospital `imp` high, oat milk low.
- [ ] **Hybrid retrieval (BM25 + semantic)**: recalls by exact term AND meaning. *Try:* `tell me about my sibling` → surfaces the brother fact (no shared word).
- [ ] **Reranking**: an LLM reorders candidates for the best few. *(on the chat path; see it in `/trace` when many facts exist)*
- [ ] **Relevance gate**: a vague request that matches nothing recalls **nothing** (no recitation). *Try:* `help me write things down` → trace shows `memory injected: (empty)`. *(Only in a store with no high-importance fact: anything scored ≥8 is **ambient** by design and stays in mind regardless of the cue, so run this before the importance item, or on a fresh store.)*
- [ ] **Semantic dedup**: the same fact reworded doesn't duplicate. *Try:* `Sam is my brother` then `my brother is called Sam` → still one fact.
- [ ] **Temporal / change-aware (the tide)**: a fact that updates an old one supersedes it, history kept. *Try:* `i stopped painting in march` then `actually i started painting again` → ask `how's my painting going?`; `changes_tracked` rises in `/state`.
- [ ] **Cross-session memory**: facts from one conversation surface in another. *Try:* mention something, click **+ new** in the sidebar, ask about it in the fresh chat.
- [ ] **Reflection → insights** ⏳, idle, it synthesizes higher-level reads of you, stored retrievably. *Trigger:* have ≥3 facts, then crank the clock (see §F), watch the **insights** tile / **latest reflection** on the dashboard.
- [ ] **Consolidation (MemGPT)** ⏳, over ~60 facts, old low-value ones are summarized + archived. *Hard to hit by hand; verified by tests. Ask me to seed facts if you want to see it.*

## C. Tools (MCP + native)
- [ ] **Read your files** (MCP, sandboxed), *Try:* `look in my files and tell me what's on my to-do list` (reads `keeper_sandbox/list.txt`). *Avoid a bare "what's on my list?", that's ambiguous with the reminders "list."*
- [ ] **Web search** (MCP), *Try:* `search the web for beginner watercolor brands`
- [ ] **Fetch a page** (MCP), *Try:* `read https://example.com and tell me what it's for`
- [ ] **Git history** (MCP), *Try:* `what have i been building on this project lately?`
- [ ] **Time / timezone** (MCP), *Try:* `what day is it, and what time in Tokyo?`
- [ ] **Read-only safety**: writes/escapes are blocked. *Try (terminal):* the snippet in `TRY_IT.md §4` → `no such tool: files__write_file`.
- [ ] **Run Python (code-as-action)**: computes exactly. *Try:* `if i save $45 a week, how much over 3 years? work it out exactly`
- [ ] **Journal (its one write)**: append-only. *Try:* `keep a note: the gallery show is in July`, then `what's in your journal?`. *(Ask for the journal by name, a bare "what have you kept?" is ambiguous with the memory it also keeps, and answers from facts instead.)*

## D. The agent: goals it pursues over time
- [ ] **Goal + planning**: decomposes a wish into steps. *Try:* `help me get back to painting, work on it with me`
- [ ] **Plan reflection**: the plan is self-critiqued/revised before it's set. *(happens inside set_goal)*
- [ ] **keeper vs person steps**: labels who does each. *See:* `next_actor` in `/state` goals.
- [ ] **Autonomous step execution (ReAct)** ⏳: it *does* its own steps with tools. *Trigger:* crank speed (§F); watch the terminal for `[goal] … EXECUTED via …`.
- [ ] **Nudges your steps** ⏳, invites you to do person-steps. `[goal] … nudged …`
- [ ] **Report progress**: *Try:* after a goal exists, `i set out my paints` → the panel advances. *(It marks the goal's CURRENT step done, whatever that step is, the real plan is model-written, so its early steps may say nothing about paints. Check `next_step` in `/state` to see what moved.)*
- [ ] **Complete / set down a goal**: *Try:* `i'm setting the painting goal aside for now`

## E. Multi-agent (specialists)
- [ ] **Specialists**: researcher / archivist / scribe / analyst. *Try:* `have your researcher dig into what a watercolor setup costs`
- [ ] **Delegate (route to one)**: picks the right specialist. *(same as above)*
- [ ] **Orchestration (decompose → parallel → synthesize)**: *Try:* `go compare watercolor and gouache for a beginner and get back to me` → terminal shows `[bg] … started`, then parallel specialists and one synthesized answer. *(The inline phrasing often answers in one turn with search + run_python instead of delegating, orchestration lives on the background path.)*
- [ ] **Background delegation (come back later)**: *Try:* `go look into watercolor vs gouache and get back to me` → immediate ack, `working_on` in `/state`, a new line + banner minutes later.
- [ ] **Graceful step budgets**: a specialist running out of steps wraps up cleanly. *(internal; verified by behavior)*
- [ ] **A2A: the Keeper as an agent**: *Try (terminal):* `curl -s localhost:8790/.well-known/agent.json | python3 -m json.tool` (its Agent Card).
- [ ] **A2A, consult another agent**: *Try:* `consult the agent at http://localhost:8790 and ask what it can help with` (loopback).

## F. Proactivity, presence & the house
- [ ] **Reaches out on its own** ⏳, *Trigger:* `curl -s -X POST localhost:8790/config -H 'content-type: application/json' -d '{"speed":600,"cooldown_min":5}'` then wait ~40s. Reset with `{"speed":1}`.
- [ ] **Native notification**: the banner from the above shows the Keeper's face, works browser-closed.
- [ ] **The banner wears the register**: frozen, tidal and turn each have their own pose. *See:* the three in the README's closing section.
- [ ] **Presence sensing**: idle time, screen lock, focused app. *See:* the **"the house"** panel; lock your screen (⌃⌘Q) and it flips to *away*.
- [ ] **Routines** ⏳, welcome_back / heads_down / wind_down. *Hard to trigger live (need presence transitions / long focus / evening); verified by tests.*
- [ ] **Circuit breaker**: can't reach out more than once/minute no matter the speed. *(the reason the speed knob is safe)*
- [ ] **Reminders (one-shot)**: *Try:* `remind me to call the dentist tomorrow at 9am`
- [ ] **Recurring reminders**: *Try:* `remind me to water the plants every day at 9am` → sub-line shows `recurring` on the dashboard.
- [ ] **Returns a reminder when due** ⏳, *Trigger:* `remind me to breathe in 1 minute`, then crank speed, it comes back with a banner.

## G. Delivery channels
- [ ] **Web (SSE)**: proactive lines appear in the open chat tab.
- [ ] **Native banner**: macOS notification (see §F).
- [ ] **Telegram (opt-in)** ⏳, set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `backend/.env` (BotFather), restart → proactive lines hit your phone too. `channels` in `/state` shows `telegram`.

## H. Observability
- [ ] **Dashboard**: <http://localhost:8790/dashboard> (energy, memory, goals, house, reminders, reflection).
- [ ] **Trace**: <http://localhost:8790/trace-view>, per turn: memory injected, register, tools fired, reply. *The bug-finding tool.*
- [ ] **/state**: the whole live state as JSON.
- [ ] **Conversation sidebar**: multiple chats, each persisted; **+ new** to start one, click to switch.

## I. Under the hood
- [ ] **Voice fidelity harness**: `.venv/bin/python backend/voice_test.py` (scores lines across every mode).
- [ ] **Test suite**: `.venv/bin/python -m pytest -q` (~233 tests, offline; evals excluded by default).

---

### The ⏳ ones (need setup/patience)
Reflection, autonomous goal execution, proactive reach-out, and due-reminder delivery all
run on the Keeper's slow internal clock. To see them in seconds, **crank the demo clock**:
```bash
curl -s -X POST localhost:8790/config -H 'content-type: application/json' -d '{"speed":600,"cooldown_min":5}'
# … watch the dashboard + the terminal running the server … then:
curl -s -X POST localhost:8790/config -H 'content-type: application/json' -d '{"speed":1}'
```
Routines and consolidation are the only two genuinely hard to trigger by hand (they need
real presence transitions / 60+ facts). They're covered by the test suite, ask me if you
want a quick way to force them.
