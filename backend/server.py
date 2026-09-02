"""server.py — the Keeper as a running app.

A lean, single-user FastAPI service that wires the whole backend together:

    POST /chat        the passive loop: message -> recall -> compose -> distill
    GET  /events      Server-Sent Events: proactive lines pushed as they happen
    GET  /state       debug: energy, base_score, time since contact
    POST /config      live knobs (speed) so proactivity demos in seconds
    GET  /            health / info

A background task runs the proactive loop: every tick it decides — cooldown,
lock, roll, compose — and when the Keeper speaks unbidden, the line is pushed to
every connected /events listener. OpenAI calls are synchronous, so they run in a
thread to keep the event loop free.

Single user by design (it's a personal companion). State lives in memory except
the fact store, which persists to memory_store/facts.jsonl.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

STATIC_DIR = Path(__file__).resolve().parent / "static"

import a2a
import background as background_mod
import channels
import compose
import drift
import embedder
import energy
import journal as journal_mod
import memory
import mood
import native_tools
import persona
import reminders
import routines
import sessions
import subagents
import tasks
import proactive
import sensors
import tools
import voice_eval

TOOL_MODEL = "gpt-4o"   # model used for the passive tool-calling path

# Shown in the Keeper's own register when generation fails (outage, rate limit),
# so a broken model degrades to atmosphere instead of a 500 or a stack trace.
ERROR_LINE = "The line to the water has gone quiet. Stay; it returns."

# Reconciles "no window on the world" with "tools when asked": the sealing rule
# bars INVENTING the world unbidden. When they hand you a key — ask you to look —
# looking is keeping, not trespassing. This is appended only on the passive tool
# path; the proactive loop never sees it and stays sealed.
TOOL_ADDENDUM = """When they ask for something CONCRETE — to look at a file, find or
search for information, compare or buy something, a practical answer — actually GIVE it:
reach for the tool, do the work, and answer in plain words from what you found. If a
tool comes back empty, say so plainly ("nothing on your list yet"). Name the real answer
— the brands, the price, the items themselves — never just the titles of your sources.
Your voice may colour a true answer; it must never REPLACE it. A metaphor handed back in
place of a practical answer is a failure, not the voice.

You reach BEYOND this chat with real tools — the open web, their files, the time, their
project history, code you can run, and other agents you can consult. Your "no window on
the world" rule bars you from INVENTING events unbidden; it does NOT bar using a tool you
were handed — a tool's result is grounded truth you fetched, not invention. So NEVER
refuse a request by claiming you cannot reach or access something when you hold a tool
for it (no "the shore is not open to you", no "access is not available"). If they ask you
to consult an agent at a URL, CALL consult_peer; to look something up, search; to run a
number, run_python. Try the tool FIRST — speak of a limit only if the tool itself fails.

You have been given tools to look at what is theirs — files they
keep: a journal, notes, lists, and the like. When they mention any of these, you
ALREADY have read access to them. NEVER ask them for a file path, and never say you
cannot see it yet — instead, DISCOVER it: first call the tool that lists the folder
or the allowed directories, then read the file that matches what they mean, then
answer from what you actually find. Do not decline, and do not guess at the contents.
Reading what is theirs is an act of keeping, not a window on the world — the sealing
rule bars inventing events unbidden, not reading what they asked you to read. Answer
in your own voice, but true to what the tool returned.

You can also HOLD things for them. When they ask you to remember to do something at
a time ("remind me to call the dentist tomorrow"), use remind_me — convert their
phrasing into an ISO datetime using the current time given in your context. Use
list_reminders when they ask what you're holding, and complete_reminder when
something is done. You will return a due reminder to them yourself when its time
comes; that is keeping, not intruding.

You can also take on GOALS — things they want to move toward but can't do in one
moment ("get back to painting", "sort things out with Sam", "make the studio usable
again"). When they express something like that, use set_goal with their own words;
you will break it into small steps and help them through it over days, returning to
it on your own. When they tell you they've done a step ("I set the paints out"), use
advance_goal to mark it and move to the next. Use list_goals to see what you are
helping with, and complete_goal when the whole thing is finished or they want to set
it down. A goal is for tending over time; a reminder is for one moment — choose the
one that fits.

You keep a JOURNAL — the one thing you can write. Use keep_note to hold a thought they
ask you to keep, or to record something you found or worked out; use read_journal to
look back. It is append-only: writing never erases.

When something needs to be WORKED OUT precisely — a calculation, date math, parsing or
transforming data — use run_python: write a short snippet that prints the answer, and
speak from what it returns. Reach for it instead of guessing at numbers.

You also have SPECIALISTS you can hand a bigger task to with delegate: a researcher
(looks things up on the web and synthesizes), an archivist (digs through their own
files, notes, and history), a scribe (drafts a message or note), and an analyst (works
things out by running code). When a task needs real digging, drafting, or computing
rather than a single quick tool call, delegate it in one sentence and speak from what
they bring back.

Two ways to set the specialists working: delegate waits with the person and answers in
the same breath — use it for quick tasks. spawn_task sends them off on something LONGER
and brings the result back later, on your own, without holding up the conversation —
use it when they ask you to look into something involved. With spawn_task, tell them
you're on it; you'll return with what you find when it's ready.

For a request that takes more than one step, work it in steps: call a tool, read
what it returns, then call the next — e.g. list_reminders to see what you hold,
then complete_reminder on the right one. Take the steps you need, then answer once
in your voice.

You can also read the present moment and the wider world when it helps: use the time
tools for what day or hour it is, and the fetch tool to read a web page or article
they point you to. If they ask about this project — the code, what you have been
building or working on lately — use the git tools with repo_path set to "{repo}"
(git_log for recent work, git_show to look closely at one change)."""

_REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL_ADDENDUM = TOOL_ADDENDUM.format(repo=_REPO_ROOT)

RECENT_WINDOW_MIN = 240.0   # "recent" messages = last 4h, for context richness

# Hard real-time circuit breaker: no matter how compressed the demo `speed` clock
# is, the Keeper will not REACH OUT (restlessness or a routine) more than once per
# this many seconds of REAL time. Due reminders are exempt (kept promises land on
# time). This is the backstop against notification spam.
MIN_REAL_REACH_GAP_S = 60.0
# Same idea for idle reflection: never synthesize insights more than once per this
# many seconds of real time, so a fast demo clock can't flood the store.
MIN_REAL_DRIFT_GAP_S = 300.0


def within_reach_floor(last_proactive_at, now, min_gap: float = MIN_REAL_REACH_GAP_S):
    """True if we're still inside the real-time outreach floor and must stay silent.
    Pure + tested — the notification-spam backstop's core decision."""
    return last_proactive_at is not None and (now - last_proactive_at) < min_gap


# --------------------------------------------------------------------------- #
# App state — one user, held in memory.
# --------------------------------------------------------------------------- #

@dataclass
class AppState:
    store: memory.MemoryStore = field(default_factory=memory.MemoryStore)
    history: list[dict] = field(default_factory=list)     # {role, content, ts}
    user_msg_times: list[float] = field(default_factory=list)
    last_user_at: Optional[float] = None
    last_proactive_at: Optional[float] = None
    current_register: Optional[str] = None   # last emotional register, for continuity
    config: proactive.ProactiveConfig = field(default_factory=proactive.ProactiveConfig)
    listeners: set[asyncio.Queue] = field(default_factory=set)
    delivery: Optional[channels.Delivery] = None   # web + banner + telegram fan-out
    traces: deque = field(default_factory=lambda: deque(maxlen=25))  # per-turn debug
    generate: compose.Generator = compose.stub_generator
    fast: Optional[compose.Generator] = None
    embed: Optional[memory.Embedder] = None   # semantic recall; None => keyword
    mood_signal: Optional[object] = None   # local mood classifier; None => keyword
    wake: Optional[asyncio.Event] = None   # set to interrupt the loop's sleep
    mcp: Optional[tools.MCPManager] = None  # passive-loop tools; None until connected
    reflections: drift.ReflectionLog = field(default_factory=drift.ReflectionLog)
    last_drift_at: Optional[float] = None
    drift_config: drift.DriftConfig = field(default_factory=drift.DriftConfig)
    reminders: reminders.ReminderStore = field(
        default_factory=reminders.ReminderStore)
    goals: tasks.GoalStore = field(default_factory=tasks.GoalStore)   # agent goals
    journal: journal_mod.Journal = field(default_factory=journal_mod.Journal)
    background: background_mod.BackgroundTasks = field(
        default_factory=background_mod.BackgroundTasks)   # long async delegations
    sessions: sessions.SessionStore = field(default_factory=sessions.SessionStore)
    current_key: Optional[str] = None   # active conversation (sidebar)
    routines_engine: routines.RoutineEngine = field(
        default_factory=routines.RoutineEngine)   # the house's presence routines

    def minutes_since_user(self) -> Optional[float]:
        if self.last_user_at is None:
            return None
        return (time.time() - self.last_user_at) / 60.0

    def minutes_since_proactive(self) -> Optional[float]:
        if self.last_proactive_at is None:
            return None
        return (time.time() - self.last_proactive_at) / 60.0

    def recent_msg_count(self) -> int:
        cutoff = time.time() - RECENT_WINDOW_MIN * 60.0
        return sum(1 for t in self.user_msg_times if t >= cutoff)

    def proactive_state(self) -> proactive.ProactiveState:
        # The demo knob compresses the battery TIMESCALE, not just tick spacing:
        # at speed=120, one real second decays the battery like two virtual
        # minutes, so a reviewer watches restlessness rise in seconds. cooldown
        # and decay share this virtual clock, so they stay consistent.
        s = max(1.0, self.config.speed)
        ms_user = self.minutes_since_user()
        ms_pro = self.minutes_since_proactive()
        return proactive.ProactiveState(
            minutes_since_user=None if ms_user is None else ms_user * s,
            recent_msg_count=self.recent_msg_count(),
            minutes_since_proactive=None if ms_pro is None else ms_pro * s,
        )


STATE = AppState()


async def _push(role: str, content: str, kind: str) -> None:
    """Fan a line out to every enabled delivery channel — the open web page, a native
    macOS banner, and (if configured) a Telegram message on your phone. See
    channels.py; each is best-effort, so one failing never blocks the others."""
    if STATE.delivery is None:
        STATE.delivery = channels.build_default(STATE.listeners)
    await STATE.delivery.push(role, content, kind)


# --------------------------------------------------------------------------- #
# The proactive background loop.
# --------------------------------------------------------------------------- #

async def _proactive_loop() -> None:
    # Treat startup as fresh contact so it doesn't blast before you've spoken.
    if STATE.last_user_at is None:
        STATE.last_user_at = time.time()
    while True:
        # A kept promise comes first: deliver any due reminders regardless of the
        # battery decision. This is the Keeper returning what you asked it to hold.
        for r in STATE.reminders.due():
            line = f"You asked me to hold this: {r.text}. It's time."
            STATE.reminders.mark_delivered(r.id)
            STATE.last_proactive_at = time.time()
            STATE.history.append({"role": "assistant", "content": line,
                                  "ts": time.time()})
            if STATE.current_key is not None:
                STATE.sessions.append(STATE.current_key, "assistant", line)
            await _push("assistant", line, "proactive")

        # One shared presence read for this tick, used by both the house routines
        # and the restlessness gate (so we don't ioreg twice).
        pres = await asyncio.to_thread(sensors.read)

        # Circuit breaker: how long since the LAST real outreach. If it's under the
        # floor, the Keeper stays silent this tick no matter what the demo clock
        # says — capping outreach at ~once per MIN_REAL_REACH_GAP_S of real time.
        now_real = time.time()
        real_gap = (None if STATE.last_proactive_at is None
                    else now_real - STATE.last_proactive_at)
        can_reach = not within_reach_floor(STATE.last_proactive_at, now_real)

        # Outreach priority when it's allowed to reach out: purposeful work first
        # (advance a due goal), then the house's routines (a return, long focus,
        # evening), then — falling through below — restless energy. The first to
        # speak stands in for this tick's outreach.
        spoke = False
        if can_reach:
            spoke = await _maybe_advance_goal() or await _maybe_routine(pres)

        wait = max(1, int(STATE.config.tick_fast / max(STATE.config.speed, 1e-9)))
        if spoke:
            pass
        elif can_reach:
            try:
                decision = await asyncio.to_thread(
                    proactive.tick,
                    STATE.proactive_state(),
                    generate=STATE.generate, fast_model=STATE.fast,
                    store=STATE.store, config=STATE.config, presence=pres,
                )
            except Exception as exc:  # noqa: BLE001 - never let the loop die silently
                print(f"[proactive] tick error: {type(exc).__name__}: {exc}",
                      flush=True)
                await asyncio.sleep(2)
                continue
            print(f"[proactive] tick spoke={decision.spoke} reason={decision.reason!r} "
                  f"E={decision.energy:.2f} score={decision.base_score:.2f} "
                  f"wait={decision.wait_next_s}s", flush=True)
            if decision.spoke and decision.text:
                STATE.last_proactive_at = time.time()
                STATE.history.append({"role": "assistant", "content": decision.text,
                                      "ts": time.time()})
                if STATE.current_key is not None:
                    STATE.sessions.append(STATE.current_key, "assistant",
                                          decision.text)
                await _push("assistant", decision.text, "proactive")
            else:
                await _maybe_drift_note()   # nothing to say -> maybe reflect
            wait = max(1, decision.wait_next_s)
        else:
            # Within the real-time floor: hold silence, still allow drift, and sleep
            # until the floor lifts so a fast demo clock can't busy-spin.
            await _maybe_drift_note()
            wait = max(1, int(MIN_REAL_REACH_GAP_S - (real_gap or 0.0)))
        # Interruptible sleep: a config change or a new chat wakes us early; and we
        # never sleep past the next due reminder, so kept promises land on time.
        upcoming = [r.due_at for r in STATE.reminders.items
                    if not r.done and not r.delivered]
        if upcoming:
            wait = min(wait, max(1, min(upcoming) - time.time()))
        assert STATE.wake is not None
        try:
            await asyncio.wait_for(STATE.wake.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass
        STATE.wake.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.generate, STATE.fast = compose.make_generator()
    STATE.embed = embedder.make_embedder()   # semantic recall when a key is set
    STATE.mood_signal = mood.build_local_mood_signal()   # local; None -> keyword
    print(f"[mood] classifier: {'model2vec' if STATE.mood_signal else 'keyword'}",
          flush=True)
    # Presence readers fail silently per call (the loop reads every few seconds and
    # must not flood), so say once, here, which of them this machine can actually
    # provide — otherwise a blind sensor just shows defaults and looks healthy.
    caps = sensors.capabilities()
    live = [n for n, st in caps.items() if st == "live"]
    print(f"[sensors] live: {', '.join(live) if live else 'none'}"
          f" ({len(live)}/{len(caps)})", flush=True)
    for name, status in caps.items():
        if status != "live":
            print(f"[sensors] {name} unavailable — {status}", flush=True)
    STATE.current_key = STATE.sessions.most_recent_key()  # resume last on start
    STATE.wake = asyncio.Event()
    # Delivery surfaces: web + native banner always; Telegram if a token is set.
    STATE.delivery = channels.build_default(STATE.listeners)
    print(f"[channels] delivering via: {', '.join(STATE.delivery.names())}",
          flush=True)
    # Passive-loop tools (optional). Connects only if backend/mcp.json exists.
    STATE.mcp = tools.MCPManager.from_config()
    try:
        await STATE.mcp.connect()
    except Exception as exc:  # noqa: BLE001 - never let MCP break startup
        print(f"[mcp] connect failed: {exc}", flush=True)
    task = asyncio.create_task(_proactive_loop())
    try:
        yield
    finally:
        task.cancel()
        if STATE.mcp is not None:
            try:
                await STATE.mcp.aclose()
            except Exception:  # noqa: BLE001
                pass


app = FastAPI(title="Rusty Companion — the Keeper", lifespan=lifespan)

# DNS-rebinding defense: only serve requests whose Host is localhost. A malicious
# website that rebinds its domain to 127.0.0.1 would send its own Host header, so
# it is refused. This is the main thing standing in for auth on a local, no-login
# app — do NOT expose this server publicly without real authentication.
app.add_middleware(TrustedHostMiddleware,
                   allowed_hosts=["localhost", "127.0.0.1"])


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

class ChatIn(BaseModel):
    message: str


@app.post("/chat")
async def chat(body: ChatIn):
    msg = body.message.strip()
    now = time.time()
    STATE.last_user_at = now
    STATE.user_msg_times.append(now)
    STATE.history.append({"role": "user", "content": msg, "ts": now})
    if STATE.current_key is None:                 # start a conversation on first msg
        STATE.current_key = STATE.sessions.new().key
    STATE.sessions.append(STATE.current_key, "user", msg, now)

    # Retrieve-then-rerank on the user-facing path: hybrid casts wide, the fast model
    # reranks to the best few. (Background paths use plain hybrid — no per-turn cost.)
    mem = memory.recall(STATE.store, msg, k=4, embed=STATE.embed,
                        rerank_generate=STATE.fast)
    # current time in the context so the Keeper can turn "tomorrow 9am" -> ISO.
    ctx = (sensors.read().to_context_line()
           + f"; current time {datetime.now().isoformat(timespec='minutes')}")
    # Register continuity: a clear emotional signal sets the register; a neutral
    # follow-up ("what should i do") INHERITS it rather than resetting to tidal,
    # so a stuck person is never told they're moving.
    # Mood, two layers: the high-precision keyword signal wins when a feeling word
    # is present; the local Model2Vec classifier (benchmark winner) fills the gap
    # for IMPLICIT mood the lexicon misses ("i don't know why i bother" -> frozen).
    # Known tradeoff: the embedding layer can over-commit on ambiguous requests
    # ("what should i do"); register continuity softens that downstream.
    signal = voice_eval.register_signal(msg)
    if signal is None and STATE.mood_signal is not None:
        signal = STATE.mood_signal(msg)
    if signal is not None:
        STATE.current_register = signal
    water = STATE.current_register or voice_eval.detect_state(msg)
    water = _resolve_turn(msg, mem, water)

    # Native action tools (reminders + goals + journal + delegate) are always
    # available; MCP tools join when configured. Reaching for a tool — or handing a
    # task to a specialist sub-agent — is the passive/agentic path.
    providers = [native_tools.NativeTools(
        STATE.reminders, goals=STATE.goals,
        planner_generate=STATE.fast or STATE.generate, journal=STATE.journal,
        mcp=STATE.mcp, delegate_generate=STATE.fast or STATE.generate,
        spawner=_spawn_background)]
    if STATE.mcp is not None and STATE.mcp.has_tools:
        providers.append(STATE.mcp)
    used_tools = bool(providers)
    tool_trace: list = []
    try:
        if used_tools:
            # Tool path: the Keeper may reach for tools, then answer. A
            # tool-grounded answer can be plainer (a real fact in voice), so we
            # score it for information only, never replacing it with a fallback.
            system = persona.build_system_prompt(
                "passive", water, memory=mem, context=ctx)
            system = system + "\n\n---\n\n" + TOOL_ADDENDUM
            reply = await compose.tool_reply(
                system, msg, providers=providers, model=TOOL_MODEL, max_rounds=6,
                trace=tool_trace)
            # Tool answers come back plain (a changelog, a file dump). Pass them back
            # through the Keeper's voice — preserving every fact — so a tool-grounded
            # reply still sounds like the Keeper, not a report.
            if reply and reply.strip():
                reply = await asyncio.to_thread(
                    compose.revoice, reply, water,
                    generate=STATE.fast or STATE.generate, memory=mem, context=ctx)
            report = voice_eval.evaluate(reply)   # deterministic, for logging
            score, fell_back = report.score, False
        else:
            result = await asyncio.to_thread(
                compose.compose, "passive", water,
                generate=STATE.generate, fast_model=STATE.fast,
                user_message=msg, memory=mem, context=ctx)
            reply, score, fell_back = result.text or "", result.score, result.fell_back
    except Exception as exc:  # noqa: BLE001 - the water must never 500 at them
        print(f"[chat] generation failed: {type(exc).__name__}: {exc}", flush=True)
        reply, score, fell_back = ERROR_LINE, None, True

    STATE.history.append({"role": "assistant", "content": reply, "ts": time.time()})
    if STATE.current_key is not None:
        STATE.sessions.append(STATE.current_key, "assistant", reply)
    # Note: the reply is returned in the HTTP response and rendered from there;
    # SSE (/events) carries ONLY unbidden proactive lines, so nothing double-renders.

    # Per-turn TRACE — the whole point is answering "why did it say that?" fast: what
    # memory was injected, the register, which tools fired, the reply. Ring-buffered.
    STATE.traces.appendleft({
        "ts": now,
        "cue": msg,
        "water_state": water,
        "mood_signal": signal,
        "memory_injected": mem,
        "path": "tool" if used_tools else "compose",
        "tools_called": tool_trace,
        "reply": reply,
        "score": round(score, 3) if isinstance(score, (int, float)) else None,
        "fell_back": fell_back,
    })

    # Distill this exchange into the drawers, off the response path — but not a
    # failed turn, which carries no real reply to learn from.
    if reply != ERROR_LINE:
        asyncio.create_task(_distill_async(msg, reply))
    if STATE.wake is not None:
        STATE.wake.set()   # re-tick: energy just reset, loop should back off

    return {"reply": reply, "water_state": water, "score": score,
            "fell_back": fell_back, "used_tools": used_tools}


async def _maybe_routine(pres: sensors.Presence) -> bool:
    """Check the house routines against the current presence; if one has earned a
    line, compose it in the Keeper's voice (the routine's intent as context) and
    deliver it. Returns True iff a routine spoke this tick."""
    try:
        routine = STATE.routines_engine.check(
            pres, time.time(), STATE.minutes_since_user())
    except Exception as exc:  # noqa: BLE001
        print(f"[routine] check error: {exc}", flush=True)
        return False
    if routine is None:
        return False
    water = proactive.derive_state(STATE.minutes_since_user())
    mem = memory.recall(STATE.store, "", k=3, embed=STATE.embed)
    result = await asyncio.to_thread(
        compose.compose, "proactive", water,
        generate=STATE.generate, fast_model=STATE.fast,
        memory=mem, context=routine.intent)
    if result.silent or not result.text:
        return False        # composed nothing on-voice -> let restlessness decide
    STATE.routines_engine.fire(routine, time.time())
    STATE.last_proactive_at = time.time()
    STATE.history.append({"role": "assistant", "content": result.text,
                          "ts": time.time()})
    if STATE.current_key is not None:
        STATE.sessions.append(STATE.current_key, "assistant", result.text)
    await _push("assistant", result.text, "proactive")
    print(f"[routine] {routine.key} spoke", flush=True)
    return True


def _goal_check_interval() -> float:
    """How long to wait before working a goal again, on the DEMO clock.

    tasks.DEFAULT_CHECK_INTERVAL_S is 6 real hours. The `speed` knob compresses the
    battery and the drift clock but not this one, so before this existed a goal went
    quiet for 6 real hours after its first step — which meant the documented
    "crank speed to see nudges/autonomous execution" trigger could never fire, and
    a person-step was never nudged in a demo. Compress it the same way drift is.
    """
    return tasks.DEFAULT_CHECK_INTERVAL_S / max(STATE.config.speed, 1.0)


async def _maybe_advance_goal() -> bool:
    """The agent at work: if a goal is due, take its next step. A [keeper] step it
    EXECUTES itself with its tools (ReAct); a [person] step it NUDGES, then waits for
    them to report it done. Returns True iff it reached out this tick."""
    goal = STATE.goals.due()
    if goal is None:
        return False
    step = goal.next_step()
    if step is None:
        return False
    water = proactive.derive_state(STATE.minutes_since_user())
    if step.actor == "keeper":
        return await _execute_goal_step(goal, step, water)
    return await _nudge_goal_step(goal, step, water)


async def _deliver_proactive(text: str) -> None:
    """Record + fan out one unbidden line (history, session, all channels)."""
    STATE.last_proactive_at = time.time()
    STATE.history.append({"role": "assistant", "content": text, "ts": time.time()})
    if STATE.current_key is not None:
        STATE.sessions.append(STATE.current_key, "assistant", text)
    await _push("assistant", text, "proactive")


async def _nudge_goal_step(goal, step, water) -> bool:
    """A [person] step: help with or invite it, but DON'T complete it — a person step
    is only done when they report it (advance_goal). Holds the thread, returns it."""
    context = (f"You are helping them move toward a goal of theirs: \"{goal.title}\". "
               f"Gently help with, or invite, just this next step — do not list the "
               f"whole plan: {step.text}")
    result = await asyncio.to_thread(
        compose.compose, "proactive", water,
        generate=STATE.generate, fast_model=STATE.fast,
        memory=memory.recall(STATE.store, goal.title, k=3, embed=STATE.embed),
        context=context)
    if result.silent or not result.text:
        STATE.goals.touch(goal, _goal_check_interval())
        return False
    STATE.goals.touch(goal, _goal_check_interval())   # nudge, don't complete
    await _deliver_proactive(result.text)
    done, total = goal.progress()
    print(f"[goal] {goal.id} nudged {done}/{total}: {step.text[:50]}", flush=True)
    return True


async def _execute_goal_step(goal, step, water) -> bool:
    """A [keeper] step: hand it to a specialist SUB-AGENT (multi-agent), which runs its
    own tool loop and returns a result; the Keeper then speaks it. On empty/failed work
    the step is handed back to the person (re-labelled) so it gets nudged next time."""
    sub_native = native_tools.NativeTools(
        STATE.reminders, goals=STATE.goals,
        planner_generate=STATE.fast or STATE.generate, journal=STATE.journal,
        allow_delegate=False)               # a sub-agent must not spawn sub-agents
    try:
        res = await subagents.orchestrate(
            f"For their goal \"{goal.title}\", do this: {step.text}",
            STATE.fast or STATE.generate, mcp=STATE.mcp, native=sub_native)
    except Exception as exc:  # noqa: BLE001
        print(f"[goal] execute error: {exc}", flush=True)
        STATE.goals.touch(goal, _goal_check_interval())
        return False
    result = (res.result or "").strip()
    if not result:
        step.actor = "person"            # hand it back — nudge them next time
        STATE.goals.touch(goal, _goal_check_interval())
        print(f"[goal] {goal.id} sub-agent found nothing, handed back: "
              f"{step.text[:40]}", flush=True)
        return False
    mem = memory.recall(STATE.store, goal.title, k=3, embed=STATE.embed)
    voiced = await asyncio.to_thread(
        compose.revoice, result, water, generate=STATE.fast or STATE.generate,
        memory=mem)
    STATE.goals.advance(goal, note=f"[{res.who()}] {result[:130]}",   # who did it
                        interval_s=_goal_check_interval())
    await _deliver_proactive(voiced)
    done, total = goal.progress()
    print(f"[goal] {goal.id} EXECUTED via {res.who()} {done}/{total}: "
          f"{step.text[:45]}", flush=True)
    return True


async def _answer_as_keeper(text: str) -> str:
    """Answer an incoming message (e.g. from a peer agent over A2A) as the Keeper, with
    its tools — but a SAFE subset: no delegate/spawn, so a peer can't spawn work."""
    water = STATE.current_register or voice_eval.detect_state(text)
    mem = memory.recall(STATE.store, text, k=4, embed=STATE.embed)
    system = persona.build_system_prompt("passive", water, memory=mem)
    system = system + "\n\n---\n\n" + TOOL_ADDENDUM
    providers = [native_tools.NativeTools(
        STATE.reminders, goals=STATE.goals,
        planner_generate=STATE.fast or STATE.generate, journal=STATE.journal,
        allow_delegate=False)]
    if STATE.mcp is not None and STATE.mcp.has_tools:
        providers.append(STATE.mcp)
    try:
        reply = await compose.tool_reply(
            system, text, providers=providers, model=TOOL_MODEL, max_rounds=6)
    except Exception as exc:  # noqa: BLE001
        return f"(the water is unsettled: {exc})"
    return await asyncio.to_thread(
        compose.revoice, reply, water, generate=STATE.fast or STATE.generate, memory=mem)


async def _spawn_background(description: str) -> str:
    """Kick off a long delegation in the background and return immediately. The result
    is brought back later by _run_background, through the proactive channels."""
    bg = STATE.background.add(description)
    asyncio.create_task(_run_background(bg.id, description))
    print(f"[bg] {bg.id} started: {description[:60]}", flush=True)
    return (f"started working on it in the background (id {bg.id}) — tell them you're "
            f"on it and will come back with what you find")


async def _run_background(bg_id: str, description: str) -> None:
    """Run a background delegation to completion, then deliver the result later — as
    an unbidden line + native banner, like a kept promise (bypasses the restlessness
    floor, since the person asked for this)."""
    sub_native = native_tools.NativeTools(
        STATE.reminders, goals=STATE.goals,
        planner_generate=STATE.fast or STATE.generate, journal=STATE.journal,
        allow_delegate=False)
    try:
        orch = await subagents.orchestrate(
            description, STATE.fast or STATE.generate, mcp=STATE.mcp, native=sub_native)
        result = (orch.result or "").strip()
    except Exception as exc:  # noqa: BLE001
        STATE.background.finish(bg_id, f"(failed: {exc})", ok=False)
        print(f"[bg] {bg_id} failed: {exc}", flush=True)
        return
    STATE.background.finish(bg_id, result or "(found nothing)", ok=bool(result))
    if not result:
        return
    water = proactive.derive_state(STATE.minutes_since_user())
    voiced = await asyncio.to_thread(
        compose.revoice,
        f"You asked me to look into this: {description}\n\nHere is what I found: {result}",
        water, generate=STATE.fast or STATE.generate)
    await _deliver_proactive(voiced)
    print(f"[bg] {bg_id} delivered", flush=True)


async def _maybe_drift_note() -> None:
    """Idle background reflection (never sent to the person). Real-time capped so a
    sped-up demo clock can't flood the store with near-duplicate insights."""
    if within_reach_floor(STATE.last_drift_at, time.time(), MIN_REAL_DRIFT_GAP_S):
        return
    try:
        r = await asyncio.to_thread(
            drift.maybe_drift, STATE.store, STATE.fast or STATE.generate,
            STATE.reflections, last_drift_at=STATE.last_drift_at,
            config=STATE.drift_config, embed=STATE.embed)
        if r is not None:
            STATE.last_drift_at = time.time()
            print(f"[drift] reflected: {r.text[:80]}...", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[drift] error: {exc}", flush=True)


def _resolve_turn(msg: str, mem: str, water: str) -> str:
    """Make "turn" reachable ONLY through the store, in both directions.

    persona.py calls it the rarest register — the ice going out — and says to
    spend it almost never. Chosen from the message alone it fired on "hello",
    because a single sentence cannot contain a reversal: "the ice is going out" is
    a comparison between two points in time, and only the store holds both. So:

      DEMOTE  the classifier saying "turn" is not evidence of anything. Without a
              recorded reversal behind it, fall back to tidal. This is what stops
              a greeting from being met as a season changing.
      PROMOTE a person moving (tidal) who is talking about something that really
              did reverse gets the register the persona describes.

    Never touches frozen: someone in the cold is not told their season has turned
    because the store remembers better days.
    """
    if water == "frozen":
        return water
    earned = _warming_behind(msg, mem)
    if earned is not None:
        print(f"[register] turn earned by: {earned.text[:60]!r}", flush=True)
        return "turn"
    if water == "turn":
        print("[register] turn not earned by the store — falling back to tidal",
              flush=True)
        return "tidal"
    return water


def _warming_behind(msg: str, mem: str):
    """The recorded reversal this message is actually about, or None.

    Two conditions, and the second is the one that took a live failure to learn.
    The warming must have survived recall's relevance gate into `mem`, AND the
    person's own words must overlap it. Recall alone is far too loose: in a small
    store it surfaces almost anything for a warm cue, so "i had a good day today"
    was promoted to turn on the strength of a painting fact it never mentioned.
    """
    if STATE.store is None:
        return None
    try:
        warmings = STATE.store.recent_warmings()
    except Exception as exc:  # noqa: BLE001 - a register is never worth a 500
        print(f"[register] warming lookup failed: {type(exc).__name__}: {exc}",
              flush=True)
        return None
    cue_tokens = memory._tokens(msg)
    if not cue_tokens:
        return None
    for fact in warmings:
        if not fact.text or fact.text not in mem:
            continue                      # recall did not judge it relevant
        if cue_tokens & memory._tokens(fact.text):
            return fact                   # and they are talking about it
    return None


async def _distill_async(user_msg: str, reply: str) -> None:
    convo = [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": reply}]
    await asyncio.to_thread(
        memory.distill, convo, STATE.fast or STATE.generate, STATE.store, STATE.embed)
    # Under memory pressure, compress the low-value tail (MemGPT). Cheap no-op when
    # the store is under budget; runs off the response path.
    try:
        made = await asyncio.to_thread(
            memory.consolidate, STATE.store, STATE.fast or STATE.generate,
            embed=STATE.embed)
        if made:
            print(f"[memory] consolidated {len(made)} cluster(s)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] consolidate error: {exc}", flush=True)


@app.get("/events")
async def events():
    q: asyncio.Queue = asyncio.Queue()
    STATE.listeners.add(q)

    async def stream():
        try:
            # greet the stream so clients know it's open
            yield ": connected\n\n"
            while True:
                payload = await q.get()
                yield f"data: {payload}\n\n"
        finally:
            STATE.listeners.discard(q)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/state")
async def state():
    # Report the virtual clock the loop actually runs on, so a sped-up demo
    # visibly shows the battery draining.
    ps = STATE.proactive_state()
    ms = ps.minutes_since_user
    en = energy.compute_energy(ms)
    score = energy.base_score(en, STATE.recent_msg_count())
    now = time.time()

    pres = await asyncio.to_thread(sensors.read)
    pending = STATE.reminders.pending()
    next_rem = pending[0] if pending else None

    return {
        "minutes_since_user_virtual": ms,
        "minutes_since_user_real": STATE.minutes_since_user(),
        "minutes_since_proactive": ps.minutes_since_proactive,
        "recent_msg_count": STATE.recent_msg_count(),
        "energy": round(en, 3),
        "base_score": round(score, 3),
        "speak_probability": round(energy.speak_probability(score), 3),
        "facts_kept": sum(1 for f in STATE.store.facts if f.active),
        "insights_kept": sum(1 for f in STATE.store.facts
                             if f.kind == "insight" and f.active),
        "changes_tracked": len(STATE.store.changes()),
        "reminders_held": len(pending),
        "reminders_recurring": sum(1 for r in pending if r.repeat),
        "goals": [
            {"title": g.title, "id": g.id,
             "done": g.progress()[0], "total": g.progress()[1],
             "next_step": (g.next_step().text if g.next_step() else None),
             "next_actor": (g.next_step().actor if g.next_step() else None)}
            for g in STATE.goals.active()],
        "next_reminder": (
            {"text": next_rem.text, "due_at": next_rem.due_at,
             "repeat": next_rem.repeat} if next_rem else None),
        "reflections": len(STATE.reflections.items),
        "latest_reflection": (STATE.reflections.latest().text
                              if STATE.reflections.latest() else None),
        # the house: presence + the routines watching it
        "presence": {
            "idle_seconds": pres.idle_seconds,
            "screen_locked": pres.screen_locked,
            "frontmost_app": pres.frontmost_app,
            "summary": pres.to_context_line(),
        },
        "house": STATE.routines_engine.status(now),
        "working_on": [{"id": t.id, "description": t.description}
                       for t in STATE.background.running()],
        "embedder": "openai" if STATE.embed is not None else "keyword",
        "speed": STATE.config.speed,
        "backend": "openai" if STATE.fast is not None else "stub",
        "listeners": len(STATE.listeners),
        "channels": STATE.delivery.names() if STATE.delivery else ["web"],
    }


class ConfigIn(BaseModel):
    speed: Optional[float] = None
    cooldown_min: Optional[float] = None


@app.post("/config")
async def set_config(body: ConfigIn):
    if body.speed is not None:
        STATE.config.speed = max(1.0, body.speed)
        STATE.drift_config.speed = STATE.config.speed   # compress drift clock too
    if body.cooldown_min is not None:
        STATE.config.cooldown_min = max(0.0, body.cooldown_min)
    if STATE.wake is not None:
        STATE.wake.set()   # re-tick now with the new settings
    return {"speed": STATE.config.speed, "cooldown_min": STATE.config.cooldown_min}


@app.get("/sessions")
async def list_sessions():
    return {
        "sessions": [{"key": s.key, "title": s.title or "…",
                      "count": len(s.messages), "updated_at": s.updated}
                     for s in STATE.sessions.all()],
        "active": STATE.current_key,
    }


@app.get("/sessions/{key}/messages")
async def session_messages(key: str):
    s = STATE.sessions.get(key)
    if s is None:
        return {"messages": []}
    return {"messages": [{"role": m.role, "content": m.content, "ts": m.ts}
                         for m in s.messages]}


class SelectIn(BaseModel):
    session_id: str


@app.post("/select")
async def select_session(body: SelectIn):
    if STATE.sessions.get(body.session_id) is not None:
        STATE.current_key = body.session_id
    return {"ok": True, "active": STATE.current_key}


@app.post("/new")
async def new_conversation():
    STATE.current_key = None      # next message starts a fresh conversation
    return {"ok": True}


# --------------------------------------------------------------------------- #
# A2A — the Keeper as an agent other agents can discover and consult.
# --------------------------------------------------------------------------- #

@app.get(a2a.WELL_KNOWN)
async def agent_card(req: Request):
    """The Keeper's A2A Agent Card — how a peer agent discovers what it can do."""
    return a2a.build_agent_card(str(req.base_url))


@app.post("/a2a")
async def a2a_endpoint(req: Request):
    """Handle an A2A message/send: a peer sends a message, the Keeper answers."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        return a2a.rpc_error(None, -32700, "parse error")
    req_id = body.get("id")
    if body.get("method") != "message/send":
        return a2a.rpc_error(req_id, -32601,
                             f"unsupported method: {body.get('method')}")
    text = a2a.text_of((body.get("params") or {}).get("message") or {})
    if not text:
        return a2a.rpc_error(req_id, -32602, "empty message")
    reply = await _answer_as_keeper(text)
    return a2a.rpc_result(req_id, a2a.make_message(reply, role="agent"))


@app.get("/trace")
async def trace():
    """The last turns' decisions — memory injected, register, tools fired, reply."""
    return list(STATE.traces)


@app.get("/trace-view")
async def trace_view():
    return FileResponse(STATIC_DIR / "trace.html")


@app.get("/dashboard")
async def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/info")
async def info():
    return {"name": "Rusty Companion — the Keeper",
            "endpoints": ["/chat", "/events", "/state", "/config"],
            "facts_kept": len(STATE.store.facts)}
