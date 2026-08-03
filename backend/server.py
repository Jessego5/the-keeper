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
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

STATIC_DIR = Path(__file__).resolve().parent / "static"

import compose
import drift
import embedder
import energy
import memory
import mood
import native_tools
import notifier
import persona
import reminders
import sessions
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
TOOL_ADDENDUM = """You have been given tools to look at what is theirs — files they
keep, and the like. When they ask you to look at something, USE the tools to look,
then answer from what you actually find. Do not decline, and do not guess at the
contents. If a tool lists the folders or files you may read, follow it to the one
they mean, then read it. Reading what they have pointed you to is an act of keeping,
not a window on the world — the sealing rule bars inventing events unbidden, not
reading what they asked you to read. Answer in your own voice, but true to what the
tool returned.

You can also HOLD things for them. When they ask you to remember to do something at
a time ("remind me to call the dentist tomorrow"), use remind_me — convert their
phrasing into an ISO datetime using the current time given in your context. Use
list_reminders when they ask what you're holding, and complete_reminder when
something is done. You will return a due reminder to them yourself when its time
comes; that is keeping, not intruding.

For a request that takes more than one step, work it in steps: call a tool, read
what it returns, then call the next — e.g. list_reminders to see what you hold,
then complete_reminder on the right one. Take the steps you need, then answer once
in your voice."""

RECENT_WINDOW_MIN = 240.0   # "recent" messages = last 4h, for context richness


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
    sessions: sessions.SessionStore = field(default_factory=sessions.SessionStore)
    current_key: Optional[str] = None   # active conversation (sidebar)

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
    """Fan a message out to every connected SSE listener (on-page rendering) and,
    for unbidden lines, fire a native OS notification — which reaches you even
    with the browser closed and carries the Keeper's own image."""
    payload = json.dumps({"role": role, "content": content, "kind": kind,
                          "ts": time.time()})
    for q in list(STATE.listeners):
        await q.put(payload)
    if kind == "proactive":
        await asyncio.to_thread(notifier.notify, "the keeper", content)


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

        try:
            decision = await asyncio.to_thread(
                proactive.tick,
                STATE.proactive_state(),
                generate=STATE.generate, fast_model=STATE.fast,
                store=STATE.store, config=STATE.config,
            )
        except Exception as exc:  # noqa: BLE001 - never let the loop die silently
            print(f"[proactive] tick error: {type(exc).__name__}: {exc}", flush=True)
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
                STATE.sessions.append(STATE.current_key, "assistant", decision.text)
            await _push("assistant", decision.text, "proactive")
        else:
            # Idle: nothing to say -> occasionally drift (reflect on memory).
            # Runs off the response path, never sent to the person.
            try:
                r = await asyncio.to_thread(
                    drift.maybe_drift, STATE.store,
                    STATE.fast or STATE.generate, STATE.reflections,
                    last_drift_at=STATE.last_drift_at, config=STATE.drift_config)
                if r is not None:
                    STATE.last_drift_at = time.time()
                    print(f"[drift] reflected: {r.text[:80]}...", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[drift] error: {exc}", flush=True)
        # Interruptible sleep: a config change or a new chat wakes us early; and we
        # never sleep past the next due reminder, so kept promises land on time.
        wait = max(1, decision.wait_next_s)
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
    STATE.current_key = STATE.sessions.most_recent_key()  # resume last on start
    STATE.wake = asyncio.Event()
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

    mem = memory.recall(STATE.store, msg, k=4, embed=STATE.embed)
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

    # Native action tools (reminders) are always available; MCP file tools join
    # when configured. Reaching for a tool is the passive/agentic path.
    providers = [native_tools.NativeTools(STATE.reminders)]
    if STATE.mcp is not None and STATE.mcp.has_tools:
        providers.append(STATE.mcp)
    used_tools = bool(providers)
    try:
        if used_tools:
            # Tool path: the Keeper may reach for tools, then answer. A
            # tool-grounded answer can be plainer (a real fact in voice), so we
            # score it for information only, never replacing it with a fallback.
            system = persona.build_system_prompt(
                "passive", water, memory=mem, context=ctx)
            system = system + "\n\n---\n\n" + TOOL_ADDENDUM
            reply = await compose.tool_reply(
                system, msg, providers=providers, model=TOOL_MODEL, max_rounds=6)
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

    # Distill this exchange into the drawers, off the response path — but not a
    # failed turn, which carries no real reply to learn from.
    if reply != ERROR_LINE:
        asyncio.create_task(_distill_async(msg, reply))
    if STATE.wake is not None:
        STATE.wake.set()   # re-tick: energy just reset, loop should back off

    return {"reply": reply, "water_state": water, "score": score,
            "fell_back": fell_back, "used_tools": used_tools}


async def _distill_async(user_msg: str, reply: str) -> None:
    convo = [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": reply}]
    await asyncio.to_thread(
        memory.distill, convo, STATE.fast or STATE.generate, STATE.store, STATE.embed)


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
    return {
        "minutes_since_user_virtual": ms,
        "minutes_since_user_real": STATE.minutes_since_user(),
        "minutes_since_proactive": ps.minutes_since_proactive,
        "recent_msg_count": STATE.recent_msg_count(),
        "energy": round(en, 3),
        "base_score": round(score, 3),
        "speak_probability": round(energy.speak_probability(score), 3),
        "facts_kept": len(STATE.store.facts),
        "reminders_held": len(STATE.reminders.pending()),
        "reflections": len(STATE.reflections.items),
        "latest_reflection": (STATE.reflections.latest().text
                              if STATE.reflections.latest() else None),
        "embedder": "openai" if STATE.embed is not None else "keyword",
        "speed": STATE.config.speed,
        "backend": "openai" if STATE.fast is not None else "stub",
        "listeners": len(STATE.listeners),
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
