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
from typing import Optional

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

STATIC_DIR = Path(__file__).resolve().parent / "static"

import compose
import embedder
import energy
import memory
import persona
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
tool returned."""

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
    wake: Optional[asyncio.Event] = None   # set to interrupt the loop's sleep
    mcp: Optional[tools.MCPManager] = None  # passive-loop tools; None until connected

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
    """Fan a message out to every connected SSE listener."""
    payload = json.dumps({"role": role, "content": content, "kind": kind,
                          "ts": time.time()})
    for q in list(STATE.listeners):
        await q.put(payload)


# --------------------------------------------------------------------------- #
# The proactive background loop.
# --------------------------------------------------------------------------- #

async def _proactive_loop() -> None:
    # Treat startup as fresh contact so it doesn't blast before you've spoken.
    if STATE.last_user_at is None:
        STATE.last_user_at = time.time()
    while True:
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
            await _push("assistant", decision.text, "proactive")
        # Interruptible sleep: a config change (or a new chat) wakes us early so
        # the next tick honors the new speed/state instead of finishing an old,
        # possibly hour-long, interval.
        assert STATE.wake is not None
        try:
            await asyncio.wait_for(STATE.wake.wait(), timeout=max(1, decision.wait_next_s))
        except asyncio.TimeoutError:
            pass
        STATE.wake.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.generate, STATE.fast = compose.make_generator()
    STATE.embed = embedder.make_embedder()   # semantic recall when a key is set
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

    mem = memory.recall(STATE.store, msg, k=4, embed=STATE.embed)
    ctx = sensors.read().to_context_line()
    # Register continuity: a clear emotional signal sets the register; a neutral
    # follow-up ("what should i do") INHERITS it rather than resetting to tidal,
    # so a stuck person is never told they're moving.
    signal = voice_eval.register_signal(msg)
    if signal is not None:
        STATE.current_register = signal
    water = STATE.current_register or voice_eval.detect_state(msg)

    used_tools = STATE.mcp is not None and STATE.mcp.has_tools
    try:
        if used_tools:
            # Tool path: the Keeper may reach for MCP tools, then answer. A
            # tool-grounded answer can be plainer (a real fact in voice), so we
            # score it for information only, never replacing it with a fallback.
            system = persona.build_system_prompt(
                "passive", water, memory=mem, context=ctx)
            system = system + "\n\n---\n\n" + TOOL_ADDENDUM
            reply = await compose.tool_reply(
                system, msg, mcp=STATE.mcp, model=TOOL_MODEL)
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
    if body.cooldown_min is not None:
        STATE.config.cooldown_min = max(0.0, body.cooldown_min)
    if STATE.wake is not None:
        STATE.wake.set()   # re-tick now with the new settings
    return {"speed": STATE.config.speed, "cooldown_min": STATE.config.cooldown_min}


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/info")
async def info():
    return {"name": "Rusty Companion — the Keeper",
            "endpoints": ["/chat", "/events", "/state", "/config"],
            "facts_kept": len(STATE.store.facts)}
