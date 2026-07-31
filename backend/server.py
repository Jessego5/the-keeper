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

STATIC_DIR = Path(__file__).resolve().parent / "static"

import compose
import energy
import memory
import proactive
import sensors
import voice_eval

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
    config: proactive.ProactiveConfig = field(default_factory=proactive.ProactiveConfig)
    listeners: set[asyncio.Queue] = field(default_factory=set)
    generate: compose.Generator = compose.stub_generator
    fast: Optional[compose.Generator] = None
    wake: Optional[asyncio.Event] = None   # set to interrupt the loop's sleep

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
    STATE.wake = asyncio.Event()
    task = asyncio.create_task(_proactive_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Rusty Companion — the Keeper", lifespan=lifespan)


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

    mem = memory.recall(STATE.store, msg, k=4)
    ctx = sensors.read().to_context_line()
    water = voice_eval.detect_state(msg)

    result = await asyncio.to_thread(
        compose.compose, "passive", water,
        generate=STATE.generate, fast_model=STATE.fast,
        user_message=msg, memory=mem, context=ctx)

    reply = result.text or ""
    STATE.history.append({"role": "assistant", "content": reply, "ts": time.time()})
    # Note: the reply is returned in the HTTP response and rendered from there;
    # SSE (/events) carries ONLY unbidden proactive lines, so nothing double-renders.

    # Distill this exchange into the drawers, off the response path.
    asyncio.create_task(_distill_async(msg, reply))
    if STATE.wake is not None:
        STATE.wake.set()   # re-tick: energy just reset, loop should back off

    return {"reply": reply, "water_state": water, "score": result.score,
            "fell_back": result.fell_back}


async def _distill_async(user_msg: str, reply: str) -> None:
    convo = [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": reply}]
    await asyncio.to_thread(
        memory.distill, convo, STATE.fast or STATE.generate, STATE.store)


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
