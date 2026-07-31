"""proactive.py — the loop that lets the Keeper reach out on its own.

Each tick asks, in order:

    1. Is the person even reachable?      (sensors: skip if the screen is locked)
    2. Am I restless enough to try?       (energy -> base_score -> a random roll)
    3. Do I actually have something?       (compose proactive; it may still fall silent)

Only if all three pass does a line go out. This is the proper fix for the "spoke
every time" problem from the first voice test: silence is the default at TWO gates
— the roll usually says no, and even when it says yes, compose can decline. A line
is earned, not scheduled.

Two ways to run it:
    tick()      one decision, given the current state. Pure, testable, no waiting.
    simulate()  fast-forward virtual time to WATCH it fire in seconds (the demo).

The `speed` knob compresses real tick intervals for a live demo (REFERENCE.md §4:
proactivity must be demoable without waiting minutes/hours).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import compose
import energy
import memory
import sensors

Generator = compose.Generator


@dataclass
class ProactiveState:
    """What the loop knows about the person right now."""

    minutes_since_user: Optional[float] = None   # None = never talked
    recent_msg_count: int = 0                     # messages in the recent window
    minutes_since_proactive: Optional[float] = None  # since the Keeper last spoke


@dataclass
class ProactiveConfig:
    p_min: float = 0.05
    p_max: float = 0.45
    tick_fast: int = 2400        # seconds between checks when restless
    tick_slow: int = 4800        # ... when satisfied
    jitter: float = 0.3
    speed: float = 1.0           # >1 compresses tick waits for demos
    respect_lock: bool = True    # never speak into a locked screen
    cooldown_min: float = 600.0  # refractory: hold silence 10h after speaking


@dataclass
class TickDecision:
    spoke: bool
    reason: str                  # why it spoke or stayed quiet
    energy: float
    base_score: float
    wait_next_s: int             # seconds until the next tick (speed-adjusted)
    text: Optional[str] = None   # the line, if it spoke
    water_state: str = "tidal"


def derive_state(minutes_since_user: Optional[float]) -> str:
    """Pick a register for an unbidden message from how long they've been gone.

    A long silence reads as the cold; recent contact reads as the moving water.
    Coarse on purpose — proactive has no fresh user signal to detect from.
    """
    if minutes_since_user is None or minutes_since_user > 2 * 24 * 60:
        return "frozen"
    return "tidal"


def tick(
    state: ProactiveState,
    *,
    generate: Generator,
    fast_model: Optional[Generator] = None,
    store: Optional[memory.MemoryStore] = None,
    config: ProactiveConfig = ProactiveConfig(),
    presence: Optional[sensors.Presence] = None,
    rng: Optional[random.Random] = None,
) -> TickDecision:
    """Run one proactive decision. Never blocks; never raises on a normal path."""
    rng = rng or random
    en = energy.compute_energy(state.minutes_since_user)
    score = energy.base_score(en, state.recent_msg_count)
    water = derive_state(state.minutes_since_user)
    wait = int(energy.next_tick_seconds(
        score, tick_fast=config.tick_fast, tick_slow=config.tick_slow,
        jitter=config.jitter, rng=rng) / max(config.speed, 1e-9))

    def quiet(reason: str) -> TickDecision:
        return TickDecision(False, reason, en, score, wait, None, water)

    # Gate 0 — cooldown. If it just reached out and got no reply, hold silence;
    # otherwise low energy would make it pester on every tick.
    if (state.minutes_since_proactive is not None
            and state.minutes_since_proactive < config.cooldown_min):
        return quiet("in cooldown since last outreach")

    # Gate 1 — reachability. Don't speak to a locked screen.
    if config.respect_lock:
        pres = presence if presence is not None else sensors.read()
        if pres.screen_locked:
            return quiet("screen locked")

    # Gate 2 — restlessness. The roll usually says no.
    if not energy.roll_speak(score, rng=rng, p_min=config.p_min, p_max=config.p_max):
        return quiet("did not roll to speak")

    # Gate 3 — content. Recall what's present, then let compose try (or decline).
    mem = memory.recall(store, "", k=3) if store is not None else ""
    result = compose.compose(
        "proactive", water, generate=generate, fast_model=fast_model, memory=mem)
    if result.silent:
        return quiet("rolled to speak, but chose silence")

    return TickDecision(True, "spoke", en, score, wait, result.text, water)


# --------------------------------------------------------------------------- #
# Demo — fast-forward a day of silence and watch when the Keeper speaks.
# --------------------------------------------------------------------------- #

@dataclass
class SimResult:
    ticks: int = 0
    spoke: int = 0
    lines: list[tuple[float, str]] = field(default_factory=list)  # (hour, text)


def simulate(
    hours: float = 48.0,
    *,
    seed: int = 7,
    msg_count: int = 4,
    store: Optional[memory.MemoryStore] = None,
    generate: Optional[Generator] = None,
    fast_model: Optional[Generator] = None,
    verbose: bool = True,
) -> SimResult:
    """Walk virtual time forward across `hours` of silence, ticking as the real
    scheduler would, and report every time the Keeper reaches out.

    Uses the offline stub by default so it runs instantly and free; pass a real
    generator to see live lines. Sensors are stubbed to 'awake' so the demo isn't
    gated by your actual screen state.
    """
    rng = random.Random(seed)
    if generate is None:
        generate, fast_model = compose.stub_generator, None

    awake = sensors.Presence(idle_seconds=60.0, screen_locked=False,
                             frontmost_app="demo")
    cfg = ProactiveConfig()
    res = SimResult()

    minutes = 0.0
    horizon = hours * 60.0
    last_spoke: Optional[float] = None
    while minutes < horizon:
        since_proactive = None if last_spoke is None else minutes - last_spoke
        state = ProactiveState(minutes_since_user=minutes, recent_msg_count=msg_count,
                               minutes_since_proactive=since_proactive)
        d = tick(state, generate=generate, fast_model=fast_model, store=store,
                 config=cfg, presence=awake, rng=rng)
        res.ticks += 1
        if d.spoke:
            res.spoke += 1
            last_spoke = minutes
            res.lines.append((minutes / 60.0, d.text or ""))
            if verbose:
                print(f"  t+{minutes/60:5.1f}h  E={d.energy:.2f} "
                      f"score={d.base_score:.2f}  ->  {d.text}")
        # advance virtual time by the (uncompressed) interval this tick chose
        step = energy.next_tick_seconds(
            d.base_score, tick_fast=cfg.tick_fast, tick_slow=cfg.tick_slow,
            jitter=cfg.jitter, rng=rng) / 60.0
        minutes += step

    if verbose:
        print(f"\n  {res.spoke} messages across {res.ticks} ticks over {hours:.0f}h "
              f"of silence (seed={seed})")
    return res


if __name__ == "__main__":
    print("Simulating 48h of silence (offline stub, so lines repeat):\n")
    simulate(48.0)
