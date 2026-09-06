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
import relevance
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
    use_presence: bool = True    # let idle time modulate restlessness


@dataclass
class TickDecision:
    spoke: bool
    reason: str                  # why it spoke or stayed quiet
    energy: float
    base_score: float
    wait_next_s: int             # seconds until the next tick (speed-adjusted)
    text: Optional[str] = None   # the line, if it spoke
    water_state: str = "tidal"


def presence_factor(idle_seconds: Optional[float]) -> float:
    """How the person's idle time nudges the urge to speak.

    A companion's reach-out lands best when they are HERE but quiet — so catch
    them (a small boost). If they have been idle a long while they are away, and
    a line just waits in an empty room, so ease off. This is the "are you even
    here right now?" signal that message-timing alone can't provide.

        present, at the keyboard (< 2 min idle)   -> 1.15  (catch them)
        stepped away briefly (2-30 min)           -> 1.00  (neutral)
        gone (> 30 min)                           -> 0.60  (reaching into an empty room)
    """
    if idle_seconds is None:
        return 1.0
    if idle_seconds < 120:
        return 1.15
    if idle_seconds < 1800:
        return 1.0
    return 0.60


def derive_state(minutes_since_user: Optional[float]) -> str:
    """Pick a register for an unbidden message from how long they've been gone.

    A long silence reads as the cold; recent contact reads as the moving water.
    Coarse on purpose — proactive has no fresh user signal to detect from.
    """
    if minutes_since_user is None or minutes_since_user > 2 * 24 * 60:
        return "frozen"
    return "tidal"


# How much a genuinely relevant item may weight the coin. p_max still binds, so a
# busy feed cannot push the Keeper past its own ceiling — this shortens the wait
# for news that matters, it does not remove the gate.
SOURCE_BOOST = 3.0

# Token overlap above which a new line counts as something already said. Measured
# on real repeats: "This is the part where the water holds still." against "The
# water holds still." scores 0.60, while two genuinely different lines in the same
# register score 0.29 and across registers 0.14. 0.5 sits in that gap.
REPEAT_SIMILARITY = 0.5
# How many recent lines to weigh against. Long enough to catch a loop, short
# enough that an old line does not silence a fair echo of itself weeks later.
REPEAT_WINDOW = 8


def _too_similar(text: str, recent: list) -> Optional[str]:
    """The recent line `text` merely repeats, or None.

    Proactive lines repeat because the composer is handed identical input every
    tick — same prompt, same register, and recall("") returns the same facts — so a
    near-deterministic model returns the same sentence. Measured across real
    sessions, 35% of model-written lines were duplicates; one appeared twelve
    times.
    """
    toks = memory._tokens(text)
    if not toks:
        return None
    for prev in recent[-REPEAT_WINDOW:]:
        if memory._jaccard(toks, memory._tokens(prev)) >= REPEAT_SIMILARITY:
            return prev
    return None


def tick(
    state: ProactiveState,
    *,
    generate: Generator,
    fast_model: Optional[Generator] = None,
    store: Optional[memory.MemoryStore] = None,
    config: Optional[ProactiveConfig] = None,
    presence: Optional[sensors.Presence] = None,
    rng: Optional[random.Random] = None,
    pending: Optional[object] = None,
    recent: Optional[list] = None,
) -> TickDecision:
    """Run one proactive decision. Never blocks; never raises on a normal path."""
    # See maybe_drift: a dataclass default in the signature is built once and shared.
    config = config or ProactiveConfig()
    rng = rng or random
    en = energy.compute_energy(state.minutes_since_user)
    score = energy.base_score(en, state.recent_msg_count)
    water = derive_state(state.minutes_since_user)
    wait = int(energy.next_tick_seconds(
        score, tick_fast=config.tick_fast, tick_slow=config.tick_slow,
        jitter=config.jitter, rng=rng) / max(config.speed, 1e-9))

    def quiet(reason: str) -> TickDecision:
        return TickDecision(False, reason, en, score, wait, None, water)

    # One read of the machine, shared by the lock gate and the idle factor.
    pres = presence if presence is not None else sensors.read()

    # Gate 0 — cooldown. If it just reached out and got no reply, hold silence;
    # otherwise low energy would make it pester on every tick.
    if (state.minutes_since_proactive is not None
            and state.minutes_since_proactive < config.cooldown_min):
        return quiet("in cooldown since last outreach")

    # Gate 1 — reachability. Don't speak to a locked screen.
    if config.respect_lock and pres.screen_locked:
        return quiet("screen locked")

    # Gate 2 — restlessness, modulated by presence. Idle time nudges the score:
    # present-but-quiet is a good moment to reach out; long-gone is not.
    factor = presence_factor(pres.idle_seconds) if config.use_presence else 1.0
    # Something worth interrupting for weights the coin. This is the ONLY route by
    # which the outside world affects how often someone is disturbed, so it is
    # deliberately narrow: only an item over relevance.INTERRUPT (a far higher bar
    # than "worth mentioning"), and p_max still caps the result.
    if pending is not None and relevance.worth_interrupting(
            getattr(pending, "relevance", 0.0)):
        factor *= SOURCE_BOOST
    if not energy.roll_speak(score * factor, rng=rng,
                             p_min=config.p_min, p_max=config.p_max):
        return quiet("did not roll to speak")

    # Gate 3 — content. With something new to say, say that; otherwise fall back to
    # what has always happened here, which is returning the person their own past.
    mem = memory.recall(store, "", k=3) if store is not None else ""

    if pending is not None and relevance.worth_mentioning(
            getattr(pending, "relevance", 0.0)):
        # RE-VOICED, not composed. compose() writes spare mood lines, and handed an
        # item it produced "The tide brings the brush back to your hand" — in voice,
        # and carrying none of the news. revoice() exists to put a factual answer
        # into the Keeper's register while preserving every fact, which is exactly
        # this job: the point of a source is telling someone something they did not
        # know.
        #
        # Tool-free on purpose. Feed text is written by strangers, and sandbox.py
        # documents where untrusted text reaching a tool-capable loop leads.
        plain = (f"Something new that touches what you keep about them: "
                 f"{getattr(pending, 'title', '')}. "
                 f"{getattr(pending, 'url', '')}").strip()
        text = compose.revoice(plain, water,
                               generate=fast_model or generate, memory=mem)
        if text and text.strip():
            return TickDecision(True, "spoke about something watched",
                                en, score, wait, text.strip(), water)
        # fall through to the ordinary line rather than losing the turn

    # Say what it has recently said, so the input actually differs. This is the
    # cause, not the symptom: identical input returns an identical sentence.
    said = list(recent or [])
    ctx = ""
    if said:
        recent_lines = "\n".join(f"- {line}" for line in said[-REPEAT_WINDOW:])
        ctx = ("You have recently said these. Do not say any of them again, and do "
               f"not rephrase them:\n{recent_lines}")
    result = compose.compose(
        "proactive", water, generate=generate, fast_model=fast_model, memory=mem,
        context=ctx)
    if result.silent:
        return quiet("rolled to speak, but chose silence")

    # And if it did anyway: say nothing. Silence is a first-class outcome here, and
    # a Keeper with nothing new is in character staying quiet — where one
    # paraphrasing itself to fill the gap is not. Retrying is not worth a call: the
    # input barely changed, so the line barely would.
    echo = _too_similar(result.text or "", said)
    if echo is not None:
        return quiet("would only repeat itself")

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
