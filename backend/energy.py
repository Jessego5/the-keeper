"""energy.py — the battery. When does the Keeper get restless enough to speak?

Ported from the reference agent's proactive_v2/energy.py (multi-timescale exponential
decay), trimmed to what v1 needs and rewritten in English. The model:

    E(t) = alpha*exp(-t/tau1) + beta*exp(-t/tau2) + gamma*exp(-t/tau3)

Three timescales of "warmth" since the person last spoke:
    tau1 = 30 min    the afterglow of a conversation
    tau2 = 4 h       still the same day
    tau3 = 48 h      the continuity of the relationship

High energy (just talked) -> the Keeper leaves you alone. As energy decays the
Keeper grows restless: it checks more often and is likelier to reach out.

Two signals combine into a base_score, which drives everything downstream:
    hunger  = 1 - energy                 the longer the silence, the higher
    context = log(1+k)/log(1+scale)      how much was recently said

base_score -> a probability of speaking (lerp between a floor and a ceiling), and
-> the wait until the next tick (higher score = shorter wait = more chances).
This file is pure math and holds no state; proactive.py drives it.
"""

from __future__ import annotations

import math
import random as _random


def compute_energy(
    minutes_since_user: float | None,
    *,
    alpha: float = 0.50,
    beta: float = 0.35,
    gamma: float = 0.15,
    tau1_min: float = 30.0,
    tau2_min: float = 240.0,
    tau3_min: float = 2880.0,
) -> float:
    """Current battery in [0, 1]. Never talked -> 0.0 (fully restless)."""
    if minutes_since_user is None:
        return 0.0
    t = max(0.0, minutes_since_user)
    return (
        alpha * math.exp(-t / tau1_min)
        + beta * math.exp(-t / tau2_min)
        + gamma * math.exp(-t / tau3_min)
    )


def hunger(energy: float) -> float:
    """Interaction hunger: low energy (long silence) -> high hunger. [0, 1]."""
    return 1.0 - max(0.0, min(1.0, energy))


def context_richness(msg_count: int, scale: float = 10.0) -> float:
    """How much was recently said, log-normalized to [0, 1].

    scale=10:  0 msgs -> 0.00,  5 -> 0.59,  10 -> 0.76,  20 -> 0.92
    """
    if msg_count <= 0:
        return 0.0
    return min(1.0, math.log1p(msg_count) / math.log1p(max(scale, 1.0)))


def base_score(
    energy: float,
    msg_count: int,
    *,
    w_hunger: float = 0.7,
    w_context: float = 0.3,
) -> float:
    """Combine hunger and recent-context into the master restlessness score [0, 1].

    Hunger dominates (a long silence is the main reason to reach out); recent
    context nudges it (a rich recent thread makes a follow-up more natural).
    """
    return w_hunger * hunger(energy) + w_context * context_richness(msg_count)


def speak_probability(
    score: float,
    *,
    p_min: float = 0.05,
    p_max: float = 0.45,
) -> float:
    """Map base_score -> probability of *attempting* to speak on this tick.

    A linear lerp between a floor and a ceiling (the reference agent's anyaction band). Even
    a restless Keeper only sometimes speaks; even a satisfied one rarely might.
    """
    s = max(0.0, min(1.0, score))
    return p_min + (p_max - p_min) * s


def roll_speak(score: float, *, rng: _random.Random | None = None,
               p_min: float = 0.05, p_max: float = 0.45) -> bool:
    """The frequency gate: draw once against speak_probability(score)."""
    p = speak_probability(score, p_min=p_min, p_max=p_max)
    return (rng or _random).random() < p


def next_tick_seconds(
    score: float,
    *,
    tick_fast: int = 2400,   # score > 0.20 -> ~40 min between checks
    tick_slow: int = 4800,   # score <= 0.20 -> ~80 min
    jitter: float = 0.3,
    rng: _random.Random | None = None,
) -> int:
    """Seconds until the next tick. Higher score -> shorter wait -> more chances."""
    base = tick_fast if score > 0.20 else tick_slow
    if jitter <= 0:
        return base
    r = (rng or _random).uniform(1.0 - jitter, 1.0 + jitter)
    return max(1, int(base * r))
