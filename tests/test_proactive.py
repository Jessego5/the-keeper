"""Tier 1 — the reach-out decision (proactive.py). Gating only; no LLM, no key."""
import random
import pytest
import proactive
import sensors

pytestmark = pytest.mark.unit


def _speaks(system, user):
    return "The tide returns what you gave it."


def _silent(system, user):
    return "SILENCE"


RESTLESS = proactive.ProactiveState(minutes_since_user=5000, recent_msg_count=6)
AWAKE = sensors.Presence(idle_seconds=30, screen_locked=False, frontmost_app="x")
LOCKED = sensors.Presence(idle_seconds=30, screen_locked=True, frontmost_app="x")


def test_cooldown_gate_holds_silence():
    st = proactive.ProactiveState(minutes_since_user=5000, recent_msg_count=6,
                                  minutes_since_proactive=5)   # < cooldown
    d = proactive.tick(st, generate=_speaks, presence=AWAKE,
                       rng=random.Random(0))
    assert not d.spoke and "cooldown" in d.reason


def test_lock_gate_holds_silence():
    d = proactive.tick(RESTLESS, generate=_speaks, presence=LOCKED,
                       rng=random.Random(0))
    assert not d.spoke and d.reason == "screen locked"


def test_silence_when_compose_declines():
    # roll may pass, but the generator declines -> stays quiet, never a bad line
    spoke = False
    for seed in range(20):
        d = proactive.tick(RESTLESS, generate=_silent, presence=AWAKE,
                           rng=random.Random(seed))
        spoke = spoke or d.spoke
    assert not spoke


def test_presence_factor_bands():
    assert proactive.presence_factor(30) > 1.0     # present -> nudge up
    assert proactive.presence_factor(600) == 1.0   # brief away -> neutral
    assert proactive.presence_factor(3600) < 1.0   # gone -> ease off


def test_presence_modulates_roll_rate():
    def rate(idle):
        pres = sensors.Presence(idle_seconds=idle, screen_locked=False,
                                frontmost_app="x")
        rng = random.Random(0)
        return sum(proactive.tick(RESTLESS, generate=_silent, presence=pres,
                                  rng=rng).reason.startswith("rolled")
                   for _ in range(200))
    assert rate(30) > rate(3600)   # present rolls more often than gone


def test_simulation_rate_is_sane_not_spammy():
    # worst case (stub never declines): still only a handful over 48h thanks to
    # cooldown + rolls, not one per tick.
    res = proactive.simulate(48.0, seed=7, generate=_speaks, verbose=False)
    assert 1 <= res.spoke <= 10
