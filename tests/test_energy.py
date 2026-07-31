"""Tier 1 — the battery math (energy.py)."""
import pytest
import energy

pytestmark = pytest.mark.unit


def test_full_when_just_talked():
    assert energy.compute_energy(0) == pytest.approx(1.0, abs=1e-6)


def test_never_talked_is_zero():
    assert energy.compute_energy(None) == 0.0


def test_decays_monotonically_with_silence():
    mins = [0, 30, 120, 720, 2880, 5760]
    vals = [energy.compute_energy(m) for m in mins]
    assert vals == sorted(vals, reverse=True)          # strictly falling
    assert vals[-1] < 0.1                               # nearly drained after days


def test_hunger_is_inverse_of_energy():
    assert energy.hunger(1.0) == pytest.approx(0.0)
    assert energy.hunger(0.0) == pytest.approx(1.0)


def test_base_score_rises_as_energy_falls():
    hi = energy.base_score(energy.compute_energy(0), msg_count=4)
    lo = energy.base_score(energy.compute_energy(3000), msg_count=4)
    assert lo > hi


def test_speak_probability_stays_in_band():
    for s in (0.0, 0.5, 1.0, -1, 2):
        p = energy.speak_probability(s, p_min=0.05, p_max=0.45)
        assert 0.05 <= p <= 0.45


def test_restless_ticks_faster_than_calm():
    fast = energy.next_tick_seconds(0.9, jitter=0)
    slow = energy.next_tick_seconds(0.1, jitter=0)
    assert fast < slow
