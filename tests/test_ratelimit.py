"""Tier 1 — the notification-spam circuit breaker (server.within_reach_floor).

The demo `speed` knob compresses the Keeper's virtual clock; this real-time floor
guarantees it can't translate into a banner flood no matter how high speed goes.
"""
import pytest
import server

pytestmark = pytest.mark.unit

GAP = server.MIN_REAL_REACH_GAP_S


def test_never_reached_out_can_reach():
    # last_proactive_at is None -> not inside the floor -> may reach out
    assert server.within_reach_floor(None, now=1000.0) is False


def test_just_spoke_is_blocked():
    # spoke 5s ago, floor is 60s -> still inside the floor -> must stay silent
    assert server.within_reach_floor(1000.0, now=1005.0) is True


def test_after_the_floor_can_reach():
    # spoke well over the floor ago -> free to reach out
    assert server.within_reach_floor(1000.0, now=1000.0 + GAP + 1) is False


def test_exactly_at_the_floor_can_reach():
    assert server.within_reach_floor(1000.0, now=1000.0 + GAP) is False


def test_floor_holds_regardless_of_speed():
    # No matter the compressed clock, two outreaches can't land within GAP real
    # seconds: if it spoke at t, every real moment before t+GAP is blocked.
    spoke_at = 5000.0
    blocked = [server.within_reach_floor(spoke_at, now=spoke_at + s)
               for s in range(0, int(GAP))]
    assert all(blocked)   # every second inside the window is silent


# --- the goal clock must follow the demo speed knob --- #

import tasks


def test_goal_check_interval_compresses_with_speed():
    """Regression: tasks.DEFAULT_CHECK_INTERVAL_S is 6 REAL hours, and the `speed`
    knob compressed the battery and the drift clock but not this one. So after a
    goal's first step it went quiet for 6 real hours whatever the speed — which made
    the documented "crank speed to see autonomous execution / nudges" trigger
    impossible, and a [person] step was never nudged in a demo."""
    original = server.STATE.config.speed
    try:
        server.STATE.config.speed = 1.0
        assert server._goal_check_interval() == tasks.DEFAULT_CHECK_INTERVAL_S
        server.STATE.config.speed = 600.0
        assert server._goal_check_interval() == tasks.DEFAULT_CHECK_INTERVAL_S / 600.0
        # 6h at 600x is 36s — inside a demo window, which is the whole point
        assert server._goal_check_interval() < 60
    finally:
        server.STATE.config.speed = original


def test_goal_check_interval_never_speeds_below_real_time():
    original = server.STATE.config.speed
    try:
        server.STATE.config.speed = 0.0          # nonsense input
        assert server._goal_check_interval() == tasks.DEFAULT_CHECK_INTERVAL_S
    finally:
        server.STATE.config.speed = original


# --- `turn` must be EARNED by the store, not matched in the message --- #



class _Fact:
    def __init__(self, text, warmed=True, created=None, active=True):
        import time as _t
        self.text, self.warmed = text, warmed
        self.created = created if created is not None else _t.time()
        self.active = active


class _Store:
    def __init__(self, warmings): self._w = warmings
    def recent_warmings(self, *a, **k): return self._w


def _with_store(monkeypatch, warmings):
    monkeypatch.setattr(server.STATE, "store", _Store(warmings))


def test_turn_is_earned_when_a_relevant_reversal_exists(monkeypatch):
    _with_store(monkeypatch, [_Fact("Started painting again.")])
    mem = "- Started painting again.\n- Is a painter."
    assert server._resolve_turn("how's the painting", mem, "tidal")[0] == "turn"


def test_an_irrelevant_reversal_does_not_earn_it(monkeypatch):
    """THE trap the naive version falls into. Any fact scored >= 8 is ambient and
    surfaces regardless of the cue, so co-presence alone would declare the ice
    going out on whatever happens to be in mind. Requiring the warming itself to
    have reached `mem` is what prevents that."""
    _with_store(monkeypatch, [_Fact("Started painting again.")])
    mem = "- Their mother is in the hospital."      # ambient, and unrelated
    assert server._resolve_turn("i had a good day", mem, "tidal")[0] == "tidal"


def test_frozen_is_never_overridden(monkeypatch):
    """Someone in the cold is not told their season has turned because the store
    remembers better days."""
    _with_store(monkeypatch, [_Fact("Started painting again.")])
    mem = "- Started painting again."
    assert server._resolve_turn("i feel numb", mem, "frozen")[0] == "frozen"


def test_no_reversal_means_no_turn(monkeypatch):
    """The greeting case: "hello" classified as turn because a single message was
    being asked to contain a reversal it cannot hold."""
    _with_store(monkeypatch, [])
    assert server._resolve_turn("hello", "", "tidal")[0] == "tidal"


def test_a_broken_store_never_costs_the_turn(monkeypatch):
    class Boom:
        def recent_warmings(self, *a, **k): raise RuntimeError("disk gone")
    monkeypatch.setattr(server.STATE, "store", Boom())
    assert server._resolve_turn("hello", "", "tidal")[0] == "tidal"


def test_the_classifier_alone_cannot_produce_a_turn(monkeypatch):
    """The greeting bug, structurally. "hello" classifies as turn — the rarest
    register — because a lone sentence was being asked to carry a reversal. With
    nothing in the store behind it, turn is demoted rather than trusted."""
    _with_store(monkeypatch, [])
    assert server._resolve_turn("hello", "", "turn")[0] == "tidal"


def test_a_relevant_reversal_keeps_a_turn(monkeypatch):
    _with_store(monkeypatch, [_Fact("Started painting again.")])
    mem = "- Started painting again."
    assert server._resolve_turn("how is the painting going", mem, "turn")[0] == "turn"


def test_the_words_must_overlap_the_reversal(monkeypatch):
    """Learned live: with a small store recall surfaces almost anything for a warm
    cue, so requiring only that the warming reached `mem` promoted "i had a good
    day today" on the strength of a painting fact it never mentioned."""
    _with_store(monkeypatch, [_Fact("Started painting again.")])
    mem = "- Started painting again."
    assert server._resolve_turn("i had a good day today", mem, "tidal")[0] == "tidal"
    assert server._resolve_turn("how is painting going", mem, "tidal")[0] == "turn"
