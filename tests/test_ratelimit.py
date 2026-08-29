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
