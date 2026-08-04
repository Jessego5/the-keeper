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
