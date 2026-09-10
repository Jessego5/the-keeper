"""
These are the Tier 1 tests for the house's presence-driven routines (routines.py). No
key, no clock.
"""
from datetime import datetime

import pytest
import routines
import sensors

pytestmark = pytest.mark.unit

# 2026-08-05 is a Wednesday; 14:00 local is a plain focused afternoon.
AFTERNOON = datetime(2026, 8, 5, 14, 0).timestamp()
EVENING = datetime(2026, 8, 5, 22, 0).timestamp()

ACTIVE = sensors.Presence(idle_seconds=30.0, screen_locked=False, frontmost_app="Code")
AWAY = sensors.Presence(idle_seconds=3600.0, screen_locked=False, frontmost_app="Code")
LOCKED = sensors.Presence(idle_seconds=30.0, screen_locked=True, frontmost_app="Code")
BLIND = sensors.Presence()   # no signals


def test_heads_down_after_long_focus():
    e = routines.RoutineEngine()
    e.observe(ACTIVE, AFTERNOON)                       # start of a present stretch
    r = e.check(ACTIVE, AFTERNOON + 91 * 60)           # 91 continuous minutes later
    assert r is not None and r.key == "heads_down"


def test_no_heads_down_before_threshold():
    e = routines.RoutineEngine()
    e.observe(ACTIVE, AFTERNOON)
    assert e.check(ACTIVE, AFTERNOON + 30 * 60) is None   # only 30 min in


def test_welcome_back_after_absence():
    e = routines.RoutineEngine()
    e.observe(AWAY, AFTERNOON)                          # they've been gone ~1h
    r = e.check(ACTIVE, AFTERNOON + 60)                 # now back at the keyboard
    assert r is not None and r.key == "welcome_back"


def test_welcome_back_consumed_after_firing():
    e = routines.RoutineEngine()
    e.observe(AWAY, AFTERNOON)
    r = e.check(ACTIVE, AFTERNOON + 60)
    e.fire(r, AFTERNOON + 60)
    # same presence moments later must not welcome again (consumed + cooldown)
    again = e.check(ACTIVE, AFTERNOON + 120)
    assert again is None or again.key != "welcome_back"


def test_wind_down_in_the_evening():
    e = routines.RoutineEngine()
    r = e.check(ACTIVE, EVENING)
    assert r is not None and r.key == "wind_down"


def test_cooldown_blocks_refire():
    e = routines.RoutineEngine()
    r = e.check(ACTIVE, EVENING)
    e.fire(r, EVENING)
    assert e.check(ACTIVE, EVENING + 5 * 60) is None    # wind_down on 10h cooldown


def test_locked_screen_fires_nothing():
    e = routines.RoutineEngine()
    assert e.check(LOCKED, EVENING) is None             # not "present"


def test_blind_sensor_fires_nothing():
    e = routines.RoutineEngine()
    assert e.check(BLIND, EVENING) is None


def test_status_snapshot():
    e = routines.RoutineEngine()
    e.observe(ACTIVE, AFTERNOON)
    s = e.status(AFTERNOON + 10 * 60)
    assert s["active_minutes"] == 10.0
    assert "welcome_back" in s["routines"]
