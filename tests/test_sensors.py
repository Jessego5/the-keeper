"""Tier 2 — OS presence sensing (sensors.py).

Logic (context rendering, gate wiring) is deterministic via injected Presence.
The live-read probe touches real macOS APIs and skips off-mac.
"""
import sys
import pytest
import sensors

pytestmark = pytest.mark.integration

requires_mac = pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS")


def test_context_line_renders_all_signals():
    p = sensors.Presence(idle_seconds=200, screen_locked=False, frontmost_app="Code")
    line = p.to_context_line()
    assert "idle" in line and "awake" in line and "Code" in line


def test_context_line_empty_when_blind():
    p = sensors.Presence()   # all None
    assert p.to_context_line() == ""
    assert not p.available


def test_human_duration_scales():
    assert sensors._human_duration(5) == "5s"
    assert sensors._human_duration(300) == "5m"
    assert sensors._human_duration(7200) == "2h"


@requires_mac
def test_live_read_returns_signals():
    p = sensors.read()
    assert p.available
    assert p.idle_seconds is not None and p.idle_seconds >= 0
    assert isinstance(p.screen_locked, bool)


@requires_mac
def test_idle_command_parses_a_number():
    val = sensors.read_idle_seconds()
    assert val is not None and val >= 0


# --- capability reporting --- #

def test_capabilities_names_every_signal():
    """Every presence signal must be accounted for, on any platform. The readers
    fail silently by design (the proactive loop reads every few seconds and must not
    flood the log), so this report is the only place a permanently blind sensor
    becomes visible instead of quietly showing defaults."""
    caps = sensors.capabilities()
    assert set(caps) == {"idle_seconds", "screen_locked", "frontmost_app"}
    assert all(isinstance(v, str) and v for v in caps.values())


@requires_mac
def test_capabilities_report_live_on_mac():
    caps = sensors.capabilities()
    assert caps["idle_seconds"] == "live"


def test_capabilities_explain_themselves_off_mac(monkeypatch):
    """Off-mac every signal must say WHY, not just be absent."""
    monkeypatch.setattr(sensors, "_IS_MAC", False)
    caps = sensors.capabilities()
    assert all(v != "live" for v in caps.values())
    assert all("platform" in v for v in caps.values())
