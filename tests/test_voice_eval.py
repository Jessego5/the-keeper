"""Tier 1 — the voice scorer (voice_eval.py) and its regressions."""
import pytest
import voice_eval as ve

pytestmark = pytest.mark.unit

ON_VOICE = [
    "The water is still. You have kept still before, in a longer cold.",
    "The tide returns what it took. You are back.",
    "The ice is going out. I have watched you cross this break before.",
]


@pytest.mark.parametrize("line", ON_VOICE)
def test_on_voice_lines_pass(line):
    assert ve.evaluate(line).passed


def test_hedging_hard_fails():
    r = ve.evaluate("Maybe things sort of get better, I think.")
    assert r.hard_fail and not r.passed


def test_greeting_card_hard_fails():
    r = ve.evaluate("Don't give up! Brighter days are coming on your healing journey.")
    assert r.hard_fail and not r.passed


def test_invented_event_reportage_fails():
    # REGRESSION: 'your brother reached out' — a stored state spoken as a fresh
    # event, motif-less notification-style. Must not clear threshold.
    for line in ["Your brother Sam has reached out again.",
                 "You got a message from Sam."]:
        assert not ve.evaluate(line).passed, line


def test_detect_state_only_returns_compose_states():
    for text in ["the ice is going out", "the tide came back", "hello", "stuck"]:
        assert ve.detect_state(text) in ("frozen", "tidal", "turn")
