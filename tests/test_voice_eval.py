"""
These are the Tier 1 tests for the voice scorer (voice_eval.py) and its regressions.
"""
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
    # REGRESSION: 'your brother reached out', a stored state spoken as a fresh
    # event, motif-less notification-style. Must not clear threshold.
    for line in ["Your brother Sam has reached out again.",
                 "You got a message from Sam."]:
        assert not ve.evaluate(line).passed, line


def test_machinery_leak_hard_fails():
    # REGRESSION: the README's own chat capture. "i stopped painting in march"
    # tripped set_goal, and the goal title came back quoted, with the tool asking
    # permission to run. Rule 7: the machinery stays invisible.
    r = ve.evaluate("The tide is set, and it knows its course. The shore has a "
                    "calling: 'return to painting.' Would you like help with this?")
    assert r.hard_fail and not r.passed
    failed = {x.rule for x in r.results if not x.passed}
    assert {"no_quoted_value", "no_service_offer"} <= failed


def test_offer_in_the_keepers_own_idiom_fails():
    # REGRESSION: told not to SET a goal, the model went on offering to, in the
    # persona's vocabulary instead of the assistant's. It is still the tool asking
    # permission, and it scored 1.0 until the guard learned this shape.
    r = ve.evaluate("The water holds still and cold. If you wish for the tide to "
                    "draw back, you may ask me to tend that hope alongside you.")
    assert not r.passed
    assert "no_service_offer" in {x.rule for x in r.results if not x.passed}


def test_keeper_may_act_without_asking():
    # The guard must catch asking permission, not acting. Nothing here is an offer,
    # and 'alongside you' is kept out of the lexicon precisely so this passes.
    r = ve.evaluate("The tide moves alongside you. I will tend this until it is done.")
    assert "no_service_offer" in {x.rule for x in r.results if x.passed}


def test_plain_practical_answer_survives_the_leak_guards():
    # Rule 4: terseness never costs them the facts. A tool-grounded answer has no
    # quoted field and offers nothing, so the guards must leave it alone.
    r = ve.evaluate("A beginner watercolour set runs about 25 dollars. Over a "
                    "year that is 48 cents a week.")
    assert not r.hard_fail
    assert {"no_quoted_value", "no_service_offer"} <= {
        x.rule for x in r.results if x.passed}


def test_leak_guards_do_not_inflate_clean_lines():
    # The guards are penalties, not credits. Scoring them as ordinary always-on
    # rules lifted test_invented_event_reportage_fails over threshold, so a clean
    # line must earn exactly nothing for lacking a leak.
    for x in ve.evaluate(ON_VOICE[0]).results:
        if x.rule in ("no_quoted_value", "no_service_offer"):
            assert x.passed and x.weight == 0.0


def test_detect_state_only_returns_compose_states():
    for text in ["the ice is going out", "the tide came back", "hello", "stuck"]:
        assert ve.detect_state(text) in ("frozen", "tidal", "turn")
