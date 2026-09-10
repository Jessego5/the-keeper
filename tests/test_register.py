"""
These are the Tier 1 tests for reading the person's register
(voice_eval.register_signal) + continuity.
"""
import pytest
import voice_eval as ve

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("msg", ["I am sad", "everything feels stuck",
                                 "i am so tired lately", "i feel hopeless"])
def test_distress_reads_frozen(msg):
    assert ve.register_signal(msg) == "frozen"


@pytest.mark.parametrize("msg", ["i feel a bit better today",
                                 "i finally went back to the studio",
                                 "things are lighter now"])
def test_upswing_reads_tidal(msg):
    assert ve.register_signal(msg) == "tidal"


@pytest.mark.parametrize("msg", ["what should i do", "what's on my list",
                                 "tell me a fact", "what time is it"])
def test_neutral_has_no_signal(msg):
    assert ve.register_signal(msg) is None


def _run_arc(messages):
    """Mirror the server's continuity logic over a sequence."""
    current = None
    out = []
    for m in messages:
        sig = ve.register_signal(m)
        if sig is not None:
            current = sig
        out.append(current or ve.detect_state(m))
    return out


def test_continuity_holds_frozen_through_neutral_followup():
    # REGRESSION: sad -> stuck -> "what should i do" must stay frozen, not flip
    # to tidal and tell a stuck person they're moving.
    arc = _run_arc(["I am sad", "I feel stuck", "what should i do"])
    assert arc == ["frozen", "frozen", "frozen"]


def test_continuity_flips_on_new_signal():
    arc = _run_arc(["I feel stuck", "what should i do",
                    "i actually feel better now", "what next"])
    assert arc == ["frozen", "frozen", "tidal", "tidal"]
