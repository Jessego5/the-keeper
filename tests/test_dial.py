"""
These are the Tier 1 tests for the proactivity dial (dial.py): what each level does
to the gates, what "off" still lets through, and that the setting survives a restart.
No network, no key.
"""
import pytest

import dial
import native_tools
import proactive
import reminders
import unbidden

pytestmark = pytest.mark.unit


@pytest.fixture
def d(tmp_path):
    return dial.Dial(tmp_path / "dial.json")


def test_usual_is_exactly_the_tuned_defaults(d):
    """proactive_bench.py measures the defaults. If "usual" drifted from them, the
    benchmark would be measuring a level nobody runs."""
    cfg = proactive.ProactiveConfig()
    assert d.name == "usual"
    assert d.shape(cfg) is cfg


def test_quiet_halves_the_band_and_lowers_the_ceiling(d):
    cfg = proactive.ProactiveConfig()
    d.set("quiet")
    q = d.shape(cfg)
    assert q.p_max == pytest.approx(cfg.p_max / 2)
    assert q.p_min == pytest.approx(cfg.p_min / 2)
    assert q.daily_max == 4
    assert q.cooldown_min == pytest.approx(cfg.cooldown_min * 2)


def test_shaping_never_touches_the_configured_knobs(d):
    """Turning the dial back must return exactly what /config set, so shape()
    works on a copy."""
    cfg = proactive.ProactiveConfig(speed=600, cooldown_min=5)
    d.set("eager")
    d.shape(cfg)
    assert (cfg.speed, cfg.cooldown_min, cfg.p_max) == (600, 5, 0.45)


def test_eager_cannot_scale_the_band_past_choosing(d):
    d.set("eager")
    assert d.shape(proactive.ProactiveConfig(p_max=0.8)).p_max == dial.P_CEILING


def test_off_silences_initiative_and_nothing_asked_for(d):
    d.set("off")
    for why in (unbidden.Why.RESTLESS, unbidden.Why.ROUTINE,
                unbidden.Why.NOTICED, unbidden.Why.GOAL):
        assert not d.allows(why), f"{why} got through at off"
    for why in unbidden.ASKED_FOR:
        assert d.allows(why), f"{why} was silenced, but the person asked for it"


def test_every_level_but_off_allows_everything(d):
    for name in ("quiet", "usual", "eager"):
        d.set(name)
        assert d.allows(unbidden.Why.RESTLESS)


def test_off_survives_a_restart(tmp_path):
    """The one setting a person would be most upset to find quietly reverted."""
    dial.Dial(tmp_path / "dial.json").set("off")
    assert dial.Dial(tmp_path / "dial.json").name == "off"


def test_an_unknown_level_changes_nothing(d):
    d.set("quiet")
    assert d.set("whisper") is None
    assert d.name == "quiet"


def test_a_corrupt_file_falls_back_to_usual(tmp_path):
    (tmp_path / "dial.json").write_text("{not json")
    assert dial.Dial(tmp_path / "dial.json").name == "usual"


@pytest.mark.parametrize("raw,expected", [
    ("/quiet", "quiet"), ("OFF", "off"), ("silent", "off"), ("normal", "usual"),
    ("more", "eager"), ("  eager ", "eager"), ("loud", None), ("", None),
])
def test_level_names_take_a_little_slack(raw, expected):
    assert dial.normalize(raw) == expected


def test_goal_patience_follows_the_dial(d):
    d.set("quiet")
    assert d.goal_interval(100.0) == 200.0
    d.set("eager")
    assert d.goal_interval(100.0) == 50.0


# --- the tool, on the person's own turn only --- #

async def test_the_person_can_turn_it_from_chat(d, tmp_path):
    tools = native_tools.NativeTools(
        reminders.ReminderStore(tmp_path / "r.jsonl"), dial=d)
    assert "set_dial" in {t["function"]["name"] for t in tools.openai_tools()}
    assert await tools.call("set_dial", {"level": "off"}) == "set to off"
    assert d.name == "off"


def test_a_sub_agent_is_not_handed_the_dial(tmp_path):
    """Sub-agents and A2A peers build NativeTools without a dial, so a peer agent
    cannot silence the Keeper, or turn it up."""
    tools = native_tools.NativeTools(
        reminders.ReminderStore(tmp_path / "r.jsonl"), allow_delegate=False)
    assert "set_dial" not in {t["function"]["name"] for t in tools.openai_tools()}
