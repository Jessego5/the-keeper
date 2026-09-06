"""Tier 1 — the reach-out decision (proactive.py). Gating only; no LLM, no key."""
import random
import pytest
import proactive
import sensors

pytestmark = pytest.mark.unit


def _speaks(system, user):
    return "The tide returns what you gave it."


def _silent(system, user):
    return "SILENCE"


RESTLESS = proactive.ProactiveState(minutes_since_user=5000, recent_msg_count=6)
AWAKE = sensors.Presence(idle_seconds=30, screen_locked=False, frontmost_app="x")
LOCKED = sensors.Presence(idle_seconds=30, screen_locked=True, frontmost_app="x")


def test_cooldown_gate_holds_silence():
    st = proactive.ProactiveState(minutes_since_user=5000, recent_msg_count=6,
                                  minutes_since_proactive=5)   # < cooldown
    d = proactive.tick(st, generate=_speaks, presence=AWAKE,
                       rng=random.Random(0))
    assert not d.spoke and "cooldown" in d.reason


def test_lock_gate_holds_silence():
    d = proactive.tick(RESTLESS, generate=_speaks, presence=LOCKED,
                       rng=random.Random(0))
    assert not d.spoke and d.reason == "screen locked"


def test_silence_when_compose_declines():
    # roll may pass, but the generator declines -> stays quiet, never a bad line
    spoke = False
    for seed in range(20):
        d = proactive.tick(RESTLESS, generate=_silent, presence=AWAKE,
                           rng=random.Random(seed))
        spoke = spoke or d.spoke
    assert not spoke


def test_presence_factor_bands():
    assert proactive.presence_factor(30) > 1.0     # present -> nudge up
    assert proactive.presence_factor(600) == 1.0   # brief away -> neutral
    assert proactive.presence_factor(3600) < 1.0   # gone -> ease off


def test_presence_modulates_roll_rate():
    def rate(idle):
        pres = sensors.Presence(idle_seconds=idle, screen_locked=False,
                                frontmost_app="x")
        rng = random.Random(0)
        return sum(proactive.tick(RESTLESS, generate=_silent, presence=pres,
                                  rng=rng).reason.startswith("rolled")
                   for _ in range(200))
    assert rate(30) > rate(3600)   # present rolls more often than gone


def test_simulation_rate_is_sane_not_spammy():
    # worst case (stub never declines): still only a handful over 48h thanks to
    # cooldown + rolls, not one per tick.
    res = proactive.simulate(48.0, seed=7, generate=_speaks, verbose=False)
    assert 1 <= res.spoke <= 10


# --- a watched source may weight the coin, but never bypass a gate --- #

import sources as _src


def _item(score):
    it = _src.SourceItem(source="feed", title="Erin Milez's Dense Paintings",
                         url="https://x/1")
    it.relevance, it.because = score, "Is a painter."
    return it


class _AlwaysRoll(random.Random):
    """Fixes only the speak draw. Subclasses Random so uniform() and the rest still
    work — next_tick_seconds jitters with uniform, and a bare stub breaks it."""
    def __init__(self, draw):
        super().__init__(0)
        self._draw = draw
    def random(self): return self._draw


def test_a_relevant_item_raises_the_odds_of_speaking():
    """The design decision: news the person cares about should shorten the wait,
    not merely improve what is said once the coin happens to come up."""
    state = proactive.ProactiveState(minutes_since_user=120.0, recent_msg_count=0)
    cfg = proactive.ProactiveConfig(cooldown_min=0.0, use_presence=False)
    pres = sensors.Presence(idle_seconds=60.0, screen_locked=False)
    base = proactive.tick(state, generate=lambda s, u: "line", config=cfg,
                          presence=pres, rng=_AlwaysRoll(0.30))
    boosted = proactive.tick(state, generate=lambda s, u: "line", config=cfg,
                             presence=pres, rng=_AlwaysRoll(0.30),
                             pending=_item(0.9))
    # Assert on the GATE, not on what the composer then did with it: the same draw
    # fails the roll without an item and passes it with one. Whether a line
    # survives voice scoring afterwards is compose's business, not this gate's.
    assert base.reason == "did not roll to speak"
    assert boosted.reason != "did not roll to speak"


def test_a_merely_mentionable_item_does_not_buy_an_interruption():
    """Over MENTION but under INTERRUPT: it may colour what is said, it may not
    make the Keeper speak."""
    state = proactive.ProactiveState(minutes_since_user=120.0, recent_msg_count=0)
    cfg = proactive.ProactiveConfig(cooldown_min=0.0, use_presence=False)
    pres = sensors.Presence(idle_seconds=60.0, screen_locked=False)
    d = proactive.tick(state, generate=lambda s, u: "line", config=cfg,
                       presence=pres, rng=_AlwaysRoll(0.30), pending=_item(0.5))
    assert not d.spoke


def test_no_item_may_break_the_cooldown():
    state = proactive.ProactiveState(minutes_since_user=120.0, recent_msg_count=0,
                                     minutes_since_proactive=1.0)
    cfg = proactive.ProactiveConfig(cooldown_min=600.0)
    d = proactive.tick(state, generate=lambda s, u: "line", config=cfg,
                       presence=sensors.Presence(idle_seconds=60.0,
                                                 screen_locked=False),
                       rng=_AlwaysRoll(0.0), pending=_item(1.0))
    assert not d.spoke and "cooldown" in d.reason


def test_no_item_may_speak_to_a_locked_screen():
    state = proactive.ProactiveState(minutes_since_user=120.0, recent_msg_count=0)
    cfg = proactive.ProactiveConfig(cooldown_min=0.0)
    d = proactive.tick(state, generate=lambda s, u: "line", config=cfg,
                       presence=sensors.Presence(idle_seconds=60.0,
                                                 screen_locked=True),
                       rng=_AlwaysRoll(0.0), pending=_item(1.0))
    assert not d.spoke and "locked" in d.reason


def test_an_item_is_re_voiced_so_the_news_survives():
    """compose() writes spare mood lines: handed this item live it produced "The
    tide brings the brush back to your hand", which carries no news at all. revoice
    preserves every fact while putting it in register, so the item's substance
    reaches the person."""
    seen = {}
    def gen(system, user):
        seen["user"] = user
        return "Erin Milez paints crowded cities."
    state = proactive.ProactiveState(minutes_since_user=120.0, recent_msg_count=0)
    cfg = proactive.ProactiveConfig(cooldown_min=0.0, use_presence=False)
    d = proactive.tick(state, generate=gen, config=cfg,
                       presence=sensors.Presence(idle_seconds=60.0,
                                                 screen_locked=False),
                       rng=_AlwaysRoll(0.0), pending=_item(0.9))
    assert "Erin Milez" in seen.get("user", ""), "the item never reached the model"
    assert d.spoke and "Erin Milez" in d.text
    assert d.reason == "spoke about something watched"
