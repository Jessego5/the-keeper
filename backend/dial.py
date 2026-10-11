"""
This is the person's say over how much the Keeper speaks on its own.

Every gate in proactive.py was tuned by the builder: the speak band, the day's
ceiling, the cooldown. The person living with the Keeper had no control over any
of it short of quitting the process. Muse and ChatGPT's dots both ship this as a
first-class control, and for the same reason: an agent that may start a
conversation is only bearable if you can tell it to start fewer.

Four levels, one dial:

    off     it starts nothing. No restless lines, no routines, nothing it noticed,
            no goal work. What you ASKED for still comes back: a reminder you set,
            a task you sent it off on, a decision it needs from you.
    quiet   about half as likely to speak, a third of the day's ceiling, twice the
            patience between goal check-ins.
    usual   exactly the tuned defaults. proactive_bench.py measures this level, so
            it must stay the identity.
    eager   more often, a higher ceiling, goals worked twice as often.

What no level can move is the real-time floor in server.py (MIN_REAL_REACH_GAP_S).
"eager" changes how willing the Keeper is, not how fast it is allowed to be: the
floor is the backstop against a bug, and a setting must not be a way around it.

The level lives in memory_store/dial.json so a restart does not quietly undo
"off", which is the one setting a person would be most upset to find reverted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import proactive
import unbidden

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
DIAL_PATH = STORE_DIR / "dial.json"


@dataclass(frozen=True)
class Level:
    name: str
    speak_scale: float        # multiplies the p_min/p_max speak band
    daily_max: Optional[int]  # None keeps the configured ceiling
    cooldown_scale: float     # multiplies cooldown_min
    goal_interval_scale: float  # multiplies how long a goal waits between checks
    initiative: bool = True   # False: the Keeper starts nothing of its own


LEVELS = {
    "off": Level("off", 0.0, None, 1.0, 1.0, initiative=False),
    "quiet": Level("quiet", 0.5, 4, 2.0, 2.0),
    "usual": Level("usual", 1.0, None, 1.0, 1.0),
    "eager": Level("eager", 1.5, 20, 0.5, 0.5),
}
DEFAULT = "usual"
# The speak band is a probability, and eager must not scale it past certainty: a
# Keeper that speaks on every roll it wins has stopped choosing.
P_CEILING = 0.9


def normalize(name: Optional[str]) -> Optional[str]:
    """A level name, or None if it is not one. Accepts a little slack in spelling,
    since it arrives from a Telegram command and a model's tool call alike."""
    if not name:
        return None
    n = name.strip().lower().lstrip("/")
    aliases = {"silent": "off", "none": "off", "quieter": "quiet", "low": "quiet",
               "normal": "usual", "default": "usual", "more": "eager",
               "high": "eager"}
    n = aliases.get(n, n)
    return n if n in LEVELS else None


class Dial:
    def __init__(self, path: Path = DIAL_PATH):
        self.path = path
        self.name = DEFAULT
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.name = normalize(json.loads(self.path.read_text()).get("level")) \
                or DEFAULT
        except (json.JSONDecodeError, OSError, AttributeError):
            self.name = DEFAULT

    def set(self, name: str) -> Optional[str]:
        """Set the level and keep it. Returns the level set, or None if `name` is
        not a level, in which case nothing changes."""
        level = normalize(name)
        if level is None:
            return None
        self.name = level
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"level": level}))
        return level

    @property
    def level(self) -> Level:
        return LEVELS[self.name]

    def allows(self, why: str) -> bool:
        """May a line of this kind go out at the current level? Only "off" ever
        says no, and only to the Keeper's own initiative."""
        return self.level.initiative or why in unbidden.ASKED_FOR

    def shape(self, config: proactive.ProactiveConfig) -> proactive.ProactiveConfig:
        """The config tick() should run with at this level. A copy: the knobs set
        through /config (speed, cooldown) stay what they were set to, and turning
        the dial back to usual returns exactly them."""
        lv = self.level
        if lv.name == DEFAULT:
            return config
        return replace(
            config,
            p_min=min(P_CEILING, config.p_min * lv.speak_scale),
            p_max=min(P_CEILING, config.p_max * lv.speak_scale),
            daily_max=lv.daily_max if lv.daily_max is not None else config.daily_max,
            cooldown_min=config.cooldown_min * lv.cooldown_scale,
        )

    def goal_interval(self, base_s: float) -> float:
        return base_s * self.level.goal_interval_scale
