"""routines.py — the housekeeper's routines. Presence-driven reasons to reach out.

The generic proactive loop (proactive.py) reaches out from restlessness — a random
roll gated by energy. Routines are the opposite: specific, earned moments the Keeper
watches for and responds to in character.

    welcome_back  they've just returned after being away a while
    heads_down    they've been focused, unbroken, for a long stretch — a breath
    wind_down     it's late in their evening — a soft close to the day

Each routine is a pure trigger over a RoutineContext plus a cooldown so it can't
nag. The engine is the only stateful part: it watches the idle signal across ticks
to notice TRANSITIONS (away -> back, and how long they've been continuously active),
which a single instantaneous sensor read can't tell you. When a routine fires, the
server composes the line with the routine's intent passed as compose() `context`, so
it comes out in the Keeper's voice — never a canned string.

All time is injectable (`now`), so the whole thing is testable offline with no clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import sensors

# Idle thresholds (seconds): below ACTIVE = actively here; above AWAY = stepped off.
ACTIVE_IDLE = 300.0
AWAY_IDLE = 1800.0
# A continuous-active stretch this long (minutes) earns a "come up for air" nudge.
HEADS_DOWN_MIN = 90.0
# Must have been away at least this long (minutes) for a return to count as one.
WELCOME_AWAY_MIN = 45.0


@dataclass
class RoutineContext:
    """Everything a routine trigger needs, computed fresh each check."""

    now: float
    local_hour: int
    idle_seconds: Optional[float]
    screen_locked: Optional[bool]
    minutes_since_user: Optional[float]
    active_minutes: float          # how long they've been continuously present
    just_returned: bool            # came back this tick after a real absence

    @property
    def present(self) -> bool:
        return (self.idle_seconds is not None and self.idle_seconds < ACTIVE_IDLE
                and not self.screen_locked)


@dataclass
class Routine:
    key: str
    intent: str                    # steers the composed line (passed as context)
    cooldown_min: float
    trigger: Callable[[RoutineContext], bool]


def default_routines() -> list[Routine]:
    """The house's built-in routines, in priority order (first match wins)."""
    return [
        Routine(
            key="welcome_back",
            intent="They've just come back after being away a while. Greet their "
                   "return quietly, without making a fuss over it.",
            cooldown_min=90.0,
            trigger=lambda c: c.just_returned and c.present),
        Routine(
            key="heads_down",
            intent="They've been heads-down and focused for a long unbroken stretch. "
                   "Offer one quiet invitation to pause or breathe — no pressure.",
            cooldown_min=120.0,
            trigger=lambda c: c.present and c.active_minutes >= HEADS_DOWN_MIN),
        Routine(
            key="wind_down",
            intent="It is late in their evening. Offer a soft close to the day; if "
                   "something is being held for tomorrow you may gesture at it.",
            cooldown_min=600.0,
            trigger=lambda c: c.present and 21 <= c.local_hour < 24),
    ]


@dataclass
class RoutineEngine:
    """Watches presence across ticks and decides when a routine has earned a line."""

    routines: list[Routine] = field(default_factory=default_routines)
    _last_fired: dict = field(default_factory=dict)
    _active_since: Optional[float] = None    # start of the current present stretch
    _away_since: Optional[float] = None      # when they stepped off (approx)
    _returned: bool = False                  # latched until a welcome_back fires
    last_fired_key: Optional[str] = None

    def observe(self, presence: sensors.Presence, now: float) -> None:
        """Update transition state from one presence read. Idempotent per tick."""
        idle = presence.idle_seconds
        if idle is None:
            return
        if idle < ACTIVE_IDLE and not presence.screen_locked:
            # actively here now
            if (self._away_since is not None
                    and (now - self._away_since) / 60.0 >= WELCOME_AWAY_MIN):
                self._returned = True
            self._away_since = None
            if self._active_since is None:
                self._active_since = now
        else:
            # idle or locked -> not continuously present
            self._active_since = None
            if idle >= AWAY_IDLE and self._away_since is None:
                # they've already been gone ~idle seconds; date the absence back
                self._away_since = now - idle

    def active_minutes(self, now: float) -> float:
        if self._active_since is None:
            return 0.0
        return (now - self._active_since) / 60.0

    def context(self, presence: sensors.Presence, now: float,
                minutes_since_user: Optional[float]) -> RoutineContext:
        return RoutineContext(
            now=now,
            local_hour=datetime.fromtimestamp(now).hour,
            idle_seconds=presence.idle_seconds,
            screen_locked=presence.screen_locked,
            minutes_since_user=minutes_since_user,
            active_minutes=self.active_minutes(now),
            just_returned=self._returned)

    def check(self, presence: sensors.Presence, now: float,
              minutes_since_user: Optional[float] = None) -> Optional[Routine]:
        """Observe, then return the first eligible routine (off cooldown, trigger
        true), or None. Does NOT mark it fired — call fire() once its line lands."""
        self.observe(presence, now)
        ctx = self.context(presence, now, minutes_since_user)
        for r in self.routines:
            last = self._last_fired.get(r.key)
            if last is not None and (now - last) / 60.0 < r.cooldown_min:
                continue
            if r.trigger(ctx):
                return r
        return None

    def fire(self, routine: Routine, now: float) -> None:
        """Record that a routine's line actually went out (starts its cooldown)."""
        self._last_fired[routine.key] = now
        self.last_fired_key = routine.key
        if routine.key == "welcome_back":
            self._returned = False   # consume the return

    def status(self, now: Optional[float] = None) -> dict:
        """A small snapshot for the dashboard."""
        now = now or (self._active_since or 0.0)
        return {
            "active_minutes": round(self.active_minutes(now), 1) if now else 0.0,
            "awaiting_return": self._returned,
            "last_routine": self.last_fired_key,
            "routines": [r.key for r in self.routines],
        }
