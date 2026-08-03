"""reminders.py — the things you ask the Keeper to hold and return at the right time.

The agentic core: you ask it to remember to do something ("remind me to call the
dentist tomorrow"); it stores that; and when the time comes the proactive loop
returns it to you. This is the one unbidden action that stays perfectly in
character — the Keeper keeping something, and giving it back.

Reminders can RECUR — the housekeeping layer. A reminder carries an optional
`repeat` ("daily", "weekly", "weekdays", or "every N minutes/hours/days/weeks");
when it's delivered, the store re-arms it to its next future occurrence instead of
retiring it, so "water the plants every day" keeps coming back.

Plain JSONL store, like the fact store. Times are epoch seconds; the model
converts natural language ("tomorrow 9am") into an ISO timestamp using the current
time given in its prompt, so there's no date-parsing dependency.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
REMINDERS_PATH = STORE_DIR / "reminders.jsonl"


@dataclass
class Reminder:
    text: str
    due_at: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: float = field(default_factory=time.time)
    done: bool = False
    delivered: bool = False
    repeat: Optional[str] = None      # None => one-shot; else a recurrence phrase
    fired_count: int = 0              # how many times a recurring one has returned


class ReminderStore:
    def __init__(self, path: Path = REMINDERS_PATH):
        self.path = path
        self.items: list[Reminder] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                self.items.append(Reminder(**json.loads(line)))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(
            json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in self.items))

    def add(self, text: str, due_at: float,
            repeat: Optional[str] = None) -> Reminder:
        repeat = normalize_repeat(repeat)
        r = Reminder(text=text.strip(), due_at=due_at, repeat=repeat)
        self.items.append(r)
        self._save()
        return r

    def pending(self) -> list[Reminder]:
        return sorted((r for r in self.items if not r.done),
                      key=lambda r: r.due_at)

    def due(self, now: Optional[float] = None) -> list[Reminder]:
        """Undelivered, not-done reminders whose time has come."""
        now = now or time.time()
        return [r for r in self.items
                if not r.done and not r.delivered and r.due_at <= now]

    def mark_delivered(self, rid: str, now: Optional[float] = None) -> None:
        """Retire a one-shot; RE-ARM a recurring one to its next future occurrence."""
        now = now or time.time()
        for r in self.items:
            if r.id != rid:
                continue
            r.fired_count += 1
            nxt = next_occurrence(r.due_at, r.repeat, now) if r.repeat else None
            if nxt is not None:
                r.due_at = nxt          # re-armed: stays undelivered for next time
            else:
                r.delivered = True      # one-shot (or unparseable recurrence): retire
        self._save()

    def complete(self, key: str) -> Optional[Reminder]:
        """Complete by id or by a text substring match. Ends recurrence too."""
        key_low = key.lower()
        for r in self.items:
            if not r.done and (r.id == key or key_low in r.text.lower()):
                r.done = True
                self._save()
                return r
        return None


# --------------------------------------------------------------------------- #
# Recurrence — small RRULE-lite over epoch seconds.
# --------------------------------------------------------------------------- #

_INTERVAL_RE = re.compile(
    r"^every\s+(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|"
    r"d|day|days|w|wk|week|weeks)$", re.I)

_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def normalize_repeat(repeat: Optional[str]) -> Optional[str]:
    """Canonicalize a recurrence phrase, or None if it isn't one we understand."""
    if not repeat:
        return None
    r = repeat.strip().lower()
    if r in ("daily", "every day"):
        return "daily"
    if r in ("hourly", "every hour"):
        return "every 1h"
    if r in ("weekly", "every week"):
        return "weekly"
    if r in ("weekdays", "every weekday", "weekday"):
        return "weekdays"
    m = _INTERVAL_RE.match(r)
    if m:
        return f"every {int(m.group(1))}{m.group(2)[0].lower()}"
    return None


def _step_seconds(repeat: str) -> Optional[int]:
    r = repeat.strip().lower()
    if r == "daily":
        return _UNIT_SECONDS["d"]
    if r == "weekly":
        return _UNIT_SECONDS["w"]
    m = _INTERVAL_RE.match(r)
    if m:
        return int(m.group(1)) * _UNIT_SECONDS[m.group(2)[0].lower()]
    return None


def next_occurrence(due_at: float, repeat: Optional[str],
                    now: Optional[float] = None) -> Optional[float]:
    """The next occurrence strictly after `now`, or None for a one-shot / unknown
    recurrence. Fixed intervals jump forward in whole steps (no catch-up storm if
    the app was off); 'weekdays' advances a day at a time skipping Sat/Sun."""
    repeat = normalize_repeat(repeat)
    if repeat is None:
        return None
    now = now or time.time()

    if repeat == "weekdays":
        nxt = due_at + _UNIT_SECONDS["d"]
        while nxt <= now or datetime.fromtimestamp(nxt).weekday() >= 5:
            nxt += _UNIT_SECONDS["d"]
        return nxt

    step = _step_seconds(repeat)
    if not step:
        return None
    nxt = due_at + step
    if nxt <= now:                      # skip forward to the next future slot
        missed = int((now - nxt) // step) + 1
        nxt += missed * step
    return nxt
