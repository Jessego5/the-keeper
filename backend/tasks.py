"""tasks.py — goals the Keeper pursues over time. The Planning pattern's state.

This is what turns the Keeper from a companion that reaches out into an AGENT that
tends to things. A Goal is something the person wants help moving toward ("get back
to painting", "sort things out with Sam"); the planner (planner.py) decomposes it
into small concrete Steps; and the proactive loop advances it over days — taking a
step with its tools, or checking in — instead of only emitting a mood line.

State is a plain JSONL file (memory_store/goals.jsonl), like the fact and reminder
stores. Each Goal carries its steps, its status, and when it is next due to be
worked, so the loop can pick the goal that has waited longest without nagging.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
GOALS_PATH = STORE_DIR / "goals.jsonl"

# How long (real seconds) to wait before a goal is due to be worked again, so the
# agent makes steady progress without pestering. ~6h by default.
DEFAULT_CHECK_INTERVAL_S = 6 * 3600.0


@dataclass
class Step:
    text: str
    done: bool = False
    note: str = ""          # what happened when it was worked (a result / a reply)


@dataclass
class Goal:
    title: str                       # what the person wants ("get back to painting")
    steps: list = field(default_factory=list)     # list[Step]
    status: str = "active"           # active | done | paused
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    next_check_at: float = field(default_factory=time.time)  # when to work it next

    def next_step(self) -> Optional[Step]:
        """The first unfinished step, or None if all are done."""
        for s in self.steps:
            if not s.done:
                return s
        return None

    def progress(self) -> tuple:
        done = sum(1 for s in self.steps if s.done)
        return done, len(self.steps)


class GoalStore:
    def __init__(self, path: Path = GOALS_PATH):
        self.path = path
        self.goals: list[Goal] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["steps"] = [Step(**s) for s in d.get("steps", [])]
            self.goals.append(Goal(**d))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(
            json.dumps(asdict(g), ensure_ascii=False) + "\n" for g in self.goals))

    def add(self, title: str, steps: list[str]) -> Goal:
        g = Goal(title=title.strip(),
                 steps=[Step(text=s.strip()) for s in steps if s.strip()])
        self.goals.append(g)
        self._save()
        return g

    def active(self) -> list[Goal]:
        return [g for g in self.goals if g.status == "active"]

    def get(self, key: str) -> Optional[Goal]:
        """By id, or by a case-insensitive substring of the title."""
        key_low = key.lower()
        for g in self.goals:
            if g.id == key or key_low in g.title.lower():
                return g
        return None

    def due(self, now: Optional[float] = None) -> Optional[Goal]:
        """The active, unfinished goal most overdue to be worked, or None."""
        now = now or time.time()
        ready = [g for g in self.active()
                 if g.next_step() is not None and g.next_check_at <= now]
        if not ready:
            return None
        return min(ready, key=lambda g: g.next_check_at)   # longest-waiting first

    def advance(self, goal: Goal, note: str = "",
                interval_s: float = DEFAULT_CHECK_INTERVAL_S,
                now: Optional[float] = None) -> Optional[Step]:
        """Mark the current step done, record what happened, and schedule the next
        check. Auto-completes the goal when the last step is done. Returns the step
        that was just completed (or None if there was nothing to do)."""
        now = now or time.time()
        step = goal.next_step()
        if step is None:
            return None
        step.done = True
        step.note = note.strip()
        goal.updated = now
        goal.next_check_at = now + interval_s
        if goal.next_step() is None:
            goal.status = "done"
        self._save()
        return step

    def complete(self, key: str) -> Optional[Goal]:
        g = self.get(key)
        if g is not None:
            g.status = "done"
            g.updated = time.time()
            self._save()
        return g

    def touch(self, goal: Goal, interval_s: float = DEFAULT_CHECK_INTERVAL_S,
              now: Optional[float] = None) -> None:
        """Push a goal's next check out without completing a step — used when the
        agent only checked in rather than making progress."""
        goal.next_check_at = (now or time.time()) + interval_s
        self._save()
