"""reminders.py — the things you ask the Keeper to hold and return at the right time.

The agentic core: you ask it to remember to do something ("remind me to call the
dentist tomorrow"); it stores that; and when the time comes the proactive loop
returns it to you. This is the one unbidden action that stays perfectly in
character — the Keeper keeping something, and giving it back.

Plain JSONL store, like the fact store. Times are epoch seconds; the model
converts natural language ("tomorrow 9am") into an ISO timestamp using the current
time given in its prompt, so there's no date-parsing dependency.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
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

    def add(self, text: str, due_at: float) -> Reminder:
        r = Reminder(text=text.strip(), due_at=due_at)
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

    def mark_delivered(self, rid: str) -> None:
        for r in self.items:
            if r.id == rid:
                r.delivered = True
        self._save()

    def complete(self, key: str) -> Optional[Reminder]:
        """Complete by id or by a text substring match."""
        key_low = key.lower()
        for r in self.items:
            if not r.done and (r.id == key or key_low in r.text.lower()):
                r.done = True
                self._save()
                return r
        return None
