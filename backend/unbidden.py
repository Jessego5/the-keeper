"""
This is the record of everything the Keeper said without being asked.

A line the Keeper decides to say is the rarest thing it produces: gated by the day's
ceiling, a cooldown, the screen lock, a roll and a composer that may still decline.
Until now it was also the most fragile. It was fanned out to whoever happened to be
listening on /events at that instant, and a tab mid-reconnect, or no tab at all,
simply never saw it. The banner fired, the person looked, and the page had nothing.

So every unbidden line is written here first, then delivered. Each gets a sequence
number, which the web page receives as the SSE event id. When the browser's
EventSource reconnects it sends that id back as Last-Event-ID, and the server replays
whatever was said after it. The browser already does the remembering; the server
only has to keep its side of the bargain.

It also answers the question an agent that works while you are away has to answer:
what did you do while I was gone? `why` records which part of the Keeper spoke (a
kept reminder, a goal step, a routine, something it noticed, plain restlessness), so
that question has a truthful answer rather than a guess.

Plain append-only JSONL, like the journal. A day's ceiling is a dozen lines, so this
grows by a few kilobytes a month and is never compacted.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
UNBIDDEN_PATH = STORE_DIR / "unbidden.jsonl"


class Why:
    """Which part of the Keeper spoke. A closed set, like proactive.Reason, so the
    dashboard can count them without matching on prose."""

    REMINDER = "reminder"        # a promise kept: something it was asked to hold
    GOAL = "goal"                # a goal step, taken or nudged
    ROUTINE = "routine"          # the house noticed a return, long focus, evening
    NOTICED = "noticed"          # something it watches bore on a kept fact
    RESTLESS = "restless"        # the battery ran low and the roll said speak
    BACKGROUND = "background"    # a task it was sent off on came back


@dataclass
class Line:
    seq: int                     # monotonic; the SSE event id
    text: str
    why: str = Why.RESTLESS
    state: Optional[str] = None  # the register it was composed in
    session: Optional[str] = None
    ts: float = field(default_factory=time.time)


class UnbiddenLog:
    def __init__(self, path: Path = UNBIDDEN_PATH):
        self.path = path
        self.lines: list[Line] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for raw in self.path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                self.lines.append(Line(**json.loads(raw)))
            except (json.JSONDecodeError, TypeError):
                # A torn last line from a crash mid-write. Losing one record is
                # better than refusing to start, which would lose all of them.
                continue

    @property
    def last_seq(self) -> int:
        return self.lines[-1].seq if self.lines else 0

    def record(self, text: str, why: str = Why.RESTLESS,
               state: Optional[str] = None, session: Optional[str] = None,
               now: Optional[float] = None) -> Line:
        """Append one line, and only then hand it back to be delivered. Writing
        first is the point: a line that was delivered but never recorded is exactly
        the line a reconnecting page cannot get back."""
        line = Line(seq=self.last_seq + 1, text=text, why=why, state=state,
                    session=session, ts=now or time.time())
        self.lines.append(line)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(line), ensure_ascii=False) + "\n")
        return line

    def after(self, seq: int, limit: int = 50) -> list[Line]:
        """What was said after `seq`, oldest first. Bounded, so a page that has been
        asleep for a month is caught up on the recent past, not flooded with it."""
        missed = [ln for ln in self.lines if ln.seq > seq]
        return missed[-limit:]

    def recent(self, limit: int = 20) -> list[Line]:
        return self.lines[-limit:]


def parse_last_event_id(header: Optional[str]) -> Optional[int]:
    """The browser's Last-Event-ID, or None if it sent nothing usable. A fresh page
    sends nothing, and gets no replay: it loads the conversation itself."""
    try:
        return int((header or "").strip())
    except ValueError:
        return None
