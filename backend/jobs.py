"""
These are the errands the Keeper runs on a schedule, because it was asked to.

A reminder hands back words at a time. A job does WORK at a time: "every weekday at
8, look at my calendar and mail and tell me what matters", "every Sunday, check
whether the gallery has posted its autumn dates". This is the piece that ChatGPT's
dots are mostly used for (their scheduled calendar reviews), and the one OpenClaw
and nanobot build in as cron. The Keeper already had every part of it except the
clock: background work that comes back on its own (spawn_task), recurrence
(reminders.next_occurrence), and a delivery path the dial and approvals govern.

Two rules, both about time:

  1. A job is re-armed BEFORE it runs. If the work crashes, or the process dies
     mid-run, the next occurrence is already scheduled and this one is not retried
     in a loop. Missing one brief is better than ten.
  2. A run that is badly late is skipped, not run. A morning brief composed at nine
     at night, because the laptop was shut all day, describes a day that is over.
     STALE_S decides "badly"; recurrence already refuses to catch up on every
     missed occurrence, so a week away costs one skipped run, not seven.

Kinds: "task" runs its own description through the specialists. "brief" is the
morning brief, whose content is assembled by server.py from what the Keeper holds
(proposals, reminders, goals) plus what a specialist reads from the calendar and
inbox. It is a kind rather than a task description so that what it contains is
code, testable, and the same every morning.

Plain JSONL, like reminders.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import reminders as reminders_mod

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
JOBS_PATH = STORE_DIR / "jobs.jsonl"

STALE_S = 6 * 3600.0
KINDS = ("task", "brief")


@dataclass
class Job:
    task: str                         # what to do, in their words (or "brief")
    next_at: float                    # when it next runs
    repeat: Optional[str] = None      # reminders.normalize_repeat; None = once
    kind: str = "task"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: float = field(default_factory=time.time)
    enabled: bool = True
    runs: int = 0
    last_run_at: Optional[float] = None
    last_status: str = ""             # done | failed | skipped (stale)
    last_result: str = ""


class JobStore:
    def __init__(self, path: Path = JOBS_PATH):
        self.path = path
        self.items: list[Job] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for raw in self.path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                self.items.append(Job(**json.loads(raw)))
            except (json.JSONDecodeError, TypeError):
                continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(
            json.dumps(asdict(j), ensure_ascii=False) + "\n" for j in self.items))

    def add(self, task: str, first_at: float, repeat: Optional[str] = None,
            kind: str = "task", now: Optional[float] = None) -> Job:
        """Schedule a job. A recurring one whose first time has already passed today
        ("every weekday at 8", said at noon) starts at its next real slot, the same
        roll native_tools does for recurring reminders."""
        now = now or time.time()
        repeat = reminders_mod.normalize_repeat(repeat)
        if repeat and first_at <= now:
            first_at = reminders_mod.next_occurrence(first_at, repeat, now) or first_at
        job = Job(task=task.strip(), next_at=first_at, repeat=repeat,
                  kind=kind if kind in KINDS else "task", created=now)
        self.items.append(job)
        self._save()
        return job

    def active(self) -> list[Job]:
        return sorted((j for j in self.items if j.enabled), key=lambda j: j.next_at)

    def due(self, now: Optional[float] = None) -> list[Job]:
        now = now or time.time()
        return [j for j in self.active() if j.next_at <= now]

    def claim(self, job: Job, now: Optional[float] = None) -> bool:
        """Take a due job for running: re-arm it (rule 1) and say whether it should
        actually run now, False when it is too late to be worth it (rule 2)."""
        now = now or time.time()
        late = now - job.next_at
        nxt = (reminders_mod.next_occurrence(job.next_at, job.repeat, now)
               if job.repeat else None)
        if nxt is None:
            job.enabled = False        # one-shot (or a recurrence it cannot read)
        else:
            job.next_at = nxt
        run = late <= STALE_S
        if not run:
            job.last_status = "skipped"
            job.last_run_at = now
            job.last_result = f"skipped: {late / 3600:.1f}h late"
        self._save()
        return run

    def finish(self, job: Job, result: str, ok: bool,
               now: Optional[float] = None) -> None:
        job.runs += 1
        job.last_run_at = now or time.time()
        job.last_status = "done" if ok else "failed"
        job.last_result = (result or "").strip()[:500]
        self._save()

    def cancel(self, key: str) -> Optional[Job]:
        """By id, or by a word from its task, like reminders."""
        key_low = key.lower().strip()
        if not key_low:
            return None
        for j in self.active():
            if j.id == key or key_low in j.task.lower():
                j.enabled = False
                self._save()
                return j
        return None

    def has_brief(self) -> bool:
        return any(j.kind == "brief" for j in self.active())
