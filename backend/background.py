"""
These are the long tasks the Keeper goes off and works on, returning later.

The other delegation model. `delegate`/`orchestrate` are SYNCHRONOUS: the Keeper
waits with you while its specialists work, then answers in the same breath. Some
tasks are too long for that: real research, digging through a lot. For those the
Keeper spawns a BACKGROUND task, tells you it's on it, and brings the result back
later: on its own, through the same channels a proactive line uses (chat + a native
banner). This is the reference agent's SpawnTool + Poller idea: fire-and-forget, report on
completion. On-persona too: a keeper that goes away, tends to something, and returns.

This module is just the registry (what's running, what finished). The actual running
and delivery live in the server, which owns the event loop and the channels.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BgTask:
    description: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "running"          # running | done | failed
    result: str = ""
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None


class BackgroundTasks:
    def __init__(self):
        self.tasks: dict[str, BgTask] = {}

    def add(self, description: str) -> BgTask:
        t = BgTask(description=description.strip())
        self.tasks[t.id] = t
        return t

    def finish(self, tid: str, result: str, ok: bool = True) -> None:
        t = self.tasks.get(tid)
        if t is not None:
            t.status = "done" if ok else "failed"
            t.result = result
            t.finished = time.time()

    def running(self) -> list[BgTask]:
        return [t for t in self.tasks.values() if t.status == "running"]
