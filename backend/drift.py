"""drift.py — the Keeper's inner life.

When the Keeper is idle (a proactive tick chose silence), it occasionally drifts:
it reflects on what it keeps about the person and writes a private note — an
evolving, honest understanding ("what do I actually know, what am I assuming, what
would I ask if they came back"). This is the original plan's self-reflection task
and the reference agent's "drift": when there's nothing to say, do a little background thinking
instead of nothing.

Design choices:
  - Reflections are the Keeper's OWN thoughts, so they live in a SEPARATE log
    (reflections.jsonl), not the fact store — they inform and are viewable, but
    they never pollute what the Keeper "was told" and recalls back to the person.
  - Rate-limited by an interval (like the reference agent's drift min_interval), so it's a
    quiet background hum, not a busy loop.
  - Model injected (compose's drift mode), so it's testable offline with the stub.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import compose
import memory

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
REFLECTIONS_PATH = STORE_DIR / "reflections.jsonl"


@dataclass
class Reflection:
    text: str
    created: float = field(default_factory=time.time)


class ReflectionLog:
    """Append-only log of the Keeper's private reflections."""

    def __init__(self, path: Path = REFLECTIONS_PATH):
        self.path = path
        self.items: list[Reflection] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                self.items.append(Reflection(**json.loads(line)))

    def add(self, r: Reflection) -> None:
        self.items.append(r)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def latest(self) -> Optional[Reflection]:
        return self.items[-1] if self.items else None


@dataclass
class DriftConfig:
    enabled: bool = True
    min_interval_min: float = 180.0   # at most one reflection every ~3h
    speed: float = 1.0                # shares the proactive demo clock


def reflect(store: memory.MemoryStore, generate: compose.Generator,
            log: Optional[ReflectionLog] = None) -> Reflection:
    """Reflect on current memory and produce a private note (and log it)."""
    mem = memory.recall(store, "", k=10)   # what's most present
    result = compose.compose("drift", generate=generate, memory=mem)
    note = (result.text or "").strip() or "Nothing has changed. I keep what I have."
    r = Reflection(text=note)
    if log is not None:
        log.add(r)
    return r


def maybe_drift(store: memory.MemoryStore, generate: compose.Generator,
                log: ReflectionLog, *, last_drift_at: Optional[float],
                config: DriftConfig = DriftConfig(),
                now: Optional[float] = None) -> Optional[Reflection]:
    """Reflect only if drift is enabled and enough (virtual) time has passed since
    the last reflection. Returns the Reflection if one was made, else None."""
    if not config.enabled:
        return None
    now = now or time.time()
    if last_drift_at is not None:
        elapsed_min = (now - last_drift_at) / 60.0 * max(config.speed, 1.0)
        if elapsed_min < config.min_interval_min:
            return None
    return reflect(store, generate, log)


if __name__ == "__main__":
    import tempfile
    store = memory.MemoryStore(Path(tempfile.mkdtemp()) / "f.jsonl")
    store.add("Is a painter; stopped in March.", "identity")
    store.add("Has a brother, Sam, not called since spring.", "state")
    log = ReflectionLog(Path(tempfile.mkdtemp()) / "reflections.jsonl")
    g, _ = compose.make_generator()
    r = reflect(store, g, log)
    print("reflection:\n ", r.text)
    print("\nlogged:", len(log.items))
