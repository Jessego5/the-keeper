"""journal.py — the Keeper's kept notes. The one thing it can WRITE.

Everything else the Keeper touches is read-only (MCP files/git/web are jailed and
stripped of mutating tools). The journal is the deliberate exception: a place it can
write, because a keeper keeps things. It is APPEND-ONLY by construction — keep() only
ever adds a line; there is no overwrite and no delete — so the Keeper can never lose
or clobber what it (or you) put here. That is what makes a write tool safe to hand an
autonomous agent.

Plain JSONL at memory_store/kept.jsonl, like the other stores.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
JOURNAL_PATH = STORE_DIR / "kept.jsonl"


@dataclass
class Entry:
    text: str
    ts: float = field(default_factory=time.time)

    def when(self) -> str:
        return datetime.fromtimestamp(self.ts).strftime("%b %d, %H:%M")


class Journal:
    def __init__(self, path: Path = JOURNAL_PATH):
        self.path = path
        self.entries: list[Entry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                self.entries.append(Entry(**json.loads(line)))

    def keep(self, text: str) -> Entry | None:
        """Append one note. The only mutation this module allows — never overwrites."""
        text = text.strip()
        if not text:
            return None
        e = Entry(text=text)
        self.entries.append(e)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:               # append mode: cannot clobber
            fh.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
        return e

    def recent(self, k: int = 10) -> list[Entry]:
        return self.entries[-k:]
