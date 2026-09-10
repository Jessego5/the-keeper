"""
This is persistent conversation history for the Keeper.

Gives the Keeper's own stack what the external bridge borrowed: multiple
conversations, each stored to disk, listable and switchable, so the UI sidebar is
real and conversations survive a restart (closing the in-memory-history gap).

One JSON file per conversation under memory_store/sessions/. The store is the
source of truth for what the sidebar shows; the server still keeps a short flat
history for the proactive/energy logic.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store" / "sessions"


@dataclass
class Msg:
    role: str          # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    key: str
    title: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    messages: list = field(default_factory=list)   # list[Msg]


class SessionStore:
    def __init__(self, directory: Path = STORE_DIR):
        self.dir = directory
        self.sessions: dict[str, Session] = {}
        self._load()

    def _load(self) -> None:
        if not self.dir.exists():
            return
        for f in self.dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                d["messages"] = [Msg(**m) for m in d.get("messages", [])]
                self.sessions[d["key"]] = Session(**d)
            except (json.JSONDecodeError, OSError, TypeError):
                continue

    def _save(self, s: Session) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{s.key}.json").write_text(
            json.dumps(asdict(s), ensure_ascii=False))

    def new(self) -> Session:
        s = Session(key="s-" + uuid.uuid4().hex[:12])
        self.sessions[s.key] = s
        self._save(s)
        return s

    def get(self, key: str) -> Optional[Session]:
        return self.sessions.get(key)

    def all(self) -> list[Session]:
        """Conversations with at least one message, newest first."""
        return sorted((s for s in self.sessions.values() if s.messages),
                      key=lambda s: s.updated, reverse=True)

    def append(self, key: str, role: str, content: str,
               ts: Optional[float] = None) -> None:
        s = self.sessions.get(key)
        if s is None:
            return
        s.messages.append(Msg(role=role, content=content, ts=ts or time.time()))
        s.updated = time.time()
        if not s.title and role == "user":
            s.title = content.strip()[:80]
        self._save(s)

    def most_recent_key(self) -> Optional[str]:
        alls = self.all()
        return alls[0].key if alls else None
