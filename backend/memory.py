"""memory.py — the drawers. What the Keeper keeps, and how it hands things back.

Two jobs:

    distill(messages, generate) -> facts     after a chat, extract durable facts
    recall(store, cue, k)       -> string     before a turn, surface relevant ones

Storage is a plain JSONL file (memory_store/facts.jsonl) — no vector DB in v1, by
design (REFERENCE.md: biggest time sink, least visible payoff). Recall is keyword
overlap + recency + how often a fact has recurred. It is deliberately simple; the
magic is not the retrieval algorithm, it is that the Keeper *has* the fact at all.

Lesson baked in from the first voice test: facts are stored as clean third-person
statements ("has a brother, Sam; not spoken since spring"), never as raw meta-notes
("mentioned a brother twice"). And recall renders them under "what you keep" so the
Keeper returns them as keeping, never recites them as a record.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

# (system_prompt, user_message) -> raw model text. Same shape compose.py uses.
Generator = Callable[[str, str], str]
# texts -> one embedding vector each. Injected like Generator; None => keyword-only.
Embedder = Callable[[list[str]], list[list[float]]]

# Above this cosine, two facts are "the same thing said differently" (semantic dedup).
_SEMANTIC_DUP = 0.86

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
STORE_PATH = STORE_DIR / "facts.jsonl"

# Words too common to help keyword matching.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "is", "was", "are", "were", "be", "been", "it", "this", "that",
    "i", "you", "he", "she", "they", "we", "me", "my", "your", "not", "no",
    "has", "have", "had", "do", "does", "did", "so", "as", "if", "then", "about",
}


@dataclass
class Fact:
    """One durable thing the Keeper keeps about the person."""

    text: str
    kind: str = "event"            # identity | state | event | preference
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    mentions: int = 1              # how many times it has resurfaced
    embedding: Optional[list[float]] = None   # semantic vector (None => keyword-only)

    def age_days(self, now: Optional[float] = None) -> float:
        return ((now or time.time()) - self.last_seen) / 86400.0

    def when(self) -> str:
        return datetime.fromtimestamp(self.created, timezone.utc).date().isoformat()


# --------------------------------------------------------------------------- #
# Store — load / append / dedup. A thin wrapper over a JSONL file.
# --------------------------------------------------------------------------- #

class MemoryStore:
    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self.facts: list[Fact] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                self.facts.append(Fact(**json.loads(line)))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(json.dumps(asdict(f), ensure_ascii=False) + "\n"
                    for f in self.facts))

    def add(self, text: str, kind: str = "event",
            embed: Optional[Embedder] = None) -> Optional[Fact]:
        """Add a fact, or bump an existing duplicate instead of duplicating.

        With an embedder, dedup is SEMANTIC (cosine): "has a brother, Sam" and
        "her brother is named Sam" collapse even though their words differ — which
        keyword jaccard misses. Without one, falls back to keyword dedup.
        Returns the Fact if stored/updated, None if it was empty.
        """
        text = text.strip()
        if not text:
            return None
        vec = None
        if embed is not None:
            try:
                vec = embed([text])[0]
            except Exception:  # noqa: BLE001 - degrade to keyword, never crash
                vec = None
        dupe = self._find_similar(text, vec)
        if dupe is not None:
            dupe.mentions += 1
            dupe.last_seen = time.time()
            if dupe.embedding is None and vec is not None:
                dupe.embedding = vec       # backfill a vector on an old fact
            self._save()
            return dupe
        fact = Fact(text=text, kind=kind, embedding=vec)
        self.facts.append(fact)
        self._save()
        return fact

    def add_many(self, items: Iterable[tuple[str, str]],
                 embed: Optional[Embedder] = None) -> list[Fact]:
        out = [self.add(text, kind, embed) for text, kind in items]
        return [f for f in out if f is not None]

    def _find_similar(self, text: str,
                      vec: Optional[list[float]] = None) -> Optional[Fact]:
        # Semantic dedup when we have a vector for the new fact and stored vectors.
        if vec is not None:
            best, best_sim = None, 0.0
            for f in self.facts:
                if f.embedding is None:
                    continue
                sim = _cosine(vec, f.embedding)
                if sim > best_sim:
                    best, best_sim = f, sim
            if best is not None and best_sim >= _SEMANTIC_DUP:
                return best
        # Keyword fallback (also covers facts with no vectors).
        toks = _tokens(text)
        if not toks:
            return None
        for f in self.facts:
            if _jaccard(toks, _tokens(f.text)) >= 0.6:
                return f
        return None


# --------------------------------------------------------------------------- #
# Recall — surface the facts that matter for this turn.
# --------------------------------------------------------------------------- #

def recall(store: MemoryStore, cue: str = "", k: int = 5,
           now: Optional[float] = None, embed: Optional[Embedder] = None) -> str:
    """Return up to k relevant facts rendered for the system prompt's memory slot.

    With an embedder and a cue, recall is SEMANTIC: it embeds the cue and ranks
    facts by cosine similarity (blended with recency/recurrence), so "tell me
    about my sibling" surfaces "has a brother, Sam" even with no shared words.
    Falls back to keyword overlap when there is no embedder, no cue (proactive/
    drift), or the query can't be embedded. Empty string if the store is empty.
    """
    if not store.facts:
        return ""
    now = now or time.time()

    # ── semantic path ──
    if embed is not None and cue.strip() and any(f.embedding for f in store.facts):
        cue_vec = None
        try:
            cue_vec = embed([cue])[0]
        except Exception:  # noqa: BLE001
            cue_vec = None
        if cue_vec is not None:
            scored = []
            for f in store.facts:
                if f.embedding is None:
                    continue
                sim = _cosine(cue_vec, f.embedding)
                score = sim * 3.0 + _presence(f, now)
                scored.append((score, f))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [f for _, f in scored[:k]]
            if top:
                return "\n".join(f"- {f.text}" for f in top)

    # ── keyword / presence fallback ──
    cue_toks = _tokens(cue)
    ranked = sorted(store.facts, key=lambda f: _score(f, cue_toks, now), reverse=True)
    top = [f for f in ranked if _score(f, cue_toks, now) > 0][:k]
    if not top:
        top = ranked[:k]
    return "\n".join(f"- {f.text}" for f in top)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. Pure Python — no heavy deps, fine at this
    scale (hundreds of facts x ~1k dims). Swap in numpy / a vector DB to scale."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _presence(f: Fact, now: float) -> float:
    """Recency + recurrence — how 'present' a fact is, independent of the cue."""
    recency = 1.0 / (1.0 + f.age_days(now))
    recurrence = min(f.mentions, 5) / 5.0
    return recency * 0.6 + recurrence * 0.4


def _score(f: Fact, cue_toks: set[str], now: float) -> float:
    overlap = len(cue_toks & _tokens(f.text))
    recency = 1.0 / (1.0 + f.age_days(now))        # 1.0 today -> decays with age
    recurrence = min(f.mentions, 5) / 5.0          # capped so it can't dominate
    # Keyword match is the strongest signal when a cue exists; otherwise
    # presence (recency + recurrence) carries the ranking.
    return overlap * 3.0 + recency * 1.5 + recurrence


# --------------------------------------------------------------------------- #
# Distill — turn a finished conversation into durable facts.
# --------------------------------------------------------------------------- #

_DISTILL_SYSTEM = """You extract durable facts about a person from a conversation, \
for a companion's long-term memory. Output ONLY facts worth keeping for months: \
who they are, their situation, relationships, ongoing struggles, things they care \
about. Skip small talk, passing moods, and anything about the companion.

Write each fact as ONE short third-person statement of the fact itself — the thing \
that is true, not the act of saying it. Write "Has a brother, Sam; not spoken since \
spring," never "Mentioned a brother." Prefix each with its kind in brackets: \
[identity] [state] [event] [preference].

One fact per line, no bullets, no numbering. If there is nothing worth keeping, \
output exactly NONE."""

_KIND_RE = re.compile(r"^\[(identity|state|event|preference)\]\s*", re.I)


def distill(messages: list[dict], generate: Generator,
            store: Optional[MemoryStore] = None,
            embed: Optional[Embedder] = None) -> list[Fact]:
    """Extract durable facts from a conversation and, if a store is given, keep them.

    messages: [{"role": "user"|"assistant", "content": str}, ...]
    generate: the injected model call (use a cheap one — the fast_model is ideal).
    embed:    optional embedder; extracted facts are embedded for semantic recall.

    Returns the Facts extracted (already stored if `store` was provided).
    """
    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages if m.get("content"))
    if not transcript.strip():
        return []

    raw = generate(_DISTILL_SYSTEM, transcript).strip()
    if not raw or raw.strip().upper().strip(".!") == "NONE":
        return []

    parsed: list[tuple[str, str]] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if not line:
            continue
        m = _KIND_RE.match(line)
        kind = m.group(1).lower() if m else "event"
        text = _KIND_RE.sub("", line).strip()
        if text:
            parsed.append((text, kind))

    if store is None:
        return [Fact(text=t, kind=k) for t, k in parsed]
    return store.add_many(parsed, embed=embed)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


if __name__ == "__main__":
    # Recall demo — pure logic, no model needed. Uses a temp store.
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "facts.jsonl"
    store = MemoryStore(tmp)
    store.add("Has a brother, Sam; not spoken since spring.", "identity")
    store.add("Is a painter; stopped working in spring.", "identity")
    store.add("Left a project unfinished.", "event")
    store.add("Feels stuck lately, can't get started.", "state")
    dup = store.add("Feels stuck, can't get started these days.", "state")  # near-dup

    print(f"stored facts: {len(store.facts)} (last add merged, "
          f"mentions={dup.mentions})\n")
    for cue in ["i saw my brother today", "back at the easel", ""]:
        print(f"cue: {cue!r}")
        print(recall(store, cue, k=3))
        print()
