"""drift.py — the Keeper's inner life.

When the Keeper is idle (a proactive tick chose silence), it occasionally drifts.
Two modes:

  reflect()    — a single private note in the Keeper's voice (the quiet hum).
  synthesize() — the Generative Agents reflection step (Park et al., 2023): from
                 what it currently holds, generate the most salient high-level
                 QUESTIONS, retrieve the facts relevant to each, and write grounded
                 INSIGHTS — higher-level understanding that follows *from* the facts
                 rather than restating them. Insights are stored back as retrievable
                 memories (kind='insight'), exactly as reflections re-enter the
                 memory stream in the paper.

Persona guardrail: an insight is the Keeper's OWN conclusion, so memory.recall
renders insights under a separate heading ("what you've come to understand"), never
as something the person said. That keeps the paper's retrievable-reflection
mechanism while preventing the "your brother reached out" recitation bug. Every
insight is also logged to reflections.jsonl for the dashboard.

Rate-limited by an interval (like the reference agent's drift min_interval) so it's a quiet
background hum. Model injected, so it's testable offline with a stub.
"""

from __future__ import annotations

import json
import re
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
    if not mem.strip():                    # empty store -> don't invent a person
        r = Reflection(text="The water is still. Nothing kept yet.")
        if log is not None:
            log.add(r)
        return r
    result = compose.compose("drift", generate=generate, memory=mem)
    note = (result.text or "").strip() or "Nothing has changed. I keep what I have."
    r = Reflection(text=note)
    if log is not None:
        log.add(r)
    return r


# --- Generative Agents reflection: questions -> retrieve -> grounded insights --- #

_MIN_FACTS_TO_SYNTHESIZE = 3

_QUESTIONS_SYSTEM = """You are the private reflective mind of a companion that keeps \
memories about one person. Given the things it currently holds, ask the 1-3 most \
salient high-level questions whose answers would deepen its understanding of who this \
person is and what they are moving through. Output only the questions, one per line, \
no numbering, no preamble."""

_INSIGHT_SYSTEM = """You are the private reflective mind of a companion. Given a \
question and the relevant things it keeps about the person, write ONE insight: a \
higher-level understanding that FOLLOWS FROM those facts — a pattern, a tension, a \
likely need — not a restatement of any single fact. Ground it only in what is given; \
never invent events. Write it as the companion's own quiet conclusion about the \
person, one sentence (e.g. "She keeps circling back to what she left unfinished"). \
Prefix it with an importance from 1 to 10 in brackets, e.g. [7]. If the facts support \
no honest higher-level read, output exactly NONE."""

_IMP_PREFIX_RE = re.compile(r"^\[(\d{1,2}(?:\.\d+)?)\]\s*")


def synthesize(store: memory.MemoryStore, generate: compose.Generator,
               log: Optional[ReflectionLog] = None,
               embed: Optional[memory.Embedder] = None,
               *, max_questions: int = 3) -> list[memory.Fact]:
    """Run one reflection cycle, storing grounded insights as retrievable memories.

    Returns the insight Facts created (already stored, embedded, and — if a log is
    given — appended to the reflection log). Returns [] when there is too little to
    reflect on (< _MIN_FACTS_TO_SYNTHESIZE facts); the caller falls back to reflect().
    """
    seed = memory.rank_facts(store.facts, "", k=12, embed=embed)
    if len(seed) < _MIN_FACTS_TO_SYNTHESIZE:
        return []

    rendered = "\n".join(f"- {f.text}" for f in seed)
    q_raw = (generate(_QUESTIONS_SYSTEM, rendered) or "").strip()
    questions = [ln.strip().lstrip("-*0123456789. ").strip()
                 for ln in q_raw.splitlines() if ln.strip()][:max_questions]

    insights: list[memory.Fact] = []
    seen: set[str] = set()
    for q in questions:
        relevant = memory.rank_facts(store.facts, q, k=6, embed=embed)
        if not relevant:
            continue
        ctx = "\n".join(f"- {f.text}" for f in relevant)
        raw = (generate(_INSIGHT_SYSTEM, f"Question: {q}\n\nWhat you keep:\n{ctx}")
               or "").strip()
        m = _IMP_PREFIX_RE.match(raw)
        imp = float(m.group(1)) if m else 6.0
        text = _IMP_PREFIX_RE.sub("", raw).strip()
        if not text or text.upper().strip(".!") == "NONE":
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        fact = store.add(text, kind="insight", embed=embed, importance=imp)
        if fact is not None:
            insights.append(fact)
            if log is not None:
                log.add(Reflection(text=fact.text))
    return insights


def maybe_drift(store: memory.MemoryStore, generate: compose.Generator,
                log: ReflectionLog, *, last_drift_at: Optional[float],
                config: DriftConfig = DriftConfig(),
                now: Optional[float] = None,
                embed: Optional[memory.Embedder] = None) -> Optional[Reflection]:
    """Drift only if enabled and enough (virtual) time has passed since the last one.

    Prefers the richer synthesize() path (grounded, retrievable insights); falls back
    to a single voice-y reflect() note when there is too little to synthesize. Returns
    the latest Reflection made, or None if it didn't drift."""
    if not config.enabled:
        return None
    # Nothing kept yet -> nothing to reflect on. Never let the idle mind invent a
    # person from an empty store (it will happily hallucinate a whole history).
    if not any(f.active for f in store.facts):
        return None
    now = now or time.time()
    if last_drift_at is not None:
        elapsed_min = (now - last_drift_at) / 60.0 * max(config.speed, 1.0)
        if elapsed_min < config.min_interval_min:
            return None
    insights = synthesize(store, generate, log, embed=embed)
    if insights:
        return log.latest() if log is not None else Reflection(text=insights[-1].text)
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
