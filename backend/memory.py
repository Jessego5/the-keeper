"""memory.py — the drawers. What the Keeper keeps, and how it hands things back.

Two jobs:

    distill(messages, generate) -> facts     after a chat, extract durable facts
    recall(store, cue, k)       -> string     before a turn, surface relevant ones

Retrieval follows the Generative Agents memory model (Park et al., 2023,
"Generative Agents: Interactive Simulacra of Human Behavior"): each memory is
scored by a weighted sum of RECENCY (exponential decay since last touched),
IMPORTANCE (a 1-10 poignancy the model assigns when the fact is formed), and
RELEVANCE (embedding cosine to the current cue). The three are min-max normalized
across the candidate set and summed, exactly as in the paper's retrieval function.
Consolidation under memory pressure follows MemGPT (Packer et al., 2023): when the
store grows past a budget, old low-importance facts are summarized and archived.

Storage is a plain JSONL file (memory_store/facts.jsonl) — no vector DB; cosine is
pure Python, fine at this scale (hundreds of facts). Swap in a vector store to grow.

Lesson baked in from the first voice test: facts are stored as clean third-person
statements ("has a brother, Sam; not spoken since spring"), never as raw meta-notes
("mentioned a brother twice"). And recall renders them under "what you keep" so the
Keeper returns them as keeping, never recites them as a record.
"""

from __future__ import annotations

import json
import math
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

# Consolidation (MemGPT memory pressure): compress the low-value tail once the
# active store grows past this many facts. Archived originals are never deleted.
_CONSOLIDATE_BUDGET = 60

# Words too common to help keyword matching.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "is", "was", "are", "were", "be", "been", "it", "this", "that",
    "i", "you", "he", "she", "they", "we", "me", "my", "your", "not", "no",
    "has", "have", "had", "do", "does", "did", "so", "as", "if", "then", "about",
}


# Generative Agents retrieval weights (paper uses 1/1/1) and the hourly recency
# decay (the paper's 0.99 per game-hour; 0.995 here for a gentler human timescale).
_W_RECENCY = 1.0
_W_IMPORTANCE = 1.0
_W_RELEVANCE = 1.0
_RECENCY_DECAY = 0.995


@dataclass
class Fact:
    """One durable thing the Keeper keeps about the person.

    `kind` is 'insight' for the Keeper's OWN synthesized understanding (from
    reflection), which recall renders separately so it is never recited back as
    something the person said — see recall() and drift.synthesize().
    """

    text: str
    kind: str = "event"            # identity | state | event | preference | insight
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    mentions: int = 1              # how many times it has resurfaced
    importance: float = 5.0        # 1-10 poignancy (Generative Agents), model-assigned
    embedding: Optional[list[float]] = None   # semantic vector (None => keyword-only)

    def age_days(self, now: Optional[float] = None) -> float:
        return ((now or time.time()) - self.last_seen) / 86400.0

    def recency(self, now: Optional[float] = None,
                decay: float = _RECENCY_DECAY) -> float:
        """Generative Agents recency: exponential decay per hour since last touched.
        1.0 the moment it's seen, decaying toward 0 as it goes untouched."""
        hours = ((now or time.time()) - self.last_seen) / 3600.0
        return decay ** max(hours, 0.0)

    def when(self) -> str:
        return datetime.fromtimestamp(self.created, timezone.utc).date().isoformat()


# --------------------------------------------------------------------------- #
# Store — load / append / dedup. A thin wrapper over a JSONL file.
# --------------------------------------------------------------------------- #

class MemoryStore:
    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        # Archived (consolidated-away) facts live beside the active store — MemGPT's
        # recall/archival tier: out of the working set, never lost.
        self.archive_path = path.parent / (path.stem + "_archive.jsonl")
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
            embed: Optional[Embedder] = None,
            importance: float = 5.0) -> Optional[Fact]:
        """Add a fact, or bump an existing duplicate instead of duplicating.

        With an embedder, dedup is SEMANTIC (cosine): "has a brother, Sam" and
        "her brother is named Sam" collapse even though their words differ — which
        keyword jaccard misses. Without one, falls back to keyword dedup.
        `importance` is the 1-10 poignancy (Generative Agents); a resurfacing fact
        keeps the higher of the old and new score. Returns the Fact if stored/
        updated, None if it was empty.
        """
        text = text.strip()
        if not text:
            return None
        importance = max(1.0, min(10.0, importance))
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
            dupe.importance = max(dupe.importance, importance)
            if dupe.embedding is None and vec is not None:
                dupe.embedding = vec       # backfill a vector on an old fact
            self._save()
            return dupe
        fact = Fact(text=text, kind=kind, embedding=vec, importance=importance)
        self.facts.append(fact)
        self._save()
        return fact

    def add_many(self, items: Iterable[tuple],
                 embed: Optional[Embedder] = None) -> list[Fact]:
        """items: (text, kind) or (text, kind, importance) tuples."""
        out = []
        for it in items:
            text, kind = it[0], it[1]
            imp = it[2] if len(it) > 2 else 5.0
            out.append(self.add(text, kind, embed, imp))
        return [f for f in out if f is not None]

    def archive(self, facts: list[Fact]) -> None:
        """Move facts out of the active working set into the archive file."""
        drop = {id(f) for f in facts}
        if not drop:
            return
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self.archive_path.open("a") as fh:
            for f in facts:
                fh.write(json.dumps(asdict(f), ensure_ascii=False) + "\n")
        self.facts = [f for f in self.facts if id(f) not in drop]
        self._save()

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
    """Return up to k relevant facts, rendered for the system prompt's memory slot.

    Ranking is the Generative Agents retrieval function (see rank_facts): recency +
    importance + relevance, each min-max normalized then summed. Insights (the
    Keeper's own synthesized understanding) are rendered under a separate heading so
    they are never returned as something the person said. Falls back to keyword
    overlap when there is no embedder or the cue can't be embedded. Empty string if
    the store is empty.
    """
    if not store.facts:
        return ""
    top = rank_facts(store.facts, cue, k=k, now=now, embed=embed)
    if not top:
        return ""
    kept = [f for f in top if f.kind != "insight"]
    insights = [f for f in top if f.kind == "insight"]
    blocks = []
    if kept:
        blocks.append("\n".join(f"- {f.text}" for f in kept))
    if insights:
        # Rendered as the Keeper's conclusions, not as reported facts.
        blocks.append("what you've come to understand (your own read, not their words):\n"
                      + "\n".join(f"- {f.text}" for f in insights))
    return "\n\n".join(blocks)


def rank_facts(facts: list[Fact], cue: str = "", k: int = 5,
               now: Optional[float] = None,
               embed: Optional[Embedder] = None) -> list[Fact]:
    """The Generative Agents retrieval function, returning the top-k Facts.

    score(f) = w_rec·recency(f) + w_imp·importance(f) + w_rel·relevance(f, cue)

    Each of the three components is min-max normalized to [0,1] across the candidate
    set before weighting (as in Park et al. 2023). `relevance` is HYBRID: dense
    (embedding cosine) and sparse (BM25) signals fused with Reciprocal Rank Fusion —
    the combo that dominates retrieval benchmarks. With no embedder it's BM25-only;
    with no cue it's dropped and ranking rests on recency + importance (the
    proactive/drift case).
    """
    if not facts:
        return []
    now = now or time.time()

    recency = [f.recency(now) for f in facts]
    importance = [f.importance / 10.0 for f in facts]

    # HYBRID relevance: a DENSE signal (embedding cosine) and a SPARSE signal (BM25),
    # fused with Reciprocal Rank Fusion — the combo that dominates retrieval benchmarks
    # (lexical catches exact terms/names dense recall misses; dense catches paraphrase).
    dense: Optional[list[float]] = None
    if embed is not None and cue.strip() and any(f.embedding for f in facts):
        try:
            cue_vec = embed([cue])[0]
        except Exception:  # noqa: BLE001 - degrade to sparse-only
            cue_vec = None
        if cue_vec is not None:
            dense = [max(0.0, _cosine(cue_vec, f.embedding)) if f.embedding
                     else 0.0 for f in facts]
    sparse: Optional[list[float]] = None
    if cue.strip():
        cue_toks = _tokens(cue)
        if cue_toks:
            sparse = _bm25(cue_toks, facts)

    # Fuse only signals that actually discriminate — a flat (all-equal) ranker carries
    # no information and would just dilute a strong one through RRF.
    signals = [s for s in (dense, sparse) if s is not None and max(s) - min(s) > 1e-12]
    if len(signals) == 2:
        relevance = _rrf(dense, sparse)          # hybrid
    elif len(signals) == 1:
        relevance = signals[0]
    else:
        relevance = dense if dense is not None else sparse   # both flat / none

    rec_n = _minmax(recency)
    imp_n = _minmax(importance)
    rel_n = _minmax(relevance) if relevance is not None else [0.0] * len(facts)

    scored = []
    for i, f in enumerate(facts):
        score = (_W_RECENCY * rec_n[i]
                 + _W_IMPORTANCE * imp_n[i]
                 + _W_RELEVANCE * rel_n[i])
        scored.append((score, i, f))
    # tie-break by original order (stable) via the index.
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    return [f for _, _, f in scored[:k]]


def _minmax(xs: list[float]) -> list[float]:
    """Min-max normalize to [0,1]; all-equal collapses to 1.0 (neutral)."""
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [1.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _bm25(cue_toks: set[str], facts: list[Fact],
          k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 lexical score of each fact against the query terms. Pure Python,
    computed over the (small) fact corpus each call — the sparse half of hybrid."""
    docs = [_token_list(f.text) for f in facts]
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = []
    for d in docs:
        dl = len(d) or 1
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in cue_toks:
            if q not in tf:
                continue
            idf = math.log(1 + (n - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5))
            freq = tf[q]
            s += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def _rrf(*rankings: list[float], k: int = 60) -> list[float]:
    """Reciprocal Rank Fusion of several score lists over the same items: convert each
    to ranks (best = 1) and sum 1/(k+rank). Robust fusion that ignores raw score
    scales — the standard way to combine dense + sparse retrieval."""
    n = len(rankings[0])
    fused = [0.0] * n
    for scores in rankings:
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        for rank, i in enumerate(order, start=1):
            fused[i] += 1.0 / (k + rank)
    return fused


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


# --------------------------------------------------------------------------- #
# Distill — turn a finished conversation into durable facts.
# --------------------------------------------------------------------------- #

_DISTILL_SYSTEM = """You extract durable facts about a person from a conversation, \
for a companion's long-term memory. Output ONLY facts worth keeping for months: \
who they are, their situation, relationships, ongoing struggles, things they care \
about. Skip small talk, passing moods, and anything about the companion.

Write each fact as ONE short third-person statement of the fact itself — the thing \
that is true, not the act of saying it. Write "Has a brother, Sam; not spoken since \
spring," never "Mentioned a brother."

Prefix each line with [kind|importance]: kind is one of identity, state, event, \
preference; importance is 1-10 for how poignant/significant this is to the person's \
life — 1 is mundane (their coffee order), 10 is life-defining (a loss, a diagnosis, \
a core relationship). Example: [identity|8] Has a brother, Sam; not spoken since spring.

One fact per line, no bullets, no numbering. If there is nothing worth keeping, \
output exactly NONE."""

_KNOWN_KINDS = {"identity", "state", "event", "preference", "insight"}
# One leading [...] tag, whatever it holds.
_LEADING_TAG_RE = re.compile(r"^\s*\[([^\]]*)\]\s*")
_NUM_RE = re.compile(r"^\d{1,2}(?:\.\d+)?$")


def _parse_prefix(line: str) -> tuple[str, float, str]:
    """Peel EVERY leading [..] tag off a distilled line and return (kind, importance,
    text). Robust to what the model actually emits: unknown tags ([emotion],
    [activity]) are dropped, several tags ([activity] [state]) are all stripped, and
    an importance number is read from a [kind|8] or a bare [8]. Nothing bracketed is
    ever left to leak into the stored fact. Defaults: kind 'event', importance 5."""
    kind: Optional[str] = None
    importance: Optional[float] = None
    while True:
        m = _LEADING_TAG_RE.match(line)
        if not m:
            break
        line = line[m.end():]
        for part in re.split(r"[|,/]", m.group(1)):
            part = part.strip().lower()
            if part in _KNOWN_KINDS and kind is None:
                kind = part
            elif _NUM_RE.match(part) and importance is None:
                importance = float(part)
    return kind or "event", importance if importance is not None else 5.0, line.strip()


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

    parsed: list[tuple[str, str, float]] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if not line:
            continue
        kind, imp, text = _parse_prefix(line)
        if text:
            parsed.append((text, kind, imp))

    if store is None:
        return [Fact(text=t, kind=k, importance=i) for t, k, i in parsed]
    return store.add_many(parsed, embed=embed)


# --------------------------------------------------------------------------- #
# Consolidation — MemGPT memory pressure: compress the low-value tail.
# --------------------------------------------------------------------------- #

_CONSOLIDATE_SYSTEM = """You compress a companion's old, minor memories about a \
person to keep its long-term store tight. You are given a small cluster of related \
facts. Merge them into ONE concise third-person fact that preserves what still \
matters and drops the trivial. Keep it faithful — do not invent anything not present. \
Output only the single merged fact, no preamble."""


def consolidate(store: MemoryStore, generate: Generator, *,
                budget: int = _CONSOLIDATE_BUDGET,
                embed: Optional[Embedder] = None,
                now: Optional[float] = None) -> list[Fact]:
    """Compress the least-valuable, mutually-similar facts when over the budget.

    MemGPT (Packer et al. 2023) manages memory pressure by evicting older content to
    external storage under recursive summarization. Here: once the active store
    exceeds `budget`, take the lowest-value tail (low importance + old + rarely
    recurred), cluster it by similarity, and summarize each cluster of >=2 into one
    fact — archiving the originals (never deleting). Insights (the Keeper's own
    conclusions) are left untouched. Returns the consolidated Facts created.
    """
    if len(store.facts) <= budget:
        return []
    now = now or time.time()
    active = [f for f in store.facts if f.kind != "insight"]

    def value(f: Fact) -> float:
        return f.importance / 10.0 + f.recency(now) + min(f.mentions, 5) / 5.0

    overflow = len(store.facts) - budget
    tail = sorted(active, key=value)[: overflow + 4]   # a small margin to find pairs

    made: list[Fact] = []
    for cluster in _cluster(tail):
        if len(cluster) < 2:
            continue
        rendered = "\n".join(f"- {f.text}" for f in cluster)
        summary = (generate(_CONSOLIDATE_SYSTEM, rendered) or "").strip()
        if not summary:
            continue
        imp = max(f.importance for f in cluster)
        seen = sum(f.mentions for f in cluster)
        store.archive(cluster)                          # originals -> archive tier
        merged = store.add(summary, kind=cluster[0].kind, embed=embed, importance=imp)
        if merged is not None:
            merged.mentions = max(merged.mentions, seen)
            store._save()
            made.append(merged)
    return made


def _cluster(facts: list[Fact]) -> list[list[Fact]]:
    """Greedy single-link clustering of facts by similarity — embedding cosine when
    vectors are present, else keyword jaccard. Coherent merges only."""
    clusters: list[list[Fact]] = []
    for f in facts:
        placed = False
        for cl in clusters:
            head = cl[0]
            if f.embedding and head.embedding:
                sim = _cosine(f.embedding, head.embedding)
                near = sim >= 0.55
            else:
                near = _jaccard(_tokens(f.text), _tokens(head.text)) >= 0.34
            if near:
                cl.append(f)
                placed = True
                break
        if not placed:
            clusters.append([f])
    return clusters


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _tokens(text: str) -> set[str]:
    return set(_token_list(text))


def _token_list(text: str) -> list[str]:
    """Content tokens WITH repeats (BM25 needs term frequencies)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


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
