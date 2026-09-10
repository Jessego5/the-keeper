"""
This repairs and de-duplicates the fact store.

It makes two fixes over an existing memory_store/facts.jsonl. De-tagging strips
the leaked bracket tags an older distill parser left behind, turning "[emotion]
Feels stuck." back into "Feels stuck." and recovering the real kind from the first
known tag. De-duplicating then merges semantic duplicates with the embedder,
keeping the highest importance, the summed mention count and the earliest created
time.

It backs the original up to facts.jsonl.bak first, and is idempotent, so running
it twice is safe. Run it with .venv/bin/python scripts/clean_memory.py, optionally
naming a path to a different facts.jsonl.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

# Load backend/.env so OPENAI_API_KEY is available for the embedder (the app does
# this in compose.py at import; standalone scripts must do it themselves).
try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND / ".env")
except ImportError:
    pass

import embedder  # noqa: E402
import memory  # noqa: E402

_DEDUP = 0.86   # same cosine threshold the store uses for semantic dedup


def clean(path: Path) -> None:
    if not path.exists():
        print(f"no store at {path}")
        return
    backup = path.with_suffix(".jsonl.bak")
    shutil.copy2(path, backup)

    store = memory.MemoryStore(path)
    before = len(store.facts)
    embed = embedder.make_embedder()          # real embedder (needs OPENAI key)
    if embed is None:
        print("no embedder (set OPENAI_API_KEY), de-tagging only, no semantic merge.")

    cleaned: list[memory.Fact] = []
    for f in store.facts:
        kind, imp, text = memory._parse_prefix(f.text)   # strip any leaked tags
        if not text:
            continue
        # keep an explicit, meaningful kind if the record already had one
        if f.kind in memory._KNOWN_KINDS and f.kind != "event":
            kind = f.kind
        imp = max(imp, float(f.importance))
        vec = None
        if embed is not None:
            try:
                vec = embed([text])[0]
            except Exception:  # noqa: BLE001
                vec = None
        new = memory.Fact(text=text, kind=kind, importance=imp,
                          created=f.created, last_seen=f.last_seen,
                          mentions=f.mentions, embedding=vec or f.embedding)

        dup = None
        for g in cleaned:
            if new.embedding and g.embedding:
                if memory._cosine(new.embedding, g.embedding) >= _DEDUP:
                    dup = g
                    break
        if dup is not None:
            dup.importance = max(dup.importance, new.importance)
            dup.mentions += new.mentions
            dup.created = min(dup.created, new.created)
            dup.last_seen = max(dup.last_seen, new.last_seen)
        else:
            cleaned.append(new)

    store.facts = cleaned
    store._save()
    print(f"cleaned {path.name}: {before} -> {len(cleaned)} facts "
          f"(backup: {backup.name})")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else memory.STORE_PATH
    clean(target)
