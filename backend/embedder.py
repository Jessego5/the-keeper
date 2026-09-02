"""embedder.py — turn text into vectors for semantic memory.

A thin, injectable wrapper over an embedding model, mirroring compose.py's
generator pattern: production passes a real embedder; tests pass None and memory
falls back to keyword recall. Keeps the whole system runnable and testable with no
key, while giving real semantic retrieval when a key is present.

Default: OpenAI text-embedding-3-small at 1024 dims (a good quality/size trade;
the model supports the `dimensions` param to shrink from its native 1536).
"""

from __future__ import annotations

import os
import retry
from typing import Callable, List, Optional

# texts -> one vector per text
Embedder = Callable[[List[str]], List[List[float]]]

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1024


def openai_embedder(model: str = DEFAULT_MODEL, dim: int = DEFAULT_DIM) -> Embedder:
    """Build an embedder backed by the OpenAI embeddings API.

    Requires OPENAI_API_KEY. Raises RuntimeError if unavailable so callers can
    fall back to keyword recall.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("openai SDK not installed") from e

    client = OpenAI()

    def embed(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        def once() -> List[List[float]]:
            resp = client.embeddings.create(model=model, input=texts,
                                            dimensions=dim)
            # API preserves input order
            return [d.embedding for d in resp.data]

        # Worth retrying harder than a chat call: a failure here makes MemoryStore
        # write the fact with embedding=None, and nothing ever comes back to embed
        # it later. A transient 429 would otherwise cost that memory permanently.
        return retry.with_retry(once, what="embed")

    return embed


def make_embedder() -> Optional[Embedder]:
    """The real OpenAI embedder if a key is set, else None (keyword fallback)."""
    try:
        return openai_embedder()
    except RuntimeError:
        return None
