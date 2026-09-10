"""
This decides whether a thing from the world matters to THIS person.

The gate that makes a feed reader into a companion. An item is worth saying only
because of something the Keeper already knows: a gallery listing matters because
the store says they paint, not because it is art news.

It runs in two stages, mirroring recall's own retrieve-then-rerank. The cheap
stage ranks the person's facts against the item's text and keeps the item only if
something clears memory's relevance floor, killing most of them for the cost of an
embedding. The judge stage then spends one model call on the survivors, and names
the fact it turned on as it goes, so the trace can show WHY the Keeper thought
this was worth an interruption.

TWO thresholds, not one, and that is the design. "Worth mentioning" and "worth
interrupting a person for" are different bars: an item over MENTION changes what
is said when the loop had already decided to speak, and only an item over
INTERRUPT is allowed to make it speak at all.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import memory

Generator = Callable[[str, str], str]

# Over this, the item is worth talking ABOUT if the Keeper speaks anyway.
MENTION = 0.45
# Over this, the item is worth SPEAKING UP for. Deliberately far higher: this is
# the only path by which the outside world can raise how often someone is
# interrupted, so mild relevance must not buy it.
INTERRUPT = 0.75

_JUDGE_SYSTEM = """You decide whether a piece of news is worth telling a specific \
person about, given what is known about them.

You are told what is kept about the person, and one item. Answer with a score from \
0 to 10 for how much THIS ITEM matters TO THIS PERSON, then a dash, then the single \
thing you know about them that makes it matter, copied word for word from the list, \
or NONE if nothing does.

  8 - Is a painter.
  0 - NONE

Score on connection to the person, never on how interesting the item is in itself. \
A wonderful story that touches nothing they care about is a 0. Be strict: most \
items are 0 to 2. Reserve 8 or more for something they would want interrupting for.

Format: a number, a dash, then the fact or NONE. Nothing else."""

# The dashes in this class are DATA, not punctuation: the prompt above asks for
# "a number, a dash, then the fact", and a model writes that dash as a hyphen, an
# en-dash or an em-dash as it pleases. They stay even where prose dashes do not,
# because this has to read what the model actually emits. (A sweep that replaced
# em-dashes across the repo turned this into [-,,:] and silently stopped the
# separator matching.)
_SCORE_RE = re.compile(r"^\s*(\d{1,2})(?:\s*[-–—:]\s*(.*))?", re.S)


def parse_verdict(raw: str) -> tuple[float, str]:
    """(score 0..1, the fact it turned on). Unreadable answers score 0.

    Lenient about shape for the reason every other verdict parser here is: the
    judge is a cheap model writing free text, and this session has watched one
    produce three spellings of its own one-word answer.
    """
    m = _SCORE_RE.match((raw or "").strip())
    if not m:
        return 0.0, ""
    score = min(10, max(0, int(m.group(1)))) / 10.0
    because = (m.group(2) or "").strip().strip('"')
    if because.upper().startswith("NONE"):
        because = ""
    return score, because


def score_item(item, store: memory.MemoryStore, *,
               generate: Optional[Generator] = None,
               embed: Optional[memory.Embedder] = None,
               k: int = 4) -> tuple[float, str]:
    """Score one SourceItem against what is kept. Returns (0..1, because)."""
    facts = [f for f in store.facts if f.active] if store is not None else []
    if not facts:
        return 0.0, ""                     # nothing known -> nothing can matter

    cue = f"{item.title} {item.body}".strip()
    # Stage 1 RANKS, it does not reject. It used to reject, with recall's gate: but
    # that floor was measured for short chat cues against short facts, and a feed
    # title plus summary is long enough to dilute the cosine below it. The judge
    # then never got asked, and a store holding "Has started painting again" scored
    # an article about someone's dense paintings at 0.0.
    #
    # The gate exists to save judge calls when candidates are many. A scan is capped
    # at eight items, so it was saving almost nothing and costing the accuracy that
    # is the whole point. As in the supersede path: the judge is the real gate, the
    # cosine only decides which facts it is shown.
    near = memory.rank_facts(facts, cue=cue, k=k, embed=embed, gate=False)
    if not near:
        return 0.0, ""
    if generate is None:
        return 0.0, ""                     # no judge -> never interrupt on a guess

    kept = "\n".join(f"- {f.text}" for f in near)
    user = f"What is kept about them:\n{kept}\n\nItem:\n{cue[:600]}"
    try:
        raw = generate(_JUDGE_SYSTEM, user) or ""
    except Exception as exc:  # noqa: BLE001 - silence is the safe failure here
        print(f"[relevance] judge failed, treating as irrelevant "
              f"({type(exc).__name__}: {exc})", flush=True)
        return 0.0, ""
    score, because = parse_verdict(raw)
    if score < MENTION:
        return score, ""       # nothing is said, so nothing is the reason
    return score, attribute(because, near)


def attribute(because: str, facts) -> str:
    """The kept fact behind a score, guaranteed to be one the store actually holds.

    The judge is asked to copy a fact word for word and does not reliably: the
    example in this prompt is itself a plausible fact, and watched live the judge
    copied THAT onto every item in a scan, birds and filmmakers included, naming a
    line the store did not hold. Replacing the example with an obvious placeholder
    fixed the copying but cost most of the gate's separation, so the example stays
    and the check lives here instead.

    A paraphrase resolves to the line it paraphrases ("Is a painter." for "Is a
    painter who stopped in March."). Anything else falls back to the fact that
    ranked highest, which is the one the judge was most prominently shown. That is
    a real fact rather than the judge's invention, which is the property that
    matters: `because` exists to show the person why it thought this mattered, and
    a fact they never gave it is a plausible lie.
    """
    ranked = [f.text if hasattr(f, "text") else str(f) for f in facts]
    norm = (because or "").strip().strip(".").lower()
    for text in ranked:
        other = text.strip().strip(".").lower()
        if norm and (norm == other or norm in other or other in norm):
            return text
    return ranked[0] if ranked else ""


def worth_interrupting(score: float) -> bool:
    return score >= INTERRUPT


def worth_mentioning(score: float) -> bool:
    return score >= MENTION
