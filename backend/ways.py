"""
These are the ways the person likes things done, learned from them, applied to work.

The fact store already keeps preferences ("prefers short emails"), and recall hands
them to a conversation when the message happens to cue them. Two things it could not
do. It could not bring them to WORK: the specialists that draft a goal's email or a
scheduled job's summary were given the task and nothing about the person, so every
draft started from scratch. And it could not learn from what the person did to its
work: a draft they rewrote before approving taught it nothing, because nothing was
watching the difference.

A way is narrower than a fact and attached to a kind of task: about "emails to Sam",
how "sign off as J, two sentences". It is learned from one of two places:

    said   they told it, in general ("from now on...", "always..."): keep_way
    edit   they changed one of its drafts before approving it, and the change said
           something about how they like that kind of thing done, rather than
           correcting a one-off fact like a date

and applied wherever work is done for them: the chat turn, a goal step, a job, a
background task. Every way records where it was learned, shows on the dashboard,
and can be forgotten; a learned habit nobody can see is a habit nobody can correct.

This is the self-improving loop Hermes Agent builds its skills around, kept small:
no generated code, no procedure files, one sentence per way, and a person who can
read every one of them.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import tasks as tasks_mod

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
WAYS_PATH = STORE_DIR / "ways.jsonl"

# How many ways a single piece of work is handed. More than this and they compete
# with the task itself for the model's attention.
MAX_APPLIED = 3


@dataclass
class Way:
    about: str                    # the kind of task: "emails to Sam", "any email"
    how: str                      # what they like: "sign off as J"
    source: str = "said"          # said | edit
    evidence: str = ""            # what taught it: their words, or the edit
    id: str = field(default_factory=lambda: "w-" + uuid.uuid4().hex[:8])
    created: float = field(default_factory=time.time)
    active: bool = True
    uses: int = 0
    replaced_by: Optional[str] = None

    def line(self) -> str:
        return f"{self.about}: {self.how}"


def _stems(text: str) -> set:
    return tasks_mod._stems(text)


def _overlap(a: set, b: set) -> int:
    """Shared words, counting two stems as one word when they share their first
    four letters. tasks._stems is crude on purpose, and it leaves "writing" as
    "writ" and "write" as "write": found in the live demo, where the way "any
    writing: sign with J" was not handed to "write a short note to Sam"."""
    def same(x: str, y: str) -> bool:
        return x == y or (len(x) >= 4 and len(y) >= 4 and x[:4] == y[:4])
    return sum(1 for x in a if any(same(x, y) for y in b))


class WayStore:
    def __init__(self, path: Path = WAYS_PATH):
        self.path = path
        self.items: list[Way] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for raw in self.path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                self.items.append(Way(**json.loads(raw)))
            except (json.JSONDecodeError, TypeError):
                continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(
            json.dumps(asdict(w), ensure_ascii=False) + "\n" for w in self.items))

    def active(self) -> list[Way]:
        return [w for w in self.items if w.active]

    def add(self, about: str, how: str, source: str = "said",
            evidence: str = "") -> Optional[Way]:
        """Keep a way. One about the same kind of task that says something different
        replaces the older one, which is closed, not deleted, like a superseded
        fact: "sign as J" then "sign as Jess" leaves one rule, and the history."""
        about, how = about.strip(), how.strip()
        if not about or not how:
            return None
        for old in self.active():
            if old.about.lower() == about.lower() and old.how.lower() == how.lower():
                return old                      # already known: nothing to learn
        way = Way(about=about, how=how, source=source, evidence=evidence.strip()[:400])
        for old in self.active():
            if _stems(old.about) == _stems(about) and _stems(old.how) & _stems(how):
                old.active, old.replaced_by = False, way.id
        self.items.append(way)
        self._save()
        return way

    def forget(self, key: str) -> Optional[Way]:
        key_low = key.lower().strip()
        if not key_low:
            return None
        for w in self.active():
            if w.id == key or key_low in w.line().lower():
                w.active = False
                self._save()
                return w
        return None

    def for_task(self, task: str, k: int = MAX_APPLIED) -> list[Way]:
        """The ways that bear on `task`. A way applies when the task shares a word
        with what it is ABOUT: "any email" applies to "email Sam about Thursday";
        "emails to Sam" applies there too and ranks higher, because it shares more.
        Matching on `about` alone, not `how`, is deliberate: "keep it short" says
        nothing about which tasks it governs."""
        cue = _stems(task)
        if not cue:
            return []
        scored = []
        for w in self.active():
            overlap = _overlap(cue, _stems(w.about))
            if overlap:
                scored.append((overlap, w.created, w))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [w for _, _, w in scored[:k]]

    def mark_used(self, used: list[Way]) -> None:
        if not used:
            return
        for w in used:
            w.uses += 1
        self._save()


def as_guidance(used: list[Way]) -> str:
    """How applied ways read to the model doing the work."""
    if not used:
        return ""
    lines = "\n".join(f"- {w.line()}" for w in used)
    return ("How they like this kind of thing done (learned from them; follow it "
            f"where it applies, and if it conflicts with this request, the request "
            f"wins):\n{lines}")


# --------------------------------------------------------------------------- #
# Learning from an edit.
# --------------------------------------------------------------------------- #

_EDIT_SYSTEM = """A person's assistant drafted something for them to approve. The \
person changed the draft before approving it. Decide whether the change shows a \
GENERAL way they like this kind of thing done (tone, length, how they sign off, \
what they always include or leave out, formatting), or only fixes a one-off fact \
for this one instance (a date, a time, a name, a typo, a number).

Ignore changes to where it goes (a path, an address, a time): those are one-off.

If it is general, answer exactly one line:
ABOUT: <the kind of thing and who it is for, as the content shows it> | HOW: <an \
instruction for the next draft, in the imperative, covering only what they changed>
If it is a one-off fix, answer exactly: NONE"""


def describe_edit(action: str, before: dict, after: dict) -> str:
    """The changed fields, before and after, for the learner to read."""
    out = [f"Action: {action}"]
    for key in after:
        if before.get(key) != after.get(key):
            out.append(f"Field {key}\nDRAFT:\n{before.get(key)}\nAPPROVED:\n{after[key]}")
    return "\n\n".join(out)


def learn_from_edit(action: str, before: dict, after: dict, generate,
                    store: WayStore) -> Optional[Way]:
    """Ask the model what an edit says about how they like things done, and keep
    the answer as a way. None for a one-off fix, which is most edits, and should
    be: a corrected date is not a habit."""
    if before == after:
        return None
    diff = describe_edit(action, before, after)
    try:
        verdict = (generate(_EDIT_SYSTEM, diff) or "").strip()
    except Exception:  # noqa: BLE001 - learning is a bonus; the approval stands
        return None
    return parse_learned(verdict, store, evidence=diff)


def parse_learned(verdict: str, store: WayStore, evidence: str = "") -> Optional[Way]:
    line = verdict.strip().splitlines()[0] if verdict.strip() else ""
    if not line or line.upper().startswith("NONE") or "|" not in line:
        return None
    about_part, _, how_part = line.partition("|")
    about = about_part.split(":", 1)[-1].strip().strip('"')
    how = how_part.split(":", 1)[-1].strip().strip('"')
    if not about_part.strip().upper().startswith("ABOUT") or not how:
        return None
    return store.add(about, how, source="edit", evidence=evidence)
