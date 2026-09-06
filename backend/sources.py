"""sources.py — what the Keeper watches, so it can speak about the world.

Until this existed, the proactive loop could only ever hand back the person's own
past: Gate 3 recalled with an empty cue, so every unbidden line was a rearrangement
of what they had already said. This gives it something to notice.

A source must genuinely PUSH. RSS qualifies: entries carry an id and a publish
time, so "new since I last looked" is a fact rather than a re-asked question. A web
search does not — running the same query again returns the same results, and a
companion announcing that as a discovery is inventing an event. This codebase has
an eval forbidding exactly that (`test_never_invents_events`, written after it
claimed "your brother reached out"), and a source that fabricates novelty walks
straight into it.

Nothing here decides whether to speak. This layer only answers "what is new?" —
relevance is judged against the person's memory in relevance.py, and the decision
to interrupt stays in proactive.tick().
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
SEEN_PATH = STORE_DIR / "sources_seen.jsonl"

# Feeds put the summary in any of these, wrapped in CDATA as often as not.
_BODY_TAGS = ("description", "summary", "{http://www.w3.org/2005/Atom}summary",
              "{http://purl.org/rss/1.0/modules/content/}encoded")


@dataclass
class SourceItem:
    """One thing that arrived. `key` is what makes "already told them" decidable."""

    source: str
    title: str
    url: str = ""
    body: str = ""
    published: Optional[float] = None
    key: str = ""
    # Filled in by the relevance pass, not by parsing.
    relevance: float = 0.0
    because: str = ""            # the fact it turned on — shown in the trace

    def __post_init__(self) -> None:
        if not self.key:
            self.key = delivery_key(self.url, self.title)

    def as_dict(self) -> dict:
        return asdict(self)


def delivery_key(url: str, title: str) -> str:
    """A stable id for "this item", so it is never delivered twice.

    Prefers the URL: the same story often gets its title tweaked after publishing,
    and a title-keyed item would then arrive a second time as though it were new.
    Falls back to the title when there is no link, truncated so a runaway body
    cannot change the key of an otherwise identical item.
    """
    basis = (url or "").strip() or (title or "").strip()[:500]
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _text(el: Optional[ET.Element]) -> str:
    if el is None or el.text is None:
        return ""
    return html.unescape(re.sub(r"<[^>]+>", " ", el.text)).strip()


def _parse_date(raw: str) -> Optional[float]:
    from email.utils import parsedate_to_datetime
    for parse in (parsedate_to_datetime,):
        try:
            return parse(raw).timestamp()
        except Exception:  # noqa: BLE001 - a feed with a bad date is still usable
            pass
    return None


def parse_feed(xml_text: str, source: str = "feed") -> list[SourceItem]:
    """RSS or Atom -> items. Pure and offline, so the tests need no network."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    out: list[SourceItem] = []
    entries = root.iter("item") if root.find(".//item") is not None else \
        root.iter("{http://www.w3.org/2005/Atom}entry")
    for e in entries:
        title = _text(e.find("title")) or _text(
            e.find("{http://www.w3.org/2005/Atom}title"))
        if not title:
            continue
        link_el = e.find("link")
        url = _text(link_el)
        if not url:                                   # Atom puts it in an attribute
            atom_link = e.find("{http://www.w3.org/2005/Atom}link")
            url = (atom_link.get("href") or "") if atom_link is not None else ""
        body = ""
        for tag in _BODY_TAGS:
            body = _text(e.find(tag))
            if body:
                break
        pub = ""
        for tag in ("pubDate", "{http://www.w3.org/2005/Atom}published",
                    "{http://www.w3.org/2005/Atom}updated"):
            pub = _text(e.find(tag))
            if pub:
                break
        out.append(SourceItem(source=source, title=title, url=url,
                              body=body[:1000], published=_parse_date(pub)))
    return out


class SeenStore:
    """Delivery keys already sent, so nothing arrives twice.

    Separate from the fact store on purpose: this is delivery bookkeeping, not
    something the Keeper knows about the person.
    """

    def __init__(self, path: Path = SEEN_PATH):
        self.path = path
        self.keys: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    self.keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue

    def seen(self, item: SourceItem) -> bool:
        return item.key in self.keys

    def mark(self, item: SourceItem) -> None:
        if item.key in self.keys:
            return
        self.keys.add(item.key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": item.key, "title": item.title[:120],
                                 "at": time.time()}, ensure_ascii=False) + "\n")

    def unseen(self, items: Iterable[SourceItem]) -> list[SourceItem]:
        return [i for i in items if not self.seen(i)]
