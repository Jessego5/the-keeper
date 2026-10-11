"""
This is what the Keeper watches, so it can speak about the world.

Until this existed, the proactive loop could only ever hand back the person's own
past: Gate 3 recalled with an empty cue, so every unbidden line was a rearrangement
of what they had already said. This gives it something to notice.

A source must genuinely PUSH. RSS qualifies: entries carry an id and a publish
time, so "new since I last looked" is a fact rather than a re-asked question. A web
search does not, running the same query again returns the same results, and a
companion announcing that as a discovery is inventing an event. This codebase has
an eval forbidding exactly that (`test_never_invents_events`, written after it
claimed "your brother reached out"), and a source that fabricates novelty walks
straight into it.

Nothing here decides whether to speak. This layer only answers "what is new?",
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
from datetime import datetime
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
    because: str = ""            # the fact it turned on, shown in the trace

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


# Lines that identify a record rather than describe it. Skipping them is what
# makes a commit read as "bound the history deque" instead of "commit a1b2c3".
_META_LINE = re.compile(r"^(commit|author|date|authordate|commitdate|merge|refs|"
                        r"from|to|subject|id|uid)\b\s*:?\s", re.I)


# The label a server puts in front of the one line worth reading, and the escape
# it prints instead of a real newline. mcp-server-git emits a whole commit message
# as ONE quoted, escaped string: Message: "subject\n\nbody...". Left alone, the
# Keeper would say 'Message: "subject\n\nbody' out loud.
_LABEL = re.compile(r"^(message|subject|summary|title|description)\s*:\s*", re.I)
_ESCAPED_NEWLINE = "\\n"


def _split_record_line(line: str) -> tuple[str, str]:
    """(title, remainder) for the one line worth reading.

    The title is that line as a person would say it: no field label, no wrapping
    quotes, cut at the first paragraph break. The REMAINDER is everything after
    that break, and it must be kept, in a commit it is the explanation under the
    subject, which is the richest thing in the record to match against memory.
    """
    body = _LABEL.sub("", line.strip())
    head, _, tail = body.partition(_ESCAPED_NEWLINE)
    title = head.strip().strip("\"'").strip()
    tail = tail.replace(_ESCAPED_NEWLINE, "\n").strip().strip("\"'").strip()
    return title, tail


def _block_title(block: str) -> tuple[str, str]:
    """(title, rest) for one record: the first line that reads as prose, and
    everything else in the block."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return "", ""
    raw = next((ln for ln in lines if not _META_LINE.match(ln)), lines[0])
    title, tail = _split_record_line(raw)
    rest = [ln for ln in lines if ln is not raw]
    if tail:
        rest.append(tail)
    # A record that was ALL metadata (its only line was a label) still needs
    # something to be called; the uncleaned line is better than nothing.
    return (title or raw.strip())[:200], "\n".join(rest)


def parse_tool_output(text: str, source: str = "tool",
                      max_items: int = 8) -> list[SourceItem]:
    """Turn one MCP tool's text result into items the same gates can judge.

    This is what lets the Keeper notice something other than a feed: a commit that
    landed, a file that changed in the folder it watches, the day's calendar. The
    servers are already configured and already running for the /chat path, so
    nothing here is a new integration, only a second reader of the same tools.

    The PUSH rule from this module's docstring still binds, and it is a property of
    the TOOL, not of this parser. `git_log` and a directory listing qualify: each
    entry carries its own identity, so "new since I last looked" is a fact. A web
    search does not, and pointing this at one would have the Keeper announce old
    results as discoveries, which an eval already forbids. That choice is made in
    mcp.json, so it is worth saying plainly: only watch tools whose entries are
    stable and identifiable.

    Output shape is not standardised across servers. Records separated by blank
    lines are tried first (git log, calendars), with a wholly indented block read as
    a continuation of the one above it, which is how git prints a commit message.
    Failing that, each line is its own record (directory listings). The key is the
    record's own text, so the same commit read twice is never delivered twice.
    """
    body = (text or "").strip()
    if not body or body.startswith("("):     # "(tool error: ...)", "(no output)"
        return []

    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    blocks: list[str] = []
    for raw in paragraphs:
        indented = all(ln.startswith((" ", "\t"))
                       for ln in raw.splitlines() if ln.strip())
        if indented and blocks:
            blocks[-1] = f"{blocks[-1]}\n{raw.strip()}"
        else:
            blocks.append(raw.strip())
    # Only when the output has no blank line anywhere is each line its own record.
    # Counting the MERGED blocks instead would misread a single commit, whose
    # message is one indented paragraph, as a listing, and hand back its hash,
    # its author and its date as three separate things that happened.
    if len(paragraphs) == 1:
        blocks = [ln.strip() for ln in body.splitlines() if ln.strip()]

    items: list[SourceItem] = []
    for block in blocks[:max_items]:
        title, rest = _block_title(block)
        if title:
            items.append(SourceItem(source=source, title=title, body=rest[:1000],
                                    key=delivery_key("", block[:500])))
    return items


# --------------------------------------------------------------------------- #
# Servers whose output the generic reader gets wrong.
# --------------------------------------------------------------------------- #
#
# The google workspace server answers in its own decorated text: a count line, an
# emoji header, numbered records, then usage tips for the model. parse_tool_output
# would read the header and the tips as things that happened. These two read the
# record shapes that server actually prints (gmail/gmail_tools.py and
# gcalendar/calendar_tools.py in workspace-mcp), and key each record by the
# server's own id, which is what makes "new since I last looked" a fact.

_GMAIL_RECORD = re.compile(r"^\s*\d+\.\s+Message ID:\s*(\S+)\s*$", re.M)
_GMAIL_FIELD = re.compile(r"^\s*(Subject|From|Date|Web Link):\s*(.*)$", re.M)


def parse_gmail_search(text: str, source: str = "inbox",
                       max_items: int = 10) -> list[SourceItem]:
    """search_gmail_messages, called with include_headers=true. One item per
    message: its subject and sender, keyed by message id, so a thread that grows
    a reply arrives as the reply rather than as the whole thread again."""
    body = text or ""
    starts = list(_GMAIL_RECORD.finditer(body))
    items: list[SourceItem] = []
    for i, m in enumerate(starts[:max_items]):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
        fields = dict(_GMAIL_FIELD.findall(body[m.end():end]))
        subject = (fields.get("Subject") or "(no subject)").strip()
        sender = (fields.get("From") or "").strip()
        title = f"{subject} (from {sender})" if sender else subject
        items.append(SourceItem(
            source=source, title=title[:200],
            url=(fields.get("Web Link") or "").strip(),
            body=f"From: {sender}\nDate: {fields.get('Date', '').strip()}",
            key=delivery_key(f"gmail:{m.group(1)}", "")))
    return items


_EVENT = re.compile(
    r'^- "(?P<title>.*?)" \(Starts: (?P<start>[^,]+), Ends: (?P<end>[^)]+)\)'
    r'(?P<rest>.*?)ID: (?P<id>\S+) \| Link: (?P<link>\S+)', re.M | re.S)


def parse_calendar_events(text: str, source: str = "calendar",
                          max_items: int = 10) -> list[SourceItem]:
    """get_events. Keyed by event id AND start time: a meeting that MOVES is news,
    and keying by id alone would have decided it had already been told."""
    items: list[SourceItem] = []
    for m in list(_EVENT.finditer(text or ""))[:max_items]:
        start = m.group("start").strip()
        items.append(SourceItem(
            source=source, title=f"{m.group('title').strip()} at {start}"[:200],
            url=m.group("link").strip(),
            body=f"Starts: {start}\nEnds: {m.group('end').strip()}",
            key=delivery_key(f"cal:{m.group('id')}@{start}", "")))
    return items


PARSERS = {"gmail": parse_gmail_search, "calendar": parse_calendar_events}


def parse_watch_output(text: str, source: str, max_items: int = 8,
                       parser: Optional[str] = None) -> list[SourceItem]:
    """A watched tool's result, read by the parser its watch names, or by the
    generic reader when it names none."""
    if (text or "").lstrip().startswith("("):   # "(tool error: ...)", "(no such tool"
        return []
    fn = PARSERS.get(parser or "")
    if fn is not None:
        return fn(text, source=source, max_items=max_items)
    return parse_tool_output(text, source=source, max_items=max_items)


_NOW_REF = re.compile(r"\$\{NOW(?:([+-])(\d+)([mhd]))?\}")
_UNIT_S = {"m": 60, "h": 3600, "d": 86400}


def expand_time(obj, now: Optional[float] = None):
    """${NOW}, ${NOW+1d}, ${NOW-6h} inside a watch's arguments, as RFC 3339 in
    local time. Expanded at each poll, not at load: a calendar watch asks about
    the next day from NOW, and the day it was configured has long passed."""
    now = now or time.time()
    if isinstance(obj, str):
        def sub(m):
            offset = 0
            if m.group(1):
                offset = int(m.group(2)) * _UNIT_S[m.group(3)]
                offset = offset if m.group(1) == "+" else -offset
            return datetime.fromtimestamp(now + offset).astimezone().isoformat(
                timespec="seconds")
        return _NOW_REF.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: expand_time(v, now) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_time(x, now) for x in obj]
    return obj


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
