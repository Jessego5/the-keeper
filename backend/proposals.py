"""
These are the things the Keeper wants to do in the world, and may not do without a yes.

Everything the Keeper could do before this was reversible or private: read a file,
search the web, hold a reminder, write in its own journal. MCP stays read-only by
default (tools._is_mutating) for exactly that reason. Muse and ChatGPT's dots go
further, they send the email and book the table, and the difference between an agent
that does that and one that is dangerous is a single step: someone saying yes to the
exact thing before it happens.

So a tool that changes the world is never handed to the model as itself. A server in
mcp.json lists such tools under "propose", and the model sees each one by its usual
name and schema, but calling it only RECORDS a proposal: the action, and the exact
arguments the model wrote. The person is shown those arguments verbatim and approves
or declines. Approving runs that stored payload and nothing else.

Two rules carry the safety, and both are easy to erode, so they are written here:

  1. What the person approves is rendered from the stored arguments by code
     (render()), never summarised by the model. A summary is a second piece of model
     output that can disagree with the first, and the one that runs is the first.
     Prompt injection lives in that gap: a fetched page asks for "a quick note to
     Sam" and the arguments carry a different address.
  2. A proposal runs at most once. It is marked approved and saved BEFORE the tool
     is called, and one found approved but unfinished at startup is failed, not
     retried: a restart must never be how an email goes out twice.

Plain JSONL, rewritten whole like the reminder store. A person sees a handful a day.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
PROPOSALS_PATH = STORE_DIR / "proposals.jsonl"

# Long enough to answer tomorrow morning; short enough that a stale draft does not
# go out a week later into a situation it no longer fits.
TTL_S = 24 * 3600.0

PENDING, APPROVED, DONE = "pending", "approved", "done"
DECLINED, EXPIRED, FAILED = "declined", "expired", "failed"
SETTLED = frozenset({DONE, DECLINED, EXPIRED, FAILED})

# What MCPManager.call answers with instead of raising. A tool that failed must
# not be reported to the person as done.
_FAILURE_PREFIXES = ("(tool error", "(tool timed out", "(no such tool")


@dataclass
class Proposal:
    action: str                 # the qualified tool name, e.g. gmail__send_message
    args: dict                  # exactly what will be passed if approved
    origin: str = "chat"        # chat | goal:<id> | background
    id: str = field(default_factory=lambda: "p-" + uuid.uuid4().hex[:12])
    status: str = PENDING
    created: float = field(default_factory=time.time)
    expires_at: float = 0.0
    decided_at: Optional[float] = None
    decided_via: Optional[str] = None   # web | telegram
    result: str = ""

    def public(self) -> dict:
        """What a surface needs to draw it: the rendering, not just the args."""
        return {**asdict(self), "rendered": render(self.action, self.args)}


def render(action: str, args: dict, schema: Optional[dict] = None) -> str:
    """The proposal as the person will read it: every argument, verbatim, in the
    order the tool declares them. Deterministic on purpose (see the module
    docstring). Nothing is truncated here; a surface too small for it says so
    rather than showing less."""
    server, _, tool = action.partition("__")
    title = tool or action
    lines = [f"{title}" + (f" ({server})" if tool else "")]
    order = list(((schema or {}).get("properties") or {}).keys())
    keys = order + [k for k in args if k not in order]
    for key in keys:
        if key not in args:
            continue
        value = args[key]
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        if "\n" in value or len(value) > 70:
            body = "\n".join("    " + ln for ln in value.splitlines() or [""])
            lines.append(f"  {key}:\n{body}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def looks_failed(result: str) -> bool:
    return (result or "").lstrip().startswith(_FAILURE_PREFIXES)


class ProposalStore:
    def __init__(self, path: Path = PROPOSALS_PATH):
        self.path = path
        self.items: list[Proposal] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for raw in self.path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                self.items.append(Proposal(**json.loads(raw)))
            except (json.JSONDecodeError, TypeError):
                continue
        # Rule 2: approved but never finished means the process died mid-call. The
        # action may or may not have happened; running it again is the one answer
        # that is certainly wrong.
        torn = [p for p in self.items if p.status == APPROVED]
        for p in torn:
            p.status = FAILED
            p.result = ("interrupted by a restart before it was confirmed; not run "
                        "again, check whether it happened")
        if torn:
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(
            json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in self.items))

    def add(self, action: str, args: dict, origin: str = "chat",
            now: Optional[float] = None) -> Proposal:
        now = now or time.time()
        p = Proposal(action=action, args=dict(args), origin=origin, created=now,
                     expires_at=now + TTL_S)
        self.items.append(p)
        self._save()
        return p

    def get(self, pid: str, now: Optional[float] = None) -> Optional[Proposal]:
        self.expire_stale(now)
        return next((p for p in self.items if p.id == pid), None)

    def pending(self, now: Optional[float] = None) -> list[Proposal]:
        self.expire_stale(now)
        return [p for p in self.items if p.status == PENDING]

    def recent(self, limit: int = 20, now: Optional[float] = None) -> list[Proposal]:
        self.expire_stale(now)
        return self.items[-limit:]

    def expire_stale(self, now: Optional[float] = None) -> list[Proposal]:
        """Retire anything left unanswered past its TTL. Lazy: checked whenever the
        queue is read, which is every time anyone could act on it."""
        now = now or time.time()
        stale = [p for p in self.items
                 if p.status == PENDING and p.expires_at and p.expires_at <= now]
        for p in stale:
            p.status = EXPIRED
            p.decided_at = now
        if stale:
            self._save()
        return stale

    def decline(self, p: Proposal, via: str, now: Optional[float] = None) -> None:
        p.status, p.decided_via = DECLINED, via
        p.decided_at = now or time.time()
        self._save()

    def mark_approved(self, p: Proposal, via: str,
                      now: Optional[float] = None) -> None:
        """Saved before the action runs. See rule 2."""
        p.status, p.decided_via = APPROVED, via
        p.decided_at = now or time.time()
        self._save()

    def finish(self, p: Proposal, result: str) -> None:
        p.result = (result or "").strip()
        p.status = FAILED if looks_failed(p.result) else DONE
        self._save()


# --------------------------------------------------------------------------- #
# The model's side: proposable tools that record instead of run.
# --------------------------------------------------------------------------- #

@dataclass
class Action:
    """A tool that may only run on a yes. `name` is what the model calls.

    `fixed` holds what the config supplies (which account to send from). It is
    merged into the stored arguments, so the approval shows it: it is part of
    what will run, and rule 1 is about everything that will run."""
    name: str
    description: str
    parameters: dict
    fixed: dict = field(default_factory=dict)


class Proposer:
    """Shaped like tools.MCPManager (openai_tools() + async call()), so it sits in
    compose.tool_reply's provider list beside the others. Its calls never touch the
    world: they add a proposal and hand it to `notify`, which asks the person."""

    def __init__(self, store: ProposalStore, actions: list[Action],
                 notify: Callable[[Proposal], Awaitable[None]],
                 origin: str = "chat"):
        self.store = store
        self.actions = {a.name: a for a in actions}
        self._notify = notify
        self.origin = origin

    def openai_tools(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": a.name,
                "description": ("PROPOSES this, it does not run until the person "
                                "approves exactly what you pass. Write the real, "
                                "final content. " + a.description)[:1024],
                "parameters": a.parameters,
            },
        } for a in self.actions.values()]

    @property
    def has_tools(self) -> bool:
        return bool(self.actions)

    async def call(self, name: str, args: dict) -> str:
        action = self.actions.get(name)
        if action is None:
            return f"(no such tool: {name})"
        missing = [k for k in (action.parameters.get("required") or [])
                   if k not in (args or {})]
        if missing:
            # Caught here, not at approval: asking someone to approve a payload the
            # tool will reject wastes the one decision they were asked to make.
            return f"(not proposed: missing {', '.join(missing)})"
        p = self.store.add(name, {**(args or {}), **action.fixed},
                           origin=self.origin)
        await self._notify(p)
        return (f"proposed, NOT done (id {p.id}). It waits for their yes, and they "
                f"see exactly what you wrote. Tell them what you drafted and that "
                f"it is waiting on them. Do not say it is done.")
