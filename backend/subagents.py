"""subagents.py — the Keeper's specialists. Multi-agent, kept legible.

For a task too involved for one tool call, the Keeper delegates to a SUB-AGENT: a
focused specialist that runs its OWN tool loop (ReAct) with a narrow role and only the
tools it needs, then returns a result. A tiny supervisor routes a task to the right
specialist. This is the reference agent's supervisor + sub-agent pattern, as three clear roles
instead of a fleet framework.

    researcher — looks things up on the web and synthesizes an answer  (search, fetch)
    archivist  — digs through the person's own files/notes/history       (files, git, time, journal)
    scribe     — drafts text: a message, a note, a plan                  (journal)

Each sub-agent is still "the Keeper's" — its output is an internal work product the
main Keeper then speaks from, so it need not be in full voice. The model is injected,
so routing and running are testable offline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

import compose

Generator = Callable[[str, str], str]


@dataclass(frozen=True)
class SubAgentProfile:
    name: str
    blurb: str              # one line, used by the router to pick
    role: str               # system-prompt fragment: who this specialist is
    servers: tuple          # MCP server names it may use (by prefix)
    use_native: bool        # whether it gets the native tools (journal, etc.)
    max_rounds: int = 6


PROFILES: dict[str, SubAgentProfile] = {
    "researcher": SubAgentProfile(
        name="researcher",
        blurb="looks things up on the open web and synthesizes an answer",
        role="You are the Keeper's researcher. Search and read what you need, then "
             "give ONE clear, accurate answer grounded in what you found. Note your "
             "sources briefly. Do not speculate beyond the evidence.",
        servers=("search", "fetch"), use_native=False),
    "archivist": SubAgentProfile(
        name="archivist",
        blurb="digs through the person's own files, notes, journal, and project history",
        role="You are the Keeper's archivist. Look through their files, journal, and "
             "project history to find what is asked, then report plainly what you "
             "found and where. Do not invent anything not in the materials.",
        servers=("files", "git", "time"), use_native=True),
    "scribe": SubAgentProfile(
        name="scribe",
        blurb="drafts text — a message, a note, a short plan",
        role="You are the Keeper's scribe. Draft exactly the text asked for, plainly "
             "and warmly, ready to use. Keep it in the journal only if asked.",
        servers=(), use_native=True),
    "analyst": SubAgentProfile(
        name="analyst",
        blurb="works things out by running code — calculations, dates, data crunching",
        role="You are the Keeper's analyst. Work the answer out by WRITING AND RUNNING "
             "Python with run_python (print the result), then report the answer in one "
             "or two plain sentences. Reach for code whenever a real computation, date "
             "math, or data transformation would settle it precisely.",
        servers=(), use_native=True),
}

DEFAULT_PROFILE = "researcher"


@dataclass
class SubAgentResult:
    profile: str
    task: str
    result: str


class _FilteredMCP:
    """A view of the MCP manager exposing only tools from certain servers, so a
    specialist sees just its own tools. Calls route to the real manager."""

    def __init__(self, mcp, allow_servers):
        self._mcp = mcp
        self._allow = set(allow_servers)

    def openai_tools(self) -> list:
        return [t for t in self._mcp.openai_tools()
                if t["function"]["name"].split("__")[0] in self._allow]

    @property
    def has_tools(self) -> bool:
        return bool(self.openai_tools())

    async def call(self, name: str, args: dict) -> str:
        return await self._mcp.call(name, args)


_ROUTE_SYSTEM = """You dispatch a task to ONE of the Keeper's specialists. Given the \
task, reply with just the specialist's name — nothing else. The specialists are:
"""


def route(task: str, generate: Generator) -> SubAgentProfile:
    """Pick the specialist that best fits `task`. LLM router with a safe default."""
    blurbs = "\n".join(f"- {p.name}: {p.blurb}" for p in PROFILES.values())
    try:
        out = (generate(_ROUTE_SYSTEM + blurbs, task) or "").strip().lower()
    except Exception:  # noqa: BLE001
        out = ""
    for name in PROFILES:
        if name in out:
            return PROFILES[name]
    return PROFILES[DEFAULT_PROFILE]


async def run(profile: SubAgentProfile, task: str, *,
              mcp=None, native=None, model: str = "gpt-4o") -> SubAgentResult:
    """Run one specialist's own tool loop on `task` and return its result."""
    providers: list = []
    if native is not None and profile.use_native:
        providers.append(native)
    if mcp is not None and profile.servers:
        view = _FilteredMCP(mcp, profile.servers)
        if view.has_tools:
            providers.append(view)
    system = (f"{profile.role}\n\nYou are working on behalf of the Keeper. Do the "
              f"task, then answer in one short paragraph — no chit-chat, no preamble.")
    result = await compose.tool_reply(
        system, task, providers=providers, model=model,
        max_rounds=profile.max_rounds)
    return SubAgentResult(profile=profile.name, task=task,
                          result=(result or "").strip())


async def delegate(task: str, generate: Generator, *,
                   mcp=None, native=None, model: str = "gpt-4o") -> SubAgentResult:
    """Route `task` to ONE specialist and run it. The single-worker primitive."""
    profile = route(task, generate)
    return await run(profile, task, mcp=mcp, native=native, model=model)


# --------------------------------------------------------------------------- #
# Orchestrator-workers: decompose -> fan out in parallel -> synthesize.
# --------------------------------------------------------------------------- #

MAX_WORKERS = 4

_DECOMPOSE_SYSTEM = """You are the Keeper's lead agent. Break a task into the FEWEST \
INDEPENDENT subtasks that together fully answer it — 1 to 4 — each doable by one \
specialist working alone, in parallel (so they must not depend on each other's output). \
If the task is already single and simple, return exactly ONE line (the task itself). \
Output one subtask per line, no numbering, no preamble."""

_SYNTH_SYSTEM = """You are the Keeper's lead agent. Your specialists each worked one \
part of a task; their findings follow. Combine them into ONE clear, coherent answer to \
the original task. Keep every concrete fact, drop repetition, resolve any conflicts \
sensibly, and invent nothing. Answer plainly — the Keeper will re-voice it."""


@dataclass
class OrchestrationResult:
    task: str
    result: str
    workers: list = field(default_factory=list)   # list[SubAgentResult]

    def who(self) -> str:
        seen, order = set(), []
        for w in self.workers:
            if w.profile not in seen:
                seen.add(w.profile)
                order.append(w.profile)
        return "+".join(order) or "keeper"

    def trace(self) -> str:
        return " | ".join(f"{w.profile}: {w.task[:40]}" for w in self.workers)


def _decompose(task: str, generate: Generator) -> list[str]:
    raw = (generate(_DECOMPOSE_SYSTEM, task) or "").strip()
    subs = [ln.strip().lstrip("-*0123456789.) ").strip()
            for ln in raw.splitlines() if ln.strip()]
    return subs[:MAX_WORKERS] or [task]


async def orchestrate(task: str, generate: Generator, *,
                      mcp=None, native=None, model: str = "gpt-4o") -> OrchestrationResult:
    """The deep multi-agent path: decompose `task` into subtasks, run each on the right
    specialist CONCURRENTLY, then synthesize their findings into one answer. Degrades to
    a single specialist (no synthesis) when the task doesn't split."""
    subtasks = _decompose(task, generate)

    if len(subtasks) == 1:
        res = await delegate(subtasks[0], generate, mcp=mcp, native=native, model=model)
        return OrchestrationResult(task=task, result=res.result, workers=[res])

    # Route each subtask (cheap, sequential), then run the workers in PARALLEL.
    assignments = [(route(st, generate), st) for st in subtasks]
    settled = await asyncio.gather(
        *[run(prof, st, mcp=mcp, native=native, model=model)
          for prof, st in assignments],
        return_exceptions=True)
    workers = [r for r in settled
               if isinstance(r, SubAgentResult) and r.result.strip()]
    if not workers:
        return OrchestrationResult(task=task, result="", workers=[])

    combined = "\n\n".join(f"[{w.profile}] {w.task}\n{w.result}" for w in workers)
    synthesis = (generate(_SYNTH_SYSTEM, f"Task: {task}\n\nFindings:\n{combined}")
                 or "").strip()
    return OrchestrationResult(
        task=task, result=synthesis or combined, workers=workers)
