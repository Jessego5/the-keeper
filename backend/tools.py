"""tools.py — the Keeper's tools, via MCP. Passive loop only.

The Keeper reaches for a tool only when the person addresses it (the /chat path).
The proactive loop stays sealed — it never gets tools and never touches the world
unbidden. That boundary is deliberate: tools make the Keeper *useful when asked*
without breaking the "no window on the world" identity that governs its own voice.

This connects the standalone server to MCP servers YOU configure (backend/mcp.json)
— it does not and cannot borrow any host's MCP connectors. Servers are launched
over stdio; their tools are exposed to the model in OpenAI's tool-call format and
executed back through MCP.

Everything here is async, matching FastAPI and the async OpenAI client, so no
thread-bridging is needed. If mcp.json is absent or the SDK is missing, the manager
simply exposes zero tools and the Keeper behaves exactly as before.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent / "mcp.json"


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key) or default)
    except ValueError:
        return default


# Every call to somebody else's process gets a deadline. Without one a server that
# hangs (a fetch against a host that never answers, a git server on a huge repo)
# blocks whatever awaited it: the person's turn, or a proactive tick, with nothing
# to break the wait. The handshake gets its own, longer budget because `npx`/`uvx`
# may be downloading the server package on first run.
CALL_TIMEOUT_S = _env_float("KEEPER_MCP_TIMEOUT_S", 20.0)
STARTUP_TIMEOUT_S = _env_float("KEEPER_MCP_STARTUP_TIMEOUT_S", 60.0)

# When set, a server that fails to start or is missing a tool it declared under
# "required" aborts startup instead of being reported. Off by default: the Keeper
# is meant to survive a missing tool. On in CI, where silence is the bug.
STRICT = bool(os.environ.get("KEEPER_MCP_STRICT"))

# Where the Keeper's files live. mcp.json can write ${KEEPER_ROOT} instead of an
# absolute path, so one config works on the host AND in the container, where the
# app sits at /app rather than the developer's home directory. Absolute paths in
# mcp.json still work untouched.
KEEPER_ROOT = os.environ.get(
    "KEEPER_ROOT", str(Path(__file__).resolve().parent.parent))


def _expand(args: list) -> list:
    """Substitute ${KEEPER_ROOT} in server args."""
    return [a.replace("${KEEPER_ROOT}", KEEPER_ROOT) if isinstance(a, str) else a
            for a in args]


def _expand_obj(obj):
    """The same substitution, anywhere inside a watch's arguments. A watched tool
    is usually pointed at a path too (git_log wants its repo), and that path has
    to survive the move into the container exactly as a server's args do."""
    if isinstance(obj, str):
        return obj.replace("${KEEPER_ROOT}", KEEPER_ROOT)
    if isinstance(obj, list):
        return [_expand_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_obj(v) for k, v in obj.items()}
    return obj

# OpenAI tool names must match ^[a-zA-Z0-9_-]+$, so we join server+tool with "__".
_QUALIFY = "__"
_NAME_OK = re.compile(r"[^a-zA-Z0-9_-]")

# Tool names hinting at mutation. Blocked by default so the Keeper can READ what
# is yours but never change it. A server spec may opt back in with
# "read_only": false (or narrow it with "tools"/"deny" allow/deny lists).
_MUTATING = (
    "write", "edit", "delete", "remove", "move", "rename", "create", "mkdir",
    "unlink", "patch", "append", "modify", "update", "truncate", "drop", "put",
)


def _safe(name: str) -> str:
    return _NAME_OK.sub("_", name)


def _is_mutating(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _MUTATING)


class MCPManager:
    """Owns the live MCP sessions for the app's lifetime.

    Usage:
        mgr = MCPManager.from_config()
        await mgr.connect()
        tools = mgr.openai_tools()          # -> list of tool defs (possibly [])
        text = await mgr.call(name, args)   # execute one tool call
        await mgr.aclose()
    """

    def __init__(self, specs: list[dict], watches: list[dict] | None = None):
        self.specs = specs
        # Tools the proactive loop polls for something new. Read here because
        # mcp.json is where the servers are already described; see sources.py for
        # why only a genuinely pushing tool belongs in this list.
        self.watches = list(watches or [])
        self._stack = AsyncExitStack()
        self._tools: list[dict] = []
        self._route: dict[str, tuple[Any, str]] = {}  # qualified -> (session, real)
        self.connected = False
        self.health: list[dict] = []    # one entry per configured server

    @classmethod
    def from_config(cls, path: Path = CONFIG_PATH) -> "MCPManager":
        """Read backend/mcp.json if present; else a manager with no servers.

        Format:  {"servers": [{"name":"files","command":"npx",
                              "args":["-y","@modelcontextprotocol/server-filesystem","/dir"]}],
                  "watch":   [{"name":"repo","tool":"git__git_log",
                               "args":{"max_count":5}}]}
        """
        if not path.exists():
            return cls([])
        try:
            data = json.loads(path.read_text())
            return cls(list(data.get("servers", [])),
                       _expand_obj(list(data.get("watch", []))))
        except (json.JSONDecodeError, OSError):
            return cls([])

    def _record(self, name: str, status: str, **fields: Any) -> None:
        self.health.append({"name": name, "status": status, **fields})

    def problems(self) -> list[dict]:
        """Servers that failed to start, or came up without a tool they declared."""
        return [h for h in self.health if h["status"] != "ok"]

    def summary(self) -> str:
        """One line per unhealthy server, for a log line or the state surface."""
        out = []
        for h in self.problems():
            if h["status"] == "failed":
                out.append(f"{h['name']}: failed ({h.get('error', 'unknown')})")
            else:
                # Name the ones OUR filter removed. Saying only "missing" would
                # send you hunting the upstream server for a tool it does offer.
                filtered = set(h.get("filtered") or ())
                names = ", ".join(
                    f"{m} (blocked by this config)" if m in filtered else m
                    for m in h.get("missing", []))
                out.append(f"{h['name']}: missing {names}")
        return "; ".join(out)

    def _enforce(self) -> None:
        if STRICT and self.problems():
            raise RuntimeError(f"MCP servers unhealthy: {self.summary()}")

    async def connect(self) -> None:
        """Launch each configured server and register its tools. Tolerant: a
        server that fails to start is skipped, not fatal (unless KEEPER_MCP_STRICT).
        Either way it is recorded in `health` rather than only printed."""
        if not self.specs:
            return
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            print("[mcp] SDK not installed; tools disabled", flush=True)
            for spec in self.specs:
                self._record(_safe(spec.get("name", "srv")), "failed",
                             error="mcp SDK not installed")
            self._enforce()
            return

        for spec in self.specs:
            name = _safe(spec.get("name", "srv"))
            read_only = spec.get("read_only", True)
            allow = set(spec.get("tools") or [])   # if set, ONLY these (by real name)
            deny = set(spec.get("deny") or [])
            required = list(spec.get("required") or [])
            try:
                params = StdioServerParameters(
                    command=spec["command"], args=_expand(spec.get("args", [])),
                    env=spec.get("env"))
                read, write = await self._stack.enter_async_context(
                    stdio_client(params))
                session = await self._stack.enter_async_context(
                    ClientSession(read, write))
                # Deadline on the handshake: a server that starts but never answers
                # would otherwise hang app startup forever.
                await asyncio.wait_for(session.initialize(), STARTUP_TIMEOUT_S)
                listed = await asyncio.wait_for(session.list_tools(),
                                                STARTUP_TIMEOUT_S)
            except asyncio.TimeoutError:
                msg = f"no response within {STARTUP_TIMEOUT_S:.0f}s"
                print(f"[mcp] server {name!r} failed: {msg}", flush=True)
                self._record(name, "failed", error=msg)
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[mcp] server {name!r} failed: {exc}", flush=True)
                self._record(name, "failed", error=str(exc))
                continue

            offered = {t.name for t in listed.tools}
            kept_real: set[str] = set()
            blocked = 0
            for t in listed.tools:
                # Filter: explicit allow/deny first, then the read-only default.
                if allow and t.name not in allow:
                    continue
                if t.name in deny:
                    continue
                if read_only and _is_mutating(t.name):
                    blocked += 1
                    continue
                qualified = f"{name}{_QUALIFY}{_safe(t.name)}"
                # SDK versions differ: input_schema (snake) vs inputSchema (camel).
                schema = (getattr(t, "input_schema", None)
                          or getattr(t, "inputSchema", None)
                          or {"type": "object", "properties": {}})
                self._tools.append({
                    "type": "function",
                    "function": {
                        "name": qualified,
                        "description": (t.description or "")[:1024],
                        "parameters": schema,
                    },
                })
                self._route[qualified] = (session, t.name)
                kept_real.add(t.name)
            note = f" ({blocked} mutating blocked, read-only)" if blocked else ""
            print(f"[mcp] {name}: {len(kept_real)} tools{note}", flush=True)

            # What this server PROMISED to provide, checked against what it did.
            # A tool that vanishes (renamed upstream, filtered by our own config,
            # server started but half-broken) used to leave no trace at all: the
            # Keeper would simply never reach for it, and the missing capability
            # looked like a choice rather than a fault.
            missing = [r for r in required if r not in kept_real]
            if missing:
                detail = ", ".join(
                    f"{m} (offered, but blocked by this server's own filter)"
                    if m in offered else m for m in missing)
                print(f"[mcp] {name}: MISSING required {detail}", flush=True)
            self._record(name, "degraded" if missing else "ok",
                         tools=len(kept_real), blocked=blocked, missing=missing,
                         # the confusing subset: present upstream, removed by us
                         filtered=[m for m in missing if m in offered])

        self.connected = True
        self._enforce()

    def openai_tools(self) -> list[dict]:
        return self._tools

    @property
    def has_tools(self) -> bool:
        return bool(self._tools)

    async def call(self, qualified_name: str, arguments: dict) -> str:
        """Execute a tool call and return its result as text for the model."""
        route = self._route.get(qualified_name)
        if route is None:
            return f"(no such tool: {qualified_name})"
        session, real = route
        try:
            result = await asyncio.wait_for(
                session.call_tool(real, arguments or {}), CALL_TIMEOUT_S)
        except asyncio.TimeoutError:
            # Answer the model rather than hanging the turn. It can say so, or
            # try something else, which is what a person would do.
            return f"(tool timed out after {CALL_TIMEOUT_S:.0f}s: {qualified_name})"
        except Exception as exc:  # noqa: BLE001
            return f"(tool error: {exc})"
        # Flatten MCP content blocks to text.
        parts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else "(no output)"

    async def aclose(self) -> None:
        await self._stack.aclose()
