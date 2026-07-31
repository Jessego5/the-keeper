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

import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Optional

CONFIG_PATH = Path(__file__).resolve().parent / "mcp.json"

# OpenAI tool names must match ^[a-zA-Z0-9_-]+$, so we join server+tool with "__".
_QUALIFY = "__"
_NAME_OK = re.compile(r"[^a-zA-Z0-9_-]")


def _safe(name: str) -> str:
    return _NAME_OK.sub("_", name)


class MCPManager:
    """Owns the live MCP sessions for the app's lifetime.

    Usage:
        mgr = MCPManager.from_config()
        await mgr.connect()
        tools = mgr.openai_tools()          # -> list of tool defs (possibly [])
        text = await mgr.call(name, args)   # execute one tool call
        await mgr.aclose()
    """

    def __init__(self, specs: list[dict]):
        self.specs = specs
        self._stack = AsyncExitStack()
        self._tools: list[dict] = []
        self._route: dict[str, tuple[Any, str]] = {}  # qualified -> (session, real)
        self.connected = False

    @classmethod
    def from_config(cls, path: Path = CONFIG_PATH) -> "MCPManager":
        """Read backend/mcp.json if present; else a manager with no servers.

        Format:  {"servers": [{"name":"files","command":"npx",
                              "args":["-y","@modelcontextprotocol/server-filesystem","/dir"]}]}
        """
        if not path.exists():
            return cls([])
        try:
            data = json.loads(path.read_text())
            return cls(list(data.get("servers", [])))
        except (json.JSONDecodeError, OSError):
            return cls([])

    async def connect(self) -> None:
        """Launch each configured server and register its tools. Tolerant: a
        server that fails to start is skipped, not fatal."""
        if not self.specs:
            return
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            print("[mcp] SDK not installed; tools disabled", flush=True)
            return

        for spec in self.specs:
            name = _safe(spec.get("name", "srv"))
            try:
                params = StdioServerParameters(
                    command=spec["command"], args=spec.get("args", []),
                    env=spec.get("env"))
                read, write = await self._stack.enter_async_context(
                    stdio_client(params))
                session = await self._stack.enter_async_context(
                    ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
            except Exception as exc:  # noqa: BLE001
                print(f"[mcp] server {name!r} failed: {exc}", flush=True)
                continue

            for t in listed.tools:
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
            print(f"[mcp] {name}: {len(listed.tools)} tools", flush=True)

        self.connected = True

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
            result = await session.call_tool(real, arguments or {})
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
