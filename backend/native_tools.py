"""native_tools.py — the Keeper's own action tools (not MCP).

Gives the Keeper things it can DO when asked: hold a reminder and give it back
when it's due, list what it's holding, mark one done. Shaped exactly like
tools.MCPManager (openai_tools() + async call()) so compose.tool_reply can take
both providers side by side and route each tool call to the right one.

Times: the model converts natural language ("tomorrow 9am") into an ISO 8601
timestamp using the current time in its prompt — no date-parsing dependency, and
the time reasoning is the model's, which is itself a small agentic step.
"""

from __future__ import annotations

from datetime import datetime

import reminders as reminders_mod


class NativeTools:
    def __init__(self, store: reminders_mod.ReminderStore):
        self.store = store
        self._defs = [
            {
                "type": "function",
                "function": {
                    "name": "remind_me",
                    "description": "Store a reminder to give back to the person at "
                                   "a future time. Convert their phrasing into an "
                                   "ISO 8601 datetime using the current time in "
                                   "your context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string",
                                     "description": "what to remind them of"},
                            "due_iso": {"type": "string",
                                        "description": "ISO 8601, e.g. "
                                                       "2026-08-03T09:00:00"},
                        },
                        "required": ["text", "due_iso"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_reminders",
                    "description": "List the reminders the Keeper is currently "
                                   "holding (not yet done).",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "complete_reminder",
                    "description": "Mark a reminder done, by a word from its text "
                                   "or its id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string",
                                    "description": "a word from the reminder, or "
                                                   "its id"},
                        },
                        "required": ["key"],
                    },
                },
            },
        ]

    def openai_tools(self) -> list[dict]:
        return self._defs

    @property
    def has_tools(self) -> bool:
        return True

    async def call(self, name: str, args: dict) -> str:
        if name == "remind_me":
            due = _parse_iso(args.get("due_iso", ""))
            if due is None:
                return "(could not read the time — need an ISO datetime)"
            r = self.store.add(args.get("text", "").strip(), due)
            when = datetime.fromtimestamp(due).strftime("%a %b %d, %H:%M")
            return f"kept: \"{r.text}\" — will return it {when} (id {r.id})"
        if name == "list_reminders":
            pend = self.store.pending()
            if not pend:
                return "holding nothing right now."
            return "\n".join(
                f"- {r.text}  ({datetime.fromtimestamp(r.due_at).strftime('%b %d %H:%M')}, id {r.id})"
                for r in pend)
        if name == "complete_reminder":
            r = self.store.complete(args.get("key", ""))
            return f"done: \"{r.text}\"" if r else "no matching reminder to complete."
        return f"(no such tool: {name})"


def _parse_iso(s: str):
    s = (s or "").strip().replace("Z", "")
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None
