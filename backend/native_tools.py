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
from typing import Callable, Optional

import planner as planner_mod
import reminders as reminders_mod
import tasks as tasks_mod


class NativeTools:
    def __init__(self, store: reminders_mod.ReminderStore,
                 goals: Optional[tasks_mod.GoalStore] = None,
                 planner_generate: Optional[Callable] = None):
        self.store = store
        self.goals = goals
        self._plan_gen = planner_generate      # used to decompose a new goal
        self._defs = [
            {
                "type": "function",
                "function": {
                    "name": "remind_me",
                    "description": "Store a reminder to give back to the person at "
                                   "a future time, optionally recurring. Convert "
                                   "their phrasing into an ISO 8601 datetime using "
                                   "the current time in your context; for a "
                                   "recurring one, set repeat and let due_iso be the "
                                   "first occurrence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string",
                                     "description": "what to remind them of"},
                            "due_iso": {"type": "string",
                                        "description": "ISO 8601 of the (first) time, "
                                                       "e.g. 2026-08-03T09:00:00"},
                            "repeat": {"type": "string",
                                       "description": "optional recurrence: 'daily', "
                                                      "'weekly', 'weekdays', or "
                                                      "'every N minutes/hours/days/"
                                                      "weeks'. Omit for one-time."},
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
            {
                "type": "function",
                "function": {
                    "name": "set_goal",
                    "description": "Take on a GOAL to help the person move toward "
                                   "over time — something they want to work toward "
                                   "or get unstuck on (not a one-time reminder). You "
                                   "will break it into small steps and help them "
                                   "through it across days. Use their own words for "
                                   "the goal.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string",
                                      "description": "the goal, in their words "
                                                     "(e.g. 'get back to painting')"},
                        },
                        "required": ["title"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_goals",
                    "description": "List the goals the Keeper is helping with and "
                                   "how far along each is.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "complete_goal",
                    "description": "Mark a goal done (finished or set down), by a "
                                   "word from its title or its id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string",
                                    "description": "a word from the goal, or its id"},
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
            r = self.store.add(args.get("text", "").strip(), due,
                               repeat=args.get("repeat"))
            when = datetime.fromtimestamp(due).strftime("%a %b %d, %H:%M")
            cadence = f", then {r.repeat}" if r.repeat else ""
            return f"kept: \"{r.text}\" — will return it {when}{cadence} (id {r.id})"
        if name == "list_reminders":
            pend = self.store.pending()
            if not pend:
                return "holding nothing right now."
            return "\n".join(
                f"- {r.text}  ({datetime.fromtimestamp(r.due_at).strftime('%b %d %H:%M')}"
                f"{', ' + r.repeat if r.repeat else ''}, id {r.id})"
                for r in pend)
        if name == "complete_reminder":
            r = self.store.complete(args.get("key", ""))
            return f"done: \"{r.text}\"" if r else "no matching reminder to complete."

        if name == "set_goal":
            if self.goals is None or self._plan_gen is None:
                return "(cannot take on goals right now)"
            title = args.get("title", "").strip()
            if not title:
                return "(need a goal to take on)"
            steps = planner_mod.plan(title, self._plan_gen)
            if not steps:
                steps = ["Take the first small step toward it."]
            g = self.goals.add(title, steps)
            plan = "\n".join(f"  {i+1}. {s.text}" for i, s in enumerate(g.steps))
            return (f"taken on: \"{g.title}\" (id {g.id}) — the plan I'll help with:\n"
                    f"{plan}")
        if name == "list_goals":
            if self.goals is None:
                return "not holding any goals."
            act = self.goals.active()
            if not act:
                return "not holding any goals right now."
            lines = []
            for g in act:
                done, total = g.progress()
                nxt = g.next_step()
                lines.append(f"- {g.title} ({done}/{total} done" +
                             (f", next: {nxt.text}" if nxt else "") +
                             f", id {g.id})")
            return "\n".join(lines)
        if name == "complete_goal":
            if self.goals is None:
                return "not holding any goals."
            g = self.goals.complete(args.get("key", ""))
            return f"set down: \"{g.title}\"" if g else "no matching goal."
        return f"(no such tool: {name})"


def _parse_iso(s: str):
    s = (s or "").strip().replace("Z", "")
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None
