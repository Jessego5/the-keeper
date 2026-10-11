"""
These are the Keeper's own action tools, the ones that are not MCP.

Gives the Keeper things it can DO when asked: hold a reminder and give it back
when it's due, list what it's holding, mark one done. Shaped exactly like
tools.MCPManager (openai_tools() + async call()) so compose.tool_reply can take
both providers side by side and route each tool call to the right one.

Times: the model converts natural language ("tomorrow 9am") into an ISO 8601
timestamp using the current time in its prompt: no date-parsing dependency, and
the time reasoning is the model's, which is itself a small agentic step.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Callable, Optional

import a2a as a2a_mod
import dial as dial_mod
import jobs as jobs_mod
import journal as journal_mod
import planner as planner_mod
import reminders as reminders_mod
import sandbox as sandbox_mod
import subagents as subagents_mod
import tasks as tasks_mod
import ways as ways_mod


class NativeTools:
    def __init__(self, store: reminders_mod.ReminderStore,
                 goals: Optional[tasks_mod.GoalStore] = None,
                 planner_generate: Optional[Callable] = None,
                 journal: Optional[journal_mod.Journal] = None,
                 mcp=None, delegate_generate: Optional[Callable] = None,
                 allow_delegate: bool = True, spawner: Optional[Callable] = None,
                 dial: Optional[dial_mod.Dial] = None,
                 jobs: Optional[jobs_mod.JobStore] = None,
                 ways: Optional[ways_mod.WayStore] = None):
        self.store = store
        self.dial = dial                       # only the person's own turn gets it
        self.jobs = jobs                       # likewise: only they schedule work
        self.ways = ways                       # and only they teach it how
        self.goals = goals
        self._plan_gen = planner_generate      # used to decompose a new goal
        self.journal = journal
        self._mcp = mcp                        # for delegating to sub-agents
        self._delegate_gen = delegate_generate
        self._allow_delegate = allow_delegate
        self._spawner = spawner                # async: start a background task
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
                                   "over time, something they want to work toward "
                                   "or get unstuck on (not a one-time reminder). You "
                                   "will break it into small steps and help them "
                                   "through it across days. Use their own words for "
                                   "the goal. Only when they ASK for help moving "
                                   "toward something. Telling you a thing they lost "
                                   "or stopped is a fact to keep, not a goal to take "
                                   "on, do not infer the wish from the loss.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string",
                                      "description": "the goal, in their words "
                                                     "(e.g. 'help me get back to "
                                                     "running')"},
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
                    "name": "advance_goal",
                    "description": "Mark the CURRENT step of a goal done, when the "
                                   "person reports they've done it, or you did it for "
                                   "them. Moves the goal to its next step (or finishes "
                                   "it if that was the last).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string",
                                    "description": "a word from the goal, or its id"},
                            "note": {"type": "string",
                                     "description": "optional: what was done"},
                        },
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "complete_goal",
                    "description": "Mark a whole goal done (finished or set down), by "
                                   "a word from its title or its id.",
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
            {
                "type": "function",
                "function": {
                    "name": "keep_note",
                    "description": "Write a note into your journal, the one thing you "
                                   "can keep in writing. Use it when they ask you to "
                                   "hold a thought, or when you want to record "
                                   "something you found or worked out. Append-only; it "
                                   "never overwrites.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string",
                                     "description": "the note to keep"},
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_journal",
                    "description": "Read back the recent notes you've kept in your "
                                   "journal.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": "Run a short Python snippet to work something out, "
                                   "a calculation, a date/time computation, parsing or "
                                   "transforming data, anything a fixed tool can't do. "
                                   "PRINT the answer. Runs fenced (timeout, no lasting "
                                   "effects). Use for figuring, not for side effects.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string",
                                     "description": "Python; print() what you want back"},
                        },
                        "required": ["code"],
                    },
                },
            },
        ]
        if dial is not None:
            self._defs.append({
                "type": "function",
                "function": {
                    "name": "set_dial",
                    "description": "Set how often you speak on your own, ONLY when "
                                   "the person asks you to: 'off' (start nothing "
                                   "yourself; reminders and things they asked for "
                                   "still come back), 'quiet', 'usual', or 'eager'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "level": {"type": "string",
                                      "enum": list(dial_mod.LEVELS)},
                        },
                        "required": ["level"],
                    },
                },
            })
        if jobs is not None:
            self._defs += [
                {
                    "type": "function",
                    "function": {
                        "name": "schedule_job",
                        "description": "Run a TASK for them on a schedule, ONLY when "
                                       "they ask ('every weekday at 8 check my "
                                       "calendar', 'brief me each morning'). Unlike "
                                       "remind_me, which hands back words, a job does "
                                       "the work and brings the result. Use kind "
                                       "'brief' for a daily brief or morning "
                                       "summary; 'task' for anything else. Convert "
                                       "their time to ISO using the current time.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string",
                                         "description": "the work, described fully"},
                                "first_iso": {"type": "string",
                                              "description": "ISO 8601 of the first "
                                                             "run"},
                                "repeat": {"type": "string",
                                           "description": "'daily', 'weekdays', "
                                                          "'weekly', 'every N hours/"
                                                          "days'. Omit for once."},
                                "kind": {"type": "string",
                                         "enum": list(jobs_mod.KINDS)},
                            },
                            "required": ["task", "first_iso"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_jobs",
                        "description": "List the jobs you run for them on a "
                                       "schedule.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "cancel_job",
                        "description": "Stop a scheduled job, by a word from it or "
                                       "its id, when they ask.",
                        "parameters": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                            "required": ["key"],
                        },
                    },
                },
            ]
        if ways is not None:
            self._defs += [
                {
                    "type": "function",
                    "function": {
                        "name": "keep_way",
                        "description": "Remember HOW they like a kind of thing done, "
                                       "when they say it in general: 'from now on', "
                                       "'always', 'never', 'whenever you'. Not for an "
                                       "instruction about this one task only ('sign "
                                       "this one J' is not a way; 'sign my emails J' "
                                       "is).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "about": {"type": "string",
                                          "description": "the kind of task, e.g. "
                                                         "'emails to Sam', 'any "
                                                         "email', 'calendar events'"},
                                "how": {"type": "string",
                                        "description": "the way, as an instruction"},
                            },
                            "required": ["about", "how"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_ways",
                        "description": "List what you have learned about how they "
                                       "like things done.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "forget_way",
                        "description": "Stop doing something the way you learned, "
                                       "when they ask, by a word from it or its id.",
                        "parameters": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                            "required": ["key"],
                        },
                    },
                },
            ]
        if allow_delegate and spawner is not None:
            self._defs.append({
                "type": "function",
                "function": {
                    "name": "spawn_task",
                    "description": "Set your specialists working on something LONGER in "
                                   "the background, and bring the result back later "
                                   "(you'll return with it on your own). Use this, not "
                                   "delegate, when a task needs real digging that "
                                   "shouldn't hold up the conversation. Describe it "
                                   "fully. You will tell them you're on it now.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string",
                                     "description": "the task, described in full"},
                        },
                        "required": ["task"],
                    },
                },
            })
        if allow_delegate:
            self._defs.append({
                "type": "function",
                "function": {
                    "name": "consult_peer",
                    "description": "Consult another AGENT over the A2A protocol: given "
                                   "its base URL, discover it (its agent card) and send "
                                   "it a message, then speak from its reply. Use when a "
                                   "task is better suited to a different agent the "
                                   "person points you to.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string",
                                    "description": "the peer agent's base URL"},
                            "task": {"type": "string",
                                     "description": "what to ask it"},
                        },
                        "required": ["url", "task"],
                    },
                },
            })
            self._defs.append({
                "type": "function",
                "function": {
                    "name": "delegate",
                    "description": "Hand an involved task to one of your specialists "
                                   "and get back what they found or made. Use it when "
                                   "a task needs real digging or drafting: researching "
                                   "something on the web, searching through their own "
                                   "files/notes/history, or drafting a message or "
                                   "note. Describe the task fully in one sentence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string",
                                     "description": "the task, described in full"},
                        },
                        "required": ["task"],
                    },
                },
            })

    def openai_tools(self) -> list[dict]:
        return self._defs

    @property
    def has_tools(self) -> bool:
        return True

    async def call(self, name: str, args: dict) -> str:
        if name == "remind_me":
            due = _parse_iso(args.get("due_iso", ""))
            if due is None:
                return "(could not read the time, need an ISO datetime)"
            repeat = args.get("repeat")
            # "every day at 9am", said in the afternoon, arrives as TODAY's 9am:
            # already past. Stored as-is it fires instantly, once, before the re-arm.
            # Roll a RECURRING reminder to its next real slot; leave a one-shot alone,
            # since a genuinely overdue one-shot should still be returned.
            if repeat:
                now = time.time()
                if due <= now:
                    due = reminders_mod.next_occurrence(due, repeat, now) or due
            r = self.store.add(args.get("text", "").strip(), due, repeat=repeat)
            when = datetime.fromtimestamp(r.due_at).strftime("%a %b %d, %H:%M")
            cadence = f", then {r.repeat}" if r.repeat else ""
            return f"kept: \"{r.text}\", will return it {when}{cadence} (id {r.id})"
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
            steps = planner_mod.plan(title, self._plan_gen, reflect=True)
            if not steps:
                steps = ["Take the first small step toward it."]
            g = self.goals.add(title, steps)
            plan = "\n".join(f"  {i+1}. {s.text}" for i, s in enumerate(g.steps))
            return (f"taken on: \"{g.title}\" (id {g.id}), the plan I'll help with:\n"
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
        if name == "advance_goal":
            if self.goals is None:
                return "not holding any goals."
            g = self.goals.get(args.get("key", ""))
            if g is None:
                return "no matching goal."
            step = self.goals.advance(g, note=args.get("note", ""))
            if step is None:
                return f"\"{g.title}\" has no open steps."
            done, total = g.progress()
            if g.status == "done":
                return f"that completes \"{g.title}\": all {total} steps done."
            return (f"marked done: \"{step.text}\" ({done}/{total}). "
                    f"next: {g.next_step().text}")
        if name == "complete_goal":
            if self.goals is None:
                return "not holding any goals."
            g = self.goals.complete(args.get("key", ""))
            return f"set down: \"{g.title}\"" if g else "no matching goal."

        if name == "keep_note":
            if self.journal is None:
                return "(no journal to keep it in)"
            e = self.journal.keep(args.get("text", ""))
            return f"kept in the journal: \"{e.text}\"" if e else "(nothing to keep)"
        if name == "read_journal":
            if self.journal is None:
                return "(no journal)"
            recent = self.journal.recent()
            if not recent:
                return "the journal is empty."
            return "\n".join(f"- ({e.when()}) {e.text}" for e in recent)

        if name == "run_python":
            res = await asyncio.to_thread(sandbox_mod.run_python, args.get("code", ""))
            return res.as_text()

        if name in ("schedule_job", "list_jobs", "cancel_job"):
            return self._job_call(name, args)
        if name in ("keep_way", "list_ways", "forget_way"):
            return self._way_call(name, args)

        if name == "set_dial":
            if self.dial is None:
                return "(cannot change that from here)"
            level = self.dial.set(args.get("level", ""))
            if level is None:
                return f"(no such level; one of {', '.join(dial_mod.LEVELS)})"
            return f"set to {level}"

        if name == "spawn_task":
            if self._spawner is None:
                return "(cannot work in the background right now)"
            task = args.get("task", "").strip()
            if not task:
                return "(need a task to work on)"
            return await self._spawner(task)

        if name == "consult_peer":
            url = args.get("url", "").strip()
            task = args.get("task", "").strip()
            if not url or not task:
                return "(need a peer URL and a task)"
            return await a2a_mod.consult(url, task)

        if name == "delegate":
            if self._mcp is None or self._delegate_gen is None:
                return "(cannot delegate right now)"
            task = args.get("task", "").strip()
            if not task:
                return "(need a task to delegate)"
            # Specialists get native tools WITHOUT delegate: no recursion.
            sub_native = NativeTools(
                self.store, goals=self.goals, planner_generate=self._plan_gen,
                journal=self.journal, allow_delegate=False)
            orch = await subagents_mod.orchestrate(
                task, self._delegate_gen, mcp=self._mcp, native=sub_native)
            if not orch.result:
                return "(the specialists came back with nothing)"
            return f"[{orch.who()}] {orch.result}"
        return f"(no such tool: {name})"


    def _way_call(self, name: str, args: dict) -> str:
        if self.ways is None:
            return "(cannot learn that from here)"
        if name == "list_ways":
            act = self.ways.active()
            if not act:
                return "nothing learned yet about how they like things done."
            return "\n".join(f"- {w.line()} (learned from {w.source}, id {w.id})"
                             for w in act)
        if name == "forget_way":
            w = self.ways.forget(args.get("key", ""))
            return f"forgotten: {w.line()}" if w else "no matching way."
        w = self.ways.add(args.get("about", ""), args.get("how", ""), source="said",
                          evidence="they said so")
        return f"kept: {w.line()}" if w else "(need what it is about, and how)"

    def _job_call(self, name: str, args: dict) -> str:
        if self.jobs is None:
            return "(cannot schedule work from here)"
        if name == "list_jobs":
            act = self.jobs.active()
            if not act:
                return "running nothing on a schedule."
            return "\n".join(
                f"- {j.task} ({j.kind}, next "
                f"{datetime.fromtimestamp(j.next_at).strftime('%a %b %d %H:%M')}"
                f"{', ' + j.repeat if j.repeat else ', once'}, id {j.id})"
                for j in act)
        if name == "cancel_job":
            j = self.jobs.cancel(args.get("key", ""))
            return f"stopped: \"{j.task}\"" if j else "no matching job."
        due = _parse_iso(args.get("first_iso", ""))
        if due is None:
            return "(could not read the time, need an ISO datetime)"
        task = (args.get("task") or "").strip()
        kind = args.get("kind") or "task"
        if kind == "brief":
            task = task or "the morning brief"
        if not task:
            return "(need a task to run)"
        j = self.jobs.add(task, due, repeat=args.get("repeat"), kind=kind)
        when = datetime.fromtimestamp(j.next_at).strftime("%a %b %d, %H:%M")
        cadence = f", then {j.repeat}" if j.repeat else ""
        return f"scheduled: \"{j.task}\", first run {when}{cadence} (id {j.id})"


def _parse_iso(s: str):
    s = (s or "").strip().replace("Z", "")
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None
