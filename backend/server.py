"""
This is the Keeper as a running app.

A lean, single-user FastAPI service that wires the whole backend together:

    POST /chat        the passive loop: message -> recall -> compose -> distill
    GET  /events      Server-Sent Events: proactive lines pushed as they happen
    GET  /state       debug: energy, base_score, time since contact
    POST /config      live knobs (speed, the dial) so proactivity demos in seconds
    GET  /proposals   what it wants to do and is waiting on a yes for
    POST /proposals/decide   approve or decline one
    GET  /            health / info

A background task runs the proactive loop: every tick it decides, cooldown,
lock, roll, compose, and when the Keeper speaks unbidden, the line is pushed to
every connected /events listener. OpenAI calls are synchronous, so they run in a
thread to keep the event loop free.

Single user by design (it's a personal companion). State lives in memory except
the fact store, which persists to memory_store/facts.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

STATIC_DIR = Path(__file__).resolve().parent / "static"

import a2a
import background as background_mod
import channels
import compose
import dial as dial_mod
import drift
import embedder
import energy
import inbox as inbox_mod
import instance
import jobs as jobs_mod
import journal as journal_mod
import memory
import mood
import native_tools
import persona
import proposals as proposals_mod
import reminders
import routines
import relevance
import sessions
import sources
import subagents
import tasks
import proactive
import sensors
import tools
import unbidden
import voice_eval
import ways as ways_mod

TOOL_MODEL = "gpt-4o"   # model used for the passive tool-calling path

# Shown in the Keeper's own register when generation fails (outage, rate limit),
# so a broken model degrades to atmosphere instead of a 500 or a stack trace.
ERROR_LINE = "The line to the water has gone quiet. Stay; it returns."

# Reconciles "no window on the world" with "tools when asked": the sealing rule
# bars INVENTING the world unbidden. When they hand you a key: ask you to look:
# looking is keeping, not trespassing. This is appended only on the passive tool
# path; the proactive loop never sees it and stays sealed.
TOOL_ADDENDUM = """When they ask for something CONCRETE, to look at a file, find or
search for information, compare or buy something, a practical answer, actually GIVE it:
reach for the tool, do the work, and answer in plain words from what you found. If a
tool comes back empty, say so plainly ("nothing on your list yet"). Name the real answer:
 the brands, the price, the items themselves, never just the titles of your sources.
Your voice may colour a true answer; it must never REPLACE it. A metaphor handed back in
place of a practical answer is a failure, not the voice.

You reach BEYOND this chat with real tools: the open web, their files, the time, their
project history, code you can run, and other agents you can consult. Your "no window on
the world" rule bars you from INVENTING events unbidden; it does NOT bar using a tool you
were handed: a tool's result is grounded truth you fetched, not invention. So NEVER
refuse a request by claiming you cannot reach or access something when you hold a tool
for it (no "the shore is not open to you", no "access is not available"). If they ask you
to consult an agent at a URL, CALL consult_peer; to look something up, search; to run a
number, run_python. Try the tool FIRST, speak of a limit only if the tool itself fails.

You have been given tools to look at what is theirs, files they
keep: a journal, notes, lists, and the like. When they mention any of these, you
ALREADY have read access to them. NEVER ask them for a file path, and never say you
cannot see it yet, instead, DISCOVER it: first call the tool that lists the folder
or the allowed directories, then read the file that matches what they mean, then
answer from what you actually find. Do not decline, and do not guess at the contents.
Reading what is theirs is an act of keeping, not a window on the world: the sealing
rule bars inventing events unbidden, not reading what they asked you to read. Answer
in your own voice, but true to what the tool returned. To name a file, write its path
relative to the INSIDE of that folder: a file directly in it is just "sam.txt", one in
a subfolder "notes/sam.txt". Never repeat the folder's own name, and never copy out a
long absolute path by hand: one wrong character and the file is refused.

You can also HOLD things for them. When they ask you to remember to do something at
a time ("remind me to call the dentist tomorrow"), use remind_me, convert their
phrasing into an ISO datetime using the current time given in your context. Use
list_reminders when they ask what you're holding, and complete_reminder when
something is done. You will return a due reminder to them yourself when its time
comes; that is keeping, not intruding.

You can also take on GOALS, things they want to move toward but can't do in one
moment ("help me get back to running", "sort things out with Sam", "make the spare
room usable again"). Take one on only when they ASK you to work on it with them.
Naming a loss is not asking: "I stopped running in March" is a fact to keep and
nothing more. Keep it and say nothing about tending it, do not offer, do not hint
that you could, do not ask whether they would like you to. An offer dressed in your
own vocabulary ("you may ask me to tend that hope alongside you") is still an offer,
and it is not yours to make. Wait to be asked. When they do ask, use set_goal with
their own words; you will break it into small steps and help them through it over
days, returning to it on your own. When they tell you they've done a step ("I set
the paints out"), use advance_goal to mark it and move to the next. Use list_goals
to see what you are helping with, and complete_goal when the whole thing is
finished or they want to set it down. A goal is for tending over time; a reminder
is for one moment, choose the one that fits.

Some of your tools CHANGE things in the world: send an email, add to a calendar,
write a file. Those never run when you call them. Calling one PROPOSES it: they are
shown exactly what you passed and decide. So pass the real thing, the final text and
the real recipient, and only what they asked for. Fill in what they left unsaid
yourself, a short subject line, a greeting, rather than asking or leaving a gap: never
write a placeholder like [Your Name]. If you do not know how they sign off, end
without a name. Afterwards,
tell them what you drafted and that it is waiting on their yes. Never say it is done.
Earlier proposals in the conversation carry what became of them in [brackets]. If
one failed or was declined and it comes up, say so plainly before anything else:
never describe a note that was never saved as though it exists.

You can also run WORK for them on a schedule with schedule_job: "every weekday at
8, look at my calendar", "each Sunday check the gallery's site". A reminder hands
back words; a job does the task and brings back what it found. A daily brief or a
morning summary is kind "brief". "Brief me every weekday at 8" is complete as it
stands: call schedule_job with kind "brief" straight away rather than asking what it
should contain.

Never say you have set, scheduled, held, kept or sent something unless the tool that
does it has just told you it did. Saying it is not doing it: if you did not call the
tool, it did not happen. Only when they ask; use list_jobs and cancel_job
when they want to see or stop one.

They decide how often you speak on your own. When they ask you to be quieter, to
leave them be for a while, or to speak up more, use set_dial: off, quiet, usual or
eager. Never change it on your own judgement, and never suggest it. Off means you
start nothing yourself; what they asked you to hold or look into still comes back.

They can teach you HOW they like things done. When they say it in general, "from
now on", "always", "never", keep it with keep_way, so every later draft follows it,
yours and your specialists'. An instruction for this one task is just followed, not
kept: "this once", "only this time", "for this email" are never ways. list_ways shows what you have learned; forget_way when they ask you to stop.

You keep a JOURNAL: the one thing you can write. Use keep_note to hold a thought they
ask you to keep, or to record something you found or worked out; use read_journal to
look back. It is append-only: writing never erases.

When something needs to be WORKED OUT precisely: a calculation, date math, parsing or
transforming data, use run_python: write a short snippet that prints the answer, and
speak from what it returns. Reach for it instead of guessing at numbers.

You also have SPECIALISTS you can hand a bigger task to with delegate: a researcher
(looks things up on the web and synthesizes), an archivist (digs through their own
files, notes, and history), a scribe (drafts a message or note), and an analyst (works
things out by running code). When a task needs real digging, drafting, or computing
rather than a single quick tool call, delegate it in one sentence and speak from what
they bring back.

Two ways to set the specialists working: delegate waits with the person and answers in
the same breath, use it for quick tasks. spawn_task sends them off on something LONGER
and brings the result back later, on your own, without holding up the conversation,
use it when they ask you to look into something involved. With spawn_task, tell them
you're on it; you'll return with what you find when it's ready.

For a request that takes more than one step, work it in steps: call a tool, read
what it returns, then call the next, e.g. list_reminders to see what you hold,
then complete_reminder on the right one. Take the steps you need, then answer once
in your voice.

You can also read the present moment and the wider world when it helps: use the time
tools for what day or hour it is, and the fetch tool to read a web page or article
they point you to. If they ask about this project: the code, what you have been
building or working on lately, use the git tools with repo_path set to "{repo}"
(git_log for recent work, git_show to look closely at one change)."""

_REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL_ADDENDUM = TOOL_ADDENDUM.format(repo=_REPO_ROOT)

RECENT_WINDOW_MIN = 240.0   # "recent" messages = last 4h, for context richness

# Hard real-time circuit breaker: no matter how compressed the demo `speed` clock
# is, the Keeper will not REACH OUT (restlessness or a routine) more than once per
# this many seconds of REAL time. Due reminders are exempt (kept promises land on
# time). This is the backstop against notification spam.
MIN_REAL_REACH_GAP_S = 60.0
# How often the watched feeds are re-fetched, and how many new items are scored
# per pass: each costs a judge call, so this is a budget, not a limit on reading.
SOURCE_POLL_S = 900.0
SOURCE_SCAN_MAX = 8
# Same idea for idle reflection: never synthesize insights more than once per this
# many seconds of real time, so a fast demo clock can't flood the store.
MIN_REAL_DRIFT_GAP_S = 300.0


def within_reach_floor(last_proactive_at, now, min_gap: float = MIN_REAL_REACH_GAP_S):
    """True if we're still inside the real-time outreach floor and must stay silent.
    Pure + tested: the notification-spam backstop's core decision."""
    return last_proactive_at is not None and (now - last_proactive_at) < min_gap


# --------------------------------------------------------------------------- #
# App state: one user, held in memory.
# --------------------------------------------------------------------------- #

@dataclass
class AppState:
    store: memory.MemoryStore = field(default_factory=memory.MemoryStore)
    # Bounded: traces have always been a deque, but this was a plain list appended
    # on every turn AND every proactive line, never trimmed. Nothing read far enough
    # back to need more, and since _recent_lines() reads it each tick an unbounded
    # list is a slow leak in something meant to run for months.
    history: deque = field(default_factory=lambda: deque(maxlen=200))
    # When each outreach happened, for the day's ceiling. Bounded well above any
    # sane daily_max so a long-running day cannot grow it without limit.
    proactive_at: deque = field(default_factory=lambda: deque(maxlen=200))
    user_msg_times: list[float] = field(default_factory=list)
    last_user_at: Optional[float] = None
    last_proactive_at: Optional[float] = None
    current_register: Optional[str] = None   # last emotional register, for continuity
    config: proactive.ProactiveConfig = field(default_factory=proactive.ProactiveConfig)
    listeners: set[asyncio.Queue] = field(default_factory=set)
    delivery: Optional[channels.Delivery] = None   # web + banner + telegram fan-out
    traces: deque = field(default_factory=lambda: deque(maxlen=25))  # per-turn debug
    generate: compose.Generator = compose.stub_generator
    fast: Optional[compose.Generator] = None
    embed: Optional[memory.Embedder] = None   # semantic recall; None => keyword
    mood_signal: Optional[object] = None   # local mood classifier; None => keyword
    llm_mood_signal: Optional[object] = None   # live-model register read; None => local
    seen_sources: Optional[object] = None      # delivery keys already sent
    pending_item: Optional[object] = None      # scored, unsent, waiting for a tick
    # The whole last scoring pass, winners and rejects alike. The rejecting is the
    # interesting half: it is what separates a companion from a feed reader: and
    # until now only the winner was ever visible, in a log line.
    last_scan: list = field(default_factory=list)
    last_poll_at: Optional[float] = None
    # What the last fetch returned, kept so a change in MEMORY can re-score it
    # without going back to the network, and a fingerprint of the memory those
    # scores were computed against. See _memory_fingerprint.
    last_items: list = field(default_factory=list)
    scored_against: Optional[tuple] = None
    wake: Optional[asyncio.Event] = None   # set to interrupt the loop's sleep
    mcp: Optional[tools.MCPManager] = None  # passive-loop tools; None until connected
    reflections: drift.ReflectionLog = field(default_factory=drift.ReflectionLog)
    last_drift_at: Optional[float] = None
    drift_config: drift.DriftConfig = field(default_factory=drift.DriftConfig)
    reminders: reminders.ReminderStore = field(
        default_factory=reminders.ReminderStore)
    goals: tasks.GoalStore = field(default_factory=tasks.GoalStore)   # agent goals
    journal: journal_mod.Journal = field(default_factory=journal_mod.Journal)
    background: background_mod.BackgroundTasks = field(
        default_factory=background_mod.BackgroundTasks)   # long async delegations
    sessions: sessions.SessionStore = field(default_factory=sessions.SessionStore)
    current_key: Optional[str] = None   # active conversation (sidebar)
    routines_engine: routines.RoutineEngine = field(
        default_factory=routines.RoutineEngine)   # the house's presence routines
    unbidden: unbidden.UnbiddenLog = field(
        default_factory=unbidden.UnbiddenLog)     # every line it said unasked
    dial: dial_mod.Dial = field(default_factory=dial_mod.Dial)  # the person's say
    proposals: proposals_mod.ProposalStore = field(
        default_factory=proposals_mod.ProposalStore)   # actions waiting on a yes
    inbox: Optional[inbox_mod.TelegramInbox] = None    # taps back from the phone
    jobs: jobs_mod.JobStore = field(
        default_factory=jobs_mod.JobStore)     # errands it runs on a schedule
    ways: ways_mod.WayStore = field(
        default_factory=ways_mod.WayStore)     # how they like things done

    def minutes_since_user(self) -> Optional[float]:
        if self.last_user_at is None:
            return None
        return (time.time() - self.last_user_at) / 60.0

    def minutes_since_proactive(self) -> Optional[float]:
        if self.last_proactive_at is None:
            return None
        return (time.time() - self.last_proactive_at) / 60.0

    def recent_msg_count(self) -> int:
        cutoff = time.time() - RECENT_WINDOW_MIN * 60.0
        return sum(1 for t in self.user_msg_times if t >= cutoff)

    def proactive_today(self) -> int:
        """Outreaches so far today, on the REAL calendar.

        Real days on purpose, even while the demo clock is compressed: the ceiling
        exists to protect a person's actual day, and a knob that made it forget
        would defeat it exactly when the loop is at its most talkative.
        """
        start = datetime.now().replace(hour=0, minute=0, second=0,
                                       microsecond=0).timestamp()
        return sum(1 for t in self.proactive_at if t >= start)

    def proactive_state(self) -> proactive.ProactiveState:
        # The demo knob compresses the battery TIMESCALE, not just tick spacing:
        # at speed=120, one real second decays the battery like two virtual
        # minutes, so a reviewer watches restlessness rise in seconds. cooldown
        # and decay share this virtual clock, so they stay consistent.
        s = max(1.0, self.config.speed)
        ms_user = self.minutes_since_user()
        ms_pro = self.minutes_since_proactive()
        return proactive.ProactiveState(
            minutes_since_user=None if ms_user is None else ms_user * s,
            recent_msg_count=self.recent_msg_count(),
            proactive_today=self.proactive_today(),
            minutes_since_proactive=None if ms_pro is None else ms_pro * s,
        )


STATE = AppState()


async def _push(role: str, content: str, kind: str,
                state: Optional[str] = None, meta: Optional[dict] = None) -> None:
    """Fan a line out to every enabled delivery channel: the open web page, a native
    macOS banner, and (if configured) a Telegram message on your phone. See
    channels.py; each is best-effort, so one failing never blocks the others.

    `state` is the register the line was written in; the native banner wears it.
    Callers that know the water state pass it, and the rest fall back to the last
    register the conversation was in, which is the same continuity the composer
    uses when a message carries no signal of its own."""
    if STATE.delivery is None:
        STATE.delivery = channels.build_default(STATE.listeners)
    await STATE.delivery.push(role, content, kind,
                              state or STATE.current_register, meta=meta)


# --------------------------------------------------------------------------- #
# The proactive background loop.
# --------------------------------------------------------------------------- #

async def _proactive_loop() -> None:
    # Treat startup as fresh contact so it doesn't blast before you've spoken.
    if STATE.last_user_at is None:
        STATE.last_user_at = time.time()
    while True:
        # A kept promise comes first: deliver any due reminders regardless of the
        # battery decision. This is the Keeper returning what you asked it to hold.
        for r in STATE.reminders.due():
            line = f"You asked me to hold this: {r.text}. It's time."
            STATE.reminders.mark_delivered(r.id)
            # A kept promise is not outreach the day's ceiling should count.
            await _deliver_proactive(line, why=unbidden.Why.REMINDER,
                                     counts_toward_day=False)

        # Then the work they scheduled. Like a reminder it is theirs, so it is
        # exempt from the outreach floor and the day's ceiling; unlike one it takes
        # a while, so it runs off the loop and comes back when it is done.
        for job in STATE.jobs.due():
            if STATE.jobs.claim(job):
                asyncio.create_task(_run_job(job))
            else:
                print(f"[job] {job.id} skipped, too late to be worth running: "
                      f"{job.task[:50]}", flush=True)

        # One shared presence read for this tick, used by both the house routines
        # and the restlessness gate (so we don't ioreg twice).
        pres = await asyncio.to_thread(sensors.read)

        # Circuit breaker: how long since the LAST real outreach. If it's under the
        # floor, the Keeper stays silent this tick no matter what the demo clock
        # says: capping outreach at ~once per MIN_REAL_REACH_GAP_S of real time.
        now_real = time.time()
        real_gap = (None if STATE.last_proactive_at is None
                    else now_real - STATE.last_proactive_at)
        can_reach = not within_reach_floor(STATE.last_proactive_at, now_real)

        # Outreach priority when it's allowed to reach out: purposeful work first
        # (advance a due goal), then the house's routines (a return, long focus,
        # evening), then: falling through below: restless energy. The first to
        # speak stands in for this tick's outreach.
        #
        # All three are the Keeper's own initiative, which the person can switch
        # off (dial.py). Off skips them here rather than composing a line and
        # discarding it, so "off" also costs nothing: no judge calls on feeds, no
        # sub-agent working a goal step whose result could not be said.
        initiative = STATE.dial.level.initiative
        spoke = False
        if can_reach and initiative:
            spoke = await _maybe_advance_goal() or await _maybe_routine(pres)

        wait = max(1, int(STATE.config.tick_fast / max(STATE.config.speed, 1e-9)))
        if spoke:
            pass
        elif can_reach and not initiative:
            # Holding its tongue by request. Reflection is private and never sent,
            # so it carries on; reminders above already went out regardless.
            await _maybe_drift_note()
            wait = max(1, int(STATE.config.tick_slow
                              / max(STATE.config.speed, 1e-9)))
        elif can_reach:
            # Refresh what is being watched before deciding. Rate-limited inside,
            # and it only ever writes a candidate: the decision stays in tick().
            try:
                await _poll_sources()
            except Exception as exc:  # noqa: BLE001 - a feed must not kill the loop
                print(f"[sources] poll error: {type(exc).__name__}: {exc}",
                      flush=True)
            try:
                decision = await asyncio.to_thread(
                    proactive.tick,
                    STATE.proactive_state(),
                    generate=STATE.generate, fast_model=STATE.fast,
                    store=STATE.store, config=STATE.dial.shape(STATE.config),
                    presence=pres,
                    pending=STATE.pending_item,
                    recent=_recent_lines(),
                )
            except Exception as exc:  # noqa: BLE001 - never let the loop die silently
                print(f"[proactive] tick error: {type(exc).__name__}: {exc}",
                      flush=True)
                await asyncio.sleep(2)
                continue
            print(f"[proactive] tick spoke={decision.spoke} reason={decision.reason!r} "
                  f"E={decision.energy:.2f} score={decision.base_score:.2f} "
                  f"wait={decision.wait_next_s}s", flush=True)
            # Trace the unbidden lines too. The trace answers "why did it say
            # that", and a line the person did not ask for is where that question
            # actually bites: yet until now only chat turns were recorded, so
            # every autonomous decision was invisible outside the terminal.
            item = STATE.pending_item
            # Only decisions worth reading. Tracing every tick floods the ring with
            # "did not roll to speak" and pushes real turns out of it: the loop
            # ticks constantly, and most ticks are the Keeper correctly saying
            # nothing. A quiet that was ABOUT something (it had a line and withheld
            # it) is worth keeping; ordinary restraint is not.
            # Was matched against the English of the reason, so rewording a log
            # line silently changed what got traced. Codes are the closed set.
            worth_tracing = decision.spoke or decision.code in (
                proactive.Reason.REPEAT, proactive.Reason.COMPOSER_SILENT)
            if worth_tracing:
                STATE.traces.appendleft({
                    "ts": time.time(),
                    "kind": "proactive",
                    "cue": ("(unbidden, the Keeper decided to speak)" if decision.spoke
                            else "(unbidden: the Keeper had a line and withheld it)"),
                    "water_state": decision.water_state,
                    "register_note": f"proactive: {decision.reason}",
                    "memory_injected": "",
                    "path": "proactive",
                    "tools_called": [],
                    "reply": decision.text or "",
                    "spoke": decision.spoke,
                    "reason": decision.reason,
                    "code": decision.code,
                    "energy": round(decision.energy, 3),
                    "base_score": round(decision.base_score, 3),
                    "source_item": ({
                        "title": item.title,
                        "url": item.url,
                        "relevance": round(item.relevance, 2),
                        "because": item.because,
                    } if item is not None and decision.spoke else None),
                    "score": None,
                    "fell_back": False,
                })
            if decision.spoke and decision.text:
                # Whatever was pending has now been said (or was at least the
                # context for what was said), so retire it before anything else can
                # deliver it a second time.
                if STATE.pending_item is not None and STATE.seen_sources is not None:
                    STATE.seen_sources.mark(STATE.pending_item)
                    print(f"[sources] delivered {STATE.pending_item.title[:50]!r} "
                          f"(relevance {STATE.pending_item.relevance:.1f}, "
                          f"because {STATE.pending_item.because!r})", flush=True)
                    STATE.pending_item = None
                why = (unbidden.Why.NOTICED
                       if decision.code == proactive.Reason.SPOKE_SOURCE
                       else unbidden.Why.RESTLESS)
                await _deliver_proactive(decision.text, decision.water_state,
                                         why=why)
            else:
                await _maybe_drift_note()   # nothing to say -> maybe reflect
            wait = max(1, decision.wait_next_s)
        else:
            # Within the real-time floor: hold silence, still allow drift, and sleep
            # until the floor lifts so a fast demo clock can't busy-spin.
            await _maybe_drift_note()
            wait = max(1, int(MIN_REAL_REACH_GAP_S - (real_gap or 0.0)))
        # Interruptible sleep: a config change or a new chat wakes us early; and we
        # never sleep past the next due reminder, so kept promises land on time.
        upcoming = [r.due_at for r in STATE.reminders.items
                    if not r.done and not r.delivered]
        upcoming += [j.next_at for j in STATE.jobs.active()]
        if upcoming:
            wait = min(wait, max(1, min(upcoming) - time.time()))
        assert STATE.wake is not None
        try:
            await asyncio.wait_for(STATE.wake.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass
        STATE.wake.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # First, before anything that could speak: one Keeper per memory. See
    # instance.py for the night this was written for.
    lock = instance.InstanceLock(instance.LOCK_PATH)
    try:
        lock.acquire(port=os.environ.get("PORT", ""))
    except instance.AlreadyRunning as exc:
        print(f"[instance] refusing to start: {exc}", flush=True)
        raise SystemExit(1) from None
    STATE.generate, STATE.fast = compose.make_generator()
    STATE.embed = embedder.make_embedder()   # semantic recall when a key is set
    STATE.mood_signal = mood.build_local_mood_signal()   # local fallback
    if STATE.fast is not None:
        STATE.llm_mood_signal = mood.make_llm_mood_signal(STATE.fast)
    primary = ("llm" if STATE.llm_mood_signal
               else "model2vec" if STATE.mood_signal else "keyword")
    fallback = "model2vec" if STATE.mood_signal else "keyword"
    print(f"[mood] classifier: {primary}"
          + (f" (falls back to {fallback})" if primary == "llm" else ""), flush=True)
    # Presence readers fail silently per call (the loop reads every few seconds and
    # must not flood), so say once, here, which of them this machine can actually
    # provide: otherwise a blind sensor just shows defaults and looks healthy.
    caps = sensors.capabilities()
    live = [n for n, st in caps.items() if st == "live"]
    print(f"[sensors] live: {', '.join(live) if live else 'none'}"
          f" ({len(live)}/{len(caps)})", flush=True)
    for name, status in caps.items():
        if status != "live":
            print(f"[sensors] {name} unavailable, {status}", flush=True)
    STATE.seen_sources = sources.SeenStore()
    feeds = [u for u in os.environ.get("KEEPER_FEEDS", "").split(",") if u.strip()]
    print(f"[sources] watching {len(feeds)} feed(s)"
          + (f"; {len(STATE.seen_sources.keys)} already delivered" if feeds else
             ", set KEEPER_FEEDS to a comma-separated list to enable"), flush=True)
    # Resume the last conversation on start, but never the phone's: the page
    # would then type into the thread the phone owns.
    STATE.current_key = STATE.sessions.most_recent_key(exclude=PHONE_KEY)
    STATE.wake = asyncio.Event()
    # Delivery surfaces: web + native banner always; Telegram if a token is set.
    STATE.delivery = channels.build_default(STATE.listeners)
    print(f"[channels] delivering via: {', '.join(STATE.delivery.names())}",
          flush=True)
    # Passive-loop tools (optional). Connects only if backend/mcp.json exists.
    STATE.mcp = tools.MCPManager.from_config()
    try:
        await STATE.mcp.connect()
    except Exception as exc:  # noqa: BLE001 - never let MCP break startup
        print(f"[mcp] connect failed: {exc}", flush=True)
    task = asyncio.create_task(_proactive_loop())
    inbox_task = _start_inbox()
    try:
        yield
    finally:
        task.cancel()
        if inbox_task is not None:
            inbox_task.cancel()
        lock.release()
        if STATE.mcp is not None:
            try:
                await STATE.mcp.aclose()
            except Exception:  # noqa: BLE001
                pass


# The phone's conversation. One long thread at a fixed key, so switching
# conversations on the page can never redirect where the phone is talking.
PHONE_KEY = "s-telegram"


def _phone_thread() -> str:
    return STATE.sessions.get_or_create(PHONE_KEY, title="telegram").key


async def _converse_from_phone(text: str) -> str:
    result = await _converse(text, _phone_thread(), surface="telegram")
    return result["reply"]


def _phone_hears(kind: str) -> bool:
    """Whether a line of this kind reaches the phone, so its thread can hold it."""
    return STATE.delivery is not None and any(
        c.name == "telegram" and c.wants(kind) for c in STATE.delivery.channels)


def _start_inbox() -> Optional[asyncio.Task]:
    """Listen for taps, dial commands and conversation from Telegram, when it is
    configured."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id or os.environ.get("KEEPER_TELEGRAM_INBOX") == "0":
        STATE.inbox = None
        return None
    STATE.inbox = inbox_mod.TelegramInbox(
        token, chat_id, decide=_decide, set_level=_set_level_from_phone,
        converse=_converse_from_phone)
    print("[inbox] listening on telegram: conversation, approvals, the dial",
          flush=True)
    return asyncio.create_task(STATE.inbox.run())


def _set_level_from_phone(level: str) -> Optional[str]:
    set_to = STATE.dial.set(level)
    if set_to is not None:
        print(f"[dial] set to {set_to} from telegram", flush=True)
        if STATE.wake is not None:
            STATE.wake.set()
    return set_to


app = FastAPI(title="The Keeper", lifespan=lifespan)

# DNS-rebinding defense: only serve requests whose Host is localhost. A malicious
# website that rebinds its domain to 127.0.0.1 would send its own Host header, so
# it is refused. This is the main thing standing in for auth on a local, no-login
# app: do NOT expose this server publicly without real authentication.
app.add_middleware(TrustedHostMiddleware,
                   allowed_hosts=["localhost", "127.0.0.1"])


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

class ChatIn(BaseModel):
    message: str


# How much of the conversation a turn sees. Long-term memory is recall's job; this
# is only the thread of the last few exchanges, so a reply can refer back to what
# was just said. Bounded in count AND in time: yesterday's last message is not
# context for this morning's first one.
RECENT_TURNS = 8
RECENT_TURNS_WINDOW_S = 2 * 3600.0


_TIMED_ASK = re.compile(
    r"\b(remind me|brief me|every (day|weekday|week|morning|evening|night|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|each (day|"
    r"morning|evening|week|weekday)|daily|weekly|schedule|tomorrow at|at \d)",
    re.I)


_STANDING_ASK = re.compile(
    r"\b(from now on|going forward|from here on|always|never|every time|whenever)\b",
    re.I)


_ACTION_ASK = re.compile(
    r"\b(send|e-?mail|reply to|write (to|back)|forward|invite|"
    r"(add|put)\b[^.]*\b(calendar|diary)|(make|book|schedule) a (meeting|call|event)|"
    r"(save|write|put)\b[^.]*\b(file|note|sandbox))\b", re.I)


def _expected_tools(msg: str, proposable: Optional[set] = None) -> Optional[set]:
    """The tools a request plainly calls for, or None. Narrow on purpose: a request
    about time, where "it is set" without setting it leaves the person waiting for
    something that never comes, and a standing instruction, where "noted, from now
    on" without keep_way is a promise nothing will keep (2 runs in 4, live). A
    false match costs one extra round and changes nothing: the nudge says to carry
    on if the tool was not called for."""
    expected: set = set()
    # A request to act in the world, when there are tools that act in the world.
    # Live demo: "send an email to ... with the subject ..." got the email typed
    # out in the reply and "you will see it ready for your approval", with no
    # proposal made: the person would have waited on a card that never came.
    if proposable and _ACTION_ASK.search(msg or ""):
        expected |= set(proposable)
    if _TIMED_ASK.search(msg or ""):
        expected |= {"remind_me", "schedule_job"}
    if _STANDING_ASK.search(msg or ""):
        expected.add("keep_way")
    return expected or None


def _now_note(now: Optional[float] = None) -> str:
    """Today, said next to the request. The time was always in the system prompt,
    and in a fresh conversation the model still asked the calendar about "this
    week" in October 2023: the date it carries from training beat the one buried in
    a long prompt. Found on camera, recording the README demo. Next to the message,
    with the weekday spelled out, is where it gets used."""
    dt = datetime.fromtimestamp(now or time.time()).astimezone()
    return (f"Right now it is {dt:%A %d %B %Y, %H:%M} ({dt.tzname()}, UTC"
            f"{dt:%z}). Use this for anything about dates or times: 'today', 'this "
            f"week', 'tomorrow' all count from here.")


def _recent_turns(key: Optional[str], now: Optional[float] = None) -> list:
    """The tail of a conversation as chat messages, oldest first.

    Unbidden lines and proposals are included: "what did he say?" is usually a
    reply to something the Keeper said first, and "make it shorter" to a draft it
    is waiting on a yes for. What became of a proposal is added as its own entry
    at the moment it was decided, not folded into the question: the question is
    saved mid-turn, so the Keeper's own "it is waiting" comes after it, and an
    outcome folded in earlier was contradicted by the last thing in the thread.
    Found in the live demo."""
    s = STATE.sessions.get(key) if key else None
    if s is None:
        return []
    now = now or time.time()
    entries = []
    for m in s.messages:
        if now - m.ts > RECENT_TURNS_WINDOW_S:
            continue
        entries.append((m.ts, "user" if m.role == "user" else "assistant",
                        m.content))
        settled = _outcome_of(m, now)
        if settled is not None:
            entries.append(settled)
    entries.sort(key=lambda e: e[0])
    return [{"role": role, "content": text}
            for _, role, text in entries[-RECENT_TURNS:]]


def _outcome_of(m, now: float) -> Optional[tuple]:
    """(when, role, note) for a decided proposal, read from the store so it is
    current; None for anything else, or one still waiting."""
    if m.kind != "proposal" or not m.ref:
        return None
    p = STATE.proposals.get(m.ref)
    if p is None or p.status == proposals_mod.PENDING or not p.decided_at:
        return None
    if p.decided_at > now:
        return None
    note = channels.outcome_line(p.public())
    if p.original_args is not None:
        note += " They edited it first; what ran was:\n" + proposals_mod.render(
            p.action, p.args)
    return (p.decided_at, "assistant", f"[{note}]")


@app.post("/chat")
async def chat(body: ChatIn):
    if STATE.current_key is None:                 # start a conversation on first msg
        STATE.current_key = STATE.sessions.new().key
    return await _converse(body.message.strip(), STATE.current_key, surface="web")


async def _converse(msg: str, key: str, surface: str = "web") -> dict:
    """One turn of conversation, from any surface, kept in conversation `key`.

    The page passes whichever conversation is open; the phone passes its own
    thread. Everything else, memory, goals, reminders, the register, is one Keeper
    whichever way the person reached it."""
    now = time.time()
    prior = _recent_turns(key, now)
    STATE.last_user_at = now
    STATE.user_msg_times.append(now)
    STATE.history.append({"role": "user", "content": msg, "ts": now})
    STATE.sessions.append(key, "user", msg, now)

    # Retrieve-then-rerank on the user-facing path: hybrid casts wide, the fast model
    # reranks to the best few. (Background paths use plain hybrid: no per-turn cost.)
    mem = memory.recall(STATE.store, msg, k=4, embed=STATE.embed,
                        rerank_generate=STATE.fast)
    # current time in the context so the Keeper can turn "tomorrow 9am" -> ISO.
    ctx = (sensors.read().to_context_line()
           + f"; current time {datetime.now().isoformat(timespec='minutes')}")
    # Register continuity: a clear emotional signal sets the register; a neutral
    # follow-up ("what should i do") INHERITS it rather than resetting to tidal,
    # so a stuck person is never told they're moving.
    # Mood. The model decides when there is one; the local pair is the offline
    # path. Measured on 32 real conversation turns:
    #
    #   llm             6% wrong        keyword   fires 9/32, right 5 of those 9
    #   model2vec      41% wrong (with the keyword layer in front of it)
    #
    # The keyword lexicon used to go first, justified as high precision. It is
    # not: it met "go look into watercolor vs gouache" in the register reserved
    # for grief, because "ache" sits inside "gouache": and once that was fixed to
    # match on word boundaries it merely failed differently, on "back to" inside
    # "get back to me". Those are word SENSE, which a word list cannot see. First
    # in the chain it had veto over a classifier three times more accurate.
    #
    # It stays, underneath, because with no key and no network it and the anchor
    # classifier are the only register detection there is.
    #
    # The model returning None is a DECISION: it read the message as neutral:
    # and is honoured. Only an unavailable model falls through.
    signal = None
    if STATE.llm_mood_signal is not None:
        try:
            signal = STATE.llm_mood_signal(msg)
        except Exception as exc:  # noqa: BLE001 - retry already ran inside compose
            print(f"[mood] llm classifier failed, using local "
                  f"({type(exc).__name__}: {exc})", flush=True)
            signal = voice_eval.register_signal(msg)
            if signal is None and STATE.mood_signal is not None:
                signal = STATE.mood_signal(msg)
    else:
        signal = voice_eval.register_signal(msg)
        if signal is None and STATE.mood_signal is not None:
            signal = STATE.mood_signal(msg)
    if signal is not None:
        STATE.current_register = signal
    water = STATE.current_register or voice_eval.detect_state(msg)
    water, register_note = _resolve_turn(msg, mem, water)

    # Native action tools (reminders + goals + journal + delegate) are always
    # available; MCP tools join when configured. Reaching for a tool: or handing a
    # task to a specialist sub-agent: is the passive/agentic path.
    providers = [native_tools.NativeTools(
        STATE.reminders, goals=STATE.goals,
        planner_generate=STATE.fast or STATE.generate, journal=STATE.journal,
        mcp=STATE.mcp, delegate_generate=STATE.fast or STATE.generate,
        spawner=_spawn_background, dial=STATE.dial, jobs=STATE.jobs,
        ways=STATE.ways)]
    if STATE.mcp is not None and STATE.mcp.has_tools:
        providers.append(STATE.mcp)
    proposer = _proposer("chat")
    if proposer is not None:
        providers.append(proposer)
    used_tools = bool(providers)
    tool_trace: list = []
    ways_used = STATE.ways.for_task(msg)
    STATE.ways.mark_used(ways_used)
    try:
        if used_tools:
            # Tool path: the Keeper may reach for tools, then answer. A
            # tool-grounded answer can be plainer (a real fact in voice), so we
            # score it for information only, never replacing it with a fallback.
            system = persona.build_system_prompt(
                "passive", water, memory=mem, context=ctx)
            system = system + "\n\n---\n\n" + TOOL_ADDENDUM
            reply = await compose.tool_reply(
                system, msg, providers=providers, model=TOOL_MODEL, max_rounds=6,
                trace=tool_trace, prior=prior,
                steer="\n\n".join(x for x in (
                    _now_note(now), ways_mod.as_guidance(ways_used)) if x),
                expect_one_of=_expected_tools(
                    msg, set(proposer.actions) if proposer is not None else None))
            # Tool answers come back plain (a changelog, a file dump). Pass them back
            # through the Keeper's voice: preserving every fact: so a tool-grounded
            # reply still sounds like the Keeper, not a report.
            if reply and reply.strip():
                reply = await asyncio.to_thread(
                    compose.revoice, reply, water,
                    generate=STATE.fast or STATE.generate, memory=mem, context=ctx)
            report = voice_eval.evaluate(reply)   # deterministic, for logging
            score, fell_back = report.score, False
        else:
            result = await asyncio.to_thread(
                compose.compose, "passive", water,
                generate=STATE.generate, fast_model=STATE.fast,
                user_message=msg, memory=mem, context=ctx)
            reply, score, fell_back = result.text or "", result.score, result.fell_back
    except Exception as exc:  # noqa: BLE001 - the water must never 500 at them
        print(f"[chat] generation failed: {type(exc).__name__}: {exc}", flush=True)
        reply, score, fell_back = ERROR_LINE, None, True

    STATE.history.append({"role": "assistant", "content": reply, "ts": time.time()})
    STATE.sessions.append(key, "assistant", reply)
    # Note: the reply is returned in the HTTP response and rendered from there;
    # SSE (/events) carries ONLY unbidden proactive lines, so nothing double-renders.

    # Per-turn TRACE: the whole point is answering "why did it say that?" fast: what
    # memory was injected, the register, which tools fired, the reply. Ring-buffered.
    STATE.traces.appendleft({
        "ts": now,
        "cue": msg,
        "surface": surface,
        "turns_seen": len(prior),
        "ways_applied": [w.line() for w in ways_used],
        "water_state": water,
        "mood_signal": signal,
        # WHY that register, not just which. The trace exists to answer "why did it
        # say that", and until now the most interesting register decision: turn
        # being granted or refused by the store: only reached stdout.
        "register_note": register_note or (
            f"classified {signal}" if signal else "inherited (no signal in this message)"),
        "memory_injected": mem,
        "path": "tool" if used_tools else "compose",
        "tools_called": tool_trace,
        "reply": reply,
        "score": round(score, 3) if isinstance(score, (int, float)) else None,
        "fell_back": fell_back,
    })

    # Distill this exchange into the drawers, off the response path: but not a
    # failed turn, which carries no real reply to learn from.
    if reply != ERROR_LINE:
        asyncio.create_task(_distill_async(msg, reply))
    if STATE.wake is not None:
        STATE.wake.set()   # re-tick: energy just reset, loop should back off

    return {"reply": reply, "water_state": water, "score": score,
            "fell_back": fell_back, "used_tools": used_tools}


async def _maybe_routine(pres: sensors.Presence) -> bool:
    """Check the house routines against the current presence; if one has earned a
    line, compose it in the Keeper's voice (the routine's intent as context) and
    deliver it. Returns True iff a routine spoke this tick."""
    try:
        routine = STATE.routines_engine.check(
            pres, time.time(), STATE.minutes_since_user())
    except Exception as exc:  # noqa: BLE001
        print(f"[routine] check error: {exc}", flush=True)
        return False
    if routine is None:
        return False
    water = proactive.derive_state(STATE.minutes_since_user())
    mem = memory.recall(STATE.store, "", k=3, embed=STATE.embed)
    result = await asyncio.to_thread(
        compose.compose, "proactive", water,
        generate=STATE.generate, fast_model=STATE.fast,
        memory=mem, context=routine.intent)
    if result.silent or not result.text:
        return False        # composed nothing on-voice -> let restlessness decide
    STATE.routines_engine.fire(routine, time.time())
    # Routines carry their own cooldowns and have never counted toward the day.
    await _deliver_proactive(result.text, water, why=unbidden.Why.ROUTINE,
                             counts_toward_day=False)
    print(f"[routine] {routine.key} spoke", flush=True)
    return True


def _goal_check_interval() -> float:
    """How long to wait before working a goal again, on the DEMO clock.

    tasks.DEFAULT_CHECK_INTERVAL_S is 6 real hours. The `speed` knob compresses the
    battery and the drift clock but not this one, so before this existed a goal went
    quiet for 6 real hours after its first step: which meant the documented
    "crank speed to see nudges/autonomous execution" trigger could never fire, and
    a person-step was never nudged in a demo. Compress it the same way drift is.
    """
    return (STATE.dial.goal_interval(tasks.DEFAULT_CHECK_INTERVAL_S)
            / max(STATE.config.speed, 1.0))


def _recent_lines(limit: int = proactive.REPEAT_WINDOW) -> list:
    """The last few things the Keeper actually said, newest last.

    Read from STATE.history, which every delivery already appends to, so this adds
    no state of its own. It resets when the process does, acceptable, because the
    repetition it guards against comes from the composer being handed identical
    input tick after tick within a run.
    """
    out = [m.get("content", "") for m in STATE.history
           if m.get("role") == "assistant" and m.get("content")]
    return out[-limit:]


async def _poll_sources() -> None:
    """Fetch what is being watched, score it against memory, keep the best one.

    Runs off the tick path deliberately. proactive.tick() is sync, pure and unit
    testable with no network, and it stays that way: this writes a scored candidate
    to STATE and the tick only reads it.
    """
    if STATE.store is None or STATE.seen_sources is None:
        return
    feeds = [u.strip() for u in os.environ.get("KEEPER_FEEDS", "").split(",")
             if u.strip()]
    # Tools the Keeper watches, from mcp.json. The same servers already serving the
    # /chat path, read a second time for what is new: a commit that landed, a file
    # that changed. sources.py explains which tools may honestly go in this list.
    watches = list(STATE.mcp.watches) if STATE.mcp is not None else []
    if not feeds and not watches:
        return
    # Nothing kept yet means nothing can be relevant yet: relevance is measured
    # against what the Keeper knows about the person, so an empty store scores
    # every item 0. Bailing BEFORE last_poll_at is stamped matters, otherwise the
    # first poll, which happens moments after boot, burns the interval and records
    # a scan of all zeros that stands for the next fifteen minutes.
    if not any(f.active for f in STATE.store.facts):
        return

    # FETCHING and SCORING are on different clocks, and conflating them was a bug.
    # A score is a claim about the person as they are known RIGHT NOW, so it goes
    # stale the moment memory changes, while a fetch is a network call that should
    # stay polite. Seen live: the first poll landed mid-conversation, when
    # "Stopped painting in March." was the only fact, and the judge quite correctly
    # scored a painting exhibition near zero for someone who had stopped painting.
    # The reversal arrived seconds later, but the all-silent scan was already
    # stamped and stood for the next fifteen minutes.
    now = time.time()
    fingerprint = _memory_fingerprint()
    due_fetch = (STATE.last_poll_at is None
                 or now - STATE.last_poll_at
                 >= SOURCE_POLL_S / max(STATE.config.speed, 1.0))
    stale_scores = bool(STATE.last_items) and fingerprint != STATE.scored_against
    if not due_fetch and not stale_scores:
        return

    batches: list[list] = []
    if not due_fetch:
        # Memory moved under scores we already have. Re-judge what was already
        # fetched; do not go back to the network for it.
        batches.append(list(STATE.last_items))
        print(f"[sources] memory changed, re-scoring {len(STATE.last_items)} "
              f"item(s) without refetching", flush=True)
    for url in (feeds if due_fetch else []):
        try:
            xml = await asyncio.to_thread(_fetch_text, url)
            batches.append(sources.parse_feed(xml, source=url))
        except Exception as exc:  # noqa: BLE001 - a dead feed is never fatal
            print(f"[sources] {url} failed: {type(exc).__name__}: {exc}", flush=True)
    for watch in (watches if due_fetch else []):
        tool = watch.get("tool", "")
        label = watch.get("name") or tool
        if not tool:
            continue
        try:
            # mcp.call answers rather than raising, and its deadline is what keeps
            # a hung server from stalling the whole proactive loop.
            text = await STATE.mcp.call(
                tool, sources.expand_time(watch.get("args") or {}, now))
            batches.append(sources.parse_watch_output(
                text, source=label, max_items=int(watch.get("max_items", 8)),
                parser=watch.get("parser")))
        except Exception as exc:  # noqa: BLE001 - a dead tool is never fatal
            print(f"[sources] watch {label} failed: {type(exc).__name__}: {exc}",
                  flush=True)

    if due_fetch:
        STATE.last_poll_at = now
        # Keep what was fetched so a later change in memory can re-judge it.
        STATE.last_items = [i for b in batches for i in b]

    best = None
    scanned: list = []
    for items in batches:
        fresh = STATE.seen_sources.unseen(items)[:SOURCE_SCAN_MAX]
        for item in fresh:
            score, because = await asyncio.to_thread(
                relevance.score_item, item, STATE.store,
                generate=STATE.fast or STATE.generate, embed=STATE.embed)
            item.relevance, item.because = score, because
            scanned.append({
                "title": item.title,
                "relevance": round(score, 2),
                "because": because,
                "verdict": ("interrupt" if relevance.worth_interrupting(score)
                            else "mention" if relevance.worth_mentioning(score)
                            else "silent"),
            })
            if relevance.worth_mentioning(score) and (
                    best is None or score > best.relevance):
                best = item
    # These scores are now a claim about memory as it stands at `fingerprint`.
    # Recording that is what lets the next tick notice they have gone stale.
    STATE.scored_against = fingerprint
    if scanned:
        STATE.last_scan = sorted(scanned, key=lambda r: -r["relevance"])
        kept = sum(1 for r in scanned if r["verdict"] != "silent")
        print(f"[sources] scanned {len(scanned)}, {len(scanned) - kept} judged "
              f"not worth saying", flush=True)
    if best is not None:
        STATE.pending_item = best
        print(f"[sources] pending {best.title[:50]!r} "
              f"relevance={best.relevance:.1f} because={best.because!r}", flush=True)


def _memory_fingerprint() -> tuple:
    """A cheap stand-in for "what the Keeper currently knows".

    Relevance is judged against the active facts, so any change to them, a new
    one distilled or an old one superseded, invalidates every score computed
    before it. Counting active facts and recorded changes catches both, and both
    are already in memory, so this costs nothing per tick.
    """
    if STATE.store is None:
        return (0, 0)
    active = sum(1 for f in STATE.store.facts if f.active)
    return (active, len(STATE.store.changes()))


def _fetch_text(url: str) -> str:
    import httpx
    r = httpx.get(url, timeout=15, follow_redirects=True,
                  headers={"user-agent": "the-keeper/1.0"})
    r.raise_for_status()
    return r.text


async def _maybe_advance_goal() -> bool:
    """The agent at work: if a goal is due, take its next step. A [keeper] step it
    EXECUTES itself with its tools (ReAct); a [person] step it NUDGES, then waits for
    them to report it done. Returns True iff it reached out this tick."""
    goal = STATE.goals.due()
    if goal is None:
        return False
    step = goal.next_step()
    if step is None:
        return False
    water = proactive.derive_state(STATE.minutes_since_user())
    if step.actor == "keeper":
        return await _execute_goal_step(goal, step, water)
    return await _nudge_goal_step(goal, step, water)


async def _deliver_proactive(text: str, state: Optional[str] = None,
                             why: str = unbidden.Why.RESTLESS,
                             counts_toward_day: bool = True) -> None:
    """Record + fan out one unbidden line (log, history, session, all channels).

    The only way an unbidden line leaves the Keeper. There used to be four, one per
    caller, each copying the bookkeeping by hand, and they had already drifted:
    the tick and goal paths counted toward the day's ceiling while the reminder
    and routine paths did not, which only an audit of all four could see.
    `counts_toward_day` keeps that split, now on purpose and in one place.

    `state` is the register it was composed in, so the native banner can wear
    the matching Keeper. `why` says which part of the Keeper spoke."""
    # The loop already skips the Keeper's own initiative when the dial is off.
    # This is the backstop for work that was in flight when it was turned: a goal
    # step a sub-agent was still working, say. Its result is kept on the step; it
    # just is not said.
    if not STATE.dial.allows(why):
        print(f"[dial] held a {why} line, the dial is {STATE.dial.name}", flush=True)
        return
    # A line with nowhere to be kept is a line the page can never show again. After
    # "begin again", or on a first run, there is no conversation, so the Keeper's
    # line opens one, the way a person's first message would.
    if STATE.current_key is None:
        STATE.current_key = STATE.sessions.new().key
    line = STATE.unbidden.record(text, why=why, state=state,
                                 session=STATE.current_key)
    STATE.last_proactive_at = line.ts
    if counts_toward_day:
        STATE.proactive_at.append(line.ts)
    STATE.history.append({"role": "assistant", "content": text, "ts": line.ts})
    STATE.sessions.append(STATE.current_key, "assistant", text, line.ts,
                          kind="proactive")
    # It reaches the phone too, so the phone's thread keeps it: a reply there
    # ("what did he say?") is usually to the line the Keeper just sent.
    if _phone_hears("proactive") and STATE.current_key != PHONE_KEY:
        STATE.sessions.append(_phone_thread(), "assistant", text, line.ts,
                              kind="proactive")
    await _push("assistant", text, "proactive", state,
                meta={"id": line.seq, "session": line.session, "why": line.why})


async def _nudge_goal_step(goal, step, water) -> bool:
    """A [person] step: help with or invite it, but DON'T complete it, a person step
    is only done when they report it (advance_goal). Holds the thread, returns it."""
    context = (f"You are helping them move toward a goal of theirs: \"{goal.title}\". "
               f"Gently help with, or invite, just this next step, do not list the "
               f"whole plan: {step.text}")
    result = await asyncio.to_thread(
        compose.compose, "proactive", water,
        generate=STATE.generate, fast_model=STATE.fast,
        memory=memory.recall(STATE.store, goal.title, k=3, embed=STATE.embed),
        context=context)
    if result.silent or not result.text:
        # Say so. A declined nudge left no trace anywhere: the goal was quietly
        # pushed forward and the only visible evidence was next_check_at moving,
        # which reads identically to the loop never having considered it. Silence
        # is a legitimate answer here, but an invisible one is indistinguishable
        # from a bug.
        STATE.goals.touch(goal, _goal_check_interval())
        print(f"[goal] {goal.id} nudge declined (chose silence) {goal.progress()[0]}"
              f"/{goal.progress()[1]}: {step.text[:44]}", flush=True)
        return False
    STATE.goals.touch(goal, _goal_check_interval())   # nudge, don't complete
    await _deliver_proactive(result.text, water, why=unbidden.Why.GOAL)
    done, total = goal.progress()
    print(f"[goal] {goal.id} nudged {done}/{total}: {step.text[:50]}", flush=True)
    return True


async def _execute_goal_step(goal, step, water) -> bool:
    """A [keeper] step: hand it to a specialist SUB-AGENT (multi-agent), which runs its
    own tool loop and returns a result; the Keeper then speaks it. On empty/failed work
    the step is handed back to the person (re-labelled) so it gets nudged next time."""
    sub_native = native_tools.NativeTools(
        STATE.reminders, goals=STATE.goals,
        planner_generate=STATE.fast or STATE.generate, journal=STATE.journal,
        allow_delegate=False)               # a sub-agent must not spawn sub-agents
    try:
        res = await subagents.orchestrate(
            _with_ways(f"For their goal \"{goal.title}\", do this: {step.text}"),
            STATE.fast or STATE.generate, mcp=STATE.mcp, native=sub_native,
            proposer=_proposer(f"goal:{goal.id}"))
    except Exception as exc:  # noqa: BLE001
        print(f"[goal] execute error: {exc}", flush=True)
        STATE.goals.touch(goal, _goal_check_interval())
        return False
    result = (res.result or "").strip()
    if not result:
        step.actor = "person"            # hand it back, nudge them next time
        STATE.goals.touch(goal, _goal_check_interval())
        print(f"[goal] {goal.id} sub-agent found nothing, handed back: "
              f"{step.text[:40]}", flush=True)
        return False
    mem = memory.recall(STATE.store, goal.title, k=3, embed=STATE.embed)
    voiced = await asyncio.to_thread(
        compose.revoice, result, water, generate=STATE.fast or STATE.generate,
        memory=mem)
    STATE.goals.advance(goal, note=f"[{res.who()}] {result[:130]}",   # who did it
                        interval_s=_goal_check_interval())
    await _deliver_proactive(voiced, water, why=unbidden.Why.GOAL)
    done, total = goal.progress()
    print(f"[goal] {goal.id} EXECUTED via {res.who()} {done}/{total}: "
          f"{step.text[:45]}", flush=True)
    return True


# --------------------------------------------------------------------------- #
# Proposals: what the Keeper wants to do in the world, waiting on a yes.
# --------------------------------------------------------------------------- #

def _proposer(origin: str) -> Optional[proposals_mod.Proposer]:
    """A provider for the tools that may only run on a yes, or None when no server
    lists any. Built per call so each proposal remembers where it came from.

    Never built for the A2A path: a peer agent may ask the Keeper things, but it
    must not be able to put actions in front of the person, even ones they would
    have to approve. A queue anyone can fill trains people to tap yes."""
    if STATE.mcp is None:
        return None
    actions = [proposals_mod.Action(t["name"], t["description"], t["parameters"],
                                    fixed=dict(t.get("fixed") or {}))
               for t in STATE.mcp.proposable()]
    if not actions:
        return None
    return proposals_mod.Proposer(STATE.proposals, actions, _deliver_proposal,
                                  origin=origin)


def _proposal_text(p: proposals_mod.Proposal) -> str:
    """Machinery, not voice: the exact payload and nothing the model wrote about it.
    The Keeper's own words about the draft are in its chat reply."""
    return f"Waiting on your yes:\n\n{proposals_mod.render(p.action, p.args)}"


async def _deliver_proposal(p: proposals_mod.Proposal) -> None:
    """Put a proposal in front of the person on every surface that can carry it.

    Logged and kept in the conversation like an unbidden line, so a page that was
    closed when it was asked still shows it, and a reconnect replays it. It does
    not touch the outreach clocks: asking permission is not the Keeper reaching out,
    and must not spend the day's ceiling or reset the cooldown."""
    if STATE.current_key is None:
        STATE.current_key = STATE.sessions.new().key
    text = _proposal_text(p)
    line = STATE.unbidden.record(text, why=unbidden.Why.PROPOSAL,
                                 session=STATE.current_key, ref=p.id)
    STATE.sessions.append(STATE.current_key, "assistant", text, line.ts,
                          kind="proposal", ref=p.id)
    if _phone_hears("proposal") and STATE.current_key != PHONE_KEY:
        STATE.sessions.append(_phone_thread(), "assistant", text, line.ts,
                              kind="proposal", ref=p.id)
    print(f"[proposal] {p.id} from {p.origin}: {p.action}", flush=True)
    await _push("assistant", text, "proposal",
                meta={"id": line.seq, "session": line.session, "why": line.why,
                      "proposal": p.public()})


async def _decide(pid: str, approve: bool, via: str,
                  edited: Optional[dict] = None) -> str:
    """Approve or decline a proposal, from any surface. Returns a short note for
    the surface that asked (a Telegram toast, the page's card).

    Idempotent: a second tap, a tap on the page after one on the phone, or a tap on
    something already expired is told what already happened and changes nothing."""
    p = STATE.proposals.get(pid)
    if p is None:
        return "No such proposal."
    if p.status != proposals_mod.PENDING:
        return f"Already {p.status}."
    if not approve:
        STATE.proposals.decline(p, via)
        print(f"[proposal] {p.id} declined via {via}", flush=True)
        await _push_outcome(p)
        return "Declined. Nothing was done."
    if edited:
        refused = STATE.proposals.apply_edit(p, edited)
        if refused:
            return f"Not approved: {refused}."
    # Saved as approved BEFORE the call. See proposals.py, rule 2.
    STATE.proposals.mark_approved(p, via)
    if STATE.mcp is None:
        result = "(tool error: no tools are connected right now)"
    else:
        result = await STATE.mcp.call_proposable(p.action, p.args)
    STATE.proposals.finish(p, result)
    print(f"[proposal] {p.id} approved via {via}: {p.status}", flush=True)
    await _push_outcome(p)
    # Learned whether or not the tool then worked: the edit says what they wanted,
    # and a refused path or a network blip does not change that.
    if p.original_args is not None:
        asyncio.create_task(_learn_from_edit(p))
    return "Done." if p.status == proposals_mod.DONE else "Approved, but it failed."


async def _push_outcome(p: proposals_mod.Proposal) -> None:
    public = p.public()
    await _push("assistant", channels.outcome_line(public), "proposal_update",
                meta={"proposal": public})


def _same_origin(req: Request) -> bool:
    """A cross-site page can make the browser POST here: a plain form submission
    carries Host: localhost, so TrustedHostMiddleware passes it. What it cannot
    forge is Origin. Absent (curl, a script) is allowed; that caller could reach
    127.0.0.1 anyway."""
    origin = req.headers.get("origin")
    if origin is None:
        return True
    return urlparse(origin).hostname in ("localhost", "127.0.0.1")


async def _answer_as_keeper(text: str) -> str:
    """Answer an incoming message (e.g. from a peer agent over A2A) as the Keeper, with
    its tools, but a SAFE subset: no delegate/spawn, so a peer can't spawn work."""
    water = STATE.current_register or voice_eval.detect_state(text)
    mem = memory.recall(STATE.store, text, k=4, embed=STATE.embed)
    system = persona.build_system_prompt("passive", water, memory=mem)
    system = system + "\n\n---\n\n" + TOOL_ADDENDUM
    providers = [native_tools.NativeTools(
        STATE.reminders, goals=STATE.goals,
        planner_generate=STATE.fast or STATE.generate, journal=STATE.journal,
        allow_delegate=False)]
    if STATE.mcp is not None and STATE.mcp.has_tools:
        providers.append(STATE.mcp)
    try:
        reply = await compose.tool_reply(
            system, text, providers=providers, model=TOOL_MODEL, max_rounds=6)
    except Exception as exc:  # noqa: BLE001
        return f"(the water is unsettled: {exc})"
    return await asyncio.to_thread(
        compose.revoice, reply, water, generate=STATE.fast or STATE.generate, memory=mem)


async def _spawn_background(description: str) -> str:
    """Kick off a long delegation in the background and return immediately. The result
    is brought back later by _run_background, through the proactive channels."""
    bg = STATE.background.add(description)
    asyncio.create_task(_run_background(bg.id, description))
    print(f"[bg] {bg.id} started: {description[:60]}", flush=True)
    return (f"started working on it in the background (id {bg.id}), tell them you're "
            f"on it and will come back with what you find")


async def _run_background(bg_id: str, description: str) -> None:
    """Run a background delegation to completion, then deliver the result later, as
    an unbidden line + native banner, like a kept promise (bypasses the restlessness
    floor, since the person asked for this)."""
    sub_native = native_tools.NativeTools(
        STATE.reminders, goals=STATE.goals,
        planner_generate=STATE.fast or STATE.generate, journal=STATE.journal,
        allow_delegate=False)
    try:
        orch = await subagents.orchestrate(
            _with_ways(description), STATE.fast or STATE.generate, mcp=STATE.mcp,
            native=sub_native, proposer=_proposer("background"))
        result = (orch.result or "").strip()
    except Exception as exc:  # noqa: BLE001
        STATE.background.finish(bg_id, f"(failed: {exc})", ok=False)
        print(f"[bg] {bg_id} failed: {exc}", flush=True)
        return
    STATE.background.finish(bg_id, result or "(found nothing)", ok=bool(result))
    if not result:
        return
    water = proactive.derive_state(STATE.minutes_since_user())
    voiced = await asyncio.to_thread(
        compose.revoice,
        f"You asked me to look into this: {description}\n\nHere is what I found: {result}",
        water, generate=STATE.fast or STATE.generate)
    await _deliver_proactive(voiced, water, why=unbidden.Why.BACKGROUND)
    print(f"[bg] {bg_id} delivered", flush=True)


def _with_ways(task: str, cue: Optional[str] = None) -> str:
    """A specialist's task, with how the person likes that kind of thing done.

    Specialists are handed a task and nothing about the person; before this, a
    draft for a goal or a job ignored everything they had taught the Keeper."""
    used = STATE.ways.for_task(cue or task)
    if not used:
        return task
    STATE.ways.mark_used(used)
    return f"{task}\n\n{ways_mod.as_guidance(used)}"


async def _learn_from_edit(p: proposals_mod.Proposal) -> None:
    """After an edited proposal ran: ask what the edit says about how they like
    this done, keep the answer as a way, and show it on the card. Off the approval
    path entirely: the approval stands whatever the learner makes of it."""
    if p.original_args is None:
        return
    # The full model, not the fast one: edits are rare, and on the fast model the
    # live demo got "removed punctuation in the sign-off" for an edit that only
    # lowercased the note, a rule that would then shape every later draft.
    way = await asyncio.to_thread(
        ways_mod.learn_from_edit, p.action, p.original_args, p.args,
        STATE.generate, STATE.ways)
    if way is None:
        return
    STATE.proposals.note_learned(p, way.line())
    print(f"[ways] learned from an edit: {way.line()}", flush=True)
    await _push_outcome(p)


async def _run_job(job: jobs_mod.Job) -> None:
    """Run one scheduled job and bring back what it found, like spawn_task does,
    through the same delivery path, so the dial and approvals govern it too. A job
    may PROPOSE (send the summary to someone, say) but never act unasked."""
    print(f"[job] {job.id} running: {job.task[:60]}", flush=True)
    try:
        if job.kind == "brief":
            text = await _brief()
        else:
            sub_native = native_tools.NativeTools(
                STATE.reminders, goals=STATE.goals,
                planner_generate=STATE.fast or STATE.generate, journal=STATE.journal,
                allow_delegate=False)
            orch = await subagents.orchestrate(
                _with_ways(job.task), STATE.fast or STATE.generate, mcp=STATE.mcp,
                native=sub_native, proposer=_proposer(f"job:{job.id}"))
            found = (orch.result or "").strip()
            text = ""
            if found:
                water = proactive.derive_state(STATE.minutes_since_user())
                text = await asyncio.to_thread(
                    compose.revoice,
                    f"You asked me to do this on a schedule: {job.task}\n\n"
                    f"Here is what I found: {found}",
                    water, generate=STATE.fast or STATE.generate)
    except Exception as exc:  # noqa: BLE001 - a failed job must not kill the loop
        STATE.jobs.finish(job, f"{type(exc).__name__}: {exc}", ok=False)
        print(f"[job] {job.id} failed: {exc}", flush=True)
        return
    STATE.jobs.finish(job, text or "(found nothing)", ok=bool(text))
    if text:
        await _deliver_proactive(text, why=unbidden.Why.JOB,
                                 counts_toward_day=False)
        print(f"[job] {job.id} delivered", flush=True)


def _has_google() -> bool:
    return STATE.mcp is not None and any(
        t["function"]["name"].startswith("google__") for t in STATE.mcp.openai_tools())


def _brief_ledger(now: Optional[float] = None) -> list:
    """The part of the brief the Keeper already KNOWS, read straight from its own
    stores. Code, not a model: these are commitments, and a commitment summarised
    by a model is one that can be summarised wrong."""
    now = now or time.time()
    end_of_day = datetime.fromtimestamp(now).replace(
        hour=23, minute=59, second=59).timestamp()
    lines = []
    pending = STATE.proposals.pending(now)
    if pending:
        what = "; ".join(proposals_mod.render(p.action, p.args).splitlines()[0]
                         for p in pending[:3])
        lines.append(f"Waiting on your yes: {len(pending)} ({what})")
    today = [r for r in STATE.reminders.pending() if r.due_at <= end_of_day]
    if today:
        lines.append("Holding for today: " + "; ".join(
            f"{r.text} ({datetime.fromtimestamp(r.due_at).strftime('%H:%M')})"
            for r in today[:5]))
    for g in STATE.goals.active()[:3]:
        nxt = g.next_step()
        done, total = g.progress()
        if nxt is not None:
            lines.append(f"Tending: {g.title} ({done}/{total}), next: {nxt.text}")
    return lines


async def _brief() -> str:
    """The morning brief: what today holds in their calendar and inbox, read by
    the archivist and put in the Keeper's voice, then the ledger, plainly.

    Two halves on purpose. The day ahead is prose worth voicing. The ledger is a
    list of things they committed to or must decide, and those stay as written."""
    day = ""
    if _has_google():
        found = await subagents.run(
            subagents.PROFILES["archivist"],
            _with_ways(
                "Read today's calendar events and the unread mail from the last "
                "day. Report today's events with their times, and only the "
                "messages that look like they need a reply or an action from them, "
                "with who sent each. Skip newsletters and notifications. Plainly, "
                "no preamble.", cue="morning brief calendar mail"),
            mcp=STATE.mcp, native=None)
        if found.result.strip():
            water = proactive.derive_state(STATE.minutes_since_user())
            day = await asyncio.to_thread(
                compose.revoice, found.result, water,
                generate=STATE.fast or STATE.generate)
    ledger = _brief_ledger()
    if not day and not ledger:
        return ""
    return "\n\n".join(part for part in (day.strip(), "\n".join(ledger)) if part)


async def _maybe_drift_note() -> None:
    """Idle background reflection (never sent to the person). Real-time capped so a
    sped-up demo clock can't flood the store with near-duplicate insights."""
    if within_reach_floor(STATE.last_drift_at, time.time(), MIN_REAL_DRIFT_GAP_S):
        return
    try:
        r = await asyncio.to_thread(
            drift.maybe_drift, STATE.store, STATE.fast or STATE.generate,
            STATE.reflections, last_drift_at=STATE.last_drift_at,
            config=STATE.drift_config, embed=STATE.embed)
        if r is not None:
            STATE.last_drift_at = time.time()
            print(f"[drift] reflected: {r.text[:80]}...", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[drift] error: {exc}", flush=True)


def _resolve_turn(msg: str, mem: str, water: str) -> tuple:
    """Make "turn" reachable ONLY through the store, in both directions.

    persona.py calls it the rarest register: the ice going out, and says to
    spend it almost never. Chosen from the message alone it fired on "hello",
    because a single sentence cannot contain a reversal: "the ice is going out" is
    a comparison between two points in time, and only the store holds both. So:

      DEMOTE  the classifier saying "turn" is not evidence of anything. Without a
              recorded reversal behind it, fall back to tidal. This is what stops
              a greeting from being met as a season changing.
      PROMOTE a person moving (tidal) who is talking about something that really
              did reverse gets the register the persona describes.

    Never touches frozen: someone in the cold is not told their season has turned
    because the store remembers better days.
    """
    if water == "frozen":
        return water, ""
    earned = _warming_behind(msg, mem)
    if earned is not None:
        note = f"turn earned by a recorded reversal: {earned.text}"
        print(f"[register] turn earned by: {earned.text[:60]!r}", flush=True)
        return "turn", note
    if water == "turn":
        note = ("turn not earned: nothing the store remembers changed, "
                "so it falls back to tidal")
        print("[register] turn not earned by the store, falling back to tidal",
              flush=True)
        return "tidal", note
    return water, ""


def _warming_behind(msg: str, mem: str):
    """The recorded reversal this message is actually about, or None.

    Two conditions, and the second is the one that took a live failure to learn.
    The warming must have survived recall's relevance gate into `mem`, AND the
    person's own words must overlap it. Recall alone is far too loose: in a small
    store it surfaces almost anything for a warm cue, so "i had a good day today"
    was promoted to turn on the strength of a painting fact it never mentioned.
    """
    if STATE.store is None:
        return None
    try:
        warmings = STATE.store.recent_warmings()
    except Exception as exc:  # noqa: BLE001 - a register is never worth a 500
        print(f"[register] warming lookup failed: {type(exc).__name__}: {exc}",
              flush=True)
        return None
    cue_tokens = memory._tokens(msg)
    if not cue_tokens:
        return None
    for fact in warmings:
        if not fact.text or fact.text not in mem:
            continue                      # recall did not judge it relevant
        if cue_tokens & memory._tokens(fact.text):
            return fact                   # and they are talking about it
    return None


async def _distill_async(user_msg: str, reply: str) -> None:
    convo = [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": reply}]
    await asyncio.to_thread(
        memory.distill, convo, STATE.fast or STATE.generate, STATE.store, STATE.embed)
    # Under memory pressure, compress the low-value tail (MemGPT). Cheap no-op when
    # the store is under budget; runs off the response path.
    try:
        made = await asyncio.to_thread(
            memory.consolidate, STATE.store, STATE.fast or STATE.generate,
            embed=STATE.embed)
        if made:
            print(f"[memory] consolidated {len(made)} cluster(s)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] consolidate error: {exc}", flush=True)


@app.get("/events")
async def events(req: Request):
    q: asyncio.Queue = asyncio.Queue()
    STATE.listeners.add(q)
    # A reconnecting EventSource sends back the id of the last line it saw. Whatever
    # was said after it went to a page that was not listening: say it again now.
    # Registered as a listener BEFORE reading the log, so nothing said after this
    # point can fall into the gap. The cost is the opposite overlap: a line already
    # recorded whose push lands a moment later is in both, and the stream skips it.
    since = unbidden.parse_last_event_id(req.headers.get("last-event-id"))
    missed = STATE.unbidden.after(since) if since is not None else []

    async def stream():
        try:
            # greet the stream so clients know it's open
            yield ": connected\n\n"
            replayed = 0
            for line in missed:
                replayed = line.seq
                yield sse_frame(_line_payload(line))
            while True:
                payload = await q.get()
                frame = sse_frame(payload)
                # The queue may hold a line the replay already sent.
                if replayed and _payload_seq(payload) <= replayed:
                    continue
                yield frame
        finally:
            STATE.listeners.discard(q)

    return StreamingResponse(stream(), media_type="text/event-stream")


def _line_payload(line: unbidden.Line) -> str:
    """An unbidden line as the web channel would have sent it live."""
    payload = {"role": "assistant", "content": line.text, "kind": "proactive",
               "ts": line.ts, "id": line.seq, "session": line.session,
               "why": line.why}
    if line.why == unbidden.Why.PROPOSAL and line.ref:
        p = STATE.proposals.get(line.ref)
        if p is not None:
            # Replayed with its CURRENT status: one decided from the phone while
            # the page was away must not come back with live buttons on it.
            payload.update(kind="proposal", proposal=p.public())
    return json.dumps(payload)


def _payload_seq(payload: str) -> int:
    try:
        return int(json.loads(payload).get("id") or 0)
    except (ValueError, TypeError, AttributeError):
        return 0


def sse_frame(payload: str) -> str:
    """One SSE frame. A line carrying a sequence number goes out with it as the
    event id, which is what the browser hands back as Last-Event-ID."""
    seq = _payload_seq(payload)
    head = f"id: {seq}\n" if seq else ""
    return f"{head}data: {payload}\n\n"


@app.get("/unbidden")
async def unbidden_lines(after: int = 0, limit: int = 50):
    """What the Keeper said on its own, and which part of it spoke. The answer to
    "what did you do while I was away", without having to scroll for it."""
    return {"lines": [{"seq": ln.seq, "text": ln.text, "why": ln.why,
                       "state": ln.state, "session": ln.session, "ts": ln.ts}
                      for ln in STATE.unbidden.after(after, limit=max(1, limit))]}


@app.get("/state")
async def state():
    # Report the virtual clock the loop actually runs on, so a sped-up demo
    # visibly shows the battery draining.
    ps = STATE.proactive_state()
    ms = ps.minutes_since_user
    en = energy.compute_energy(ms)
    score = energy.base_score(en, STATE.recent_msg_count())
    now = time.time()
    # The band the loop actually rolls against, which the dial reshapes. Reporting
    # the defaults while the dial was at quiet would show a number the Keeper is
    # not using: the same lie as a channel advertised that cannot fire.
    shaped = STATE.dial.shape(STATE.config)

    pres = await asyncio.to_thread(sensors.read)
    pending = STATE.reminders.pending()
    next_rem = pending[0] if pending else None

    return {
        "minutes_since_user_virtual": ms,
        "minutes_since_user_real": STATE.minutes_since_user(),
        "minutes_since_proactive": ps.minutes_since_proactive,
        "recent_msg_count": STATE.recent_msg_count(),
        "energy": round(en, 3),
        "base_score": round(score, 3),
        "speak_probability": (round(energy.speak_probability(
            score, p_min=shaped.p_min, p_max=shaped.p_max), 3)
            if STATE.dial.level.initiative else 0.0),
        "level": STATE.dial.name,
        "daily_max": shaped.daily_max,
        "facts_kept": sum(1 for f in STATE.store.facts if f.active),
        "insights_kept": sum(1 for f in STATE.store.facts
                             if f.kind == "insight" and f.active),
        "changes_tracked": len(STATE.store.changes()),
        "reminders_held": len(pending),
        "reminders_recurring": sum(1 for r in pending if r.repeat),
        "goals": [
            {"title": g.title, "id": g.id,
             "done": g.progress()[0], "total": g.progress()[1],
             "next_step": (g.next_step().text if g.next_step() else None),
             "next_actor": (g.next_step().actor if g.next_step() else None)}
            for g in STATE.goals.active()],
        "next_reminder": (
            {"text": next_rem.text, "due_at": next_rem.due_at,
             "repeat": next_rem.repeat} if next_rem else None),
        "reflections": len(STATE.reflections.items),
        "latest_reflection": (STATE.reflections.latest().text
                              if STATE.reflections.latest() else None),
        # the house: presence + the routines watching it
        "presence": {
            "idle_seconds": pres.idle_seconds,
            "screen_locked": pres.screen_locked,
            "frontmost_app": pres.frontmost_app,
            "summary": pres.to_context_line(),
        },
        "house": STATE.routines_engine.status(now),
        "working_on": [{"id": t.id, "description": t.description}
                       for t in STATE.background.running()],
        "ways": [{"id": w.id, "line": w.line(), "source": w.source, "uses": w.uses}
                 for w in STATE.ways.active()],
        "jobs": [{"id": j.id, "task": j.task, "kind": j.kind, "next_at": j.next_at,
                  "repeat": j.repeat, "last_status": j.last_status}
                 for j in STATE.jobs.active()],
        # Shown once Google is connected and no brief exists: the "offered once"
        # is the page's to make, never the Keeper's voice, which may not offer.
        "brief_hint": _has_google() and not STATE.jobs.has_brief(),
        "proposals": {
            "pending": [{"id": p.id, "action": p.action, "origin": p.origin,
                         "created": p.created}
                        for p in STATE.proposals.pending()],
            "behind_approval": ([t["name"] for t in STATE.mcp.proposable()]
                                if STATE.mcp else []),
            "inbox": ("telegram" if STATE.inbox is not None
                      and STATE.inbox.stopped_because is None
                      else "stopped" if STATE.inbox is not None else "off"),
        },
        "embedder": "openai" if STATE.embed is not None else "keyword",
        "speed": STATE.config.speed,
        "backend": "openai" if STATE.fast is not None else "stub",
        "listeners": len(STATE.listeners),
        "considered": STATE.last_scan,
        "channels": STATE.delivery.names() if STATE.delivery else ["web"],
        # Tools the Keeper can actually reach right now, and any server that
        # promised something it did not deliver. A half-broken MCP server used to
        # be visible only in the startup log, so a capability could go missing
        # mid-demo and read as the Keeper simply choosing not to use it.
        "tools": {
            "count": len(STATE.mcp.openai_tools()) if STATE.mcp else 0,
            "servers": STATE.mcp.health if STATE.mcp else [],
            "problems": STATE.mcp.summary() if STATE.mcp else "",
        },
    }


class ConfigIn(BaseModel):
    speed: Optional[float] = None
    cooldown_min: Optional[float] = None
    level: Optional[str] = None      # the proactivity dial: off/quiet/usual/eager


@app.post("/config")
async def set_config(body: ConfigIn):
    if body.speed is not None:
        STATE.config.speed = max(1.0, body.speed)
        STATE.drift_config.speed = STATE.config.speed   # compress drift clock too
    if body.cooldown_min is not None:
        STATE.config.cooldown_min = max(0.0, body.cooldown_min)
    error = None
    if body.level is not None and STATE.dial.set(body.level) is None:
        error = f"unknown level {body.level!r}: one of {', '.join(dial_mod.LEVELS)}"
    elif body.level is not None:
        print(f"[dial] set to {STATE.dial.name}", flush=True)
    if STATE.wake is not None:
        STATE.wake.set()   # re-tick now with the new settings
    out = {"speed": STATE.config.speed, "cooldown_min": STATE.config.cooldown_min,
           "level": STATE.dial.name}
    if error:
        out["error"] = error
    return out


# What the calendar holds for the coming week, for the dashboard. Cached: the
# dashboard refreshes every few seconds, and the calendar does not need asking that
# often, or at the cost of a call to Google each time.
UPCOMING_TTL_S = 300.0
_upcoming: dict = {"at": 0.0, "events": [], "connected": False}


@app.get("/upcoming")
async def upcoming():
    now = time.time()
    if now - _upcoming["at"] < UPCOMING_TTL_S:
        return {k: _upcoming[k] for k in ("events", "connected")}
    connected = _has_google()
    events: list = []
    if connected:
        out = await STATE.mcp.call("google__get_events", sources.expand_time(
            {"time_min": "${NOW}", "time_max": "${NOW+7d}", "max_results": 8}))
        if out.lstrip().startswith("("):
            # A refused call (an expired sign-in, say) is shown as such, not as an
            # empty week: "nothing coming up" would be a claim about the calendar.
            connected = False
        else:
            events = sources.calendar_events(out, max_items=8)
    _upcoming.update(at=now, events=events, connected=connected)
    return {"events": events, "connected": connected}


@app.get("/proposals")
async def list_proposals():
    return {"pending": [p.public() for p in STATE.proposals.pending()],
            "recent": [p.public() for p in reversed(STATE.proposals.recent())]}


class DecideIn(BaseModel):
    id: str
    decision: str     # "approve" | "decline"
    args: Optional[dict] = None   # the person's edits to the draft, if any


@app.post("/proposals/decide")
async def decide_proposal(body: DecideIn, req: Request):
    """A JSON body, deliberately: a cross-site form cannot send one without a CORS
    preflight, which nothing here answers. _same_origin is the second lock."""
    if not _same_origin(req):
        return {"ok": False, "note": "refused: not from this page"}
    if body.decision not in ("approve", "decline"):
        return {"ok": False, "note": "decision must be approve or decline"}
    note = await _decide(body.id, body.decision == "approve", "web",
                         edited=body.args if body.decision == "approve" else None)
    p = STATE.proposals.get(body.id)
    return {"ok": p is not None, "note": note,
            "proposal": p.public() if p is not None else None}


class ForgetIn(BaseModel):
    id: str


@app.post("/ways/forget")
async def forget_way(body: ForgetIn, req: Request):
    if not _same_origin(req):
        return {"ok": False, "note": "refused: not from this page"}
    w = STATE.ways.forget(body.id)
    return {"ok": w is not None}


@app.get("/sessions")
async def list_sessions():
    return {
        "sessions": [{"key": s.key, "title": s.title or "…",
                      "count": len(s.messages), "updated_at": s.updated}
                     for s in STATE.sessions.all()],
        "active": STATE.current_key,
    }


@app.get("/sessions/{key}/messages")
async def session_messages(key: str):
    s = STATE.sessions.get(key)
    if s is None:
        return {"messages": []}
    return {"messages": [{"role": m.role, "content": m.content, "ts": m.ts,
                          "kind": m.kind, "ref": m.ref}
                         for m in s.messages]}


class SelectIn(BaseModel):
    session_id: str


@app.post("/select")
async def select_session(body: SelectIn):
    if STATE.sessions.get(body.session_id) is not None:
        STATE.current_key = body.session_id
    return {"ok": True, "active": STATE.current_key}


@app.post("/new")
async def new_conversation():
    STATE.current_key = None      # next message starts a fresh conversation
    return {"ok": True}


# --------------------------------------------------------------------------- #
# A2A: the Keeper as an agent other agents can discover and consult.
# --------------------------------------------------------------------------- #

@app.get(a2a.WELL_KNOWN)
async def agent_card(req: Request):
    """The Keeper's A2A Agent Card, how a peer agent discovers what it can do."""
    return a2a.build_agent_card(str(req.base_url))


@app.post("/a2a")
async def a2a_endpoint(req: Request):
    """Handle an A2A message/send: a peer sends a message, the Keeper answers.

    Closed unless KEEPER_A2A_TOKEN is set. Answering means recalling memory, so an
    open endpoint lets any local process ask the Keeper what it knows about the
    person and get it back in prose. TrustedHostMiddleware is not that control: it
    validates the Host header, not who is calling.
    """
    token = a2a.auth_token()
    if not token:
        return a2a.rpc_error(None, -32001,
                             "a2a is disabled: set KEEPER_A2A_TOKEN to enable it")
    presented = (req.headers.get("authorization") or "")
    if not secrets.compare_digest(presented.strip(), f"Bearer {token}"):
        return a2a.rpc_error(None, -32002, "unauthorized")
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        return a2a.rpc_error(None, -32700, "parse error")
    req_id = body.get("id")
    if body.get("method") != "message/send":
        return a2a.rpc_error(req_id, -32601,
                             f"unsupported method: {body.get('method')}")
    text = a2a.text_of((body.get("params") or {}).get("message") or {})
    if not text:
        return a2a.rpc_error(req_id, -32602, "empty message")
    reply = await _answer_as_keeper(text)
    return a2a.rpc_result(req_id, a2a.make_message(reply, role="agent"))


@app.get("/trace")
async def trace():
    """The last turns' decisions, memory injected, register, tools fired, reply."""
    return list(STATE.traces)


@app.get("/trace-view")
async def trace_view():
    return FileResponse(STATIC_DIR / "trace.html")


@app.get("/dashboard")
async def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/info")
async def info():
    return {"name": "The Keeper",
            "endpoints": ["/chat", "/events", "/state", "/config"],
            "facts_kept": len(STATE.store.facts)}
