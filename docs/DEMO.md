# Demo script

One story, not a feature list: a person who stopped painting, starts again, and is
tended. Every feature earns its place by belonging to that arc.

**Run on the host** (`./run.sh`), not in Docker — the container is Linux, so it
loses presence sensing and the native banner, which are two of the beats below.
Start from a clean store:

```bash
mv memory_store memory_store_backup_$(date +%s) && mkdir memory_store
./run.sh
```

Set `KEEPER_FEEDS` in `backend/.env` first, and `DISCORD_WEBHOOK_URL` if you want
the notification beat visible on screen rather than in the corner of your desktop.

Two windows: **chat** at `/`, and **dashboard** at `/dashboard`. Keep the trace
(`/trace-view`) in a third tab — it is the payoff, not the opener.

---

## Act 1 — It learns you  *(~40s)*

```
i stopped painting in march
```
> Dashboard: **facts kept** goes to 1.

```
actually i started painting again
```
> Dashboard: **changes tracked** goes to 1. The old fact is closed, not deleted.

```
how's my painting going?
```
> The reply is in the **turn** register — the ice going out. Terminal shows
> `[register] turn earned by: 'Started painting again.'`

Then, to show it cannot be talked into it:

```
everything is finally turning around for me
```
> The classifier reads that as **turn** — the words are right there. The reply comes
> back **tidal**, and the terminal says
> `turn not earned by the store — falling back to tidal`.
>
> **The person said the words of a turn and the Keeper declined them**, because
> nothing it remembers actually changed. Contrast with the line before, where the
> same register was granted on the strength of a recorded reversal. Saying it is
> not evidence; the store is.

## Act 2 — It works on things with you  *(~60s)*

```
help me get back to painting, work on it with me
```
> Dashboard: a **goal** appears with 5 steps and `next_actor`. Some steps are
> labelled for the Keeper, some for the person — it does not hand you a to-do list
> and walk away.

```
find a typical price for a beginner watercolour set and work out the cost per week over a year
```
> Trace: **search → fetch → run_python**. A real price, and a *computed* figure —
> not a guessed one.

```
go compare watercolour and gouache for a beginner and get back to me
```
> Immediate acknowledgement. Dashboard: **working on** populates. It has handed
> the job to specialists and will return unprompted.

## Act 3 — It holds things  *(~20s)*

```
remind me to gesso the canvas tomorrow at 9am
```
> Dashboard: **next reminder**. Natural language became an ISO datetime.

```
keep a note: the gallery show is in July
```
> Its one write. Append-only.

## Act 4 — It has judgment about the world  *(~40s)*

Terminal, from the feed poll:

```
[sources] pending 'Denizens of a Crowded City Populate Erin Milez's Dense Paintings'
          relevance=0.9 because='Is a painter.'
```

Show the whole scored batch — this is the strongest beat in the demo, because you
watch it **reject** things:

| item | score | |
|---|---|---|
| dense paintings | 0.9 | interrupt |
| art installation | 0.6–0.7 | mention only |
| Audubon photography awards | 0.0 | silent |
| 40 filmmakers | 0.0 | silent |

Two thresholds, not one: *worth mentioning* changes what it says when it was going
to speak anyway; only *worth interrupting* lets the outside world make it speak.

## Act 5 — Then close the browser  *(~60s)*

```bash
curl -s -X POST localhost:8790/config -H 'content-type: application/json' \
  -d '{"speed":600,"cooldown_min":5}'
```

**Close the browser.** One crank makes everything autonomous fire at once:

- a **native banner** arrives with the Keeper's face — telling you about the
  painting show, *because* it knows you paint
- `[goal] … EXECUTED via researcher` — it takes its own step, with tools
- `[goal] … nudged` — it invites you to take yours
- the background comparison returns on its own
- Discord/Telegram get the same line, if configured

Reset when you are done:
```bash
curl -s -X POST localhost:8790/config -H 'content-type: application/json' -d '{"speed":1}'
```

## Act 6 — Why it said any of it  *(~40s)*

Open **`/trace-view`**. Per turn: memory injected, the register and why, every tool
call with arguments and results, and the reply.

This is the credibility shot. Not "it said something apt" — *here is the fact that
made it worth saying.*

Finish on the **dashboard**, now fully populated: energy and speak-probability, facts
and changes and insights, the goal's progress, the house (idle, screen, focused app),
reminders held.

## Act 7 — It is an agent, not just an app  *(~25s)*

Open **`localhost:8790/.well-known/agent.json`** in a tab.
> An **Agent Card**: name, skills, and an endpoint. This is the server half of
> A2A — other agents can discover the Keeper and call it.

Then, in chat:

```
consult the agent at http://localhost:8790 and ask what it can help with
```
> The **trace** shows the client half:
> ```
> tool : consult_peer
> args : {"url": "http://localhost:8790", "task": "What can you help with?"}
> reply: [The Keeper] The tide brings many things...
> ```
> It discovers an agent by URL, sends it a task, and reads the reply. Here it is
> pointed at itself for want of a second agent to hand — say so, and note the URL
> could be anyone's.

---

## Timing

About **5 minutes**. Acts 1 and 5 are the ones that cannot be cut: memory that
changes, and a machine that speaks first with the browser shut.

## Rehearsal notes

- Act 5 needs the **60s real-time floor** to pass between outreaches — that gap is
  deliberate and cannot be cranked away. Plan for a pause, or cut around it.
- The register is **inherited** on neutral messages, so a `tidal` reading after a
  question is continuity, not a misread. Say so if it shows.
- `hello` will NOT show the demote line: the classifier reads a greeting as neutral,
  so it never claims `turn` and there is nothing to demote. You need a message that
  genuinely sounds like a turn but that the store cannot back, which is why Act 1
  uses "everything is finally turning around for me".
- Feed items are deduplicated permanently. To re-demo the same item,
  `rm memory_store/sources_seen.jsonl`.
