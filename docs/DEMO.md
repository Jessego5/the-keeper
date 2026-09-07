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

**Pause about three seconds after "actually i started painting again."** The reply
comes back before the memory is written; distillation lands roughly a second later.
Ask "how is my painting going?" too fast and the reversal is not in the store yet,
so `turn` is not earned and the beat silently becomes an ordinary tidal reply. This
is the one place in the demo where typing quickly loses you the moment.

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
go and research watercolour versus gouache for a beginner, take your time and get back to me later
```
> Immediate acknowledgement. Dashboard: **working on** populates. It has handed
> the job to specialists and will return unprompted.

> The long phrasing is deliberate. The shorter "go compare X and get back to me"
> spawned in only 1 of 3 rehearsal attempts; the rest of the time the model chose
> `delegate`, which answers in the same breath and leaves **working on** empty.
> Naming the delay ("take your time", "later") spawned 4 times out of 4. Both are
> reasonable readings of the request, so this is a script fix, not a bug.

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

Terminal, from the poll:

```
[sources] scanned 13, 11 judged not worth saying
[sources] pending 'Denizens of a Crowded City Populate Erin Milez's Dense Paintings'
          relevance=0.7 because='Is a painter who stopped painting in March.'
```

Show the whole scored batch. This is the strongest beat in the demo, because you
watch it **reject** things. A real run:

| item | source | score | |
|---|---|---|---|
| dense paintings | feed | 0.7-0.9 | speaks |
| Hans Op de Beeck installation | feed | 0.6 | mention only |
| Audubon photography awards | feed | 0.2 | silent |
| 40 filmmakers | feed | 0.0 | silent |
| "Give the Keeper another agent to actually talk to" | **its own repo** | 0.2 | silent |
| "Refuse a push a laptop can already tell is broken" | **its own repo** | 0.0 | silent |

Two things to say over this.

**It is not just reading feeds.** The bottom rows are real commits, pulled through
the same MCP git server it uses in chat. Anything that genuinely pushes can be
watched: a commit log, a folder, a calendar. Only the tool changes; the gate does
not.

**It refuses to tell you about your own work.** Every commit scores near zero, and
that is correct, not a miss: you wrote them an hour ago. Something is only news if
it is news *to this person*. That is the difference between a feed reader and a
companion, and it is the same judgment that keeps the bird photography quiet.

Two thresholds, not one: *worth mentioning* changes what it says when it was going
to speak anyway; only *worth interrupting* lets the outside world make it speak.

The `because` is the fact it turned on, copied from the store. On the silent rows
it is empty, because nothing it keeps is the reason.

## Act 5 — Then close the browser  *(~60s)*

```bash
curl -s -X POST localhost:8790/config -H 'content-type: application/json' \
  -d '{"speed":600,"cooldown_min":5}'
```

The cooldown is measured in the Keeper's own compressed minutes, so **cranking is
not optional**: at `speed 1` with `cooldown_min 600` it will not speak for ten
hours. Cranked, expect a line roughly every **60 to 130 seconds**, never faster,
because the 60s real-time floor cannot be cranked away.

**Close the browser.** One crank makes everything autonomous fire at once:

- a **native banner** arrives with the Keeper's face — telling you about the
  painting show, *because* it knows you paint
- `[goal] … EXECUTED via researcher` — it takes its own step, with tools
- possibly `[goal] … nudged`, inviting you to take yours. **Do not promise this
  one.** A nudge goes through the ordinary proactive composer, which may choose
  silence, and in a full rehearsal it declined every time inside four minutes.
  When it declines it now logs `nudge declined (chose silence)`, which is worth
  showing in its own right: restraint is a designed outcome here, not a failure.
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

Start the peer first, in another terminal:

```bash
.venv/bin/python -m uvicorn scripts.almanac_agent:app --port 8791
```

It is **The Almanac**: a painter's reference with no memory, no voice and no model
behind it, sharing no code with the Keeper's own A2A module. Show its card at
`localhost:8791/.well-known/agent.json` beside the Keeper's, and note it asks for
no auth while the Keeper's requires a token — because answering there means
reading someone's memory.

Then, in chat:

```
consult the agent at http://localhost:8791 and ask what gouache is
```
> The **trace** shows the client half:
> ```
> tool : consult_peer
> args : {"url": "http://localhost:8791", "task": "What is gouache?"}
> reply: [The Almanac] Gouache is watercolour made opaque, usually with added chalk...
> ```
> Two independently written implementations agreeing about a wire format, which is
> the only thing that actually demonstrates a protocol. The Keeper then answers in
> its own voice using something it did not know a moment ago.

Requires the origin to be allowed, since it is loopback:
`KEEPER_A2A_ALLOW=http://localhost:8790,http://localhost:8791`

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
- Feed items are deduplicated permanently, and so are watched commits. To re-demo
  the same batch, `rm memory_store/sources_seen.jsonl`.
- The top item's score moves between **0.7 and 0.9** run to run; the judge is a
  model, not a lookup. At 0.7 it reads *mention*, not *interrupt*. Narrate it as
  "worth saying" rather than promising the word on screen.
- **Watch the dashboard, not the reply, when setting the goal.** Once in a
  rehearsal the Keeper answered "I will hold this goal for you" and called no tool
  at all, leaving the goals tile empty. The reply reads exactly like success, which
  makes it the easiest failure to narrate straight past. It set the goal 3 times out
  of 3 on retry, so if the tile stays empty, simply ask again.
- **Act 1's opening register varies.** "i stopped painting in march" has come back
  both as `tidal` (no signal, so it inherits) and as `frozen` (classified). Both are
  defensible and `frozen` is arguably the truer reading of a stopped practice. Do
  not re-take for it; the thing that must be right is **facts kept 1**.
- The **git watch will not make it speak.** A commit you wrote is not news to you,
  so those rows are there to be rejected. Do not wait for one to become a line.
