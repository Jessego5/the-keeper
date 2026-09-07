"""proactive_bench.py — hold the interruption policy to evidence, not taste.

The register classifier was chosen with data: mood_bench.py compares four methods
on a held-out split and reports accuracy and latency. The policy that decides
whether to INTERRUPT SOMEONE was chosen by intuition — a dozen constants, none
measured:

    alpha 0.50 / beta 0.35 / gamma 0.15     the three decay horizons
    w_hunger 0.7 / w_context 0.3            hunger against recent context
    p_min 0.05 / p_max 0.45                 the coin's bounds
    cooldown_min 600                        silence after an ignored outreach
    daily_max 12                            the day's ceiling

That asymmetry is backwards: a wrong threshold in the classifier gives a reply the
wrong tone, while a wrong constant here pesters a person all day. (the reference agent tunes
the same kind of numbers by hand too, so this is not a gap peculiar to us — it is
just one worth closing.)

Nothing here asserts a policy is CORRECT; there is no ground truth for how often a
companion should speak. It reports what a configuration actually does across
simulated days, so a change is a decision with numbers attached rather than a
feeling.

Read the numbers knowing what the simulation assumes. Presence is held AWAKE
throughout, so the screen-lock gate never fires and the "asleep" column is an upper
bound on night-time outreaches rather than a prediction. The person's messages
arrive on a jittered fixed rhythm, which is a cartoon of real use. What the
simulation is good for is COMPARING configurations under identical conditions, not
forecasting a real week.

Three things it said on first run, none of which were obvious beforehand:

  * the cooldown is doing almost all the work. Removing it takes the shipped
    policy from ~2 outreaches a day to 8-10, with bursts and night-time lines.
  * p_max barely matters next to it. 0.20 against 0.70 moves the rate from 1.4 to
    2.0 a day, because a ten-hour refractory dominates the coin.
  * the daily ceiling never binds at shipped settings — even a ceiling of 4 is
    identical to no ceiling. It is a backstop against a pathological case such as a
    busy feed, not a shaper of ordinary behaviour, and it should be described that
    way rather than credited with restraint the cooldown is providing.

Since every decision now carries a proactive.Reason code, the run also reports
which gate ended each tick, which turns the first finding above from an inference
into a count. Pooled across all three profiles at shipped settings:

    cooldown  78.2%     dice  16.0%     spoke  5.8%

and daily_max, composer_silent and repeat do not appear at all. The ceiling really
never binds; the coin is a minor character; the refractory period IS the policy.
(The two composer outcomes are absent because the simulation runs on the offline
stub, which never declines and never repeats itself. Those gates are real, they
just cannot be exercised without a live model.)

    python backend/proactive_bench.py
    python backend/proactive_bench.py --days 14 --runs 40
"""

from __future__ import annotations

import argparse
import collections
import itertools
import random
import statistics
from dataclasses import dataclass, field, replace

import compose
import proactive
import sensors

DAY_MIN = 24 * 60.0


@dataclass
class Profile:
    """A way of using the Keeper, as minutes-between-messages and waking hours."""

    name: str
    gap_min: float          # typical minutes between the person's messages
    awake_from: float = 8.0
    awake_to: float = 23.0


PROFILES = [
    Profile("daily user", gap_min=6 * 60),
    Profile("occasional", gap_min=36 * 60),
    Profile("gone quiet", gap_min=14 * 24 * 60),
]


@dataclass
class Outcome:
    per_day: float          # outreaches per day
    longest_silence_h: float
    burst_1h: float         # times it spoke twice inside an hour
    asleep: float           # outreaches while they were asleep
    spoke_total: int
    # Which gate actually held the silence, counted by proactive.Reason code.
    # Without this the benchmark can say HOW OFTEN it stayed quiet but not why,
    # and "why" is the half that tells you which knob is load-bearing.
    codes: dict = field(default_factory=dict)


def _run(cfg: proactive.ProactiveConfig, prof: Profile, days: float,
         seed: int) -> Outcome:
    """Walk virtual minutes, ticking as the scheduler would."""
    rng = random.Random(seed)
    awake = sensors.Presence(idle_seconds=120.0, screen_locked=False,
                             frontmost_app="bench")
    t = 0.0
    horizon = days * DAY_MIN
    last_user = 0.0
    last_spoke: float | None = None
    spoke_at: list[float] = []
    codes: collections.Counter = collections.Counter()
    today_count, today_start = 0, 0.0

    while t < horizon:
        if t - today_start >= DAY_MIN:
            today_start, today_count = t, 0
        # The person speaks on their own rhythm, jittered so a run is not a metronome.
        if (t - last_user) >= rng.uniform(0.6, 1.4) * prof.gap_min:
            last_user = t
        state = proactive.ProactiveState(
            minutes_since_user=t - last_user,
            recent_msg_count=1 if (t - last_user) < 240 else 0,
            minutes_since_proactive=None if last_spoke is None else t - last_spoke,
            proactive_today=today_count)
        d = proactive.tick(state, generate=compose.stub_generator, config=cfg,
                           presence=awake, rng=rng)
        codes[d.code] += 1
        if d.spoke:
            spoke_at.append(t)
            last_spoke, today_count = t, today_count + 1
        t += max(1.0, d.wait_next_s / 60.0)

    # pairwise, not zip(xs, xs[1:], strict=True): those lengths differ by one by
    # design, and strict= is for iterables that should match.
    gaps = [b - a for a, b in itertools.pairwise(spoke_at)]
    hour_of = [(x % DAY_MIN) / 60.0 for x in spoke_at]
    return Outcome(
        per_day=len(spoke_at) / days,
        longest_silence_h=(max(gaps) / 60.0) if gaps else days * 24,
        burst_1h=sum(1 for g in gaps if g < 60) / days,
        asleep=sum(1 for h in hour_of
                   if h < prof.awake_from or h > prof.awake_to) / days,
        spoke_total=len(spoke_at),
        codes=dict(codes))


def evaluate(cfg: proactive.ProactiveConfig, days: float = 7.0,
             runs: int = 20) -> dict:
    """Average a configuration over several seeds and every usage profile."""
    out = {}
    for prof in PROFILES:
        rs = [_run(cfg, prof, days, seed) for seed in range(runs)]
        out[prof.name] = Outcome(
            per_day=statistics.mean(r.per_day for r in rs),
            longest_silence_h=statistics.mean(r.longest_silence_h for r in rs),
            burst_1h=statistics.mean(r.burst_1h for r in rs),
            asleep=statistics.mean(r.asleep for r in rs),
            spoke_total=int(statistics.mean(r.spoke_total for r in rs)),
            codes=dict(sum((collections.Counter(r.codes) for r in rs),
                           collections.Counter())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    base = proactive.ProactiveConfig()
    variants = [
        ("shipped", base),
        ("no daily ceiling", replace(base, daily_max=0)),
        ("no cooldown", replace(base, cooldown_min=0.0)),
        ("eager  p_max 0.70", replace(base, p_max=0.70)),
        ("reticent p_max 0.20", replace(base, p_max=0.20)),
        ("ceiling 4/day", replace(base, daily_max=4)),
    ]

    print(f"\n{'='*74}")
    print(f"  PROACTIVE POLICY   {args.days:.0f} simulated days x {args.runs} seeds")
    print(f"{'='*74}")
    print(f"  {'config':22} {'profile':12} {'/day':>6} {'quiet h':>8} "
          f"{'burst':>6} {'asleep':>7}")
    print("  " + "-" * 70)
    for name, cfg in variants:
        res = evaluate(cfg, days=args.days, runs=args.runs)
        for i, (prof, o) in enumerate(res.items()):
            label = name if i == 0 else ""
            print(f"  {label:22} {prof:12} {o.per_day:6.1f} "
                  f"{o.longest_silence_h:8.1f} {o.burst_1h:6.2f} {o.asleep:7.2f}")
        print()
    # The question the first version of this benchmark could not answer. It could
    # report that the Keeper spoke 1.4 times a day and stayed quiet the rest, but
    # not WHICH gate held the silence, which is what tells you where the policy
    # actually lives and which knob is worth turning.
    pooled: collections.Counter = collections.Counter()
    for o in evaluate(base, days=args.days, runs=args.runs).values():
        pooled.update(o.codes)
    total = sum(pooled.values()) or 1
    print("  WHY EACH TICK ENDED   (shipped config, every profile pooled)")
    print("  " + "-" * 70)
    for code, n in pooled.most_common():
        share = 100.0 * n / total
        print(f"  {code:17} {share:5.1f}%  {'#' * int(round(share / 2))}")
    print()
    print("  /day   outreaches per day        quiet h  longest silence, hours")
    print("  burst  second line within an hour, per day")
    print("  asleep outreaches outside waking hours, per day")
    print(f"{'='*74}\n")


if __name__ == "__main__":
    main()
