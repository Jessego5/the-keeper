"""
These are the Tier 1 tests for the policy benchmark (proactive_bench.py). Offline: the
stub generator and a fixed seed, so it runs in a second and asserts the harness, never a
verdict.
"""
import pytest

import proactive
import proactive_bench as pb

pytestmark = pytest.mark.unit


def test_a_run_reports_the_shape_we_measure():
    o = pb._run(proactive.ProactiveConfig(), pb.PROFILES[0], days=3.0, seed=1)
    assert o.per_day >= 0 and o.longest_silence_h > 0
    assert o.burst_1h >= 0 and o.asleep >= 0


def test_a_seed_is_reproducible():
    """A benchmark that moves on its own cannot settle an argument."""
    a = pb._run(proactive.ProactiveConfig(), pb.PROFILES[0], days=3.0, seed=7)
    b = pb._run(proactive.ProactiveConfig(), pb.PROFILES[0], days=3.0, seed=7)
    assert a == b


def test_different_seeds_differ():
    a = pb._run(proactive.ProactiveConfig(), pb.PROFILES[0], days=5.0, seed=1)
    b = pb._run(proactive.ProactiveConfig(), pb.PROFILES[0], days=5.0, seed=2)
    assert a != b, "the roll is not being exercised"


def test_removing_the_cooldown_makes_it_talk_more():
    """The finding the benchmark exists to make checkable: the cooldown, not the
    coin, is what holds the Keeper quiet. If this ever stops being true the policy
    has changed shape and someone should notice."""
    from dataclasses import replace
    base = proactive.ProactiveConfig()
    shipped = pb._run(base, pb.PROFILES[0], days=7.0, seed=3)
    loosed = pb._run(replace(base, cooldown_min=0.0), pb.PROFILES[0],
                     days=7.0, seed=3)
    assert loosed.per_day > shipped.per_day * 2


def test_a_ceiling_of_zero_is_no_ceiling():
    from dataclasses import replace
    base = proactive.ProactiveConfig()
    assert pb._run(replace(base, daily_max=0), pb.PROFILES[0], days=5.0, seed=4) \
        == pb._run(base, pb.PROFILES[0], days=5.0, seed=4), \
        "at shipped settings the ceiling never binds, so removing it changes nothing"


def test_a_hard_ceiling_does_bind_when_the_policy_is_loud():
    """It is a backstop, so prove it actually catches the case it exists for."""
    from dataclasses import replace
    loud = replace(proactive.ProactiveConfig(), cooldown_min=0.0)
    capped = replace(loud, daily_max=2)
    assert pb._run(capped, pb.PROFILES[0], days=7.0, seed=5).per_day \
        < pb._run(loud, pb.PROFILES[0], days=7.0, seed=5).per_day
