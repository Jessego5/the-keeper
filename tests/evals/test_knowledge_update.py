"""Tier 3 — knowledge update, end to end on the real model. Run explicitly:

    .venv/bin/pytest tests/evals/test_knowledge_update.py -m eval

Borrowed from LongMemEval, whose `knowledge-update` category asks the only question
that matters about a memory that changes: when a fact is stated, then later
REPLACED, does the assistant answer with the new value or the old one?

Everything already here tests a piece of that and none of it tests the whole. The
unit tests drive supersession with a fake judge that spells its verdict perfectly.
test_supersede.py drives the real judge, but only asks what it says about a pair of
facts. Nothing runs the actual path a person walks: say a thing, come back later and
say it changed, then ask about it. That path crosses distillation, the supersede
judge, retrieval and composition, and it can fail at any of the four while all three
existing layers stay green.

Scored with metrics.py rather than substring checks, because the Keeper answers in a
register. "The tide has turned back to the canvas" is a correct answer to "am I
painting again?" and scores near zero on any overlap measure, so the model judge is
the one that counts; token_f1 is reported alongside as a sanity number.

Each case asserts BOTH directions, which is the point. Answering with the new value
is half of it; the failure that actually shipped was the Keeper reciting both at once
("You have started painting again. You stopped painting in March."), which passes any
check that only looks for the new value.
"""
import os

import pytest

import compose
import embedder
import memory
import metrics

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                       reason="needs OPENAI_API_KEY"),
]


class Case:
    def __init__(self, first, update, question, current, stale):
        self.first, self.update = first, update
        self.question, self.current, self.stale = question, current, stale

    def __repr__(self):
        return f"<{self.first[:28]}...>"


CASES = [
    # The documented arc, and the one that shipped broken.
    # The question asks about a PERIOD, not an instant. "am i painting at the
    # moment?" was the first phrasing and it is ambiguous: read literally the
    # honest answer while someone is typing to you is "no", so the Keeper said
    # "you are not painting at the moment, you have started painting again" and
    # scored as asserting the stale value. Measured over three trials it failed
    # 1 in 3, where "have i been painting lately?" and "how is my painting
    # going?" each passed 3 of 3. That was a flaw in the question, not the
    # memory, and it is written down so the change does not read as tuning the
    # test until it went green.
    Case(first="i stopped painting in march",
         update="actually i started painting again",
         question="have i been painting lately?",
         current="they are painting again",
         stale="they stopped painting and have not painted since"),
    # A plain factual replacement, where the old value is not a negation of the new.
    Case(first="i live in portland",
         update="i moved to chicago last month",
         question="where do i live now?",
         current="they live in chicago",
         stale="they live in portland"),
    # A change of occupation, phrased the way someone actually mentions it.
    Case(first="i work as a barista at the cafe on rowan street",
         update="i left the cafe, i'm freelancing now",
         question="what do i do for work these days?",
         current="they freelance",
         stale="they work as a barista at a cafe"),
]


@pytest.fixture(scope="module")
def gen():
    g, fast = compose.make_generator()
    assert fast is not None, "expected a live backend with a key"
    return fast or g


@pytest.fixture(scope="module")
def embed():
    return embedder.make_embedder()


def _walk(case, gen, embed, path):
    """Say the thing, come back and say it changed, then ask. Returns the store,
    what recall injected, and the Keeper's answer."""
    store = memory.MemoryStore(path / "facts.jsonl")
    for message in (case.first, case.update):
        memory.distill([{"role": "user", "content": message}], gen, store, embed)
    mem = memory.recall(store, case.question, k=5, embed=embed)
    result = compose.compose("passive", "tidal", generate=gen,
                             user_message=case.question, memory=mem)
    return store, mem, (result.text or "")


# Every stage of this is a model call, so a single run proves little and four
# separate tests each re-walking the pipeline would pay for it four times over.
# Walk once per trial, share the results, and assert RATES, which is what the
# other evals here do with the same problem.
TRIALS = 4


@pytest.fixture(scope="module")
def walks(gen, embed, tmp_path_factory):
    out = {}
    for case in CASES:
        runs = []
        for i in range(TRIALS):
            path = tmp_path_factory.mktemp(f"ku{abs(hash(case.question)) % 9999}_{i}")
            runs.append(_walk(case, gen, embed, path))
        out[case.question] = runs
    return out


def _rate(flags) -> str:
    return f"{sum(flags)}/{len(flags)}"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.question)
def test_the_old_fact_is_closed_not_kept_alongside(case, walks):
    """Layer one: the store. Both facts left active is the shape of the bug, and
    no amount of prompt work downstream can recover from it. Measured 6/6 on each
    case once the supersede floor stopped hiding event-phrased updates from the
    judge, so this one is held to every run."""
    for store, _, _ in walks[case.question]:
        active = [f.text for f in store.facts if f.active]
        assert active, "distillation kept nothing at all"
        assert len(store.changes()) >= 1, f"no change recorded; active={active}"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.question)
def test_retrieval_does_not_hand_back_the_stale_value(case, gen, walks):
    """Layer two: what recall injects. A composer handed both facts will recite
    both."""
    flags = [bool(metrics.judge_answer(case.question, case.current, mem, gen))
             for _, mem, _ in walks[case.question]]
    assert all(flags), (
        f"recall lost the update in {_rate(flags)} runs:\n"
        + "\n---\n".join(m for _, m, _ in walks[case.question]))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.question)
def test_the_answer_uses_the_current_value(case, gen, walks):
    """Layer three: the whole path, scored by a model because the answer arrives
    in a register rather than as a bare fact. Measured 6/6 against a correct
    store; one run in four is allowed so a single unlucky sample does not cry
    wolf, and a real regression still fails."""
    answers = [a for _, _, a in walks[case.question]]
    assert all(a.strip() for a in answers), "the Keeper said nothing"
    flags = [bool(metrics.judge_answer(case.question, case.current, a, gen))
             for a in answers]
    assert sum(flags) >= TRIALS - 1, (
        f"conveyed {case.current!r} in only {_rate(flags)}:\n"
        + "\n---\n".join(answers))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.question)
def test_the_answer_does_not_also_assert_the_old_one(case, gen, walks):
    """The half the shipped bug passed. Reciting both is not partial success; to
    the person it reads as the Keeper not knowing which of the two is true."""
    answers = [a for _, _, a in walks[case.question]]
    flags = [bool(metrics.judge_answer(case.question, case.stale, a, gen))
             for a in answers]
    assert sum(flags) <= 1, (
        f"still asserts the superseded {case.stale!r} in {_rate(flags)}:\n"
        + "\n---\n".join(answers))
