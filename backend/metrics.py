"""
This scores a free-text answer against a known one.

Every eval in this repo checks answers by looking for a substring: `"paint" in
reply.lower()`. That is fine for "did the fact survive distillation" and useless
for "did it answer the question", because it cannot tell a right answer from a
wrong one that happens to share a word. An answer saying the person STOPPED
painting passes a `"paint" in reply` check exactly as well as one saying they
started again, which is the distinction the whole tide mechanism exists to make.

Three ways to score, cheapest first:

  exact_match   after normalisation. Brittle on purpose, and only useful for
                short factual answers ("Chicago").
  token_f1      the standard extractive-QA overlap measure. Partial credit for a
                right answer wrapped in other words, which is most of them here:
                the Keeper answers in a voice, not in single tokens.
  judge_answer  a model reading both. The only one that survives the Keeper's
                register, where "the tide has turned back to the canvas" is a
                correct answer to "am I painting again?" and scores near zero on
                the other two.

Borrowed in shape from LongMemEval's scoring, which pairs an overlap metric with
a model judge for the same reason.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Iterable, Optional

Generator = Callable[[str, str], str]

_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace.

    The SQuAD convention: "The Chicago." and "chicago" are the same answer, and a
    metric that says otherwise measures typography rather than correctness.
    """
    lowered = _PUNCT.sub(" ", (text or "").lower())
    words = [w for w in lowered.split() if w not in _ARTICLES]
    return _SPACE.sub(" ", " ".join(words)).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def token_f1(pred: str, gold: str) -> float:
    """Harmonic mean of token precision and recall, 0..1.

    Two empty strings match; one empty against a real answer does not, which is
    the case that a naive implementation gets wrong by dividing by zero.
    """
    p_toks, g_toks = normalize(pred).split(), normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    shared = Counter(p_toks) & Counter(g_toks)
    same = sum(shared.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p_toks), same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


_JUDGE_SYSTEM = """You check whether an assistant's answer conveys a known fact.

You are given a question, the correct answer, and the assistant's reply. Say YES if \
the reply conveys the correct answer, even loosely, in metaphor, or wrapped in other \
words. Say NO if it conveys something different, contradicts it, or dodges the \
question without answering it.

Judge the MEANING, not the wording. An answer may be poetic and still be right. An \
answer that merely mentions the same topic without asserting the fact is NO.

Reply with one word: YES or NO."""


def reads_as_yes(verdict: str) -> bool:
    """Lenient, for the reason every verdict parser in this repo is lenient: the
    judge is a model writing free text, and one in this codebase produced three
    spellings of its own one-word answer."""
    first = (verdict or "").strip().strip('"').upper()
    return first.startswith("Y")


def judge_answer(question: str, gold: str, answer: str,
                 generate: Optional[Generator]) -> bool:
    """Does `answer` convey `gold`? False when there is no judge, or it fails.

    Failing to False is deliberate: an eval that treats an outage as a pass is
    worse than no eval, because it goes green precisely when it cannot see.
    """
    if generate is None or not (answer or "").strip():
        return False
    user = (f"Question:\n{question}\n\nCorrect answer:\n{gold}\n\n"
            f"The assistant replied:\n{answer}")
    try:
        return reads_as_yes(generate(_JUDGE_SYSTEM, user))
    except Exception as exc:  # noqa: BLE001
        print(f"[metrics] judge failed, scoring as wrong "
              f"({type(exc).__name__}: {exc})", flush=True)
        return False


def score_answers(preds: Iterable[str], golds: Iterable[str]) -> dict:
    """Aggregate the two string metrics over a run. Lengths must match."""
    pairs = list(zip(preds, golds, strict=True))
    if not pairs:
        return {"n": 0, "exact_match": 0.0, "token_f1": 0.0}
    return {
        "n": len(pairs),
        "exact_match": sum(exact_match(p, g) for p, g in pairs) / len(pairs),
        "token_f1": sum(token_f1(p, g) for p, g in pairs) / len(pairs),
    }
