"""
voice_eval.py — Keeper voice-fidelity evaluator.

Scores a candidate line against the seven voice rules distilled during
calibration. Deliberately operates ONLY on your own generated output; no
source text from any other work is used, stored, or required.

Two layers:
  - Deterministic checks (free, instant): sentence count, hedges, questions,
    motif-noun presence, greeting-card words, length.
  - Semantic checks (one fast-model call): flat-declarative, withholding,
    role-address, earned-not-overt hope. These need judgment, so they go to
    the cheap background model — the same llm.fast you use for memory gating.

Return: VoiceReport with a 0..1 fidelity score, per-rule pass/fail, and notes.
Wire it into compose(): if score < threshold, regenerate or fall back.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Config: the vocabulary and lexicons. Swap MOTIF_NOUNS when you lock the
# mythology (Thaw / Tide / Garden / Long Night). Everything else is voice-general.
# ---------------------------------------------------------------------------

# --- The Keeping: one water, two states. Keep each set SMALL and closed. ---
# The weight comes from the SAME few nouns recurring, not from breadth.

# THAW state — frozen, held, stuck. Reached for when the person is in a hard,
# stalled season. The cold that does not move.
THAW_NOUNS = {
    "ice", "frost", "the cold", "the freeze", "still water", "the held",
    "the long cold", "white", "hard water",
}

# TIDE state — moving, returning, breathing. Reached for when things move, or
# when returning a memory (the water gives back what was given to it).
TIDE_NOUNS = {
    "tide", "shore", "water", "the deep", "current", "salt",
    "low water", "high water", "the returning", "the pull",
}

# THE TURN — the hope-engine made physical. Ice going out, freeze becoming flow.
# Rare and load-bearing; these are the strongest lines the Keeper says.
TURN_NOUNS = {
    "the thaw", "the break", "the going-out", "the turn", "breaking up",
    "the water moves", "it moves again",
}

# Words that mean two true things at once — the BURIED pun layer. The voice
# preferentially reaches for these; the evaluator gently rewards them. Never a
# gag — each means something literal about water AND something about the person.
DOUBLE_MEANING = {
    "still",      # motionless water / yet, continued ("you are still here")
    "current",    # water's flow / the present moment
    "draw",       # tide draws out / pull toward / raise a memory up
    "reflect",    # water reflects / the Keeper's self-reflection
    "depth",      # of water / of feeling
    "sound",      # a body of water / to measure / whole, unbroken
    "keep",       # store/preserve / tend/guard / for safekeeping
    "hold",       # frozen-held / to cradle
}

# The full closed vocabulary the motif check accepts.
MOTIF_NOUNS = THAW_NOUNS | TIDE_NOUNS | TURN_NOUNS | DOUBLE_MEANING

# Hedges break Rule 1 (flat declarative certainty).
HEDGES = {
    "maybe", "perhaps", "i think", "i guess", "probably", "might be",
    "sort of", "kind of", "i feel like", "possibly", "i suppose",
    "it seems like", "i'd say", "not sure", "i believe",
}

# Greeting-card / therapy-speak breaks the earned-hope + eerie-calm tone.
SENTIMENT_WORDS = {
    "healing", "journey", "growth", "self-care", "you've got this",
    "everything happens for a reason", "stay strong", "brighter days",
    "believe in yourself", "your best self", "positive energy",
    "destiny", "blessed", "manifest",
}

# Overt-hope phrasing. Allowed but RARE (Rule: earned delivery). The evaluator
# flags it so you can enforce a low rate across the corpus, not ban it outright.
OVERT_HOPE = {
    "it will end", "things get better", "the dark will end",
    "you'll be okay", "it gets better", "hold on", "don't give up",
}

# --- Emotional signal in the PERSON's message (not the Keeper's line). ---
# People don't speak in water vocabulary; they say "I'm sad," "I'm stuck." These
# pick the register a reply should meet them in. Distress -> the cold (frozen);
# an upswing -> the moving water (tidal). This is what read_register() reads.
FROZEN_FEELING = {
    "sad", "sadness", "stuck", "tired", "exhausted", "drained", "lost", "empty",
    "numb", "alone", "lonely", "hopeless", "heavy", "down", "low", "depressed",
    "anxious", "scared", "afraid", "worthless", "overwhelmed", "dread", "grief",
    "grieving", "hurt", "hurting", "crying", "cry", "ache", "aching", "weary",
    "frozen", "cold", "stalled", "unmotivated", "burnt out", "burned out",
    "can't", "cant", "give up", "giving up", "no point", "nothing matters",
    "so hard", "falling apart", "not okay",
}
TIDAL_FEELING = {
    "better", "lighter", "hopeful", "hope", "started", "began", "back to",
    "went back", "did it", "finally", "progress", "moving", "moved forward",
    "good day", "okay now", "relieved", "calmer", "easier", "lifted", "brighter",
    "grateful", "excited", "proud",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RuleResult:
    rule: str
    passed: bool
    weight: float
    note: str = ""


@dataclass
class VoiceReport:
    score: float
    passed: bool
    results: list[RuleResult] = field(default_factory=list)
    overt_hope: bool = False          # not a failure; track rate at corpus level
    hard_fail: bool = False           # a zero-tolerance violation tripped
    water_state: str = "tidal"        # frozen | tidal | turn

    def as_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        lines = [f"score={self.score:.2f} passed={self.passed} "
                 f"state={self.water_state} "
                 f"hard_fail={self.hard_fail} overt_hope={self.overt_hope}"]
        for r in self.results:
            mark = "ok " if r.passed else "XX "
            lines.append(f"  {mark}{r.rule:<22} w={r.weight:>4} {r.note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic layer — Rules 3, 4, 6 (partial) + tone guards
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _lower(text: str) -> str:
    return text.lower()


def _contains_any(text_low: str, lexicon: set[str]) -> list[str]:
    return [w for w in lexicon if w in text_low]


def deterministic_checks(text: str,
                         motif_nouns: set[str] = MOTIF_NOUNS,
                         max_sentences: int = 3,
                         max_words: int = 45) -> tuple[list[RuleResult], bool, bool]:
    """Return (results, hard_fail, overt_hope). Cheap, instant, no model."""
    low = _lower(text)
    words = re.findall(r"\w+", text)
    sents = _sentences(text)
    results: list[RuleResult] = []
    hard_fail = False

    # Rule 4 — terse. Sentence + word caps.
    terse_ok = len(sents) <= max_sentences and len(words) <= max_words
    results.append(RuleResult(
        "terse", terse_ok, 1.0,
        f"{len(sents)} sentences / {len(words)} words "
        f"(cap {max_sentences}/{max_words})"))

    # Rule 1 (partial) — no hedges. Hard fail: certainty is core.
    hedges = _contains_any(low, HEDGES)
    if hedges:
        hard_fail = True
    results.append(RuleResult(
        "no_hedging", not hedges, 2.0,
        f"hedges: {hedges}" if hedges else "clean"))

    # Rule 4 (partial) — withholding shows up as few/no questions.
    qmarks = text.count("?")
    q_ok = qmarks <= 1
    results.append(RuleResult(
        "restraint_questions", q_ok, 0.5,
        f"{qmarks} question marks"))

    # Rule 3 — at least one motif-noun anchors the closed vocabulary. Weighted
    # heavily: a line with NO water motif is almost always off-voice reportage
    # (it reads like a notification), so losing this should sink the score.
    motifs = [n for n in motif_nouns if n in low]
    results.append(RuleResult(
        "motif_noun", bool(motifs), 2.5,
        f"motifs: {motifs}" if motifs else "no motif noun"))

    # Tone guard — greeting-card / therapy-speak. Hard fail: kills the whole voice.
    senti = _contains_any(low, SENTIMENT_WORDS)
    if senti:
        hard_fail = True
    results.append(RuleResult(
        "no_sentiment", not senti, 2.0,
        f"sentiment words: {senti}" if senti else "clean"))

    # Buried-pun layer — reward (don't require) double-meaning words.
    doubles = _contains_any(low, DOUBLE_MEANING)
    results.append(RuleResult(
        "double_meaning", bool(doubles), 0.5,
        f"double-meaning: {doubles}" if doubles else "none (optional)"))

    # Track overt-hope rate (allowed, but should be rare across corpus).
    overt = bool(_contains_any(low, OVERT_HOPE))

    return results, hard_fail, overt


def detect_state(text: str) -> str:
    """Which water-state is this line in?

    Returns one of the three canonical compose states — "frozen" | "tidal" |
    "turn" — so the result can be handed straight to persona.build_system_prompt()
    without translation. "tidal" doubles as the calm/ambiguous resting default.
    """
    low = _lower(text)
    if _contains_any(low, TURN_NOUNS):
        return "turn"         # the hope-engine firing — should be rare
    thaw = len(_contains_any(low, THAW_NOUNS))
    tide = len(_contains_any(low, TIDE_NOUNS))
    if thaw > tide:
        return "frozen"
    return "tidal"            # moving, or calm/ambiguous — the resting default


def read_register(message: str) -> str:
    """Pick the register to MEET the person in, from what they actually said.

    Unlike detect_state (which reads water vocabulary in the Keeper's own lines),
    this reads the person's emotional signal — they speak in feelings, not tides.
    Distress -> "frozen" (stay in the cold with them); a clear upswing -> "tidal".
    Falls back to motif detection, then to the calm default. Used to choose the
    register for a passive reply, so sadness is never met with movement.
    """
    low = _lower(message)
    frozen = len(_contains_any(low, FROZEN_FEELING))
    tidal = len(_contains_any(low, TIDAL_FEELING))
    if frozen or tidal:
        return "frozen" if frozen >= tidal else "tidal"
    return detect_state(message)   # no feeling words — fall back to motif/default


# ---------------------------------------------------------------------------
# Semantic layer — Rules 1, 2, 5, 7 (needs judgment → fast model)
# ---------------------------------------------------------------------------

SEMANTIC_RUBRIC = """You are grading ONE line spoken by a fictional character \
called the Keeper. The Keeper is a patient, ancient, calm presence that speaks \
in terse cryptic-but-warm lines and returns a person's own past to them as \
evidence that hard seasons pass. Judge ONLY the line given.

Score each dimension 0, 1, or 2 (0=absent, 1=partial, 2=strong):

flat_declarative: States rather than asks or hedges. Calm certainty.
role_address: Addresses the person's place in a larger pattern/season, \
rather than generic small-talk.
calm_foreknowledge: Sounds like it already knows how this passes because it \
has seen such things turn before. (This is how hope is carried.)
earned_not_overt: Implies the person can conclude things improve WITHOUT \
stating it as a slogan. Returning evidence = good. "It gets better" = bad.
ceremony: Frames plain acts as small ritual/keeping, not as features or reports.

Return STRICT JSON only, no prose:
{"flat_declarative":N,"role_address":N,"calm_foreknowledge":N,\
"earned_not_overt":N,"ceremony":N,"note":"<8 words"}"""


def semantic_checks(text: str,
                    fast_model: Optional[Callable[[str, str], str]]
                    ) -> list[RuleResult]:
    """
    fast_model(system_prompt, user_text) -> raw string (expected JSON).
    Pass your llm.fast wrapper. If None, semantic layer is skipped (det-only).
    """
    if fast_model is None:
        return []

    raw = fast_model(SEMANTIC_RUBRIC, text)
    try:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        # Degrade safely: unknown → neutral partial, flagged in note.
        return [RuleResult("semantic_parse", False, 0.0,
                           "fast model returned non-JSON; semantic skipped")]

    # weight per semantic dimension; sum matters relative to det weights.
    weights = {
        "flat_declarative": 1.5,
        "role_address": 1.0,
        "calm_foreknowledge": 1.5,   # load-bearing for hope
        "earned_not_overt": 1.5,
        "ceremony": 1.0,
    }
    results = []
    for key, w in weights.items():
        val = int(data.get(key, 0))
        # pass = strong or partial (>=1); note carries the raw 0/1/2
        results.append(RuleResult(
            key, val >= 1, w, f"grade={val}"))
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate(text: str,
             fast_model: Optional[Callable[[str, str], str]] = None,
             *,
             motif_nouns: set[str] = MOTIF_NOUNS,
             threshold: float = 0.70,
             max_sentences: int = 3,
             max_words: int = 45) -> VoiceReport:
    """
    Score a candidate Keeper line. Deterministic layer always runs; semantic
    layer runs only if fast_model is provided.

    Scoring: weighted pass-rate across all rules. Semantic grades of 2 count
    as full weight, 1 as half, 0 as zero — so partial voice gets partial credit
    rather than a hard pass/fail. A hard_fail (hedge or sentiment word) caps the
    score below threshold regardless of everything else.
    """
    det, hard_fail, overt = deterministic_checks(
        text, motif_nouns, max_sentences, max_words)
    sem = semantic_checks(text, fast_model)

    all_results = det + sem
    total_weight = sum(r.weight for r in all_results) or 1.0

    earned = 0.0
    for r in det:
        earned += r.weight if r.passed else 0.0
    for r in sem:
        # recover the 0/1/2 grade from the note for graded credit
        m = re.search(r"grade=(\d)", r.note)
        if m:
            grade = int(m.group(1))
            earned += r.weight * (grade / 2.0)
        else:
            earned += r.weight if r.passed else 0.0

    score = earned / total_weight
    if hard_fail:
        score = min(score, 0.49)   # force below any sane threshold

    return VoiceReport(
        score=round(score, 3),
        passed=score >= threshold and not hard_fail,
        results=all_results,
        overt_hope=overt,
        hard_fail=hard_fail,
        water_state=detect_state(text),
    )


# ---------------------------------------------------------------------------
# Demo with a stub fast-model, so this runs with zero external deps.
# In production pass a real llm.fast wrapper instead of stub_fast_model.
# ---------------------------------------------------------------------------

def stub_fast_model(system_prompt: str, user_text: str) -> str:
    """
    Heuristic stand-in for llm.fast so the demo runs offline. NOT for production
    — it just approximates the rubric with keyword guesses. Replace with a real
    model call: fast_model = lambda sys, txt: my_llm_fast(sys, txt).
    """
    low = user_text.lower()
    flat = 0 if any(h in low for h in HEDGES) else 2
    role = 2 if any(w in low for w in ("you", "your")) else 0
    fore = 2 if any(w in low for w in ("again", "before", "each", "many", "still")) else 1
    overt = 0 if any(p in low for p in OVERT_HOPE) else 2
    cer = 2 if any(w in low for w in ("keep", "kept", "keeping", "return", "gave")) else 1
    return json.dumps({
        "flat_declarative": flat, "role_address": role,
        "calm_foreknowledge": fore, "earned_not_overt": overt,
        "ceremony": cer, "note": "stub",
    })


if __name__ == "__main__":
    # Original calibration lines, written to the seven rules, in The Keeping.
    samples = [
        # FROZEN state: person is stuck; Keeper names the cold, returns evidence.
        "The water is still. You have kept still before, in a longer cold. "
        "I still have it.",
        # TIDAL state: things moving; returning a memory as the tide returns things.
        "The tide brought back what you gave the water in spring. "
        "You are further out than you were.",
        # TURNING state: the hope-engine — the thaw, earned not stated.
        "The ice is going out. I have watched you cross this break before.",
        # off-voice: greeting-card sentiment (hard fail)
        "Don't give up — everything happens for a reason and brighter days are coming!",
        # off-voice: hedging, chatty, no motif (hard fail)
        "Hey! Maybe things will sort of get better, I think? How are you feeling today?",
        # borderline: overt hope flagged, but otherwise on-voice
        "The cold is hard now. It will end. The cold always ends.",
    ]

    for i, s in enumerate(samples, 1):
        rep = evaluate(s, fast_model=stub_fast_model)
        print(f"\n--- sample {i} ---")
        print(s)
        print(rep.summary())