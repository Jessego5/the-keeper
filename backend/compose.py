"""compose.py — produce ONE Keeper line, guaranteed on-voice.

The heartbeat every part of the system calls. It ties the three pieces together:

    persona.build_system_prompt()   ->  who the Keeper is, in this mode/state
    generate()  (injected)          ->  the model writes a candidate line
    voice_eval.evaluate()           ->  score it; retry or fall back if off-voice

The model is INJECTED, not imported, for two reasons: the loop is testable offline
with a stub, and swapping models (or providers) never touches this file. Pass a real
generator built by `openai_generator()` in production.

Silence is a first-class outcome. In proactive mode the model may decline to speak;
compose returns a result with `silent=True` and no text. That is the loop working,
not failing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import persona
import voice_eval

# Load backend/.env (if present) so OPENAI_API_KEY is available without the
# caller having to export it. Silent no-op when the file or the lib is absent.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # pragma: no cover
    pass

# A generator turns (system_prompt, user_message) into the model's raw text.
Generator = Callable[[str, str], str]

# The sentinel the model emits to decline speaking (proactive/drift only).
SILENCE_TOKEN = "SILENCE"

# Per-state fallbacks: hand-written, known on-voice, used when generation keeps
# missing the register. Last line of defense so a bad line never reaches the human.
SAFE_LINES = {
    "frozen": "The water is still. You have kept still before.",
    "tidal": "The tide went out and came back, the way it does.",
    "turn": "The ice is going out.",
}

# What we ask the model to do, per mode, as the user turn. Passive gets the human's
# real message; the others get an internal instruction.
_PROACTIVE_INSTRUCTION = (
    f"Speak to them now, unbidden, in one line — or, if you have no true reason to "
    f"speak, output exactly {SILENCE_TOKEN} and nothing else. Silence is the usual "
    f"outcome."
)
_DRIFT_INSTRUCTION = (
    "Reflect. Write your private note to memory now."
)


@dataclass
class ComposeResult:
    text: Optional[str]          # the line to send, or None if silent
    silent: bool                 # model chose not to speak (proactive/drift)
    water_state: str
    mode: str
    attempts: int                # how many generations were tried
    fell_back: bool              # a hand-written SAFE_LINE was used
    score: Optional[float]       # final voice-fidelity score (None if silent)
    report: Optional[voice_eval.VoiceReport] = None

    def __str__(self) -> str:
        if self.silent:
            return f"[silent] ({self.mode}/{self.water_state}, {self.attempts} tries)"
        tag = " FALLBACK" if self.fell_back else ""
        score = f"{self.score:.2f}" if self.score is not None else "note"
        return f"[{score}{tag}] {self.text}"


def compose(
    mode: str = "passive",
    water_state: str = persona.DEFAULT_WATER_STATE,
    *,
    generate: Generator,
    user_message: str = "",
    memory: str = "",
    context: str = "",
    fast_model: Optional[Generator] = None,
    threshold: float = 0.70,
    max_attempts: int = 2,
) -> ComposeResult:
    """Generate one on-voice Keeper line (or silence).

    generate:      (system_prompt, user_message) -> raw model text. Injected.
    user_message:  the human's text in passive mode; ignored otherwise.
    fast_model:    optional cheap model for voice_eval's semantic layer. If None,
                   scoring is deterministic-only (still catches hedges/sentiment/
                   length/motif — the hard fails).
    threshold:     min fidelity to accept a generated line.
    max_attempts:  generations tried before giving up on passing threshold. In
                   passive mode the best real attempt is then kept unless it hard-
                   failed (sentiment/hedge), in which case a SAFE_LINE stands in.

    In drift mode the output is a private memory note, not a sent line: it is
    returned verbatim (no scoring, no fallback), since the voice rests there.
    """
    system = persona.build_system_prompt(
        mode=mode, water_state=water_state, memory=memory, context=context)

    user_turn = {
        "passive": user_message,
        "proactive": _PROACTIVE_INSTRUCTION,
        "drift": _DRIFT_INSTRUCTION,
    }[mode]

    # Drift: private reflection, returned as-is.
    if mode == "drift":
        text = generate(system, user_turn).strip()
        silent = _is_silence(text)
        return ComposeResult(
            text=None if silent else text, silent=silent, water_state=water_state,
            mode=mode, attempts=1, fell_back=False, score=None)

    best: Optional[voice_eval.VoiceReport] = None
    best_text: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        raw = generate(system, user_turn).strip()

        # The model may decline to speak (proactive).
        if _is_silence(raw):
            return ComposeResult(
                text=None, silent=True, water_state=water_state, mode=mode,
                attempts=attempt, fell_back=False, score=None)

        report = voice_eval.evaluate(
            raw, fast_model=fast_model, threshold=threshold)

        if best is None or report.score > best.score:
            best, best_text = report, raw

        if report.passed:
            return ComposeResult(
                text=raw, silent=False, water_state=water_state, mode=mode,
                attempts=attempt, fell_back=False, score=report.score,
                report=report)

    # Every attempt missed threshold. In proactive mode a mediocre unbidden line
    # is worse than silence, so we stay quiet rather than interrupt.
    if mode == "proactive":
        return ComposeResult(
            text=None, silent=True, water_state=water_state, mode=mode,
            attempts=max_attempts, fell_back=False, score=None, report=best)

    # Passive: they asked, so they are owed a reply. The canned safe line is a net
    # for genuinely BAD output — a hard fail (sentiment / hedge). A merely
    # low-motif answer is usually an honest, plain reply to a plain question
    # ("100 degrees Celsius"); clobbering it with a water non-answer would break
    # "a plain question gets a true answer." So keep the best real attempt unless
    # it hard-failed (or there is none).
    if best_text is not None and not best.hard_fail:
        return ComposeResult(
            text=best_text, silent=False, water_state=water_state, mode=mode,
            attempts=max_attempts, fell_back=False,
            score=best.score, report=best)

    return ComposeResult(
        text=SAFE_LINES.get(water_state, SAFE_LINES["tidal"]),
        silent=False, water_state=water_state, mode=mode,
        attempts=max_attempts, fell_back=True,
        score=best.score if best else None, report=best)


_REVOICE_INSTRUCTION = """The message below is a true answer you have just worked out \
(often from a tool — a file, your memory, the web, your own history). Re-voice it so it \
sounds like YOU — your register, your economy — WITHOUT changing what it says.

Rules, in order of importance:
- Keep every concrete fact exactly: names, numbers, dates, quotes, file or code contents.
- Add nothing that is not already there. Invent no events, no feelings, no claims.
- If a line is a plain fact your voice would distort (a number, an address), keep it plain.
- Do not turn it into a riddle; clarity first, voice second.

Output only the re-voiced message — no preface, no explanation."""


def revoice(text: str, water_state: str = persona.DEFAULT_WATER_STATE, *,
            generate: Generator, memory: str = "", context: str = "") -> str:
    """Rewrite a factual/tool-grounded answer in the Keeper's voice, preserving every
    fact. Used so tool answers (git history, a file, the time) still sound like the
    Keeper instead of a flat changelog. Falls back to the original on any failure —
    a true-but-plain answer always beats a lost fact."""
    text = (text or "").strip()
    if not text:
        return text
    system = persona.build_system_prompt(
        mode="passive", water_state=water_state, memory=memory, context=context)
    system = system + "\n\n---\n\n" + _REVOICE_INSTRUCTION
    try:
        out = (generate(system, text) or "").strip()
    except Exception:  # noqa: BLE001 - never lose the answer to a re-voice failure
        return text
    return out or text


def _is_silence(text: str) -> bool:
    return not text or text.strip().upper().strip(".!") == SILENCE_TOKEN


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #

def openai_generator(
    model: str = "gpt-4o",
    fast: str = "gpt-4o-mini",
    max_tokens: int = 600,     # room for re-voiced tool answers; short lines stay short
    fast_max_tokens: int = 600,   # see fast_model below — a CAP, never a target
) -> tuple[Generator, Generator]:
    """Build (generate, fast_model) backed by the OpenAI API.

    `generate` writes Keeper lines with the strong model; `fast_model` runs the
    cheap model for voice_eval's semantic rubric. Requires OPENAI_API_KEY. Raises
    RuntimeError if unavailable, so callers can fall back to the stub. Change
    `model`/`fast` here to use other OpenAI models.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("openai SDK not installed") from e

    client = OpenAI()

    def _call(model_id: str, system: str, user: str, mt: int) -> str:
        resp = client.chat.completions.create(
            model=model_id, max_tokens=mt,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user or "."},
            ],
        )
        return resp.choices[0].message.content or ""

    def generate(system: str, user: str) -> str:
        return _call(model, system, user, max_tokens)

    def fast_model(system: str, user: str) -> str:
        # This was capped at 128 tokens, which is plenty for what the fast model was
        # first used for (a one-word SUPERSEDES/DISTINCT verdict, a rerank list, the
        # voice rubric) but silently truncated everything else. Background research
        # is synthesized AND re-voiced through this same callable, so a multi-
        # paragraph answer arrived cut off mid-sentence at ~640 characters. max_tokens
        # is only an upper bound — a short verdict still costs one short answer.
        return _call(fast, system, user, fast_max_tokens)

    return generate, fast_model


async def tool_reply(
    system: str,
    user: str,
    *,
    providers: Optional[list] = None,
    model: str = "gpt-4o",
    max_rounds: int = 4,
    max_tokens: int = 400,
    trace: Optional[list] = None,
) -> str:
    """Passive-only: let the Keeper use tools, then answer in voice.

    providers: objects shaped like tools.MCPManager — each exposes
        openai_tools() -> list[def]  and  async call(name, args) -> str.
    Pass any mix (MCP file tools + native action tools like reminders); each tool
    call is routed to the provider that declared it. The model may call tools
    across several rounds, sees results, and writes a final line. Returns that
    line's text (scoring is the caller's job — a tool-grounded answer may be
    plainer than a proactive line and must not be replaced by a canned fallback).

    Never used by the proactive loop, which stays sealed.
    """
    from openai import AsyncOpenAI

    aclient = AsyncOpenAI()
    providers = providers or []
    tools: list[dict] = []
    route: dict[str, object] = {}
    for p in providers:
        for t in p.openai_tools():
            tools.append(t)
            route[t["function"]["name"]] = p

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user or "."},
    ]

    for i in range(max_rounds):
        resp = await aclient.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            tools=tools or None,
            tool_choice="auto" if tools else "none",
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        # Record the assistant's tool-call turn, then satisfy each call.
        messages.append({
            "role": "assistant", "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            provider = route.get(tc.function.name)
            out = (await provider.call(tc.function.name, args)
                   if provider is not None else f"(no such tool: {tc.function.name})")
            if trace is not None:      # observability: what the Keeper actually did
                trace.append({"tool": tc.function.name, "args": args,
                              "result": out[:160]})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                            "content": out[:4000]})
        # Step budget running low — nudge it to wrap up on its own before the wall.
        if i == max_rounds - 2:
            messages.append({"role": "system", "content": _BUDGET_WARN})

    # Ran out of rounds — force a GRACEFUL final answer with tools off, so a worker
    # is never cut off mid-thought with nothing to show (the reference agent's forced-cleanup).
    messages.append({"role": "system", "content": _BUDGET_DONE})
    resp = await aclient.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens)
    return resp.choices[0].message.content or ""


_BUDGET_WARN = ("You are almost out of tool steps. If you can answer from what you "
                "already have, do it now instead of calling more tools.")
_BUDGET_DONE = ("Your tool steps are used up. Answer now with what you found — your "
                "best result, noting briefly if something stayed incomplete. Do not "
                "call tools; do not apologize at length.")


def make_generator() -> tuple[Generator, Optional[Generator]]:
    """Best available (generate, fast_model). Real OpenAI if OPENAI_API_KEY is set
    in the environment (or backend/.env), else the offline stub with no semantic
    layer."""
    try:
        return openai_generator()
    except RuntimeError:
        return stub_generator, None


def stub_generator(system: str, user: str) -> str:
    """Offline stand-in so the loop is exercisable with no API key.

    Not a model — it just echoes a plausible on-voice line by peeking at the
    system prompt's active water state. Enough to test the plumbing end to end.
    """
    if "The ice is going out" in system:
        return "The ice is going out. I have watched you cross this break before."
    if "They are in the cold" in system:
        return "The water is still. You have kept still before, in a longer cold."
    return "The tide brought back what you gave the water in spring."


if __name__ == "__main__":
    generate, fast = make_generator()
    backend = "openai" if os.environ.get("OPENAI_API_KEY") else "stub"
    print(f"generator: {backend}\n")

    print("— passive —")
    print(compose("passive", "frozen", generate=generate, fast_model=fast,
                  user_message="i can't get started on anything lately"))

    print("\n— proactive —")
    print(compose("proactive", "tidal", generate=generate, fast_model=fast,
                  memory="- left a project unfinished in spring",
                  context="idle for 8h; screen awake"))

    print("\n— drift —")
    print(compose("drift", generate=generate, fast_model=fast,
                  memory="- brother Sam, not spoken since spring"))
