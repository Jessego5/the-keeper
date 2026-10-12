"""
These are the Tier 1 tests for compose.revoice: tool answers re-voiced without losing
facts.
"""
import pytest
import compose

pytestmark = pytest.mark.unit


def test_revoice_uses_the_model_output():
    def gen(system, user):
        # a stand-in "voiced" rewrite that keeps the fact (the number 3)
        assert "re-voice" in system.lower()      # the instruction is in the prompt
        return "The record shows three commits, held in the deep."
    out = compose.revoice("You have 3 commits.", generate=gen)
    assert "three" in out


def test_revoice_falls_back_on_empty_output():
    # if the model returns nothing, keep the original answer (never lose the fact)
    out = compose.revoice("The file lists: milk, bread, eggs.",
                          generate=lambda s, u: "   ")
    assert out == "The file lists: milk, bread, eggs."


def test_revoice_falls_back_on_error():
    def boom(system, user):
        raise RuntimeError("model down")
    out = compose.revoice("last commit: fix the notifier", generate=boom)
    assert out == "last commit: fix the notifier"


def test_revoice_empty_input():
    assert compose.revoice("", generate=lambda s, u: "whatever") == ""


def test_revoice_passes_facts_to_the_model():
    # the original factual text must reach the model as the user turn, verbatim
    seen = {}
    def gen(system, user):
        seen["user"] = user
        return "voiced"
    compose.revoice("commit a1b2c3 at 09:00", generate=gen)
    assert seen["user"] == "commit a1b2c3 at 09:00"


# --- the re-voice guard: a name or a number must survive the voice --- #

def test_a_revoice_that_loses_a_name_is_not_used():
    """Seen live: "the note to Sam" came back as "a note from J". Sam was gone,
    and the plain answer is what they get instead."""
    original = "The note to Sam says: I can't make Thursday, but Saturday works. J"
    out = compose.revoice(original, generate=lambda s, u: (
        "You have a note from J saying they cannot meet Thursday, but Saturday "
        "works."))
    assert out == original


def test_a_revoice_that_loses_a_time_is_not_used():
    original = "Your dentist is at 3:15 on Friday."
    out = compose.revoice(original, generate=lambda s, u: "The dentist waits, Friday.")
    assert out == original


def test_a_faithful_revoice_is_kept():
    out = compose.revoice("Your dentist is at 3:15 on Friday.",
                          generate=lambda s, u: "Friday at 3:15, the dentist.")
    assert out == "Friday at 3:15, the dentist."


# --- a timed request that nothing was called for gets one more round --- #

class _FakeOpenAI:
    """Scripted chat.completions: each create() pops the next canned message."""
    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        outer = self

        class _Completions:
            async def create(self, **kw):
                outer.seen.append(list(kw["messages"]))   # a snapshot: it grows
                msg = outer.script.pop(0)
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})

        self.chat = type("Chat", (), {"completions": _Completions()})()


def _msg(content=None, call=None):
    calls = None
    if call:
        fn = type("F", (), {"name": call, "arguments": "{}"})()
        calls = [type("T", (), {"id": "c1", "function": fn})()]
    return type("M", (), {"content": content, "tool_calls": calls})()


class _Tools:
    def __init__(self):
        self.called = []

    def openai_tools(self):
        return [{"type": "function", "function": {"name": "schedule_job",
                                                   "parameters": {}}}]

    async def call(self, name, args):
        self.called.append(name)
        return "scheduled"


async def test_claiming_a_schedule_without_making_it_gets_one_more_round(monkeypatch):
    """Live tier: "brief me every weekday at 8" got "the tide will bring a summary
    at 8" and no call, so no brief existed."""
    fake = _FakeOpenAI([_msg("The tide will bring it at 8."),
                        _msg(call="schedule_job"), _msg("Set for 8.")])
    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda: fake)
    tools = _Tools()
    out = await compose.tool_reply("sys", "brief me every weekday at 8",
                                   providers=[tools],
                                   expect_one_of={"schedule_job", "remind_me"})
    assert tools.called == ["schedule_job"] and out == "Set for 8."
    assert "You answered without calling" in fake.seen[1][-1]["content"]


async def test_the_nudge_happens_once(monkeypatch):
    """If it still will not call the tool, its second answer stands: one extra
    round, never a loop."""
    fake = _FakeOpenAI([_msg("Not now."), _msg("I have not set anything.")])
    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda: fake)
    out = await compose.tool_reply("sys", "remind me at 9", providers=[_Tools()],
                                   expect_one_of={"remind_me"})
    assert out == "I have not set anything." and len(fake.seen) == 2


def test_only_timed_requests_expect_a_tool():
    import server
    assert server._expected_tools("brief me every weekday at 8am")
    assert server._expected_tools("remind me tomorrow at 9 to call sam")
    assert server._expected_tools("what's in my inbox?") is None
    assert server._expected_tools("how is my painting going") is None


async def test_a_refused_call_gets_one_retry(monkeypatch):
    """Live tier: a draft refused for a "[Your Name]" slot was described to the
    person as drafted. A call that came back "(not ...)" gets one retry, with why."""
    fake = _FakeOpenAI([_msg(call="schedule_job"), _msg("Drafted."),
                        _msg(call="schedule_job"), _msg("Proposed.")])
    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda: fake)
    results = iter(["(not proposed: body still has a placeholder)", "proposed"])

    class _Refusing(_Tools):
        async def call(self, name, args):
            self.called.append(name)
            return next(results)

    tools = _Refusing()
    out = await compose.tool_reply("sys", "email sam", providers=[tools])
    assert out == "Proposed." and tools.called == ["schedule_job", "schedule_job"]
    assert "did not go through" in fake.seen[2][-1]["content"]


def test_a_standing_instruction_expects_keep_way():
    import server
    assert server._expected_tools("from now on, sign my emails J") == {"keep_way"}
    assert server._expected_tools("always remind me at 9") == {
        "keep_way", "remind_me", "schedule_job"}
