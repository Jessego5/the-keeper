"""
These are the Tier 1 tests for the unbidden log (unbidden.py) and the two seams it
leans on: sessions remembering which lines were the Keeper's own, and the web channel
being the only one handed a line's sequence number. No network, no key.
"""
import json

import pytest

import channels
import sessions
import unbidden

pytestmark = pytest.mark.unit


def test_lines_are_numbered_in_order(tmp_path):
    log = unbidden.UnbiddenLog(tmp_path / "u.jsonl")
    a = log.record("the tide is in", why=unbidden.Why.RESTLESS)
    b = log.record("you asked me to hold this", why=unbidden.Why.REMINDER)
    assert (a.seq, b.seq) == (1, 2)
    assert [ln.text for ln in log.after(1)] == ["you asked me to hold this"]


def test_numbering_survives_a_restart(tmp_path):
    """The seq is the SSE event id a browser hands back. If a restart began again
    at 1, a page holding id 7 would be told nothing new had been said, forever."""
    path = tmp_path / "u.jsonl"
    log = unbidden.UnbiddenLog(path)
    for i in range(3):
        log.record(f"line {i}")
    reopened = unbidden.UnbiddenLog(path)
    assert reopened.record("after the restart").seq == 4


def test_a_torn_last_line_does_not_cost_the_rest(tmp_path):
    """A crash mid-write leaves half a JSON object at the end of the file. That
    one record is lost; refusing to load would lose every other one."""
    path = tmp_path / "u.jsonl"
    log = unbidden.UnbiddenLog(path)
    log.record("kept")
    with path.open("a") as f:
        f.write('{"seq": 2, "text": "half a li')
    reopened = unbidden.UnbiddenLog(path)
    assert [ln.text for ln in reopened.lines] == ["kept"]


def test_replay_is_bounded(tmp_path):
    """A page asleep for a month is caught up on the recent past, not buried."""
    log = unbidden.UnbiddenLog(tmp_path / "u.jsonl")
    for i in range(80):
        log.record(f"line {i}")
    missed = log.after(0, limit=50)
    assert len(missed) == 50 and missed[-1].text == "line 79"


def test_the_record_is_written_before_it_is_returned(tmp_path):
    path = tmp_path / "u.jsonl"
    unbidden.UnbiddenLog(path).record("said", why=unbidden.Why.GOAL, state="tidal",
                                      session="s-1")
    row = json.loads(path.read_text().splitlines()[0])
    assert row["why"] == "goal" and row["state"] == "tidal" and row["session"] == "s-1"


@pytest.mark.parametrize("header,expected", [
    ("7", 7), (" 12 ", 12), ("", None), (None, None), ("not-a-number", None),
])
def test_last_event_id_parsing(header, expected):
    assert unbidden.parse_last_event_id(header) == expected


# --- sessions keep the kind of a line --- #

def test_a_reload_still_knows_which_lines_were_unbidden(tmp_path):
    store = sessions.SessionStore(tmp_path)
    s = store.new()
    store.append(s.key, "user", "hello")
    store.append(s.key, "assistant", "the water stirs", kind="proactive")
    msgs = sessions.SessionStore(tmp_path).get(s.key).messages
    assert [m.kind for m in msgs] == ["reply", "proactive"]


def test_old_session_files_load_as_replies(tmp_path):
    """Written before `kind` existed. They were shown as replies, so that is what
    they stay; failing to load them would empty the sidebar."""
    (tmp_path / "s-old.json").write_text(json.dumps({
        "key": "s-old", "title": "hi", "created": 1.0, "updated": 1.0,
        "messages": [{"role": "user", "content": "hi", "ts": 1.0}]}))
    msgs = sessions.SessionStore(tmp_path).get("s-old").messages
    assert msgs[0].kind == "reply"


def test_a_conversation_the_keeper_opens_is_titled_by_its_line(tmp_path):
    store = sessions.SessionStore(tmp_path)
    s = store.new()
    store.append(s.key, "assistant", "You asked me to hold this: call Sam.",
                 kind="proactive")
    assert store.get(s.key).title.startswith("You asked me to hold this")


def test_an_ordinary_reply_never_titles_a_conversation(tmp_path):
    store = sessions.SessionStore(tmp_path)
    s = store.new()
    store.append(s.key, "assistant", "a reply with no question before it")
    assert store.get(s.key).title == ""


# --- only the web channel is handed the sequence number --- #

class _Plain:
    """A channel written to the original contract, as every fake and every surface
    added later will be. It must keep working untouched."""
    name = "plain"

    def __init__(self):
        self.got = []

    def wants(self, kind):
        return True

    async def deliver(self, role, content, kind, state=None):
        self.got.append(content)


async def test_meta_reaches_the_web_page_and_nothing_else():
    import asyncio
    q: asyncio.Queue = asyncio.Queue()
    plain = _Plain()
    d = channels.Delivery([channels.SSEChannel({q}), plain])
    await d.push("assistant", "the tide", "proactive",
                 meta={"id": 3, "session": "s-1"})
    payload = json.loads(q.get_nowait())
    assert payload["id"] == 3 and payload["session"] == "s-1"
    assert plain.got == ["the tide"]
