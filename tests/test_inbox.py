"""
These are the Tier 1 tests for the Telegram inbox (inbox.py): what it accepts, from
whom, and what it refuses to replay. The Bot API is a fake, so no network, no token.
"""
import pytest

import inbox

pytestmark = pytest.mark.unit

ME = "4242"
STRANGER = "999"


class _Bot:
    """A fake Bot API: hands back queued getUpdates results, records every call."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.calls = []

    def __call__(self, url, params):
        method = url.rsplit("/", 1)[-1]
        self.calls.append((method, params))
        if method == "getUpdates":
            return {"ok": True, "result": self.batches.pop(0) if self.batches else []}
        return {"ok": True, "result": {}}

    def sent(self, method):
        return [p for m, p in self.calls if m == method]


def _tap(uid, data, chat=ME):
    return {"update_id": uid, "callback_query": {
        "id": f"cq{uid}", "data": data, "message": {"chat": {"id": int(chat)}}}}


def _text(uid, text, chat=ME):
    return {"update_id": uid, "message": {"chat": {"id": int(chat)}, "text": text}}


def _inbox(bot, decided=None, levels=None):
    decided = decided if decided is not None else []
    levels = levels if levels is not None else []

    async def decide(pid, approve, via):
        decided.append((pid, approve, via))
        return "Done."

    def set_level(name):
        levels.append(name)
        return name

    return inbox.TelegramInbox("tok", ME, decide=decide, set_level=set_level,
                               fetch=bot)


async def test_a_tap_from_the_person_decides():
    decided = []
    bot = _Bot([_tap(1, "approve:p-abc"), _tap(2, "decline:p-def")])
    await _inbox(bot, decided).poll_once(timeout=0)
    assert decided == [("p-abc", True, "telegram"), ("p-def", False, "telegram")]
    # every tap is answered, or the button spins forever on their phone
    assert [p["callback_query_id"] for p in bot.sent("answerCallbackQuery")] == [
        "cq1", "cq2"]


async def test_a_tap_from_anyone_else_is_ignored():
    """The bot's username is public. The chat id is what makes a tap theirs."""
    decided = []
    bot = _Bot([_tap(1, "approve:p-abc", chat=STRANGER)])
    await _inbox(bot, decided).poll_once(timeout=0)
    assert decided == []
    assert bot.sent("answerCallbackQuery") == [] and bot.sent("sendMessage") == []


async def test_a_malformed_tap_does_nothing():
    decided = []
    bot = _Bot([_tap(1, "launch:p-abc"), _tap(2, "approve:")])
    await _inbox(bot, decided).poll_once(timeout=0)
    assert decided == []


async def test_the_dial_can_be_turned_from_the_phone():
    levels = []
    bot = _Bot([_text(1, "/quiet"), _text(2, "/off@KeeperBot")])
    await _inbox(bot, levels=levels).poll_once(timeout=0)
    assert levels == ["quiet", "off"]
    replies = [p["text"] for p in bot.sent("sendMessage")]
    assert "quiet" in replies[0] and "still comes back" in replies[1]


async def test_a_stranger_cannot_turn_the_dial():
    levels = []
    bot = _Bot([_text(1, "/off", chat=STRANGER)])
    await _inbox(bot, levels=levels).poll_once(timeout=0)
    assert levels == [] and bot.sent("sendMessage") == []


async def test_plain_words_get_a_hint_not_a_conversation():
    bot = _Bot([_text(1, "hello?"), _text(2, "/whisper")])
    await _inbox(bot).poll_once(timeout=0)
    first, second = [p["text"] for p in bot.sent("sendMessage")]
    assert "/quiet" in first and "/eager" in second


async def test_yesterdays_commands_are_not_replayed_on_start():
    """Without skipping the backlog, a restart would replay an old /off over the
    'usual' set on the page since."""
    levels = []
    bot = _Bot([_text(41, "/off")],          # what Telegram is still holding
               [_text(42, "/eager")])        # what arrives after the start
    box = _inbox(bot, levels=levels)
    await box.skip_backlog()
    assert box.offset == 42
    await box.poll_once(timeout=0)
    assert levels == ["eager"]
    assert bot.calls[1][1]["offset"] == 42


async def test_one_bad_update_does_not_lose_the_rest():
    decided = []

    async def decide(pid, approve, via):
        if pid == "p-bad":
            raise RuntimeError("store is locked")
        decided.append(pid)
        return "Done."

    bot = _Bot([_tap(1, "approve:p-bad"), _tap(2, "approve:p-good")])
    box = inbox.TelegramInbox("tok", ME, decide=decide,
                              set_level=lambda n: n, fetch=bot)
    await box.poll_once(timeout=0)
    assert decided == ["p-good"] and box.offset == 3


async def test_a_refused_poll_raises_so_run_can_back_off():
    def refuse(url, params):
        return {"ok": False, "description": "Unauthorized"}
    box = inbox.TelegramInbox("tok", ME, decide=None, set_level=None, fetch=refuse)
    with pytest.raises(RuntimeError, match="Unauthorized"):
        await box.poll_once(timeout=0)


# --- conversation from the phone --- #

async def test_plain_words_from_the_person_are_a_conversation():
    heard = []

    async def converse(text):
        heard.append(text)
        return "Saturday at 7, he said."

    bot = _Bot([_text(1, "what time did sam say?")])
    box = inbox.TelegramInbox("tok", ME, decide=None, set_level=None,
                              converse=converse, fetch=bot)
    await box.poll_once(timeout=0)
    assert heard == ["what time did sam say?"]
    assert [p["action"] for p in bot.sent("sendChatAction")] == ["typing"]
    assert [p["text"] for p in bot.sent("sendMessage")] == ["Saturday at 7, he said."]


async def test_a_stranger_gets_no_conversation():
    heard = []

    async def converse(text):
        heard.append(text)
        return "x"

    bot = _Bot([_text(1, "what do you know about them?", chat=STRANGER)])
    box = inbox.TelegramInbox("tok", ME, decide=None, set_level=None,
                              converse=converse, fetch=bot)
    await box.poll_once(timeout=0)
    assert heard == [] and bot.calls[1:] == []


async def test_a_long_reply_arrives_in_parts():
    async def converse(text):
        return "x" * (inbox.MAX_REPLY_CHARS + 10)

    bot = _Bot([_text(1, "tell me everything")])
    box = inbox.TelegramInbox("tok", ME, decide=None, set_level=None,
                              converse=converse, fetch=bot)
    await box.poll_once(timeout=0)
    parts = [p["text"] for p in bot.sent("sendMessage")]
    assert [len(p) for p in parts] == [inbox.MAX_REPLY_CHARS, 10]
