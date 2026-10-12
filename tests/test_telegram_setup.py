"""
These are the Tier 1 tests for the Telegram setup script (scripts/telegram_setup.py):
who it accepts as the owner, what it writes into .env, and what it asks the Bot API
to do. The Bot API is a fake, so no network and no token.
"""
import json

import pytest

from scripts import telegram_setup as ts

pytestmark = pytest.mark.unit


def _start(uid, text, chat_id=42, kind="private"):
    return {"update_id": uid, "message": {"text": text,
                                          "chat": {"id": chat_id, "type": kind,
                                                   "first_name": "Jess"}}}


def test_only_the_coded_start_is_the_owner():
    """A bot's username is public; whoever finds it first must not become the
    person the Keeper answers to."""
    updates = [_start(1, "/start", chat_id=7), _start(2, "hi", chat_id=8),
               _start(3, "/start 000000", chat_id=9),
               _start(4, "/start 482113", chat_id=42)]
    assert ts.owner_from(updates, "482113")["id"] == 42


def test_a_group_cannot_be_the_owner():
    assert ts.owner_from([_start(1, "/start 482113", kind="group")], "482113") is None


class _Bot:
    """A fake Bot API recording every call, with scripted getUpdates batches."""

    def __init__(self, *batches, me="keeper_test_bot"):
        self.batches = list(batches)
        self.calls = []
        self.me = me

    def __call__(self, method, params, files):
        self.calls.append((method, params, files))
        if method == "getMe":
            return {"ok": True, "result": {"username": self.me}}
        if method == "getUpdates":
            if params.get("timeout") == 0:
                return {"ok": True, "result": []}
            return {"ok": True, "result": self.batches.pop(0) if self.batches else []}
        return {"ok": True, "result": True}

    def methods(self):
        return [m for m, _, _ in self.calls]


def test_waiting_acknowledges_what_it_read():
    """The exchange must not sit in the backlog for the Keeper to find later."""
    bot = _Bot([_start(10, "hello")], [_start(11, "/start 123456")])
    chat = ts.wait_for_owner(bot, "123456", wait_s=60)
    assert chat["id"] == 42
    last = bot.calls[-1]
    assert last[0] == "getUpdates" and last[1]["offset"] == 12


def test_it_gives_up_at_the_deadline():
    ticks = iter([0, 1, 2, 999])
    assert ts.wait_for_owner(_Bot([], []), "1", wait_s=5,
                             clock=lambda: next(ticks)) is None


def test_set_env_var_replaces_in_place_and_keeps_everything_else(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# keys\nOPENAI_API_KEY=sk-x\nTELEGRAM_CHAT_ID=\nKEEPER_FEEDS=a\n")
    ts.set_env_var(env, "TELEGRAM_CHAT_ID", "42")
    assert env.read_text() == ("# keys\nOPENAI_API_KEY=sk-x\nTELEGRAM_CHAT_ID=42\n"
                               "KEEPER_FEEDS=a\n")
    ts.set_env_var(env, "NEW_ONE", "x")
    assert env.read_text().endswith("KEEPER_FEEDS=a\nNEW_ONE=x\n")


def test_the_whole_setup_against_a_fake_bot(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_CHAT_ID=\n")
    monkeypatch.setattr(ts.secrets, "randbelow", lambda n: 654321)
    bot = _Bot([_start(1, "/start 654321", chat_id=4242)])
    assert ts.main([], call=bot, env_path=env) == 0
    assert bot.methods()[:5] == ["getMe", "setMyProfilePhoto", "setMyCommands",
                                 "setMyDescription", "setMyShortDescription"]
    photo_params, photo_files = bot.calls[1][1], bot.calls[1][2]
    assert json.loads(photo_params["photo"]) == {"type": "static",
                                                 "photo": "attach://avatar"}
    assert photo_files["avatar"][1][:2] == b"\xff\xd8"        # a real JPEG
    cmds = [c["command"] for c in json.loads(bot.calls[2][1]["commands"])]
    assert cmds == ["off", "quiet", "usual", "eager"]
    assert "TELEGRAM_CHAT_ID=4242" in env.read_text()
    assert "t.me/keeper_test_bot?start=654321" in capsys.readouterr().out


def test_an_existing_owner_is_left_alone(tmp_path):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_CHAT_ID=99\n")
    bot = _Bot()
    assert ts.main(["--no-photo"], call=bot, env_path=env) == 0
    assert "getUpdates" not in bot.methods() and "setMyProfilePhoto" not in bot.methods()
    assert "TELEGRAM_CHAT_ID=99" in env.read_text()


def test_no_token_says_where_to_get_one(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=x\n")
    assert ts.main([], call=_Bot(), env_path=env) == 1
    assert "BotFather" in capsys.readouterr().out


def test_multipart_carries_the_file_and_the_fields():
    body, ctype = ts._multipart({"photo": '{"type":"static"}'},
                                {"avatar": ("k.jpg", b"\xff\xd8DATA", "image/jpeg")})
    boundary = ctype.split("boundary=")[1]
    assert body.count(boundary.encode()) == 3
    assert b'name="avatar"; filename="k.jpg"' in body and b"\xff\xd8DATA" in body
