"""Tier 1 — delivery channels (channels.py). No network, no key."""
import pytest
import channels

pytestmark = pytest.mark.unit


class _RecordChannel:
    """A fake channel that records what it was asked to deliver."""
    def __init__(self, name, wants_kind="proactive"):
        self.name = name
        self._wants = wants_kind
        self.delivered = []

    def wants(self, kind):
        return kind == self._wants

    async def deliver(self, role, content, kind):
        self.delivered.append((role, content, kind))


async def test_delivery_fans_out_to_wanting_channels():
    a = _RecordChannel("a", wants_kind="proactive")
    b = _RecordChannel("b", wants_kind="proactive")
    d = channels.Delivery([a, b])
    await d.push("assistant", "the tide turns", "proactive")
    assert a.delivered == b.delivered == [("assistant", "the tide turns", "proactive")]


async def test_delivery_skips_channels_that_dont_want_kind():
    only_proactive = _RecordChannel("np", wants_kind="proactive")
    d = channels.Delivery([only_proactive])
    await d.push("assistant", "echo", "chat")     # not a proactive kind
    assert only_proactive.delivered == []


async def test_one_failing_channel_never_blocks_others():
    class Boom:
        name = "boom"
        def wants(self, kind): return True
        async def deliver(self, *a): raise RuntimeError("down")
    good = _RecordChannel("good", wants_kind="proactive")
    d = channels.Delivery([Boom(), good])
    await d.push("assistant", "still lands", "proactive")   # must not raise
    assert good.delivered == [("assistant", "still lands", "proactive")]


async def test_none_channels_are_dropped():
    good = _RecordChannel("good")
    d = channels.Delivery([None, good, None])
    assert d.names() == ["good"]


# --- native banner gating --- #

def test_notifier_channel_only_wants_proactive():
    ch = channels.NotifierChannel()
    assert ch.wants("proactive") and not ch.wants("chat")


# --- Telegram (opt-in, injectable sender) --- #

def test_telegram_disabled_without_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert channels.TelegramChannel.from_env() is None


def test_telegram_enabled_with_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    ch = channels.TelegramChannel.from_env()
    assert ch is not None and ch.wants("proactive")


async def test_telegram_builds_correct_request():
    sent = {}
    def fake_sender(url, data):
        sent["url"] = url
        sent["data"] = data
    ch = channels.TelegramChannel("123:abc", "42", sender=fake_sender)
    await ch.deliver("assistant", "the light is on", "proactive")
    assert sent["url"] == "https://api.telegram.org/bot123:abc/sendMessage"
    assert sent["data"]["chat_id"] == "42"
    assert "the light is on" in sent["data"]["text"]
    assert channels.KEEPER_NAME in sent["data"]["text"]


def test_build_default_has_web_and_native(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    d = channels.build_default(set())
    assert "web" in d.names() and "native" in d.names()
    assert "telegram" not in d.names()          # off without a token


# --- a surface is only offered if it can actually deliver --- #

def test_native_banner_is_offered_when_the_notifier_exists(monkeypatch):
    monkeypatch.setattr(channels.notifier, "available", lambda: "keeper.app")
    assert "native" in channels.build_default(set()).names()


def test_native_banner_is_withheld_where_it_cannot_fire(monkeypatch):
    """Regression: inside the Linux container /state advertised
    channels: ["web", "native"] while notifier.available() was "none" — a surface
    that silently swallowed every line sent to it. Claiming a capability you do
    not have reads as working until something depends on it."""
    monkeypatch.setattr(channels.notifier, "available", lambda: "none")
    names = channels.build_default(set()).names()
    assert "native" not in names
    assert "web" in names, "the web surface must survive regardless"


# --- Discord --- #

def test_discord_is_off_unless_a_webhook_is_set(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert channels.DiscordChannel.from_env() is None
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "   ")
    assert channels.DiscordChannel.from_env() is None, "whitespace is not a webhook"


def test_discord_is_built_when_configured(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x")
    assert channels.DiscordChannel.from_env() is not None


async def test_discord_sends_the_line_as_json():
    sent = {}
    c = channels.DiscordChannel("https://x/hook",
                                sender=lambda u, d: sent.update(url=u, data=d))
    await c.deliver("assistant", "The tide returns.", "proactive")
    assert sent["url"] == "https://x/hook"
    assert "The tide returns." in sent["data"]["content"]


async def test_discord_truncates_below_the_api_limit():
    """Discord rejects anything over 2000 characters outright, and a re-voiced
    source item is not guaranteed to be short."""
    sent = {}
    c = channels.DiscordChannel("https://x/h",
                                sender=lambda u, d: sent.update(data=d))
    await c.deliver("assistant", "x" * 5000, "proactive")
    assert len(sent["data"]["content"]) <= channels.DiscordChannel.MAX_CHARS


def test_discord_does_not_echo_your_own_chat():
    """A phone or a channel is for unbidden lines. Mirroring the web conversation
    into it would make both useless."""
    c = channels.DiscordChannel("https://x/h")
    assert c.wants("proactive") and not c.wants("chat")


def test_a_configured_discord_joins_the_default_surfaces(monkeypatch):
    monkeypatch.setattr(channels.notifier, "available", lambda: "none")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/x")
    names = channels.build_default(set()).names()
    assert "discord" in names and "web" in names
