"""
This is where the Keeper's unbidden lines go.

The Keeper reaches out through delivery SURFACES, not one hardcoded path. Each is a
Channel with the same tiny contract; a proactive line fans out to every enabled one
at once, so a single reach-out can land as a browser message, a native macOS banner,
AND a phone message, whatever is switched on.

    SSEChannel       -> the open web page (Server-Sent Events)
    NotifierChannel  -> a native macOS banner (works with the browser closed)
    TelegramChannel  -> a message on your phone (opt-in: needs a bot token)

This is the reference agent's channel abstraction (infra/channels: web/telegram/qq behind one
contract) fitted to the Keeper. Adding a surface is adding a Channel: the loop that
decides WHEN to speak never changes. Channels are best-effort: one failing (a closed
tab, a network blip) never blocks the others, and never crashes the loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Callable, Optional, Protocol

import notifier

# What the Keeper calls itself in a notification / message header.
KEEPER_NAME = "The Keeper"


class Channel(Protocol):
    name: str
    def wants(self, kind: str) -> bool: ...
    async def deliver(self, role: str, content: str, kind: str,
                      state: Optional[str] = None) -> None: ...


# --------------------------------------------------------------------------- #
# Concrete channels
# --------------------------------------------------------------------------- #

class SSEChannel:
    """The open web page. Holds a reference to the live listener set so it always
    fans out to whoever is currently connected."""

    name = "web"

    def __init__(self, listeners: set):
        self.listeners = listeners

    def wants(self, kind: str) -> bool:
        return True

    async def deliver(self, role: str, content: str, kind: str,
                      state: Optional[str] = None) -> None:
        payload = json.dumps({"role": role, "content": content, "kind": kind,
                              "ts": time.time()})
        for q in list(self.listeners):
            await q.put(payload)


class NotifierChannel:
    """A native macOS banner, fired from the backend (reaches you browser-closed)."""

    name = "native"

    def wants(self, kind: str) -> bool:
        return kind == "proactive"     # only unbidden lines become banners

    async def deliver(self, role: str, content: str, kind: str,
                      state: Optional[str] = None) -> None:
        await asyncio.to_thread(notifier.notify, KEEPER_NAME, content, state)


class TelegramChannel:
    """A message on your phone via the Telegram Bot API. Opt-in, only built when
    TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set. The HTTP send is injectable so
    it can be tested without the network."""

    name = "telegram"

    def __init__(self, token: str, chat_id: str,
                 sender: Optional[Callable[[str, dict], None]] = None):
        self.token = token
        self.chat_id = chat_id
        self._sender = sender or _http_post

    @classmethod
    def from_env(cls) -> Optional["TelegramChannel"]:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            return None
        return cls(token, chat_id)

    def wants(self, kind: str) -> bool:
        return kind == "proactive"     # don't echo your own web chat to your phone

    async def deliver(self, role: str, content: str, kind: str,
                      state: Optional[str] = None) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": f"{KEEPER_NAME}\n\n{content}"}
        await asyncio.to_thread(self._sender, url, data)


class DiscordChannel:
    """A message in a Discord channel via an incoming webhook. Opt-in, only built
    when DISCORD_WEBHOOK_URL is set.

    Simpler than Telegram on purpose: a webhook is one URL with no bot
    registration and no chat id to look up, so it is the quickest surface to prove
    that browser-closed delivery works. It is also the only kind of surface that
    works everywhere: the native banner is macOS-only, and cannot fire at all from
    inside the Linux container, where this can.
    """

    name = "discord"
    # Discord rejects a message over 2000 characters outright. The Keeper's lines
    # are short, but a re-voiced source item is not guaranteed to be.
    MAX_CHARS = 1900

    def __init__(self, webhook_url: str,
                 sender: Optional[Callable[[str, dict], None]] = None):
        self.webhook_url = webhook_url
        self._sender = sender or _json_post

    @classmethod
    def from_env(cls) -> Optional["DiscordChannel"]:
        url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
        return cls(url) if url else None

    def wants(self, kind: str) -> bool:
        return kind == "proactive"     # don't echo your own web chat to a channel

    async def deliver(self, role: str, content: str, kind: str,
                      state: Optional[str] = None) -> None:
        body = f"**{KEEPER_NAME}**\n{content}"[:self.MAX_CHARS]
        await asyncio.to_thread(self._sender, self.webhook_url, {"content": body})


def _json_post(url: str, data: dict) -> None:
    """POST JSON. Stdlib only, short timeout; Delivery logs anything that raises."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310 - operator-set URL
        resp.read()


def _http_post(url: str, data: dict) -> None:
    """POST form-encoded data. Stdlib only; short timeout; never raises upward-
    important errors are logged by the Delivery wrapper."""
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310 - fixed https host
        resp.read()


# --------------------------------------------------------------------------- #
# Delivery: fan one line out to every enabled channel, best-effort.
# --------------------------------------------------------------------------- #

class Delivery:
    def __init__(self, channels: list):
        self.channels = [c for c in channels if c is not None]

    async def push(self, role: str, content: str, kind: str,
                   state: Optional[str] = None) -> None:
        for ch in self.channels:
            if not ch.wants(kind):
                continue
            try:
                await ch.deliver(role, content, kind, state)
            except Exception as exc:  # noqa: BLE001 - a dead channel never blocks others
                print(f"[channel:{ch.name}] {type(exc).__name__}: {exc}", flush=True)

    def names(self) -> list[str]:
        return [c.name for c in self.channels]


def build_default(listeners: set) -> Delivery:
    """The Keeper's standard surfaces: web always; the native banner only where it
    can actually fire; Telegram and Discord if configured.

    The banner is conditional because it was not, and /state advertised
    channels: ["web", "native"] from inside a Linux container where
    notifier.available() is "none": a delivery surface that silently swallowed
    every line sent to it. Reporting a capability that does not exist is the same
    failure as the presence sensors quietly returning defaults: it reads as
    working right up until someone depends on it.
    """
    surfaces: list = [SSEChannel(listeners)]
    if notifier.available() != "none":
        surfaces.append(NotifierChannel())
    else:
        print("[channels] native banner unavailable here "
              "(no macOS notifier), not offering it", flush=True)
    surfaces.append(TelegramChannel.from_env())
    surfaces.append(DiscordChannel.from_env())
    return Delivery(surfaces)
