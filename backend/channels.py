"""channels.py — where the Keeper's unbidden lines go.

The Keeper reaches out through delivery SURFACES, not one hardcoded path. Each is a
Channel with the same tiny contract; a proactive line fans out to every enabled one
at once, so a single reach-out can land as a browser message, a native macOS banner,
AND a phone message — whatever is switched on.

    SSEChannel       -> the open web page (Server-Sent Events)
    NotifierChannel  -> a native macOS banner (works with the browser closed)
    TelegramChannel  -> a message on your phone (opt-in: needs a bot token)

This is the reference agent's channel abstraction (infra/channels: web/telegram/qq behind one
contract) fitted to the Keeper. Adding a surface is adding a Channel — the loop that
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
    async def deliver(self, role: str, content: str, kind: str) -> None: ...


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

    async def deliver(self, role: str, content: str, kind: str) -> None:
        payload = json.dumps({"role": role, "content": content, "kind": kind,
                              "ts": time.time()})
        for q in list(self.listeners):
            await q.put(payload)


class NotifierChannel:
    """A native macOS banner, fired from the backend (reaches you browser-closed)."""

    name = "native"

    def wants(self, kind: str) -> bool:
        return kind == "proactive"     # only unbidden lines become banners

    async def deliver(self, role: str, content: str, kind: str) -> None:
        await asyncio.to_thread(notifier.notify, KEEPER_NAME, content)


class TelegramChannel:
    """A message on your phone via the Telegram Bot API. Opt-in — only built when
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

    async def deliver(self, role: str, content: str, kind: str) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": f"{KEEPER_NAME}\n\n{content}"}
        await asyncio.to_thread(self._sender, url, data)


def _http_post(url: str, data: dict) -> None:
    """POST form-encoded data. Stdlib only; short timeout; never raises upward-
    important errors are logged by the Delivery wrapper."""
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310 - fixed https host
        resp.read()


# --------------------------------------------------------------------------- #
# Delivery — fan one line out to every enabled channel, best-effort.
# --------------------------------------------------------------------------- #

class Delivery:
    def __init__(self, channels: list):
        self.channels = [c for c in channels if c is not None]

    async def push(self, role: str, content: str, kind: str) -> None:
        for ch in self.channels:
            if not ch.wants(kind):
                continue
            try:
                await ch.deliver(role, content, kind)
            except Exception as exc:  # noqa: BLE001 - a dead channel never blocks others
                print(f"[channel:{ch.name}] {type(exc).__name__}: {exc}", flush=True)

    def names(self) -> list[str]:
        return [c.name for c in self.channels]


def build_default(listeners: set) -> Delivery:
    """The Keeper's standard surfaces: web + native banner always; Telegram if a bot
    token is configured."""
    return Delivery([
        SSEChannel(listeners),
        NotifierChannel(),
        TelegramChannel.from_env(),
    ])
