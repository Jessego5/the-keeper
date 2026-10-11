"""
This is the way back in: the person answering the Keeper from their phone.

Every channel in channels.py points outward. That was enough while the Keeper only
said things, and stopped being enough the moment it could ask: a proposal waiting
on a yes is useless if the yes can only be given at the desk it was sent away from.
ChatGPT's dots message you in Slack when they need a decision; this is the same idea
on the surface the Keeper already uses for your phone.

It long-polls Telegram's getUpdates on the same bot that TelegramChannel sends with.
Two things are accepted, and only from TELEGRAM_CHAT_ID:

    a tap on Approve / Decline    -> decide(proposal_id, approve)
    /off /quiet /usual /eager     -> set_level(name), the proactivity dial

Anything from any other chat is ignored without a reply. A bot's username is
public; the chat id is what makes a tap the person's.

The backlog is skipped on start. Telegram holds updates for a day, so without that a
restart would replay yesterday's "/off" over the "usual" set on the page since.
A tap made while the Keeper was down is lost the same way, which is the right side
to err on: an approval should be given to a Keeper that is there to act on it.

Polling, not a webhook, because a webhook needs a public URL and this binds to
127.0.0.1 on purpose. The HTTP call is injectable so all of it runs offline.
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
import urllib.request
from typing import Awaitable, Callable, Optional

import dial as dial_mod

# Long-poll window. Telegram holds the request open this long when there is
# nothing to say, so an idle Keeper costs one request every half minute.
POLL_TIMEOUT_S = 25
# After a failed poll (network down, Telegram hiccup), wait before trying again
# rather than spinning on an error.
BACKOFF_S = 15.0

Fetch = Callable[[str, dict], dict]
Decide = Callable[[str, bool, str], Awaitable[str]]
SetLevel = Callable[[str], Optional[str]]


def _http_fetch(url: str, params: dict) -> dict:
    body = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    # The read timeout must outlast the long poll, or every quiet poll is an error.
    with urllib.request.urlopen(req, timeout=POLL_TIMEOUT_S + 10) as resp:  # noqa: S310
        return json.loads(resp.read())


class TelegramInbox:
    def __init__(self, token: str, chat_id: str, *, decide: Decide,
                 set_level: SetLevel, fetch: Optional[Fetch] = None):
        self.token = token
        self.chat_id = str(chat_id)
        self._decide = decide
        self._set_level = set_level
        self._fetch = fetch or _http_fetch
        self.offset: Optional[int] = None
        self.stopped_because: Optional[str] = None

    def _api(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    async def _call(self, method: str, params: dict) -> dict:
        return await asyncio.to_thread(self._fetch, self._api(method), params)

    async def skip_backlog(self) -> None:
        """Start after the newest update Telegram is holding. See the module
        docstring for why the backlog is not replayed."""
        resp = await self._call("getUpdates", {"offset": -1, "timeout": 0})
        updates = resp.get("result") or []
        if updates:
            self.offset = updates[-1]["update_id"] + 1

    async def poll_once(self, timeout: int = POLL_TIMEOUT_S) -> int:
        """One long poll; handles what came back. Returns how many updates."""
        params: dict = {"timeout": timeout,
                        "allowed_updates": json.dumps(["message", "callback_query"])}
        if self.offset is not None:
            params["offset"] = self.offset
        resp = await self._call("getUpdates", params)
        if not resp.get("ok", True):
            raise RuntimeError(resp.get("description") or "getUpdates refused")
        updates = resp.get("result") or []
        for update in updates:
            self.offset = update["update_id"] + 1
            try:
                await self.handle(update)
            except Exception as exc:  # noqa: BLE001 - one bad update never stops the rest
                print(f"[inbox] update {update.get('update_id')} failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        return len(updates)

    def _from_the_person(self, chat: Optional[dict]) -> bool:
        return chat is not None and str(chat.get("id")) == self.chat_id

    async def handle(self, update: dict) -> None:
        cq = update.get("callback_query")
        if cq is not None:
            if not self._from_the_person((cq.get("message") or {}).get("chat")):
                return
            verb, _, pid = (cq.get("data") or "").partition(":")
            if verb not in ("approve", "decline") or not pid:
                return
            note = await self._decide(pid, verb == "approve", "telegram")
            # Answering the callback stops the button's spinner; the note is the
            # short toast Telegram shows. The message itself is edited by the
            # channel once the outcome is known.
            await self._call("answerCallbackQuery",
                             {"callback_query_id": cq.get("id"), "text": note[:190]})
            return

        msg = update.get("message")
        if msg is None or not self._from_the_person(msg.get("chat")):
            return
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            await self._reply("Approvals and the dial live here (/off /quiet "
                              "/usual /eager). Talk to me on the page.")
            return
        # "/quiet@SomeBot" is how a command arrives when the bot was mentioned.
        command = text.split()[0].split("@")[0]
        level = dial_mod.normalize(command)
        if level is None:
            await self._reply("I know /off, /quiet, /usual and /eager.")
            return
        set_to = self._set_level(level)
        await self._reply(f"The dial is at {set_to}." + (
            " I will start nothing of my own. What you asked me to hold still "
            "comes back." if set_to == "off" else ""))

    async def _reply(self, text: str) -> None:
        await self._call("sendMessage", {"chat_id": self.chat_id, "text": text})

    async def run(self) -> None:
        """Poll until cancelled. A conflict (another poller, or a webhook set on
        the same bot) stops it for good: two consumers would each see half the
        taps, which is worse than one clear failure in the log."""
        try:
            await self.skip_backlog()
        except Exception as exc:  # noqa: BLE001 - a dead network at boot is not fatal
            print(f"[inbox] could not reach telegram at start: {exc}", flush=True)
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep listening through blips
                text = str(exc)
                if "409" in text or "Conflict" in text:
                    self.stopped_because = text
                    print(f"[inbox] stopped: another consumer holds this bot "
                          f"({text})", flush=True)
                    return
                print(f"[inbox] poll failed, retrying in {BACKOFF_S:.0f}s: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                await asyncio.sleep(BACKOFF_S)
