"""
This sets up the Keeper's Telegram bot, once, from the token alone.

Everything after BotFather's /newbot used to be by hand: upload the avatar in a chat
with BotFather, paste in the command list, write the descriptions, then open
getUpdates in a browser and dig a chat id out of raw JSON. The Bot API can do all of
it (setMyProfilePhoto arrived in February 2026), so this does:

    the avatar        the Keeper's icon, cropped head-and-shoulders for a circle
    the menu          /off /quiet /usual /eager, the dial (see backend/dial.py)
    the words         the description and the short line on the profile
    the owner         TELEGRAM_CHAT_ID, found and written into backend/.env

The owner is the part worth doing carefully. A bot's username is public, so "the
first chat to message the bot" is whoever finds it first. Instead this prints a link
carrying a one-time code, t.me/<bot>?start=<code>; pressing Start there sends
"/start <code>", and only the chat that sends that code is accepted.

Run it while the Keeper is NOT running: both read the same bot's updates, and
Telegram hands each update to one reader only.

    .venv/bin/python scripts/telegram_setup.py              # everything
    .venv/bin/python scripts/telegram_setup.py --no-photo   # keep the current avatar
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "backend" / ".env"
ICON = ROOT / "backend" / "static" / "keeper.png"

COMMANDS = [
    ("off", "start nothing on its own"),
    ("quiet", "speak up less"),
    ("usual", "the default"),
    ("eager", "speak up more"),
]
DESCRIPTION = ("A companion that decides on its own when to speak. Talk to it here, "
               "approve what it wants to do, and turn it down with /quiet or /off.")
SHORT_DESCRIPTION = "A companion that decides on its own when to speak."

# Head and shoulders, in keeper.png's 512px space: the face and the lamp stay
# readable at the size Telegram shows avatars in a chat list.
PORTRAIT_BOX = (126, 28, 366, 268)

Call = Callable[[str, dict, Optional[dict]], dict]


# --------------------------------------------------------------------------- #
# The Bot API, stdlib only.
# --------------------------------------------------------------------------- #

def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """multipart/form-data, for the one call that uploads a file."""
    boundary = "keeper" + secrets.token_hex(12)
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; "
                  f'name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data, ctype) in files.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; "
                  f'name="{name}"; filename="{filename}"\r\n'
                  f"Content-Type: {ctype}\r\n\r\n".encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def make_caller(token: str) -> Call:
    def call(method: str, params: dict, files: Optional[dict] = None) -> dict:
        url = f"https://api.telegram.org/bot{token}/{method}"
        if files:
            body, ctype = _multipart(params, files)
        else:
            body = urllib.parse.urlencode(params).encode()
            ctype = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"content-type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:  # noqa: S310
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            # Telegram explains itself in the body of a 4xx; that is the useful part.
            try:
                return json.loads(exc.read())
            except ValueError:
                return {"ok": False, "description": f"HTTP {exc.code}"}
    return call


def _ok(resp: dict, what: str) -> bool:
    if resp.get("ok"):
        return True
    print(f"· {what} failed: {resp.get('description', 'no reason given')}", flush=True)
    return False


# --------------------------------------------------------------------------- #
# The avatar.
# --------------------------------------------------------------------------- #

def avatar_jpg(icon: Path = ICON) -> Optional[bytes]:
    """The Keeper's icon as a square .jpg, head and shoulders, on its own teal.

    JPG because setMyProfilePhoto takes nothing else for a still photo, and solid
    because the icon's corners are transparent, which JPG cannot hold. Pillow when
    it is installed (it is pinned in requirements.lock); macOS's sips otherwise,
    which converts but cannot crop; None if neither is there."""
    try:
        from PIL import Image
    except ImportError:
        return _sips_jpg(icon)
    src = Image.open(icon).convert("RGBA")
    bg = Image.new("RGBA", src.size, src.getpixel((256, 30))[:3] + (255,))
    bg.alpha_composite(src)
    out = io.BytesIO()
    bg.convert("RGB").crop(PORTRAIT_BOX).resize((640, 640), Image.LANCZOS).save(
        out, "JPEG", quality=92)
    return out.getvalue()


def _sips_jpg(icon: Path) -> Optional[bytes]:
    if not shutil.which("sips"):
        return None
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "keeper.jpg"
        done = subprocess.run(["sips", "-s", "format", "jpeg", str(icon), "--out",
                               str(dst)], capture_output=True, check=False)
        return dst.read_bytes() if done.returncode == 0 and dst.exists() else None


# --------------------------------------------------------------------------- #
# The owner.
# --------------------------------------------------------------------------- #

def owner_from(updates: list, code: str) -> Optional[dict]:
    """The private chat that sent "/start <code>", or None. Any other message, from
    anyone, is not the owner, whatever it says."""
    for u in updates:
        msg = u.get("message") or {}
        chat = msg.get("chat") or {}
        if chat.get("type") != "private":
            continue
        if (msg.get("text") or "").strip() == f"/start {code}":
            return chat
    return None


def wait_for_owner(call: Call, code: str, wait_s: float,
                   clock: Callable[[], float] = time.monotonic) -> Optional[dict]:
    """Long-poll until the coded /start arrives. Every update read is acknowledged
    on the way, so the Keeper does not later find this exchange in its backlog."""
    offset = None
    deadline = clock() + wait_s
    while clock() < deadline:
        params: dict = {"timeout": 20,
                        "allowed_updates": json.dumps(["message"])}
        if offset is not None:
            params["offset"] = offset
        resp = call("getUpdates", params, None)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("description") or "getUpdates refused")
        updates = resp.get("result") or []
        if updates:
            offset = updates[-1]["update_id"] + 1
        chat = owner_from(updates, code)
        if chat is not None:
            call("getUpdates", {"offset": offset, "timeout": 0}, None)   # ack
            return chat
    return None


def set_env_var(path: Path, key: str, value: str) -> None:
    """Set KEY=value in a .env file: replace the line if it is there, else append.
    Every other line, comments and secrets included, is left exactly as it was."""
    lines = path.read_text().splitlines() if path.exists() else []
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #

def read_env(path: Path = ENV_PATH) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def main(argv: Optional[list] = None, call: Optional[Call] = None,
         env_path: Path = ENV_PATH) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--no-photo", action="store_true",
                    help="leave the bot's current avatar alone")
    ap.add_argument("--wait", type=float, default=600.0,
                    help="seconds to wait for you to press Start")
    args = ap.parse_args(argv)

    env = read_env(env_path)
    token = env.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("· set TELEGRAM_BOT_TOKEN in backend/.env first (from @BotFather, "
              "/newbot)", flush=True)
        return 1
    call = call or make_caller(token)

    me = call("getMe", {}, None)
    if not _ok(me, "checking the token"):
        return 1
    username = me["result"]["username"]
    print(f"· bot: @{username}", flush=True)

    if not args.no_photo:
        jpg = avatar_jpg()
        if jpg is None:
            print("· no way to make a .jpg here; skipping the avatar", flush=True)
        elif _ok(call("setMyProfilePhoto",
                      {"photo": json.dumps({"type": "static",
                                            "photo": "attach://avatar"})},
                      {"avatar": ("keeper.jpg", jpg, "image/jpeg")}), "the avatar"):
            print("· avatar set", flush=True)
    if _ok(call("setMyCommands", {"commands": json.dumps(
            [{"command": c, "description": d} for c, d in COMMANDS])}, None),
           "the command menu"):
        print("· command menu set: /off /quiet /usual /eager", flush=True)
    if (_ok(call("setMyDescription", {"description": DESCRIPTION}, None),
            "the description")
            and _ok(call("setMyShortDescription",
                         {"short_description": SHORT_DESCRIPTION}, None),
                    "the short description")):
        print("· descriptions set", flush=True)

    if env.get("TELEGRAM_CHAT_ID"):
        print("· TELEGRAM_CHAT_ID is already set; leaving it. Done.", flush=True)
        return 0

    code = f"{secrets.randbelow(10**6):06d}"
    print(f"\n· open this and press Start, from your own Telegram:\n\n"
          f"  https://t.me/{username}?start={code}\n", flush=True)
    try:
        chat = wait_for_owner(call, code, args.wait)
    except RuntimeError as exc:
        why = str(exc)
        if "Conflict" in why or "409" in why:
            why += " (is the Keeper running? Stop it and run this again.)"
        print(f"· could not read messages: {why}", flush=True)
        return 1
    if chat is None:
        print("· timed out waiting for Start. Run this again when ready.", flush=True)
        return 1
    set_env_var(env_path, "TELEGRAM_CHAT_ID", str(chat["id"]))
    who = chat.get("first_name") or chat.get("username") or "you"
    call("sendMessage", {"chat_id": chat["id"],
                         "text": "The Keeper is connected. Messages from this chat "
                                 "are the only ones it will answer."}, None)
    print(f"· connected to {who} (chat {chat['id']}); TELEGRAM_CHAT_ID written to "
          f"backend/.env. Restart the Keeper to start using it.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
