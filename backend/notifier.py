"""notifier.py — native macOS notifications, fired from the backend.

Why backend-fired: it works even when the browser is CLOSED (the server is the
one notifying, not a tab) — closing the "missed a proactive line while the tab was
shut" gap — and it lets the Keeper's own image ride along instead of the browser's
logo.

macOS reality: unsigned Python can't use the modern notification API (auth is
denied), so we shell out, best-first:
  1. Keeper.app -> a rebranded copy of terminal-notifier (built by
     scripts/build_keeper_notifier.sh) that carries the Keeper's OWN icon and name
     as the PRIMARY badge — no thumbnail needed, so the banner stays clean.
  2. plain terminal-notifier -> falls back to a -contentImage thumbnail (the only
     way to show the Keeper without the rebranded app), plus terminal-notifier's badge.
  3. osascript -> reliable, but a generic icon.
No-op off macOS.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ICON = _HERE / "static" / "keeper.png"
KEEPER_APP_BIN = _HERE / "notify" / "Keeper.app" / "Contents" / "MacOS" / "terminal-notifier"
_IS_MAC = sys.platform == "darwin"


def _binary() -> str | None:
    """The best terminal-notifier binary: the rebranded Keeper.app if built, else
    a plain terminal-notifier on PATH, else None."""
    if KEEPER_APP_BIN.exists():
        return str(KEEPER_APP_BIN)
    return shutil.which("terminal-notifier")


def available() -> str:
    """Which backend will be used: 'keeper.app' | 'terminal-notifier' | 'osascript' | 'none'."""
    if not _IS_MAC:
        return "none"
    if KEEPER_APP_BIN.exists():
        return "keeper.app"
    return "terminal-notifier" if shutil.which("terminal-notifier") else "osascript"


def _as_string(text: str) -> str:
    """Make `text` safe to sit inside an AppleScript double-quoted string."""
    out = text.replace("\\", "")          # drop escapes before anything else
    out = out.replace('"', "'")           # cannot close the string
    return " ".join(out.split())          # collapse newlines/tabs into spaces


def notify(title: str, message: str) -> None:
    """Fire one native notification. Blocking but fast (~tens of ms); call via
    asyncio.to_thread from an async loop. Never raises."""
    if not _IS_MAC:
        return
    tn = _binary()
    try:
        if tn:
            cmd = [tn, "-title", title, "-message", message, "-sound", "default"]
            # Keeper.app already shows the Keeper as the PRIMARY badge, so no
            # thumbnail is needed. Only the plain terminal-notifier fallback needs
            # -contentImage to show the Keeper at all.
            if tn != str(KEEPER_APP_BIN):
                cmd += ["-contentImage", str(ICON)]
            subprocess.run(cmd, timeout=5, capture_output=True)
        else:
            # osascript: the text is interpolated into an AppleScript string, so it
            # has to survive AppleScript's own escaping. Backslashes go FIRST: "\\"
            # is an escape character there, so a message ending in one would escape
            # the closing quote and let the string run on into the script. Newlines
            # would break the one-line -e form. (terminal-notifier above takes argv,
            # so it needs none of this.)
            t = _as_string(title)
            m = _as_string(message)
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{m}" with title "{t}"'],
                timeout=5, capture_output=True)
    except Exception as exc:  # noqa: BLE001 - a failed banner is never fatal
        print(f"[notify] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    print("backend:", available())
    notify("The Keeper", "You asked me to hold this. It's time.")
    print("fired — check the top-right")
