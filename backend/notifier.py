"""
This fires native macOS notifications from the backend.

Why backend-fired: it works even when the browser is CLOSED (the server is the
one notifying, not a tab), closing the "missed a proactive line while the tab was
shut" gap, and it lets the Keeper's own image ride along instead of the browser's
logo.

macOS reality: unsigned Python can't use the modern notification API, which
denies it authorisation, so it shells out to whatever is best available. First
choice is Keeper.app, a rebranded copy of terminal-notifier built by
scripts/build_keeper_notifier.sh, which carries the Keeper's OWN icon and name as
the PRIMARY badge, so no thumbnail is needed and the banner stays clean. Failing
that a plain terminal-notifier takes a -contentImage thumbnail, the only way to
show the Keeper at all without the rebranded app, alongside terminal-notifier's
own badge. Failing that osascript is reliable but shows a generic icon. It is a
no-op off macOS.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ICON = _HERE / "static" / "keeper.png"
_NOTIFY_DIR = _HERE / "notify"
KEEPER_APP_BIN = _NOTIFY_DIR / "Keeper.app" / "Contents" / "MacOS" / "terminal-notifier"
_IS_MAC = sys.platform == "darwin"

# The Keeper wears the weather it is speaking in. A banner's PRIMARY badge is the
# icon inside the .app bundle, fixed at build time: so a per-register badge means
# a bundle per register, not an argument. build_keeper_notifier.sh builds all of
# them; any that is missing simply falls back to the plain Keeper.app.
#
# Every state bundle is named "The Keeper.app" and sits in a directory named for
# its register. macOS labels both its permission prompt and the Notification
# Centre entry from the bundle's FILENAME rather than CFBundleDisplayName, so a
# bundle named for the state would introduce the companion as "Keeper-turning".
#
# Three registers, because there are three: frozen, tidal, turn. The concept
# sheet also carries a "resting" pose and this map once had a fourth entry for
# it, but no register ever produces "resting" - detect_state and persona both
# know only the three, and tidal IS the resting default. A bundle nothing can
# select is dead weight that implies a register the system does not have.
STATE_APPS = {
    "frozen": "frozen/The Keeper.app",
    "tidal": "tidal/The Keeper.app",
    "turn": "turning/The Keeper.app",
}
STATE_ICONS = {
    "frozen": "keeper_frozen.png",
    "tidal": "keeper_tidal.png",
    "turn": "keeper_turning.png",
}


def _icon_for(state: str | None) -> Path:
    """The still image for `state`, for the -contentImage fallback path."""
    name = STATE_ICONS.get(state or "")
    if name:
        candidate = _HERE / "static" / name
        if candidate.exists():
            return candidate
    return ICON


def _state_binary(state: str | None) -> str | None:
    """The binary of the bundle built for `state`, if that bundle exists."""
    app = STATE_APPS.get(state or "")
    if not app:
        return None
    candidate = _NOTIFY_DIR / app / "Contents" / "MacOS" / "terminal-notifier"
    return str(candidate) if candidate.exists() else None


def _binary(state: str | None = None) -> str | None:
    """The best terminal-notifier binary: the register's own Keeper bundle if it
    was built, else the plain Keeper.app, else a terminal-notifier on PATH."""
    return (_state_binary(state)
            or (str(KEEPER_APP_BIN) if KEEPER_APP_BIN.exists() else None)
            or shutil.which("terminal-notifier"))


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


def notify(title: str, message: str, state: str | None = None) -> None:
    """Fire one native notification, in the register's own likeness where a bundle
    for it was built. Blocking but fast (~tens of ms); call via asyncio.to_thread
    from an async loop. Never raises."""
    if not _IS_MAC:
        return
    tn = _binary(state)
    try:
        if tn:
            cmd = [tn, "-title", title, "-message", message, "-sound", "default"]
            # Whether to hang a thumbnail beside the badge, which is only ever
            # worth doing when it would say something the badge does not:
            #   a state bundle  -> already wearing this register. Nothing to add.
            #   Keeper.app      -> wearing the default pose, so carry the register
            #                      as a thumbnail IF we have a distinct one; with
            #                      no state icon it would just be the same figure
            #                      printed twice.
            #   terminal-notifier -> no Keeper at all in the badge. Always show one.
            icon = _icon_for(state)
            if tn == _state_binary(state):
                pass
            elif tn == str(KEEPER_APP_BIN):
                if icon != ICON:
                    cmd += ["-contentImage", str(icon)]
            else:
                cmd += ["-contentImage", str(icon)]
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
    print("fired, check the top-right")
