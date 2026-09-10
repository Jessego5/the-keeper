"""
These are the Keeper's senses, read-only macOS presence signals.

It answers three cheap questions, none of which trips a permission dialog:
idle_seconds, how long since the human last touched keyboard or mouse;
screen_locked, whether they have stepped away; and frontmost_app, which app has
focus, by name only and never by window contents.

Three rules govern it. It is read-only, observing and never acting. It asks for no
frightening permissions, because NSWorkspace.frontmostApplication() gives the app
NAME without Accessibility access, and only reading INSIDE windows would need
that, which it deliberately never does: that boundary is the whole privacy story.
And it degrades to an empty dict off macOS or when a framework is missing, so
nothing downstream has to special-case the platform.

What it feeds is the water-state read, where idle at 2am reads frozen and an
active daytime reads tidal; the proactive gate, which never speaks into a locked
screen; and the Keeper's voice, for which the focused app is a concrete, on-voice
detail.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

_IS_MAC = sys.platform == "darwin"

# Import the macOS frameworks once, tolerantly. If pyobjc isn't present we simply
# lose the lock/app signals and keep idle (which is pure shell).
# Why an import failed, kept so capabilities() can say WHY a signal is missing
# instead of just reporting it absent.
_IMPORT_ERRORS: dict[str, str] = {}

try:  # pragma: no cover - platform dependent
    from AppKit import NSWorkspace  # type: ignore
except Exception as exc:  # noqa: BLE001
    NSWorkspace = None
    _IMPORT_ERRORS["frontmost_app"] = f"{type(exc).__name__}: {exc}"

try:  # pragma: no cover - platform dependent
    from Quartz import CGSessionCopyCurrentDictionary  # type: ignore
except Exception as exc:  # noqa: BLE001
    CGSessionCopyCurrentDictionary = None
    _IMPORT_ERRORS["screen_locked"] = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class Presence:
    """One read of the machine. Any field may be None if unreadable."""

    idle_seconds: float | None = None
    screen_locked: bool | None = None
    frontmost_app: str | None = None

    @property
    def available(self) -> bool:
        """True if we got at least one real signal."""
        return any(v is not None for v in (
            self.idle_seconds, self.screen_locked, self.frontmost_app))

    def to_context_line(self) -> str:
        """Render for the system prompt's context block. Empty string if blind.

        Kept factual and terse; the model is told elsewhere never to recite it.
        """
        if not self.available:
            return ""
        bits: list[str] = []
        if self.idle_seconds is not None:
            bits.append(f"idle for {_human_duration(self.idle_seconds)}")
        if self.screen_locked is not None:
            bits.append("screen locked" if self.screen_locked else "screen awake")
        if self.frontmost_app:
            bits.append(f"focused app: {self.frontmost_app}")
        return "; ".join(bits)


# --------------------------------------------------------------------------- #
# Individual reads. Each returns None rather than raising.
# --------------------------------------------------------------------------- #

def read_idle_seconds() -> float | None:
    """Seconds since the last HID (keyboard/mouse) event.

    Parses `ioreg -c IOHIDSystem`'s HIDIdleTime, which is in nanoseconds. Pure
    shell, no dependency, no permission.
    """
    if not _IS_MAC:
        return None
    try:
        out = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "HIDIdleTime" in line:
            # line looks like:  "HIDIdleTime" = 113780083
            try:
                ns = int(line.split("=")[-1].strip())
            except ValueError:
                return None
            return ns / 1_000_000_000
    return None


def read_screen_locked() -> bool | None:
    """True if the login session's screen is locked. No permission needed."""
    if not _IS_MAC or CGSessionCopyCurrentDictionary is None:
        return None
    try:
        d = CGSessionCopyCurrentDictionary()
    except Exception:  # noqa: BLE001
        return None
    if not d:
        return None
    return bool(d.get("CGSSessionScreenIsLocked", 0))


def read_frontmost_app() -> str | None:
    """Localized name of the focused app (e.g. 'Safari'). Name only, never
    window titles or contents, so no Accessibility prompt."""
    if not _IS_MAC or NSWorkspace is None:
        return None
    try:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
    except Exception:  # noqa: BLE001
        return None
    if app is None:
        return None
    name = app.localizedName()
    return str(name) if name else None


def read() -> Presence:
    """Take one full read of the machine. Never raises."""
    return Presence(
        idle_seconds=read_idle_seconds(),
        screen_locked=read_screen_locked(),
        frontmost_app=read_frontmost_app(),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _human_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 90:
        return f"{s}s"
    m = s // 60
    if m < 90:
        return f"{m}m"
    h = m // 60
    if h < 48:
        return f"{h}h"
    return f"{h // 24}d"


if __name__ == "__main__":
    p = read()
    print("Presence:", p)
    print("context line:", repr(p.to_context_line()))
    print("available:", p.available)


def capabilities() -> dict[str, str]:
    """Which presence signals this machine can actually provide, {name: status},
    where status is "live" or the reason it is not.

    The individual readers return None on failure and stay SILENT on purpose: the
    proactive loop reads presence every few seconds, so logging per failure would
    flood the log. The cost of that silence is that a permanently blind sensor is
    invisible: the "house" panel just shows defaults and looks like it works. This
    is read once at startup and logged, the way the mood classifier reports itself.
    """
    if not _IS_MAC:
        reason = f"unsupported platform ({sys.platform}); macOS only"
        return {name: reason
                for name in ("idle_seconds", "screen_locked", "frontmost_app")}

    out: dict[str, str] = {}
    out["idle_seconds"] = ("live" if read_idle_seconds() is not None
                           else "ioreg gave no HIDIdleTime")
    if CGSessionCopyCurrentDictionary is None:
        out["screen_locked"] = _IMPORT_ERRORS.get("screen_locked", "Quartz unavailable")
    else:
        out["screen_locked"] = ("live" if read_screen_locked() is not None
                                else "CGSessionCopyCurrentDictionary gave nothing")
    if NSWorkspace is None:
        out["frontmost_app"] = _IMPORT_ERRORS.get("frontmost_app", "AppKit unavailable")
    else:
        out["frontmost_app"] = ("live" if read_frontmost_app() is not None
                                else "NSWorkspace gave no frontmost app")
    return out
