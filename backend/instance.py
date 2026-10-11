"""
This is the guarantee that there is only one Keeper.

On 2026-08-03 the Keeper reached out 39 times in nine hours. Nothing in it was
broken: three copies were running, each with its own proactive loop, one of them
stuck at 600x, because "restart" had been starting new servers without stopping
the old. Every gate the Keeper has (the day's ceiling, the cooldown, the real-time
floor) is per process, so three processes had three of each.

The fix the postmortem deferred was to refuse a second server on an occupied port.
A port is the wrong thing to hold: dev.sh serves on 8737 and run.sh on 8790, and
both read and write the same memory_store/. What must be single is the memory, so
the lock is on the memory: an OS file lock on memory_store/keeper.lock, held for
the life of the process. The OS drops it when the process dies, however it dies,
so there is no stale lock file to clean up after a crash and no pid to check.

A second Keeper refuses to start and says who holds the lock. Under launchd that
means it retries quietly until the other one stops, then takes over, which is the
behaviour a supervisor wants.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Optional

STORE_DIR = Path(__file__).resolve().parent.parent / "memory_store"
LOCK_PATH = STORE_DIR / "keeper.lock"


class AlreadyRunning(RuntimeError):
    pass


class InstanceLock:
    def __init__(self, path: Path = LOCK_PATH):
        self.path = path
        self._fd: Optional[int] = None

    def acquire(self, port: str = "") -> None:
        """Take the lock or raise AlreadyRunning naming the holder. The holder's
        pid and port are written into the file for exactly that message."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.read(fd, 200).decode(errors="replace").strip()
            os.close(fd)
            raise AlreadyRunning(
                f"another Keeper already holds {self.path} ({holder or 'unknown'}). "
                f"Two would mean two proactive loops on one memory: stop that one "
                f"first.") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()}{f' port {port}' if port else ''}".encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
