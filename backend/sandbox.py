"""sandbox.py — run short Python the agent writes, safely enough. Code-as-action.

The most powerful action space is code. This lets the Keeper (and its analyst
sub-agent) COMPUTE — calculate, transform, parse, reason numerically — the things
fixed tools can't. It's the reference agent's shell.py, scoped down to Python and fenced.

Threat model, stated plainly: this runs code the MODEL writes to help the person, not
untrusted attacker input. The fences — a separate process in isolated mode (`-I`), a
hard wall-clock timeout, CPU + memory rlimits, a throwaway temp working directory, a
stripped environment, and capped output — stop runaways and accidents (infinite loops,
memory blowups, floods of output). What they do NOT do: block network access or the
filesystem, or defend against deliberately malicious code — that needs a container /
nsjail / macOS sandbox-exec profile. Kept this way on purpose: legible, dependency-free,
right-sized for a single-user companion running its own helper code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:
    import resource            # POSIX only
except ImportError:            # pragma: no cover - non-POSIX
    resource = None            # type: ignore

_CPU_SECONDS = 5               # hard CPU-time cap (kills infinite loops)
_MEM_BYTES = 512 * 1024 * 1024  # address-space cap where the OS enforces it
_OUTPUT_CAP = 4000             # chars of stdout/stderr kept
_DEFAULT_TIMEOUT = 8           # wall-clock seconds


@dataclass
class Result:
    ok: bool
    stdout: str
    stderr: str
    timed_out: bool = False

    def as_text(self) -> str:
        """A compact rendering for a tool result."""
        if self.timed_out:
            return "(timed out — the code took too long)"
        if self.ok:
            return self.stdout.strip() or "(ran, no output)"
        return f"(error)\n{(self.stderr or self.stdout).strip()}"


def _apply_limits() -> None:  # pragma: no cover - runs in the child process
    if resource is None:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_MEM_BYTES, _MEM_BYTES))
    except (ValueError, OSError):
        pass    # RLIMIT_AS is flaky on macOS; the timeout is the real backstop
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


def run_python(code: str, timeout: int = _DEFAULT_TIMEOUT) -> Result:
    """Run `code` in a fenced subprocess and capture its output. Never raises."""
    code = (code or "").strip()
    if not code:
        return Result(False, "", "(no code)")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=tmp, capture_output=True, text=True, timeout=timeout,
                preexec_fn=_apply_limits if os.name == "posix" else None,
                env={"PATH": "/usr/bin:/bin", "HOME": tmp},
            )
        except subprocess.TimeoutExpired:
            return Result(False, "", "", timed_out=True)
        except Exception as exc:  # noqa: BLE001
            return Result(False, "", f"{type(exc).__name__}: {exc}")
    out = (proc.stdout or "")[:_OUTPUT_CAP]
    err = (proc.stderr or "")[:_OUTPUT_CAP]
    return Result(ok=(proc.returncode == 0), stdout=out, stderr=err)
