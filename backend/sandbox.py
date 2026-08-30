"""sandbox.py — run short Python the agent writes, safely enough. Code-as-action.

The most powerful action space is code. This lets the Keeper (and its analyst
sub-agent) COMPUTE — calculate, transform, parse, reason numerically — the things
fixed tools can't. It's the reference agent's shell.py, scoped down to Python and fenced.

Threat model, stated plainly. The fences — a separate process in isolated mode (`-I`),
a hard wall-clock timeout, CPU + memory rlimits, a throwaway temp working directory, a
stripped environment (HOME points at that temp dir), and capped output — stop runaways
and accidents: infinite loops, memory blowups, floods of output.

What they do NOT do: block the filesystem or the network. Measured, not assumed — code
run here can read backend/.env (which holds the API key) by absolute path, list the
real home directory, and open outbound sockets. The temp cwd/HOME only redirect
relative paths and `~`.

READ THIS BEFORE ASSUMING THE INPUT IS TRUSTED. An earlier version of this note said
the code here is written by the model "to help the person, not untrusted attacker
input". That is not true of this app. The Keeper's own showcase flow searches the web,
fetches a page, and calls run_python IN THE SAME TURN, so attacker-controlled text from
a fetched page reaches the same model that then writes this code. A prompt injection in
a page is therefore a path to reading local secrets and sending them out.

Closing that needs real isolation — the app's own Dockerfile with no host mount beyond
keeper_sandbox/, or a macOS sandbox-exec profile (fiddly: a naive deny-by-default
profile aborts CPython at startup) — or a rule that no turn may both fetch untrusted
content and run code. Until one of those lands, treat this as a single-user, local,
trusted-network tool and do not point it at hostile pages.
"""

from __future__ import annotations

import ast
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


def _autoprint(code: str) -> str:
    """If the last top-level statement is a bare expression, print its value — so
    REPL-style code the model naturally writes ('cost_per_week' on the last line, like
    Jupyter) actually returns something, instead of running silently. A trailing `print`
    or an assignment is left alone."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    tree.body[-1] = ast.Assign(
        targets=[ast.Name(id="__v", ctx=ast.Store())], value=last.value)
    tree.body.append(
        ast.parse("if __v is not None:\n    print(__v)").body[0])
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def run_python(code: str, timeout: int = _DEFAULT_TIMEOUT) -> Result:
    """Run `code` in a fenced subprocess and capture its output. Never raises."""
    code = (code or "").strip()
    if not code:
        return Result(False, "", "(no code)")
    code = _autoprint(code)      # REPL-style last-expression echo
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
