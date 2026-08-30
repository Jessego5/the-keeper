"""Tier 1 — the code sandbox (sandbox.py) + the run_python tool. Code-as-action."""
import pytest
import native_tools
import reminders as reminders_mod
from pathlib import Path

import sandbox

pytestmark = pytest.mark.unit


def test_runs_and_captures_output():
    r = sandbox.run_python("print(2 + 2)")
    assert r.ok and r.stdout.strip() == "4"


def test_computation():
    r = sandbox.run_python("print(sum(range(1, 101)))")
    assert r.stdout.strip() == "5050"


def test_bare_expression_is_auto_printed():
    # REPL-style: a bare expression on the last line returns its value (no print())
    assert sandbox.run_python("2 + 2").stdout.strip() == "4"


def test_auto_print_uses_computed_variable():
    r = sandbox.run_python("cost = 35\nweeks = 52\ncost / weeks")
    assert r.stdout.strip().startswith("0.673")   # the miss from the trace, fixed


def test_explicit_print_not_doubled():
    # a trailing print() call is left alone — its value (None) isn't re-printed
    assert sandbox.run_python("print(6 * 7)").stdout.strip() == "42"


def test_trailing_assignment_stays_silent():
    # an assignment isn't an expression -> nothing to auto-print
    assert sandbox.run_python("x = 5").as_text() == "(ran, no output)"


def test_error_is_captured_not_raised():
    r = sandbox.run_python("print(1/0)")
    assert not r.ok
    assert "ZeroDivisionError" in r.stderr


def test_empty_code():
    r = sandbox.run_python("")
    assert not r.ok and "no code" in r.stderr


def test_timeout_is_bounded():
    r = sandbox.run_python("while True: pass", timeout=2)
    assert r.timed_out and not r.ok
    assert "too long" in r.as_text()


def test_output_is_capped():
    r = sandbox.run_python("print('x' * 100000)")
    assert len(r.stdout) <= sandbox._OUTPUT_CAP


def test_as_text_ok_and_empty():
    assert sandbox.Result(True, "42\n", "").as_text() == "42"
    assert "no output" in sandbox.Result(True, "", "").as_text()


# --- via the tool (async) --- #

async def test_run_python_tool():
    nt = native_tools.NativeTools(
        reminders_mod.ReminderStore.__new__(reminders_mod.ReminderStore))
    nt.store.items = []          # minimal store without file I/O
    out = await nt.call("run_python", {"code": "print(6 * 7)"})
    assert out.strip() == "42"


# --- the fence's documented LIMITS, pinned by execution --- #

def test_cwd_and_home_are_a_throwaway_dir():
    """The fences that DO hold: relative paths and ~ never reach the real home."""
    out = sandbox.run_python(
        "import os\nprint(os.getcwd())\nprint(os.environ.get('HOME'))").stdout
    cwd, home = out.strip().splitlines()[:2]
    assert cwd != str(Path.home())
    # resolve(): macOS reports getcwd() through /private but leaves HOME as /var
    assert Path(home).resolve() == Path(cwd).resolve()   # ~ is the throwaway dir


def test_known_gap_filesystem_is_not_fenced():
    """NOT a wish — a pin on a documented limitation, so it cannot drift unnoticed.

    sandbox.py states the fence does not block the filesystem or the network, and
    that matters here because the Keeper fetches web pages and calls run_python in
    the SAME turn: attacker-controlled text reaches the model that writes this code.

    If real isolation ever lands (the Dockerfile with no host mount, a sandbox-exec
    profile), THIS TEST WILL FAIL. That failure is the signal to rewrite the threat
    model in sandbox.py — currently the only place a reader is told about the path
    from a fetched page to local code execution.
    """
    r = sandbox.run_python("import pathlib\nprint(pathlib.Path('/etc/hosts').exists())")
    assert r.stdout.strip() == "True", (
        "filesystem reads now appear fenced — update sandbox.py's threat model")
