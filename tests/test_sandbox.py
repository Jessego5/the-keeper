"""Tier 1 — the code sandbox (sandbox.py) + the run_python tool. Code-as-action."""
import pytest
import native_tools
import reminders as reminders_mod
import sandbox

pytestmark = pytest.mark.unit


def test_runs_and_captures_output():
    r = sandbox.run_python("print(2 + 2)")
    assert r.ok and r.stdout.strip() == "4"


def test_computation():
    r = sandbox.run_python("print(sum(range(1, 101)))")
    assert r.stdout.strip() == "5050"


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
