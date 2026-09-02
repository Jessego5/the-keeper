"""Shared fixtures and tier gating for the Keeper test suite.

Tiers (see pyproject.toml markers):
  unit         — pure logic, always runnable
  integration  — OS sensors, MCP (needs node), running app
  eval         — LLM behavior, needs OPENAI_API_KEY (run explicitly)

Run:
  .venv/bin/pytest                     # unit + integration (evals excluded)
  .venv/bin/pytest -m unit             # fast logic only
  .venv/bin/pytest -m integration      # subsystems (auto-skips if node/mcp absent)
  .venv/bin/pytest tests/evals -m eval # LLM evals (needs OPENAI_API_KEY)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

HAVE_NODE = shutil.which("npx") is not None
try:
    import mcp  # noqa: F401
    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False


# --- capability skips ------------------------------------------------------- #

requires_node = pytest.mark.skipif(
    not (HAVE_NODE and HAVE_MCP),
    reason="needs node/npx + mcp SDK for a live MCP server")

requires_mac = pytest.mark.skipif(
    sys.platform != "darwin", reason="needs macOS")


# --- fixtures --------------------------------------------------------------- #

@pytest.fixture
def store(tmp_path):
    """A fresh, isolated MemoryStore backed by a temp file."""
    import memory
    return memory.MemoryStore(tmp_path / "facts.jsonl")


@pytest.fixture
def sandbox(tmp_path):
    """A folder with two known files, for MCP filesystem tests."""
    d = tmp_path / "sandbox"
    d.mkdir()
    (d / "journal.txt").write_text(
        "March 3 - sat with tea, did not paint.\n"
        "March 19 - Sam called, did not pick up.\n")
    (d / "list.txt").write_text("- gesso the canvas\n- reply to Sam\n")
    return d


@pytest.fixture
def awake():
    """A stubbed 'present, screen awake' sensor read."""
    import sensors
    return sensors.Presence(idle_seconds=30.0, screen_locked=False,
                            frontmost_app="test")


def silent_gen(system, user):
    """A generator that always declines to speak (for gate tests)."""
    return "SILENCE"


def echo_gen(text):
    """Build a generator that always returns `text` (for delivery tests)."""
    return lambda system, user: text
