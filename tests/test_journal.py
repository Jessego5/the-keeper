"""Tier 1 — the Keeper's append-only journal (journal.py) + its tools."""
import pytest
import journal as journal_mod
import native_tools
import reminders as reminders_mod

pytestmark = pytest.mark.unit


def test_keep_appends_and_persists(tmp_path):
    p = tmp_path / "kept.jsonl"
    j = journal_mod.Journal(p)
    j.keep("gessoed the canvas")
    j.keep("replied to Sam")
    j2 = journal_mod.Journal(p)                    # reload from disk
    assert [e.text for e in j2.entries] == ["gessoed the canvas", "replied to Sam"]


def test_keep_is_append_only(tmp_path):
    # writing a second note must NOT lose the first (no overwrite, ever)
    p = tmp_path / "kept.jsonl"
    j = journal_mod.Journal(p)
    j.keep("first")
    j.keep("second")
    j.keep("third")
    assert len(journal_mod.Journal(p).entries) == 3


def test_keep_ignores_empty(tmp_path):
    j = journal_mod.Journal(tmp_path / "k.jsonl")
    assert j.keep("   ") is None and j.entries == []


def test_recent_returns_tail(tmp_path):
    j = journal_mod.Journal(tmp_path / "k.jsonl")
    for i in range(15):
        j.keep(f"note {i}")
    recent = j.recent(5)
    assert [e.text for e in recent] == [f"note {i}" for i in range(10, 15)]


# --- via the tools (async) --- #

@pytest.fixture
def nt(tmp_path):
    rem = reminders_mod.ReminderStore(tmp_path / "r.jsonl")
    j = journal_mod.Journal(tmp_path / "kept.jsonl")
    return native_tools.NativeTools(rem, journal=j)


async def test_keep_note_tool(nt):
    out = await nt.call("keep_note", {"text": "buy ultramarine"})
    assert "kept in the journal" in out
    assert nt.journal.recent()[-1].text == "buy ultramarine"


async def test_read_journal_tool(nt):
    await nt.call("keep_note", {"text": "the gallery show is in July"})
    out = await nt.call("read_journal", {})
    assert "gallery show is in July" in out


async def test_journal_tools_graceful_without_store(tmp_path):
    bare = native_tools.NativeTools(reminders_mod.ReminderStore(tmp_path / "r.jsonl"))
    assert "no journal" in await bare.call("keep_note", {"text": "x"})
    assert "no journal" in await bare.call("read_journal", {})
