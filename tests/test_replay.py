"""Tier 1 — the register replay harness (replay_register.py). No key, no model:
these pin the extraction and the shape of what it emits, not any classification.
"""
import json

import pytest

import replay_register as rr

pytestmark = pytest.mark.unit


@pytest.fixture
def sessions(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"messages": [
        {"role": "user", "content": "i feel stuck"},
        {"role": "assistant", "content": "The ice holds."},
        {"role": "user", "content": "what time is it"},
    ]}))
    (d / "b.json").write_text(json.dumps({"messages": [
        {"role": "user", "content": "I feel stuck"},      # same, different case
        {"role": "user", "content": "  "},                # empty
        {"role": "user", "content": "hello"},
    ]}))
    (d / "broken.json").write_text("{not json")
    return d


def test_extracts_user_turns_only(sessions):
    out = rr.user_messages(sessions)
    assert "The ice holds." not in out


def test_deduplicates_case_insensitively_and_keeps_order(sessions):
    out = rr.user_messages(sessions)
    assert out == ["i feel stuck", "what time is it", "hello"]


def test_a_corrupt_session_file_is_skipped_not_fatal(sessions):
    """One unreadable file must not cost the whole corpus."""
    assert rr.user_messages(sessions)


def test_candidates_leave_the_label_empty():
    """The predictions come from the system under test, so a candidate must arrive
    UNLABELLED — promoting its own guesses would be marking its own homework."""
    rows = rr.classify_both(["i feel stuck"])
    assert rows[0]["predicted"] in (None, "frozen", "tidal", "turn")
    assert set(rows[0]) == {"text", "keyword", "model2vec", "predicted", "why"}


def test_already_labelled_reads_the_real_dataset():
    known = rr.already_labelled()
    assert "i am sad" in known, "expected the shipped dataset to be readable"


# --- guards on the output file --- #

def test_refuses_to_overwrite_labelled_work(tmp_path):
    """Regression, from doing exactly this: re-running the harness silently
    overwrote a fully labelled candidates file. Labelling is the expensive part,
    and the harness can always regenerate what it produced."""
    out = tmp_path / "cands.json"
    out.write_text(json.dumps([{"text": "hello", "register": "neutral"}]))
    assert rr._has_labels(out)


def test_an_unlabelled_file_is_free_to_overwrite(tmp_path):
    out = tmp_path / "cands.json"
    out.write_text(json.dumps([{"text": "hello", "register": ""}]))
    assert not rr._has_labels(out)


def test_a_missing_or_corrupt_file_does_not_block(tmp_path):
    assert not rr._has_labels(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert not rr._has_labels(bad)
