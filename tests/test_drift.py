"""Tier 1 — drift / idle self-reflection (drift.py). No key: stub generator."""
import time
import pytest
import drift
import memory

pytestmark = pytest.mark.unit


def _stub(system, user):
    return "They have been quiet. I keep what they gave me."


@pytest.fixture
def store(tmp_path):
    s = memory.MemoryStore(tmp_path / "f.jsonl")
    s.add("Is a painter; stopped in March.", "identity")
    return s


@pytest.fixture
def log(tmp_path):
    return drift.ReflectionLog(tmp_path / "reflections.jsonl")


def test_reflect_produces_and_logs(store, log):
    r = drift.reflect(store, _stub, log)
    assert r.text and len(log.items) == 1
    assert log.latest().text == r.text


def test_reflection_log_persists(tmp_path):
    p = tmp_path / "reflections.jsonl"
    l1 = drift.ReflectionLog(p)
    l1.add(drift.Reflection(text="a quiet note"))
    l2 = drift.ReflectionLog(p)                     # reload from disk
    assert len(l2.items) == 1 and l2.items[0].text == "a quiet note"


def test_maybe_drift_respects_interval(store, log):
    cfg = drift.DriftConfig(min_interval_min=180)
    now = time.time()
    # just drifted 10 min ago -> should NOT drift again
    assert drift.maybe_drift(store, _stub, log, last_drift_at=now - 600,
                             config=cfg, now=now) is None
    # last drift 4h ago -> should drift
    assert drift.maybe_drift(store, _stub, log, last_drift_at=now - 4 * 3600,
                             config=cfg, now=now) is not None


def test_disabled_never_drifts(store, log):
    cfg = drift.DriftConfig(enabled=False)
    assert drift.maybe_drift(store, _stub, log, last_drift_at=None, config=cfg) is None
