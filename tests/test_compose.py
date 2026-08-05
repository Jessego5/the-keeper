"""Tier 1 — compose.revoice: tool answers re-voiced without losing facts."""
import pytest
import compose

pytestmark = pytest.mark.unit


def test_revoice_uses_the_model_output():
    def gen(system, user):
        # a stand-in "voiced" rewrite that keeps the fact (the number 3)
        assert "re-voice" in system.lower()      # the instruction is in the prompt
        return "The record shows three commits, held in the deep."
    out = compose.revoice("You have 3 commits.", generate=gen)
    assert "three" in out


def test_revoice_falls_back_on_empty_output():
    # if the model returns nothing, keep the original answer (never lose the fact)
    out = compose.revoice("The file lists: milk, bread, eggs.",
                          generate=lambda s, u: "   ")
    assert out == "The file lists: milk, bread, eggs."


def test_revoice_falls_back_on_error():
    def boom(system, user):
        raise RuntimeError("model down")
    out = compose.revoice("last commit: fix the notifier", generate=boom)
    assert out == "last commit: fix the notifier"


def test_revoice_empty_input():
    assert compose.revoice("", generate=lambda s, u: "whatever") == ""


def test_revoice_passes_facts_to_the_model():
    # the original factual text must reach the model as the user turn, verbatim
    seen = {}
    def gen(system, user):
        seen["user"] = user
        return "voiced"
    compose.revoice("commit a1b2c3 at 09:00", generate=gen)
    assert seen["user"] == "commit a1b2c3 at 09:00"
