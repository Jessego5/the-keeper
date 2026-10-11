"""
These are the Tier 1 tests for compose.revoice: tool answers re-voiced without losing
facts.
"""
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


# --- the re-voice guard: a name or a number must survive the voice --- #

def test_a_revoice_that_loses_a_name_is_not_used():
    """Seen live: "the note to Sam" came back as "a note from J". Sam was gone,
    and the plain answer is what they get instead."""
    original = "The note to Sam says: I can't make Thursday, but Saturday works. J"
    out = compose.revoice(original, generate=lambda s, u: (
        "You have a note from J saying they cannot meet Thursday, but Saturday "
        "works."))
    assert out == original


def test_a_revoice_that_loses_a_time_is_not_used():
    original = "Your dentist is at 3:15 on Friday."
    out = compose.revoice(original, generate=lambda s, u: "The dentist waits, Friday.")
    assert out == original


def test_a_faithful_revoice_is_kept():
    out = compose.revoice("Your dentist is at 3:15 on Friday.",
                          generate=lambda s, u: "Friday at 3:15, the dentist.")
    assert out == "Friday at 3:15, the dentist."
