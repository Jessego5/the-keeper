"""Tier 1 — answer scoring (metrics.py). Pure string work, no key, no model."""
import pytest
import metrics

pytestmark = pytest.mark.unit


# --- normalisation --- #

def test_normalize_drops_punctuation_articles_and_case():
    assert metrics.normalize("The Chicago.") == "chicago"
    assert metrics.normalize("  a  PAINTER,  ") == "painter"


def test_normalize_survives_empty_and_none():
    assert metrics.normalize("") == ""
    assert metrics.normalize(None) == ""


# --- exact match --- #

def test_exact_match_ignores_surface_differences():
    assert metrics.exact_match("The Chicago.", "chicago")


def test_exact_match_is_still_strict_about_content():
    assert not metrics.exact_match("chicago", "portland")
    assert not metrics.exact_match("chicago illinois", "chicago")


# --- token f1 --- #

def test_f1_gives_partial_credit_for_a_right_answer_in_a_sentence():
    """The case substring checks get wrong in the generous direction and exact
    match gets wrong in the strict one."""
    assert 0.0 < metrics.token_f1("they live in chicago now", "chicago") < 1.0


def test_f1_is_one_for_the_same_answer_worded_the_same():
    assert metrics.token_f1("chicago", "The Chicago.") == 1.0


def test_f1_is_zero_for_a_wrong_answer():
    assert metrics.token_f1("portland", "chicago") == 0.0


def test_f1_handles_empty_without_dividing_by_zero():
    assert metrics.token_f1("", "chicago") == 0.0
    assert metrics.token_f1("chicago", "") == 0.0
    assert metrics.token_f1("", "") == 1.0


def test_f1_is_symmetric():
    a, b = "started painting again", "she has started painting"
    assert metrics.token_f1(a, b) == pytest.approx(metrics.token_f1(b, a))


def test_f1_does_not_reward_padding():
    """Recall alone would score a rambling answer perfectly; precision is what
    stops "everything and also chicago" from counting as a clean answer."""
    tight = metrics.token_f1("chicago", "chicago")
    padded = metrics.token_f1("well it might be chicago or anywhere really", "chicago")
    assert padded < tight


# --- the model judge --- #

def test_judge_reads_a_lenient_yes():
    for raw in ("YES", "yes", " Yes.", '"YES"', "Y"):
        assert metrics.reads_as_yes(raw)


def test_judge_reads_anything_else_as_no():
    for raw in ("NO", "no", "", "   ", "unclear", None):
        assert not metrics.reads_as_yes(raw)


def test_judge_without_a_model_scores_wrong_not_right():
    """An eval that treats an outage as a pass goes green exactly when it has
    stopped being able to see."""
    assert metrics.judge_answer("q", "gold", "answer", None) is False


def test_judge_treats_an_empty_answer_as_wrong():
    assert not metrics.judge_answer("q", "gold", "", lambda s, u: "YES")
    assert not metrics.judge_answer("q", "gold", "   ", lambda s, u: "YES")


def test_judge_failure_is_a_no_not_a_crash():
    def boom(system, user):
        raise RuntimeError("rate limited")
    assert metrics.judge_answer("q", "gold", "an answer", boom) is False


def test_judge_passes_question_gold_and_answer_to_the_model():
    seen = {}
    def spy(system, user):
        seen["user"] = user
        return "YES"
    metrics.judge_answer("where do they live?", "chicago", "up north now", spy)
    assert "where do they live?" in seen["user"]
    assert "chicago" in seen["user"] and "up north now" in seen["user"]


# --- aggregate --- #

def test_score_answers_averages_both_metrics():
    out = metrics.score_answers(["chicago", "portland"], ["chicago", "chicago"])
    assert out["n"] == 2
    assert out["exact_match"] == 0.5
    assert out["token_f1"] == 0.5


def test_score_answers_on_nothing():
    assert metrics.score_answers([], [])["n"] == 0


def test_score_answers_refuses_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics.score_answers(["a"], ["a", "b"])
