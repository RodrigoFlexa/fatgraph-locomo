"""The scorer must reproduce the official LoCoMo numbers exactly."""

from __future__ import annotations

import pytest

from fgl.evaluation import (
    aggregate,
    evidence_recall,
    f1_multi,
    f1_score,
    is_abstention,
    markdown_table,
    normalize_answer,
    score_question,
)
from fgl.data.locomo import ABSTAIN_ANSWER, Question


# --------------------------------------------------------------------------- #
# Upstream parity                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The Cat, and a Dog!", "cat dog"),
        ("7 May, 2023", "7 may 2023"),
        ("  MIXED   Case  ", "mixed case"),
    ],
)
def test_normalize_answer_matches_upstream(raw, expected):
    assert normalize_answer(raw) == expected


def test_f1_is_token_level_and_symmetric_in_the_usual_way():
    assert f1_score("7 May 2023", "7 May 2023") == pytest.approx(1.0)
    assert f1_score("May 2023", "7 May 2023") == pytest.approx(2 * 2 / (2 + 3))
    assert f1_score("nothing common", "7 May 2023") == 0.0


def test_f1_multi_splits_on_commas_for_category_1():
    # every gold sub-answer finds its best matching predicted sub-answer
    assert f1_multi("Psychology, counseling", "counseling, Psychology") == pytest.approx(1.0)
    assert f1_multi("Psychology", "Psychology, counseling") == pytest.approx(0.5)


def test_category_3_uses_only_the_text_before_the_first_semicolon():
    q = Question("q", "Psychology; counseling certification", 3, [])
    assert score_question(q, "Psychology") == pytest.approx(1.0)


def test_adversarial_rule_is_a_substring_check():
    assert is_abstention("Not mentioned in the conversation")
    assert is_abstention("no information available about that")
    assert not is_abstention("She drives a red Toyota")

    q = Question("q", ABSTAIN_ANSWER, 5, [])
    assert score_question(q, ABSTAIN_ANSWER) == 1.0
    assert score_question(q, "a red Toyota") == 0.0


def test_unknown_category_is_a_hard_error():
    with pytest.raises(ValueError):
        score_question(Question("q", "a", 9, []), "a")


# --------------------------------------------------------------------------- #
# Recall                                                                       #
# --------------------------------------------------------------------------- #


def test_evidence_recall_matches_the_upstream_definition():
    assert evidence_recall(["D1:3", "D2:5"], ["D1:3"]) == pytest.approx(0.5)
    assert evidence_recall(["D1:3"], ["D9:9", "D1:3"]) == 1.0
    assert evidence_recall([], ["anything"]) == 1.0, "no evidence -> counted as 1"


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #


def test_aggregate_reports_every_category_separately():
    from fgl.evaluation import QAOutcome

    outcomes = [
        QAOutcome("a", 1, "x", "x", 1.0, recall={"recall@5": 1.0}),
        QAOutcome("b", 1, "x", "y", 0.0, recall={"recall@5": 0.0}),
        QAOutcome("c", 5, ABSTAIN_ANSWER, ABSTAIN_ANSWER, 1.0, abstained=True),
    ]
    agg = aggregate(outcomes)
    assert agg["per_category"]["multi-hop"]["f1"] == 0.5
    assert agg["per_category"]["multi-hop"]["recall@5"] == 0.5
    assert agg["per_category"]["adversarial"]["f1"] == 1.0
    assert agg["overall"]["f1_micro"] == pytest.approx(2 / 3, abs=1e-4)
    assert agg["overall"]["f1_macro"] == pytest.approx(0.75)


def test_markdown_table_lists_every_condition():
    results = {
        "B3-rag-facts": aggregate([]) | {
            "per_category": {"multi-hop": {"f1": 0.3, "n": 1}},
            "overall": {"f1_macro": 0.3, "f1_micro": 0.3},
        },
        "G1-fatgraph-min": aggregate([]) | {
            "per_category": {"multi-hop": {"f1": 0.4, "n": 1}},
            "overall": {"f1_macro": 0.4, "f1_micro": 0.4},
        },
    }
    table = markdown_table(results)
    assert "B3-rag-facts" in table and "G1-fatgraph-min" in table
    assert "multi-hop" in table
