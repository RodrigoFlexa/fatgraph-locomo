"""Tests for the runnable scope conditions (docs/ASSUMPTIONS.md).

The point of these checks is that they FAIL on a corpus the method does not
suit -- a check that always passes declares nothing. So each test builds a
corpus that violates the condition and pins that the report says so, alongside
one that satisfies it.
"""

from __future__ import annotations

import pytest

from fgl.core import FatGraph
from fgl.data.locomo import Conversation, Question, Session, Turn
from fgl.evaluation.scope import (
    AUDIT,
    RUNTIME,
    check_degree_scale,
    check_episode_structure,
    check_evidence_belongs_to_named_actor,
    check_participants,
    check_question_names_one_actor,
    check_temporal_granularity,
    format_scope,
    run_scope,
)
from fgl.memory.slots import KIND_CONCEPT, KIND_EPISODE


def _conv(speakers, questions=(), sample_id="c1") -> Conversation:
    session = Session(num=1, date_time_raw="1:56 pm on 8 May, 2023",
                      timestamp="2023-05-08T13:56:00")
    session.turns = [
        Turn(f"D1:{i+1}", sp, f"{sp} said something about a shelter.", 1)
        for i, sp in enumerate(speakers)
    ]
    return Conversation(
        sample_id=sample_id,
        speaker_a=speakers[0],
        speaker_b=speakers[1] if len(speakers) > 1 else speakers[0],
        sessions=[session],
        questions=list(questions),
    )


def _q(text, evidence=(), category=1) -> Question:
    return Question(question=text, answer="x", category=category,
                    evidence=list(evidence))


# --------------------------------------------------------------------------- #
# S1                                                                           #
# --------------------------------------------------------------------------- #


def test_s1_holds_on_two_party_dialogue_and_fails_on_a_meeting():
    two = check_participants([_conv(["Jon", "Gina"])])
    assert two.holds is True and two.value == 2.0

    meeting = check_participants([_conv([f"P{i}" for i in range(9)])])
    assert meeting.holds is False
    assert "1/n_speakers" in meeting.degrades_to, (
        "a failing condition must name what the design falls back to"
    )


# --------------------------------------------------------------------------- #
# S2 / S3                                                                      #
# --------------------------------------------------------------------------- #


def test_s2_counts_questions_that_name_exactly_one_participant():
    conv = _conv(
        ["Jon", "Gina"],
        [
            _q("What did Jon adopt?"),
            _q("What did Gina cook?"),
            _q("What did they talk about?"),          # names nobody
            _q("What did Jon say to Gina?"),          # names both
        ],
    )
    c = check_question_names_one_actor([conv])
    assert c.kind == RUNTIME
    assert c.detail["named_one"] == 2
    assert c.detail["named_none"] == 1
    assert c.detail["named_multiple"] == 1
    assert c.holds is False, "0.5 is below the 0.80 criterion"


def test_s3_is_labelled_as_an_audit_because_it_needs_gold_evidence():
    """The dependence the design cannot engineer away -- only declare."""
    conv = _conv(
        ["Jon", "Gina"],
        [
            _q("What did Jon adopt?", evidence=["D1:1"]),   # Jon's own turn
            _q("What did Gina cook?", evidence=["D1:1"]),   # Jon's turn: a miss
        ],
    )
    c = check_evidence_belongs_to_named_actor([conv])
    assert c.kind == AUDIT
    assert c.detail == {"n_single_actor_questions": 2, "n_with_actor_evidence": 1}
    assert c.value == pytest.approx(0.5)
    assert c.holds is False


def test_audit_checks_do_not_count_toward_the_runtime_tally():
    """Mixing them would let a paper claim its assumptions were verified on
    data that had to be annotated first."""
    report = run_scope([_conv(["Jon", "Gina"], [_q("What did Jon adopt?")])])
    audits = [c for c in report["checks"] if c["kind"] == AUDIT]
    assert audits, "S3 must still be reported"
    assert report["runtime_conditions_total"] == len(
        [c for c in report["checks"]
         if c["kind"] == RUNTIME and c["holds"] is not None]
    )


# --------------------------------------------------------------------------- #
# S4                                                                           #
# --------------------------------------------------------------------------- #


def test_s4_reports_the_grain_without_gating_on_it():
    """The granularity parameter was deleted, so S4 records the distribution
    and never fails. A criterion here would be reintroducing the knob."""
    conv = _conv(
        ["Jon", "Gina"],
        [
            _q("What did Jon adopt in April 2022?"),
            _q("What happened on 7 May 2023?"),
            _q("What happened in 2023?"),
            _q("What is her favourite colour?"),
        ],
    )
    c = check_temporal_granularity([conv])
    assert c.value == {"day": 1, "month": 1, "year": 1}
    assert c.detail["n_with_a_date"] == 3
    assert c.holds is True
    assert "deleted" in c.degrades_to


# --------------------------------------------------------------------------- #
# S6 / S7                                                                      #
# --------------------------------------------------------------------------- #


def _episode_graph(shares) -> FatGraph:
    g = FatGraph()
    for i, share in enumerate(shares):
        g.add_vertex(name=f"ep{i}", vertex_id=f"ep:{i}",
                     meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{i}", f"D1:{i+1}"],
                           "speaker_content": share})
    return g


def test_s6_fails_on_a_monologue_corpus():
    dialogue = _episode_graph([{"a": 4, "b": 2}] * 5)
    assert check_episode_structure([dialogue]).holds is True

    monologue = _episode_graph([{"a": 6}] * 5)
    c = check_episode_structure([monologue])
    assert c.holds is False
    assert "sibling_frac" in c.degrades_to


def test_s7_flags_an_absolute_cut_off_that_is_eating_the_graph():
    from fgl.config import Config

    cfg = Config.load("L2")
    g = FatGraph()
    eps = [
        g.add_vertex(name=f"ep{i}", vertex_id=f"ep:{i}",
                     meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{i}"],
                           "speaker_content": {"a": 2, "b": 1}})
        for i in range(200)
    ]
    # every concept above the absolute cut-off of 60: the mechanism has
    # switched itself off without saying so
    for j in range(20):
        vid = g.add_vertex(name=f"c{j}", vertex_id=f"concept:c{j}",
                           meta={"kind": KIND_CONCEPT, "key": f"c{j}"})
        for i in range(120):
            g.add_edge(eps[i], vid, {"text": "t", "turn_ids": [f"D1:{i}"]})

    c = check_degree_scale([g], cfg)
    assert c.holds is False
    assert c.value[KIND_CONCEPT] == pytest.approx(1.0)
    derived = c.detail["derived_hub_degree_by_kind"][KIND_CONCEPT]
    assert derived > cfg.slots.hub_degree, (
        "the derived cut-off must follow the corpus up, not stay at 60"
    )


# --------------------------------------------------------------------------- #
# The report                                                                   #
# --------------------------------------------------------------------------- #


def test_report_states_the_degradation_path_for_every_condition():
    """A premise with no declared degradation path is a hidden requirement,
    not a premise."""
    report = run_scope([_conv(["Jon", "Gina"], [_q("What did Jon adopt?")])])
    for c in report["checks"]:
        if c.get("detail", {}).get("skipped"):
            continue
        assert c["statement"], c["id"]
        assert c["degrades_to"], f"{c['id']} declares no fallback"
        assert c["kind"] in (RUNTIME, AUDIT)


def test_format_scope_marks_the_audit_checks():
    report = run_scope([_conv(["Jon", "Gina"], [_q("What did Jon adopt?")])])
    text = format_scope(report)
    assert "audit (needs gold labels)" in text
    assert "runtime scope conditions hold" in text
