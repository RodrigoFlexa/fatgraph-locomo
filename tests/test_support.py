"""The support attestation -- the abstention decision, moved into the graph.

What is pinned here, in the order it matters:

* the score's two halves combine the way the design says (the gate is
  conjunctive because its terms are refutations; the evidence is averaged
  because none of its terms is decisive), and no weight exists to tune;
* the cut is label-free -- Otsu on the corpus's own histogram -- and a
  degenerate distribution yields 0.0, which deletes nothing, rather than an
  invented valley;
* the four shapes are produced by the structural conditions they claim, not by
  a question-template classifier;
* ``support.enabled: false`` is a byte-for-byte no-op, so L1-L6 keep their
  numbers.

See :mod:`fgl.retrieval.support` and ``docs/PROPOSTA_ATESTADO.md``.
"""

from __future__ import annotations

import pytest

from fgl.config import Config, ConfigError
from fgl.retrieval.support import (
    REASON_EMPTY_CORNER,
    REASON_LOW_SUPPORT,
    SHAPE_ABSENT,
    SHAPE_COMPOSED,
    SHAPE_CONFLICT,
    SHAPE_DIRECT,
    SupportInputs,
    attest,
    auc,
    calibrate_threshold,
    concentration,
    cooccurrence,
    margin,
    operating_curve,
    otsu_threshold,
    support_score,
    vocabulary_presence,
)

# --------------------------------------------------------------------------- #
# Features                                                                     #
# --------------------------------------------------------------------------- #


def test_vocabulary_presence_is_a_fraction_and_undefined_when_nothing_was_asked():
    assert vocabulary_presence(3, 3) == 1.0
    assert vocabulary_presence(3, 1) == pytest.approx(1 / 3)
    assert vocabulary_presence(2, 0) == 0.0
    assert vocabulary_presence(0, 0) is None, "no content words is not a vote"


def test_cooccurrence_is_asymmetric_on_purpose():
    """A rare slot fully inside a common one is full co-occurrence.

    "the painting Melanie mentioned" sits inside "Melanie", not the other way
    round, and normalising by the larger orbit would call that weak support.
    """
    rare = frozenset({"e1"})
    common = frozenset({"e1", "e2", "e3", "e4"})
    assert cooccurrence({"a": rare, "b": common}) == 1.0
    assert cooccurrence({"a": frozenset({"e9"}), "b": common}) == 0.0


def test_cooccurrence_is_undefined_with_a_single_slot():
    assert cooccurrence({"a": frozenset({"e1"})}) is None, "undefined, not zero"
    assert cooccurrence({}) is None


def test_concentration_separates_a_spike_from_a_plateau():
    spike = concentration([10.0, 0.1, 0.1, 0.1], top_k=4)
    plateau = concentration([1.0, 1.0, 1.0, 1.0], top_k=4)
    assert plateau == pytest.approx(0.0, abs=1e-9)
    assert spike > 0.5
    assert concentration([1.0], top_k=4) is None


def test_margin_is_the_relative_gap():
    assert margin([10.0, 5.0]) == pytest.approx(0.5)
    assert margin([10.0, 10.0]) == pytest.approx(0.0)
    assert margin([1.0]) is None
    assert margin([0.0, 0.0]) == 0.0


# --------------------------------------------------------------------------- #
# The score                                                                    #
# --------------------------------------------------------------------------- #


def _inputs(**kw) -> SupportInputs:
    base = dict(
        asked_specific=2, linked_specific=2, corner=1.0, corner_reason="",
        slot_orbits={"s1": frozenset({"e1"}), "s2": frozenset({"e1", "e2"})},
        candidate_scores=[5.0, 1.0, 0.5],
        episode_slots={"e1": frozenset({"s1", "s2"}), "e2": frozenset({"s2"})},
        episode_sessions={"e1": "S1", "e2": "S1"},
        dense_top=0.6,
        is_set=False,
    )
    base.update(kw)
    return SupportInputs(**base)


def test_the_gate_is_conjunctive():
    """One refutation is enough: a zeroed corner cannot be averaged away."""
    full, _ = support_score(_inputs())
    refuted, feats = support_score(_inputs(corner=0.0))
    assert full > 0.0
    assert refuted == 0.0
    assert feats["gate"] == 0.0


def test_absent_vocabulary_also_zeroes_the_gate():
    score, feats = support_score(_inputs(asked_specific=2, linked_specific=0,
                                         slot_orbits={}))
    assert feats["vocabulary"] == 0.0
    assert score == 0.0


def test_the_evidence_is_averaged_not_multiplied():
    """A single weak soft feature must degrade the score, not annihilate it."""
    strong, _ = support_score(_inputs(dense_top=0.9))
    weak, feats = support_score(_inputs(dense_top=0.0))
    assert 0.0 < weak < strong, "averaging keeps a weak term from being fatal"
    assert feats["dense_top"] == 0.0


def test_a_set_question_is_not_punished_for_a_flat_distribution():
    """The orbit enumeration answers set questions; flat IS the right shape.

    Reading that as weak support would abstain on exactly the questions
    `enumerate_sets` exists to answer.
    """
    flat = [1.0, 1.0, 1.0, 1.0]
    ranked, _ = support_score(_inputs(candidate_scores=flat, is_set=False))
    as_set, feats = support_score(_inputs(candidate_scores=flat, is_set=True))
    assert as_set > ranked
    assert "concentration" not in feats and "margin" not in feats


def test_the_score_has_no_weights():
    """Regression guard for the whole calibration line (D30).

    Every number the score reports is a feature or a combinator, never a
    coefficient -- there is nothing here for a sweep to fit.
    """
    _, feats = support_score(_inputs())
    assert set(feats) <= {
        "vocabulary", "corner", "cooccurrence", "concentration", "margin",
        "dense_top", "gate", "strength",
    }


# --------------------------------------------------------------------------- #
# The verdict                                                                  #
# --------------------------------------------------------------------------- #


def test_a_structural_refutation_wins_over_the_score():
    a = attest(_inputs(corner=0.0, corner_reason=REASON_EMPTY_CORNER))
    assert a.shape == SHAPE_ABSENT
    assert a.reason == REASON_EMPTY_CORNER
    assert a.abstains is True
    assert a.witness == [], "an absent verdict names no evidence"


def test_a_low_score_abstains_with_its_own_reason():
    a = attest(_inputs(), threshold=0.99)
    assert a.shape == SHAPE_ABSENT
    assert a.reason == REASON_LOW_SUPPORT


def test_threshold_zero_abstains_on_nothing():
    """The default must never delete an answer by accident."""
    a = attest(_inputs(corner=0.0001), threshold=0.0)
    assert a.shape != SHAPE_ABSENT


def test_direct_when_one_episode_carries_the_whole_tuple():
    a = attest(_inputs())
    assert a.shape == SHAPE_DIRECT
    assert a.witness == ["e1"]


def test_composed_when_only_a_pair_carries_it():
    a = attest(_inputs(
        slot_orbits={"s1": frozenset({"e1"}), "s2": frozenset({"e2"})},
        episode_slots={"e1": frozenset({"s1"}), "e2": frozenset({"s2"})},
        episode_sessions={"e1": "S1", "e2": "S2"},
    ))
    assert a.shape == SHAPE_COMPOSED
    assert sorted(a.witness) == ["e1", "e2"], "both sides of the join are the witness"


def test_conflict_when_two_full_witnesses_sit_in_different_sessions():
    a = attest(_inputs(
        slot_orbits={"s1": frozenset({"e1", "e2"}), "s2": frozenset({"e1", "e2"})},
        episode_slots={"e1": frozenset({"s1", "s2"}), "e2": frozenset({"s1", "s2"})},
        episode_sessions={"e1": "S1", "e2": "S7"},
    ))
    assert a.shape == SHAPE_CONFLICT
    assert len(a.witness) == 2


def test_two_full_witnesses_in_the_same_session_are_not_a_conflict():
    a = attest(_inputs(
        slot_orbits={"s1": frozenset({"e1", "e2"}), "s2": frozenset({"e1", "e2"})},
        episode_slots={"e1": frozenset({"s1", "s2"}), "e2": frozenset({"s1", "s2"})},
        episode_sessions={"e1": "S1", "e2": "S1"},
    ))
    assert a.shape == SHAPE_DIRECT


def test_attestation_serialises_every_number_that_produced_it():
    d = attest(_inputs()).as_dict()
    assert set(d) == {"shape", "score", "reason", "threshold", "features", "witness"}
    assert d["features"]["gate"] >= 0.0


# --------------------------------------------------------------------------- #
# Where the cut comes from                                                     #
# --------------------------------------------------------------------------- #


def test_otsu_finds_the_valley_of_a_bimodal_distribution():
    low = [0.05 + 0.01 * i for i in range(40)]     # ~0.05 - 0.44
    high = [0.70 + 0.005 * i for i in range(60)]   # ~0.70 - 0.99
    t = otsu_threshold(low + high)
    assert max(low) < t < min(high), "the cut must land in the empty band"


def test_otsu_refuses_to_invent_a_valley():
    """A single mode has no valley; returning one would delete correct answers."""
    assert otsu_threshold([0.5] * 100) == 0.0
    assert otsu_threshold([]) == 0.0
    assert otsu_threshold([0.4]) == 0.0


def test_otsu_needs_no_labels():
    """Same scores, opposite labels -> same cut. The cut cannot be fitted."""
    scores = [0.1] * 30 + [0.9] * 70
    assert otsu_threshold(scores) == otsu_threshold(list(reversed(scores)))


def test_calibrate_reports_its_provenance():
    scores = [0.1] * 30 + [0.9] * 70
    t, src = calibrate_threshold(scores, method="otsu")
    assert src == "otsu" and 0.1 < t < 0.9

    t, src = calibrate_threshold([0.5] * 10, method="otsu", floor=0.0)
    assert (t, src) == (0.0, "fallback"), "degenerate -> abstain on nothing"

    t, src = calibrate_threshold(scores, method="quantile", quantile=0.3)
    assert src == "quantile"

    assert calibrate_threshold(scores, method="absolute", floor=0.42) == (0.42, "absolute")

    with pytest.raises(ValueError, match="unknown"):
        calibrate_threshold(scores, method="magic")


# --------------------------------------------------------------------------- #
# The two-sided objective                                                      #
# --------------------------------------------------------------------------- #


def test_auc_is_one_for_perfect_separation_and_half_for_none():
    assert auc([0.9, 0.8, 0.7], [0.1, 0.2, 0.3]) == 1.0
    assert auc([0.1, 0.2, 0.3], [0.9, 0.8, 0.7]) == 0.0
    assert auc([0.5] * 4, [0.5] * 4) == 0.5, "ties count half"
    assert auc([], [0.1]) == 0.5


def test_the_curve_does_the_arithmetic_the_tables_invite_you_to_skip():
    """282 multi-hop against 446 adversarial: both columns or neither."""
    sub = [0.9] * 1540
    adv = [0.1] * 446
    rows = operating_curve(sub, adv, thresholds=[0.5])
    row = rows[0]
    assert row["adversarial_caught"] == 1.0
    assert row["substantive_deleted"] == 0.0
    # 446 questions won at (1 - 0.5762) each, nothing destroyed
    assert row["net_questions"] == pytest.approx(446 * (1 - 0.5762), abs=0.5)
    assert row["net_micro"] > 0.09


def test_a_cut_that_deletes_answers_shows_as_a_loss():
    sub = [0.1] * 1540
    adv = [0.1] * 446
    row = operating_curve(sub, adv, thresholds=[0.5])[0]
    assert row["substantive_deleted"] == 1.0
    assert row["net_questions"] < 0, "catching everything by abstaining is not a win"


def test_a_threshold_below_everything_is_a_no_op():
    row = operating_curve([0.5] * 10, [0.5] * 10, thresholds=[0.0])[0]
    assert row["net_questions"] == 0.0


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


def test_support_is_off_by_default():
    cfg = Config.load("L2d")
    assert cfg.support.enabled is False, "L1-L6 must stay byte-identical"
    assert cfg.support.method == "otsu"


def test_support_requires_the_typed_slot_ingest():
    with pytest.raises(ConfigError, match="ingest.mode=slots"):
        Config.load("G1", overrides=["support.enabled=true"])


@pytest.mark.parametrize(
    "override,match",
    [
        ("support.method=magic", "support.method"),
        ("support.quantile=1.5", "support.quantile"),
        ("support.floor=2.0", "support.floor"),
        ("support.bins=2", "support.bins"),
        ("support.top_k=1", "support.top_k"),
    ],
)
def test_support_config_is_validated(override, match):
    with pytest.raises(ConfigError, match=match):
        Config.load("L2d", overrides=["support.enabled=true", override])


# --------------------------------------------------------------------------- #
# Wiring: on a real graph                                                      #
# --------------------------------------------------------------------------- #

def _conversation():
    from fgl.data.locomo import Conversation, Session, Turn

    session = Session(num=1, date_time_raw="1:56 pm on 8 May, 2023",
                      timestamp="2023-05-08T13:56:00")
    session.turns = [
        Turn("D1:1", "Jon", "How did the dance competition go last weekend?", 1),
        Turn("D1:2", "Gina", "We just did a contemporary piece called Finding Freedom.", 1),
        Turn("D1:3", "Jon", "I adopted a pup from a shelter in Stamford.", 1),
        Turn("D1:4", "Gina", "Roasted chicken is one of my favorites, I cooked it again.", 1),
    ]
    return Conversation(
        sample_id="conv-test", speaker_a="Jon", speaker_b="Gina",
        sessions=[session], questions=[],
    )


@pytest.fixture
def graph_cfg(cfg, embedder, prompts):
    pytest.importorskip("spacy")  # in the fixture: the unit tests above run anyway
    from fgl.llm import FakeLLM
    from fgl.memory.ingest_slots import SlotIngestor

    c = Config.load("L2d")
    for attr in ("facts_cache", "graphs_dir", "results_dir", "logs_dir"):
        setattr(c.paths, attr, getattr(cfg.paths, attr))
    c.embeddings = cfg.embeddings
    c.llm = cfg.llm
    graph, _ = SlotIngestor(c, FakeLLM(c.llm), embedder, prompts).ingest(_conversation())
    return graph, c


def test_disabled_support_is_a_byte_identical_no_op(graph_cfg, embedder):
    """The guarantee L1-L6 depend on: an untouched section changes nothing."""
    from fgl.retrieval.slots import SlotRetriever

    graph, cfg = graph_cfg
    q = "What did Jon adopt?"

    cfg.support.enabled = False
    off = SlotRetriever(graph, embedder, cfg, {}).retrieve(q)
    assert off.support_shape == ""
    assert off.support_score == 0.0

    cfg.support.enabled = True
    cfg.support.abstain = False
    on = SlotRetriever(graph, embedder, cfg, {}).retrieve(q)
    assert on.support_shape != "", "the attestation is computed"
    assert [f.text for f in on.facts] == [f.text for f in off.facts], (
        "measuring support must not change what is retrieved"
    )
    assert on.tokens_used == off.tokens_used


def test_the_attestation_scores_a_supported_question_above_an_unsupported_one(
    graph_cfg, embedder
):
    """The separation the whole proposal rests on, on a real graph.

    Jon adopted the pup; Gina did not. Both questions retrieve something -- the
    budget is always spent -- so a scorer that cannot tell them apart is the
    proposal dying.
    """
    from fgl.retrieval.slots import SlotRetriever

    graph, cfg = graph_cfg
    cfg.support.enabled = True
    cfg.support.abstain = False
    r = SlotRetriever(graph, embedder, cfg, {})

    supported = r.retrieve("What did Jon adopt?")
    unsupported = r.retrieve("What did Gina adopt?")
    assert supported.support_score > unsupported.support_score


def test_acting_on_absent_empties_the_context(graph_cfg, embedder):
    from fgl.retrieval.slots import SlotRetriever

    graph, cfg = graph_cfg
    cfg.support.enabled = True
    cfg.support.abstain = True
    r = SlotRetriever(graph, embedder, cfg, {})
    r.support_threshold = 1.01  # above any possible score

    result = r.retrieve("What did Jon adopt?")
    assert result.facts == []
    assert result.support_shape == SHAPE_ABSENT
    assert result.abstain_reason == REASON_LOW_SUPPORT


def test_abstain_false_reports_without_deleting(graph_cfg, embedder):
    """How the operating curve is measured before it is paid for."""
    from fgl.retrieval.slots import SlotRetriever

    graph, cfg = graph_cfg
    cfg.support.enabled = True
    cfg.support.abstain = False
    r = SlotRetriever(graph, embedder, cfg, {})
    r.support_threshold = 1.01

    result = r.retrieve("What did Jon adopt?")
    assert result.support_shape == SHAPE_ABSENT
    assert result.facts, "reporting an absent verdict must not act on it"
