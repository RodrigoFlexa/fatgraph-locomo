"""Tests for the corpus-derived thresholds -- condition L2d.

Each test pins one claim the calibration work makes, in the form the claim was
argued in ``docs/ASSUMPTIONS.md``, so a regression names the argument it breaks
rather than a line number:

* the derived hub cut-off is **scale-free** -- the property the absolute 60
  lacked, and the reason an absolute count is a latent bug rather than a matter
  of taste;
* the derived framing stoplist **recovers the hand-written list** on a
  templated question set -- the evidence that the estimator subsumes the hack
  instead of merely replacing it;
* the derived actor prior **sharpens as participants are added**, so a
  multi-party corpus does not need a re-sweep;
* multi-resolution time is a **strict generalisation**: with
  ``time_granularities=month`` the retriever behaves exactly as it did;
* ``calibration=absolute`` reproduces the swept numbers verbatim, which is what
  makes every measurement already in ``DECISIONS.md`` still true.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from fgl.config import Config, ConfigError
from fgl.core import FatGraph
from fgl.data.locomo import Conversation, Session, Turn
from fgl.llm import FakeLLM
from fgl.memory.calibration import (
    ABSOLUTE,
    DERIVED,
    actor_prior_from_graph,
    calibrate,
    concept_link_threshold_by_quantile,
    degrees_by_kind,
    derive_question_stop,
    hub_degree_by_quantile,
)
from fgl.memory.slots import (
    KIND_CONCEPT,
    KIND_EPISODE,
    KIND_TIME,
    LEGACY_QUESTION_NOUN_STOP,
    granularity_of,
    parse_granularities,
    question_time_slots,
    time_buckets,
)

pytest.importorskip("spacy")


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _conversation() -> Conversation:
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
def slot_cfg(cfg) -> Config:
    c = Config.load("L2")
    for attr in ("facts_cache", "graphs_dir", "results_dir", "logs_dir"):
        setattr(c.paths, attr, getattr(cfg.paths, attr))
    c.embeddings = cfg.embeddings
    c.llm = cfg.llm
    return c


@pytest.fixture
def derived_cfg(cfg) -> Config:
    c = Config.load("L2d")
    for attr in ("facts_cache", "graphs_dir", "results_dir", "logs_dir"):
        setattr(c.paths, attr, getattr(cfg.paths, attr))
    c.embeddings = cfg.embeddings
    c.llm = cfg.llm
    return c


def _build(config, embedder, prompts):
    from fgl.memory.ingest_slots import SlotIngestor

    ing = SlotIngestor(config, FakeLLM(config.llm), embedder, prompts)
    return ing.ingest(_conversation())


def _synthetic_graph(degree_by_vertex: dict[str, int], kind: str) -> FatGraph:
    """A star-free bipartite graph with exactly the degrees asked for.

    Built by hand rather than ingested: the point of the scale-freeness test is
    to control the degree distribution, and an ingest gives whatever the text
    gives.
    """
    g = FatGraph()
    n_ep = max(degree_by_vertex.values(), default=1)
    eps = [
        g.add_vertex(name=f"ep{i}", vertex_id=f"ep:{i}",
                     meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{i}"],
                           "speaker_content": {"a": 3, "b": 1}})
        for i in range(n_ep)
    ]
    for name, deg in degree_by_vertex.items():
        vid = g.add_vertex(name=name, vertex_id=f"{kind}:{name}",
                           meta={"kind": kind, "key": name})
        for i in range(deg):
            g.add_edge(eps[i], vid, {"text": f"{name} {i}", "turn_ids": [f"D1:{i}"]})
    return g


# --------------------------------------------------------------------------- #
# S7 -- the hub cut-off is scale-free                                          #
# --------------------------------------------------------------------------- #


def test_absolute_hub_degree_breaks_under_rescaling():
    """The bug the derived cut-off exists to fix, stated as a test.

    Two corpora with the SAME shape and different sizes: with an absolute
    cut-off of 60 the small one has no hubs at all and the large one is
    entirely hubs, so the mechanism silently switches itself off in both
    directions. This is not a matter of taste -- it is a latent failure that a
    config comment ("measure it on your own graph") acknowledged but could not
    prevent.
    """
    small = _synthetic_graph({f"c{i}": 2 + i % 5 for i in range(40)}, KIND_CONCEPT)
    large = _synthetic_graph({f"c{i}": 100 + i % 5 for i in range(40)}, KIND_CONCEPT)

    small_above = [d for d in degrees_by_kind(small)[KIND_CONCEPT] if d >= 60]
    large_above = [d for d in degrees_by_kind(large)[KIND_CONCEPT] if d >= 60]
    assert not small_above, "absolute 60 finds no hub in a small corpus"
    assert len(large_above) == 40, "absolute 60 calls every slot a hub in a large one"


def test_derived_hub_degree_selects_the_same_tail_at_any_scale():
    """The property the absolute number lacked: the share above the cut-off is
    a function of the DISTRIBUTION, not of the corpus size."""
    small = _synthetic_graph({f"c{i}": 2 + (i % 10) for i in range(60)}, KIND_CONCEPT)
    large = _synthetic_graph({f"c{i}": 20 * (2 + (i % 10)) for i in range(60)},
                             KIND_CONCEPT)

    t_small, ev_small = hub_degree_by_quantile(small, 0.90, 2, 60)
    t_large, ev_large = hub_degree_by_quantile(large, 0.90, 2, 60)

    assert t_large[KIND_CONCEPT] > t_small[KIND_CONCEPT], "threshold follows the scale"
    frac_s = ev_small["per_kind"][KIND_CONCEPT]["frac_above"]
    frac_l = ev_large["per_kind"][KIND_CONCEPT]["frac_above"]
    assert abs(frac_s - frac_l) <= 0.05, (
        f"the selected tail must be scale-invariant, got {frac_s} vs {frac_l}"
    )


def test_derived_hub_degree_is_per_kind(slot_cfg, embedder, prompts, derived_cfg):
    """An actor incident to half the episodes and a concept incident to half
    the episodes are not the same event, and one absolute cut-off cannot say
    so. The derived cut-offs therefore differ by kind."""
    graph, _ = _build(derived_cfg, embedder, prompts)
    thresholds, _ = hub_degree_by_quantile(graph, 0.99, 2, 60)
    assert set(thresholds) >= {"actor", "concept", "predicate", "time"}


def test_hub_quantile_falls_back_on_a_tiny_kind():
    """Below a handful of vertices a quantile is an artefact of the sample.
    The fallback is recorded, never silent."""
    tiny = _synthetic_graph({"c0": 3, "c1": 4}, KIND_CONCEPT)
    thresholds, ev = hub_degree_by_quantile(tiny, 0.99, 8, 60)
    assert thresholds[KIND_CONCEPT] == 60
    assert ev["per_kind"][KIND_CONCEPT]["source"] == "fallback"


# --------------------------------------------------------------------------- #
# S5 -- the derived stoplist recovers the hand-written one                     #
# --------------------------------------------------------------------------- #


def test_derived_stoplist_separates_template_words_from_topic_words():
    """The claim the whole estimator rests on.

    A topic word is common in the questions BECAUSE it is common in the
    conversations, so both frequencies move together. A template word is common
    in the questions and absent from what anyone said. The ratio -- not the
    frequency -- is what tells them apart, which is why a plain
    document-frequency cut-off would keep "dog" and drop nothing useful.
    """
    question_df = {
        "conversation": 0.098,  # template: everywhere in questions
        "date": 0.059,          # template
        "type": 0.023,          # template, and too rare for a df-only rule
        "dog": 0.030,           # topic: also common in the memory
        "dance competition": 0.012,
    }
    memory_df = {
        "conversation": 0.002,
        "date": 0.0,
        "type": 0.001,
        "dog": 0.055,
        "dance competition": 0.020,
    }
    stop, ev = derive_question_stop(question_df, memory_df, min_df=0.01,
                                    min_ratio=3.0)
    assert {"conversation", "date", "type"} <= stop
    assert "dog" not in stop
    assert "dance competition" not in stop
    assert ev["top"][0]["word"] in {"conversation", "date"}


def test_derived_stoplist_recovers_words_from_the_legacy_list():
    """Evidence that the mechanism subsumes the hack rather than replacing it.

    Everything selected here is already in the hand-written list -- i.e. the
    estimator does not merely find *a* set of words, it finds the ones the
    author found by reading errors, without being told what a template looks
    like.
    """
    question_df = {w: 0.06 for w in ("conversation", "date", "answer", "type")}
    question_df["shelter"] = 0.04
    memory_df = {"shelter": 0.08}
    stop, _ = derive_question_stop(question_df, memory_df, 0.01, 3.0)
    assert stop <= LEGACY_QUESTION_NOUN_STOP
    assert len(stop) == 4


def test_a_frequency_only_rule_would_not_have_worked():
    """Why the estimator is a RATIO and not a document-frequency cut-off.

    Any df threshold that catches "type" (2.3% of questions) also catches every
    topic word above 2.3%, and any threshold that spares the topic words misses
    "type" and "answer". The contrast against the memory is what breaks the
    tie, so this test fails the moment someone simplifies it away.
    """
    question_df = {"type": 0.023, "dog": 0.030}
    memory_df = {"type": 0.001, "dog": 0.055}
    only_df = {w for w, d in question_df.items() if d >= 0.023}
    assert only_df == {"type", "dog"}, "a df-only rule cannot separate these"

    stop, _ = derive_question_stop(question_df, memory_df, 0.01, 3.0)
    assert stop == {"type"}


# --------------------------------------------------------------------------- #
# S1/S3 -- the actor prior re-derives itself                                   #
# --------------------------------------------------------------------------- #


def _graph_with_speakers(n_speakers: int) -> FatGraph:
    g = FatGraph()
    share = {f"s{i}": 10 for i in range(n_speakers)}
    for e in range(6):
        g.add_vertex(name=f"ep{e}", vertex_id=f"ep:{e}",
                     meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{e}"],
                           "speaker_content": dict(share)})
    return g


def test_actor_prior_sharpens_as_participants_are_added():
    """The behaviour the absolute 0.35 could not have.

    Naming one of eight participants excludes far more of the corpus than
    naming one of two, so the prior SHOULD be stronger there -- and with
    `floor = 1/n_speakers` it becomes stronger on its own, with no sweep and no
    second corpus to fit on.
    """
    floor2, full2, ev2 = actor_prior_from_graph(_graph_with_speakers(2))
    floor8, _full8, ev8 = actor_prior_from_graph(_graph_with_speakers(8))

    assert ev2["n_speakers"] == 2 and ev8["n_speakers"] == 8
    assert floor8 < floor2, "more participants must mean a stronger prior"
    assert floor2 == pytest.approx(0.5)
    assert floor8 == pytest.approx(0.125)
    assert 0.0 < full2 <= 1.0


def test_actor_prior_full_follows_the_typical_dominant_share():
    """`full` is the point at which 'this exchange is theirs' is already true,
    which is a property of the corpus rather than a constant."""
    g = FatGraph()
    for e, share in enumerate(({"a": 9, "b": 1}, {"a": 8, "b": 2}, {"a": 7, "b": 3})):
        g.add_vertex(name=f"ep{e}", vertex_id=f"ep:{e}",
                     meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{e}"],
                           "speaker_content": share})
    _floor, full, _ev = actor_prior_from_graph(g)
    assert full == pytest.approx(0.8), "median of 0.9 / 0.8 / 0.7"


# --------------------------------------------------------------------------- #
# The concept-link threshold                                                   #
# --------------------------------------------------------------------------- #


def test_concept_link_threshold_follows_the_embedding_geometry():
    """An absolute cosine is a property of the ENCODER as much as of the task.
    A corpus whose concepts sit closer together must get a higher bar, or the
    paraphrase fallback links everything to everything."""
    rng = np.random.default_rng(0)
    spread = rng.normal(size=(120, 24))
    spread /= np.linalg.norm(spread, axis=1, keepdims=True)
    # a tight corpus: every concept is a small perturbation of one direction
    tight = np.tile(spread[0], (120, 1)) + 0.15 * rng.normal(size=(120, 24))
    tight /= np.linalg.norm(tight, axis=1, keepdims=True)

    loose_cut, _ = concept_link_threshold_by_quantile(spread, 0.99, 0.0)
    tight_cut, _ = concept_link_threshold_by_quantile(tight, 0.99, 0.0)
    assert tight_cut > loose_cut


def test_concept_link_threshold_respects_its_floor():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(80, 16))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    cut, ev = concept_link_threshold_by_quantile(m, 0.99, 0.55)
    assert cut >= 0.55
    assert ev["source"] == "derived"


# --------------------------------------------------------------------------- #
# S4 -- multi-resolution time                                                  #
# --------------------------------------------------------------------------- #


def test_time_buckets_index_every_level():
    assert time_buckets(datetime(2023, 5, 7)) == ["2023", "2023-05", "2023-05-07"]
    assert time_buckets(datetime(2023, 5, 7), ("month",)) == ["2023-05"]
    assert time_buckets(None) == []


def test_granularity_is_recoverable_from_the_key_alone():
    assert granularity_of("2023") == "year"
    assert granularity_of("2023-05") == "month"
    assert granularity_of("2023-05-07") == "day"
    assert granularity_of("") == ""


def test_question_emits_the_level_it_names_plus_backoff():
    """Finest first, coarser behind it -- so a question asking by day still
    reaches a memory that only knows the month."""
    assert question_time_slots("What did James adopt on 7 May 2023?") == [
        "2023-05-07", "2023-05", "2023",
    ]
    assert question_time_slots("What happened in April 2022?") == ["2022-04", "2022"]
    assert question_time_slots("What happened in 2023?") == ["2023"]
    assert question_time_slots("What is her favourite colour?") == []


def test_month_only_is_exactly_the_old_behaviour():
    """The change is a strict generalisation: pinning the old granularity has
    to reproduce the old function, or L2's reported numbers stop being about
    the thing that was measured."""
    from fgl.memory.slots import question_time_buckets

    for q in (
        "What did James adopt in April 2022?",
        "What happened in 2023?",
        "What is her favourite colour?",
        "What did she do on 7 May 2023?",
    ):
        assert question_time_slots(q, ("month",)) == question_time_buckets(q)


def test_parse_granularities_orders_and_rejects():
    assert parse_granularities("day,year") == ("year", "day")
    assert parse_granularities(None) == ("year", "month", "day")
    with pytest.raises(ValueError):
        parse_granularities("fortnight")


def test_ingest_builds_time_vertices_at_every_configured_level(
    derived_cfg, embedder, prompts
):
    graph, report = _build(derived_cfg, embedder, prompts)
    levels = {
        granularity_of(vx.meta.get("key", ""))
        for vx in graph.vertices.values()
        if vx.meta.get("kind") == KIND_TIME
    }
    assert {"year", "month", "day"} <= levels
    assert report.graph_stats["time_granularities"] == ["year", "month", "day"]


# --------------------------------------------------------------------------- #
# Provenance and reproducibility                                               #
# --------------------------------------------------------------------------- #


def test_absolute_mode_reproduces_the_swept_numbers(slot_cfg, embedder, prompts):
    """L2 must not have changed under anyone. Every number already reported in
    DECISIONS.md was produced with these literals, so `calibration=absolute`
    returning anything else would make those numbers false."""
    graph, _ = _build(slot_cfg, embedder, prompts)
    cal = calibrate(slot_cfg, graph)
    assert cal.concept_link_threshold == slot_cfg.slots.concept_link_threshold
    assert cal.actor_prior_floor == slot_cfg.slots.actor_prior_floor
    assert cal.actor_prior_full == slot_cfg.slots.actor_prior_full
    assert cal.hub_degree("concept") == slot_cfg.slots.hub_degree
    assert cal.question_noun_stop == LEGACY_QUESTION_NOUN_STOP
    assert set(cal.source.values()) == {ABSOLUTE}


def test_derived_mode_records_its_provenance(derived_cfg, embedder, prompts):
    """A number without provenance is indistinguishable from a number someone
    typed. The report has to say which happened, per knob."""
    graph, report = _build(derived_cfg, embedder, prompts)
    cal = calibrate(derived_cfg, graph)
    assert DERIVED in set(cal.source.values())
    assert "hub_degree" in cal.evidence
    recorded = report.graph_stats["calibration"]
    assert recorded["source"]["hub_degree"] == DERIVED
    assert "n_question_noun_stop" in recorded


def test_l2_pins_the_swept_form_and_l2d_does_not():
    """The two conditions have to differ in exactly the intended way, or the
    L2-vs-L2d comparison stops measuring the calibration debt."""
    l2 = Config.load("L2")
    l2d = Config.load("L2d")
    assert (l2.slots.calibration, l2.slots.question_stop,
            l2.slots.time_granularities) == ("absolute", "literal", "month")
    assert (l2d.slots.calibration, l2d.slots.question_stop) == ("derived", "derived")
    assert parse_granularities(l2d.slots.time_granularities) == (
        "year", "month", "day"
    )
    # everything that encodes a claim about the MODEL rather than the corpus
    # must be identical, or the comparison confounds two changes
    for knob in ("dense_weight", "actor_weight", "predicate_weight",
                 "concept_weight", "type_weight", "time_weight",
                 "mention_weight", "sibling_frac", "slot_damping",
                 "set_orbit_boost", "corner_actor_min"):
        assert getattr(l2.slots, knob) == getattr(l2d.slots, knob), knob
    assert l2.retrieval.budget_tokens == l2d.retrieval.budget_tokens


def test_config_rejects_a_nonsense_calibration_setting():
    c = Config.load("L2")
    c.slots.calibration = "vibes"
    with pytest.raises(ConfigError):
        c.validate()
    c = Config.load("L2")
    c.slots.time_granularities = "fortnight"
    with pytest.raises(ConfigError):
        c.validate()
    c = Config.load("L2")
    c.slots.question_stop_ratio = 0.5
    with pytest.raises(ConfigError):
        c.validate()


def test_retriever_uses_the_calibration_not_the_literal(
    derived_cfg, embedder, prompts
):
    """The wiring test. A derived threshold that the retriever never reads is
    a comment, not a mechanism."""
    from fgl.retrieval.slots import SlotRetriever

    graph, _ = _build(derived_cfg, embedder, prompts)
    r = SlotRetriever(graph, embedder, derived_cfg, {})
    assert r.calibration.source["hub_degree"] in ("derived", "fallback")
    # the retriever must query the levels the GRAPH has, not the ones the
    # config happens to name
    assert set(r.time_granularities) == {"year", "month", "day"}
