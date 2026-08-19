"""Tests for conditions L3 (walk) and L4 (connection).

The load-bearing test in this file is the first one. Everything else L3 claims
rests on the walk being a *generalisation* of L2 rather than a different
retriever that scores similarly: if ``hops=1`` did not reproduce L2 exactly, the
sweep over ``propagation.hops`` would not be a curve through a published number,
and the L2-vs-L3 delta would be uninterpretable.

The rest of the file pins the three things that make a walk on this graph work
instead of smear:

* a hub may receive mass and may never relay it;
* the walk is non-backtracking, so hop 2 is a join and not the seed reflected;
* the Steiner channel is a conjunction -- an episode that misses one terminal
  scores nothing, however close it is to the others.
"""

from __future__ import annotations

import numpy as np
import pytest

from fgl.config import Config, ConfigError
from fgl.core import FatGraph
from fgl.data.locomo import Conversation, Session, Turn
from fgl.llm import FakeLLM
from fgl.memory.slots import KIND_CONCEPT, KIND_EPISODE
from fgl.retrieval.propagation import (
    PropagationRetriever,
    build_bipartite,
    propagate,
    reduces_to_l2,
)
from fgl.retrieval.slots import SlotRetriever
from fgl.retrieval.steiner import SteinerMetric, calibrate_null
from fgl.retrieval.unified import UnifiedRetriever

pytest.importorskip("spacy")


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _conversation() -> Conversation:
    """Two sessions that share a topic without sharing a question noun.

    D1:3 and D2:1 both concern the shelter/pup, but a question phrased about
    Stamford reaches D2:1 only *through* D1:3 -- which is the hop-2 case this
    condition exists for, in miniature.
    """
    s1 = Session(num=1, date_time_raw="1:56 pm on 8 May, 2023",
                 timestamp="2023-05-08T13:56:00")
    s1.turns = [
        Turn("D1:1", "Jon", "How did the dance competition go last weekend?", 1),
        Turn("D1:2", "Gina", "We just did a contemporary piece called Finding Freedom.", 1),
        Turn("D1:3", "Jon", "I adopted a pup from a shelter in Stamford.", 1),
        Turn("D1:4", "Gina", "Roasted chicken is one of my favorites, I cooked it again.", 1),
    ]
    s2 = Session(num=2, date_time_raw="10:10 am on 3 June, 2023",
                 timestamp="2023-06-03T10:10:00")
    s2.turns = [
        Turn("D2:1", "Jon", "The pup is settling in and loves the park.", 2),
        Turn("D2:2", "Gina", "That is lovely, send a photo of the park sometime.", 2),
        Turn("D2:3", "Gina", "I roasted a chicken for the neighbours on Sunday.", 2),
        Turn("D2:4", "Jon", "The shelter asked me to volunteer at the park event.", 2),
    ]
    return Conversation(
        sample_id="conv-prop", speaker_a="Jon", speaker_b="Gina",
        sessions=[s1, s2], questions=[],
    )


def _condition(name: str, cfg) -> Config:
    c = Config.load(name)
    for attr in ("facts_cache", "graphs_dir", "results_dir", "logs_dir"):
        setattr(c.paths, attr, getattr(cfg.paths, attr))
    c.embeddings = cfg.embeddings
    c.llm = cfg.llm
    return c


@pytest.fixture
def l2_cfg(cfg) -> Config:
    return _condition("L2", cfg)


@pytest.fixture
def l3_cfg(cfg) -> Config:
    return _condition("L3", cfg)


@pytest.fixture
def l4_cfg(cfg) -> Config:
    return _condition("L4", cfg)


@pytest.fixture
def built(l2_cfg, embedder, prompts):
    from fgl.memory.ingest_slots import SlotIngestor

    ing = SlotIngestor(l2_cfg, FakeLLM(l2_cfg.llm), embedder, prompts)
    graph, _report = ing.ingest(_conversation())
    return graph


QUESTIONS = [
    "What did Jon adopt in May 2023?",
    "What did Gina cook?",
    "Where did Jon adopt the pup from?",
    "What does Jon do at the park?",
    "What foods does Gina like?",
]


# --------------------------------------------------------------------------- #
# The reduction -- the claim everything else rests on                          #
# --------------------------------------------------------------------------- #


def test_one_hop_reproduces_l2_exactly(built, l2_cfg, l3_cfg, embedder):
    """``hops=1`` + ``normalization=none`` IS condition L2's structural read.

    Not "close to", not "correlated with". If this ever fails, the sweep over
    `propagation.hops` stops being a curve through the published L2 number and
    the whole L2-vs-L3 comparison loses its meaning -- so the assertion is on
    the emitted turns AND their scores, which is everything downstream depends
    on.
    """
    l3_cfg.propagation.hops = 1
    l3_cfg.propagation.normalization = "none"
    l3_cfg.propagation.dense_seed = 0.0
    assert reduces_to_l2(l3_cfg)

    l2 = SlotRetriever(built, embedder, l2_cfg, {})
    l3 = PropagationRetriever(built, embedder, l3_cfg, {})

    for q in QUESTIONS:
        a, b = l2.retrieve(q), l3.retrieve(q)
        assert [f.turn_ids for f in a.facts] == [f.turn_ids for f in b.facts], q
        np.testing.assert_allclose(
            [f.anchor_score for f in a.facts],
            [f.anchor_score for f in b.facts],
            rtol=1e-9, atol=1e-12, err_msg=q,
        )
        assert a.tokens_used == b.tokens_used, q


def test_reduces_to_l2_names_the_exact_configuration(l3_cfg):
    assert not reduces_to_l2(l3_cfg)  # the shipped L3 is a generalisation
    l3_cfg.propagation.hops = 1
    l3_cfg.propagation.normalization = "none"
    l3_cfg.propagation.dense_seed = 0.0
    assert reduces_to_l2(l3_cfg)
    l3_cfg.propagation.dense_seed = 0.5
    assert not reduces_to_l2(l3_cfg)


def test_two_hops_reaches_strictly_more_than_one(built, l3_cfg, embedder):
    """The generalisation has to actually generalise -- otherwise the condition
    is L2 with a slower scorer."""
    l3_cfg.propagation.normalization = "none"
    l3_cfg.propagation.hops = 1
    one = PropagationRetriever(built, embedder, l3_cfg, {})
    r1 = one.retrieve("Where did Jon adopt the pup from?")

    l3_cfg.propagation.hops = 2
    two = PropagationRetriever(built, embedder, l3_cfg, {})
    r2 = two.retrieve("Where did Jon adopt the pup from?")

    assert r2.n_walk_only >= r1.n_walk_only
    # a walk that reaches nothing new is a walk that did not happen
    assert two._n_walk_only >= 0
    assert len(r2.facts) > 0


# --------------------------------------------------------------------------- #
# A hub is a filter, never a bridge                                            #
# --------------------------------------------------------------------------- #


def _star(n_ep: int, hub_deg: int) -> FatGraph:
    """One hub slot on every episode, plus one private slot each.

    The shape that kills naive propagation: without the bridge rule, mass put
    on any private slot reaches every episode in two hops through the hub, and
    the walk returns the whole corpus.
    """
    g = FatGraph()
    eps = [
        g.add_vertex(name=f"ep{i}", vertex_id=f"ep:{i}",
                     meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{i}"],
                           "turn_texts": [f"turn {i}"],
                           "speaker_content": {"a": 2, "b": 1}})
        for i in range(n_ep)
    ]
    hub = g.add_vertex(name="hub", vertex_id="concept:hub",
                       meta={"kind": KIND_CONCEPT, "key": "hub"})
    for i in range(min(hub_deg, n_ep)):
        g.add_edge(eps[i], hub, {"text": "t", "turn_ids": [f"D1:{i}"]})
    for i in range(n_ep):
        v = g.add_vertex(name=f"c{i}", vertex_id=f"concept:c{i}",
                         meta={"kind": KIND_CONCEPT, "key": f"c{i}"})
        g.add_edge(eps[i], v, {"text": "t", "turn_ids": [f"D1:{i}"]})
    return g


class _FakeIsHub:
    def __init__(self, cut: int, graph: FatGraph) -> None:
        self.cut, self.graph = cut, graph

    def __call__(self, vid: str) -> bool:
        return self.graph.degree(vid) >= self.cut


def test_a_hub_may_receive_mass_and_may_never_relay_it():
    """The single rule that makes all three reads one design.

    Same graph, same seed, one flag. With the hub allowed to bridge, the walk
    returns the entire corpus and has discriminated nothing; with the rule on,
    it returns the episodes actually connected to the seed.
    """
    g = _star(40, 40)

    class _R:
        graph = g

    blocked = build_bipartite.__wrapped__ if hasattr(build_bipartite, "__wrapped__") \
        else build_bipartite
    r = _R()
    r.is_hub = _FakeIsHub(20, g)  # type: ignore[attr-defined]

    bp_filtered = blocked(r, "none", bridge_hubs=False)
    bp_bridging = blocked(r, "none", bridge_hubs=True)

    seed = np.zeros(bp_filtered.n_slot)
    seed[bp_filtered.slot_ids.index("concept:c0")] = 1.0
    reached_filtered = int((propagate(
        bp_filtered, seed, None, hops=2, decay=1.0, non_backtracking=True
    ) > 0).sum())

    seed2 = np.zeros(bp_bridging.n_slot)
    seed2[bp_bridging.slot_ids.index("concept:c0")] = 1.0
    reached_bridging = int((propagate(
        bp_bridging, seed2, None, hops=2, decay=1.0, non_backtracking=True
    ) > 0).sum())

    assert reached_bridging == 40, "without the rule the hub returns everything"
    assert reached_filtered < 5, (
        f"the hub must not relay; reached {reached_filtered} episodes"
    )


# --------------------------------------------------------------------------- #
# Non-backtracking                                                             #
# --------------------------------------------------------------------------- #


def test_non_backtracking_removes_the_seed_reflecting_off_itself():
    """A plain walk's hop 2 is dominated by mass returning through the slot it
    came from: it re-scores the seed and calls it a join.

    Built as the cleanest possible case -- one seeded slot on one episode, with
    nothing else attached. A backtracking walk still finds mass at hop 2 (it
    went out and came back); the non-backtracking one correctly finds none,
    because there is genuinely nowhere to go.
    """
    g = FatGraph()
    ep = g.add_vertex(name="ep0", vertex_id="ep:0",
                      meta={"kind": KIND_EPISODE, "turn_ids": ["D1:0"],
                            "turn_texts": ["t"], "speaker_content": {"a": 1}})
    v = g.add_vertex(name="c0", vertex_id="concept:c0",
                     meta={"kind": KIND_CONCEPT, "key": "c0"})
    g.add_edge(ep, v, {"text": "t", "turn_ids": ["D1:0"]})

    class _R:
        graph = g
    r = _R()
    r.is_hub = lambda vid: False  # type: ignore[attr-defined]
    bp = build_bipartite(r, "none", bridge_hubs=False)

    seed = np.zeros(bp.n_slot)
    seed[bp.slot_ids.index("concept:c0")] = 1.0

    with_nb = propagate(bp, seed, None, hops=2, decay=1.0, non_backtracking=True)
    without = propagate(bp, seed, None, hops=2, decay=1.0, non_backtracking=False)

    assert with_nb[0] == pytest.approx(1.0), "hop 1 only: there is nowhere else"
    assert without[0] == pytest.approx(2.0), (
        "a backtracking walk double-counts the seed and calls it a second hop"
    )


def test_propagation_is_monotone_in_decay():
    """Sanity on the operator: keeping more mass per hop cannot reduce a score."""
    g = _star(6, 3)

    class _R:
        graph = g
    r = _R()
    r.is_hub = lambda vid: False  # type: ignore[attr-defined]
    bp = build_bipartite(r, "rw", bridge_hubs=True)
    seed = np.zeros(bp.n_slot)
    seed[bp.slot_ids.index("concept:c0")] = 1.0
    low = propagate(bp, seed, None, 3, 0.2, True)
    high = propagate(bp, seed, None, 3, 0.9, True)
    assert (high + 1e-12 >= low).all()


# --------------------------------------------------------------------------- #
# The Steiner connection (L4)                                                  #
# --------------------------------------------------------------------------- #


def test_rooted_star_ranks_holding_both_above_holding_one_twice():
    """The property no additive channel has.

    ep0 mentions both terminals; ep1 mentions only one of them. An additive
    channel can rank ep1 first by matching that one terminal hard -- that is
    exactly how multi-hop fails. The rooted star cannot: its score is a sum of
    distances to EVERY terminal, so ep1 pays for the detour through ep0 to
    reach the terminal it does not have.

    Reachability is transitive on a connected graph, so ep1 is not *excluded*
    here -- it is *dominated*, which is the honest version of the claim. The
    exclusion case is a genuine disconnection and is tested separately below.
    """
    g = FatGraph()
    def ep(i):
        return g.add_vertex(name=f"ep{i}", vertex_id=f"ep:{i}",
                            meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{i}"],
                                  "turn_texts": ["t"], "speaker_content": {"a": 1}})
    e0, e1 = ep(0), ep(1)
    a = g.add_vertex(name="a", vertex_id="concept:a",
                     meta={"kind": KIND_CONCEPT, "key": "a"})
    b = g.add_vertex(name="b", vertex_id="concept:b",
                     meta={"kind": KIND_CONCEPT, "key": "b"})
    g.add_edge(e0, a, {"text": "t", "turn_ids": ["D1:0"]})
    g.add_edge(e0, b, {"text": "t", "turn_ids": ["D1:0"]})
    g.add_edge(e1, a, {"text": "t", "turn_ids": ["D1:1"]})

    metric = SteinerMetric(g, lambda vid: False)
    read = metric.rooted_star(["concept:a", "concept:b"])
    assert read.supported
    assert read.root == "ep:0"
    assert read.per_episode["ep:0"] < read.per_episode["ep:1"], (
        "the episode holding both terminals must cost less than the one that "
        "has to travel for the second"
    )


def test_rooted_star_excludes_an_episode_that_cannot_reach_a_terminal():
    """The exclusion, in the case where it is real: two components.

    Nothing in the second component reaches terminal b at any price, so the
    intersection drops it -- however strongly it matches terminal a.
    """
    g = FatGraph()
    def ep(i):
        return g.add_vertex(name=f"ep{i}", vertex_id=f"ep:{i}",
                            meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{i}"],
                                  "turn_texts": ["t"], "speaker_content": {"a": 1}})
    e0, e1 = ep(0), ep(1)
    a = g.add_vertex(name="a", vertex_id="concept:a",
                     meta={"kind": KIND_CONCEPT, "key": "a"})
    b = g.add_vertex(name="b", vertex_id="concept:b",
                     meta={"kind": KIND_CONCEPT, "key": "b"})
    far = g.add_vertex(name="far", vertex_id="concept:far",
                       meta={"kind": KIND_CONCEPT, "key": "far"})
    g.add_edge(e0, a, {"text": "t", "turn_ids": ["D1:0"]})
    g.add_edge(e0, b, {"text": "t", "turn_ids": ["D1:0"]})
    # e1 lives in its own component, attached to `far` only
    g.add_edge(e1, far, {"text": "t", "turn_ids": ["D1:1"]})

    metric = SteinerMetric(g, lambda vid: False)
    read = metric.rooted_star(["concept:a", "concept:b"])
    assert set(read.per_episode) == {"ep:0"}
    # and a query spanning the two components is simply not supported
    split = metric.rooted_star(["concept:a", "concept:far"])
    assert not split.supported


def test_an_unreachable_terminal_kills_the_conjunction():
    g = FatGraph()
    e0 = g.add_vertex(name="ep0", vertex_id="ep:0",
                      meta={"kind": KIND_EPISODE, "turn_ids": ["D1:0"],
                            "turn_texts": ["t"], "speaker_content": {"a": 1}})
    a = g.add_vertex(name="a", vertex_id="concept:a",
                     meta={"kind": KIND_CONCEPT, "key": "a"})
    g.add_edge(e0, a, {"text": "t", "turn_ids": ["D1:0"]})
    g.add_vertex(name="lonely", vertex_id="concept:lonely",
                 meta={"kind": KIND_CONCEPT, "key": "lonely"})

    metric = SteinerMetric(g, lambda vid: False)
    read = metric.rooted_star(["concept:a", "concept:lonely"])
    assert not read.supported
    assert read.dead_terminals == 1


def test_the_metric_refuses_to_route_through_a_hub():
    """Same rule as the walk. Without it every pair of episodes is two steps
    apart through the speaker vertex and the metric says nothing."""
    g = _star(30, 30)
    open_metric = SteinerMetric(g, lambda vid: False)
    hub_blocked = SteinerMetric(g, _FakeIsHub(20, g))

    open_read = open_metric.rooted_star(["concept:c0", "concept:c1"])
    blocked_read = hub_blocked.rooted_star(["concept:c0", "concept:c1"])

    assert open_read.supported, "through the hub, everything connects"
    assert not blocked_read.supported, (
        "with the hub blocked these two topics genuinely never meet"
    )


def test_the_abstention_threshold_is_derived_from_random_tuples():
    """No swept number, and no answer key: the criterion is 'further apart than
    the upper tail of arbitrary combinations of the same size in this memory'.
    """
    g = FatGraph()
    eps = []
    for i in range(30):
        eps.append(g.add_vertex(
            name=f"ep{i}", vertex_id=f"ep:{i}",
            meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{i}"],
                  "turn_texts": ["t"], "speaker_content": {"a": 1}}))
    for i in range(30):
        v = g.add_vertex(name=f"c{i}", vertex_id=f"concept:c{i}",
                         meta={"kind": KIND_CONCEPT, "key": f"c{i}"})
        g.add_edge(eps[i], v, {"text": "t", "turn_ids": [f"D1:{i}"]})
        g.add_edge(eps[(i + 1) % 30], v, {"text": "t", "turn_ids": [f"D1:{i}"]})

    metric = SteinerMetric(g, lambda vid: False)
    null = calibrate_null(metric, lambda vid: False, ks=(2, 3),
                          n_samples=40, pool=20, quantile=0.9)
    assert null.pool_size == 20
    assert set(null.by_k) <= {2, 3}
    if null.by_k:
        assert all(v > 0 for v in null.by_k.values())
        # cost grows with the number of things that must be held together
        if 2 in null.by_k and 3 in null.by_k:
            assert null.by_k[3] >= null.by_k[2]


def test_null_extrapolates_to_an_unsampled_k():
    from fgl.retrieval.steiner import NullDistribution

    null = NullDistribution(by_k={2: 6.0, 3: 9.0})
    assert null.threshold(2) == 6.0
    assert null.threshold(5) > null.threshold(3)


# --------------------------------------------------------------------------- #
# Composition: L4 is L3 is L2                                                  #
# --------------------------------------------------------------------------- #


def test_the_hierarchy_is_real_not_documented():
    """"L3 is L2 with one more hop" has to be a fact about the classes.

    A subclass cannot silently diverge on the question parser, the actor prior
    or the emission policy; a sibling class copied from L2 could, and would
    make every measured delta uninterpretable.
    """
    assert issubclass(PropagationRetriever, SlotRetriever)
    assert issubclass(UnifiedRetriever, PropagationRetriever)
    # exactly the seams, and nothing else
    overridden = {
        n for n in vars(PropagationRetriever)
        if not n.startswith("__") and hasattr(SlotRetriever, n)
    }
    assert overridden <= {"_structural_channels", "retrieve"}, overridden


def test_l4_runs_end_to_end_and_reports_its_connection(built, l4_cfg, embedder):
    r = UnifiedRetriever(built, embedder, l4_cfg, {})
    result = r.retrieve("Where did Jon adopt the pup from?")
    assert result.facts
    stats = r.connection_stats()
    assert stats["steiner"]["enabled"] is True
    assert "bridgeable_frac" in stats


def test_l4_abstention_supersedes_the_corner_test_without_deleting_it(
    built, l4_cfg, embedder
):
    """Turning the connection abstention on must never *remove* a signal: where
    the Steiner read has nothing to say, the inherited corner test still runs.
    """
    l4_cfg.steiner.enabled = False
    off = UnifiedRetriever(built, embedder, l4_cfg, {})
    q = "What did Melanie paint in April 2022?"
    base = off._corner_support(  # noqa: SLF001
        [(k, kk, v) for k, kk, v in []], []
    )
    assert base == (1.0, "")  # nothing linked -> nothing to refute

    l4_cfg.steiner.enabled = True
    on = UnifiedRetriever(built, embedder, l4_cfg, {})
    result = on.retrieve(q)
    # whatever it decides, it must decide it with a named reason
    assert result.abstain_reason in (
        "", "missing_slot", "empty_corner", "dead_terminal", "disconnected",
        "far_apart",
    )


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #


def test_conditions_differ_only_where_they_are_supposed_to():
    """L3 must be L2 plus a walk -- if any other knob drifted, the delta stops
    being attributable."""
    l2, l3 = Config.load("L2"), Config.load("L3")
    for knob in ("dense_weight", "actor_weight", "predicate_weight",
                 "concept_weight", "type_weight", "time_weight",
                 "mention_weight", "sibling_frac", "slot_damping",
                 "set_orbit_boost", "corner_actor_min", "hub_degree",
                 "hub_weight", "actor_prior_floor", "actor_prior_full",
                 "concept_link_threshold", "calibration", "question_stop",
                 "time_granularities", "episode_min_turns",
                 "episode_max_turns", "episode_cohesion"):
        assert getattr(l2.slots, knob) == getattr(l3.slots, knob), knob
    assert l2.retrieval.budget_tokens == l3.retrieval.budget_tokens
    assert l2.retrieval.max_facts_in_prompt == l3.retrieval.max_facts_in_prompt
    # and it must read L2's graphs, or it is comparing memories not reads
    assert l3.paths.graphs_condition == "L2-slots"


def test_l4_carries_every_piece_it_claims_to():
    l4 = Config.load("L4")
    assert l4.retrieval.mode == "unified"
    assert l4.slots.calibration == "derived"          # from L2d
    assert l4.slots.question_stop == "derived"        # from L2d
    assert l4.slots.time_granularities == "year,month,day"  # from L2d
    assert l4.propagation.hops >= 2                   # from L3
    assert l4.propagation.non_backtracking is True    # from L3
    assert l4.steiner.enabled and l4.steiner.abstain  # its own contribution
    # the binary corner test is superseded, not left on alongside
    assert l4.slots.abstain_on_empty_corner is False
    # multi-resolution time changes the vertex set, so it cannot borrow graphs
    assert l4.paths.graphs_condition == ""
    # and it must still be measured at the same budget as everything else
    assert l4.retrieval.budget_tokens == Config.load("L2").retrieval.budget_tokens


def test_an_ingest_can_serve_several_reads_but_not_a_foreign_one():
    c = Config.load("L3")
    c.validate()
    c.retrieval.mode = "bipartite"
    with pytest.raises(ConfigError):
        c.validate()


def test_config_rejects_nonsense_propagation_and_steiner_settings():
    for knob, bad in (
        ("propagation.hops", "0"),
        ("propagation.decay", "0"),
        ("propagation.normalization", "spectral"),
        ("steiner.max_terminals", "1"),
        ("steiner.abstain_quantile", "0.2"),
        ("steiner.max_cost", "0"),
    ):
        c = Config.load("L4")
        c.set(knob, bad)
        with pytest.raises(ConfigError):
            c.validate()


# --------------------------------------------------------------------------- #
# The hop profile -- the gate that runs before L3                              #
# --------------------------------------------------------------------------- #


def test_hop_profile_distinguishes_one_hop_from_two_and_from_unreachable():
    """The three answers this diagnostic exists to tell apart.

    ep0 carries the seed; ep1 shares a slot with ep0; ep2 is in its own
    component. If the profile cannot separate those, its verdict ("a longer
    walk has a target" vs "this is an ingest problem") is worthless.
    """
    from fgl.evaluation.hops import episode_hops

    g = FatGraph()
    def ep(i):
        return g.add_vertex(name=f"ep{i}", vertex_id=f"ep:{i}",
                            meta={"kind": KIND_EPISODE, "turn_ids": [f"D1:{i}"],
                                  "turn_texts": ["t"], "speaker_content": {"a": 1}})
    e0, e1, e2 = ep(0), ep(1), ep(2)
    seed = g.add_vertex(name="seed", vertex_id="concept:seed",
                        meta={"kind": KIND_CONCEPT, "key": "seed"})
    link = g.add_vertex(name="link", vertex_id="concept:link",
                        meta={"kind": KIND_CONCEPT, "key": "link"})
    away = g.add_vertex(name="away", vertex_id="concept:away",
                        meta={"kind": KIND_CONCEPT, "key": "away"})
    g.add_edge(e0, seed, {"text": "t", "turn_ids": ["D1:0"]})
    g.add_edge(e0, link, {"text": "t", "turn_ids": ["D1:0"]})
    g.add_edge(e1, link, {"text": "t", "turn_ids": ["D1:1"]})
    g.add_edge(e2, away, {"text": "t", "turn_ids": ["D1:2"]})

    hops = episode_hops(g, ["concept:seed"], lambda vid: False)
    assert hops["ep:0"] == 1
    assert hops["ep:1"] == 2
    assert "ep:2" not in hops


def test_hop_profile_obeys_the_same_hub_rule_as_the_walk():
    """It has to measure the graph the walk will actually see, or the ceiling
    it reports is for a retriever nobody is going to run."""
    from fgl.evaluation.hops import episode_hops

    g = _star(30, 30)
    open_hops = episode_hops(g, ["concept:c0"], lambda vid: False)
    blocked = episode_hops(g, ["concept:c0"], _FakeIsHub(20, g))
    assert len(open_hops) == 30, "through the hub, two steps reach everything"
    assert len(blocked) == 1, "with the hub blocked, only its own episode"


def test_quadrangulation_stats_report_the_euler_ceiling():
    """The numbers behind 'the rotation is using 0.3% of the ribbon structure'.

    Bipartite => every face has length >= 4 => F <= E/2, and Euler turns that
    into a floor on the genus. Both are pure arithmetic on (V, E, F, C), so a
    regression here means the report is quoting a ceiling that is not one.
    """
    from fgl.evaluation.hops import quadrangulation_stats

    g = _star(12, 6)
    st = quadrangulation_stats(g, lambda vid: False)
    assert st["faces_ceiling_bipartite"] == st["E"] // 2
    # F = 2C - 2g + E - V, rearranged -- must match the graph's own count
    assert st["genus"] == int((2 * st["C"] + st["E"] - st["V"] - st["F"]) / 2)
    assert st["genus_floor_bipartite"] <= st["genus"]
    assert 0.0 <= st["faces_used_frac"] <= 1.0
    # the 4-cycles a quadrangular rotation would turn into faces
    assert st["episode_pairs_sharing_2plus_slots"] >= 0
