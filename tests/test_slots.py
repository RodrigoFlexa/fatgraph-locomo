"""Unit tests for condition L2 -- the typed-slot memory.

Each test pins one of the four bridges the model exists to build, using the
worked example the design was derived from, so a regression names the failure
mode instead of a line number:

* the **reply**: a turn that carries the value and none of the topic must land
  in the same episode as the turn that carries the topic;
* the **predicate**: "adopt" is a vertex, not just a word inside a turn;
* the **type**: "chicken" reaches a question asking about "food";
* the **actor**: the speaker is a vertex, and a high-degree one is a partition
  (weighted by contribution) rather than a hub.

Plus the two structural invariants everything else rests on: ``sigma`` at an
episode follows ``SLOT_ORDER`` (which is what makes the corners meaningful),
and the corner test fires on an unsupported combination without firing on a
supported one.
"""

from __future__ import annotations

import pytest

from fgl.config import Config
from fgl.core import FatGraph
from fgl.data.locomo import Conversation, Session, Turn
from fgl.llm import FakeLLM
from fgl.memory.slots import (
    KIND_ACTOR,
    KIND_CONCEPT,
    KIND_EPISODE,
    KIND_PREDICATE,
    KIND_TIME,
    KIND_TYPE,
    SLOT_ORDER,
    EpisodeSegmenter,
    actor_key,
    lift_types,
    match_actor,
    month_bucket,
    question_time_buckets,
    types_available,
)

pytest.importorskip("spacy")


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _conversation() -> Conversation:
    """A miniature LoCoMo conversation carrying every measured failure mode.

    D1:1/D1:2 is the reply case in its purest form: the question turn names the
    topic ("dance piece"), the answer turn names only the value ("Finding
    Freedom") -- the exact shape ("What kind of dance piece did Gina's team
    perform?" / "We just did a contemporary piece called 'Finding Freedom.'")
    that L1 cannot retrieve, because the two turns share no noun.
    """
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
def built(slot_cfg, embedder, prompts):
    from fgl.memory.ingest_slots import SlotIngestor

    ing = SlotIngestor(slot_cfg, FakeLLM(slot_cfg.llm), embedder, prompts)
    graph, report = ing.ingest(_conversation())
    return graph, report, slot_cfg


def _by_kind(graph: FatGraph, kind: str) -> dict[str, str]:
    """``normalised name -> vertex id`` for one slot kind."""
    return {
        vx.name.lower(): vid
        for vid, vx in graph.vertices.items()
        if vx.meta.get("kind") == kind
    }


# --------------------------------------------------------------------------- #
# The four bridges                                                             #
# --------------------------------------------------------------------------- #


def test_reply_shares_an_episode_with_its_topic(built):
    """D1:2 answers D1:1 and shares no noun with it: same episode or nothing.

    This is failure mode (1) and the reason the unit is not the turn. If this
    test fails, the value turn has become unreachable from the only turn that
    names what it is about, exactly as in L1.
    """
    graph, _, _ = built
    episodes = [
        vx.meta["turn_ids"] for vx in graph.vertices.values()
        if vx.meta.get("kind") == KIND_EPISODE
    ]
    holding_both = [t for t in episodes if "D1:1" in t and "D1:2" in t]
    assert holding_both, f"the adjacency pair was split across episodes: {episodes}"


def test_predicate_becomes_a_vertex(built):
    """"What did James ADOPT" -- failure mode (2). L1 indexes only nouns."""
    graph, _, _ = built
    predicates = _by_kind(graph, KIND_PREDICATE)
    assert "adopt" in predicates
    assert "cook" in predicates
    assert graph.degree(predicates["adopt"]) >= 1


def test_concept_lifts_to_its_type(built):
    """"What FOODS does Audrey like" reaching "roasted chicken" -- mode (3)."""
    if not types_available():
        pytest.skip("WordNet corpus not installed; the type channel is additive")
    graph, _, _ = built
    types = _by_kind(graph, KIND_TYPE)
    assert "food" in types, sorted(types)[:20]


def test_actor_is_a_vertex_weighted_by_contribution(built):
    """The speaker is a vertex -- failure mode (4) -- and an episode is not
    equally *about* both people in it, which is what stops a degree-N actor
    from behaving like a hub."""
    graph, _, _ = built
    actors = _by_kind(graph, KIND_ACTOR)
    assert {"jon", "gina"} <= set(actors)
    episodes = [
        vx for vx in graph.vertices.values() if vx.meta.get("kind") == KIND_EPISODE
    ]
    shares = [vx.meta["speaker_content"] for vx in episodes]
    assert any(len(s) > 1 and len(set(s.values())) > 1 for s in shares), (
        "no episode has an uneven contribution split, so the actor prior "
        "would be a constant and could not discriminate"
    )


# --------------------------------------------------------------------------- #
# Structure                                                                    #
# --------------------------------------------------------------------------- #


def test_sigma_at_an_episode_follows_slot_order(built):
    """``sigma`` at an episode *is* which corners exist, so its order is part
    of the model rather than an implementation detail: consecutive half-edges
    must be (actor, predicate), (predicate, concept), (concept, type), ...
    """
    graph, _, _ = built
    rank = {kind: i for i, kind in enumerate(SLOT_ORDER)}
    for vid, vx in graph.vertices.items():
        if vx.meta.get("kind") != KIND_EPISODE:
            continue
        kinds = [
            graph.vertices[graph.H[graph.alpha[h]].vertex_id].meta.get("kind", KIND_CONCEPT)
            for h in graph.sigma[vid]
        ]
        ranks = [rank[k] for k in kinds]
        assert ranks == sorted(ranks), f"{vid}: rotation out of SLOT_ORDER: {kinds}"


def test_ingest_is_deterministic_and_costs_no_llm(slot_cfg, embedder, prompts):
    """Two ingests of the same conversation give the same graph, and neither
    calls the LLM -- the guarantee L1 established and L2 must not quietly drop.
    """
    from fgl.memory.ingest_slots import SlotIngestor

    llm = FakeLLM(slot_cfg.llm)
    a, _ = SlotIngestor(slot_cfg, llm, embedder, prompts).ingest(_conversation())
    b, _ = SlotIngestor(slot_cfg, llm, embedder, prompts).ingest(_conversation())
    assert a.stats()["fingerprint"] == b.stats()["fingerprint"]
    assert llm.usage.to_dict()["calls"] == 0


def test_graph_invariants_hold(built):
    graph, report, _ = built
    graph.check_invariants()
    assert report.graph_stats["n_episodes"] >= 1
    assert set(report.graph_stats["slot_kinds"]) <= {
        KIND_EPISODE, KIND_ACTOR, KIND_PREDICATE, KIND_CONCEPT, KIND_TYPE, KIND_TIME
    }


# --------------------------------------------------------------------------- #
# Retrieval                                                                    #
# --------------------------------------------------------------------------- #


def test_question_parses_into_the_memory_vocabulary(built, embedder):
    from fgl.retrieval.slots import SlotRetriever

    graph, _, cfg = built
    r = SlotRetriever(graph, embedder, cfg, {"S1": "1:56 pm on 8 May, 2023"})
    slots = r.parse_question("What did Jon adopt in May 2023?")
    assert "jon" in slots.actors
    assert "adopt" in slots.predicates
    assert "2023-05" in slots.times


def test_question_boilerplate_is_not_a_concept(built, embedder):
    """"conversation" was linked to a graph vertex in 194 of L1's 1986
    questions and "date" in 118, both straight out of the question template
    and the temporal suffix. They are filtered on the question side.
    """
    from fgl.retrieval.slots import SlotRetriever

    graph, _, cfg = built
    r = SlotRetriever(graph, embedder, cfg, {})
    slots = r.parse_question(
        "What activities does Gina do? Use DATE of CONVERSATION to answer "
        "with an approximate date."
    )
    assert "conversation" not in slots.concepts
    assert "date" not in slots.concepts
    assert "activity" not in slots.concepts


def test_retrieval_respects_the_budget(built, embedder):
    from fgl.retrieval.slots import SlotRetriever

    graph, _, cfg = built
    cfg.retrieval.budget_tokens = 40
    r = SlotRetriever(graph, embedder, cfg, {})
    result = r.retrieve("What did Jon adopt?")
    assert result.tokens_used <= 40 or len(result.facts) == 1
    assert result.facts


def test_corner_test_fires_only_on_an_unsupported_pair(built, embedder):
    """The deterministic abstention: "did GINA adopt anything" has no corner
    (Jon did), while "did JON adopt anything" does. A test that fires on both
    is worthless, and one that fires on neither is the same.
    """
    from fgl.retrieval.slots import SlotRetriever

    graph, _, cfg = built
    r = SlotRetriever(graph, embedder, cfg, {})
    supported = r.retrieve("What did Jon adopt?")
    assert supported.abstain_reason == ""
    unsupported = r.retrieve("What did Gina adopt?")
    assert unsupported.abstain_reason == "empty_corner"


def test_abstention_flag_actually_empties_the_context(built, embedder):
    from fgl.retrieval.slots import SlotRetriever

    graph, _, cfg = built
    cfg.slots.abstain_on_empty_corner = True
    r = SlotRetriever(graph, embedder, cfg, {})
    result = r.retrieve("What did Gina adopt?")
    assert result.facts == []  # Answerer abstains on an empty context


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def test_segmenter_always_glues_the_adjacency_pair():
    seg = EpisodeSegmenter(min_turns=2, max_turns=4, cohesion_min=0.5)
    # no concept overlap anywhere: only the min_turns rule can group these
    groups = seg.segment([frozenset({"a"}), frozenset({"b"}), frozenset({"c"})])
    assert groups[0] == [0, 1]


def test_segmenter_respects_the_ceiling():
    seg = EpisodeSegmenter(min_turns=2, max_turns=2, cohesion_min=0.0)
    groups = seg.segment([frozenset({"a"})] * 5)
    assert all(len(g) <= 2 for g in groups)
    assert sum(len(g) for g in groups) == 5


def test_contentless_turn_joins_rather_than_starting_an_episode():
    seg = EpisodeSegmenter(min_turns=1, max_turns=6, cohesion_min=0.9)
    groups = seg.segment([frozenset({"dance"}), frozenset(), frozenset({"dance"})])
    assert groups == [[0, 1, 2]]


def test_actor_matching_handles_nicknames_and_word_boundaries():
    assert match_actor("What did Mel paint?", ["melanie", "caroline"]) == ["melanie"]
    assert match_actor("They ate the same meal", ["sam", "evan"]) == []
    assert actor_key("Melanie Chen") == "melanie"


def test_type_lift_drops_the_useless_abstractions():
    if not types_available():
        pytest.skip("WordNet corpus not installed")
    types = lift_types("chicken")
    assert "food" in types
    assert "entity" not in types and "physical_entity" not in types


def test_question_time_buckets():
    assert question_time_buckets("What did James adopt in April 2022?") == ["2022-04"]
    assert question_time_buckets("What happened in 2023?") == ["2023"]
    assert question_time_buckets("What is her favourite colour?") == []


def test_month_bucket_of_nothing_is_empty():
    from datetime import datetime

    assert month_bucket(None) == ""
    assert month_bucket(datetime(2023, 5, 8)) == "2023-05"
