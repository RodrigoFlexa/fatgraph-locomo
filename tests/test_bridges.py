"""Tests for condition L6 -- LLM-synthesised bridges between episodes that
share no slot at all.

Design: ``docs/L6_DESIGN_bridge_synthesis.md``; implementation:
:mod:`fgl.memory.bridges`. Three things are pinned here:

* **stage 1 is correct and cheap** -- the corpus-derived threshold is applied
  correctly, and a pair the slot graph already connects within a hop is
  never sent to the LLM (:func:`find_bridge_candidates`);
* **stage 2 materialises what it is told, and only that** -- a "linked: true"
  becomes a real episode incident to the two *pre-existing* vertices it
  names, with a turn id that can never collide with a real dialogue turn id,
  and a "linked: false" (the default fake responder's answer) changes
  nothing (:func:`synthesize_bridges`);
* **the connection is visible in rendered text** -- both a bridge fact and a
  Steiner-joined fact now say what they connect, not the literal channel
  name (the bug this condition's own design doc found in L4/L5).

Stage 1's graph is built by hand rather than through :class:`SlotIngestor`,
so the corpus-derived-quantile fallback and the hop-1 reachability filter can
be pinned to an exact, deterministic outcome, independent of what spaCy's
NER happens to extract from a sentence.
"""

from __future__ import annotations

import json

import pytest

from fgl.config import Config, ConfigError
from fgl.core import FatGraph
from fgl.data.locomo import Conversation, Session, Turn
from fgl.llm import FakeLLM
from fgl.memory.bridges import BridgeReport, find_bridge_candidates, synthesize_bridges
from fgl.memory.slots import KIND_ACTOR, KIND_CONCEPT, KIND_EPISODE, slot_vertex_id
from fgl.retrieval.faces import RetrievalResult, RetrievedFact, render_context
from fgl.retrieval.slots import SOURCE_SLOT_BRIDGE, SOURCE_SLOT_STEINER
from fgl.retrieval.unified import UnifiedRetriever

pytest.importorskip("spacy")


# --------------------------------------------------------------------------- #
# a small synthetic graph, built by hand (no NER)                             #
# --------------------------------------------------------------------------- #


def _episode(g: FatGraph, vid: str, text: str, turn_id: str, vec) -> None:
    g.add_vertex(
        name=turn_id, vertex_id=vid, embedding=vec,
        meta={
            "kind": KIND_EPISODE, "turn_ids": [turn_id], "turn_texts": [text],
            "speakers": [], "speaker_content": {}, "mentioned_actors": [],
        },
    )


def _slot_edge(g: FatGraph, ep_vid: str, kind: str, key: str) -> None:
    slot_vid = slot_vertex_id(kind, key)
    if slot_vid not in g.vertices:
        g.add_vertex(name=key, vertex_id=slot_vid, meta={"kind": kind, "key": key})
    g.add_edge(ep_vid, slot_vid, {"text": key, "turn_ids": [], "session_id": "s1"})


#: Hand-picked wording: HashingEmbedder is a bag-of-hashed-word/char-trigram
#: cosine model, so similarity tracks shared vocabulary. "a" and "c" share
#: "watercolor"/"painting"/"downtown" without sharing a SLOT in the graph
#: below (a is tagged actor=caroline/concept=painting; c is tagged
#: actor=aunt/concept=gallery) -- exactly the "semantically close,
#: topologically far" shape stage 1 exists to find. "b" is the same story as
#: "a" (same slots too, on purpose). "d" shares nothing with any of them.
TEXTS = {
    "a": "Jon: Caroline just started a new watercolor painting class downtown this week.",
    "b": "Gina: Caroline mentioned she loves watercolor painting, she started a class downtown.",
    "c": "Jon: My aunt paints watercolor pieces every week and just started selling them downtown.",
    "d": "Gina: I tried a new pasta recipe last night with garlic and olive oil.",
}


def _graph(embedder) -> FatGraph:
    vecs = dict(zip(TEXTS, embedder.encode(list(TEXTS.values()))))
    g = FatGraph()
    _episode(g, "ep:D1:1", TEXTS["a"], "D1:1", vecs["a"])
    _episode(g, "ep:D1:5", TEXTS["b"], "D1:5", vecs["b"])
    _episode(g, "ep:D1:9", TEXTS["c"], "D1:9", vecs["c"])
    _episode(g, "ep:D1:13", TEXTS["d"], "D1:13", vecs["d"])
    # a and b are the SAME story (Caroline's class): they share both an actor
    # and a concept slot, so the slot graph already connects them -- stage 1
    # must not spend an LLM call rediscovering that.
    _slot_edge(g, "ep:D1:1", KIND_ACTOR, "caroline")
    _slot_edge(g, "ep:D1:1", KIND_CONCEPT, "painting")
    _slot_edge(g, "ep:D1:5", KIND_ACTOR, "caroline")
    _slot_edge(g, "ep:D1:5", KIND_CONCEPT, "painting")
    # c shares no slot with a or b at all.
    _slot_edge(g, "ep:D1:9", KIND_ACTOR, "aunt")
    _slot_edge(g, "ep:D1:9", KIND_CONCEPT, "gallery")
    _slot_edge(g, "ep:D1:13", KIND_ACTOR, "gina")
    _slot_edge(g, "ep:D1:13", KIND_CONCEPT, "pasta")
    return g


@pytest.fixture
def bridge_cfg(cfg) -> Config:
    c = Config.load("L6")
    for attr in ("facts_cache", "graphs_dir", "results_dir", "logs_dir"):
        setattr(c.paths, attr, getattr(cfg.paths, attr))
    c.embeddings = cfg.embeddings
    c.llm = cfg.llm
    # Pins the fallback threshold used below: fewer than 12 episodes means
    # concept_link_threshold_by_quantile returns `floor` verbatim.
    c.bridges.floor = 0.3
    return c


# --------------------------------------------------------------------------- #
# Stage 1: candidates, zero LLM                                               #
# --------------------------------------------------------------------------- #


def test_candidates_pass_the_similarity_floor_and_skip_slot_reachable_pairs(
    bridge_cfg, embedder
):
    g = _graph(embedder)
    candidates, meta = find_bridge_candidates(g, bridge_cfg)

    pairs = {frozenset((c.ep_a, c.ep_b)) for c in candidates}
    assert frozenset(("ep:D1:1", "ep:D1:9")) in pairs        # a-c: related, no shared slot
    assert frozenset(("ep:D1:5", "ep:D1:9")) in pairs        # b-c: related, no shared slot
    assert frozenset(("ep:D1:1", "ep:D1:5")) not in pairs    # a-b: same story, already linked
    assert frozenset(("ep:D1:1", "ep:D1:13")) not in pairs   # d: below the floor entirely

    assert meta["n_pairs_over_threshold"] == 3   # a-b, a-c, b-c
    assert meta["n_skipped_reachable"] == 1      # a-b, dropped
    assert meta["n_candidates"] == len(candidates) == 2
    assert meta["threshold"] == bridge_cfg.bridges.floor
    assert meta["threshold_evidence"]["source"] == "fallback"  # n < MIN_VERTICES_FOR_QUANTILE


def test_candidates_are_capped_at_max_candidates_by_similarity(bridge_cfg, embedder):
    g = _graph(embedder)
    bridge_cfg.bridges.max_candidates = 1
    candidates, _meta = find_bridge_candidates(g, bridge_cfg)
    assert len(candidates) == 1
    # the higher-similarity pair (a-c) wins over the lower one (b-c)
    assert frozenset((candidates[0].ep_a, candidates[0].ep_b)) == frozenset(
        ("ep:D1:1", "ep:D1:9")
    )


def test_fewer_than_two_episodes_finds_nothing(bridge_cfg, embedder):
    g = FatGraph()
    _episode(g, "ep:D1:1", TEXTS["a"], "D1:1", embedder.encode_one(TEXTS["a"]))
    candidates, meta = find_bridge_candidates(g, bridge_cfg)
    assert candidates == []
    assert meta["n_episodes"] == 1


# --------------------------------------------------------------------------- #
# Stage 2: judgment + synthesis                                               #
# --------------------------------------------------------------------------- #


def _linked_responder(
    bridge_text="Caroline's new class and the aunt's gallery both feature watercolor painting.",
    entity_1="Caroline",
    entity_2="gallery",
):
    def responder(prompt, system):
        return json.dumps(
            {"linked": True, "bridge_text": bridge_text,
             "entity_1": entity_1, "entity_2": entity_2}
        )

    return responder


def test_synthesize_bridges_materializes_onto_pre_existing_vertices(
    bridge_cfg, embedder, prompts
):
    g = _graph(embedder)
    before = set(g.vertices)
    llm = FakeLLM(bridge_cfg.llm, responder=_linked_responder())

    report = synthesize_bridges(g, bridge_cfg, llm, embedder, prompts)

    assert report.n_candidates == 2
    assert report.n_llm_calls == 2
    assert report.n_linked == 2
    assert report.n_rejected == 0

    bridge_vid = "ep:bridge:ep:D1:1|ep:D1:9"
    assert bridge_vid in g.vertices
    vx = g.vertices[bridge_vid]
    assert vx.meta["kind"] == KIND_EPISODE
    assert vx.meta["bridge"] is True
    assert vx.meta["bridge_of"] == ["ep:D1:1", "ep:D1:9"]
    assert vx.meta["bridge_entities"] == ["Caroline", "gallery"]
    assert vx.embedding is not None

    # Resolves onto the two PRE-EXISTING vertices, not new ones: the whole
    # point of a bridge is to join what is already there, and creating a
    # fresh "Caroline" vertex instead would leave the two halves of the
    # graph just as disconnected as before.
    neighbours = {g.H[g.alpha[hid]].vertex_id for hid in g.sigma[bridge_vid]}
    assert neighbours == {"actor:caroline", "concept:gallery"}
    new_vertices = set(g.vertices) - before
    assert new_vertices == {
        "ep:bridge:ep:D1:1|ep:D1:9", "ep:bridge:ep:D1:5|ep:D1:9",
    }, "no orphan concept/actor vertex should have been created"

    for hid in g.sigma[bridge_vid]:
        he = g.H[hid]
        assert he.turn_ids == [
            "BRIDGE:ep:D1:1|ep:D1:9", "D1:1", "D1:9",
        ]
        assert he.meta["source"] == "ingest_bridge"
        # Never collides with a real dialogue turn id ("D<session>:<n>") --
        # evidence_recall in fgl.pipeline does exact turn-id-set membership.
        assert not he.turn_ids[0].startswith("D")
    assert g.vertices["actor:caroline"].meta["kind"] == KIND_ACTOR
    edge_to_actor = next(
        g.H[hid] for hid in g.sigma[bridge_vid]
        if g.H[g.alpha[hid]].vertex_id == "actor:caroline"
    )
    assert edge_to_actor.meta["slot_kind"] == KIND_ACTOR  # not hardcoded to concept


@pytest.mark.parametrize(
    "payload",
    [[], {"linked": "false"}, {"linked": True, "bridge_text": [], "entity_1": "Caroline", "entity_2": "gallery"}],
)
def test_invalid_bridge_schema_is_rejected_without_mutating_graph(
    bridge_cfg, embedder, prompts, payload
):
    g = _graph(embedder)
    before = set(g.vertices)

    def responder(prompt, system):
        return json.dumps(payload)

    report = synthesize_bridges(
        g, bridge_cfg, FakeLLM(bridge_cfg.llm, responder=responder), embedder, prompts
    )

    assert report.n_linked == 0
    assert report.n_rejected == report.n_candidates
    assert set(g.vertices) == before


def test_unresolved_bridge_entities_are_rejected_without_orphans(
    bridge_cfg, embedder, prompts
):
    g = _graph(embedder)
    before = set(g.vertices)
    llm = FakeLLM(
        bridge_cfg.llm,
        responder=_linked_responder(entity_1="the thing", entity_2="it"),
    )

    report = synthesize_bridges(g, bridge_cfg, llm, embedder, prompts)

    assert report.n_linked == 0
    assert report.n_rejected == report.n_candidates
    assert set(g.vertices) == before


def test_bridge_with_duplicate_entities_is_rejected(bridge_cfg, embedder, prompts):
    g = _graph(embedder)
    before = set(g.vertices)
    llm = FakeLLM(
        bridge_cfg.llm,
        responder=_linked_responder(entity_1="Caroline", entity_2="Caroline"),
    )

    report = synthesize_bridges(g, bridge_cfg, llm, embedder, prompts)

    assert report.n_linked == 0
    assert report.n_rejected == report.n_candidates
    assert set(g.vertices) == before


def test_default_fake_responder_rejects_every_bridge(bridge_cfg, embedder, prompts):
    """The conservative default (like every other marker) is a 'no'."""
    g = _graph(embedder)
    before = set(g.vertices)
    llm = FakeLLM(bridge_cfg.llm)  # default responder, no override

    report = synthesize_bridges(g, bridge_cfg, llm, embedder, prompts)

    assert report.n_candidates == 2
    assert report.n_llm_calls == 2
    assert report.n_linked == 0
    assert report.n_rejected == 2
    assert set(g.vertices) == before


def test_a_too_short_bridge_text_is_rejected(bridge_cfg, embedder, prompts):
    g = _graph(embedder)
    bridge_cfg.bridges.min_bridge_chars = 1000
    llm = FakeLLM(bridge_cfg.llm, responder=_linked_responder())

    report = synthesize_bridges(g, bridge_cfg, llm, embedder, prompts)

    assert report.n_linked == 0
    assert report.n_rejected == report.n_candidates == 2


def test_disabled_synthesize_bridges_is_a_noop(bridge_cfg, embedder, prompts):
    g = _graph(embedder)
    bridge_cfg.bridges.enabled = False
    llm = FakeLLM(bridge_cfg.llm)

    report = synthesize_bridges(g, bridge_cfg, llm, embedder, prompts)

    assert report == BridgeReport(enabled=False)
    assert llm.prompts == []  # zero LLM calls, the L1-L5 guarantee


# --------------------------------------------------------------------------- #
# Wired into SlotIngestor                                                     #
# --------------------------------------------------------------------------- #


def _tiny_conversation() -> Conversation:
    s1 = Session(num=1, date_time_raw="1:56 pm on 8 May, 2023", timestamp="2023-05-08T13:56:00")
    s1.turns = [
        Turn("D1:1", "Jon", "How did the dance competition go last weekend?", 1),
        Turn("D1:2", "Gina", "We just did a contemporary piece called Finding Freedom.", 1),
        Turn("D1:3", "Jon", "I adopted a pup from a shelter in Stamford.", 1),
        Turn("D1:4", "Gina", "Roasted chicken is one of my favorites, I cooked it again.", 1),
    ]
    return Conversation(
        sample_id="conv-l6", speaker_a="Jon", speaker_b="Gina",
        sessions=[s1], questions=[],
    )


def test_bridges_disabled_makes_zero_llm_calls_and_leaves_no_trace(
    bridge_cfg, embedder, prompts
):
    from fgl.memory.ingest_slots import SlotIngestor

    bridge_cfg.bridges.enabled = False
    llm = FakeLLM(bridge_cfg.llm)
    ing = SlotIngestor(bridge_cfg, llm, embedder, prompts)
    _graph_out, report = ing.ingest(_tiny_conversation())

    assert llm.prompts == []
    assert "bridges" not in report.graph_stats


def test_bridges_enabled_runs_the_pass_and_records_it(bridge_cfg, embedder, prompts):
    from fgl.memory.ingest_slots import SlotIngestor

    bridge_cfg.bridges.enabled = True
    llm = FakeLLM(bridge_cfg.llm)  # default responder: conservative, rejects everything
    ing = SlotIngestor(bridge_cfg, llm, embedder, prompts)
    _graph_out, report = ing.ingest(_tiny_conversation())

    assert "bridges" in report.graph_stats
    stats = report.graph_stats["bridges"]
    assert stats["enabled"] is True
    # the pass must have actually run over this conversation's episodes, even
    # though the conservative default responder links none of them
    assert stats["n_episodes"] >= 1


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


def test_l6_ships_with_bridges_enabled_and_builds_its_own_graphs():
    c = Config.load("L6")
    assert c.bridges.enabled is True
    assert c.ingest.mode == "slots"
    # cannot borrow another condition's graphs -- ingestion itself changed
    assert c.paths.graphs_condition == ""


def test_bridges_enabled_requires_slot_ingest():
    c = Config.load("L6")
    c.set("ingest.mode", "bipartite")
    with pytest.raises(ConfigError):
        c.validate()


@pytest.mark.parametrize(
    "knob,bad",
    [
        ("bridges.top_k", "0"),
        ("bridges.quantile", "1.0"),
        ("bridges.quantile", "0.4"),
        ("bridges.floor", "-0.1"),
        ("bridges.floor", "1.1"),
        ("bridges.skip_within_hops", "0"),
        ("bridges.max_candidates", "0"),
        ("bridges.min_bridge_chars", "0"),
    ],
)
def test_bridges_rejects_bad_values(knob, bad):
    c = Config.load("L6")
    c.set(knob, bad)
    with pytest.raises(ConfigError):
        c.validate()


# --------------------------------------------------------------------------- #
# The connection is visible in rendered text                                  #
# --------------------------------------------------------------------------- #


def test_render_context_labels_a_bridge_as_a_chain():
    fact = RetrievedFact(
        edge_id="e1", text="Caroline's class and the gallery both feature watercolor.",
        timestamp="", date_raw="", session_id="bridge", turn_ids=["BRIDGE:a|b"],
        state="emergente", level=1, anchor_rank=0, anchor_score=1.0,
        face_id="ep:bridge:a|b", position_in_face=0,
        source=SOURCE_SLOT_BRIDGE, via_vertex="", via_entity="Caroline and gallery",
    )
    text = render_context(RetrievalResult(facts=[fact]))
    assert "chain linking Caroline and gallery" in text


def _dance_pup_conversation() -> Conversation:
    """Two sessions that share a topic without sharing a question noun --
    the exact fixture used by test_propagation.py to exercise the Steiner
    join channel, reproduced here so this file does not import from that one.
    """
    s1 = Session(num=1, date_time_raw="1:56 pm on 8 May, 2023", timestamp="2023-05-08T13:56:00")
    s1.turns = [
        Turn("D1:1", "Jon", "How did the dance competition go last weekend?", 1),
        Turn("D1:2", "Gina", "We just did a contemporary piece called Finding Freedom.", 1),
        Turn("D1:3", "Jon", "I adopted a pup from a shelter in Stamford.", 1),
        Turn("D1:4", "Gina", "Roasted chicken is one of my favorites, I cooked it again.", 1),
    ]
    s2 = Session(num=2, date_time_raw="10:10 am on 3 June, 2023", timestamp="2023-06-03T10:10:00")
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


def test_steiner_channel_reports_the_terminals_it_actually_joined(cfg, embedder, prompts):
    """Achado 2 of the L6 design doc: the join channel found the connection
    but the label used to say the literal word "steiner" -- unusable for a
    reader, and indistinguishable from any other fact in the same trail.
    """
    from fgl.memory.ingest_slots import SlotIngestor

    ing = SlotIngestor(_condition("L2", cfg), FakeLLM(cfg.llm), embedder, prompts)
    graph, _report = ing.ingest(_dance_pup_conversation())

    r = UnifiedRetriever(graph, embedder, _condition("L5", cfg), {})
    result = r.retrieve("Where did Jon adopt the pup from?")

    steiner_facts = [f for f in result.facts if f.source == SOURCE_SLOT_STEINER]
    assert steiner_facts, "expected the join channel to fire on this question"
    for f in steiner_facts:
        assert f.via_entity not in ("", "steiner")

    text = render_context(result)
    assert "chain linking" in text
    assert "steiner" not in text.lower()  # the channel name must never leak into the text
