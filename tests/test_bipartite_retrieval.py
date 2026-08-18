"""Tests for :class:`fgl.retrieval.bipartite.BipartiteRetriever` -- L1's
degree-aware retrieval over the real graph :class:`BipartiteIngestor` builds
(not a hand-built mock graph: exercising the actual NER -> incidence ->
retrieval path end to end is the point, the same way
``test_bipartite_retrieval`` for the triples graph exercises
``Ingestor`` -> ``FaceRetriever`` in ``test_integration.py``).
"""

from __future__ import annotations

import pytest
from conftest import PATHS

from fgl.data.locomo import Conversation, Session, Turn, normalize_timestamp
from fgl.llm.prompts import PromptLibrary
from fgl.memory.ingest_bipartite import BipartiteIngestor
from fgl.retrieval import HashingEmbedder
from fgl.retrieval.bipartite import (
    BipartiteRetriever,
    SOURCE_BP_BRIDGE,
    SOURCE_BP_DENSE,
    SOURCE_BP_ENTITY,
)

pytest.importorskip("spacy")


def _conversation(sessions_spec) -> Conversation:
    sessions = []
    for num, date, turns in sessions_spec:
        s = Session(num=num, date_time_raw=date, timestamp=normalize_timestamp(date))
        s.turns = [
            Turn(dia_id=d, speaker=sp, text=tx, session_num=num) for d, sp, tx in turns
        ]
        sessions.append(s)
    return Conversation(
        sample_id="bipartite-retrieval-synthetic",
        speaker_a="Caroline", speaker_b="Melanie",
        sessions=sessions, questions=[],
    )


def _build(cfg, sessions_spec):
    embedder = HashingEmbedder(cfg.embeddings.dim)
    prompts = PromptLibrary(PATHS.prompts)
    graph, _ = BipartiteIngestor(cfg, llm=None, embedder=embedder, prompts=prompts).ingest(
        _conversation(sessions_spec)
    )
    dates = {s.id: s.date_time_raw for s in _conversation(sessions_spec).sessions}
    retriever = BipartiteRetriever(graph, embedder, cfg, dates)
    return graph, retriever


BRIDGE_SESSIONS = [
    (1, "1:56 pm on 8 May, 2023", [
        ("D1:1", "Caroline", "I painted a sunset at the beach."),
    ]),
    (2, "10:10 am on 20 May, 2023", [
        ("D2:1", "Melanie", "I painted a sunset with watercolors."),
    ]),
]


def test_direct_hit_for_a_degree_one_entity(cfg):
    _, retriever = _build(cfg, BRIDGE_SESSIONS)
    result = retriever.retrieve("What did Caroline paint at the beach?")
    sources = {f.source for f in result.facts}
    turn_ids = {t for f in result.facts for t in f.turn_ids}
    assert SOURCE_BP_ENTITY in sources
    assert "D1:1" in turn_ids


def test_two_entity_bridge_is_found_by_neighbourhood_intersection(cfg):
    # Neither "beach" nor "watercolors" is shared between the two turns --
    # only "sunset" is, and it is never named in the question. The bridge
    # mechanism must find it by intersecting each linked entity's OTHER
    # incident entities, not by the question mentioning it directly.
    _, retriever = _build(cfg, BRIDGE_SESSIONS)
    # "watercolor" (singular) to exact-match the vertex name: NER lemmatises
    # plural common nouns at ingest time ("watercolors" -> "watercolor"),
    # and the linker's cheap exact-surface cascade does not itself lemmatise
    # the question, so a plural here would fall through to the embedding
    # cascade and depend on HashingEmbedder similarity clearing threshold --
    # exercising a different code path than the one this test is about.
    question = (
        "What did Caroline paint at the beach and what did Melanie "
        "paint with watercolor?"
    )
    result = retriever.retrieve(question)
    turn_ids = {t for f in result.facts for t in f.turn_ids}
    assert {"D1:1", "D2:1"} <= turn_ids

    bridge_facts = [f for f in result.facts if f.source == SOURCE_BP_BRIDGE]
    assert bridge_facts, "expected at least one fact surfaced via the bridge"
    assert all(f.via_entity == "sunset" for f in bridge_facts)


def test_whole_sigma_orbit_returned_for_a_repeated_entity(cfg):
    sessions = [
        (1, "1:56 pm on 8 May, 2023", [
            ("D1:1", "Caroline", "I started pottery this week.")
        ]),
        (2, "10:10 am on 20 May, 2023", [
            ("D2:1", "Caroline", "I did more pottery yesterday.")
        ]),
        (3, "9:00 am on 1 June, 2023", [
            ("D3:1", "Melanie", "I also enjoy pottery.")
        ]),
    ]
    _, retriever = _build(cfg, sessions)
    result = retriever.retrieve("What has everyone said about pottery?")
    turn_ids = {t for f in result.facts for t in f.turn_ids}
    # every turn incident to "pottery" comes back, not just the top-ranked one
    assert {"D1:1", "D2:1", "D3:1"} <= turn_ids


def test_hub_entity_is_never_enumerated_only_used_as_a_filter_bonus(cfg):
    # Force a low bridge_max_degree so a 3-mention entity qualifies as a hub
    # in this small synthetic graph (mirrors "dog"/"thank" in the real data).
    cfg.bipartite.bridge_max_degree = 3
    sessions = [
        (1, "1:56 pm on 8 May, 2023", [
            ("D1:1", "Caroline", "I went hiking with my dog."),
        ]),
        (2, "10:10 am on 20 May, 2023", [
            ("D2:1", "Melanie", "I played fetch with my dog."),
        ]),
        (3, "9:00 am on 1 June, 2023", [
            ("D3:1", "Caroline", "I took my dog to the vet."),
        ]),
    ]
    graph, retriever = _build(cfg, sessions)
    dog_vid = next(
        vid for vid, vx in graph.vertices.items()
        if vx.meta.get("kind") != "turn" and vx.name == "dog"
    )
    assert graph.degree(dog_vid) == 3  # >= bridge_max_degree=3 -> hub

    result = retriever.retrieve("What did Caroline do with her dog?")
    # a hub is never a direct SOURCE_BP_ENTITY hit on its own
    assert not any(
        f.source == SOURCE_BP_ENTITY and f.via_entity == "dog" for f in result.facts
    )


def test_speaker_named_explicitly_boosts_coverage_of_both(cfg):
    sessions = [
        (1, "1:56 pm on 8 May, 2023", [
            ("D1:1", "Caroline", "I painted a sunset at the beach."),
        ]),
        (2, "10:10 am on 20 May, 2023", [
            ("D2:1", "Melanie", "I baked cookies for the party."),
        ]),
    ]
    _, retriever = _build(cfg, sessions)
    result = retriever.retrieve("What did Caroline and Melanie each do?")
    speakers_present = set()
    for f in result.facts[: max(len(result.facts), 1)]:
        # RetrievedFact does not carry speaker directly; recover it from the
        # rendered text, which is always "<Speaker>: ...".
        speakers_present.add(f.text.split(":", 1)[0])
    assert {"Caroline", "Melanie"} <= speakers_present


def test_dense_backstop_returns_candidates_with_no_lexical_overlap(cfg):
    # No noun in the question overlaps any entity vertex -- only the dense
    # backstop over turn embeddings can find anything at all.
    sessions = [
        (1, "1:56 pm on 8 May, 2023", [
            ("D1:1", "Caroline", "I painted a sunset at the beach."),
        ]),
    ]
    _, retriever = _build(cfg, sessions)
    result = retriever.retrieve("Tell me something completely unrelated.")
    assert result.facts  # dense backstop still returns turn D1:1
    assert all(f.source == SOURCE_BP_DENSE for f in result.facts)


def test_top_edges_and_turn_ids_for_edges_roundtrip(cfg):
    _, retriever = _build(cfg, BRIDGE_SESSIONS)
    edges = retriever.top_edges("What did Caroline paint at the beach?", k=5)
    assert edges
    turn_ids = retriever.turn_ids_for_edges(edges)
    assert turn_ids  # every returned edge maps back to a real turn


def test_empty_graph_returns_empty_result_not_an_exception(cfg):
    _, retriever = _build(cfg, [(1, "1:56 pm on 8 May, 2023", [("D1:1", "Caroline", "Thanks!")])])
    result = retriever.retrieve("Anything at all?")
    assert result.facts == []
