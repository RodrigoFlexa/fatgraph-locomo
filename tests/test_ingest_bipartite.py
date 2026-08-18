"""Integration tests for :class:`fgl.memory.ingest_bipartite.BipartiteIngestor`
-- condition L1's ingest path.

Mirrors the pattern in ``test_integration.py`` (a small, hand-traceable
synthetic conversation with a known expected graph shape), but for the
bipartite turn/entity graph instead of the triples fatgraph: no LLM script is
needed here because there is no LLM in this ingest path at all -- that is
itself one of the things under test.
"""

from __future__ import annotations

import pytest
from conftest import PATHS

from fgl.data.locomo import Conversation, Session, Turn, normalize_timestamp
from fgl.llm.prompts import PromptLibrary
from fgl.memory.ingest_bipartite import BipartiteIngestor
from fgl.retrieval import HashingEmbedder

pytest.importorskip("spacy")

SESSIONS = [
    (1, "1:56 pm on 8 May, 2023", [
        ("D1:1", "Caroline", "Hey Mel! I started pottery this week."),
        ("D1:2", "Melanie", "That's great, Caroline! I love pottery too."),
        ("D1:3", "Caroline", "Thanks!"),  # small talk only -> no turn vertex
    ]),
    (2, "10:10 am on 20 May, 2023", [
        ("D2:1", "Caroline", "I did pottery yesterday."),
        ("D2:2", "Melanie", "Nice! I also enjoy pottery."),
    ]),
]


def build_conversation() -> Conversation:
    sessions = []
    for num, date, turns in SESSIONS:
        s = Session(num=num, date_time_raw=date, timestamp=normalize_timestamp(date))
        s.turns = [
            Turn(dia_id=d, speaker=sp, text=tx, session_num=num) for d, sp, tx in turns
        ]
        sessions.append(s)
    return Conversation(
        sample_id="bipartite-synthetic", speaker_a="Caroline", speaker_b="Melanie",
        sessions=sessions, questions=[],
    )


@pytest.fixture
def ingested(cfg):
    embedder = HashingEmbedder(cfg.embeddings.dim)
    prompts = PromptLibrary(PATHS.prompts)
    ingestor = BipartiteIngestor(cfg, llm=None, embedder=embedder, prompts=prompts)
    graph, report = ingestor.ingest(build_conversation())
    return graph, report


def _entity_vertices(graph):
    return {vid: vx for vid, vx in graph.vertices.items() if vx.meta.get("kind") != "turn"}


def _turn_vertices(graph):
    return {vid: vx for vid, vx in graph.vertices.items() if vx.meta.get("kind") == "turn"}


def test_zero_llm_calls(ingested):
    _, report = ingested
    assert report.llm_usage == {}


def test_recurring_entity_merges_into_one_vertex(ingested):
    graph, _ = ingested
    entities = _entity_vertices(graph)
    pottery = [vid for vid, vx in entities.items() if vx.name == "pottery"]
    assert len(pottery) == 1, f"expected one 'pottery' vertex, found {len(pottery)}"
    assert graph.degree(pottery[0]) == 4  # D1:1, D1:2, D2:1, D2:2


def test_speaker_names_never_become_vertices(ingested):
    graph, _ = ingested
    names = {vx.name for vx in graph.vertices.values()}
    assert "caroline" not in names
    assert "melanie" not in names


def test_nickname_of_a_speaker_is_excluded_too(ingested):
    # "Mel" (D1:1) is a prefix-match nickname of speaker_b "Melanie" -- must
    # not slip through and become its own vertex (the exact regression this
    # design fixed after measuring it on the real dataset).
    graph, _ = ingested
    names = {vx.name for vx in graph.vertices.values()}
    assert "mel" not in names


def test_turn_with_no_surviving_candidates_gets_no_vertex(ingested):
    graph, _ = ingested
    turn_names = {vx.name for vx in _turn_vertices(graph).values()}
    assert "D1:3" not in turn_names  # "Thanks!" -- nothing to link


def test_turn_vertices_created_for_every_other_turn(ingested):
    graph, _ = ingested
    turn_names = {vx.name for vx in _turn_vertices(graph).values()}
    assert turn_names == {"D1:1", "D1:2", "D2:1", "D2:2"}


def test_report_counts_one_incidence_per_surviving_turn(ingested):
    _, report = ingested
    # each of the 4 surviving turns contributes exactly one incidence
    # ("pottery" is the only candidate left after speaker exclusion)
    assert report.n_facts == 4
    assert report.n_edges == 4


def test_sigma_at_entity_vertex_is_chronological(ingested):
    graph, _ = ingested
    entities = _entity_vertices(graph)
    pottery_vid = next(vid for vid, vx in entities.items() if vx.name == "pottery")
    turn_order = [
        graph.H[graph.alpha[h]].vertex_id for h in graph.sigma[pottery_vid]
    ]
    turn_names = [graph.vertices[vid].name for vid in turn_order]
    assert turn_names == ["D1:1", "D1:2", "D2:1", "D2:2"]


def test_graph_invariants_hold(ingested):
    graph, _ = ingested
    graph.check_invariants()  # raises on violation


def test_provenance_text_is_the_turns_own_rendered_text(ingested):
    graph, _ = ingested
    texts = {graph.H[h].text for h in graph.H}
    assert any(t.startswith("Caroline: Hey Mel!") for t in texts)
    assert any(t.startswith("Melanie: Nice!") for t in texts)
