"""Milestone M5 (extraction), offline: a scripted :class:`FakeLLM`
responder stands in for the model, so every OTHER piece of the pipeline
that a real LLM would otherwise mask -- prompt construction, context
assembly, schema/Sigma validation, temporal resolution, staging, and
consolidation -- runs for real and is checked for real.

What a scripted response cannot check is extraction QUALITY (would a real
model actually produce this JSON from this turn) -- that needs
``test_clio_ingest_live.py``, gated on real credentials.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from fgl.clio.catalog import load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.consolidate.journal import FoldJournal
from fgl.clio.consolidate.pipeline import consolidate
from fgl.clio.graph.store import GraphStore
from fgl.clio.index import EntityIndex
from fgl.clio.ingest.context import build_extraction_context
from fgl.clio.ingest.extractor import _coerce_propositions
from fgl.clio.ingest.pipeline import ingest_turn
from fgl.clio.ingest.validate import validate_and_bind
from fgl.clio.log.mentions import MentionStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.unmapped import UnmappedQueue
from fgl.llm.client import FakeLLM
from fgl.llm.prompts import PromptLibrary
from fgl.paths import Paths, project_root
from fgl.retrieval.embeddings import HashingEmbedder

PROMPTS = PromptLibrary(Paths.build(project_root()).prompts)


def _fake_config():
    from fgl.config import LLMConfig

    return LLMConfig(provider="fake", cache_enabled=False)


# --------------------------------------------------------------------- #
# build_extraction_context, in isolation                                  #
# --------------------------------------------------------------------- #
def test_context_finds_known_entities_as_candidates():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    log = LogStore()
    graph = GraphStore()
    melanie = graph.create_entity("Melanie", "Person")
    vertex = graph.create_entity("Vertex", "Organization")
    index = EntityIndex(HashingEmbedder(dim=64))
    index.rebuild(graph)

    ep = log.append(
        session_id="s1",
        speaker="Melanie",
        text="I still work at Vertex remotely",
        ts_ingest=datetime(2023, 6, 20),
    )
    ctx = build_extraction_context(ep, log, index, catalog, max_candidates=5)
    ids = {c.id for c in ctx.candidates}
    assert vertex.id in ids
    assert melanie.id not in ids  # not mentioned by name in this turn
    assert ctx.speaker_entity_id == melanie.id
    assert any(r.name == "works_at" for r in ctx.relations)


def test_context_carries_the_coreference_window_not_further():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    log = LogStore()
    graph = GraphStore()
    index = EntityIndex(HashingEmbedder(dim=64))
    for i in range(5):
        log.append(
            session_id="s1",
            speaker="Melanie",
            text=f"turn {i}",
            ts_ingest=datetime(2023, 1, 1),
        )
    ep = log.append(
        session_id="s1", speaker="Melanie", text="turn 5", ts_ingest=datetime(2023, 1, 1)
    )
    index.rebuild(graph)
    ctx = build_extraction_context(ep, log, index, catalog, coref_window=2)
    assert [e.text for e in ctx.previous_turns] == ["turn 3", "turn 4"]


# --------------------------------------------------------------------- #
# _coerce_propositions: every shape observed on a real deployment          #
# (gpt-4.1-mini, Azure JSON mode) once the top-level-array bug in the      #
# prompt was found and fixed -- Azure/OpenAI's response_format=json_object #
# FORBIDS a top-level array outright, which is why the prompt now asks    #
# for {"propositions": [...]}, and why this still has to tolerate a model #
# that skips the wrapper on a good day.                                   #
# --------------------------------------------------------------------- #
_ONE_FACT = {"operation": "assert", "relation": "works_at", "span": "..."}


def test_coerce_accepts_the_documented_wrapped_shape():
    assert _coerce_propositions({"propositions": [_ONE_FACT]}) == [_ONE_FACT]


def test_coerce_accepts_wrapper_holding_a_bare_dict_not_a_list():
    assert _coerce_propositions({"propositions": _ONE_FACT}) == [_ONE_FACT]


def test_coerce_accepts_the_wrapper_skipped_entirely():
    """Observed for real: the model emits the one fact directly as the
    top-level object, with no "propositions" key at all."""
    assert _coerce_propositions(_ONE_FACT) == [_ONE_FACT]


def test_coerce_accepts_a_bare_list_too():
    """Not what the prompt asks for, but not wrong either -- FakeLLM and
    any future non-JSON-mode backend can return this directly."""
    assert _coerce_propositions([_ONE_FACT]) == [_ONE_FACT]


def test_coerce_empty_object_means_nothing_extracted():
    """Observed for real: json_object mode's closest legal stand-in for
    "nothing to report" when the model wants to return []."""
    assert _coerce_propositions({}) == []
    assert _coerce_propositions({"propositions": []}) == []


def test_coerce_an_unrelated_object_produces_no_crash():
    """Observed for real: '{"error": "Output must be a JSON array."}' --
    the model correctly diagnosing the OLD prompt's bug and still being
    unable to comply under json_object mode. Must not crash; validate_and_bind
    downstream will reject it for missing required fields regardless."""
    assert _coerce_propositions({"error": "Output must be a JSON array."}) == [
        {"error": "Output must be a JSON array."}
    ]


def test_coerce_tolerates_garbage_types():
    assert _coerce_propositions(None) == []
    assert _coerce_propositions("not json-shaped") == []
    assert _coerce_propositions(42) == []


# --------------------------------------------------------------------- #
# validate_and_bind                                                        #
# --------------------------------------------------------------------- #
def test_validate_rejects_a_relation_outside_sigma():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    graph = GraphStore()
    ep_text = "I adopted a cat"
    from fgl.clio.types import Episode

    ep = Episode(
        id="e1",
        session_id="s1",
        speaker="Melanie",
        text=ep_text,
        ts_ingest=datetime(2023, 1, 1),
        seq=0,
    )
    raw = [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "adopted",
            "object_id": "new:cat",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "I adopted a cat",
        }
    ]
    valid, unmapped = validate_and_bind(raw, ep, graph, catalog)
    assert valid == []
    assert unmapped == []  # "adopted" isn't UNMAPPED either -- just rejected


def test_validate_routes_unmapped_relation_without_touching_valid():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    graph = GraphStore()
    from fgl.clio.types import Episode

    ep = Episode(
        id="e1",
        session_id="s1",
        speaker="Melanie",
        text="I adopted a cat",
        ts_ingest=datetime(2023, 1, 1),
        seq=0,
    )
    raw = [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "UNMAPPED",
            "suggested_relation": "adopted_pet",
            "object_id": "new:cat",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "I adopted a cat",
        }
    ]
    valid, unmapped = validate_and_bind(raw, ep, graph, catalog)
    assert valid == []
    assert len(unmapped) == 1
    assert unmapped[0]["suggested_relation"] == "adopted_pet"


def test_validate_downgrades_evidence_when_span_is_not_verbatim():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    graph = GraphStore()
    from fgl.clio.types import Episode

    ep = Episode(
        id="e1",
        session_id="s1",
        speaker="Melanie",
        text="I work at Vertex now",
        ts_ingest=datetime(2023, 1, 1),
        seq=0,
    )
    raw = [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "works_at",
            "object_id": "new:Vertex",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "this span was never said",
        }
    ]
    valid, _ = validate_and_bind(raw, ep, graph, catalog)
    assert valid[0]["evidence_kind"] == "contextual"


def test_validate_ignores_only_decorative_outer_quotes_on_a_literal_span():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    graph = GraphStore()
    from fgl.clio.types import Episode

    ep = Episode(
        id="e1",
        session_id="s1",
        speaker="Melanie",
        text="Yeah, I play clarinet!",
        ts_ingest=datetime(2023, 1, 1),
        seq=0,
    )
    raw = [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "practices",
            "object_id": "new:clarinet",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": '"I play clarinet!"',
        }
    ]

    result = validate_and_bind(raw, ep, graph, catalog)

    assert result.span_downgrades == 0
    assert result.valid[0]["evidence_kind"] == "literal"
    assert result.valid[0]["span"] == "I play clarinet!"


def test_validate_rebinds_singular_first_person_from_a_collective_to_speaker():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    graph = GraphStore()
    speaker = graph.create_entity("Caroline", "Person")
    collective = graph.create_entity("Caroline and her mentee", "Person")
    from fgl.clio.types import Episode

    ep = Episode(
        id="e1",
        session_id="s1",
        speaker="Caroline",
        text="I went hiking last week",
        ts_ingest=datetime(2023, 1, 1),
        seq=0,
    )
    raw = [
        {
            "operation": "assert",
            "subject_id": collective.id,
            "relation": "attended",
            "object_id": "new:hike",
            "polarity": True,
            "time_expression": "last week",
            "evidence_kind": "literal",
            "span": "I went hiking last week",
        }
    ]

    result = validate_and_bind(raw, ep, graph, catalog)

    assert result.rebindings == 1
    assert result.valid[0]["subject_id"] == speaker.id


def test_validate_rejects_type_mismatch_against_a_known_entity():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    graph = GraphStore()
    vertex = graph.create_entity("Vertex", "Organization")
    from fgl.clio.types import Episode

    ep = Episode(
        id="e1",
        session_id="s1",
        speaker="Melanie",
        text="I live in Vertex",
        ts_ingest=datetime(2023, 1, 1),
        seq=0,
    )
    raw = [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "lives_in",
            "object_id": vertex.id,
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "I live in Vertex",
        }
    ]
    valid, _ = validate_and_bind(raw, ep, graph, catalog)
    assert valid == []  # Vertex is an Organization, lives_in wants a Place


# --------------------------------------------------------------------- #
# ingest_turn end to end, FakeLLM scripted to a small real conversation   #
# --------------------------------------------------------------------- #
_SCRIPT: dict[str, list[dict]] = {
    "I started at Vertex this week, I'm living in Recife": [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "works_at",
            "object_id": "new:Vertex",
            "polarity": True,
            "time_expression": "this week",
            "evidence_kind": "literal",
            "span": "I started at Vertex this week",
        },
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "lives_in",
            "object_id": "new:Recife",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "I'm living in Recife",
        },
    ],
    "My manager here is Bia, she also likes climbing": [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "managed_by",
            "object_id": "new:Bia",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "My manager here is Bia",
        },
        {
            "operation": "assert",
            "subject_id": "new:Bia",
            "relation": "practices",
            "object_id": "new:climbing",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "she also likes climbing",
        },
    ],
    "I moved to Salvador last month, I'm still at Vertex remotely": [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "lives_in",
            "object_id": "new:Salvador",
            "polarity": True,
            "time_expression": "last month",
            "evidence_kind": "literal",
            "span": "I moved to Salvador last month",
        },
        {
            "operation": "reassert",
            "subject_id": "new:Melanie",
            "relation": "works_at",
            "object_id": "new:Vertex",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "I'm still at Vertex remotely",
        },
    ],
}


def _scripted_responder(prompt: str, system):
    # Matched against the "THIS TURN:" section specifically, not anywhere
    # in the prompt -- turn N's own text is also quoted verbatim in turn
    # N+1's "PREVIOUS TURNS" context block, and a bare substring check
    # would replay turn N's facts on turn N+1 as a false match.
    for turn_text, facts in _SCRIPT.items():
        if f'THIS TURN:\n"{turn_text}"' in prompt:
            return json.dumps({"propositions": facts})
    return "[]"


@pytest.fixture
def memory():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    return {
        "catalog": catalog,
        "log": LogStore(),
        "graph": GraphStore(),
        "staging": StagingStore(),
        "mentions": MentionStore(),
        "journal": FoldJournal(),
        "index": EntityIndex(HashingEmbedder(dim=64)),
        "unmapped": UnmappedQueue(),
        "llm": FakeLLM(_fake_config(), responder=_scripted_responder),
    }


def _ingest(memory, text, ts):
    return ingest_turn(
        text=text,
        speaker="Melanie",
        session_id="s1",
        ts=ts,
        log=memory["log"],
        graph=memory["graph"],
        staging=memory["staging"],
        mentions=memory["mentions"],
        entity_index=memory["index"],
        catalog=memory["catalog"],
        llm=memory["llm"],
        prompts=PROMPTS,
        unmapped_queue=memory["unmapped"],
    )


def test_full_pipeline_text_to_consolidated_graph(memory):
    config = ClioConfig.default()

    _ingest(
        memory, "I started at Vertex this week, I'm living in Recife", datetime(2023, 1, 14)
    )
    consolidate(
        memory["catalog"],
        memory["graph"],
        memory["staging"],
        config,
        log=memory["log"],
        journal=memory["journal"],
    )

    _ingest(memory, "My manager here is Bia, she also likes climbing", datetime(2023, 3, 2))
    consolidate(
        memory["catalog"],
        memory["graph"],
        memory["staging"],
        config,
        log=memory["log"],
        journal=memory["journal"],
    )

    _ingest(
        memory,
        "I moved to Salvador last month, I'm still at Vertex remotely",
        datetime(2023, 6, 20),
    )
    consolidate(
        memory["catalog"],
        memory["graph"],
        memory["staging"],
        config,
        log=memory["log"],
        journal=memory["journal"],
    )

    graph = memory["graph"]
    names = {e.canonical_name for e in graph.all_entities()}
    assert {"Melanie", "Vertex", "Recife", "Bia", "climbing", "Salvador"} <= names

    melanie = next(e for e in graph.all_entities() if e.canonical_name == "Melanie")
    vertex_edge = next(
        e for e in graph.all_edges() if e.src_id == melanie.id and e.label == "works_at"
    )
    # "this week" resolved for real, through the real temporal resolver --
    # not hand-fed, this time produced from raw text by ingest_turn.
    assert vertex_edge.t_valid.start == datetime(2023, 1, 8)
    assert vertex_edge.reinforcement == 2  # the E3-equivalent reassert landed

    recife_edge = next(
        e
        for e in graph.all_edges()
        if e.src_id == melanie.id and e.label == "lives_in" and e.t_valid.end is not None
    )
    assert recife_edge.t_valid.end == datetime(
        2023, 5, 1
    )  # cardinality, from raw text end to end


def test_count_by_entity_name_finds_mentions_a_real_ingest_recorded(memory):
    """Regression: mentions recorded by a real `ingest_turn` never carry
    `entity_id` (it isn't known until consolidation resolves "new:X"), so
    `count(entity=...)` has to match by canonical name, not by that field
    -- an entity_id-keyed lookup would silently return 0 here."""
    config = ClioConfig.default()
    _ingest(memory, "My manager here is Bia, she also likes climbing", datetime(2023, 3, 2))
    consolidate(
        memory["catalog"],
        memory["graph"],
        memory["staging"],
        config,
        log=memory["log"],
        journal=memory["journal"],
    )

    from fgl.clio.access.movements import count

    assert all(
        m.entity_id is None for m in memory["mentions"].all()
    )  # documents the premise
    assert count(memory["mentions"], memory["graph"], entity="climbing") == 1


def test_unmapped_relation_is_queued_not_written(memory):
    memory["llm"] = FakeLLM(
        _fake_config(),
        responder=lambda prompt, system: json.dumps(
            [
                {
                    "operation": "assert",
                    "subject_id": "new:Melanie",
                    "relation": "UNMAPPED",
                    "suggested_relation": "adopted_pet",
                    "object_id": "new:cat",
                    "polarity": True,
                    "time_expression": None,
                    "evidence_kind": "literal",
                    "span": "I adopted a cat",
                }
            ]
        ),
    )
    result = _ingest(memory, "I adopted a cat", datetime(2023, 1, 1))
    assert result.propositions == []
    assert len(result.unmapped) == 1
    assert memory["unmapped"].grouped_by_suggestion()["adopted_pet"]
