"""Architectural contracts for the compiled CLIO2 reader."""

from __future__ import annotations

import json
from datetime import datetime

from tests.clio.helpers import HandFedMemory, load_melanie

from fgl.clio.catalog import load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.facade import Clio
from fgl.clio.index import EpisodeIndex
from fgl.clio.ingest.validate import validate_and_bind
from fgl.clio.types import Episode, Interval
from fgl.clio2.answer import render_answer, verify_answer
from fgl.clio2.executor import QueryExecutor
from fgl.clio2.ledger import FactIndex, SemanticLedger
from fgl.clio2.model import (
    AnswerType,
    ExecutionResult,
    QueryConstraints,
    QueryOperator,
    QueryPlan,
    StructuredAnswer,
)
from fgl.clio2.planner import heuristic_plan
from fgl.config import LLMConfig
from fgl.llm.client import FakeLLM
from fgl.llm.prompts import PromptLibrary
from fgl.retrieval.embeddings import HashingEmbedder


def _memory() -> HandFedMemory:
    return HandFedMemory(load_catalog(ClioConfig.default().catalog_path))


def _ingest(memory, episode_id, date, speaker, text, propositions):
    memory.ingest_episode(
        {
            "id": episode_id,
            "date": date,
            "speaker": speaker,
            "text": text,
            "propositions": propositions,
        }
    )


def _prop(subject, relation, object_, span, time_expression=None):
    return {
        "subject": f"new:{subject}",
        "relation": relation,
        "object": f"new:{object_}",
        "operation": "assert",
        "evidence_kind": "literal",
        "time_expression": time_expression,
        "span": span,
    }


def _executor(memory):
    memory.episode_index = EpisodeIndex(HashingEmbedder(dim=128))
    memory.episode_index.rebuild(memory.log)
    ledger = SemanticLedger(memory)
    index = FactIndex(ledger, HashingEmbedder(dim=128))
    return ledger, QueryExecutor(memory, ledger, index)


def test_ledger_materializes_facts_and_event_views_without_becoming_truth():
    memory = _memory()
    _ingest(
        memory,
        "e1",
        "2023-05-01",
        "Melanie",
        "I painted a sunset",
        [_prop("Melanie", "created", "sunset painting", "I painted a sunset")],
    )

    ledger, _ = _executor(memory)

    assert [(fact.relation, fact.object_name) for fact in ledger.facts] == [
        ("created", "sunset painting")
    ]
    assert ledger.events[0].episode_id == "e1"
    assert ledger.events[0].proposition_ids == (ledger.facts[0].proposition_id,)


def test_intersection_is_set_algebra_not_a_graph_wander():
    memory = _memory()
    _ingest(
        memory,
        "e1",
        "2023-05-01",
        "Caroline",
        "I painted a sunset",
        [_prop("Caroline", "created", "sunset painting", "I painted a sunset")],
    )
    _ingest(
        memory,
        "e2",
        "2023-06-01",
        "Melanie",
        "I painted a sunset and a horse",
        [
            _prop("Melanie", "created", "painting of a sunset", "painted a sunset"),
            _prop("Melanie", "created", "horse painting", "painted a horse"),
        ],
    )
    _, executor = _executor(memory)
    plan = QueryPlan(
        QueryOperator.INTERSECTION,
        subjects=("Caroline", "Melanie"),
        relations=("created",),
        answer_type=AnswerType.ENTITY_SET,
    )

    result = executor.execute("What subject have Caroline and Melanie both painted?", plan)

    assert [item.value for item in result.items] == ["sunset painting"]
    assert set(result.items[0].episode_ids) == {"e1", "e2"}


def test_intersection_rejects_generic_pronoun_objects_as_false_shared_concepts():
    memory = _memory()
    _ingest(
        memory,
        "e1",
        "2023-05-01",
        "Caroline",
        "This painting is blue",
        [_prop("Caroline", "created", "this painting", "This painting")],
    )
    _ingest(
        memory,
        "e2",
        "2023-06-01",
        "Melanie",
        "This painting is red",
        [_prop("Melanie", "created", "this painting", "This painting")],
    )
    _, executor = _executor(memory)
    plan = QueryPlan(
        QueryOperator.INTERSECTION,
        subjects=("Caroline", "Melanie"),
        relations=("created",),
        answer_type=AnswerType.ENTITY_SET,
    )

    result = executor.execute("What did Caroline and Melanie both paint?", plan)

    assert result.items == []


def test_count_distinct_counts_events_not_mentions_or_folded_edges():
    memory = _memory()
    _ingest(
        memory,
        "e1",
        "2023-05-01",
        "Melanie",
        "We went to the beach",
        [_prop("Melanie", "practices", "beach visit", "went to the beach")],
    )
    _ingest(
        memory,
        "e2",
        "2023-08-01",
        "Melanie",
        "We camped at the beach",
        [_prop("Melanie", "practices", "beach visit", "camped at the beach")],
    )
    _ingest(
        memory,
        "e3",
        "2023-09-01",
        "Melanie",
        "I went running",
        [_prop("Melanie", "practices", "running", "went running")],
    )
    _, executor = _executor(memory)
    plan = QueryPlan(
        QueryOperator.COUNT_DISTINCT,
        subjects=("Melanie",),
        relations=("practices",),
        constraints=QueryConstraints(
            start=datetime(2023, 1, 1),
            end=datetime(2024, 1, 1),
            terms=("beach",),
        ),
        answer_type=AnswerType.NUMBER,
    )

    result = executor.execute("How many times did Melanie go to the beach in 2023?", plan)

    assert result.scalar == 2
    assert {fact.episode_id for fact in result.candidate_facts} == {"e1", "e2", "e3"}


def test_count_distinct_collapses_same_day_restatement_of_one_event():
    memory = _memory()
    _ingest(
        memory,
        "e1",
        "2023-05-01",
        "Melanie",
        "We went to the beach",
        [_prop("Melanie", "practices", "beach visit", "went to the beach")],
    )
    _ingest(
        memory,
        "e2",
        "2023-05-01",
        "Melanie",
        "Yes, that beach visit was fun",
        [_prop("Melanie", "practices", "beach visit", "that beach visit")],
    )
    _, executor = _executor(memory)
    plan = QueryPlan(
        QueryOperator.COUNT_DISTINCT,
        subjects=("Melanie",),
        relations=("practices",),
        constraints=QueryConstraints(terms=("beach",)),
        answer_type=AnswerType.NUMBER,
    )

    result = executor.execute("How many times did Melanie go to the beach?", plan)

    assert result.scalar == 1
    assert set(result.items[0].episode_ids) == {"e1", "e2"}


def test_first_person_attribution_uses_speaker_and_repairs_collective_subjects():
    memory = _memory()
    memory.graph.create_entity("Caroline", "Person")
    _ingest(
        memory,
        "e1",
        "2023-05-01",
        "Caroline",
        "I went to the beach",
        [_prop("Melanie", "practices", "beach visit", "I went to the beach")],
    )
    _ingest(
        memory,
        "e2",
        "2023-06-01",
        "Melanie",
        "We painted a sunset",
        [_prop("family", "owns", "sunset painting", "We painted a sunset")],
    )
    _, executor = _executor(memory)

    melanie_beach = executor.execute(
        "Did Melanie go to the beach?",
        QueryPlan(
            QueryOperator.ENUMERATE,
            subjects=("Melanie",),
            relations=("practices",),
            answer_type=AnswerType.ENTITY_SET,
        ),
    )
    melanie_art = executor.execute(
        "What did Melanie paint?",
        QueryPlan(
            QueryOperator.ENUMERATE,
            subjects=("Melanie",),
            relations=("owns",),
            answer_type=AnswerType.ENTITY_SET,
        ),
    )

    assert melanie_beach.items == []
    assert [item.value for item in melanie_art.items] == ["sunset painting"]


def test_latest_orders_by_world_time_after_structural_filtering():
    memory = _memory()
    _ingest(
        memory,
        "e1",
        "2023-05-01",
        "Melanie",
        "I painted a sunrise",
        [_prop("Melanie", "created", "sunrise", "painted a sunrise")],
    )
    _ingest(
        memory,
        "e2",
        "2023-08-01",
        "Melanie",
        "I painted a sunset",
        [_prop("Melanie", "created", "sunset", "painted a sunset")],
    )
    _, executor = _executor(memory)
    plan = QueryPlan(
        QueryOperator.LATEST,
        subjects=("Melanie",),
        relations=("created",),
        answer_type=AnswerType.ENTITY,
    )

    result = executor.execute("What did Melanie paint recently?", plan)

    assert [item.value for item in result.items] == ["sunset"]
    assert result.evidence_episode_ids[0] == "e2"


def test_ledger_reads_consolidated_bitemporal_windows_not_stale_proposition_windows():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    memory = load_melanie(catalog)
    _, executor = _executor(memory)
    plan = QueryPlan(
        QueryOperator.ENUMERATE,
        subjects=("Melanie",),
        relations=("works_at",),
        constraints=QueryConstraints(current_only=True),
        answer_type=AnswerType.ENTITY_SET,
    )

    result = executor.execute("Where does Melanie currently work?", plan)

    assert [item.value for item in result.items] == ["Kaia"]


def test_heuristic_compiler_produces_a_typed_count_plan():
    memory = _memory()
    memory.graph.create_entity("Melanie", "Person")

    plan = heuristic_plan("How many times did Melanie go to the beach in 2023?", memory)

    assert plan.operator == QueryOperator.COUNT_DISTINCT
    assert plan.answer_type == AnswerType.NUMBER
    assert plan.subjects == ("Melanie",)
    assert plan.relations == ("practices", "attended")
    assert plan.constraints.start == datetime(2023, 1, 1)


def test_compiler_distinguishes_frequency_duration_and_attribute_projection():
    memory = _memory()
    memory.graph.create_entity("Melanie", "Person")

    frequency = heuristic_plan(
        "How often does Melanie go to the beach with her kids?", memory
    )
    duration = heuristic_plan("How long has Melanie been practicing art?", memory)
    attribute = heuristic_plan(
        "What is Melanie's bowl a reminder of?", memory
    )

    assert (frequency.operator, frequency.answer_type) == (
        QueryOperator.FREQUENCY,
        AnswerType.FREQUENCY,
    )
    assert (duration.operator, duration.answer_type) == (
        QueryOperator.DURATION,
        AnswerType.DURATION,
    )
    assert (attribute.operator, attribute.answer_type) == (
        QueryOperator.ATTRIBUTE_LOOKUP,
        AnswerType.TEXT,
    )


def test_compiler_resolves_short_nickname_without_typing_kids_as_the_answer():
    memory = _memory()
    memory.graph.create_entity("Melanie", "Person")

    plan = heuristic_plan(
        "What did Mel and her kids paint in their latest project?", memory
    )

    assert plan.subjects == ("Melanie",)
    assert plan.constraints.object_types == ()


def test_temporal_execution_preserves_source_expression_and_defers_projection():
    memory = _memory()
    memory.graph.create_entity("Melanie", "Person")
    _ingest(
        memory,
        "e1",
        "2023-10-20",
        "Melanie",
        "My family went on a road trip last weekend",
        [
            _prop(
                "family",
                "practices",
                "road trip",
                "went on a road trip last weekend",
                "last weekend",
            )
        ],
    )
    # Simulate a snapshot written by an older resolver that persisted this
    # expression as unanchored.  The CLIO2 read model must be rebuildable from
    # the literal expression and episode timestamp after a resolver upgrade.
    stale = memory.staging.all()[0]
    stale.t_valid = Interval(datetime(2023, 10, 20), None)
    stale.unanchored = True
    _, executor = _executor(memory)
    plan = QueryPlan(
        QueryOperator.TEMPORAL_LOOKUP,
        subjects=("Melanie",),
        relations=("practices",),
        constraints=QueryConstraints(terms=("road", "trip")),
        answer_type=AnswerType.DATE,
    )

    result = executor.execute("When did Melanie's family go on a road trip?", plan)

    assert result.scalar is None
    assert result.items[0].time_expression == "last weekend"
    assert result.items[0].t_valid.granularity == "week"
    assert result.items[0].resolved_value == "8 October 2023"
    assert result.candidate_episode_ids[0] == "e1"


def test_temporal_verifier_projects_selected_event_instead_of_rejecting_arithmetic():
    memory = _memory()
    memory.graph.create_entity("Caroline", "Person")
    _ingest(
        memory,
        "e1",
        "2023-05-08",
        "Caroline",
        "I went to a LGBTQ support group yesterday",
        [
            _prop(
                "Caroline",
                "attended",
                "LGBTQ support group",
                "went to a LGBTQ support group yesterday",
                "yesterday",
            )
        ],
    )
    _, executor = _executor(memory)
    result = executor.execute(
        "When did Caroline go to the LGBTQ support group?",
        QueryPlan(
            QueryOperator.TEMPORAL_LOOKUP,
            subjects=("Caroline",),
            relations=("attended",),
            constraints=QueryConstraints(terms=("LGBTQ", "support", "group")),
            answer_type=AnswerType.DATE,
        ),
    )
    # The date is derived from the immutable episode timestamp and therefore
    # does not occur literally in the dialogue text.  This was the regression:
    # the generic lexical verifier used to delete it and force an abstention.
    proposed = StructuredAnswer(
        AnswerType.DATE,
        ("7 May 2023",),
        ("e1",),
    )

    verified = verify_answer(proposed, result, memory, ["e1"])

    assert verified.values == ("7 May 2023",)
    assert not verified.abstain


def test_temporal_verifier_preserves_week_granularity_with_episode_anchor():
    memory = _memory()
    memory.graph.create_entity("Melanie", "Person")
    _ingest(
        memory,
        "e1",
        "2023-10-20",
        "Melanie",
        "My family went on a road trip last weekend",
        [
            _prop(
                "family",
                "practices",
                "road trip",
                "went on a road trip last weekend",
                "last weekend",
            )
        ],
    )
    _, executor = _executor(memory)
    result = executor.execute(
        "When did Melanie's family go on a road trip?",
        QueryPlan(
            QueryOperator.TEMPORAL_LOOKUP,
            subjects=("Melanie",),
            relations=("practices",),
            constraints=QueryConstraints(terms=("road", "trip")),
            answer_type=AnswerType.DATE,
        ),
    )

    verified = verify_answer(
        StructuredAnswer(AnswerType.DATE, ("last weekend",), ("e1",)),
        result,
        memory,
        ["e1"],
    )

    assert verified.values == ("the weekend before 20 October 2023",)
    assert not verified.abstain


def test_temporal_verifier_does_not_project_an_unselected_event():
    memory = _memory()
    result = ExecutionResult(
        QueryPlan(QueryOperator.TEMPORAL_LOOKUP, answer_type=AnswerType.DATE)
    )

    verified = verify_answer(
        StructuredAnswer(AnswerType.DATE, ("7 May 2023",), ("unrelated",)),
        result,
        memory,
        [],
    )

    assert verified.values == ()
    assert verified.abstain


def test_answer_renderer_humanizes_canonical_identifier_values():
    answer = StructuredAnswer(
        AnswerType.ENTITY_SET,
        ("exploring_forests", "roasting_marshmallows"),
    )

    assert render_answer(answer) == "exploring forests, roasting marshmallows"


def test_premise_check_abstains_when_only_an_unrelated_pet_fact_exists():
    memory = _memory()
    _ingest(
        memory,
        "e1",
        "2023-05-01",
        "Melanie",
        "I have a dog named Bailey",
        [_prop("Melanie", "owns", "Bailey", "a dog named Bailey")],
    )
    _, executor = _executor(memory)
    plan = QueryPlan(
        QueryOperator.PREMISE_CHECK,
        subjects=("Melanie",),
        relations=("owns",),
        constraints=QueryConstraints(terms=("Oscar", "pet")),
        answer_type=AnswerType.BOOLEAN,
    )

    result = executor.execute("Is Oscar Melanie's pet?", plan)

    assert result.scalar is None
    assert "premise has no matching fact" in result.diagnostics


def test_declared_unmapped_alias_is_normalized_into_the_ledger_relation():
    catalog = load_catalog(ClioConfig.default().catalog_path)
    memory = _memory()
    speaker = memory.graph.create_entity("Caroline", "Person")
    episode = Episode(
        id="e1",
        session_id="s",
        speaker="Caroline",
        text="Researching adoption agencies",
        ts_ingest=datetime(2023, 5, 1),
        seq=0,
    )
    raw = [
        {
            "operation": "assert",
            "subject_id": speaker.id,
            "relation": "UNMAPPED",
            "suggested_relation": "researches_adoption_agencies",
            "object_id": "new:adoption agencies",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "Researching adoption agencies",
        }
    ]

    result = validate_and_bind(raw, episode, memory.graph, catalog)

    assert result.unmapped == []
    assert result.valid[0]["relation"] == "researches"


def test_clio2_end_to_end_compiles_executes_and_answers_a_count():
    memory = _memory()
    _ingest(
        memory,
        "e1",
        "2023-05-01",
        "Melanie",
        "We went to the beach",
        [_prop("Melanie", "practices", "beach visit", "went to the beach")],
    )
    _ingest(
        memory,
        "e2",
        "2023-08-01",
        "Melanie",
        "We camped at the beach",
        [_prop("Melanie", "practices", "beach visit", "camped at the beach")],
    )

    def responder(prompt, _system):
        assert "# TASK: clio2_plan" in prompt
        return json.dumps(
            {
                "operator": "count_distinct",
                "subjects": ["Melanie"],
                "relations": ["practices"],
                "constraints": {
                    "start": "2023-01-01",
                    "end": "2024-01-01",
                    "object_types": [],
                    "terms": ["beach"],
                    "companions": [],
                    "current_only": False,
                },
                "answer_type": "number",
                "projection": "event",
                "confidence": 0.99,
                "rationale": "count beach events",
            }
        )

    llm = FakeLLM(LLMConfig(provider="fake", cache_enabled=False), responder=responder)
    clio = Clio(
        memory.catalog,
        llm,
        HashingEmbedder(dim=128),
        PromptLibrary("prompts"),
        memory.config,
    )
    clio.log, clio.graph, clio.staging = memory.log, memory.graph, memory.staging

    trace = clio.ask2("How many times did Melanie go to the beach in 2023?")

    assert trace.answer == "2"
    assert trace.count_result == 2
    assert [step.action for step in trace.steps] == ["compile", "execute", "verify"]
    assert set(trace.final_state.evidence_ids) == {"e1", "e2"}
    assert llm.usage.by_purpose["clio2_plan"]["calls"] == 1
