"""Architectural contracts for CLIO3's open-schema memory."""

from __future__ import annotations

import json
import re
from datetime import datetime

from fgl.clio.catalog import load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.facade import Clio
from fgl.clio.persist import load_memory, save_memory
from fgl.clio3.model import MemoryQuery, MemoryRecord, Participant
from fgl.clio3.retrieval import OpenGraphRetriever
from fgl.config import LLMConfig
from fgl.llm.client import FakeLLM
from fgl.llm.prompts import PromptLibrary
from fgl.retrieval.embeddings import HashingEmbedder


def _clio(responder):
    config = ClioConfig.default()
    config.reader = "clio3"
    llm = FakeLLM(LLMConfig(provider="fake", cache_enabled=False), responder=responder)
    return Clio(
        load_catalog(config.catalog_path),
        llm,
        HashingEmbedder(dim=128),
        PromptLibrary("prompts"),
        config,
    )


def test_clio3_extracts_an_open_event_type_without_a_relation_catalog():
    def responder(prompt, _system):
        assert "# TASK: clio3_extract" in prompt
        speaker_id = re.search(r"SPEAKER ENTITY: (c3e_\d+)", prompt).group(1)
        return json.dumps(
            {
                "entities": [],
                "records": [
                    {
                        "kind": "event",
                        "type": "prototype construction",
                        "participants": [{"entity": speaker_id, "role": "inventor"}],
                        "attributes": {"artifact": "solar desalination membrane"},
                        "time_expression": "yesterday",
                        "operation": "assert",
                        "polarity": True,
                        "supersedes": [],
                        "related_records": [],
                        "evidence": "prototyped a solar desalination membrane yesterday",
                    }
                ],
            }
        )

    clio = _clio(responder)
    result = clio.ingest(
        "I prototyped a solar desalination membrane yesterday",
        speaker="Asha",
        session_id="s1",
        ts=datetime(2025, 2, 10),
        episode_id="e1",
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record.type == "prototype construction"
    assert record.attributes == {"artifact": "solar desalination membrane"}
    assert record.valid_time.start == datetime(2025, 2, 9)
    assert clio.graph.all_edges() == []  # no hidden dependency on CLIO2's catalog graph
    assert clio.clio3_store.edge_count() == 1


def test_clio3_end_to_end_lets_the_model_plan_but_code_bounds_evidence():
    def responder(prompt, _system):
        if "# TASK: clio3_extract" in prompt:
            speaker_id = re.search(r"SPEAKER ENTITY: (c3e_\d+)", prompt).group(1)
            return json.dumps(
                {
                    "entities": [],
                    "records": [
                        {
                            "kind": "event",
                            "type": "prototype construction",
                            "participants": [{"entity": speaker_id, "role": "inventor"}],
                            "attributes": {"artifact": "solar desalination membrane"},
                            "time_expression": "yesterday",
                            "operation": "assert",
                            "polarity": True,
                            "supersedes": [],
                            "related_records": [],
                            "evidence": "prototyped a solar desalination membrane yesterday",
                        }
                    ],
                }
            )
        if "# TASK: clio3_plan" in prompt:
            return json.dumps(
                {
                    "operation": "retrieve",
                    "answer_type": "entity",
                    "focal_entities": ["Asha"],
                    "concepts": ["prototype construction", "artifact"],
                    "start": None,
                    "end": None,
                    "current_only": False,
                    "max_hops": 0,
                    "rationale": "retrieve Asha's constructed artifact",
                }
            )
        assert "# TASK: clio3_answer" in prompt
        return json.dumps(
            {
                "answer_type": "entity",
                "values": ["solar desalination membrane"],
                "support": ["e1", "invented-evidence"],
                "abstain": False,
                "explanation": "the event attribute names the artifact",
            }
        )

    clio = _clio(responder)
    clio.ingest(
        "I prototyped a solar desalination membrane yesterday",
        speaker="Asha",
        session_id="s1",
        ts=datetime(2025, 2, 10),
        episode_id="e1",
    )

    trace = clio.ask("What did Asha prototype?")

    assert trace.answer == "solar desalination membrane"
    assert trace.final_state.evidence_ids == ("e1",)
    assert [step.action for step in trace.steps] == ["compile", "retrieve", "verify"]


def test_clio3_expands_between_records_through_shared_entities():
    clio = _clio(lambda _prompt, _system: "{}")
    asha = clio.clio3_store.add_entity("Asha", "person", "e1")
    lab = clio.clio3_store.add_entity("North Lab", "organization", "e1")
    clio.clio3_store.add_record(
        MemoryRecord(
            id=clio.clio3_store.new_record_id(),
            kind="event",
            type="prototype construction",
            participants=[Participant(asha.id, "inventor"), Participant(lab.id, "site")],
            attributes={"artifact": "membrane"},
            episode_id="e1",
            evidence="built a membrane",
            episode_ts=datetime(2025, 1, 1),
        )
    )
    clio.clio3_store.add_record(
        MemoryRecord(
            id=clio.clio3_store.new_record_id(),
            kind="event",
            type="field deployment",
            participants=[Participant(lab.id, "operator")],
            attributes={"location": "Atacama"},
            episode_id="e2",
            evidence="deployed it in Atacama",
            episode_ts=datetime(2025, 2, 1),
        )
    )
    clio.log.append("s", "Asha", "built a membrane", datetime(2025, 1, 1), episode_id="e1")
    clio.log.append("s", "Asha", "deployed it in Atacama", datetime(2025, 2, 1), episode_id="e2")
    clio.config.clio3.seed_records = 1
    retriever = OpenGraphRetriever(clio)

    result = retriever.execute(
        "Where was the membrane project deployed?",
        MemoryQuery(
            "retrieve",
            "entity",
            concepts=("prototype membrane",),
            max_hops=1,
        ),
    )

    assert {record.id for record, _ in result.records} == {"c3r_0000000", "c3r_0000001"}


def test_clio3_memory_round_trips_with_open_types(tmp_path):
    clio = _clio(lambda _prompt, _system: "{}")
    entity = clio.clio3_store.add_entity("Asha", "researcher", "e1")
    clio.log.append("s", "Asha", "I built a membrane", datetime(2025, 1, 1), episode_id="e1")
    clio.clio3_store.add_record(
        MemoryRecord(
            id=clio.clio3_store.new_record_id(),
            kind="event",
            type="prototype construction",
            participants=[Participant(entity.id, "inventor")],
            attributes={"artifact": "membrane"},
            episode_id="e1",
            evidence="built a membrane",
            episode_ts=datetime(2025, 1, 1),
        )
    )
    path = save_memory(clio, tmp_path / "memory.json")

    loaded = load_memory(
        path,
        clio.catalog,
        llm=clio.llm,
        embedder=HashingEmbedder(dim=128),
        prompts=PromptLibrary("prompts"),
        config=clio.config,
    )

    assert loaded.clio3_store.entities()[0].type == "researcher"
    assert loaded.clio3_store.records()[0].type == "prototype construction"
    assert loaded.clio3_store.records()[0].attributes == {"artifact": "membrane"}


def test_clio3_preserves_homonyms_and_lifecycle_history():
    clio = _clio(lambda _prompt, _system: "{}")
    person = clio.clio3_store.add_entity("Aurora", "person", "e1")
    project = clio.clio3_store.add_entity("Aurora", "project", "e1")
    assert person.id != project.id

    prior = clio.clio3_store.add_record(
        MemoryRecord(
            id=clio.clio3_store.new_record_id(),
            kind="state",
            type="project status",
            participants=[Participant(project.id, "project")],
            attributes={"status": "active"},
            episode_id="e1",
            evidence="Aurora is active",
            episode_ts=datetime(2025, 1, 1),
        )
    )
    clio.clio3_store.add_record(
        MemoryRecord(
            id=clio.clio3_store.new_record_id(),
            kind="event",
            type="project closure",
            participants=[Participant(project.id, "project")],
            attributes={},
            episode_id="e2",
            evidence="Aurora was closed",
            episode_ts=datetime(2025, 2, 1),
            operation="close",
            supersedes=[prior.id],
        )
    )

    assert prior.status == "closed"
    assert len(clio.clio3_store.records()) == 2
