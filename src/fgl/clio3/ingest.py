"""Autonomous, open-schema ingestion for CLIO3."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from fgl.clio.catalog.spec import RelationSpec
from fgl.clio.temporal import resolve_time
from fgl.clio3.model import MemoryRecord, Participant, RecordLink
from fgl.retrieval.embeddings import cosine

SYSTEM_EXTRACT = (
    "You organize long-term memory into open-schema entities and event/state "
    "records. Extract only what the current episode supports. Return JSON only."
)


@dataclass
class Clio3IngestResult:
    episode: object
    records: list[MemoryRecord] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    raw_count: int = 0

    def summary(self) -> str:
        return (
            f"raw={self.raw_count} kept={len(self.records)} "
            f"rejected={len(self.rejected)}"
        )


def _rank(text: str, rows, render, embedder, limit: int):
    if not rows:
        return []
    query = embedder.encode_one(text)
    vectors = embedder.encode([render(row) for row in rows])
    scored = [(float(cosine(query, vectors[i])), row) for i, row in enumerate(rows)]
    scored.sort(key=lambda pair: -pair[0])
    return [row for _, row in scored[:limit]]


def _context(memory, episode, entity_limit: int, record_limit: int) -> dict:
    store = memory.clio3_store
    entities = [entity for entity in store.entities() if not entity.merged_into]
    entity_candidates = _rank(
        episode.text,
        entities,
        lambda entity: " ".join([entity.name, entity.type, *entity.aliases]),
        memory.entity_index.embedder,
        entity_limit,
    )
    records = store.records(active_only=True)
    record_candidates = _rank(
        episode.text,
        records,
        lambda record: record.render(store._entities),
        memory.entity_index.embedder,
        record_limit,
    )
    previous = memory.log.previous_turns(episode, memory.config.clio3.coref_window)
    return {
        "previous": "\n".join(f"{row.speaker}: {row.text}" for row in previous)
        or "(none)",
        "entities": json.dumps(
            [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "aliases": entity.aliases,
                }
                for entity in entity_candidates
            ],
            ensure_ascii=False,
        ),
        "records": json.dumps(
            [
                {
                    "id": record.id,
                    "kind": record.kind,
                    "type": record.type,
                    "summary": record.render(store._entities),
                    "episode_id": record.episode_id,
                }
                for record in record_candidates
            ],
            ensure_ascii=False,
        ),
    }


def _generic_time_spec(kind: str) -> RelationSpec:
    volatility = "slow" if kind in {"state", "preference", "plan", "fact"} else "fast"
    return RelationSpec(
        name=f"clio3_{kind}",
        signature=("entity", "record"),
        cardinality="multi",
        volatility=volatility,
    )


def _entity_from_ref(ref: object, definitions: dict[str, dict], memory, episode_id: str):
    if not isinstance(ref, str) or not ref.strip():
        return None
    store = memory.clio3_store
    if ref in store._entities:
        return store.get_entity(ref)
    definition = definitions.get(ref, {})
    name = str(definition.get("name") or ref.removeprefix("new:")).strip()
    type_name = str(definition.get("type") or "entity").strip()
    if not name:
        return None
    return store.add_entity(name, type_name, episode_id)


def ingest_turn(
    *,
    text: str,
    speaker: str,
    session_id: str,
    ts: datetime,
    memory,
    episode_id: str | None = None,
) -> Clio3IngestResult:
    episode = memory.log.append(
        session_id=session_id,
        speaker=speaker,
        text=text,
        ts_ingest=ts,
        episode_id=episode_id,
    )
    store = memory.clio3_store
    speaker_entity = store.add_entity(speaker, "person", episode.id)
    context = _context(
        memory,
        episode,
        memory.config.clio3.entity_candidate_limit,
        memory.config.clio3.record_candidate_limit,
    )
    prompt = memory.prompts.render(
        "clio3_extract",
        speaker=speaker,
        speaker_entity_id=speaker_entity.id,
        turn_date=ts.strftime("%d %B %Y"),
        previous_turns=context["previous"],
        turn_text=text,
        entity_candidates=context["entities"],
        record_candidates=context["records"],
    )
    raw = memory.llm.complete_json(
        prompt,
        system=SYSTEM_EXTRACT,
        purpose="clio3_extract",
        default={"entities": [], "records": []},
        max_tokens=memory.config.extraction.max_tokens,
    )
    if not isinstance(raw, dict):
        raw = {}
    raw_entities = raw.get("entities", []) if isinstance(raw.get("entities"), list) else []
    raw_records = raw.get("records", []) if isinstance(raw.get("records"), list) else []
    definitions: dict[str, dict] = {}
    for index, item in enumerate(raw_entities):
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or item.get("id") or f"new_entity_{index}")
        definitions[ref] = item
        entity = _entity_from_ref(ref, definitions, memory, episode.id)
        alias_of = item.get("alias_of")
        if entity is not None and isinstance(alias_of, str) and alias_of in store._entities:
            store.merge_entity(entity.id, alias_of)

    kept: list[MemoryRecord] = []
    rejected: list[str] = []
    pending_links: list[tuple[str, dict]] = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, dict):
            rejected.append(f"record {index}: not an object")
            continue
        evidence = str(item.get("evidence") or "").strip()
        if not evidence or evidence.casefold() not in text.casefold():
            rejected.append(f"record {index}: evidence is not verbatim")
            continue
        kind = str(item.get("kind") or "event").casefold()
        if kind not in {"event", "state", "preference", "plan", "fact"}:
            kind = "fact"
        participants: list[Participant] = []
        raw_participants = item.get("participants", [])
        if isinstance(raw_participants, list):
            for participant in raw_participants:
                if not isinstance(participant, dict):
                    continue
                entity = _entity_from_ref(
                    participant.get("entity"), definitions, memory, episode.id
                )
                if entity is None:
                    continue
                role = str(participant.get("role") or "participant").strip().casefold()
                binding = Participant(entity.id, role)
                if binding not in participants:
                    participants.append(binding)
        if not participants:
            participants.append(Participant(speaker_entity.id, "speaker"))
        attributes = item.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
        attributes = {
            str(key).strip(): str(value).strip()
            for key, value in attributes.items()
            if str(key).strip() and str(value).strip()
        }
        expression = item.get("time_expression")
        expression = str(expression).strip() if expression else None
        interval, tconf = resolve_time(
            expression,
            ts,
            _generic_time_spec(kind),
            locale=memory.config.temporal.locale,
        )
        operation = str(item.get("operation") or "assert").casefold()
        if operation not in {"assert", "update", "close", "retract"}:
            operation = "assert"
        supersedes = [
            value
            for value in item.get("supersedes", [])
            if isinstance(value, str) and value in store._records
        ] if isinstance(item.get("supersedes", []), list) else []
        record = MemoryRecord(
            id=store.new_record_id(),
            kind=kind,
            type=str(item.get("type") or "unspecified").strip().casefold(),
            participants=participants,
            attributes=attributes,
            episode_id=episode.id,
            evidence=evidence,
            episode_ts=ts,
            time_expression=expression,
            valid_time=interval,
            operation=operation,
            polarity=bool(item.get("polarity", True)),
            confidence=0.9 if tconf >= 0.8 else 0.75,
            supersedes=supersedes,
        )
        store.add_record(record)
        kept.append(record)
        for link in item.get("related_records", []):
            if isinstance(link, dict):
                pending_links.append((record.id, link))

    for source_id, raw_link in pending_links:
        target_id = raw_link.get("record_id")
        relation = str(raw_link.get("relation") or "related_to").strip().casefold()
        if isinstance(target_id, str):
            store.add_link(RecordLink(source_id, relation, target_id, episode.id))

    memory._clio3_runtime = None
    return Clio3IngestResult(
        episode=episode,
        records=kept,
        rejected=rejected,
        raw_count=len(raw_records),
    )


__all__ = ["Clio3IngestResult", "ingest_turn"]
