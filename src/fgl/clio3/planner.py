"""LLM-assisted compiler for open-schema CLIO3 queries."""

from __future__ import annotations

import json
import re
from datetime import datetime

from fgl.clio3.model import MemoryQuery

SYSTEM_PLANNER = (
    "You compile a question into a domain-independent event-memory query. "
    "Use only supplied entity names, but use open semantic concepts. JSON only."
)


def _fallback(question: str) -> MemoryQuery:
    low = question.casefold().strip()
    operation = "retrieve"
    answer_type = "entity"
    if low.startswith("how many"):
        operation, answer_type = "count", "number"
    elif low.startswith("when"):
        operation, answer_type = "timeline", "date"
    elif low.startswith("how long"):
        operation, answer_type = "timeline", "duration"
    elif re.match(r"^(did|does|do|is|are|was|were|has|have|had|can|could)\b", low):
        operation, answer_type = "retrieve", "boolean"
    elif re.search(r"\b(why|would|likely|feel|reaction|mean|importance)\b", low):
        operation, answer_type = "infer", "text"
    elif re.search(r"\b(what .* and |which .* and |both|compare)\b", low):
        operation, answer_type = "compare", "entity_set"
    concepts = tuple(
        token
        for token in re.findall(r"[a-z0-9]+", low)
        if token
        not in {
            "the", "a", "an", "did", "does", "do", "is", "are", "was", "were",
            "what", "when", "where", "which", "who", "how", "why", "to", "of",
            "in", "on", "for", "with", "her", "his", "their", "has", "have",
        }
    )
    return MemoryQuery(operation, answer_type, concepts=concepts, max_hops=1)


def _date(value) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d") if isinstance(value, str) else None
    except ValueError:
        return None


def compile_question(question: str, memory) -> MemoryQuery:
    fallback = _fallback(question)
    store = memory.clio3_store
    entities = memory.clio3_retriever.entity_candidates(question, limit=15)
    record_types = sorted({record.type for record in store.records(active_only=True)})
    fallback_payload = {
        "operation": fallback.operation,
        "answer_type": fallback.answer_type,
        "focal_entities": list(fallback.focal_entities),
        "concepts": list(fallback.concepts),
        "start": None,
        "end": None,
        "current_only": fallback.current_only,
        "max_hops": fallback.max_hops,
    }
    prompt = memory.prompts.render(
        "clio3_plan",
        question=question,
        entities=json.dumps(
            [{"id": entity.id, "name": entity.name, "type": entity.type} for entity in entities],
            ensure_ascii=False,
        ),
        record_types=json.dumps(record_types[:80], ensure_ascii=False),
        fallback=json.dumps(fallback_payload, ensure_ascii=False),
    )
    raw = memory.llm.complete_json(
        prompt,
        system=SYSTEM_PLANNER,
        purpose="clio3_plan",
        default=fallback_payload,
        max_tokens=700,
    )
    if not isinstance(raw, dict):
        return fallback
    operation = str(raw.get("operation", fallback.operation))
    if operation not in {"retrieve", "count", "compare", "timeline", "infer"}:
        operation = fallback.operation
    answer_type = str(raw.get("answer_type", fallback.answer_type))
    if answer_type not in {
        "entity", "entity_set", "number", "boolean", "date", "duration", "text"
    }:
        answer_type = fallback.answer_type
    known_names = {entity.name.casefold(): entity.name for entity in store.entities()}
    focal = []
    for value in raw.get("focal_entities", []):
        if not isinstance(value, str):
            continue
        if value in store._entities:
            name = store.get_entity(value).name
        else:
            name = known_names.get(value.casefold())
        if name and name not in focal:
            focal.append(name)
    concepts = tuple(
        str(value).strip()
        for value in raw.get("concepts", fallback.concepts)
        if str(value).strip()
    )
    return MemoryQuery(
        operation=operation,
        answer_type=answer_type,
        focal_entities=tuple(focal),
        concepts=concepts or fallback.concepts,
        start=_date(raw.get("start")),
        end=_date(raw.get("end")),
        current_only=bool(raw.get("current_only", False)),
        max_hops=max(0, min(memory.config.clio3.max_hops, int(raw.get("max_hops", 1)))),
        rationale=str(raw.get("rationale", "compiled by open-schema planner"))[:240],
    )


__all__ = ["compile_question"]
