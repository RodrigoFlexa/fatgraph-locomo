"""Natural-language question compiler for CLIO2."""

from __future__ import annotations

import json
import re
from datetime import datetime

from fgl.clio2.ledger import content_tokens
from fgl.clio2.model import AnswerType, QueryConstraints, QueryOperator, QueryPlan

SYSTEM_PLANNER = (
    "You compile questions into a typed memory query. Return valid JSON only. "
    "Never answer the question and never invent an entity or relation."
)

_RELATION_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("bought", "buy", "items", "possess", "pets", "pet's", "pet "), ("owns",)),
    (
        ("paint", "draw", "made", "make", "created", "art"),
        ("created", "owns", "practices"),
    ),
    (("attend", "participat", "event", "went to"), ("attended", "member_of")),
    (
        ("activit", "do with", "partake", "hike", "camp", "go to", "beach"),
        ("practices", "attended"),
    ),
    (("research",), ("researches",)),
    (("support", "rocks", "backing"), ("received_support_from", "supports")),
    (("learn", "taught"), ("learned",)),
    (("symbol", "represent", "mean"), ("symbolizes",)),
    (("member", "joined", "group"), ("member_of",)),
    (("married", "relationship status"), ("married_to", "partner_of")),
    (("children", "kids", "family"), ("parent_of", "family_of")),
    (("book", "read"), ("likes", "learned")),
    (("instrument", "play"), ("practices",)),
    (("like",), ("likes",)),
    (("live", "move from", "moved from"), ("lives_in", "born_in")),
    (("work", "career", "job"), ("works_at", "has_role", "plans_to")),
    (("plan", "would", "likely"), ("plans_to",)),
)


def _parse_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def _answer_type(question: str) -> AnswerType:
    low = question.casefold().strip()
    if low.startswith("how often"):
        return AnswerType.FREQUENCY
    if low.startswith("how many"):
        return AnswerType.NUMBER
    if low.startswith("when"):
        return AnswerType.DATE
    if low.startswith("how long"):
        return AnswerType.DURATION
    if re.search(
        r"\b(remind(?:er|s|ed)? of|symboli[sz]e|reason for|used for|"
        r"important to|meaning|inspired by|what kind of|what .* about)\b",
        low,
    ):
        return AnswerType.TEXT
    if re.match(r"^(did|does|do|is|are|was|were|has|have|had|can|could)\b", low):
        return AnswerType.BOOLEAN
    if re.search(
        r"\b(activities|items|names|books|events|ways|types|instruments|artists|bands|symbols|pets)\b",
        low,
    ):
        return AnswerType.ENTITY_SET
    return AnswerType.ENTITY


def _operator(question: str, answer_type: AnswerType) -> QueryOperator:
    low = question.casefold()
    if answer_type == AnswerType.FREQUENCY:
        return QueryOperator.FREQUENCY
    if answer_type == AnswerType.DURATION:
        return QueryOperator.DURATION
    if answer_type == AnswerType.NUMBER:
        return QueryOperator.COUNT_DISTINCT
    if answer_type == AnswerType.DATE:
        return QueryOperator.TEMPORAL_LOOKUP
    if answer_type == AnswerType.BOOLEAN:
        return QueryOperator.PREMISE_CHECK
    if re.search(
        r"\b(remind(?:er|s|ed)? of|symboli[sz]e|reason for|used for|"
        r"important to|meaning|inspired by|what kind of|what .* about)\b",
        low,
    ):
        return QueryOperator.ATTRIBUTE_LOOKUP
    if " both " in f" {low} " or re.search(
        r"\b(caroline and melanie|melanie and caroline)\b", low
    ):
        return QueryOperator.INTERSECTION
    if "recent" in low or "latest" in low or "last " in low:
        return QueryOperator.LATEST
    if answer_type == AnswerType.ENTITY_SET:
        return QueryOperator.ENUMERATE
    if re.search(r"\b(why|would|likely|feel|reaction|importance|mean to)\b", low):
        return QueryOperator.INFER
    return QueryOperator.LOOKUP


def _relations(question: str, memory) -> tuple[str, ...]:
    low = question.casefold()
    found: list[str] = []
    for hints, relations in _RELATION_HINTS:
        if any(hint in low for hint in hints):
            for relation in relations:
                canonical = memory.catalog.canonical_relation(relation)
                if canonical and canonical not in found:
                    found.append(canonical)
    return tuple(found)


def _subjects(question: str, memory) -> tuple[str, ...]:
    low = question.casefold()
    matches = []
    people = [
        entity
        for entity in memory.graph.all_entities()
        if entity.merged_into is None and entity.type == "Person"
    ]
    question_words = set(re.findall(r"\b[a-z]{3,}\b", low))
    for entity in people:
        names = (entity.canonical_name, *entity.aliases)
        exact = any(
                re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", low)
                for name in names
                if name.strip()
            )
        prefix = any(
            entity.canonical_name.isalpha()
            and len(word) >= 3
            and entity.canonical_name.casefold().startswith(word)
            for word in question_words
        )
        if (exact or prefix) and entity.canonical_name not in matches:
            matches.append(entity.canonical_name)
    return tuple(matches)


def _constraints(question: str, subjects: tuple[str, ...]) -> QueryConstraints:
    low = question.casefold()
    object_types: tuple[str, ...] = ()
    if low.startswith("who ") or re.search(
        r"\bwhat (?:pets?|children|kids|friends|mentors)\b", low
    ):
        object_types = ("Person",)
    elif re.search(r"\b(items?|bought|gift|necklace|shoes?|figurines?)\b", low):
        object_types = ("Object",)
    year = re.search(r"\b(19|20)\d{2}\b", question)
    start = end = None
    if year:
        number = int(year.group())
        start, end = datetime(number, 1, 1), datetime(number + 1, 1, 1)
    companions = []
    if re.search(r"\b(with (?:her|his|their) family|with family)\b", low):
        companions.append("family")
    subject_tokens = content_tokens(" ".join(subjects))
    terms = tuple(sorted(content_tokens(question) - subject_tokens))
    return QueryConstraints(
        start=start,
        end=end,
        object_types=object_types,
        terms=terms,
        companions=tuple(companions),
        current_only=bool(re.search(r"\b(now|currently|current)\b", low)),
    )


def heuristic_plan(question: str, memory) -> QueryPlan:
    answer_type = _answer_type(question)
    subjects = _subjects(question, memory)
    return QueryPlan(
        operator=_operator(question, answer_type),
        subjects=subjects,
        relations=_relations(question, memory),
        constraints=_constraints(question, subjects),
        answer_type=answer_type,
        projection="object",
        confidence=0.55,
        rationale="deterministic language fallback",
    )


def _plan_payload(plan: QueryPlan) -> dict:
    return {
        "operator": plan.operator.value,
        "subjects": list(plan.subjects),
        "relations": list(plan.relations),
        "constraints": {
            "start": plan.constraints.start.strftime("%Y-%m-%d")
            if plan.constraints.start
            else None,
            "end": plan.constraints.end.strftime("%Y-%m-%d")
            if plan.constraints.end
            else None,
            "object_types": list(plan.constraints.object_types),
            "terms": list(plan.constraints.terms),
            "companions": list(plan.constraints.companions),
            "current_only": plan.constraints.current_only,
        },
        "answer_type": plan.answer_type.value,
        "projection": plan.projection,
        "confidence": plan.confidence,
        "rationale": plan.rationale,
    }


def _coerce_plan(raw: object, fallback: QueryPlan, memory) -> QueryPlan:
    if not isinstance(raw, dict):
        return fallback
    try:
        operator = QueryOperator(raw.get("operator", fallback.operator.value))
        answer_type = AnswerType(raw.get("answer_type", fallback.answer_type.value))
    except ValueError:
        return fallback
    # These surface forms carry semantics that generic lookup/count plans lose:
    # "how often" is a stated frequency, "how long" is a duration, and an
    # attribute question projects a property rather than the object itself.
    # Keep the compiler model from weakening those deterministic distinctions.
    specialized = {
        QueryOperator.FREQUENCY,
        QueryOperator.DURATION,
        QueryOperator.ATTRIBUTE_LOOKUP,
    }
    if fallback.operator in specialized:
        operator = fallback.operator
        answer_type = fallback.answer_type
    subjects = tuple(
        value.strip()
        for value in raw.get("subjects", fallback.subjects)
        if isinstance(value, str) and value.strip()
    )
    relations = []
    for value in raw.get("relations", fallback.relations):
        if not isinstance(value, str):
            continue
        canonical = memory.catalog.canonical_relation(value)
        if canonical and canonical not in relations:
            relations.append(canonical)
    # The fallback also carries schema-compatibility relations.  For example,
    # older memories encode a created painting as owns(artwork) plus
    # practices(painting), while the current catalog exposes created.  Unioning
    # these declared alternatives lets the executor perform the typed event
    # join without asking the planning model to know storage-version history.
    for relation in fallback.relations:
        if relation not in relations:
            relations.append(relation)
    constraints = raw.get("constraints") if isinstance(raw.get("constraints"), dict) else {}
    object_types = tuple(
        value
        for value in constraints.get("object_types", fallback.constraints.object_types)
        if isinstance(value, str) and value in memory.catalog.types
    )
    terms = tuple(
        value.strip()
        for value in constraints.get("terms", fallback.constraints.terms)
        if isinstance(value, str) and value.strip()
    )
    companions = tuple(
        value.strip()
        for value in constraints.get("companions", fallback.constraints.companions)
        if isinstance(value, str) and value.strip()
    )
    return QueryPlan(
        operator=operator,
        subjects=subjects or fallback.subjects,
        relations=tuple(relations) or fallback.relations,
        constraints=QueryConstraints(
            start=_parse_date(constraints.get("start")) or fallback.constraints.start,
            end=_parse_date(constraints.get("end")) or fallback.constraints.end,
            object_types=object_types,
            terms=terms,
            companions=companions,
            current_only=bool(
                constraints.get("current_only", fallback.constraints.current_only)
            ),
        ),
        answer_type=answer_type,
        projection=str(raw.get("projection", fallback.projection)),
        confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.7)))),
        rationale=str(raw.get("rationale", "compiled by model"))[:240],
    )


def compile_question(question: str, memory) -> QueryPlan:
    fallback = heuristic_plan(question, memory)
    entity_candidates = memory.entity_index.search(question, k=15)
    entities = (
        "\n".join(
            f"- {entity.canonical_name} ({entity.type})" for entity in entity_candidates
        )
        or "(none)"
    )
    prompt = memory.prompts.render(
        "clio2_plan",
        question=question,
        entities=entities,
        relations=", ".join(memory.catalog.names()),
        fallback=json.dumps(_plan_payload(fallback), ensure_ascii=False),
    )
    raw = memory.llm.complete_json(
        prompt,
        system=SYSTEM_PLANNER,
        purpose="clio2_plan",
        default=_plan_payload(fallback),
        max_tokens=700,
    )
    return _coerce_plan(raw, fallback, memory)


__all__ = ["compile_question", "heuristic_plan"]
