"""Evidence-bounded structured answering for CLIO2."""

from __future__ import annotations

import json
import re

from fgl.clio2.ledger import content_tokens, normalized_value
from fgl.clio2.model import (
    AnswerType,
    ExecutionResult,
    QueryOperator,
    StructuredAnswer,
)
from fgl.llm.prompts import SYSTEM_ANSWERER


def _ordered_evidence_ids(result: ExecutionResult, limit: int) -> list[str]:
    structured = []
    for item in result.items:
        for episode_id in item.episode_ids:
            if episode_id not in structured:
                structured.append(episode_id)
    raw_direct = list(dict.fromkeys(result.candidate_episode_ids))

    temporal_evidence_first = {
        QueryOperator.TEMPORAL_LOOKUP,
        QueryOperator.DURATION,
    }
    is_temporal = (
        result.plan.operator in temporal_evidence_first
        or result.plan.operator == QueryOperator.JOIN
        and result.plan.answer_type in (AnswerType.DATE, AnswerType.DURATION)
    )
    if is_temporal:
        return raw_direct[:limit]

    # The reranked episodic channel already includes structured provenance as
    # one of its signals. Give it most of the context budget, then reserve four
    # slots for high-confidence ledger support that did not survive reranking.
    direct_budget = min(len(raw_direct), max(1, (3 * limit) // 4))
    out = raw_direct[:direct_budget]
    for episode_id in structured:
        if len(out) >= limit:
            break
        if episode_id not in out:
            out.append(episode_id)
    for episode_id in raw_direct[direct_budget:]:
        if len(out) >= limit:
            break
        if episode_id not in out:
            out.append(episode_id)
    return out


def _render_evidence(memory, episode_ids: list[str]) -> str:
    lines = []
    for episode_id in episode_ids:
        try:
            episode = memory.log.get(episode_id)
        except KeyError:
            continue
        lines.append(
            f"[{episode.id}; {episode.ts_ingest.strftime('%d %B %Y')}] "
            f"{episode.speaker}: {episode.text}"
        )
    return "\n".join(lines) or "(none)"


def _render_candidates(result: ExecutionResult) -> str:
    rows = []
    for item in result.items[:30]:
        rows.append(
            {
                "value": item.value,
                "subject": item.subject,
                "relation": item.relation,
                "object_type": item.object_type,
                "episode_ids": item.episode_ids,
                "score": round(item.score, 4),
                "time_expression": item.time_expression,
                "resolved_value": item.resolved_value,
                "valid_start": item.t_valid.start.strftime("%d %B %Y")
                if item.t_valid and item.t_valid.start
                else None,
                "granularity": item.t_valid.granularity if item.t_valid else None,
            }
        )
    return json.dumps(rows, indent=2, ensure_ascii=False)


def _deterministic_answer(result: ExecutionResult) -> StructuredAnswer | None:
    # Deterministic algebra may inspect a broad recall backstop, but its proof
    # is only the episodes attached to the values it actually selected.
    support = tuple(
        dict.fromkeys(
            episode_id
            for item in result.items
            for episode_id in item.episode_ids
        )
    )
    if result.plan.operator == QueryOperator.COUNT_DISTINCT and isinstance(
        result.scalar, int
    ):
        return StructuredAnswer(AnswerType.NUMBER, (str(result.scalar),), support)
    if result.plan.operator == QueryOperator.PREMISE_CHECK and isinstance(
        result.scalar, bool
    ):
        return StructuredAnswer(
            AnswerType.BOOLEAN, ("Yes" if result.scalar else "No",), support
        )
    if result.plan.operator == QueryOperator.PREMISE_CHECK and result.scalar is None:
        return StructuredAnswer(AnswerType.BOOLEAN, (), (), abstain=True)
    if result.plan.operator == QueryOperator.INTERSECTION and result.items:
        return StructuredAnswer(
            AnswerType.ENTITY_SET,
            tuple(item.value for item in result.items),
            support,
        )
    return None


def _default_payload(result: ExecutionResult) -> dict:
    values = [item.value for item in result.items]
    if result.plan.answer_type not in (AnswerType.ENTITY_SET, AnswerType.TEXT):
        values = values[:1]
    return {
        "answer_type": result.plan.answer_type.value,
        "values": values[:12],
        "support": _ordered_evidence_ids(result, 12),
        "abstain": not bool(values),
        "explanation": "deterministic fallback",
    }


def _coerce_answer(raw: object, result: ExecutionResult) -> StructuredAnswer:
    default = _default_payload(result)
    if not isinstance(raw, dict):
        raw = default
    try:
        answer_type = AnswerType(raw.get("answer_type", result.plan.answer_type.value))
    except ValueError:
        answer_type = result.plan.answer_type
    values = tuple(
        str(value).strip()
        for value in raw.get("values", default["values"])
        if str(value).strip()
    )
    support = tuple(
        str(value)
        for value in raw.get("support", default["support"])
        if isinstance(value, str) and value
    )
    return StructuredAnswer(
        answer_type=answer_type,
        values=values,
        support=support,
        abstain=bool(raw.get("abstain", default["abstain"])),
        explanation=str(raw.get("explanation", ""))[:300],
    )


def _value_supported(value: str, result: ExecutionResult, evidence_text: str) -> bool:
    key = normalized_value(value)
    if any(
        key == normalized_value(item.value)
        or key in normalized_value(item.value)
        or normalized_value(item.value) in key
        for item in result.items
    ):
        return True
    tokens = content_tokens(value)
    if not tokens:
        return False
    evidence_tokens = content_tokens(evidence_text)
    return len(tokens & evidence_tokens) / len(tokens) >= 0.6


_WEEKDAY_NAMES = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}


def _episode_anchor(memory, item):
    for episode_id in item.episode_ids:
        try:
            return memory.log.get(episode_id).ts_ingest
        except KeyError:
            continue
    return None


def _selected_temporal_item(answer: StructuredAnswer, result: ExecutionResult):
    """Use the answerer's evidence choice as event selection.

    Temporal arithmetic is deliberately not delegated to the model.  Once it
    selects an event by provenance, the typed interval attached to that event
    is the only legal projection.
    """

    support = set(answer.support)
    if support:
        for item in result.items:
            if support.intersection(item.episode_ids) and item.resolved_value:
                return item
    proposed = {normalized_value(value) for value in answer.values}
    for item in result.items:
        candidates = (item.value, item.time_expression, item.resolved_value)
        if proposed.intersection(
            normalized_value(value) for value in candidates if value
        ):
            return item
    return None


def _project_temporal_value(item, memory) -> str | None:
    """Render a resolved temporal value at the source expression's grain."""

    resolved = item.resolved_value
    if not resolved:
        return None
    expression = (item.time_expression or "").strip()
    low = expression.casefold()
    anchor = _episode_anchor(memory, item)
    if anchor is None or not expression:
        return resolved
    anchor_text = f"{anchor.day} {anchor.strftime('%B %Y')}"

    # LoCoMo asks for the relation to the dated conversation for week-shaped
    # expressions.  Keeping that anchor preserves the evidence's natural
    # granularity and avoids pretending that a whole week is one exact day.
    if re.search(r"\blast\s+week\b", low):
        return f"the week before {anchor_text}"
    if re.search(r"\b(?:last\s+weekend|this\s+past\s+weekend)\b", low):
        return f"the weekend before {anchor_text}"
    weekends_ago = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"weekends?\s+ago\b",
        low,
    )
    if weekends_ago:
        return f"{weekends_ago.group(1)} weekends before {anchor_text}"
    weekday = re.search(
        r"\blast\s+(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|"
        r"thu(?:r|rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
        low,
    )
    if weekday:
        name = _WEEKDAY_NAMES.get(weekday.group(1), weekday.group(1).title())
        return f"the {name} before {anchor_text}"
    return resolved


def verify_answer(
    answer: StructuredAnswer,
    result: ExecutionResult,
    memory,
    allowed_episode_ids: list[str],
) -> StructuredAnswer:
    allowed = set(allowed_episode_ids)
    support = tuple(dict.fromkeys(ep for ep in answer.support if ep in allowed))
    known_episode_ids = {episode.id for episode in memory.log.all()}
    evidence_text = " ".join(
        memory.log.get(episode_id).text
        for episode_id in allowed_episode_ids
        if episode_id in known_episode_ids
    )
    values = tuple(
        dict.fromkeys(
            value
            for value in answer.values
            if _value_supported(value, result, evidence_text)
        )
    )
    if answer.answer_type in (AnswerType.BOOLEAN, AnswerType.NUMBER):
        values = answer.values[:1]
    elif answer.answer_type == AnswerType.DATE and not answer.abstain and answer.values:
        selected = _selected_temporal_item(answer, result)
        if selected is not None:
            projected = _project_temporal_value(selected, memory)
            values = (projected,) if projected else values
    abstain = answer.abstain or not values
    return StructuredAnswer(
        answer.answer_type,
        values,
        support or tuple(allowed_episode_ids if values else ()),
        abstain,
        answer.explanation,
    )


def render_answer(answer: StructuredAnswer) -> str:
    if answer.abstain or not answer.values:
        return "Not mentioned in the conversation"
    values = tuple(value.replace("_", " ") for value in answer.values)
    if answer.answer_type in (
        AnswerType.BOOLEAN,
        AnswerType.NUMBER,
        AnswerType.DATE,
        AnswerType.DURATION,
        AnswerType.FREQUENCY,
        AnswerType.ENTITY,
    ):
        return values[0]
    if answer.answer_type == AnswerType.ENTITY_SET:
        return ", ".join(values)
    return "; ".join(values)


def answer_query(question: str, result: ExecutionResult, memory):
    deterministic = _deterministic_answer(result)
    limit = memory.config.clio2.answer_evidence_limit
    evidence_ids = _ordered_evidence_ids(result, limit)
    if deterministic is not None:
        verified = verify_answer(deterministic, result, memory, evidence_ids)
        return render_answer(verified), verified

    prompt = memory.prompts.render(
        "clio2_answer",
        question=question,
        operator=result.plan.operator.value,
        answer_type=result.plan.answer_type.value,
        candidates=_render_candidates(result),
        evidence=_render_evidence(memory, evidence_ids),
    )
    raw = memory.llm.complete_json(
        prompt,
        system=SYSTEM_ANSWERER,
        purpose="clio2_answer",
        default=_default_payload(result),
        max_tokens=700,
    )
    proposed = _coerce_answer(raw, result)
    verified = verify_answer(proposed, result, memory, evidence_ids)
    return render_answer(verified), verified


__all__ = ["answer_query", "render_answer", "verify_answer"]
