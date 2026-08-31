"""Evidence-bounded answer synthesis for CLIO3."""

from __future__ import annotations

import json
from datetime import timedelta

from fgl.clio3.model import Clio3Answer, RetrievalResult

SYSTEM_ANSWER = (
    "Answer only from supplied memory records and raw episodes. Select the "
    "smallest sufficient evidence set and return JSON only."
)


def _format_interval(interval) -> str | None:
    if interval is None or interval.start is None:
        return None
    start = interval.start
    if interval.granularity == "year":
        return str(start.year)
    if interval.granularity == "month":
        return start.strftime("%B %Y")
    if interval.granularity == "week" and interval.end:
        end = (interval.end - timedelta(days=1)).date()
        if start.month == end.month and start.year == end.year:
            return f"{start.day}–{end.day} {start.strftime('%B %Y')}"
        return f"{start.day} {start.strftime('%B %Y')} – {end.day} {end.strftime('%B %Y')}"
    return f"{start.day} {start.strftime('%B %Y')}"


def _render_records(result: RetrievalResult, memory) -> str:
    rows = []
    for record, score in result.records:
        rows.append(
            {
                "id": record.id,
                "kind": record.kind,
                "type": record.type,
                "participants": [
                    {
                        "entity": memory.clio3_store.get_entity(p.entity_id).name,
                        "role": p.role,
                    }
                    for p in record.participants
                ],
                "attributes": record.attributes,
                "time_expression": record.time_expression,
                "resolved_time": _format_interval(record.valid_time),
                "status": record.status,
                "polarity": record.polarity,
                "episode_id": record.episode_id,
                "evidence": record.evidence,
                "score": round(score, 5),
            }
        )
    return json.dumps(rows, indent=2, ensure_ascii=False)


def _render_episodes(result: RetrievalResult, memory) -> str:
    lines = []
    for episode_id in result.episode_ids:
        try:
            episode = memory.log.get(episode_id)
        except KeyError:
            continue
        lines.append(
            f"[{episode.id}; {episode.ts_ingest.strftime('%d %B %Y')}] "
            f"{episode.speaker}: {episode.text}"
        )
    return "\n".join(lines) or "(none)"


def _coerce(raw, result: RetrievalResult) -> Clio3Answer:
    if not isinstance(raw, dict):
        raw = {}
    values = tuple(
        str(value).strip()
        for value in raw.get("values", [])
        if str(value).strip()
    )
    support = tuple(
        str(value)
        for value in raw.get("support", [])
        if isinstance(value, str) and value
    )
    return Clio3Answer(
        answer_type=str(raw.get("answer_type", result.query.answer_type)),
        values=values,
        support=support,
        abstain=bool(raw.get("abstain", not values)),
        explanation=str(raw.get("explanation", ""))[:300],
    )


def _verify(answer: Clio3Answer, result: RetrievalResult, memory) -> Clio3Answer:
    allowed = set(result.episode_ids)
    support = tuple(dict.fromkeys(value for value in answer.support if value in allowed))
    if answer.abstain or not answer.values or not support:
        return Clio3Answer(answer.answer_type, (), support, True, answer.explanation)

    values = answer.values
    if answer.answer_type == "date":
        selected = next(
            (
                record
                for record, _ in result.records
                if record.episode_id in support and record.valid_time is not None
            ),
            None,
        )
        if selected is not None:
            projected = _format_interval(selected.valid_time)
            if projected:
                values = (projected,)
    return Clio3Answer(answer.answer_type, values, support, False, answer.explanation)


def _render(answer: Clio3Answer) -> str:
    if answer.abstain or not answer.values:
        return "Not mentioned in the conversation"
    if answer.answer_type in {"entity", "number", "boolean", "date", "duration"}:
        return answer.values[0]
    return ", ".join(answer.values) if answer.answer_type == "entity_set" else "; ".join(answer.values)


def answer_query(question: str, result: RetrievalResult, memory):
    prompt = memory.prompts.render(
        "clio3_answer",
        question=question,
        operation=result.query.operation,
        answer_type=result.query.answer_type,
        records=_render_records(result, memory),
        episodes=_render_episodes(result, memory),
    )
    raw = memory.llm.complete_json(
        prompt,
        system=SYSTEM_ANSWER,
        purpose="clio3_answer",
        default={
            "answer_type": result.query.answer_type,
            "values": [],
            "support": [],
            "abstain": True,
            "explanation": "no model answer",
        },
        max_tokens=700,
    )
    verified = _verify(_coerce(raw, result), result, memory)
    return _render(verified), verified


__all__ = ["answer_query"]
