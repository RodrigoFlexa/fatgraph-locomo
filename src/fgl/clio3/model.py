"""Typed contracts for CLIO3's open-schema event graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from fgl.clio.types import Interval

RecordKind = Literal["event", "state", "preference", "plan", "fact"]
RecordOperation = Literal["assert", "update", "close", "retract"]


@dataclass
class OpenEntity:
    id: str
    name: str
    type: str = "entity"
    aliases: list[str] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    merged_into: str | None = None


@dataclass(frozen=True)
class Participant:
    entity_id: str
    role: str


@dataclass
class MemoryRecord:
    id: str
    kind: RecordKind
    type: str
    participants: list[Participant]
    attributes: dict[str, str]
    episode_id: str
    evidence: str
    episode_ts: datetime
    time_expression: str | None = None
    valid_time: Interval | None = None
    operation: RecordOperation = "assert"
    polarity: bool = True
    confidence: float = 0.9
    status: Literal["active", "closed", "retracted", "superseded"] = "active"
    supersedes: list[str] = field(default_factory=list)

    def render(self, entities: dict[str, OpenEntity]) -> str:
        actors = ", ".join(
            f"{entities[p.entity_id].name} as {p.role}"
            for p in self.participants
            if p.entity_id in entities
        )
        attrs = ", ".join(f"{key}: {value}" for key, value in self.attributes.items())
        pieces = [self.kind, self.type, actors, attrs, self.evidence]
        return " | ".join(piece for piece in pieces if piece)


@dataclass(frozen=True)
class RecordLink:
    source_id: str
    relation: str
    target_id: str
    episode_id: str


@dataclass(frozen=True)
class MemoryQuery:
    operation: Literal["retrieve", "count", "compare", "timeline", "infer"]
    answer_type: Literal[
        "entity", "entity_set", "number", "boolean", "date", "duration", "text"
    ]
    focal_entities: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()
    start: datetime | None = None
    end: datetime | None = None
    current_only: bool = False
    max_hops: int = 1
    rationale: str = ""


@dataclass
class RetrievalResult:
    query: MemoryQuery
    records: list[tuple[MemoryRecord, float]] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Clio3Answer:
    answer_type: str
    values: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    abstain: bool = False
    explanation: str = ""


__all__ = [
    "Clio3Answer",
    "MemoryQuery",
    "MemoryRecord",
    "OpenEntity",
    "Participant",
    "RecordLink",
    "RetrievalResult",
]
