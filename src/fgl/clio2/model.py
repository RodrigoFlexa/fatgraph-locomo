"""Typed intermediate representation shared by every CLIO2 layer.

This module is intentionally free of stores, prompts, and LLM clients.  A
query plan has one meaning whether it came from a model, a deterministic
fallback, or a unit test.  That separation is the architectural boundary
that prevents natural-language interpretation from leaking into execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from fgl.clio.types import Interval


class QueryOperator(str, Enum):
    LOOKUP = "lookup"
    ENUMERATE = "enumerate"
    COUNT_DISTINCT = "count_distinct"
    INTERSECTION = "intersection"
    JOIN = "join"
    LATEST = "latest"
    TEMPORAL_LOOKUP = "temporal_lookup"
    PREMISE_CHECK = "premise_check"
    INFER = "infer"


class AnswerType(str, Enum):
    ENTITY = "entity"
    ENTITY_SET = "entity_set"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DURATION = "duration"
    TEXT = "text"


@dataclass(frozen=True)
class QueryConstraints:
    start: datetime | None = None
    end: datetime | None = None
    object_types: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    companions: tuple[str, ...] = ()
    current_only: bool = False

    @property
    def interval(self) -> Interval | None:
        if self.start is None and self.end is None:
            return None
        return Interval(self.start, self.end)


@dataclass(frozen=True)
class QueryPlan:
    operator: QueryOperator
    subjects: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    constraints: QueryConstraints = field(default_factory=QueryConstraints)
    answer_type: AnswerType = AnswerType.TEXT
    projection: str = "object"
    subqueries: tuple[QueryPlan, ...] = ()
    confidence: float = 0.0
    rationale: str = ""


@dataclass(frozen=True)
class LedgerFact:
    proposition_id: str
    subject_id: str
    subject_name: str
    subject_type: str
    speaker_id: str
    speaker_name: str
    relation: str
    object_id: str
    object_name: str
    object_type: str
    t_valid: Interval
    t_tx: Interval
    episode_id: str
    episode_ts: datetime
    span: str
    episode_text: str
    confidence: float
    polarity: bool
    promoted: bool
    unanchored: bool

    def render(self) -> str:
        polarity = "not " if not self.polarity else ""
        return (
            f"{self.subject_name} {polarity}{self.relation} {self.object_name}. "
            f"Evidence: {self.span or self.episode_text}"
        )


@dataclass(frozen=True)
class MemoryEvent:
    """An event-shaped materialized view over one episode's ledger facts.

    CLIO's current extractor writes binary propositions.  CLIO2 does not
    pretend those are already a perfect n-ary event ontology: it groups the
    facts that share an episode, retains their roles, and remains fully
    rebuildable.  A future event-native extractor can populate the same
    contract without changing the query executor.
    """

    id: str
    episode_id: str
    ts: datetime
    event_types: tuple[str, ...]
    participant_ids: tuple[str, ...]
    participant_names: tuple[str, ...]
    value_ids: tuple[str, ...]
    value_names: tuple[str, ...]
    proposition_ids: tuple[str, ...]
    text: str


@dataclass
class EvidenceItem:
    value: str
    episode_ids: list[str] = field(default_factory=list)
    proposition_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    subject: str = ""
    relation: str = ""
    object_type: str = ""
    t_valid: Interval | None = None


@dataclass
class ExecutionResult:
    plan: QueryPlan
    items: list[EvidenceItem] = field(default_factory=list)
    candidate_facts: list[LedgerFact] = field(default_factory=list)
    candidate_episode_ids: list[str] = field(default_factory=list)
    scalar: str | int | bool | None = None
    diagnostics: list[str] = field(default_factory=list)

    @property
    def evidence_episode_ids(self) -> list[str]:
        out: list[str] = []
        for item in self.items:
            for episode_id in item.episode_ids:
                if episode_id not in out:
                    out.append(episode_id)
        for episode_id in self.candidate_episode_ids:
            if episode_id not in out:
                out.append(episode_id)
        return out


@dataclass(frozen=True)
class StructuredAnswer:
    answer_type: AnswerType
    values: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    abstain: bool = False
    explanation: str = ""
