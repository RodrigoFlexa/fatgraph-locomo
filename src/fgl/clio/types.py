"""Core data types for CLIO: Chronologically Layered Interval Ontology.

See ``CLIO-especificacao-tecnica.md`` section 3 for the design rationale.
Two invariants hold everywhere in this module and are load-bearing for the
rest of CLIO:

* Intervals are half-open ``[start, end)`` -- this is what makes functional
  closure exact, with no overlap at the boundary instant (spec 3.1).
* Nothing here is ever mutated to erase history: ``Edge.t_valid``/``t_tx``
  are narrowed (an end date is written) but a row is never deleted. See
  P1/P2 in the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

Granularity = Literal["day", "week", "month", "year"]
#: "week" is not in the spec's own literal (section 5.5 names only
#: day/month/year), but table 5.2 has a week-deictic row ("semana
#: passada") that would otherwise lose its 7-day precision by being
#: rounded up to "month". Extending the type by one value is cheaper than
#: silently discarding information the resolver already has.


def _max_opt(a: datetime | None, b: datetime | None) -> datetime | None:
    """max(a, b) with None treated as -infinity. None only if both are."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _min_opt(a: datetime | None, b: datetime | None) -> datetime | None:
    """min(a, b) with None treated as +infinity. None only if both are."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


@dataclass(frozen=True)
class Interval:
    """A half-open interval ``[start, end)``. ``None`` is an open bound:
    ``start=None`` means "since always", ``end=None`` means "until now"."""

    start: datetime | None = None
    end: datetime | None = None
    #: precision the bound was resolved at, e.g. "some point in May 2023"
    #: (month) vs. "14 January 2023" (day). See spec 5.5.
    granularity: Granularity | None = None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"invalid interval: start {self.start} > end {self.end}")

    def intersect(self, other: Interval) -> Interval | None:
        """The interval both intervals agree on, or None if they never do.

        This is the optimistic reading spec 5.5 asks for: two month-grain
        intervals "overlap" if the months could coincide, not only if they
        provably do. That is intentional -- the pessimistic reading would
        kill valid access trails (P4).
        """
        s = _max_opt(self.start, other.start)
        e = _min_opt(self.end, other.end)
        if s is not None and e is not None and s >= e:
            return None
        return Interval(s, e)

    def contains(self, t: datetime) -> bool:
        if self.start is not None and t < self.start:
            return False
        return not (self.end is not None and t >= self.end)

    def overlaps(self, other: Interval) -> bool:
        return self.intersect(other) is not None

    def is_open(self) -> bool:
        return self.end is None


@dataclass
class Episode:
    id: str
    session_id: str
    speaker: str
    text: str
    ts_ingest: datetime
    seq: int
    meta: dict = field(default_factory=dict)


@dataclass
class Entity:
    id: str
    canonical_name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    created_from: str = ""  # episode id
    merged_into: str | None = None  # non-None => this id is an alias
    #: True until a later mention confirms it independently, or a fold
    #: (spec section 8, milestone M6) merges it into an existing vertex.
    provisional: bool = False


class EvidenceKind(str, Enum):
    LITERAL = "literal"  # verbatim span
    COREFERENCE = "coreference"  # literal, with a pronoun resolved
    IMPLICATURE = "implicature"  # "also", "again", "still", "back to"
    CONTEXTUAL = "contextual"  # inferred from broad context, not stated


class Operation(str, Enum):
    ASSERT = "assert"  # states a new fact
    REASSERT = "reassert"  # restates an already-known fact
    CLOSE = "close"  # the fact stopped being true -> closes t_valid
    RETRACT = "retract"  # the fact was never true -> closes t_tx


@dataclass
class Proposition:
    id: str
    subject_id: str
    relation: str
    object_id: str
    operation: Operation
    polarity: bool = True  # False = explicit negation
    time_expression: str | None = None  # LITERAL span, never a date
    t_valid: Interval | None = None  # filled in by the temporal resolver
    #: True when the time expression could not be resolved at all (spec
    #: 5.4). ``t_valid`` then holds the relation's volatility DEFAULT so
    #: the fact still has a plausible shape, but the flag records that no
    #: date was actually read from the text -- which is what stops
    #: ``restrict(axis="valid")`` from reaching it. Writing
    #: ``Interval(None, None)`` here instead (the old behaviour) made an
    #: unresolved date mean "true for all time", which intersects every
    #: window and so can never kill a trail: the exact opposite of what
    #: spec 5.4 asks for.
    unanchored: bool = False
    #: the extractor's ORIGINAL references, before phase 1 rewrote
    #: ``subject_id``/``object_id`` into vertex ids. Kept because rebuild
    #: (spec 12.3) has to replay consolidation from the log against a
    #: fresh, empty graph, and phase 1's in-place rewrite would otherwise
    #: have destroyed the only inputs it needs.
    subject_ref: str = ""
    object_ref: str = ""
    t_tx: Interval = field(default_factory=Interval)  # [episode ts, None)
    evidence_kind: EvidenceKind = EvidenceKind.LITERAL
    confidence: float = 0.0  # derived from evidence_kind, never from the LLM
    span: str = ""  # exact excerpt of the source text
    episode_id: str = ""
    status: Literal["staged", "promoted", "rejected"] = "staged"


@dataclass
class Edge:
    id: str
    src_id: str
    label: str  # a name in Sigma, or "name+inverse suffix"
    dst_id: str
    t_valid: Interval
    t_tx: Interval
    provenance: list[str] = field(default_factory=list)  # proposition ids
    reinforcement: int = 1  # number of independent confirmations
    last_confirmed: datetime | None = None
    confidence: float = 0.0  # max confidence among contributing propositions
    conflict_flag: bool = False
    #: False = this edge records an explicit NEGATION ("I don't live in
    #: Recife"). Carrying it here is what stops a denial from being
    #: written as, and then read back as, the corresponding affirmation:
    #: ``Proposition`` has always had the field, but before this it was
    #: dropped at the graph boundary and only ever compared between two
    #: propositions of one batch. ``follow`` does not walk a negative
    #: edge -- a denial is a fact ABOUT a path, not a path.
    polarity: bool = True
    #: True when the contributing proposition(s) had no resolvable date
    #: (spec 5.4). ``t_valid`` is the relation's volatility default, not
    #: something read from the text, so a query that explicitly restricts
    #: validity must not be answered from this edge.
    unanchored: bool = False


@dataclass
class Mention:
    """Raw occurrence record, for counting. Never touched by consolidation.

    Folding (spec section 8, M6) merges repeated occurrences into one
    canonical edge -- which is exactly what destroys multiplicity. A
    question like "how many times did she mention climbing" cannot be
    answered from the graph; it is answered from this table instead.
    """

    id: str
    episode_id: str
    entity_id: str | None
    surface: str
    ts: datetime
    #: the proposition this mention came from, when it came from one.
    #: ``entity_id`` cannot be filled at ingest time -- the object is
    #: still a ``"new:X"`` reference until consolidation's phase 1
    #: resolves it -- so this is the handle
    #: :meth:`~fgl.clio.log.mentions.MentionStore.relink` uses afterwards
    #: to write ``entity_id`` back. Without it every mention a real
    #: ingest ever produced stayed unlinked forever.
    proposition_id: str | None = None


@dataclass(frozen=True)
class EdgeAddress:
    """The deterministic write address: ``(subject_id, relation)``.

    Not a semantic search, not an LLM decision -- a hash key. This is what
    makes consolidation reproducible (spec section 7.3).
    """

    src: str
    label: str
