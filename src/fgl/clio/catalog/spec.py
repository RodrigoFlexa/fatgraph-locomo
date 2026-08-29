"""``RelationSpec``: one row of Sigma, the relation catalog.

Sigma is the only place temporal/structural intelligence about a relation
lives (spec section 4). The consolidation code reads these fields; it never
special-cases a relation name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Cardinality = Literal["functional", "multi"]
Volatility = Literal["static", "slow", "fast"]


@dataclass(frozen=True)
class RelationSpec:
    name: str
    signature: tuple[str, str]  # (subject type, object type)
    cardinality: Cardinality
    volatility: Volatility
    invertible: bool = False
    inverse_name: str | None = None
    #: seconds are not worth it here; a bare integer count of days is all
    #: the "fast" volatility default window needs (spec 5.3).
    default_duration_days: int | None = None
    #: if False, a new functional value does NOT close the previous one
    #: even though cardinality is "functional" -- reserved for relations
    #: where overlap is itself the thing to flag, not to resolve.
    closes_on_new: bool = True
    #: closing this relation's t_valid also closes these, same date
    #: (spec 7.6). Must form a DAG across the whole catalog.
    dependents: tuple[str, ...] = field(default_factory=tuple)
    aliases_surface: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.invertible and not self.inverse_name:
            raise ValueError(
                f"relation {self.name!r} is invertible but has no inverse_name"
            )
