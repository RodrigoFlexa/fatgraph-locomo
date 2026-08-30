"""``Trail`` and ``AccessState`` (spec section 9.1): the unit and the state
of the access algebra. Every movement in :mod:`fgl.clio.access.movements`
takes one ``AccessState`` and returns another -- there is no other kind of
memory-read call.

Invariant I1 (spec 9.1): for every ``Trail``, ``window`` is exactly the
intersection of every edge interval on its path, and a trail whose window
would empty is never constructed -- ``follow``/``restrict`` drop it instead.
Path coherence is a property of this data structure, not a check a caller
can forget to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from fgl.clio.types import Interval


@dataclass(frozen=True)
class Trail:
    vertex_id: str
    #: the intersection of every edge interval traversed so far
    window: Interval
    #: proposition ids traversed, in order -- resolved to episode text by
    #: the ``evidence`` movement
    path: tuple[str, ...] = ()
    #: labels traversed, kept only to explain a trail to the reader/agent
    labels: tuple[str, ...] = ()
    #: retrieval relevance carried across movements.  Algebraic validity is
    #: still decided exclusively by the intervals/path above; this score only
    #: decides which of many valid trails fit in the reader's finite budget.
    score: float = 0.0


@dataclass
class AccessState:
    trails: list[Trail]
    #: point of view on ``t_tx`` -- "what did the agent believe as of this
    #: instant". Defaults to now; ``restrict(axis="tx", ...)`` changes it.
    tx_point: datetime = field(default_factory=datetime.now)
    #: how many trails died on the movement that produced this state
    dead_count: int = 0
    #: why they died -- one of "no_edge_with_label", "all_edges_retracted",
    #: "empty_temporal_window", or None (no death, or genuinely mixed
    #: causes not worth collapsing to one label)
    death_cause: str | None = None
    budget_used: int = 0
    #: True once ``restrict(axis="valid", ...)`` has been applied. Spec
    #: 5.4: a proposition whose date could not be resolved is reachable by
    #: the log's partial order but NOT by a restriction on validity, so
    #: ``follow`` stops walking ``unanchored`` edges once this is set.
    #: Without the flag there is no way to tell an open window that was
    #: never narrowed from one the caller deliberately narrowed to
    #: everything.
    valid_restricted: bool = False
    #: Episodes found by the episodic half of ``anchor``. Candidates are shown
    #: to the reader, but are not answer evidence until an explicit
    #: ``select_evidence`` movement promotes them. This distinction prevents a
    #: merely similar turn from silently supporting a false premise.
    candidate_episode_ids: tuple[str, ...] = ()
    #: Candidate episodes explicitly selected as support by the reader.  They
    #: live on the state rather than on a synthetic graph vertex: P1 says the
    #: log is truth, and an UNMAPPED turn must remain retrievable even when
    #: Sigma has no edge capable of locating it yet.
    evidence_ids: tuple[str, ...] = ()
    #: Original query retained for deterministic ranking and diagnostics.
    query: str = ""

    @property
    def is_alive(self) -> bool:
        return bool(self.trails or self.evidence_ids)
