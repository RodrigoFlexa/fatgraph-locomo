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

    @property
    def is_alive(self) -> bool:
        return bool(self.trails)
