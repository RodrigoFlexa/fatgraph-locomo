"""Staging: propositions the extractor produced but consolidation has not
(yet) written to the graph (spec section 2's three-layer table). A
proposition leaves this store only by promotion (status becomes
``"promoted"``) -- it is never deleted, matching P1/P2 for this layer too.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable

from fgl.clio.types import Proposition


class StagingStore:
    def __init__(self) -> None:
        self._propositions: list[Proposition] = []
        self._by_id: dict[str, Proposition] = {}

    def insert(self, props: Iterable[Proposition]) -> None:
        props = list(props)
        self._propositions.extend(props)
        self._by_id.update({p.id: p for p in props})

    def get(self, prop_id: str) -> Proposition:
        """Looks up a proposition by id, promoted or not -- what
        ``evidence()`` (access, M7) needs to walk a trail's ``path`` back
        to the episodes that support it."""
        return self._by_id[prop_id]

    def all(self) -> list[Proposition]:
        """Every proposition ever staged, promoted or not, in insertion
        order. Rebuild (spec 12.3) needs the promoted ones back."""
        return list(self._propositions)

    def pending(self) -> list[Proposition]:
        """Staged propositions, in the order consolidation must apply them:
        by transaction time, i.e. the order the episodes that produced them
        were ingested. Phase 3 (spec 7.4) depends on this order -- a CLOSE
        processed before its ASSERT would find nothing to close.
        """
        staged = [p for p in self._propositions if p.status == "staged"]
        return sorted(staged, key=lambda p: (p.t_tx.start, p.id))

    def pending_count(self) -> int:
        return sum(1 for p in self._propositions if p.status == "staged")

    def keep(self, p: Proposition) -> None:
        """Below tau_promote: stays staged, waiting for independent
        confirmation (spec 7.4, ASSERT branch). No state change needed --
        it is already staged; this exists so the call site reads as a
        decision, not a no-op.
        """

    def orphan(self, p: Proposition) -> None:
        """CLOSE/RETRACT with no existing edge to act on (spec 6.5): stays
        staged and is retried on the next consolidation run, once whatever
        it depends on has (hopefully) arrived.
        """

    def mark_promoted(self, group: Iterable[Proposition]) -> None:
        for p in group:
            p.status = "promoted"

    def group_by(self, key: Callable[[Proposition], Hashable]) -> list[list[Proposition]]:
        """Groups currently-staged propositions by ``key``, preserving each
        group's relative order. Used by phase 7 (spec 7.7) to find
        propositions that corroborate each other."""
        groups: dict[Hashable, list[Proposition]] = defaultdict(list)
        for p in self.pending():
            groups[key(p)].append(p)
        return list(groups.values())
