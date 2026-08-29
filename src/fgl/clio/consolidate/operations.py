"""Phase 3 (spec 7.4): turn one proposition into a graph write.

The CLOSE/RETRACT split is the single most important distinction in this
module: CLOSE narrows ``t_valid`` (the fact stopped being true in the
world), RETRACT narrows ``t_tx`` (the agent was wrong that it was ever
true). Different field, different observable effect, and a unit test
(``tests/clio/test_operations.py``) pins that the other field never moves.

A proposition's ``status`` becomes ``"promoted"`` exactly when this
function writes or touches a graph edge because of it -- whether that
happens here directly (high confidence) or later, via phase 7's
accumulated-evidence path. That is what makes :meth:`StagingStore.pending`
safe to reprocess on every consolidation run: anything already reflected
in the graph is excluded from then on, and only genuinely unresolved
propositions (below threshold, or an orphaned CLOSE/RETRACT) come back.
"""

from __future__ import annotations

from datetime import datetime

from fgl.clio.graph.store import GraphStore
from fgl.clio.staging import StagingStore
from fgl.clio.types import Edge, EdgeAddress, Interval, Operation, Proposition

_EPOCH = datetime.min


def address(p: Proposition) -> EdgeAddress:
    """The deterministic write address: (subject, relation). Not search,
    not an LLM decision -- a hash key (spec 7.3)."""
    return EdgeAddress(src=p.subject_id, label=p.relation)


def find_live_edge(edges: list[Edge], dst: str, at: datetime) -> Edge | None:
    """Among ``edges`` at one address, the one that targets ``dst`` and was
    both believed (``t_tx`` open) and in force (``t_valid`` open, or
    covering ``at``) at the moment ``at``. Used by REASSERT/CLOSE to find
    what they are reinforcing or ending.
    """
    candidates = [
        e
        for e in edges
        if e.dst_id == dst
        and e.t_tx.contains(at)
        and (e.t_valid.end is None or e.t_valid.contains(at))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.t_valid.start or _EPOCH)


def find_edge_any(edges: list[Edge], dst: str) -> Edge | None:
    """The edge RETRACT should act on: any edge at the address targeting
    ``dst``, preferring the most recently-believed one still not retracted
    (a retraction should hit what the agent currently holds, not history).
    """
    candidates = [e for e in edges if e.dst_id == dst]
    if not candidates:
        return None
    not_retracted = [e for e in candidates if e.t_tx.end is None]
    pool = not_retracted or candidates
    return max(pool, key=lambda e: e.t_tx.start or _EPOCH)


def apply(
    p: Proposition, graph: GraphStore, staging: StagingStore, tau_promote: float
) -> list[Edge]:
    addr = address(p)
    existing = graph.edges_at(addr)

    if p.operation == Operation.ASSERT:
        if p.confidence < tau_promote:
            staging.keep(p)
            return []
        edge = graph.create_edge(
            src_id=p.subject_id,
            label=p.relation,
            dst_id=p.object_id,
            t_valid=p.t_valid if p.t_valid is not None else Interval(None, None),
            t_tx=Interval(p.t_tx.start, None),
            provenance=[p.id],
            confidence=p.confidence,
            last_confirmed=p.t_tx.start,
        )
        p.status = "promoted"
        return [edge]

    if p.operation == Operation.REASSERT:
        e = find_live_edge(existing, p.object_id, p.t_tx.start)
        if e is None:
            p.operation = Operation.ASSERT
            return apply(p, graph, staging, tau_promote)
        e.reinforcement += 1
        e.last_confirmed = p.t_tx.start
        e.provenance.append(p.id)
        e.confidence = max(e.confidence, p.confidence)
        p.status = "promoted"
        return [e]

    if p.operation == Operation.CLOSE:
        e = find_live_edge(existing, p.object_id, p.t_tx.start)
        if e is None:
            staging.orphan(p)
            return []
        close_at = p.t_valid.start if p.t_valid is not None else p.t_tx.start
        e.t_valid = Interval(e.t_valid.start, close_at)
        p.status = "promoted"
        return [e]

    if p.operation == Operation.RETRACT:
        e = find_edge_any(existing, p.object_id)
        if e is None:
            staging.orphan(p)
            return []
        e.t_tx = Interval(e.t_tx.start, p.t_tx.start)
        p.status = "promoted"
        return [e]

    raise ValueError(f"unknown operation {p.operation!r}")
