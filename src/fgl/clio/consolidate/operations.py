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

from fgl.clio.catalog.spec import RelationSpec
from fgl.clio.graph.store import GraphStore
from fgl.clio.staging import StagingStore
from fgl.clio.temporal.resolver import default_for_volatility
from fgl.clio.types import Edge, EdgeAddress, Interval, Operation, Proposition

_EPOCH = datetime.min


def address(p: Proposition) -> EdgeAddress:
    """The deterministic write address: (subject, relation). Not search,
    not an LLM decision -- a hash key (spec 7.3)."""
    return EdgeAddress(src=p.subject_id, label=p.relation)


def find_live_edge(
    edges: list[Edge], dst: str, at: datetime, polarity: bool = True
) -> Edge | None:
    """Among ``edges`` at one address, the one that targets ``dst`` with
    the same ``polarity`` and was both believed (``t_tx`` open) and in
    force (``t_valid`` open, or covering ``at``) at the moment ``at``.
    Used by REASSERT/CLOSE to find what they are reinforcing or ending.

    Polarity is part of the match, not an afterthought: "I don't like X"
    must not reinforce, and must not close, the edge saying she does.
    Those two are contradictory claims about one address, and phase 8 is
    what flags them -- silently collapsing them here would hide it.
    """
    candidates = [
        e
        for e in edges
        if e.dst_id == dst
        and e.polarity == polarity
        and e.t_tx.contains(at)
        and (e.t_valid.end is None or e.t_valid.contains(at))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.t_valid.start or _EPOCH)


def find_edge_any(edges: list[Edge], dst: str, polarity: bool = True) -> Edge | None:
    """The edge RETRACT should act on: any edge at the address targeting
    ``dst`` with the same polarity, preferring the most recently-believed
    one still not retracted (a retraction should hit what the agent
    currently holds, not history).
    """
    candidates = [e for e in edges if e.dst_id == dst and e.polarity == polarity]
    if not candidates:
        return None
    not_retracted = [e for e in candidates if e.t_tx.end is None]
    pool = not_retracted or candidates
    return max(pool, key=lambda e: e.t_tx.start or _EPOCH)


def _find_reinforceable(
    existing: list[Edge], p: Proposition, spec: RelationSpec | None = None
) -> Edge | None:
    """An edge this ASSERT is merely restating: same destination, same
    polarity, still believed, still in force, and starting no later than
    the new claim.

    "Still in force" (``t_valid.end is None``) is the load-bearing part:
    an edge a CLOSE or a functional supersession has already ended must
    NOT absorb a later assertion of the same value. "She lived in Recife,
    then Salvador, then Recife again" is genuinely two Recife intervals,
    and merging them would erase the gap.

    A new claim that starts EARLIER than the edge is still the same fact,
    not a second one -- :func:`_reinforce` widens the edge backwards to the
    earlier evidence rather than opening a rival interval beside it.

    ``spec`` closes the one hole "still in force" cannot see on its own.
    Phase 4 is what CLOSES a superseded functional value, and it runs
    after every proposition in the batch has been applied -- so inside one
    batch, "Recife, then Salvador, then Recife again" reaches this
    function with the first Recife edge still open, and would absorb the
    third statement into it, erasing the Salvador interval between them.
    For a functional relation, an intervening value with a different
    destination therefore blocks reinforcement, exactly as phase 4 would
    have if it had already run.
    """
    for e in existing:
        if e.dst_id != p.object_id or e.polarity != p.polarity:
            continue
        if e.t_tx.end is not None or e.t_valid.end is not None:
            continue
        functional = (
            spec is not None and spec.cardinality == "functional" and spec.closes_on_new
        )
        if functional and _superseded_between(existing, e, p):
            continue
        return e
    return None


def _superseded_between(existing: list[Edge], e: Edge, p: Proposition) -> bool:
    """True if some other believed edge at this address holds a DIFFERENT
    value starting after ``e`` and no later than the new claim -- i.e. the
    fact changed and changed back, and these are two intervals."""
    new_start = p.t_valid.start if p.t_valid is not None else p.t_tx.start
    if new_start is None:
        return False
    for other in existing:
        if other is e or other.dst_id == p.object_id or other.t_tx.end is not None:
            continue
        other_start = other.t_valid.start
        if other_start is None:
            continue
        if e.t_valid.start is not None and other_start <= e.t_valid.start:
            continue
        if other_start <= new_start:
            return True
    return False


def _reinforce(e: Edge, p: Proposition) -> None:
    """One more independent confirmation of an edge already in the graph.

    Widens ``t_valid`` backwards when the new evidence places the fact
    earlier than the edge does (spec 7.7's own principle, applied to a
    direct assertion rather than a promotion): the earliest evidence sets
    the start. Never widens forward -- that is what a CLOSE decides.

    ``t_tx`` widens backwards on the same rule, and that one is what
    spec 17.4's order-invariance property actually rests on. Transaction
    time is when the agent came to believe the fact, and the agent
    believes it because an EPISODE said so -- so the start belongs to the
    earliest episode that said it, not to whichever consolidation call
    happened to reach the graph first. Without this, ingesting the same
    sessions in a different order gives the same fact a different ``t_tx``
    and the memory is not canonical (caught by
    ``test_order_invariance_over_shuffled_sessions``).
    """
    e.reinforcement += 1
    e.last_confirmed = max(e.last_confirmed or p.t_tx.start, p.t_tx.start)
    e.provenance.append(p.id)
    e.confidence = max(e.confidence, p.confidence)
    p_valid_start = p.t_valid.start if p.t_valid is not None else None
    if (
        p_valid_start is not None
        and e.t_valid.start is not None
        and p_valid_start < e.t_valid.start
    ):
        e.t_valid = Interval(p_valid_start, e.t_valid.end, e.t_valid.granularity)
    if (
        p.t_tx.start is not None
        and e.t_tx.start is not None
        and p.t_tx.start < e.t_tx.start
    ):
        e.t_tx = Interval(p.t_tx.start, e.t_tx.end, e.t_tx.granularity)
    # an anchored confirmation redeems an edge written from a guess
    if not p.unanchored:
        e.unanchored = False


def apply(
    p: Proposition,
    graph: GraphStore,
    staging: StagingStore,
    tau_promote: float,
    spec: RelationSpec | None = None,
) -> list[Edge]:
    if p.operation == Operation.ASSERT:
        return _apply_assert(p, graph, staging, tau_promote, spec)

    if p.operation == Operation.REASSERT:
        return _apply_reassert(p, graph, staging, tau_promote, spec)

    if p.operation == Operation.CLOSE:
        return _apply_close(p, graph, staging)

    if p.operation == Operation.RETRACT:
        return _apply_retract(p, graph, staging)

    raise ValueError(f"unknown operation {p.operation!r}")


def _apply_assert(
    p: Proposition,
    graph: GraphStore,
    staging: StagingStore,
    tau_promote: float,
    spec: RelationSpec | None = None,
) -> list[Edge]:
    existing = graph.edges_at(address(p))
    if p.confidence < tau_promote:
        staging.keep(p)
        return []
    # An ASSERT of something the graph already holds is a REASSERT in
    # everything but the label the extractor happened to choose. Spec
    # 7.4 relies on the model saying "reassert"; measured against a
    # real deployment it does not, so the same fact stated in two
    # turns used to write two identical edges -- inflating the edge
    # count (and with it the compression-rate metric), emitting the
    # same vertex twice from one `follow`, and throwing away the
    # reinforcement signal that is the whole point of noticing a
    # repeat. Nothing else can catch it downstream: phase 4 skips
    # pairs with the same destination, and fold merges VERTICES, not
    # edges.
    existing_equivalent = _find_reinforceable(existing, p, spec)
    if existing_equivalent is not None:
        _reinforce(existing_equivalent, p)
        p.status = "promoted"
        return [existing_equivalent]
    # A proposition still carrying `t_valid=None` here never went through
    # ingestion's own fallback (hand-built in a test, or replayed from an
    # older store). Writing `Interval(None, None)` for it would make it
    # true for all time, which intersects every query window and so can
    # never kill a trail -- the exact failure spec 5.4 is written to
    # prevent. The invariant belongs on this side of the boundary too, so
    # nothing that reaches the graph can bypass it.
    t_valid, unanchored = p.t_valid, p.unanchored
    if t_valid is None:
        unanchored = True
        t_valid = (
            default_for_volatility(p.t_tx.start, spec)
            if spec is not None and p.t_tx.start is not None
            else Interval(None, None)
        )
    edge = graph.create_edge(
        src_id=p.subject_id,
        label=p.relation,
        dst_id=p.object_id,
        t_valid=t_valid,
        t_tx=Interval(p.t_tx.start, None),
        provenance=[p.id],
        confidence=p.confidence,
        last_confirmed=p.t_tx.start,
        polarity=p.polarity,
        unanchored=unanchored,
    )
    p.status = "promoted"
    return [edge]


def _apply_reassert(
    p: Proposition,
    graph: GraphStore,
    staging: StagingStore,
    tau_promote: float,
    spec: RelationSpec | None = None,
) -> list[Edge]:
    existing = graph.edges_at(address(p))
    e = find_live_edge(existing, p.object_id, p.t_tx.start, p.polarity)
    if e is None:
        # Nothing to reinforce -- treat it as a fresh assertion, but
        # WITHOUT rewriting p.operation. The proposition is the log's
        # derived record; rebuild (spec 12.3) replays it against an
        # empty graph, and a REASSERT silently rewritten to ASSERT by
        # a previous run would replay as something the extractor
        # never said.
        return _apply_assert(p, graph, staging, tau_promote, spec)
    _reinforce(e, p)
    p.status = "promoted"
    return [e]


def _apply_close(p: Proposition, graph: GraphStore, staging: StagingStore) -> list[Edge]:
    existing = graph.edges_at(address(p))
    e = find_live_edge(existing, p.object_id, p.t_tx.start, p.polarity)
    if e is None:
        staging.orphan(p)
        return []
    close_at = p.t_valid.start if p.t_valid is not None else p.t_tx.start
    e.t_valid = Interval(e.t_valid.start, close_at)
    p.status = "promoted"
    return [e]


def _apply_retract(p: Proposition, graph: GraphStore, staging: StagingStore) -> list[Edge]:
    existing = graph.edges_at(address(p))
    e = find_edge_any(existing, p.object_id, p.polarity)
    if e is None:
        staging.orphan(p)
        return []
    e.t_tx = Interval(e.t_tx.start, p.t_tx.start)
    p.status = "promoted"
    return [e]
