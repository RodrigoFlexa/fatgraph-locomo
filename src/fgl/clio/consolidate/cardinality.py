"""Phase 4 (spec 7.5): a new value for a functional relation closes the
old one -- automatically, with no CLOSE proposition needed. "Melanie moved
to Salvador" alone is enough; nobody has to also say "I stopped living in
Recife".

The closing date is the NEW fact's validity start, never the episode date
it was reported in (spec's own emphasis, verified against
``tests/fixtures/melanie.yaml`` assertion 1): if in June she says she moved
in May, the move closes the old residence in May. That is why the
temporal resolver runs before this phase, not after.
"""

from __future__ import annotations

from datetime import datetime

from fgl.clio.catalog import Catalog
from fgl.clio.graph.store import GraphStore
from fgl.clio.types import Edge, EdgeAddress, Interval

_EPOCH = datetime.min


def phase_4_cardinality(
    touched: set[EdgeAddress], graph: GraphStore, catalog: Catalog
) -> list[Edge]:
    """Returns the edges this phase closed, so phase 5 can propagate from
    them too (a functional supersession is just as much a closure as an
    explicit CLOSE proposition)."""
    closed: list[Edge] = []
    for addr in touched:
        spec = catalog[addr.label]
        if spec.cardinality != "functional" or not spec.closes_on_new:
            continue
        edges = sorted(graph.live_edges_at(addr), key=lambda e: e.t_valid.start or _EPOCH)
        for a, b in zip(edges, edges[1:], strict=False):
            if a.dst_id == b.dst_id:
                continue  # same value restated, not a conflict
            if not a.t_valid.overlaps(b.t_valid):
                continue
            a_start = a.t_valid.start or _EPOCH
            b_start = b.t_valid.start or _EPOCH
            if a_start < b_start:
                a.t_valid = Interval(a.t_valid.start, b.t_valid.start)
                closed.append(a)
            else:
                # identical start: which one is "the past" is undecidable
                # from the timeline alone -- flag both, resolve neither.
                a.conflict_flag = True
                b.conflict_flag = True
    return closed
