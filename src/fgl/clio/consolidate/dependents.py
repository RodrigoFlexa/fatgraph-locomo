"""Phase 5 (spec 7.6): closing a relation can close others on the same
date, as declared by ``Sigma.dependents`` -- e.g. leaving a job also ends
who manages you there, with no separate statement needed. The catalog
loader already checked ``dependents`` forms a DAG (a cycle here would spin
forever), so this is a plain bounded breadth-first propagation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from fgl.clio.catalog import Catalog
from fgl.clio.graph.store import GraphStore
from fgl.clio.types import Edge, EdgeAddress, Interval


def phase_5_propagate_dependents(
    closed: Iterable[Edge], graph: GraphStore, catalog: Catalog
) -> None:
    queue: deque[Edge] = deque(closed)
    seen: set[str] = set()
    while queue:
        e = queue.popleft()
        if e.id in seen or e.t_valid.end is None:
            continue
        seen.add(e.id)
        spec = catalog[e.label]
        for dep_label in spec.dependents:
            addr = EdgeAddress(e.src_id, dep_label)
            for d in graph.live_edges_at(addr):
                if d.id == e.id:
                    continue
                # A dependent that only starts at/after the cutoff is the
                # REPLACEMENT fact (e.g. the new manager under the new
                # job), not the predecessor closing this relation is meant
                # to end -- without this guard, a manager asserted in the
                # very same batch as the job change would be born already
                # closed (caught by tests/clio/test_melanie_fixture.py).
                if d.t_valid.start is not None and d.t_valid.start >= e.t_valid.end:
                    continue
                if d.t_valid.end is None or d.t_valid.end > e.t_valid.end:
                    d.t_valid = Interval(d.t_valid.start, e.t_valid.end)
                    queue.append(d)
