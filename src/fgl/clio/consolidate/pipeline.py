"""Orchestrates the consolidation phases (spec 7.1 + section 8). The order
is not incidental -- entities must resolve before addressing, operations
must apply before cardinality can compare live edges, and cardinality's
own closures must feed dependents alongside phase 3's explicit CLOSEs.

Folding runs LAST here, after promotion and conflict detection -- spec 7.1
lists it earlier (phase 6, before phase 7's promotion), but phase 7 can
promote a group of propositions staged across several EARLIER
consolidation calls (spec 7.7: ``combine_confidence`` over whatever is
still pending), so the vertices its edge touches are not always among
THIS call's own ``props`` and are only known once phase 7 has actually
run. Folding after it, scoped to every vertex ``props`` OR the newly
promoted edges touch, is what lets a same-round merge opportunity (like
spec's own "Rui" / "Rui Sampaio") actually get caught in the round that
created it, rather than waiting one extra cycle.

Folding is opt-in: pass ``log`` and ``journal`` to run it, or omit both to
get exactly milestone M4's behaviour (addresses keyed on whatever vertex
ids phase 1's exact-name matching produced -- see
:mod:`fgl.clio.consolidate.entities`). Every M4 test still calls this
without them on purpose: that scope cut is documented, not accidental, and
a required parameter would have forced every one of those tests to care
about fold to keep passing.

Idempotent and resumable by construction: :meth:`StagingStore.pending`
returns only propositions still ``"staged"``, so re-running consolidation
after a crash reprocesses exactly the propositions that never got
reflected in the graph, and nothing else (spec 7's opening paragraph).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fgl.clio.catalog import Catalog
from fgl.clio.config import ClioConfig
from fgl.clio.consolidate.cardinality import phase_4_cardinality
from fgl.clio.consolidate.conflicts import phase_8_detect_conflicts
from fgl.clio.consolidate.dependents import phase_5_propagate_dependents
from fgl.clio.consolidate.entities import phase_1_resolve_entities
from fgl.clio.consolidate.fold import FoldConfig
from fgl.clio.consolidate.fold import fold as run_fold
from fgl.clio.consolidate.journal import FoldJournal, FoldRecord
from fgl.clio.consolidate.operations import address, apply
from fgl.clio.consolidate.promote import phase_7_promote_staged
from fgl.clio.graph.store import GraphStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.types import Edge, EdgeAddress, Operation


@dataclass
class ConsolidationReport:
    #: edges phase 3 wrote to or touched directly, in processing order
    applied: list[Edge] = field(default_factory=list)
    #: edges phase 7 created from accumulated staged evidence
    promoted: list[Edge] = field(default_factory=list)
    #: addresses phase 3 touched, fed to cardinality/dependents/fold
    touched: set[EdgeAddress] = field(default_factory=set)
    #: merges phase 6 performed, if folding was enabled for this call
    folded: list[FoldRecord] = field(default_factory=list)


def consolidate(
    catalog: Catalog,
    graph: GraphStore,
    staging: StagingStore,
    config: ClioConfig,
    log: LogStore | None = None,
    journal: FoldJournal | None = None,
) -> ConsolidationReport:
    """Runs phases 1-5, (6,) 7-8 over everything currently in ``staging``.

    Callers insert new propositions into ``staging`` before calling this
    (``staging.insert(props)``) -- consolidation always operates over the
    full pending backlog, not just what was just inserted, which is what
    lets an orphaned CLOSE or an under-confident implicature from a much
    earlier call get picked up again here.
    """
    props = staging.pending()
    phase_1_resolve_entities(props, graph, catalog, log)

    tau_promote = config.thresholds.tau_promote
    applied: list[Edge] = []
    explicitly_closed: list[Edge] = []
    touched: set[EdgeAddress] = set()
    for p in props:
        touched.add(address(p))
        edges = apply(p, graph, staging, tau_promote)
        applied.extend(edges)
        if p.operation == Operation.CLOSE:
            explicitly_closed.extend(edges)

    superseded = phase_4_cardinality(touched, graph, catalog)
    phase_5_propagate_dependents([*explicitly_closed, *superseded], graph, catalog)

    promoted = phase_7_promote_staged(staging, graph, tau_promote)
    phase_8_detect_conflicts(props, graph)

    folded: list[FoldRecord] = []
    if log is not None and journal is not None:
        touched_vertices = {p.subject_id for p in props} | {p.object_id for p in props}
        touched_vertices |= {e.src_id for e in promoted} | {e.dst_id for e in promoted}
        if touched_vertices:
            trigger_episode = props[-1].episode_id if props else ""
            folded = run_fold(
                touched_vertices,
                graph,
                log,
                catalog,
                journal,
                FoldConfig(tau_fold=config.thresholds.tau_fold),
                trigger_episode=trigger_episode,
            )

    return ConsolidationReport(
        applied=applied, promoted=promoted, touched=touched, folded=folded
    )
