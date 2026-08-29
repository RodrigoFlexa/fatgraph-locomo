"""Phase 8 (spec 7.8): mark contradictions, never resolve them -- the
agent decides what to do with a ``conflict_flag``, possibly by asking the
user.

Two of the table's three rows are already covered elsewhere: a functional
relation with two same-start values is flagged by
:func:`~fgl.clio.consolidate.cardinality.phase_4_cardinality` itself, and
"folding would merge incompatible vertices" belongs to milestone M6. What
is left for this phase is the row phase 4 cannot see: it treats "same
destination" as "the same value restated" and skips it, so a proposition
that names the SAME (subject, relation, object) but with opposite
``polarity`` -- one episode saying so, another denying it -- passes phase 4
unnoticed. This pass catches that, scoped to one consolidation batch: a
polarity contradiction against an edge from a *previous* run would need
that edge's contributing propositions indexed by id, which M1-M4 does not
persist. What is caught here is still real -- every contradiction that
arrives within the same batch.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from fgl.clio.graph.store import GraphStore
from fgl.clio.types import EdgeAddress, Proposition


def phase_8_detect_conflicts(props: list[Proposition], graph: GraphStore) -> None:
    by_key: dict[tuple[str, str, str], list[Proposition]] = defaultdict(list)
    for p in props:
        by_key[(p.subject_id, p.relation, p.object_id)].append(p)

    for group in by_key.values():
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            if a.polarity == b.polarity:
                continue
            if a.t_valid is None or b.t_valid is None:
                continue
            if not a.t_valid.overlaps(b.t_valid):
                continue
            addr = EdgeAddress(a.subject_id, a.relation)
            for e in graph.edges_at(addr):
                if e.dst_id == a.object_id:
                    e.conflict_flag = True
