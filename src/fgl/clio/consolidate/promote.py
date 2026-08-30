"""Phase 7 (spec 7.7): propositions too weak on their own (a single
implicature, confidence 0.55) reach the graph once a SECOND, independent
episode corroborates them -- noisy-OR, not averaging, because two
independent weak signals should outweigh one.

The promoted edge's validity starts at the EARLIEST corroborating
evidence, even if that evidence sat in staging for months before the
second one arrived (spec's own example: an implicature in March,
confirmed in November, is asserted as having been true since March). Its
transaction time, by contrast, starts at the LATEST evidence -- the
promotion happens, and the agent starts believing it, at the moment enough
evidence exists, not retroactively. Confirmed against
``tests/fixtures/melanie.yaml``'s ``practices`` edge (assertion 4): valid
from March, tx from the November episode that tipped it over threshold.
"""

from __future__ import annotations

from fgl.clio.graph.store import GraphStore
from fgl.clio.staging import StagingStore
from fgl.clio.types import Edge, Interval


def combine_confidence(cs: list[float]) -> float:
    """Noisy-OR: P(at least one of these independent weak signals is
    right), not their average."""
    # NB: the grouping key includes polarity, so a denial never
    # corroborates the corresponding affirmation into the graph.
    prod = 1.0
    for c in cs:
        prod *= 1 - c
    return 1 - prod


def phase_7_promote_staged(
    staging: StagingStore, graph: GraphStore, tau_promote: float
) -> list[Edge]:
    created: list[Edge] = []
    for group in staging.group_by(
        lambda p: (p.subject_id, p.relation, p.object_id, p.polarity)
    ):
        if len(group) < 2:
            continue
        if len({p.episode_id for p in group}) < 2:
            continue  # independence: the same episode cannot corroborate itself
        combined = combine_confidence([p.confidence for p in group])
        if combined < tau_promote:
            continue
        starts = [p.t_valid.start for p in group if p.t_valid and p.t_valid.start]
        earliest = min(starts) if starts else None
        latest_tx = max(p.t_tx.start for p in group)
        edge = graph.create_edge(
            src_id=group[0].subject_id,
            label=group[0].relation,
            dst_id=group[0].object_id,
            t_valid=Interval(earliest, None),
            t_tx=Interval(latest_tx, None),
            provenance=[p.id for p in group],
            confidence=combined,
            reinforcement=len(group),
            last_confirmed=latest_tx,
            polarity=group[0].polarity,
            # only if EVERY contributing proposition was a guess: one
            # dated confirmation is enough to anchor the promoted edge
            unanchored=all(pr.unanchored for pr in group),
        )
        staging.mark_promoted(group)
        created.append(edge)
    return created


__all__ = ["combine_confidence", "phase_7_promote_staged"]
