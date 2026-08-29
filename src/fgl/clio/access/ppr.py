"""Personalized PageRank for ``expand`` (spec 9.5): the associative
fallback used when a question does not name a relation. Deliberately
loose -- ``expand`` only picks candidate entry points; it is ``follow``
that re-establishes temporal coherence on top of them, and spec 9.5 is
explicit that a trail should never be answered from ``expand`` alone.

Restricted to ``max_hops`` from the seeds so one call cannot wander the
whole graph -- "associative" is meant to mean "nearby", not "anywhere
reachable at all".
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np

from fgl.clio.graph.store import GraphStore


def _adjacency(graph: GraphStore, tx_point: datetime) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for e in graph.all_edges():
        if not e.t_tx.contains(tx_point):
            continue  # retracted in this view -- not a real association
        adj[e.src_id].append(e.dst_id)
        adj[e.dst_id].append(e.src_id)
    return adj


def _reachable_within(
    adj: dict[str, list[str]], seeds: set[str], max_hops: int
) -> set[str]:
    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(max_hops):
        nxt = {u for v in frontier for u in adj.get(v, []) if u not in visited}
        if not nxt:
            break
        visited |= nxt
        frontier = nxt
    return visited


def personalized_pagerank(
    graph: GraphStore,
    seeds: dict[str, float],
    tx_point: datetime,
    alpha: float = 0.15,
    max_hops: int = 2,
    iterations: int = 20,
) -> dict[str, float]:
    """``alpha`` is the restart probability back to ``seeds``. Returns a
    score per vertex reachable from the seeds within ``max_hops``,
    including the seeds themselves."""
    if not seeds:
        return {}
    adj = _adjacency(graph, tx_point)
    nodes = sorted(_reachable_within(adj, set(seeds), max_hops))
    if not nodes:
        return dict(seeds)
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    total = sum(seeds.values()) or 1.0
    p0 = np.zeros(n)
    for v, w in seeds.items():
        if v in idx:
            p0[idx[v]] = w / total

    p = p0.copy()
    for _ in range(iterations):
        nxt = np.zeros(n)
        for v in nodes:
            neighbours = [u for u in adj.get(v, []) if u in idx]
            if not neighbours:
                nxt[idx[v]] += p[idx[v]]  # dangling vertex: mass stays put
                continue
            share = p[idx[v]] / len(neighbours)
            for u in neighbours:
                nxt[idx[u]] += share
        p = (1 - alpha) * nxt + alpha * p0

    return {v: float(p[idx[v]]) for v in nodes}
