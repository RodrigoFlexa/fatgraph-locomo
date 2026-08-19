"""Group-Steiner connection over the typed bipartite graph -- condition L4.

The question this asks that no other channel asks
--------------------------------------------------
Every read in L1, L2 and L3 asks the same shape of question: *which episodes are
near the things the query mentioned?* Nearness is a per-slot quantity summed at
the end, so an episode scores well by matching one slot very strongly, and an
episode that sits between three of them scores no better than the sum of its
parts.

But a multi-hop question is not asking for nearness. It is asking:

    what is the smallest piece of this memory that holds ALL of these together?

That is the group Steiner tree problem, and it is the classical formulation of
keyword search over graphs (BANKS, BLINKS, DPBF). Two things follow from posing
it properly, and both matter here.

**A join score with an AND in it.** The rooted-star relaxation -- for each
candidate root, the sum of its distances to every terminal -- is the standard
tractable stand-in for the optimal tree, and it is a *conjunction*: a root that
cannot reach one terminal at all scores nothing, however close it is to the
others. Multi-hop was 0.376 in the last run precisely because the additive
channels have no way to say "and".

**An abstention signal with resolution.** The corner test is binary: the
(actor, slot) pair co-occurs or it does not, and measured on this benchmark it
caught 20 of 446 adversarials at the cost of 38 false positives on 1540
substantive questions -- a losing trade. The Steiner cost is continuous. "These
slots exist, but nothing in this memory holds them within distance *d* of each
other" is a much finer statement than "this pair never co-occurred", and it is
exactly what an adversarial question looks like: real ingredients, a combination
that never happened.

The metric, and why the hub rule reappears
-------------------------------------------
Unit distances would make every pair of episodes two steps apart through the
``actor`` vertex, and the whole structure would collapse -- the classic failure
of keyword search over graphs with hubs. So the cost of *entering* a slot is
``1 + log(degree)``: routing through something forty episodes mention is
expensive, routing through something two episodes mention is cheap. And above
the calibrated per-kind cut-off a slot is not traversable at all.

That is the same rule the walk in :mod:`fgl.retrieval.propagation` uses and the
same cut-off L2's scorer uses -- **a hub is a filter, never a bridge** -- stated
once in :meth:`fgl.retrieval.slots.SlotRetriever.is_hub` and obeyed by all three
reads. It is what makes L2, L3 and L4 one design rather than three retrievers
that happen to share a graph.

Calibrating the abstention without labels
------------------------------------------
The threshold is not a swept number. For a question with *k* terminals, the
memory is asked how far apart *k random non-hub slots* usually are, by sampling
tuples and computing the same rooted-star cost; the question abstains when it
lands in the upper tail of that null distribution. So the criterion is "these
slots are further apart than 95% of arbitrary combinations of the same size in
this corpus" -- a property of the memory, measurable on unlabelled data, and
free of the benchmark's answer key. See ``docs/ASSUMPTIONS.md``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from fgl.memory.slots import KIND_EPISODE

#: Returned when a terminal reaches nothing, so callers can treat "no support"
#: as a number rather than as a special case.
UNREACHABLE = float("inf")


@dataclass
class SteinerRead:
    """What the connection read found for one question."""

    #: rooted-star cost of the best episode: sum of its distances to every
    #: terminal. ``inf`` when no episode reaches all of them.
    cost: float = UNREACHABLE
    #: the episode achieving it
    root: str = ""
    #: cost per episode that reaches EVERY terminal (the conjunction)
    per_episode: dict[str, float] = field(default_factory=dict)
    n_terminals: int = 0
    #: terminals that reached no episode at all -- the strongest non-support
    #: shape, and distinct from "they exist but do not meet"
    dead_terminals: int = 0

    @property
    def supported(self) -> bool:
        return bool(self.per_episode) and math.isfinite(self.cost)

    def as_dict(self) -> dict:
        return {
            "cost": None if not math.isfinite(self.cost) else round(self.cost, 4),
            "root": self.root,
            "n_reaching_all": len(self.per_episode),
            "n_terminals": self.n_terminals,
            "dead_terminals": self.dead_terminals,
        }


class SteinerMetric:
    """Shortest paths from a slot to every episode, under the degree metric.

    One instance per conversation. Distances are cached per source slot: a
    question reuses the sources of every other question that named the same
    slot, and the null-distribution sampling reuses them again, which is what
    keeps the calibration affordable.
    """

    def __init__(
        self,
        graph,
        is_hub: Callable[[str], bool],
        max_cost: float = 12.0,
        cache_size: int = 4096,
    ) -> None:
        self.graph = graph
        self.max_cost = max_cost
        self._cache: dict[str, dict[str, float]] = {}
        self._cache_size = cache_size

        # adjacency, precomputed once: slot -> episodes and episode -> slots.
        # Non-bridgeable slots are dropped from the episode->slot direction, so
        # a hub can still be a terminal (you may ask about a common thing) but
        # can never be a step on a path between two other things.
        self.eps_of_slot: dict[str, list[str]] = {}
        self.slots_of_ep: dict[str, list[str]] = {}
        self.slot_cost: dict[str, float] = {}
        for hid, he in graph.H.items():
            vid = he.vertex_id
            other = graph.H[graph.alpha[hid]].vertex_id
            if graph.vertices[vid].meta.get("kind") != KIND_EPISODE:
                continue
            self.eps_of_slot.setdefault(other, []).append(vid)
            if not is_hub(other):
                self.slots_of_ep.setdefault(vid, []).append(other)
        for slot, eps in self.eps_of_slot.items():
            # entering a slot mentioned by many episodes tells you little, so it
            # costs more. Same shape as the scorer's damping, used as a metric.
            self.slot_cost[slot] = 1.0 + math.log(max(len(eps), 1))

    # ------------------------------------------------------------ distances --
    def distances_from(self, slot_vid: str) -> dict[str, float]:
        """Cost of the cheapest path from ``slot_vid`` to each episode.

        Dijkstra rather than BFS because the metric is weighted, and bounded by
        ``max_cost`` because a path costing more than that is not a connection
        anybody would call one -- the bound is what keeps this linear in the
        neighbourhood that matters instead of in the whole conversation.
        """
        hit = self._cache.get(slot_vid)
        if hit is not None:
            return hit

        dist_ep: dict[str, float] = {}
        seen_slot: dict[str, float] = {slot_vid: 0.0}
        # (cost, slot): the frontier lives on the slot side, because episodes
        # are what we are measuring TO and never routed through twice.
        frontier: list[tuple[float, str]] = [(0.0, slot_vid)]
        while frontier:
            cost, slot = heapq.heappop(frontier)
            if cost > seen_slot.get(slot, UNREACHABLE):
                continue
            if cost > self.max_cost:
                break
            for ep in self.eps_of_slot.get(slot, ()):
                # arriving at an episode costs one step
                d = cost + 1.0
                if d < dist_ep.get(ep, UNREACHABLE):
                    dist_ep[ep] = d
                    for nxt in self.slots_of_ep.get(ep, ()):
                        nd = d + self.slot_cost.get(nxt, 1.0)
                        if nd <= self.max_cost and nd < seen_slot.get(
                            nxt, UNREACHABLE
                        ):
                            seen_slot[nxt] = nd
                            heapq.heappush(frontier, (nd, nxt))

        if len(self._cache) >= self._cache_size:
            self._cache.clear()  # bounded, and cheap to refill within a conv
        self._cache[slot_vid] = dist_ep
        return dist_ep

    # ------------------------------------------------------------- the read --
    def rooted_star(self, terminals: Sequence[str]) -> SteinerRead:
        """Sum of distances to all terminals, per episode that reaches them all.

        The rooted-star relaxation of group Steiner: it is a ``k``-approximation
        of the optimal tree, and unlike the optimal tree it is a handful of
        Dijkstras. Named for what it is, so nobody later reports it as an exact
        Steiner cost.
        """
        read = SteinerRead(n_terminals=len(terminals))
        if not terminals:
            return read

        totals: Optional[dict[str, float]] = None
        for t in terminals:
            d = self.distances_from(t)
            if not d:
                read.dead_terminals += 1
                return read  # a terminal reaching nothing kills the conjunction
            if totals is None:
                totals = dict(d)
                continue
            # intersect: the AND that the additive channels cannot express
            totals = {
                ep: acc + d[ep] for ep, acc in totals.items() if ep in d
            }
            if not totals:
                break

        if not totals:
            return read
        read.per_episode = totals
        read.root = min(totals, key=totals.get)
        read.cost = totals[read.root]
        return read


# --------------------------------------------------------------------------- #
# Label-free calibration of the abstention threshold                           #
# --------------------------------------------------------------------------- #


@dataclass
class NullDistribution:
    """How far apart *k* arbitrary slots usually are, in this memory.

    The reference against which a question's Steiner cost is judged. Built by
    sampling random non-hub slot tuples -- no question text, no answer, no
    evidence, no category -- so the abstention threshold is a property of the
    corpus rather than a number fitted to an answer key.
    """

    by_k: dict[int, float] = field(default_factory=dict)
    quantile: float = 0.95
    n_samples: int = 0
    pool_size: int = 0
    unreachable_frac: dict[int, float] = field(default_factory=dict)

    def threshold(self, k: int) -> float:
        """Cost above which a ``k``-terminal question is judged unsupported."""
        if not self.by_k:
            return UNREACHABLE
        if k in self.by_k:
            return self.by_k[k]
        # a k we did not sample: use the nearest one we did, which is a better
        # guess than a global constant because the cost grows with k
        nearest = min(self.by_k, key=lambda kk: abs(kk - k))
        return self.by_k[nearest] * (k / max(nearest, 1))

    def as_dict(self) -> dict:
        return {
            "quantile": self.quantile,
            "n_samples": self.n_samples,
            "pool_size": self.pool_size,
            "threshold_by_k": {k: round(v, 3) for k, v in sorted(self.by_k.items())},
            "unreachable_frac_by_k": {
                k: round(v, 3) for k, v in sorted(self.unreachable_frac.items())
            },
        }


def calibrate_null(
    metric: SteinerMetric,
    is_hub: Callable[[str], bool],
    ks: Iterable[int] = (2, 3, 4),
    n_samples: int = 120,
    pool: int = 64,
    quantile: float = 0.95,
    seed: int = 1234,
) -> NullDistribution:
    """Sample random slot tuples and read off the upper tail of their cost.

    ``pool`` is deliberately small and drawn once: every sampled tuple is built
    from the same few dozen slots, so the metric's per-source cache is warm
    after the first handful of Dijkstras and the whole calibration costs about
    as much as ``pool`` shortest-path computations rather than
    ``n_samples * k``.
    """
    rng = np.random.default_rng(seed)
    candidates = [s for s in metric.eps_of_slot if not is_hub(s)]
    null = NullDistribution(quantile=quantile, n_samples=n_samples)
    if len(candidates) < max(ks, default=2) + 1:
        return null

    take = min(pool, len(candidates))
    sources = [candidates[i] for i in rng.choice(len(candidates), take, replace=False)]
    null.pool_size = take

    for k in ks:
        if k > take:
            continue
        costs: list[float] = []
        dead = 0
        for _ in range(n_samples):
            idx = rng.choice(take, size=k, replace=False)
            read = metric.rooted_star([sources[i] for i in idx])
            if read.supported:
                costs.append(read.cost)
            else:
                dead += 1
        null.unreachable_frac[k] = dead / float(n_samples)
        if costs:
            null.by_k[k] = float(np.quantile(costs, quantile))
    return null
