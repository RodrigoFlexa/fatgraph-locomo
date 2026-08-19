"""Condition L4 -- every read of the typed memory that earned its place.

What L4 is, precisely
---------------------
``UnifiedRetriever`` is ``PropagationRetriever`` (L3) plus one channel and one
replaced abstention. It is not a fourth memory model: L2, L3 and L4 build and
read the *same* typed episode-slot fatgraph. What changes across the three is
the question asked of it, and each question strictly contains the one before::

    L2   which episodes TOUCH the query's slots?          one hop
    L3   which episodes are REACHED from them?            a bounded walk
    L4   which episodes HOLD THEM TOGETHER?               a connection

The three are one design and not three retrievers because they share a single
structural rule -- **a hub is a filter, never a bridge** -- stated once in
:meth:`fgl.retrieval.slots.SlotRetriever.is_hub` and obeyed by the scorer, the
walk and the metric alike. Everything else L4 uses (typed slots, episodes as the
index unit, actor as a multiplicative partition, orbit enumeration for set
questions, turn-level emission with sibling propagation, corpus-derived
thresholds, multi-resolution time) is inherited rather than restated.

The two additions
-----------------
**A join channel with an AND in it** (:mod:`fgl.retrieval.steiner`). Every
channel up to here is a sum, so an episode scores well by matching one slot
hard, and nothing in the arithmetic can say "and also the other two". The
rooted-star group-Steiner score is a conjunction by construction: an episode
that cannot reach one terminal drops out of the intersection however close it is
to the rest. That is the shape multi-hop needs.

**An abstention with resolution.** The corner test is binary and, measured, a
losing trade on this benchmark: 20 of 446 adversarials caught for 38 false
positives on 1540 substantive questions -- roughly ``+0.004`` against
``-0.010`` in micro F1. It is replaced, not tuned. The Steiner cost is
continuous, and the threshold is the upper tail of the cost of *random* slot
tuples of the same size in this same memory, so the criterion reads "these
things sit further apart than 95% of arbitrary combinations here" and needs no
answer key to set.

That second point is also the answer to the regression in the last run.
Adversarial F1 fell 0.666 -> 0.608 from L1 to L2 while retrieval improved:
better recall means the context always contains something plausible, so the
answerer stops abstaining on its own. Abstention has to come back from the
structure, and a binary co-occurrence test does not have the resolution.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from fgl.memory.slots import KIND_ACTOR, SPECIFIC_KINDS
from fgl.retrieval.propagation import PropagationRetriever
from fgl.retrieval.steiner import (
    NullDistribution,
    SteinerMetric,
    SteinerRead,
    calibrate_null,
)


class UnifiedRetriever(PropagationRetriever):
    """L3 + the connection read. See the module docstring for the progression."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._metric: Optional[SteinerMetric] = None
        self._null: Optional[NullDistribution] = None
        #: the read for the question in flight, so `_join_channels` and the
        #: abstention hook share one computation instead of doing it twice
        self._steiner: SteinerRead = SteinerRead()
        self._steiner_done: bool = False
        self._terminals_used: list[str] = []

    # ------------------------------------------------------------ lazy parts --
    @property
    def metric(self) -> SteinerMetric:
        if self._metric is None:
            self._metric = SteinerMetric(
                self.graph, self.is_hub,
                max_cost=self.cfg.steiner.max_cost,
                cache_size=self.cfg.steiner.cache_size,
            )
        return self._metric

    @property
    def null(self) -> NullDistribution:
        """The null distribution of connection cost, built on first use.

        Lazily, because a condition running with ``steiner_abstain=false`` should
        not pay for a calibration it will not read -- and because building it
        warms the metric's cache, which the questions then reuse.
        """
        if self._null is None:
            st = self.cfg.steiner
            self._null = calibrate_null(
                self.metric, self.is_hub,
                ks=tuple(range(2, st.max_terminals + 1)),
                n_samples=st.null_samples,
                pool=st.null_pool,
                quantile=st.abstain_quantile,
                seed=self.cfg.seed,
            )
        return self._null

    # -------------------------------------------------------------- terminals --
    def _terminals(self, linked: Sequence[tuple[str, str, str]]) -> list[str]:
        """Which linked slots the connection must hold together.

        The *rarest* specific slots, capped at ``max_terminals``. Rarest for the
        same reason the set-question enumeration picks the rarest orbit: a
        common slot is satisfied by half the memory, so adding it to the
        conjunction costs a Dijkstra and constrains nothing. Actors are excluded
        -- the actor is a partition applied multiplicatively at the end, and
        making it a terminal would demand that the evidence be *near* the
        speaker vertex, which every episode of theirs trivially is.
        """
        st = self.cfg.steiner
        cands = [
            (self.graph.degree(vid), vid)
            for kind, _key, vid in linked
            if kind in SPECIFIC_KINDS and not self.is_hub(vid)
            and self.graph.degree(vid) > 0
        ]
        cands.sort()
        return [vid for _deg, vid in cands[: st.max_terminals]]

    # ------------------------------------------------------------- the read --
    def _ensure_steiner(self, linked) -> SteinerRead:
        """Compute the connection read once per question, memoised.

        Both hooks need it and the base class calls them in the wrong order for
        a naive "compute it in the channel" arrangement: ``_corner_support``
        runs *before* any scoring, ``_join_channels`` after. Rather than reorder
        the base class -- which would change what L2 does and break the claim
        that the conditions differ only in the stage they override -- whichever
        hook runs first pays for the read and the other reuses it.
        """
        if self._steiner_done:
            return self._steiner
        self._steiner_done = True
        st = self.cfg.steiner
        if not st.enabled:
            return self._steiner
        self._terminals_used = self._terminals(linked)
        # one terminal has nothing to connect to: a rooted star over a single
        # point is just distance, which the walk already scored better.
        if len(self._terminals_used) < 2:
            return self._steiner
        self._steiner = self.metric.rooted_star(self._terminals_used)
        return self._steiner

    # ------------------------------------------------------------ the channel --
    def _join_channels(self, linked, slots, result, touch) -> None:
        st = self.cfg.steiner
        read = self._ensure_steiner(linked)
        result.steiner_cost = None if not read.supported else round(read.cost, 4)
        result.steiner_root = read.root
        result.n_steiner_reaching = len(read.per_episode)
        if not st.enabled or not read.supported:
            return
        terminals = self._terminals_used

        best = read.cost
        for ep_vid, total in read.per_episode.items():
            if total <= 0.0:
                continue
            # scale-free by construction: the best root gets the full weight,
            # something twice as far gets half. No sharpness exponent to sweep,
            # which is the point -- this project has spent an iteration
            # removing numbers that were chosen by looking at the answers.
            touch(ep_vid, st.weight * (best / total), "", via=terminals[0],
                  label="steiner")

    # ----------------------------------------------------------- abstention --
    def _corner_support(self, linked, unlinked=()):
        """Replace the binary corner test with the continuous connection cost.

        Falls back to the inherited corner test whenever the Steiner read has
        nothing to say -- fewer than two terminals, or the channel switched off
        -- so turning ``steiner.abstain`` on never *removes* a signal, it only
        supersedes it where it has more resolution.

        Three shapes of non-support, reported distinctly in
        ``RetrievalResult.abstain_reason`` because they are different claims:

        ``missing_slot``       the question's vocabulary is absent (inherited);
        ``dead_terminal``      a terminal exists but reaches no episode within
                               the metric's bound;
        ``disconnected``       every terminal reaches something, but nothing
                               reaches all of them;
        ``far_apart``          they do meet, but further apart than
                               ``abstain_quantile`` of random tuples of the same
                               size in this memory.
        """
        st = self.cfg.steiner
        base_support, base_reason = super()._corner_support(linked, unlinked)
        if not st.enabled or not st.abstain:
            return base_support, base_reason
        if base_reason == "missing_slot":
            return base_support, base_reason  # nothing to connect; keep the claim

        read = self._ensure_steiner(linked)
        if read.n_terminals < 2:
            return base_support, base_reason
        if read.dead_terminals:
            return 0.0, "dead_terminal"
        if not read.supported:
            return 0.0, "disconnected"

        threshold = self.null.threshold(read.n_terminals)
        if math.isfinite(threshold) and read.cost > threshold:
            return 0.0, "far_apart"
        # support in [0, 1]: how comfortably inside the null this sits, so a
        # downstream reader can threshold differently without re-running
        support = 1.0 if not math.isfinite(threshold) else min(
            1.0, threshold / max(read.cost, 1e-9)
        )
        return support, ""

    # ------------------------------------------------------------- ordering --
    def retrieve(self, question: str):
        """Reset the per-question memo, then run the inherited pipeline.

        Without the reset the abstention of question *n+1* would read the
        connection cost of question *n* -- a bug that produces plausible
        numbers, which is the worst kind.
        """
        self._steiner = SteinerRead()
        self._steiner_done = False
        self._terminals_used = []
        return super().retrieve(question)

    # ------------------------------------------------------------ reporting --
    def connection_stats(self) -> dict:
        stats = self.walk_stats()
        st = self.cfg.steiner
        stats["steiner"] = {
            "enabled": st.enabled,
            "abstain": st.abstain,
            "weight": st.weight,
            "max_terminals": st.max_terminals,
            "max_cost": st.max_cost,
            "null": self._null.as_dict() if self._null is not None else None,
        }
        return stats
