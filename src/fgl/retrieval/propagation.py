"""Condition L3 -- the same memory, read by propagation instead of by one hop.

The observation this starts from
--------------------------------
L2's structural score is::

    score(episode) = SUM over query slots incident to it of  w_kind / (1+log deg)^d

That is not a heuristic that happens to resemble a graph algorithm. It *is* one
iteration of a random walk with restart: the query's slots are the
personalisation vector, the incidence is the transition, and the degree damping
is a hand-rolled version of the degree normalisation a proper transition matrix
would apply. Written that way, the limitation is obvious -- **the walk stops
after one step**, so an episode can only ever be found by a slot the question
itself named.

Measured on L1 and reproduced in the L2 error analysis, that is precisely where
multi-hop fails. A multi-hop question asks about something reached *through*
something else, and on a bipartite episode-slot graph "through" has an exact
meaning::

    hop 1   slot_q -> episode                 the episode says what was asked
    hop 2   slot_q -> episode -> slot -> episode'
                                             an episode that shares a slot with
                                             an episode the question named

Hop 2 IS the join. And the closed case of hop 2 -- coming back to an episode
already reached, by a *different* slot -- is the 4-cycle
``e1 - s_a - e2 - s_b - e1``, which is the face of a quadrangular embedding of a
bipartite graph. So the walk of length two is the soft, weighted version of the
object the ribbon-graph theory points at, and it needs no rotation, no genus
minimisation and no embedding to compute. (``fgl hop-profile`` measures how much
evidence actually lives at hop 2 before any of this is run.)

Three things make it work rather than smear
-------------------------------------------
**1. A hub is a filter, never a bridge.** The failure mode of every walk on a
graph with hubs: mass enters the ``actor`` vertex, which is incident to half the
episodes, and comes out spread evenly over the corpus. The calibrated per-kind
hub cut-off (:mod:`fgl.memory.calibration`) already decides which vertices those
are, and here it gets the job it was always better suited to -- a hub may
*receive* mass at hop 1, where it acts as the filter L2 uses it as, and may
never *relay* it. One rule, stated once in :meth:`SlotRetriever.is_hub`, used by
the scorer, this walk, and the Steiner metric in L4.

**2. The walk is non-backtracking.** A plain walk's second hop is dominated by
mass that goes ``slot -> episode -> same slot`` and comes straight back: it
re-scores the seed and calls it a join. Tracking flow on *directed half-edges*
instead of on vertices and subtracting each edge's own incoming contribution
gives the Hashimoto non-backtracking operator, for which the bipartite graph's
half-edge structure -- which this codebase already has as ``alpha`` -- is the
natural data structure. It costs one extra ``bincount`` per hop.

**3. Reduction to L2 is exact, not approximate.** ``hops=1`` with
``normalization="none"`` reproduces condition L2 vertex for vertex and score for
score; ``tests/test_propagation.py`` asserts it on real graphs. So the sweep
over ``propagation.hops`` is a curve whose leftmost point IS the published L2
number, and the delta cannot be attributed to anything else -- L3 borrows L2's
graphs byte for byte (``paths.graphs_condition``) and inherits every other stage
of the retriever by subclassing rather than by copying.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from fgl.memory.slots import KIND_CONCEPT, KIND_EPISODE
from fgl.retrieval.slots import SlotRetriever

#: Transition weightings, i.e. what a step of the walk divides by.
#:
#: ``none``  no normalisation. An episode accumulates the raw weight of every
#:           incident seeded slot -- which is exactly L2's sum, and the reason
#:           this option exists is to make the reduction exact.
#: ``rw``    row-stochastic (``D^-1 A``): mass leaving a vertex is split among
#:           its incidences. The literal random walk.
#: ``sym``   symmetric (``D^-1/2 A D^-1/2``): the spectral normalisation. It
#:           damps on BOTH sides, so an episode that mentions forty things is
#:           discounted for it the same way a slot on forty episodes is --
#:           which the ad-hoc ``1/(1+log deg)`` never did, because it only ever
#:           looked at the slot side.
NORMALIZATIONS = ("none", "rw", "sym")


@dataclass
class _Bipartite:
    """The episode-slot incidence as flat arrays, built once per conversation.

    Deliberately not scipy: the whole operator is two ``np.bincount`` calls per
    direction, the dependency list stays at numpy, and the half-edge layout
    below is what makes the non-backtracking correction a subtraction rather
    than a special case.
    """

    ep_ids: list[str]
    slot_ids: list[str]
    #: one entry per (episode, slot) incidence
    ep_of: np.ndarray          # int index into ep_ids
    slot_of: np.ndarray        # int index into slot_ids
    w_se: np.ndarray           # transition weight, slot -> episode
    w_es: np.ndarray           # transition weight, episode -> slot
    #: may this slot relay mass onward? (a hub may receive, never bridge)
    slot_bridgeable: np.ndarray

    @property
    def n_ep(self) -> int:
        return len(self.ep_ids)

    @property
    def n_slot(self) -> int:
        return len(self.slot_ids)


def build_bipartite(
    retriever: SlotRetriever, normalization: str, bridge_hubs: bool
) -> _Bipartite:
    """Flatten the fatgraph's episode-slot incidences into arrays.

    Reads the graph, never the rotation: this condition makes no claim about
    sigma, so it must not accidentally depend on it. (L2's orbit enumeration
    for set questions still does, and is inherited unchanged.)
    """
    graph = retriever.graph
    ep_ids = [
        vid for vid, vx in graph.vertices.items()
        if vx.meta.get("kind") == KIND_EPISODE
    ]
    ep_pos = {vid: i for i, vid in enumerate(ep_ids)}
    slot_ids: list[str] = []
    slot_pos: dict[str, int] = {}

    ep_of: list[int] = []
    slot_of: list[int] = []
    for hid, he in graph.H.items():
        vid = he.vertex_id
        if vid not in ep_pos:
            continue
        other = graph.H[graph.alpha[hid]].vertex_id
        j = slot_pos.get(other)
        if j is None:
            j = len(slot_ids)
            slot_pos[other] = j
            slot_ids.append(other)
        ep_of.append(ep_pos[vid])
        slot_of.append(j)

    ep_arr = np.asarray(ep_of, dtype=np.int64)
    slot_arr = np.asarray(slot_of, dtype=np.int64)
    deg_ep = np.bincount(ep_arr, minlength=len(ep_ids)).astype(float)
    deg_slot = np.bincount(slot_arr, minlength=len(slot_ids)).astype(float)
    deg_ep[deg_ep == 0.0] = 1.0
    deg_slot[deg_slot == 0.0] = 1.0

    if normalization == "rw":
        w_se = 1.0 / deg_slot[slot_arr]
        w_es = 1.0 / deg_ep[ep_arr]
    elif normalization == "sym":
        inv = 1.0 / np.sqrt(deg_slot[slot_arr] * deg_ep[ep_arr])
        w_se = inv
        w_es = inv
    else:  # "none" -- the L2-equivalent operator
        w_se = np.ones(len(ep_arr))
        w_es = np.ones(len(ep_arr))

    bridgeable = np.ones(len(slot_ids), dtype=bool)
    if not bridge_hubs:
        for j, vid in enumerate(slot_ids):
            bridgeable[j] = not retriever.is_hub(vid)

    return _Bipartite(
        ep_ids=ep_ids, slot_ids=slot_ids, ep_of=ep_arr, slot_of=slot_arr,
        w_se=w_se, w_es=w_es, slot_bridgeable=bridgeable,
    )


def propagate(
    bp: _Bipartite,
    seed_slot: np.ndarray,
    seed_ep: Optional[np.ndarray],
    hops: int,
    decay: float,
    non_backtracking: bool,
) -> np.ndarray:
    """Score every episode by a truncated, non-backtracking bipartite walk.

    State lives on **directed half-edges**, not on vertices. ``f_se[i]`` is the
    flow crossing incidence ``i`` from its slot to its episode; ``f_es[i]`` the
    reverse. The non-backtracking rule is then a subtraction and nothing more::

        arrive[e]  =  sum of f_se over incidences at e
        f_es[i]    =  w_es[i] * (arrive[e_i] - f_se[i])
                                             ^^^^^^^^^ do not send back what
                                                       this edge just delivered

    which is the Hashimoto operator restricted to this bipartite graph. Without
    that subtraction the second hop is mostly the seed reflected off its own
    episodes -- it looks like a join and is not one.

    ``decay`` is the restart probability's complement: the mass a walk keeps for
    each further hop. It is why the sum converges and why hop 1 always outranks
    hop 2 at equal support, which is the right prior -- a direct mention is
    better evidence than a shared neighbour.
    """
    score = np.zeros(bp.n_ep) if seed_ep is None else np.array(seed_ep, dtype=float)
    if hops < 1 or not seed_slot.any():
        return score

    f_se = bp.w_se * seed_slot[bp.slot_of]
    for h in range(hops):
        arrive = np.bincount(bp.ep_of, weights=f_se, minlength=bp.n_ep)
        score += (decay ** h) * arrive
        if h + 1 >= hops:
            break
        incoming = arrive[bp.ep_of]
        if non_backtracking:
            incoming = incoming - f_se
        f_es = bp.w_es * incoming
        # a hub may receive but never relay
        f_es = f_es * bp.slot_bridgeable[bp.slot_of]
        back = np.bincount(bp.slot_of, weights=f_es, minlength=bp.n_slot)
        outgoing = back[bp.slot_of]
        if non_backtracking:
            outgoing = outgoing - f_es
        f_se = bp.w_se * outgoing
    return score


class PropagationRetriever(SlotRetriever):
    """L3. Identical to :class:`SlotRetriever` except for the structural read.

    Everything that made L2 work is inherited, not re-implemented: the question
    parser, the typed slot resolution with its paraphrase fallback, the
    multiplicative actor prior, the orbit enumeration for set questions, the
    corner test, and the turn-level emission with sibling propagation. The only
    override is :meth:`_structural_channels`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pg = self.cfg.propagation
        self._bp: Optional[_Bipartite] = None
        self._pg = pg
        #: best turn-level cosine per episode for the question in flight, set
        #: by `retrieve` and cleared after it. Per instance, never class-level:
        #: a shared mutable default would leak one conversation's dense scores
        #: into the next.
        self._last_dense_by_episode: dict[str, float] = {}
        #: episodes the last walk reached that no nameable linked slot touches.
        #: Reported so a run can prove the extra hops did something: if this is
        #: ~0 the condition has quietly become L2 whatever `hops` says.
        self._n_walk_only: int = 0

    @property
    def bipartite(self) -> _Bipartite:
        if self._bp is None:
            self._bp = build_bipartite(
                self, self._pg.normalization, self._pg.bridge_hubs
            )
        return self._bp

    # ---------------------------------------------------------- the override --
    def _structural_channels(self, linked, touch) -> None:
        pg = self._pg
        bp = self.bipartite
        seed = self.slot_seed(linked)
        if not seed:
            return

        slot_pos = {vid: i for i, vid in enumerate(bp.slot_ids)}
        seed_vec = np.zeros(bp.n_slot)
        for vid, w in seed.items():
            j = slot_pos.get(vid)
            if j is not None:
                seed_vec[j] += w

        seed_ep = None
        if pg.dense_seed > 0.0 and self._last_dense_by_episode:
            # The dense channel as a SECOND personalisation source rather than
            # a separate additive term. In L2 the two channels never spoke: a
            # turn that resembled the question could not lend that resemblance
            # to the episode next to it in the graph. Here it can, and it costs
            # nothing -- it is the same walk with a second seed.
            seed_ep = np.zeros(bp.n_ep)
            for i, vid in enumerate(bp.ep_ids):
                d = self._last_dense_by_episode.get(vid)
                if d:
                    seed_ep[i] = pg.dense_seed * d
        scores = propagate(
            bp, seed_vec, seed_ep, pg.hops, pg.decay, pg.non_backtracking
        )

        # attribution: label an episode by the strongest linked slot actually
        # incident to it, so `source` / `via_entity` keep meaning what they
        # meant in L2 even though the score no longer comes from one hop.
        kind_of = {vid: kind for kind, _k, vid in linked}
        best_via: dict[str, tuple[float, str, str]] = {}
        for vid, w in seed.items():
            if self.is_hub(vid):
                continue
            name = self.graph.vertices[vid].name
            for ep_vid in self._orbit_episodes(vid):
                cur = best_via.get(ep_vid)
                if cur is None or w > cur[0]:
                    best_via[ep_vid] = (w, vid, name)

        nz = np.nonzero(scores)[0]
        walk_only = 0
        for i in nz:
            ep_vid = bp.ep_ids[i]
            via = best_via.get(ep_vid)
            if via is None:
                # Reached only through the walk (or only through a hub): no
                # linked slot of a nameable kind touches it, so there is no
                # honest per-kind label and it gets none -- exactly what L2
                # does for a hub-only hit, which is what keeps the reduction
                # at hops=1 exact down to the `source` column.
                walk_only += 1
                touch(ep_vid, float(scores[i]), "")
            else:
                _w, vid, name = via
                touch(ep_vid, float(scores[i]), kind_of[vid], via=vid, label=name)
        self._n_walk_only = walk_only

    # ------------------------------------------------------------- plumbing --
    def retrieve(self, question: str):
        # `_structural_channels` runs before the dense scores exist in the base
        # class's local scope, so the dense seed is handed over on the instance.
        # Reset per question: a stale vector would silently seed the next walk.
        self._last_dense_by_episode = self._dense_by_episode(question)
        self._n_walk_only = 0
        try:
            result = super().retrieve(question)
            result.n_walk_only = self._n_walk_only
            return result
        finally:
            self._last_dense_by_episode = {}

    def _dense_by_episode(self, question: str) -> dict[str, float]:
        """Best turn-level cosine per episode -- only computed when it is used."""
        if self.cfg.propagation.dense_seed <= 0.0:
            return {}
        qvec = self.embedder.encode_one(question)
        out: dict[str, float] = {}
        for turn_id, score in self.turn_index.search(qvec, self._n_turns):
            ep_vid = self._episode_of_turn.get(turn_id)
            if ep_vid is not None:
                out[ep_vid] = max(out.get(ep_vid, 0.0), float(score))
        return out

    # ------------------------------------------------------------ reporting --
    def walk_stats(self) -> dict:
        """What the operator looks like on this graph -- for the run manifest.

        Reported rather than trusted: ``bridgeable`` is the share of slots the
        walk is allowed to relay through, and if that number is near zero the
        second hop is doing nothing and the condition has quietly become L2.
        """
        bp = self.bipartite
        return {
            "n_episodes": bp.n_ep,
            "n_slots": bp.n_slot,
            "n_incidences": int(len(bp.ep_of)),
            "bridgeable_slots": int(bp.slot_bridgeable.sum()),
            "bridgeable_frac": round(float(bp.slot_bridgeable.mean()), 4),
            "hops": self._pg.hops,
            "decay": self._pg.decay,
            "normalization": self._pg.normalization,
            "non_backtracking": self._pg.non_backtracking,
            "dense_seed": self._pg.dense_seed,
        }


def reduces_to_l2(cfg) -> bool:
    """Is this configuration the identity on L2's structural read?

    Used by the tests and printed by ``fgl config show`` consumers: a condition
    that claims to generalise L2 should be able to say, mechanically, when it
    stops being a generalisation and becomes a copy.
    """
    pg = cfg.propagation
    return pg.hops == 1 and pg.normalization == "none" and pg.dense_seed == 0.0
