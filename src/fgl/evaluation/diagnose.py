"""Where does the answer stop being reachable?

Every condition from G1 to G10 varied *retrieval policy* while assuming the
memory graph encodes the answer and that the answer is reachable in it.  That
assumption was never tested.  If the extraction dropped an evidence turn, or the
two facts a multi-hop question needs sit in different components, then no
ranking, no orbit, no face and no rotation can recover it -- and a fortnight of
policy experiments is measuring noise below a ceiling nobody looked at.

This module walks the chain of ceilings, in order, and reports where each one
bites::

    1. extraction   is every gold evidence turn carried by some extracted fact?
    2. component    are the facts carrying the evidence even connected?
    3. distance     how far apart are they in the graph?
    4. sigma        do they share a vertex, i.e. is a one-hop join possible?
    5. face         do they lie on a common face?
    6. ranking      where does the *hardest* evidence fact rank by cosine?

Each row is conditional on the previous one passing, so the first big drop is
the layer that actually costs the points.  Step 6 is the only one a retrieval
policy can address; a drop before it is an ingest problem wearing a retrieval
costume.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from fgl.core import FatGraph
from fgl.data.locomo import Conversation, Question

#: LoCoMo category ids that have annotated evidence worth chasing
EVIDENCE_CATEGORIES = (1, 2, 3, 4)


@dataclass
class QuestionTrace:
    """The waterfall for a single question."""

    question: str
    category: int
    gold: str
    evidence: list[str]
    #: evidence turns carried by at least one extracted fact
    covered: list[str] = field(default_factory=list)
    #: edge ids carrying the evidence, one group per evidence turn
    edges: list[str] = field(default_factory=list)
    same_component: Optional[bool] = None
    #: graph distance between the two furthest evidence-bearing facts
    distance: Optional[int] = None
    shares_vertex: Optional[bool] = None
    same_face: Optional[bool] = None
    #: cosine rank of the worst-ranked evidence fact (0-based), None if absent
    worst_rank: Optional[int] = None

    @property
    def fully_extracted(self) -> bool:
        return bool(self.evidence) and len(self.covered) == len(self.evidence)


class Diagnostician:
    """Answers 'is the answer even in there, and can anything reach it?'"""

    def __init__(self, graph: FatGraph, embedder, index=None) -> None:
        self.graph = graph
        self.embedder = embedder
        self.index = index
        self._turn_to_edges: dict[str, list[str]] = {}
        for eid in graph.edges():
            for t in graph.get_edge_attr(eid, "turn_ids") or ():
                self._turn_to_edges.setdefault(t, []).append(eid)
        self._adj: dict[str, set[str]] = {}
        for vid, halves in graph.sigma.items():
            self._adj[vid] = {graph.H[graph.alpha[h]].vertex_id for h in halves}
        self._face_of_edge: dict[str, set[str]] = {}
        for face in graph.faces():
            for e in face.edges:
                self._face_of_edge.setdefault(e, set()).add(face.id)
        self._components = graph._components()  # noqa: SLF001

    # ---------------------------------------------------------------- api ---
    def trace(self, q: Question) -> QuestionTrace:
        ev = [e for e in (q.evidence or []) if e]
        t = QuestionTrace(
            question=q.question, category=q.category, gold=q.answer, evidence=ev
        )
        t.covered = [e for e in ev if e in self._turn_to_edges]
        if not t.fully_extracted:
            return t  # ceiling 1: the memory never recorded it

        # one representative edge per evidence turn: the question needs them
        # *together*, so what matters is the relation between the groups
        groups = [self._turn_to_edges[e] for e in ev]
        t.edges = [g[0] for g in groups]
        if len(t.edges) < 2:
            # single-fact question: reachability is trivially satisfied
            t.same_component = True
            t.distance = 0
            t.shares_vertex = True
            t.same_face = True
            t.worst_rank = self._rank_of(t.edges, q)
            return t

        verts = [set(self.graph.edge_endpoints(e)) for e in t.edges]
        comps = {self._components[v] for vs in verts for v in vs}
        t.same_component = len(comps) == 1

        t.shares_vertex = any(
            verts[i] & verts[j]
            for i in range(len(verts))
            for j in range(i + 1, len(verts))
        )
        faces = [self._face_of_edge.get(e, set()) for e in t.edges]
        t.same_face = bool(set.intersection(*faces)) if all(faces) else False

        if t.same_component:
            t.distance = max(
                self._distance(verts[i], verts[j])
                for i in range(len(verts))
                for j in range(i + 1, len(verts))
            )
        t.worst_rank = self._rank_of(t.edges, q)
        return t

    # --------------------------------------------------------- internals ---
    def _distance(self, a: set[str], b: set[str], cap: int = 8) -> int:
        """Shortest vertex distance between two edges' endpoint sets."""
        if a & b:
            return 0
        seen, frontier, d = set(a), deque(a), 0
        while frontier and d < cap:
            d += 1
            for _ in range(len(frontier)):
                for nb in self._adj.get(frontier.popleft(), ()):
                    if nb in b:
                        return d
                    if nb not in seen:
                        seen.add(nb)
                        frontier.append(nb)
        return cap + 1

    def _rank_of(self, edges: Sequence[str], q: Question) -> Optional[int]:
        """Worst cosine rank among the evidence edges (0-based, None if absent).

        This is the only rung a retrieval policy can climb.  A rank of 12 says
        'raise k'; a rank of 900 says the query and the evidence do not look
        alike and no k will help.
        """
        if self.index is None:
            return None
        qvec = self.embedder.encode_one(q.prompt_question())
        hits = self.index.search(qvec, len(self.graph.H))
        order: dict[str, int] = {}
        for hid, _ in hits:
            eid = self.graph.H[hid].edge_id
            order.setdefault(eid, len(order))
        ranks = [order.get(e) for e in edges]
        return max((r for r in ranks if r is not None), default=None)


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #


def waterfall(traces: Sequence[QuestionTrace]) -> dict:
    """The chain of ceilings, each rung conditional on the previous."""
    if not traces:
        return {}
    n = len(traces)
    ev_turns = [e for t in traces for e in t.evidence]
    covered_turns = [e for t in traces for e in t.covered]

    extracted = [t for t in traces if t.fully_extracted]
    multi = [t for t in extracted if len(t.edges) >= 2]
    connected = [t for t in multi if t.same_component]
    shares = [t for t in connected if t.shares_vertex]
    faced = [t for t in connected if t.same_face]
    ranked = [t for t in extracted if t.worst_rank is not None]

    def pct(a, b):
        return round(len(a) / b, 4) if b else 0.0

    out = {
        "n_questions": n,
        "n_evidence_turns": len(ev_turns),
        # ceiling 1 -- nothing below this can be recovered by any policy
        "evidence_turns_extracted": round(
            len(covered_turns) / len(ev_turns), 4
        )
        if ev_turns
        else 0.0,
        "questions_fully_extracted": pct(extracted, n),
        # ceilings 2-5, over the multi-fact questions that survived ceiling 1
        "n_multi_fact": len(multi),
        "same_component": pct(connected, len(multi)),
        "shares_a_vertex": pct(shares, len(connected)),
        "on_a_common_face": pct(faced, len(connected)),
    }
    dists = [t.distance for t in connected if t.distance is not None]
    if dists:
        out["distance_median"] = float(np.median(dists))
        out["distance_hist"] = {
            str(d): int(sum(1 for x in dists if x == d)) for d in sorted(set(dists))
        }
    ranks = [t.worst_rank for t in ranked]
    if ranks:
        out["worst_evidence_rank_median"] = float(np.median(ranks))
        for k in (5, 10, 20, 50, 100):
            out[f"evidence_within_top_{k}"] = round(
                float(np.mean([r < k for r in ranks])), 4
            )
    return out


def by_category(traces: Sequence[QuestionTrace]) -> dict:
    from fgl.data.locomo import CATEGORY_NAMES

    out = {}
    for cat in sorted({t.category for t in traces}):
        items = [t for t in traces if t.category == cat]
        out[CATEGORY_NAMES.get(cat, str(cat))] = waterfall(items)
    return out


def failing_cases(
    traces: Sequence[QuestionTrace], limit: int = 5
) -> list[QuestionTrace]:
    """Concrete questions whose evidence the memory never recorded, or buried.

    One readable case is worth a page of percentages: it shows whether the
    extraction lost the turn, or kept it and put it out of reach.
    """
    lost = [t for t in traces if not t.fully_extracted]
    far = [
        t
        for t in traces
        if t.fully_extracted and (t.worst_rank or 0) > 50
    ]
    return (lost[: limit // 2 + 1] + far)[:limit]
