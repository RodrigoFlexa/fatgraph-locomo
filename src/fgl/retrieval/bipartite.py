"""Degree-aware retrieval over the bipartite turn/entity graph -- L1.

``FaceRetriever`` (the G1-G10 mechanism) treats every anchor the same way:
rank half-edges by cosine, ``walk_face`` from the winners. That is the right
move on a triples graph, where a "fact" and a "turn" are the same kind of
thing. On a bipartite graph they are not, and neither is every entity vertex
comparable to every other one -- so this retriever's central move is
classifying a *linked* entity vertex by its degree before deciding what to
do with it, instead of treating every anchor uniformly:

* **degree 1** -- the entity was mentioned exactly once. Its one incident
  turn is the direct hit; nothing to rank.
* **2 <= degree < bridge_max_degree** -- a real, specific entity ("sunset",
  "pottery class"). Its whole sigma-orbit (every turn that mentions it) is
  small enough to enumerate and IS the multi-hop answer for the dominant
  LoCoMo pattern, where a question asks about everything one person said
  about a topic across sessions.
* **degree >= bridge_max_degree** -- a hub (a common activity mentioned
  constantly, or a name that slipped past the speaker filter). Never
  enumerated into the walk -- that is exactly the mistake that made the
  triples graph a star. Used only as a FILTER: a candidate turn found some
  other way gets a small bonus if it also touches the hub, the same
  distinction retrieval.sigma_skip_hub_degree already draws for sigma
  expansion on the triples graph.

The two-entity bridge (the concrete case: "what did Caroline AND Melanie
both paint?") is found by literally intersecting neighbourhoods: take the
candidate turns reached from each linked entity, look at THEIR other
incident entities, and any entity appearing on both sides is a bridge. This
is two sigma-lookups and a set intersection -- no LLM call, no embedding
similarity needed to find it, because incidence itself is the signal.

Dense retrieval is not a fallback bolted on at the end; it runs on equal
footing with the entity mechanism from the start, over turn-vertex
embeddings. NER misses adjectives, feelings, and anything category 3
(open-domain) asks about ("would Caroline consider..."), and this is the
same recall floor B3/G1 already rely on for those cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from fgl.config import Config
from fgl.core import Face, FatGraph, HalfEdge
from fgl.retrieval.embeddings import Embedder, VectorIndex, build_index
from fgl.retrieval.faces import (
    RetrievalResult,
    RetrievedFact,
    _ngrams,
    _unit,
)
from fgl.memory.entities import normalize_name

#: RetrievedFact.source values specific to this retriever.
SOURCE_BP_ENTITY = "bp_entity"  # turn incident to a linked, non-hub entity
SOURCE_BP_BRIDGE = "bp_bridge"  # turn incident to an entity found by 2-hop intersection
SOURCE_BP_DENSE = "bp_dense"  # cosine backstop over turn embeddings

BIPARTITE_SOURCES = (SOURCE_BP_ENTITY, SOURCE_BP_BRIDGE, SOURCE_BP_DENSE)


@dataclass
class _Candidate:
    turn_vid: str
    score: float = 0.0
    source: str = SOURCE_BP_DENSE  # highest-priority source wins for display
    via_vertex: str = ""
    via_entity: str = ""
    hub_hits: int = 0


class _EntityLinker:
    """Same n-gram-then-embedding approach as
    :class:`fgl.retrieval.faces.QuestionLinker`, scoped to ENTITY vertices
    only (``meta.get("kind") != "turn"``). Reusing ``QuestionLinker``
    unmodified was considered and rejected: it indexes every vertex with an
    embedding, and turn vertices have one here (needed for the dense
    backstop above), so it would happily "link" the question straight to a
    turn -- which downstream degree-classification cannot make sense of, a
    turn's degree means something different from an entity's.
    """

    def __init__(self, graph: FatGraph, embedder: Embedder, threshold: float):
        self.graph = graph
        self.embedder = embedder
        self.threshold = threshold
        self._by_surface: dict[str, str] = {}
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for vid, vx in graph.vertices.items():
            if vx.meta.get("kind") == "turn":
                continue
            for surface in (vx.name, *vx.aliases):
                key = normalize_name(surface)
                if key:
                    self._by_surface.setdefault(key, vid)
            if vx.embedding is not None:
                ids.append(vid)
                rows.append(_unit(vx.embedding))
        self._ids = ids
        self._matrix = np.vstack(rows) if rows else None

    def link(self, question: str, max_entities: int = 6) -> list[tuple[str, float]]:
        found: dict[str, float] = {}
        for gram in _ngrams(normalize_name(question), 3):
            vid = self._by_surface.get(gram)
            if vid is not None:
                found[vid] = max(found.get(vid, 0.0), 1.0 + 0.01 * gram.count(" "))
        if len(found) < max_entities and self._matrix is not None:
            sims = self._matrix @ _unit(self.embedder.encode_one(question))
            for i in np.argsort(-sims)[: max_entities * 4]:
                score = float(sims[int(i)])
                if score < self.threshold:
                    break
                found.setdefault(self._ids[int(i)], score)
        ranked = sorted(found.items(), key=lambda kv: -kv[1])
        return ranked[:max_entities]


class BipartiteRetriever:
    """Same public contract as :class:`fgl.retrieval.faces.FaceRetriever`:
    ``retrieve(question) -> RetrievalResult``, ``top_edges``,
    ``turn_ids_for_edges`` -- so :class:`fgl.pipeline.Runner` can dispatch to
    either with one ``if`` on ``cfg.retrieval.mode``.
    """

    def __init__(
        self,
        graph: FatGraph,
        embedder: Embedder,
        cfg: Config,
        date_by_session: dict[str, str] | None = None,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.cfg = cfg
        self.dates = date_by_session or {}

        self.turn_index: VectorIndex = build_index(cfg.index, embedder.dim)
        turn_ids: list[str] = []
        turn_vecs: list[np.ndarray] = []
        for vid, vx in graph.vertices.items():
            if vx.meta.get("kind") == "turn" and vx.embedding is not None:
                turn_ids.append(vid)
                turn_vecs.append(vx.embedding)
        if turn_ids:
            self.turn_index.add(turn_ids, np.vstack(turn_vecs))

        self.linker = _EntityLinker(
            graph, embedder, threshold=cfg.retrieval.coverage_entity_threshold
        )
        self._face_by_half_edge: dict[str, Face] | None = None
        #: every speaker that appears on a turn vertex. The speaker is still
        #: not a vertex -- this reads ``meta["speaker"]``, a turn attribute --
        #: so nothing here changes the topology or can recreate the hub the
        #: speaker exclusion exists to prevent.
        self.speakers: list[str] = sorted(
            {
                vx.meta["speaker"]
                for vx in graph.vertices.values()
                if vx.meta.get("kind") == "turn" and vx.meta.get("speaker")
            }
        )

    # -------------------------------------------------------- face lookup ----
    def face_of(self, half_edge_id: str) -> Face:
        """Memoised, same rationale as ``FaceRetriever.face_of``: the graph
        is read-only during QA, so caching the decomposition once is sound.
        Used only for diagnostics here (see module docstring): retrieval
        itself never walks phi, it reads sigma-orbits directly.
        """
        cache = self._face_by_half_edge
        if cache is None:
            cache = {}
            for face in self.graph.faces():
                for h in face.half_edges:
                    cache[h] = face
            self._face_by_half_edge = cache
        face = cache.get(half_edge_id)
        return face if face is not None else self.graph.face_of(half_edge_id)

    # ------------------------------------------------------------------ api --
    def retrieve(self, question: str) -> RetrievalResult:
        bp = self.cfg.bipartite
        r = self.cfg.retrieval
        qvec = self.embedder.encode_one(question)

        linked = self.linker.link(question, max_entities=6)
        result = RetrievalResult(
            all_anchor_ranking=[(vid, s) for vid, s in linked],
        )
        result.question_vertices = [vid for vid, _ in linked]
        result.question_entities = [self.graph.vertices[vid].name for vid, _ in linked]

        leaves_and_bridges: list[tuple[str, float, int]] = []
        hubs: list[tuple[str, float]] = []
        for vid, score in linked:
            deg = self.graph.degree(vid)
            if deg == 0:
                continue
            if deg >= bp.bridge_max_degree:
                hubs.append((vid, score))
            else:
                leaves_and_bridges.append((vid, score, deg))
        result.sigma_vertices = [vid for vid, _, _ in leaves_and_bridges]

        candidates: dict[str, _Candidate] = {}

        def touch(turn_vid: str, add: float, source: str, via_vertex: str = "",
                  via_entity: str = "", priority: int = 0) -> _Candidate:
            c = candidates.get(turn_vid)
            if c is None:
                c = _Candidate(turn_vid=turn_vid)
                candidates[turn_vid] = c
            c.score += add
            # higher-priority source wins the displayed label; priority order
            # is bridge > entity > dense, matching how informative each is
            if priority >= _SOURCE_PRIORITY.get(c.source, -1):
                c.source = source
                if via_vertex:
                    c.via_vertex = via_vertex
                    c.via_entity = via_entity
            return c

        # 1. dense backstop over turn embeddings -- a full participant, not
        # a last resort (see module docstring). Breadth is tied to
        # max_facts_in_prompt, not to a constant: the prompt budget decides how
        # many units can be shown, so generating fewer candidates than that
        # leaves budget unspendable however good the ranking is (measured: with
        # the cap at 80 and breadth at 32, L1 filled 1001 of its 2000 tokens).
        breadth = max(r.top_m_anchors * 4, r.max_facts_in_prompt, 24)
        for turn_vid, score in self.turn_index.search(qvec, breadth):
            touch(turn_vid, bp.dense_weight * score, SOURCE_BP_DENSE, priority=0)

        # 2. direct entity hits: every turn in a linked non-hub entity's
        # whole sigma-orbit (small by construction, degree < bridge_max_degree)
        entity_turns: dict[str, set[str]] = {}  # entity_vid -> turn_vids touched
        for vid, score, deg in leaves_and_bridges:
            name = self.graph.vertices[vid].name
            touched: set[str] = set()
            for h in self.graph.sigma.get(vid, ()):
                turn_vid = self.graph.H[self.graph.alpha[h]].vertex_id
                touched.add(turn_vid)
                touch(
                    turn_vid, bp.entity_weight * score, SOURCE_BP_ENTITY,
                    via_vertex=vid, via_entity=name, priority=1,
                )
            entity_turns[vid] = touched

        # 3. hub filter: a turn already found some other way gets a bonus
        # for ALSO touching a linked hub -- never enumerated on its own.
        if hubs and candidates:
            hub_turn_sets: dict[str, set[str]] = {}
            for vid, _ in hubs:
                hub_turn_sets[vid] = {
                    self.graph.H[self.graph.alpha[h]].vertex_id
                    for h in self.graph.sigma.get(vid, ())
                }
            for turn_vid, c in candidates.items():
                for vid, hub_turns in hub_turn_sets.items():
                    if turn_vid in hub_turns:
                        c.score += bp.hub_weight
                        c.hub_hits += 1

        # 4. the bridge: two linked entities whose candidate turns share
        # ANOTHER entity between them. For each non-hub candidate turn found
        # via entity A, look at its OTHER incident entities (bounded scan);
        # any that also appears as an "other entity" of a turn found via a
        # DIFFERENT linked entity B is a bridge -- the "sunset" between
        # Caroline's and Melanie's independent painting turns.
        if len(entity_turns) >= 2:
            other_entities: dict[str, dict[str, set[str]]] = {}
            # entity_vid -> {other_entity_vid -> {turn_vids that connect them}}
            for vid, touched in entity_turns.items():
                mapping: dict[str, set[str]] = {}
                for turn_vid in list(touched)[: bp.max_bridge_scan]:
                    for h2 in self.graph.sigma.get(turn_vid, ())[: bp.max_bridge_scan]:
                        other_vid = self.graph.H[self.graph.alpha[h2]].vertex_id
                        if other_vid == vid or self.graph.degree(other_vid) >= bp.bridge_max_degree:
                            continue
                        mapping.setdefault(other_vid, set()).add(turn_vid)
                other_entities[vid] = mapping

            seen_pairs: set[frozenset] = set()
            ent_ids = list(other_entities)
            for i in range(len(ent_ids)):
                for j in range(i + 1, len(ent_ids)):
                    a, b = ent_ids[i], ent_ids[j]
                    common = set(other_entities[a]) & set(other_entities[b])
                    for bridge_vid in common:
                        pair = frozenset((a, b, bridge_vid))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        bridge_name = self.graph.vertices[bridge_vid].name
                        for h in self.graph.sigma.get(bridge_vid, ()):
                            turn_vid = self.graph.H[self.graph.alpha[h]].vertex_id
                            touch(
                                turn_vid, bp.bridge_weight, SOURCE_BP_BRIDGE,
                                via_vertex=bridge_vid, via_entity=bridge_name,
                                priority=2,
                            )

        # 5. speaker partition: when the question names exactly ONE speaker,
        # the other one's turns are almost never the evidence, so spending
        # context slots on them is pure loss. Measured on this condition's own
        # predictions: 98.5-99.7% of questions name exactly one speaker, the
        # evidence turn is that speaker's in 96-100% of cases, and 24% of
        # every context was going to the other one.
        if bp.speaker_partition and candidates:
            self._partition_by_speaker(question, candidates)

        # 6. speaker-set boost: when the question names BOTH speakers
        # explicitly, nudge the ranking so at least one turn from each
        # survives truncation (measured minority pattern -- see
        # BipartiteConfig.speaker_filter docstring). Disjoint from the
        # partition above by construction: that one only fires on exactly one
        # named speaker, this one only on two.
        if bp.speaker_filter and candidates:
            self._boost_speaker_coverage(question, candidates)

        if not candidates:
            return result  # empty facts -> Answerer abstains, no change needed there

        ranked = sorted(candidates.values(), key=lambda c: -c.score)

        used = 0
        budget = r.budget_tokens
        for c in ranked:
            if len(result.facts) >= r.max_facts_in_prompt or used >= budget:
                break
            # any half-edge at this turn vertex carries the turn's text; take
            # the first (all identical, see ingest_bipartite.py)
            halves = self.graph.sigma.get(c.turn_vid, ())
            if not halves:
                continue
            he = self.graph.H[halves[0]]
            cost = self.graph._token_counter(he.text)  # noqa: SLF001
            if used and used + cost > budget:
                continue
            used += cost
            result.facts.append(self._make_fact(he, c))

        result.tokens_used = used
        result.faces = sorted({f.face_id for f in result.facts})
        return result

    # ------------------------------------------------------------ internals --
    def _named_speakers(self, question: str) -> list[str]:
        """Speakers the question names, by whole-word match on the first name.

        Whole word, not substring: "Sam" must not fire on "same", and the
        normalised question is already word-split, so the check is exact
        rather than a prefix heuristic.
        """
        words = set(normalize_name(question).split())
        return [
            spk for spk in self.speakers
            if (normalize_name(spk).split() or [""])[0] in words
        ]

    def _partition_by_speaker(
        self, question: str, candidates: dict[str, _Candidate]
    ) -> None:
        """Drop candidates spoken by someone the question did not name.

        Mutates ``candidates`` in place. Refuses to fire when it would leave
        fewer than ``speaker_partition_min`` candidates: on the few questions
        where the named person is not the one who said it, an empty context is
        a forced abstention, which is strictly worse than a ranked miss.
        """
        named = self._named_speakers(question)
        if len(named) != 1:
            return
        keep = {
            vid for vid in candidates
            if self.graph.vertices[vid].meta.get("speaker") == named[0]
        }
        if len(keep) < self.cfg.bipartite.speaker_partition_min:
            return
        for vid in [v for v in candidates if v not in keep]:
            del candidates[vid]

    def _boost_speaker_coverage(self, question: str, candidates: dict[str, _Candidate]) -> None:
        named = set(self._named_speakers(question))
        if len(named) < 2:
            return
        # at least one candidate per named speaker gets a boost proportional
        # to the current best score, so it is competitive with -- not
        # necessarily above -- the rest of the ranking
        if not candidates:
            return
        best = max(c.score for c in candidates.values())
        for spk in named:
            per_speaker = [
                c for vid, c in candidates.items()
                if self.graph.vertices[vid].meta.get("speaker") == spk
            ]
            if not per_speaker:
                continue
            top = max(per_speaker, key=lambda c: c.score)
            if top.score < best:
                top.score = best + 0.01

    def _make_fact(self, he: HalfEdge, c: _Candidate) -> RetrievedFact:
        face_id = (
            f"{c.source}:{c.via_vertex}" if c.source != SOURCE_BP_DENSE else "bp_dense"
        )
        return RetrievedFact(
            edge_id=he.edge_id,
            text=he.text,
            timestamp=he.timestamp,
            date_raw=self.dates.get(he.session_id, he.timestamp),
            session_id=he.session_id,
            turn_ids=list(he.turn_ids),
            state=he.state,
            level=he.level,
            anchor_rank=0,
            anchor_score=c.score,
            face_id=face_id,
            position_in_face=0,
            source=c.source,
            via_vertex=c.via_vertex,
            via_entity=c.via_entity,
        )

    def top_edges(self, question: str, k: int) -> list[str]:
        """Top-k *edges* by turn relevance -- used for the recall@k metric.

        An "edge" in this graph is one (turn, entity) incidence, not a fact
        with independent identity the way a triples-graph edge is. For
        recall@k comparability with every other condition (which counts
        edges, i.e. facts), this returns up to k incidences per ranked turn,
        walking turns in the same order ``retrieve`` would score them by
        dense similarity alone -- the metric every other condition's
        top_edges is defined against.
        """
        qvec = self.embedder.encode_one(question)
        out: list[str] = []
        for turn_vid, _ in self.turn_index.search(qvec, max(k, 8)):
            for h in self.graph.sigma.get(turn_vid, ()):
                eid = self.graph.H[h].edge_id
                if eid not in out:
                    out.append(eid)
                if len(out) >= k:
                    break
            if len(out) >= k:
                break
        return out

    def turn_ids_for_edges(self, edge_ids: Sequence[str]) -> list[str]:
        out: list[str] = []
        for e in edge_ids:
            for t in self.graph.get_edge_attr(e, "turn_ids"):
                if t not in out:
                    out.append(t)
        return out


_SOURCE_PRIORITY = {SOURCE_BP_DENSE: 0, SOURCE_BP_ENTITY: 1, SOURCE_BP_BRIDGE: 2}
