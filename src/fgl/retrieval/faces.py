"""Face-based retrieval and question answering.

For each LoCoMo question:

1. top-``m`` anchor half-edges by cosine similarity to the question, with a
   boost for level-2 (consolidation) edges and a penalty for shadowed ones;
2. ``walk_face`` from every anchor, sharing a single global token budget;
3. optionally, **sigma expansion** of the top anchors (see below);
4. the answer prompt lists the facts **in face order**, each prefixed with the
   session date, deduplicated across faces;
5. short extractive answer, or the abstention string when the faces do not
   contain the information (or when the anchors are ``incongruente``).

Sigma expansion (``retrieval.sigma_expand``, condition G4)
----------------------------------------------------------
A multi-hop question needs two memories that **share an entity**.  In a
fatgraph that is, exactly, two half-edges in the same ``sigma``-orbit -- the
cyclic order around one vertex.  The second memory is by construction *not*
similar to the question (it only becomes relevant once the first hop names the
bridging entity), so the cosine anchor ranking cannot find it, which is why
single-hop can be strong while multi-hop is weak.

``phi = sigma o alpha`` *leaves* the vertex at every step, so a face does
contain the sigma-neighbours -- but only after a full lap around the surface,
usually past ``budget_tokens``.  Walking ``sigma`` directly from the anchor (and
from ``alpha(anchor)``, i.e. from both entities of the anchor memory) is the
join operation itself, costs no LLM call, and puts the bridge first instead of
last.

Everything here is inert unless ``retrieval.sigma_expand`` is true: with the
flag off, :meth:`FaceRetriever.retrieve` walks exactly the same code path as
before, so G1/G2/G3 keep producing byte-identical numbers.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from fgl.config import Config
from fgl.core import STATE_INCONGRUENT, Face, FatGraph, HalfEdge
from fgl.retrieval.embeddings import Embedder, VectorIndex, build_index
from fgl.llm import LLMClient
from fgl.data.locomo import ABSTAIN_ANSWER, Conversation, Question
from fgl.llm.prompts import SYSTEM_ANSWERER, SYSTEM_ANSWERER_OPEN, PromptLibrary


#: ``RetrievedFact.source`` values -- how a fact entered the context.
SOURCE_FACE = "face"  # phi-orbit walk from an anchor (the original path)
SOURCE_SIGMA = "sigma"  # sigma-orbit neighbour of an anchor (the multi-hop join)
SOURCE_COVERAGE = "coverage"  # face selected for covering the question's entities
SOURCE_GEODESIC = "geodesic"  # shortest path between two of those entities
SOURCE_FACE_UNIT = "face_unit"  # whole face containing a top-ranked fact (G10)

#: sources that only exist because of a multi-hop mechanism; used by the
#: counterfactual recalls and by the truncation guard.
JOIN_SOURCES = (SOURCE_SIGMA, SOURCE_COVERAGE, SOURCE_GEODESIC)


@dataclass
class RetrievedFact:
    edge_id: str
    text: str
    timestamp: str
    date_raw: str
    session_id: str
    turn_ids: list[str]
    state: str
    level: int
    anchor_rank: int
    anchor_score: float
    face_id: str
    position_in_face: int
    #: provenance, so a run can be audited fact by fact
    source: str = SOURCE_FACE
    #: for ``SOURCE_SIGMA``: the vertex whose orbit produced this fact
    via_vertex: str = ""
    #: display name of ``via_vertex`` -- the bridging entity
    via_entity: str = ""


@dataclass
class RetrievalResult:
    facts: list[RetrievedFact] = field(default_factory=list)
    anchors: list[tuple[str, float]] = field(default_factory=list)  # (half_edge, score)
    faces: list[str] = field(default_factory=list)
    all_anchor_ranking: list[tuple[str, float]] = field(default_factory=list)
    tokens_used: int = 0
    any_incongruent: bool = False

    # --- sigma expansion telemetry (all zero when the flag is off) ---------
    #: whether the retriever ran with ``retrieval.sigma_expand`` on
    sigma_expand: bool = False
    #: half-edges scored while scanning the orbits (cost of the expansion)
    sigma_scanned: int = 0
    #: candidates dropped because the face walk had already retrieved them.
    #: ``scanned`` high + ``facts`` low + this high means the orbit adds
    #: nothing *because phi already covered it* -- which happens exactly when
    #: the neighbouring vertices have degree 1 (phi then marches along the
    #: hub's own orbit). ``scanned`` near zero instead means the orbits are
    #: empty: the entities are not being shared, an ingest problem, not a
    #: retrieval one. The two call for opposite fixes, hence two counters.
    sigma_dup: int = 0
    #: candidates dropped for lack of budget
    sigma_over_budget: int = 0
    #: orbits skipped for belonging to a hub vertex (the stopword rule).
    #: Read together with ``n_sigma_facts``: high skips AND facts still arriving
    #: means the join found a real entity once the speakers were out of the way;
    #: high skips and no facts means the speaker WAS the only thing the two
    #: memories had in common, which is the hypothesis dying rather than the
    #: filter misfiring.
    sigma_hubs_skipped: int = 0

    # --- coverage retrieval telemetry (all zero when the flag is off) ------
    face_coverage: bool = False
    #: vertices the question was linked to, in link order
    question_vertices: list[str] = field(default_factory=list)
    #: their display names -- what the audit column actually shows
    question_entities: list[str] = field(default_factory=list)
    #: best coverage achieved by a single face, in [0, 1]
    coverage_best: float = 0.0
    #: candidate faces scored
    coverage_faces_scored: int = 0
    #: faces that covered 2+ of the question's entities: the real bridges
    coverage_faces_multi: int = 0
    #: the geodesic fallback ran (no face covered 2+)
    geodesic_used: bool = False
    #: hops of the retrieved shortest path (0 = none)
    geodesic_len: int = 0
    coverage_tokens: int = 0
    #: vertices whose orbit contributed at least one fact -- the bridges
    sigma_vertices: list[str] = field(default_factory=list)
    #: tokens spent on sigma facts (subset of ``tokens_used``)
    sigma_tokens: int = 0

    # --- face-as-a-unit telemetry (G10) -------------------------------------
    face_units: bool = False
    #: how many whole faces fitted in the budget. 1 means the method degenerated
    #: into "one big face", i.e. the genus search did not separate the memory
    #: and the unit is not a unit -- the check that G5's saturated coverage
    #: needed and did not have.
    face_units_used: int = 0
    #: best-member similarity of each retrieved face, in retrieval order
    face_unit_scores: list[float] = field(default_factory=list)
    #: Facts in the prompt that a k-NN over the same facts would NOT have
    #: returned: they do not resemble the question, they merely belong to the
    #: same narrative unit as something that does. This is the method's claim
    #: reduced to one number -- if it is ~0 the faces are adding nothing and
    #: G10 is B3 with extra steps, whatever the F1 says.
    corroborating_facts: int = 0

    # --- typed-slot telemetry (L2; zero/empty for every other condition) ----
    #: which slot keys the question linked, per kind -- the audit column that
    #: says whether a miss was a linking failure or a scoring one
    slot_channels: dict[str, list[str]] = field(default_factory=dict)
    #: fraction of the question's specific-slot episodes that also carry its
    #: actor: 1.0 when the corner test does not apply, 0.0 when it fired
    slot_support: float = 1.0
    #: why the corner test fired ("empty_corner"), empty when it did not
    abstain_reason: str = ""

    # --- support attestation (all inert when `support.enabled` is false) ---
    #: "direct" | "composed" | "conflict" | "absent" -- how the question's slot
    #: tuple projects into this memory. See fgl.retrieval.support.
    support_shape: str = ""
    #: the continuous support score and the cut it was compared against
    support_score: float = 0.0
    support_threshold: float = 0.0
    #: every feature that produced the score, for the audit column
    support_features: dict = field(default_factory=dict)
    #: episodes that justify the shape: one for `direct`, two for `composed`
    #: and `conflict`, none for `absent`
    support_witness: list[str] = field(default_factory=list)
    #: the question asks for a LIST, detected from its wording alone. Routes
    #: the Answerer to the enumerating prompt -- see fgl.memory.slots.
    set_question: bool = False
    #: how many episodes the enumerated orbit contributed, for auditing
    n_enumerated: int = 0

    # --- propagation / connection telemetry (L3, L4) ------------------------
    #: episodes the walk reached that no linked slot is incident to -- i.e. the
    #: ones that exist only because of hop 2+. If this is ~0 the walk collapsed
    #: to L2 whatever `propagation.hops` says, and the condition is not testing
    #: what it claims to.
    n_walk_only: int = 0
    #: rooted-star group-Steiner cost of the best episode: how tightly this
    #: memory holds the question's slots together. ``None`` when the read did
    #: not apply (fewer than two terminals) or nothing connected them.
    steiner_cost: Optional[float] = None
    #: the episode achieving that cost
    steiner_root: str = ""
    #: episodes reaching EVERY terminal -- the size of the conjunction, which
    #: is what distinguishes "one strong match" from "a real join"
    n_steiner_reaching: int = 0

    @property
    def turn_ids(self) -> list[str]:
        out: list[str] = []
        for f in self.facts:
            for t in f.turn_ids:
                if t not in out:
                    out.append(t)
        return out

    @property
    def sigma_facts(self) -> list[RetrievedFact]:
        """Facts that only the sigma expansion could have brought in."""
        return [f for f in self.facts if f.source == SOURCE_SIGMA]

    @property
    def n_sigma_facts(self) -> int:
        return len(self.sigma_facts)

    @property
    def coverage_facts(self) -> list[RetrievedFact]:
        return [
            f for f in self.facts if f.source in (SOURCE_COVERAGE, SOURCE_GEODESIC)
        ]

    @property
    def n_coverage_facts(self) -> int:
        return len(self.coverage_facts)

    @property
    def n_geodesic_facts(self) -> int:
        return sum(1 for f in self.facts if f.source == SOURCE_GEODESIC)

    @property
    def n_face_unit_facts(self) -> int:
        return sum(1 for f in self.facts if f.source == SOURCE_FACE_UNIT)

    def turn_ids_excluding(self, *sources: str) -> list[str]:
        """Turns the context would have had without those retrieval sources."""
        out: list[str] = []
        for f in self.facts:
            if f.source in sources:
                continue
            for t in f.turn_ids:
                if t not in out:
                    out.append(t)
        return out

    def turn_ids_only_from(self, *sources: str) -> list[str]:
        """Turns reached *only* by those sources -- the marginal contribution.

        Turns another source already retrieved are excluded, so this measures
        what the mechanism added, not merely what it touched.
        """
        others = {
            t for f in self.facts if f.source not in sources for t in f.turn_ids
        }
        out: list[str] = []
        for f in self.facts:
            if f.source not in sources:
                continue
            for t in f.turn_ids:
                if t not in others and t not in out:
                    out.append(t)
        return out

    @property
    def sigma_turn_ids(self) -> list[str]:
        """Evidence turns reachable *only* through the sigma expansion."""
        return self.turn_ids_only_from(SOURCE_SIGMA)

    @property
    def coverage_turn_ids(self) -> list[str]:
        """Evidence turns reachable *only* through coverage/geodesic."""
        return self.turn_ids_only_from(SOURCE_COVERAGE, SOURCE_GEODESIC)


# --------------------------------------------------------------------------- #
# Question -> vertices                                                         #
# --------------------------------------------------------------------------- #


class QuestionLinker:
    """Maps a question to the vertices it names.  Read-only, no LLM call.

    Deliberately *not* :class:`fgl.memory.entities.EntityResolver`: that one
    creates a vertex when nothing matches, which during QA would mutate the
    memory the protocol says we may only read (spec section 5).  Here a miss is
    simply a miss.

    Two passes, cheapest first: literal surface match of the question's n-grams
    against vertex names and aliases, then -- only if that found little -- the
    nearest vertices by embedding above a threshold.  Surface match carries
    score 1.0 because in LoCoMo the entities are mostly proper names, which the
    ingest already canonicalised.
    """

    def __init__(self, graph: FatGraph, embedder: Embedder, threshold: float = 0.75):
        self.graph = graph
        self.embedder = embedder
        self.threshold = threshold
        self._by_surface: dict[str, str] = {}
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for vid, vx in graph.vertices.items():
            for surface in (vx.name, *vx.aliases):
                key = normalize_name(surface)
                if key:
                    self._by_surface.setdefault(key, vid)
            if vx.embedding is not None:
                ids.append(vid)
                rows.append(_unit(vx.embedding))
        self._ids = ids
        self._matrix = np.vstack(rows) if rows else None

    def link(self, question: str, max_entities: int = 4) -> list[tuple[str, float]]:
        """``[(vertex_id, score), ...]``, best first, deduplicated."""
        found: dict[str, float] = {}
        for gram in _ngrams(normalize_name(question), 3):
            vid = self._by_surface.get(gram)
            if vid is not None:
                # longer surface wins ties: "support group" over "group"
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


def _ngrams(text: str, n: int) -> list[str]:
    words = text.split()
    out = []
    for size in range(min(n, len(words)), 0, -1):  # longest first
        out += [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]
    return out


def normalize_name(name: str) -> str:
    """Same normalisation the ingest used, imported lazily to avoid a cycle."""
    from fgl.memory.entities import normalize_name as _n

    return _n(name)


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _aggregate(values: Sequence[float], how: str) -> float:
    """Face-level similarity from its members'.  ``mean`` punishes long faces."""
    if not values:
        return 0.0
    if how == "mean":
        return float(np.mean(values))
    top = sorted(values, reverse=True)
    if how == "max":
        return float(top[0])
    return float(np.mean(top[:2]))


# --------------------------------------------------------------------------- #
# Retriever                                                                    #
# --------------------------------------------------------------------------- #


class FaceRetriever:
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
        self.index: VectorIndex = build_index(cfg.index, embedder.dim)
        ids, vecs = [], []
        for hid, he in graph.H.items():
            if he.embedding is None:
                continue
            ids.append(hid)
            vecs.append(he.embedding)
        if ids:
            self.index.add(ids, np.vstack(vecs))
        self.linker = (
            QuestionLinker(graph, embedder, cfg.retrieval.coverage_entity_threshold)
            if cfg.retrieval.face_coverage
            else None
        )
        self._adjacency: dict[str, list[tuple[str, str]]] | None = None
        self._face_by_half_edge: dict[str, Face] | None = None
        self._hubs_skipped = 0

    # -------------------------------------------------------- face lookup ----
    def face_of(self, half_edge_id: str) -> Face:
        """``graph.face_of`` memoised for the lifetime of this retriever.

        ``face_of`` walks the whole phi-cycle, and LoCoMo faces run to hundreds
        of half-edges (COERENCIA C9), so ``faces_through_vertex`` on a hub costs
        ``degree x |face|`` -- measured at 200x a single full decomposition on a
        degree-400 vertex. The graph is *read-only* during QA (spec section 5),
        which is exactly the precondition that makes memoising it sound.

        The cached ``Face`` is the canonical one, so ``half_edges`` may start at
        a different rotation than a fresh ``face_of(h)`` would.  Everything used
        downstream -- ``id``, the touched-vertex set, the member similarities --
        is rotation-invariant.
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

    def faces_through_vertex(self, vertex_id: str) -> list[Face]:
        seen: dict[str, Face] = {}
        for h in self.graph.sigma.get(vertex_id, ()):
            f = self.face_of(h)
            seen.setdefault(f.id, f)
        return list(seen.values())

    # ------------------------------------------------------------------ api --
    def retrieve(self, question: str) -> RetrievalResult:
        r = self.cfg.retrieval
        self._hubs_skipped = 0  # per question, not per retriever
        qvec = self.embedder.encode_one(question)
        raw = self.index.search(qvec, max(r.top_m_anchors * 8, 32))
        scored = [(hid, self._adjust(hid, s)) for hid, s in raw]
        scored.sort(key=lambda kv: -kv[1])

        # one anchor per edge: the two halves of a memory are near-duplicates
        anchors: list[tuple[str, float]] = []
        seen_edges: set[str] = set()
        for hid, score in scored:
            eid = self.graph.H[hid].edge_id
            if eid in seen_edges:
                continue
            seen_edges.add(eid)
            anchors.append((hid, score))
            if len(anchors) >= r.top_m_anchors:
                break

        result = RetrievalResult(
            anchors=anchors,
            all_anchor_ranking=scored,
            sigma_expand=r.sigma_expand,
            face_coverage=r.face_coverage,
            face_units=r.face_units,
        )
        if r.face_units:
            return self._retrieve_face_units(result, anchors, qvec)
        if not anchors and not r.face_coverage:
            return result

        # Each mechanism gets its slice carved out *up front*, otherwise the
        # face walk of anchor 0 eats the whole budget and the joins never run.
        # With both flags off the slices are zero and the loop below sees the
        # original budget, byte for byte as before.
        budget = r.budget_tokens
        sigma_budget = int(budget * r.sigma_budget_frac) if r.sigma_expand else 0
        cov_budget = int(budget * r.coverage_budget_frac) if r.face_coverage else 0
        face_budget = budget - sigma_budget - cov_budget

        used = 0
        seen_facts: set[str] = set()

        # Coverage runs FIRST: its facts are the ones selected for naming the
        # question's entities, so they belong at the head of the prompt, and
        # whatever they leave unspent rolls into the anchor walk.
        if r.face_coverage:
            used += self._expand_coverage(
                result, question=question, qvec=qvec,
                seen_facts=seen_facts, budget=cov_budget,
            )
            face_budget += max(0, cov_budget - used)

        # the anchor walk spends `face_budget` on top of whatever coverage used
        face_limit = used + face_budget
        for rank, (hid, score) in enumerate(anchors):
            if used >= face_limit:
                break
            if self.graph.H[hid].state == STATE_INCONGRUENT:
                result.any_incongruent = True
            face = self.face_of(hid)
            walk = self.graph.walk_face(hid, budget_tokens=face_limit - used)
            result.faces.append(face.id)
            for pos, he in enumerate(walk):
                used += self.graph._token_counter(he.text)  # noqa: SLF001
                if he.edge_id in seen_facts:
                    continue
                seen_facts.add(he.edge_id)
                if he.state == STATE_INCONGRUENT:
                    result.any_incongruent = True
                result.facts.append(
                    self._make_fact(
                        he,
                        anchor_rank=rank,
                        anchor_score=score,
                        face_id=face.id,
                        position_in_face=pos,
                        source=SOURCE_FACE,
                    )
                )

        if r.sigma_expand:
            # unspent face budget rolls into the expansion, never the reverse;
            # and the total is clamped so the expansion can never push a run
            # over budget_tokens (walk_face may already have overshot it by one
            # fact, since it always returns at least one).
            used += self._expand_sigma(
                result,
                anchors=anchors,
                qvec=qvec,
                seen_facts=seen_facts,
                budget=min(
                    sigma_budget + max(0, face_limit - used),
                    max(0, budget - used),
                ),
            )

        result.tokens_used = used
        self._truncate(result, r.max_facts_in_prompt)
        return result

    # ------------------------------------------------------ face-as-a-unit ---
    def _retrieve_face_units(
        self,
        result: RetrievalResult,
        anchors: Sequence[tuple[str, str]],
        qvec: np.ndarray,
    ) -> RetrievalResult:
        """Retrieve the *faces containing* the top facts, whole.

        One line of difference from the k-NN baseline:

            B3   -> return the top-k facts
            here -> return the faces those facts belong to

        and that line is where the ribbon structure enters.  What a face adds is
        the memory that does **not** match the question but belongs to the same
        narrative unit as one that does -- corroboration, which k independent
        matches cannot produce by construction.

        A face is treated as a *set*, not as a path.  Three measurements forced
        that reading: walking phi from an anchor lost 0.21 of multi-hop recall;
        choosing which face to walk by entity coverage was null because coverage
        saturated at 0.955; and permuting the prompt was null in multi-hop, so
        the sequence never carried the signal.  What survives is membership --
        and membership only became meaningful once the genus search turned 19
        monster faces holding 75% of the memory into a unimodal distribution
        (median face 263 -> 36 half-edges).  Hence the precondition below.

        Ranking is ``max`` similarity over the face's members, deliberately: a
        face containing a relevant fact *is* the entity-coverage signal, without
        an entity linker, a threshold, an aggregation mode or a geodesic
        fallback.  No weights, nothing to tune.
        """
        r = self.cfg.retrieval
        budget = r.budget_tokens
        seen_facts: set[str] = set()
        used = 0

        # dedup faces first: several top anchors usually share one face
        ranked: list[tuple[float, str, Face]] = []
        by_id: dict[str, Face] = {}
        for hid, _score in anchors:
            face = self.face_of(hid)
            if face.id in by_id:
                continue
            by_id[face.id] = face
            sims = [
                float(np.dot(qvec, self.graph.H[h].embedding))
                for h in face.half_edges
                if self.graph.H[h].embedding is not None
            ]
            ranked.append((max(sims) if sims else -1.0, face.id, face))
        ranked.sort(key=lambda t: (-t[0], t[1]))

        for score, _fid, face in ranked:
            if used >= budget:
                break
            # one half-edge per edge, in face order; the walk is only a
            # traversal device here, the unit is the whole face
            members = []
            seen_edges: set[str] = set()
            for h in face.half_edges:
                he = self.graph.H[h]
                if he.edge_id in seen_edges:
                    continue
                seen_edges.add(he.edge_id)
                members.append(he)
            # If the best face alone overflows the budget, keep its most
            # relevant memories rather than an arbitrary prefix of the cycle.
            cost = sum(self.graph._token_counter(m.text) for m in members)  # noqa: SLF001
            if cost > budget - used:
                members.sort(
                    key=lambda m: (
                        -float(np.dot(qvec, m.embedding))
                        if m.embedding is not None
                        else 1.0,
                        m.edge_id,
                    )
                )
            result.faces.append(face.id)
            result.face_unit_scores.append(round(float(score), 4))
            used += self._collect(
                result, members, seen_facts=seen_facts, budget=budget - used,
                face_id=face.id, source=SOURCE_FACE_UNIT, coverage=score,
                # the best-ranked unit plays the role rank-0 plays elsewhere, so
                # the incongruence rule keeps applying to "the top evidence"
                anchor_rank=0 if not result.faces[:-1] else 1,
            )
        result.tokens_used = used
        result.face_units_used = len(result.faces)
        self._truncate(result, r.max_facts_in_prompt)

        # what a k-NN over the same facts could not have produced: the k best
        # matches are what B3 would return, so anything else in the prompt is
        # there purely by face membership
        knn = {
            self.graph.H[h].edge_id
            for h, _ in self.index.search(qvec, max(r.top_m_anchors * 2, 20))
        }
        result.corroborating_facts = sum(1 for f in result.facts if f.edge_id not in knn)
        return result

    # ---------------------------------------------------- coverage retrieval --
    def score_faces(
        self, question_vertices: Sequence[str], qvec: np.ndarray
    ) -> list[tuple[Face, float, float]]:
        """``[(face, score, coverage), ...]`` best first.

        The unit of retrieval is the *trail*, not the fact.  Two terms:

        ``sim``       aggregated similarity of the face's memories to the
                      question -- ``max``/``top2`` rather than ``mean``, since
                      the mean punishes long faces and those are exactly the
                      cross-session ones a multi-hop question needs;
        ``coverage``  fraction of the question's entities the face touches.

        Coverage is the term that cosine cannot produce: a face through both
        ``Melanie`` and ``Bangkok`` is a candidate bridge even when none of its
        individual facts resembles the question.  ``faces_through_vertex`` makes
        the candidate set cheap -- it is bounded by the vertices' degree, not by
        the size of the graph.
        """
        r = self.cfg.retrieval
        if not question_vertices:
            return []
        wanted = list(dict.fromkeys(question_vertices))
        wanted_set = set(wanted)

        # Round-robin, not entity-by-entity: a bridge face is by definition one
        # that appears under more than one of the question's entities, and
        # draining the first entity's list would spend the whole cap before the
        # second one is ever consulted -- which is precisely the face we want.
        per_vertex = [iter(self.faces_through_vertex(vid)) for vid in wanted]
        candidates: dict[str, Face] = {}
        while per_vertex and len(candidates) < r.coverage_max_faces:
            alive = []
            for it in per_vertex:
                if len(candidates) >= r.coverage_max_faces:
                    break
                face = next(it, None)
                if face is None:
                    continue
                candidates.setdefault(face.id, face)
                alive.append(it)
            if not alive:
                break
            per_vertex = alive

        out: list[tuple[Face, float, float]] = []
        for face in candidates.values():
            touched = {self.graph.H[h].vertex_id for h in face.half_edges}
            coverage = len(touched & wanted_set) / len(wanted)
            sims = [
                float(np.dot(qvec, self.graph.H[h].embedding))
                for h in face.half_edges
                if self.graph.H[h].embedding is not None
            ]
            sim = _aggregate(sims, r.coverage_sim_aggregate)
            out.append((face, sim + r.coverage_weight * coverage, coverage))
        out.sort(key=lambda t: (-t[1], t[0].id))
        return out

    def geodesic(self, source: str, target: str, max_depth: int) -> list[str]:
        """Edge ids of a shortest path between two vertices (``[]`` if none).

        When no single face covers both entities, the memories that chain them
        are, by definition, a path between their vertices -- length 2 being the
        dominant case: ``A -- m1 -- B -- m2 -- C``.
        """
        if source == target:
            return []
        adj = self._adjacency_map()
        seen = {source}
        frontier = [(source, [])]
        for _ in range(max_depth):
            nxt = []
            for vid, path in frontier:
                for neighbour, edge_id in adj.get(vid, ()):
                    if neighbour in seen:
                        continue
                    if neighbour == target:
                        return [*path, edge_id]
                    seen.add(neighbour)
                    nxt.append((neighbour, [*path, edge_id]))
            if not nxt:
                break
            frontier = nxt
        return []

    def _adjacency_map(self) -> dict[str, list[tuple[str, str]]]:
        if self._adjacency is None:
            adj: dict[str, list[tuple[str, str]]] = {}
            for vid, halves in self.graph.sigma.items():
                adj[vid] = [
                    (self.graph.H[self.graph.alpha[h]].vertex_id, self.graph.H[h].edge_id)
                    for h in halves
                ]
            self._adjacency = adj
        return self._adjacency

    def _expand_coverage(
        self,
        result: RetrievalResult,
        question: str,
        qvec: np.ndarray,
        seen_facts: set[str],
        budget: int,
    ) -> int:
        """Retrieve whole trails chosen for covering the question's entities."""
        r = self.cfg.retrieval
        linked = self.linker.link(question, r.coverage_max_entities) if self.linker else []
        result.question_vertices = [v for v, _ in linked]
        result.question_entities = [
            self.graph.vertices[v].name for v, _ in linked if v in self.graph.vertices
        ]
        if not linked:
            return 0

        ranked = self.score_faces(result.question_vertices, qvec)
        n_wanted = len(result.question_vertices)
        result.coverage_faces_scored = len(ranked)
        # `cov` is a fraction; count entities as an integer rather than trusting
        # `cov * n >= 2` not to land a hair under the boundary
        result.coverage_faces_multi = sum(
            1 for _, _, cov in ranked if round(cov * n_wanted) >= 2
        )
        result.coverage_best = round(max((c for _, _, c in ranked), default=0.0), 4)

        used = 0
        for face, _score, coverage in ranked:
            if used >= budget:
                break
            if coverage <= 0:
                continue
            # start the walk at the face's most relevant memory, not at an
            # arbitrary half-edge: the budget cut should keep what matters.
            # The half-edge id breaks ties so the choice does not depend on the
            # rotation the face happens to be cached in.
            start = max(
                face.half_edges,
                key=lambda h: (
                    float(np.dot(qvec, self.graph.H[h].embedding))
                    if self.graph.H[h].embedding is not None
                    else -1.0,
                    h,
                ),
            )
            result.faces.append(face.id)
            used += self._collect(
                result, self.graph.walk_face(start, budget_tokens=budget - used),
                seen_facts=seen_facts, budget=budget - used,
                face_id=face.id, source=SOURCE_COVERAGE, coverage=coverage,
                max_facts=r.coverage_max_facts_per_face,
            )

        if r.coverage_geodesic_fallback and result.coverage_faces_multi == 0:
            used += self._expand_geodesic(
                result, seen_facts=seen_facts, budget=max(0, budget - used)
            )
        result.coverage_tokens = used
        return used

    def _expand_geodesic(
        self, result: RetrievalResult, seen_facts: set[str], budget: int
    ) -> int:
        """Fallback: the shortest chain between two of the question's entities."""
        r = self.cfg.retrieval
        vertices = result.question_vertices
        for i, a in enumerate(vertices):
            for b in vertices[i + 1 :]:
                path = self.geodesic(a, b, r.coverage_geodesic_max_depth)
                if not path:
                    continue
                halves = []
                for eid in path:
                    h1, _ = self.graph.edge_half_edges(eid)
                    halves.append(self.graph.H[h1])
                result.geodesic_used = True
                result.geodesic_len = len(path)
                return self._collect(
                    result, halves, seen_facts=seen_facts, budget=budget,
                    face_id=f"geodesic:{a}:{b}", source=SOURCE_GEODESIC,
                    coverage=1.0,
                    via_entity=" → ".join(
                        self.graph.vertices[x].name for x in (a, b)
                        if x in self.graph.vertices
                    ),
                )
        return 0

    def _collect(
        self,
        result: RetrievalResult,
        half_edges: Sequence[HalfEdge],
        seen_facts: set[str],
        budget: int,
        face_id: str,
        source: str,
        coverage: float = 0.0,
        via_entity: str = "",
        max_facts: int = 0,
        anchor_rank: int = -1,
    ) -> int:
        """Append half-edges as facts, honouring dedup, budget and fact cap."""
        used = 0
        taken = 0
        for pos, he in enumerate(half_edges):
            if max_facts and taken >= max_facts:
                break
            if he.edge_id in seen_facts:
                continue
            cost = self.graph._token_counter(he.text)  # noqa: SLF001
            if used + cost > budget:
                break
            used += cost
            taken += 1
            seen_facts.add(he.edge_id)
            if he.state == STATE_INCONGRUENT:
                result.any_incongruent = True
            result.facts.append(
                self._make_fact(
                    he,
                    # -1 = selected structurally, not by the anchor ranking
                    anchor_rank=anchor_rank,
                    anchor_score=coverage,
                    face_id=face_id,
                    position_in_face=pos,
                    source=source,
                    via_entity=via_entity,
                )
            )
        return used

    # ------------------------------------------------------- sigma expansion --
    def sigma_neighborhood(
        self, half_edge_id: str, qvec: Optional[np.ndarray] = None
    ) -> list[tuple[str, str]]:
        """``[(half_edge_id, via_vertex_id), ...]`` -- the join candidates.

        The orbit of ``sigma`` at a vertex is the set of memories touching that
        entity, so these are exactly the facts that share an entity with the
        anchor memory: the second hop.  Both ends of the anchor edge are
        scanned when ``sigma_expand_both_ends`` is on, because the bridge may
        live on either entity of the anchor.

        With ``sigma_rerank`` the orbit is ranked by similarity to the question
        rather than taken in cyclic order -- under ``sigma-time`` the cyclic
        successor is merely the chronologically adjacent fact.  Note this is
        *not* the global k-NN: candidates are constrained to the orbit, i.e. we
        ask "which fact **about this entity** answers the question", which is
        the whole point.

        With ``sigma_skip_hub_degree`` a vertex above that degree is skipped
        entirely, the way an index skips a stopword: the two speakers touch 86%
        of the edges, so their orbit is not "memories about this entity", it is
        "memories from this conversation", and the join through them is noise
        wearing the shape of a bridge.
        """
        r = self.cfg.retrieval
        starts = [half_edge_id]
        if r.sigma_expand_both_ends:
            twin = self.graph.alpha.get(half_edge_id)
            if twin is not None:
                starts.append(twin)

        out: list[tuple[str, str]] = []
        for start in starts:
            vid = self.graph.H[start].vertex_id
            if self._is_hub(vid):
                self._hubs_skipped += 1
                continue
            orbit = self._orbit(start, r.sigma_max_orbit_scan)
            if r.sigma_rerank and qvec is not None:
                orbit = self._rank_by_query(orbit, qvec)
            out.extend((h, vid) for h in orbit[: r.sigma_expand_k])
        return out

    def _is_hub(self, vertex_id: str) -> bool:
        """Is this vertex the graph's equivalent of a stopword?"""
        limit = self.cfg.retrieval.sigma_skip_hub_degree
        return bool(limit) and self.graph.degree(vertex_id) >= limit

    def _orbit(self, start: str, cap: int) -> list[str]:
        """Half-edges of ``sigma`` at ``start``'s vertex, excluding ``start``.

        ``cap <= 0`` means no cap, consistent with ``sigma_expand_max_anchors``
        and ``ingest.max_facts_per_session``.  (It used to clamp to one, so a
        configured 0 silently crippled the expansion instead of widening it.)
        """
        limit = cap if cap > 0 else self.graph.degree(self.graph.H[start].vertex_id)
        out: list[str] = []
        cur = start
        while len(out) < limit:
            cur = self.graph.sigma_next(cur)
            if cur == start:
                break
            out.append(cur)
        return out

    def _rank_by_query(self, half_edges: Sequence[str], qvec: np.ndarray) -> list[str]:
        scored: list[tuple[str, float]] = []
        for h in half_edges:
            emb = self.graph.H[h].embedding
            if emb is None:
                continue
            scored.append((h, self._adjust(h, float(np.dot(qvec, emb)))))
        scored.sort(key=lambda kv: -kv[1])
        return [h for h, _ in scored]

    def _expand_sigma(
        self,
        result: RetrievalResult,
        anchors: Sequence[tuple[str, float]],
        qvec: np.ndarray,
        seen_facts: set[str],
        budget: int,
    ) -> int:
        """Add sigma-orbit neighbours of the top anchors.  Returns tokens used."""
        r = self.cfg.retrieval
        limit = r.sigma_expand_max_anchors or len(anchors)
        used = 0
        for rank, (hid, score) in enumerate(anchors[:limit]):
            for h, vid in self.sigma_neighborhood(hid, qvec):
                result.sigma_scanned += 1
                he = self.graph.H[h]
                if he.edge_id in seen_facts:
                    result.sigma_dup += 1
                    continue
                cost = self.graph._token_counter(he.text)  # noqa: SLF001
                if used + cost > budget:
                    result.sigma_over_budget += 1
                    continue
                used += cost
                seen_facts.add(he.edge_id)
                if he.state == STATE_INCONGRUENT:
                    result.any_incongruent = True
                if vid not in result.sigma_vertices:
                    result.sigma_vertices.append(vid)
                vertex = self.graph.vertices.get(vid)
                result.facts.append(
                    self._make_fact(
                        he,
                        anchor_rank=rank,
                        anchor_score=score,
                        face_id=f"sigma:{vid}",
                        position_in_face=len(result.sigma_vertices),
                        source=SOURCE_SIGMA,
                        via_vertex=vid,
                        via_entity=vertex.name if vertex else vid,
                    )
                )
        result.sigma_tokens = used
        result.sigma_hubs_skipped = self._hubs_skipped
        return used

    # ------------------------------------------------------------ internals --
    def _make_fact(
        self,
        he: HalfEdge,
        *,
        anchor_rank: int,
        anchor_score: float,
        face_id: str,
        position_in_face: int,
        source: str,
        via_vertex: str = "",
        via_entity: str = "",
    ) -> RetrievedFact:
        return RetrievedFact(
            edge_id=he.edge_id,
            text=he.text,
            timestamp=he.timestamp,
            date_raw=self.dates.get(he.session_id, he.timestamp),
            session_id=he.session_id,
            turn_ids=list(he.turn_ids),
            state=he.state,
            level=he.level,
            anchor_rank=anchor_rank,
            anchor_score=anchor_score,
            face_id=face_id,
            position_in_face=position_in_face,
            source=source,
            via_vertex=via_vertex,
            via_entity=via_entity,
        )

    def _truncate(self, result: RetrievalResult, max_facts: int) -> None:
        """Cap the prompt size without either pool starving the other.

        The blunt ``facts[:max]`` would silently drop whatever the multi-hop
        mechanisms contributed, and the ablation would then measure nothing.
        But absolute priority for the joins is the same mistake mirrored: with a
        long covering trail the joins alone overflow ``max_facts`` and *every*
        anchor fact is evicted, so G5/G6 stop being supersets of G1 and the
        comparison the conditions exist to make quietly stops holding.

        So each pool is guaranteed its share and only the *unused* part of a
        share is lent to the other -- the same discipline the token budget
        already follows.
        """
        if len(result.facts) <= max_facts:
            return
        if not (result.sigma_expand or result.face_coverage):
            result.facts = result.facts[:max_facts]
            return

        joins = [f for f in result.facts if f.source in JOIN_SOURCES]
        faces = [f for f in result.facts if f.source not in JOIN_SOURCES]
        # at least one slot each, whenever that pool has anything to offer
        join_quota = max(1, int(max_facts * self.cfg.retrieval.max_facts_join_frac))
        n_joins = min(len(joins), max(join_quota, max_facts - len(faces)))
        n_faces = min(len(faces), max_facts - n_joins)

        keep = {id(f) for f in joins[:n_joins]} | {id(f) for f in faces[:n_faces]}
        result.facts = [f for f in result.facts if id(f) in keep]  # order preserved

    def top_edges(self, question: str, k: int) -> list[str]:
        """Top-k *edges* by anchor score -- used for the recall@k metric."""
        qvec = self.embedder.encode_one(question)
        out: list[str] = []
        for hid, _ in self.index.search(qvec, max(k * 4, 16)):
            eid = self.graph.H[hid].edge_id
            if eid not in out:
                out.append(eid)
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

    # ------------------------------------------------------------ internals --
    def _adjust(self, half_edge_id: str, score: float) -> float:
        he = self.graph.H[half_edge_id]
        if he.level == 2:
            score += self.cfg.retrieval.level2_boost
        if he.shadowed:
            score -= self.cfg.retrieval.shadowed_penalty
        return score


# --------------------------------------------------------------------------- #
# Answering                                                                    #
# --------------------------------------------------------------------------- #


def render_context(result: RetrievalResult, shuffle_seed: Optional[int] = None) -> str:
    """Facts grouped by trail, in face order, each prefixed with its date.

    Sigma groups are labelled with the entity they hinge on: telling the model
    *where* two trails meet is exactly the composition step a multi-hop
    question asks for, and it is free -- the retriever already knows it.

    With ``shuffle_seed`` the facts are permuted and the trail headers dropped,
    which keeps the *content* of the context byte-identical while destroying its
    *order*.  That isolates the one thing a ribbon graph contributes over a
    plain graph: sigma is an ordering, phi is the walk it induces, and faces are
    the trails that walk produces.  If the score does not move under this
    permutation, ordering carries no signal and optimising sigma cannot pay.
    """
    if shuffle_seed is not None:
        facts = list(result.facts)
        random.Random(shuffle_seed).shuffle(facts)
        return "\n".join(
            f"[{f.date_raw or f.timestamp}]"
            f"{' (summary)' if f.level == 2 else ''}"
            f"{' [INCONSISTENT]' if f.state == STATE_INCONGRUENT else ''} {f.text}"
            for f in facts
        ) or "(no memories retrieved)"

    lines: list[str] = []
    current_face: Optional[str] = None
    trail_no = 0
    for f in result.facts:
        if f.face_id != current_face:
            current_face = f.face_id
            if f.source == SOURCE_SIGMA:
                lines.append(f"--- other memories about {f.via_entity} ---")
            elif f.source == SOURCE_GEODESIC:
                lines.append(f"--- chain linking {f.via_entity} ---")
            elif f.source == SOURCE_FACE_UNIT:
                trail_no += 1
                lines.append(f"--- related memories, group {trail_no} ---")
            elif f.source == "bp_entity":  # fgl.retrieval.bipartite.SOURCE_BP_ENTITY
                # (string literals here, not imported: bipartite.py imports
                # FROM this module, so importing back would be circular)
                lines.append(f"--- also about {f.via_entity} ---")
            elif f.source == "bp_bridge":
                lines.append(f"--- links back to {f.via_entity} ---")
            elif f.source == "bp_dense":
                lines.append("--- similar memories ---")
            # L2 channels (fgl.retrieval.slots). String literals for the same
            # reason as the bp_* ones above: that module imports FROM here.
            elif f.source == "slot_concept":
                lines.append(f"--- about {f.via_entity} ---")
            elif f.source == "slot_predicate":
                lines.append(f"--- times someone {f.via_entity} ---")
            elif f.source == "slot_type":
                lines.append(f"--- a kind of {f.via_entity} ---")
            elif f.source == "slot_actor":
                lines.append(f"--- {f.via_entity} ---")
            elif f.source == "slot_time":
                lines.append(f"--- around {f.via_entity} ---")
            elif f.source == "slot_dense":
                lines.append("--- similar memories ---")
            elif f.source == "slot_steiner":
                lines.append(f"--- chain linking {f.via_entity} ---")
            elif f.source == "slot_bridge":
                lines.append(f"--- chain linking {f.via_entity} ---")
            else:
                trail_no += 1
                lines.append(f"--- trail {trail_no} ---")
        marker = " (summary)" if f.level == 2 else ""
        flag = " [INCONSISTENT]" if f.state == STATE_INCONGRUENT else ""
        lines.append(f"[{f.date_raw or f.timestamp}]{marker}{flag} {f.text}")
    return "\n".join(lines) if lines else "(no memories retrieved)"


class Answerer:
    def __init__(self, llm: LLMClient, prompts: PromptLibrary, cfg: Config) -> None:
        self.llm = llm
        self.prompts = prompts
        self.cfg = cfg

    def answer(
        self, conv: Conversation, question: Question, result: RetrievalResult
    ) -> str:
        if not result.facts:
            return ABSTAIN_ANSWER
        if self.cfg.retrieval.incongruent_abstain and result.any_incongruent:
            # The rule is about the *anchors'* faces; join facts are extra
            # evidence and must not dilute it either way. Note the guard on the
            # empty list: under G5/G6 the anchor walk can legitimately
            # contribute nothing (coverage spent the budget, or truncation kept
            # only joins), and `all([])` is True -- which used to abstain on a
            # context whose every surviving fact was perfectly congruent.
            anchor_facts = [
                f
                for f in result.facts
                if f.anchor_rank == 0 and f.source not in JOIN_SOURCES
            ]
            if anchor_facts and all(
                f.state == STATE_INCONGRUENT for f in anchor_facts
            ):
                return ABSTAIN_ANSWER
        # Open-domain (category 3) asks what is *likely*, so the extractive
        # instruction actively forbids the task and the model abstains on a
        # prompt that already holds the evidence. Routed to its own prompt when
        # `retrieval.open_domain_inference` is on.
        open_domain = (
            self.cfg.retrieval.open_domain_inference and question.category == 3
        )
        # A question asking for a list gets the enumerating prompt. The flag is
        # set by the retriever from the QUESTION's wording, never from the gold
        # category, so this stays a property of the input rather than of the
        # label -- and open-domain keeps its own prompt, which already asks for
        # an inference and would fight an instruction to enumerate.
        enumerate_set = bool(getattr(result, "set_question", False)) and not open_domain
        # Seeded per question, so the permutation is fixed across runs but not
        # identical for every question. `crc32`, not `hash()`: string hashing is
        # salted per process, which would make the ablation unreproducible --
        # exactly the property it needs most, since its whole claim rests on
        # comparing two runs that differ in nothing but order.
        seed = (
            self.cfg.seed + zlib.crc32(question.question.encode())
            if self.cfg.retrieval.shuffle_context
            else None
        )
        prompt = self.prompts.render(
            "answer_open" if open_domain else ("answer_set" if enumerate_set else "answer"),
            speaker_a=conv.speaker_a,
            speaker_b=conv.speaker_b,
            context=render_context(result, shuffle_seed=seed),
            question=question.prompt_question(),
        )
        out = self.llm.complete(
            prompt,
            system=SYSTEM_ANSWERER_OPEN if open_domain else SYSTEM_ANSWERER,
            purpose=(
                "qa/answer_open" if open_domain
                else ("qa/answer_set" if enumerate_set else "qa/answer")
            ),
            max_tokens=self.cfg.retrieval.answer_max_tokens,
        )
        return clean_answer(out)


def clean_answer(text: str) -> str:
    """Strip the framing models add around short extractive answers."""
    t = (text or "").strip()
    for prefix in ("SHORT ANSWER:", "Short answer:", "Answer:", "A:"):
        if t.startswith(prefix):
            t = t[len(prefix) :].strip()
    t = t.strip().strip('"').strip()
    if not t:
        return ABSTAIN_ANSWER
    return t.split("\n")[0].strip()
