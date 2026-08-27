"""Reading a proposition store: the query plan, and the two readers.

A question is a proposition with a hole::

    ?  subject=Melanie   predicate~"paint"   object=?   as_of=?   modality=factual

Structure first, similarity second:

1. resolve the question's named entities against the entity store;
2. match its predicate by embedding against the predicate vocabulary;
3. look up propositions satisfying the bound arguments;
4. if none, ask whether the *entities* are known at all -- "I have never heard
   of this person" and "I know them and nobody ever said that" are different
   answers, and only a store of resolved propositions can tell them apart;
5. if the hole needs a join -- the question binds A and asks about B, and no
   single proposition holds both -- take the entities of step 3's propositions
   as new bindings and query again. **Bounded to one extra step.** This is the
   evidence closure, over propositions where it is exact, rather than over a
   similarity graph where L3 measured it as noise;
6. emit the propositions with their spans, grouped by the step that found them.

M1 and M2: one store, two readers
---------------------------------
Both conditions run the same ingest and consolidation, and the store is always
laid out as a fatgraph carrying the rotation. The flat reader ignores it. The
two are allowed to disagree about exactly four things, and no others:

===========================  ==========================  =========================
what                         flat                        ribbon
===========================  ==========================  =========================
an entity's timeline         inverted index, then sort   sigma-orbit (already
                                                         chronological)
the hole test                field comparison            corner at the
                                                         proposition vertex
finding the join             re-query the index          phi-face walk
emission order under         by score                    by orbit/face position
truncation
===========================  ==========================  =========================

The last two are the only places rotation can genuinely pay: under a budget the
orbit gives a principled order where the index gives a score order, and a face
walk can reach a chain whose intermediate binding never made a top-k.

``meca.ribbon_order=score`` plus ``meca.ribbon_join=index`` makes the ribbon
reader reproduce the flat reader fact for fact. That identity is pinned by a
test -- the same discipline that proved L5 reduces to L2, and without it the
comparison would be worth nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from fgl.config import Config
from fgl.core import FatGraph
from fgl.memory.propositions import (
    KIND_ENTITY,
    KIND_PROPOSITION,
    Proposition,
    PropositionStore,
    TimePoint,
    normalise,
    orbit_pids,
    proposition_vertex_index,
    store_from_graph,
    vertices_of_kind,
)
from fgl.retrieval.embeddings import Embedder
from fgl.retrieval.faces import RetrievalResult, RetrievedFact

#: provenance labels, mirrored in ``render_context``
SOURCE_MECA_DIRECT = "meca_direct"
SOURCE_MECA_JOIN = "meca_join"
SOURCE_MECA_DENSE = "meca_dense"

#: Closed-class tokens that never bind an argument. Short and blunt: this is a
#: question parser, not a tagger, and what counts as an entity is decided by
#: the store rather than by this list.
_STOP_WORDS = (
    "a an the of in on at to for from by with about into over after before "
    "what when where who whom which why how did do does is are was were be been "
    "has have had will would can could should may might must and or but if then "
    "that this these those it its his her their there here as than"
)
_STOP = frozenset(_STOP_WORDS.split(" "))


# --------------------------------------------------------------------------- #
# The question, as a proposition with a hole                                   #
# --------------------------------------------------------------------------- #


@dataclass
class Target:
    """What the question asks for, in the store's own vocabulary."""

    entities: list[str] = field(default_factory=list)
    predicate: str = ""
    predicate_key: str = ""
    as_of: Optional[TimePoint] = None
    #: the question asks for a fact about the world, not about someone's plans
    factual_only: bool = True
    text: str = ""

    def as_dict(self) -> dict:
        return {
            "entities": list(self.entities),
            "predicate": self.predicate,
            "as_of": self.as_of.value if self.as_of else None,
            "factual_only": self.factual_only,
        }


_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october"
    r"|november|december)\b", re.I,
)
_MONTH_NAMES = (
    "january february march april may june july august september october "
    "november december"
)
_MONTHS = {m: f"{i + 1:02d}" for i, m in enumerate(_MONTH_NAMES.split(" "))}
#: A question about what someone *would* do, or *wants*, is answered from a
#: non-factual proposition; one about what happened is not. Read off the
#: question's own wording, never off a benchmark category.
_NON_FACTUAL_CUE = re.compile(
    r"\b(plan|planning|plans|intend|intends|hope|hopes|want|wants|wish|wishes"
    r"|plann?ed to|would like|thinking of|considering)\b", re.I,
)


def parse_question(question: str, store: PropositionStore) -> Target:
    """Turn a question into a target proposition. Reads only the question text.

    Entity candidates are matched against the store's own entity table, so what
    counts as an entity is a fact about the memory rather than a tagger's guess
    -- and a name the memory never heard simply does not bind, which is what
    makes the "never heard of them" answer possible.
    """
    words = [w for w in re.findall(r"[\w'-]+", question)]
    lowered = [w.lower() for w in words]

    entities: list[str] = []
    # longest match first, so "Melanie Carter" wins over "Melanie"
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i:i + size])
            key = normalise(phrase)
            if not key or key in _STOP:
                continue
            if store.knows_entity(key) and not any(
                normalise(e) == key or key in normalise(e) for e in entities
            ):
                entities.append(phrase)

    content = [w for w in lowered if w not in _STOP and len(w) > 2]
    predicate = " ".join(content[:4])

    as_of = None
    my = _MONTH.search(question)
    yr = _YEAR.search(question)
    if yr and my:
        as_of = TimePoint.parse(f"{yr.group(0)}-{_MONTHS[my.group(0).lower()]}")
    elif yr:
        as_of = TimePoint.parse(yr.group(0))

    return Target(
        entities=entities,
        predicate=predicate,
        predicate_key=normalise(predicate),
        as_of=as_of,
        factual_only=not bool(_NON_FACTUAL_CUE.search(question)),
        text=question,
    )


# --------------------------------------------------------------------------- #
# Readers                                                                      #
# --------------------------------------------------------------------------- #


class _Reader:
    """Shared plan; the two subclasses differ only where the table above says."""

    def __init__(self, retriever: MecaRetriever) -> None:
        self.r = retriever

    # -- the four seams ----------------------------------------------------
    def timeline(self, entity: str) -> list[Proposition]:
        raise NotImplementedError

    def join(self, seeds: Sequence[Proposition], target: Target) -> list[Proposition]:
        raise NotImplementedError

    def order(
        self, props: Sequence[Proposition], scores: dict[str, float]
    ) -> list[Proposition]:
        raise NotImplementedError

    def fills_hole(self, prop: Proposition, target: Target) -> bool:
        raise NotImplementedError


class FlatReader(_Reader):
    """Inverted indexes and scores. No rotation is consulted anywhere."""

    def timeline(self, entity: str) -> list[Proposition]:
        return self.r.store.about(entity)  # fetch, then sort

    def join(self, seeds: Sequence[Proposition], target: Target) -> list[Proposition]:
        """Second step: re-query the index with the seeds' own arguments.

        Through ANY argument, not only through resolved entities. The ribbon
        reader's phi-walk pivots through every argument vertex, so a flat join
        restricted to entities would differ from it for a reason that has
        nothing to do with the rotation -- and the whole comparison rests on
        the two differing only where the design says they may.
        """
        seen = {p.pid for p in seeds}
        bound = {normalise(e) for e in target.entities}
        out: list[Proposition] = []
        for seed in seeds:
            for value in seed.arguments():
                if value in bound:
                    # pivoting through what the question already bound walks
                    # straight back into the region we came from
                    continue
                for pid in self.r.store.by_argument.get(value, ()):
                    if pid in seen:
                        continue
                    prop = self.r.store.get(pid)
                    if prop is None:
                        continue
                    seen.add(pid)
                    self.r._via[pid] = value
                    out.append(prop)
        return out

    def order(self, props, scores):
        return sorted(props, key=lambda p: (-scores.get(p.pid, 0.0), p.pid))

    def fills_hole(self, prop: Proposition, target: Target) -> bool:
        """Field comparison: does this proposition carry a value to return?"""
        return bool(prop.object) or bool(prop.qualifiers) or prop.when() is not None


class RibbonReader(_Reader):
    """The same plan, read off the rotation.

    Three operations change, and they are named in the module docstring. With
    ``ribbon_order=score`` and ``ribbon_join=index`` this class is required by
    test to produce byte-identical output to :class:`FlatReader`.
    """

    def timeline(self, entity: str) -> list[Proposition]:
        vid = self.r.entity_vertices.get(normalise(entity))
        if vid is None:
            return []
        # sigma at an entity vertex is chronological BY CONSTRUCTION -- the
        # ingest adds propositions in time order and each incidence appends --
        # so this is the timeline with no sort at all, at cost O(degree).
        out = []
        for pid in orbit_pids(self.r.graph, vid):
            prop = self.r.store.get(pid)
            if prop is not None:
                out.append(prop)
        return out

    def join(self, seeds: Sequence[Proposition], target: Target) -> list[Proposition]:
        if self.r.cfg.meca.ribbon_join == "index":
            return FlatReader(self.r).join(seeds, target)
        return self._face_join(seeds, target)

    def _face_join(self, seeds: Sequence[Proposition], target: Target
                   ) -> list[Proposition]:
        """Walk phi from each seed's proposition vertex.

        The face through a proposition alternates proposition and argument
        vertices, so walking it enumerates the claims that share an argument
        with the seed -- in trail order rather than in score order. That is the
        one thing the index cannot do: reach a chain whose intermediate binding
        never made a top-k.
        """
        graph = self.r.graph
        seen = {p.pid for p in seeds}
        bound = {normalise(e) for e in target.entities}
        out: list[Proposition] = []
        budget = self.r.cfg.meca.ribbon_walk_max
        for seed in seeds:
            svid = self.r.prop_vertices.get(seed.pid)
            if svid is None:
                continue
            for h0 in graph.sigma.get(svid, ()):
                # never pivot through an argument the question already bound:
                # that walks back into the region we came from
                arg_vid = graph.H[graph.alpha[h0]].vertex_id
                arg = graph.vertices[arg_vid]
                if arg.meta.get("key") in bound:
                    continue
                steps = 0
                h = h0
                while steps < budget:
                    h = graph.phi(h)
                    steps += 1
                    if h == h0:
                        break
                    vx = graph.vertices[graph.H[h].vertex_id]
                    if vx.meta.get("kind") != KIND_PROPOSITION:
                        continue
                    pid = vx.meta.get("pid", "")
                    if not pid or pid in seen:
                        continue
                    prop = self.r.store.get(pid)
                    if prop is None:
                        continue
                    seen.add(pid)
                    self.r._via[pid] = arg.name
                    out.append(prop)
        return out

    def order(self, props, scores):
        if self.r.cfg.meca.ribbon_order == "score":
            return FlatReader(self.r).order(props, scores)
        # orbit order: chronological and local, which is what the rotation
        # gives for free. Under a tight budget this is where it can differ.
        return sorted(
            props,
            key=lambda p: (p.when().value if p.when() else "9999", p.pid),
        )

    def fills_hole(self, prop: Proposition, target: Target) -> bool:
        """The corner test: does this proposition have an argument to return?

        A corner is a pair of adjacent arguments in the rotation, so a
        proposition whose ring holds more than (subject, predicate) has
        something to answer with. Same verdict as the flat field comparison on
        a well-formed store -- and that agreement is itself worth pinning,
        because a disagreement means the rotation and the fields have drifted.
        """
        vid = self.r.prop_vertices.get(prop.pid)
        if vid is None:
            return False
        return self.r.graph.degree(vid) > 2


# --------------------------------------------------------------------------- #
# The retriever                                                                #
# --------------------------------------------------------------------------- #


class MecaRetriever:
    """Same public contract as every other retriever: ``retrieve(question)``."""

    WANTS_QUESTION_CORPUS = False

    def __init__(
        self,
        graph: FatGraph,
        embedder: Embedder,
        cfg: Config,
        dates: dict[str, str] | None = None,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.cfg = cfg
        self.dates = dates or {}
        self.store = store_from_graph(graph)
        self.entity_vertices = vertices_of_kind(graph, KIND_ENTITY)
        self.prop_vertices = proposition_vertex_index(graph)
        self._via: dict[str, str] = {}

        self.reader: _Reader = (
            RibbonReader(self) if cfg.meca.reader == "ribbon" else FlatReader(self)
        )

        # dense fallback over the statements themselves, so a question whose
        # entities never bind still retrieves something rather than nothing
        self._pids = list(self.store.propositions)
        if self._pids:
            statements = [self.store.propositions[p].statement() for p in self._pids]
            self._matrix = _unit(
                np.asarray(self.embedder.encode(statements), dtype=np.float32)
            )
        else:
            self._matrix = np.zeros((0, 1), dtype=np.float32)

    # ------------------------------------------------------------------ api --
    def retrieve(self, question: str) -> RetrievalResult:
        m = self.cfg.meca
        self._via = {}
        result = RetrievalResult()
        target = parse_question(question, self.store)
        result.question_entities = list(target.entities)

        # --- step 1-3: bound arguments ------------------------------------
        seeds: list[Proposition] = []
        seen: set[str] = set()
        for ent in target.entities:
            for prop in self.reader.timeline(ent):
                if prop.pid not in seen:
                    seen.add(prop.pid)
                    seeds.append(prop)

        # --- step 4: the honest abstention --------------------------------
        # Not a score with a threshold: a lookup with an answer.
        result.slot_support = 1.0 if seeds else 0.0
        if not seeds and target.entities:
            result.abstain_reason = "unknown_entity"
        elif not seeds:
            result.abstain_reason = "unbound_question"

        qvec = _unit(np.asarray(self.embedder.encode_one(question), dtype=np.float32))
        scores = self._score(seeds, target, qvec)

        # --- step 5: the join ---------------------------------------------
        joined: list[Proposition] = []
        if m.join_steps > 0 and seeds:
            ranked_seeds = self.reader.order(seeds, scores)[: m.join_budget]
            joined = self.reader.join(ranked_seeds, target)
            join_scores = self._score(joined, target, qvec)
            # A joined proposition is evidence about the connector, not about
            # the question, so it never outranks a directly bound one.
            for pid, value in join_scores.items():
                scores[pid] = value * 0.5

        # --- dense fallback ------------------------------------------------
        dense: list[Proposition] = []
        if len(seeds) + len(joined) < m.join_budget and self._pids:
            known = {p.pid for p in seeds} | {p.pid for p in joined}
            sims = self._matrix @ qvec
            order = np.argsort(-sims)[: m.join_budget * 2]
            for i in order:
                pid = self._pids[int(i)]
                if pid in known:
                    continue
                prop = self.store.propositions[pid]
                dense.append(prop)
                scores[pid] = float(sims[int(i)]) * m.dense_weight

        # --- emission -------------------------------------------------------
        result.tokens_used = self._emit(target, seeds, joined, dense, scores, result)
        result.faces = sorted({f.face_id for f in result.facts})
        result.question_vertices = [
            self.entity_vertices.get(normalise(e), "") for e in target.entities
        ]
        return result

    # -------------------------------------------------------------- scoring --
    def _score(
        self, props: Sequence[Proposition], target: Target, qvec: np.ndarray
    ) -> dict[str, float]:
        """Binding coverage first, resemblance second, recency as tie-break."""
        if not props:
            return {}
        bound = {normalise(e) for e in target.entities}
        pred_words = set(target.predicate_key.split())
        vecs = _unit(
            np.asarray(
                self.embedder.encode([p.statement() for p in props]), dtype=np.float32
            )
        )
        sims = vecs @ qvec
        out: dict[str, float] = {}
        for prop, sim in zip(props, sims, strict=False):
            score = 0.0
            hits = sum(1 for e in prop.entities() if normalise(e) in bound)
            score += 1.5 * hits
            if pred_words:
                pw = set(normalise(prop.predicate).split()) | set(
                    normalise(prop.object).split()
                )
                score += 1.0 * len(pred_words & pw) / len(pred_words)
            score += float(sim)
            if target.as_of is not None:
                point = prop.when()
                if point is not None and (
                    target.as_of.overlaps(point)
                ):
                    score += 1.0
            if not prop.is_factual and target.factual_only:
                # still emitted and still labelled -- but a plan must not
                # outrank the fact that answers the question
                score -= 1.0
            if not prop.is_current:
                score -= 0.5
            if prop.is_derived:
                score -= 0.25
            if self.reader.fills_hole(prop, target):
                score += 0.5
            out[prop.pid] = score
        return out

    # ------------------------------------------------------------- emission --
    def _emit(
        self, target: Target, seeds: Sequence[Proposition],
        joined: Sequence[Proposition], dense: Sequence[Proposition],
        scores: dict[str, float], result: RetrievalResult,
    ) -> int:
        m = self.cfg.meca
        budget = self.cfg.retrieval.budget_tokens
        max_facts = self.cfg.retrieval.max_facts_in_prompt
        groups = (
            ("what the memory holds", SOURCE_MECA_DIRECT, seeds),
            ("linked through", SOURCE_MECA_JOIN, joined),
            ("possibly related", SOURCE_MECA_DENSE, dense),
        )
        used = 0
        rank = 0
        for label, source, props in groups:
            keep = [
                p for p in props
                if (m.emit_non_factual or p.is_factual)
                and (m.emit_superseded or p.is_current)
            ]
            for prop in self.reader.order(keep, scores):
                if rank >= max_facts:
                    return used
                text = self._render(prop)
                # the graph's own counter, so a MECA budget and an L-line
                # budget are the same unit and the comparison at "equal
                # budget" means what it says
                cost = self.graph._token_counter(text)
                if used + cost > budget and rank > 0:
                    return used
                used += cost
                ev = prop.evidence[0] if prop.evidence else None
                result.facts.append(RetrievedFact(
                    edge_id=prop.pid,
                    text=text,
                    timestamp=ev.timestamp if ev else "",
                    date_raw=ev.date_raw if ev else "",
                    session_id="",
                    turn_ids=[e.doc_id for e in prop.evidence],
                    state="emergent",
                    level=1,
                    anchor_rank=rank,
                    anchor_score=scores.get(prop.pid, 0.0),
                    face_id=source,
                    position_in_face=rank,
                    source=source,
                    via_vertex="",
                    via_entity=self._via.get(prop.pid, label),
                ))
                rank += 1
        return used

    def _render(self, prop: Proposition) -> str:
        """The claim, then the exact text it came from. Both, always.

        The statement is what the generator can compose over; the span is the
        proof and the wording the answer should reuse. Emitting only the
        statement would make the memory unfalsifiable to its own reader.
        """
        lines = [f"{prop.label()} {prop.statement()}"]
        for ev in prop.evidence[:2]:
            where = ev.date_raw or ev.timestamp
            who = f"{ev.author}, " if ev.author else ""
            lines.append(f'    "{ev.span.strip()}"  ({who}{where})')
        return "\n".join(lines)

    # ----------------------------------------------------- retriever contract -
    def top_edges(self, question: str, k: int = 10) -> list[tuple[str, float]]:
        result = self.retrieve(question)
        return [(f.edge_id, f.anchor_score) for f in result.facts[:k]]

    def turn_ids_for_edges(self, edge_ids: Sequence[str]) -> list[str]:
        out: list[str] = []
        for pid in edge_ids:
            prop = self.store.get(pid)
            if prop is None:
                continue
            for ev in prop.evidence:
                if ev.doc_id not in out:
                    out.append(ev.doc_id)
        return out


def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)
