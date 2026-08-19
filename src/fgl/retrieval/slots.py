"""Typed-slot retrieval over the episode graph -- condition L2.

The move
--------
L1 scores a turn by "does it mention a noun the question mentions?", with a
dense backstop. Measured on L1's own output, 87-95% of the evidence turns it
missed share *no* noun with the question at all -- there is no path to score.

Here a question is parsed into the same typed vocabulary the memory is built
from (:mod:`fgl.memory.slots`) and an episode is scored by how many of the
question's slots are incident to it, per channel::

    actor      who the question is about  -- weighted by how much of the
               episode that person actually contributed, so an episode where
               they merely say "nice!" does not outrank one where they talk
    predicate  the verb that frames the question ("adopt", "paint", "perform")
    concept    the nouns, entity-resolved exactly as in L1
    type       WordNet hypernyms, so a question asking by category ("foods")
               reaches a turn answering by instance ("roasted chicken")
    time       month bucket, so "in April 2022" is a real constraint
    dense      cosine over episode embeddings, a full participant as in L1

Every structural channel is degree-damped (``1 / (1 + log(deg))``): a slot
incident to 200 episodes cannot discriminate between them, so it contributes
almost nothing, while a slot incident to three carries the query. Above the
hub cut-off a slot stops being enumerated altogether and becomes a filter
bonus -- the same "do not index a stopword" rule L1 applies to entities, now
applied per *kind*, which is what makes a high-degree ``actor`` a partition
instead of a hub.

Where the thresholds come from
------------------------------
Nothing in this module reads a literal out of the config any more. The hub
cut-off, the paraphrase cosine floor, the actor prior and the question-side
stoplist all arrive through a :class:`fgl.memory.calibration.Calibration`
built once per conversation, which either returns the swept constants
(``slots.calibration="absolute"``) or estimates each one from the unlabelled
graph and the question *texts* (``"derived"``), recording per knob which of
the two happened. That is the difference between a parameter and a fitted
number: a derived threshold can be inherited by a corpus nobody annotated.
The time channel goes further and deletes its parameter outright -- it indexes
year, month and day and lets the damping term pick the level (see
:mod:`fgl.memory.slots`).

Abstention without an LLM
-------------------------
An adversarial question names a combination that never happened. In this
graph that is a *corner test*: take the question's most specific slot and ask
whether it has any episode in common with the question's actor. Two failure
shapes, both deterministic and both reported in
``RetrievalResult.abstain_reason``:

``missing_slot``  the slot has no vertex at all -- nobody ever painted;
``empty_corner``  the slot exists but never co-occurs with that actor.

Enabled by ``slots.abstain_on_empty_corner``. Measured false-positive rate on
substantive questions is reported by ``fgl slots-oracle`` before it is worth
turning on -- it is off by default precisely so that decision is made from a
number rather than from the fact that the mechanism is elegant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from fgl.config import Config
from fgl.core import Face, FatGraph, HalfEdge
from fgl.memory.entities import normalize_name
from fgl.memory.calibration import Calibration, calibrate
from fgl.memory.ner import NonGenerativeExtractor
from fgl.memory.slots import (
    KIND_ACTOR,
    KIND_CONCEPT,
    KIND_EPISODE,
    KIND_PREDICATE,
    KIND_TIME,
    KIND_TYPE,
    QUESTION_PREDICATE_STOP,
    SPECIFIC_KINDS,
    granularity_of,
    is_set_question,
    lift_types,
    match_actor,
    parse_granularities,
    question_time_slots,
)
from fgl.retrieval.embeddings import Embedder, VectorIndex, build_index
from fgl.retrieval.faces import RetrievalResult, RetrievedFact, _unit

#: ``RetrievedFact.source`` values specific to this retriever. One per
#: channel, so a results directory can be audited channel by channel instead
#: of trusted -- "which of the five actually earned the recall" is the whole
#: question this condition exists to answer.
SOURCE_SLOT_ACTOR = "slot_actor"
SOURCE_SLOT_PREDICATE = "slot_predicate"
SOURCE_SLOT_CONCEPT = "slot_concept"
SOURCE_SLOT_TYPE = "slot_type"
SOURCE_SLOT_TIME = "slot_time"
SOURCE_SLOT_DENSE = "slot_dense"

SLOT_SOURCES = (
    SOURCE_SLOT_CONCEPT,
    SOURCE_SLOT_PREDICATE,
    SOURCE_SLOT_TYPE,
    SOURCE_SLOT_ACTOR,
    SOURCE_SLOT_TIME,
    SOURCE_SLOT_DENSE,
)

_SOURCE_BY_KIND = {
    KIND_ACTOR: SOURCE_SLOT_ACTOR,
    KIND_PREDICATE: SOURCE_SLOT_PREDICATE,
    KIND_CONCEPT: SOURCE_SLOT_CONCEPT,
    KIND_TYPE: SOURCE_SLOT_TYPE,
    KIND_TIME: SOURCE_SLOT_TIME,
}

#: Which channel gets to label an episode when several hit it. Ordered by how
#: much the hit actually tells you: a concept match is specific, a type match
#: is a category guess, a dense match is a resemblance.
_SOURCE_PRIORITY = {
    SOURCE_SLOT_DENSE: 0,
    SOURCE_SLOT_TIME: 1,
    SOURCE_SLOT_ACTOR: 2,
    SOURCE_SLOT_TYPE: 3,
    SOURCE_SLOT_PREDICATE: 4,
    SOURCE_SLOT_CONCEPT: 5,
}


@dataclass
class QuestionSlots:
    """The question, parsed into the memory's own vocabulary."""

    actors: list[str] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    #: the question asks for a LIST ("what foods", "what has X read"), decided
    #: from its wording alone -- never from the LoCoMo category
    is_set: bool = False

    def as_pairs(self) -> list[tuple[str, str]]:
        return (
            [(KIND_ACTOR, k) for k in self.actors]
            + [(KIND_PREDICATE, k) for k in self.predicates]
            + [(KIND_CONCEPT, k) for k in self.concepts]
            + [(KIND_TYPE, k) for k in self.types]
            + [(KIND_TIME, k) for k in self.times]
        )

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.actors, self.predicates, self.concepts, self.types, self.times)
        )


@dataclass
class _Episode:
    """Scoring state for one episode: what the typed channels found."""

    vid: str
    score: float = 0.0
    source: str = SOURCE_SLOT_DENSE
    via_vertex: str = ""
    via_entity: str = ""
    channels: set[str] = field(default_factory=set)


class SlotRetriever:
    """Same public contract as the other two retrievers: ``retrieve``,
    ``top_edges``, ``turn_ids_for_edges`` -- so :class:`fgl.pipeline.Runner`
    dispatches on ``cfg.retrieval.mode`` alone.
    """

    def __init__(
        self,
        graph: FatGraph,
        embedder: Embedder,
        cfg: Config,
        date_by_session: dict[str, str] | None = None,
        question_corpus: Sequence[str] | None = None,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.cfg = cfg
        self.dates = date_by_session or {}
        #: question TEXTS only -- never an answer, an evidence list or a
        #: category. The framing-stoplist estimator needs the query
        #: distribution and nothing else; keeping the parameter typed as bare
        #: strings is what makes that checkable at every call site.
        self.question_corpus = list(question_corpus or ())
        self._face_by_half_edge: dict[str, Face] | None = None

        # --- episode side: dense index + a text handle per episode ----------
        self.episode_vids: list[str] = [
            vid for vid, vx in graph.vertices.items()
            if vx.meta.get("kind") == KIND_EPISODE
        ]
        self.episode_index: VectorIndex = build_index(cfg.index, embedder.dim)
        rows = [
            (vid, graph.vertices[vid].embedding)
            for vid in self.episode_vids
            if graph.vertices[vid].embedding is not None
        ]
        if rows:
            self.episode_index.add([v for v, _ in rows], np.vstack([e for _, e in rows]))

        # A second, TURN-level dense index over the same episodes. The episode
        # vector is the vector of the joined turns, which is the right thing
        # for "what is this exchange about" and the wrong thing for "which
        # single line resembles the question" -- averaging two turns dilutes
        # whichever one actually matches. Measured: with the episode vector
        # alone, open-domain recall_context fell from L1's 0.615 to 0.500 at an
        # identical budget, and open-domain is the category that lives entirely
        # on the dense channel. So the dense hit is scored per turn and lifted
        # to its episode by max, while every structural channel stays at the
        # episode. Built once per conversation, not per question, and the
        # embedder is cached, so a rerun pays nothing.
        self.turn_index: VectorIndex = build_index(cfg.index, embedder.dim)
        self._episode_of_turn: dict[str, str] = {}
        turn_keys: list[str] = []
        turn_texts: list[str] = []
        for vid in self.episode_vids:
            meta = graph.vertices[vid].meta
            for turn_id, text in zip(meta.get("turn_ids", []), meta.get("turn_texts", [])):
                self._episode_of_turn[turn_id] = vid
                turn_keys.append(turn_id)
                turn_texts.append(text)
        if turn_texts:
            self.turn_index.add(turn_keys, np.vstack(embedder.encode(turn_texts)))
        self._n_turns = max(len(turn_keys), 1)

        # --- slot side: exact lookup per kind -------------------------------
        # Exact, not nearest-neighbour: a typed slot key is already a lemma or
        # a bucket, so "close enough" would only blur the kinds into each
        # other. Concepts keep L1's embedding fallback below, where it earns
        # its keep on paraphrase.
        self._by_kind: dict[str, dict[str, str]] = {
            k: {} for k in (KIND_ACTOR, KIND_PREDICATE, KIND_CONCEPT, KIND_TYPE, KIND_TIME)
        }
        concept_ids: list[str] = []
        concept_rows: list[np.ndarray] = []
        for vid, vx in graph.vertices.items():
            kind = vx.meta.get("kind")
            if kind == KIND_EPISODE or kind is None:
                continue
            table = self._by_kind.get(kind)
            if table is None:
                continue
            for surface in (vx.meta.get("key", ""), vx.name, *vx.aliases):
                key = normalize_name(surface) if kind == KIND_CONCEPT else (surface or "")
                if key:
                    table.setdefault(key, vid)
            if kind == KIND_CONCEPT and vx.embedding is not None:
                concept_ids.append(vid)
                concept_rows.append(_unit(vx.embedding))
        self._concept_ids = concept_ids
        self._concept_matrix = np.vstack(concept_rows) if concept_rows else None

        self.actor_keys: list[str] = sorted(self._by_kind[KIND_ACTOR])
        self._extractor: Optional[NonGenerativeExtractor] = None

        # Which time levels this graph actually carries, read off the graph
        # rather than off the config. The two normally agree, but a graph
        # built under one setting and queried under another would otherwise
        # emit question keys no vertex can answer -- a silent recall loss that
        # looks like a scoring bug. Falls back to the config for an empty
        # time channel.
        observed = {
            granularity_of(vx.meta.get("key", ""))
            for vx in graph.vertices.values()
            if vx.meta.get("kind") == KIND_TIME
        }
        observed.discard("")
        self.time_granularities: tuple[str, ...] = (
            parse_granularities(",".join(sorted(observed)))
            if observed
            else parse_granularities(cfg.slots.time_granularities)
        )

        # Every threshold that used to be a literal, resolved once per
        # conversation with its provenance attached. `slots.calibration=
        # "absolute"` returns the swept numbers unchanged; "derived" measures
        # them here off the unlabelled graph (and, for the framing stoplist,
        # off the question TEXTS). Exposed as an attribute so a run can dump
        # `retriever.calibration.as_dict()` and show which is which.
        self.calibration: Calibration = calibrate(
            cfg,
            graph,
            concept_matrix=self._concept_matrix,
            question_corpus=self.question_corpus,
            extractor=self.extractor if self.question_corpus else None,
        )

    # ------------------------------------------------------------- helpers --
    @property
    def extractor(self) -> NonGenerativeExtractor:
        """Lazily built, and configured exactly like the ingest's.

        The question has to be parsed by the same pipeline that parsed the
        turns, or the two vocabularies do not meet: a question lemmatised
        differently from the turn cannot match it however good the scoring is.
        """
        if self._extractor is None:
            sl = self.cfg.slots
            self._extractor = NonGenerativeExtractor(
                model_name=sl.ner_model,
                max_chunk_words=sl.max_chunk_words,
                min_chars=sl.min_concept_chars,
                extract_verbs=True,
                split_persons=True,
            )
        return self._extractor

    def face_of(self, half_edge_id: str) -> Face:
        """Memoised face lookup, kept for diagnostics only -- retrieval reads
        sigma-orbits directly and never walks phi (see the module docstring of
        :mod:`fgl.memory.slots` for why the face is not the object here).
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

    # ------------------------------------------------------- question side --
    def parse_question(self, question: str) -> QuestionSlots:
        sl = self.cfg.slots
        ex = self.extractor.extract(question)

        actors = match_actor(question, self.actor_keys)

        predicates = [
            v.text for v in ex.verbs if v.text not in QUESTION_PREDICATE_STOP
        ][: sl.max_question_predicates]

        stop = self.calibration.question_noun_stop
        concepts: list[str] = []
        for cand in ex.candidates:
            key = normalize_name(cand.text)
            if not key or key in stop or key in concepts:
                continue
            concepts.append(key)
        concepts = concepts[: sl.max_question_concepts]

        # The question's own nouns are matched against the TYPE index too:
        # "What foods..." is not looking for a concept called "food", it is
        # looking for everything lifted TO food. Its hypernyms are added as
        # well, one level of slack for when the question is itself specific.
        types: list[str] = []
        if sl.lift_types:
            for c in concepts:
                head = c.split()[-1]
                if head in self._by_kind[KIND_TYPE] and head not in types:
                    types.append(head)
                for t in lift_types(c, sl.max_types_per_concept):
                    if t not in types:
                        types.append(t)
            types = types[: sl.max_question_types]

        # finest level first, coarser ones behind it as backoff: see
        # fgl.memory.slots.question_time_slots for why emitting all of them is
        # free once scoring is degree-damped.
        times = question_time_slots(question, self.time_granularities)
        return QuestionSlots(
            actors=actors, predicates=predicates, concepts=concepts,
            types=types, times=times,
            is_set=sl.enumerate_sets and is_set_question(question, ex.doc),
        )

    def _resolve_slot(self, kind: str, key: str) -> Optional[str]:
        """Slot key -> vertex id, with L1's embedding fallback for concepts."""
        table = self._by_kind.get(kind, {})
        if kind == KIND_TIME:
            hit = table.get(key)
            if hit is not None:
                return hit
            # a year-only bucket from the question ("2022") matches any month
            # vertex inside it; first match wins, and the caller scores every
            # episode in its orbit, so which month is picked does not matter
            for bucket, vid in sorted(table.items()):
                if bucket.startswith(key):
                    return vid
            return None
        hit = table.get(key)
        if hit is not None:
            return hit
        if kind == KIND_CONCEPT and self._concept_matrix is not None:
            sims = self._concept_matrix @ _unit(self.embedder.encode_one(key))
            best = int(np.argmax(sims))
            if float(sims[best]) >= self.calibration.concept_link_threshold:
                return self._concept_ids[best]
        return None

    # ------------------------------------------------------------------ api --
    def retrieve(self, question: str) -> RetrievalResult:
        sl = self.cfg.slots
        r = self.cfg.retrieval
        result = RetrievalResult()

        slots = self.parse_question(question)
        linked: list[tuple[str, str, str]] = []  # (kind, key, vertex_id)
        for kind, key in slots.as_pairs():
            vid = self._resolve_slot(kind, key)
            if vid is not None:
                linked.append((kind, key, vid))

        unlinked = [
            (kind, key) for kind, key in slots.as_pairs()
            if kind in SPECIFIC_KINDS
            and not any(k == kind and kk == key for k, kk, _ in linked)
        ]

        result.question_vertices = [vid for _, _, vid in linked]
        result.question_entities = [f"{kind}:{key}" for kind, key, _ in linked]
        result.all_anchor_ranking = [(vid, 1.0) for _, _, vid in linked]
        result.set_question = slots.is_set
        result.slot_channels = {
            kind: [k for kk, k, _ in linked if kk == kind]
            for kind in (KIND_ACTOR, KIND_PREDICATE, KIND_CONCEPT, KIND_TYPE, KIND_TIME)
        }

        # --- the corner test, before any scoring ---------------------------
        support, reason = self._corner_support(linked, unlinked)
        result.slot_support = support
        result.abstain_reason = reason
        if reason and sl.abstain_on_empty_corner:
            return result  # no facts -> Answerer abstains

        candidates: dict[str, _Episode] = {}

        def touch(ep_vid: str, add: float, kind: str, via: str = "",
                  label: str = "") -> None:
            c = candidates.get(ep_vid)
            if c is None:
                c = _Episode(vid=ep_vid)
                candidates[ep_vid] = c
            c.score += add
            source = _SOURCE_BY_KIND.get(kind, SOURCE_SLOT_DENSE)
            c.channels.add(source)
            if _SOURCE_PRIORITY[source] > _SOURCE_PRIORITY.get(c.source, -1):
                c.source = source
                if via:
                    c.via_vertex = via
                    c.via_entity = label
                    
        # 1. dense channel, a full participant and not a fallback (as in L1).
        # Breadth follows max_facts_in_prompt for the same reason it does
        # there: candidates the prompt could have held but were never generated
        # are budget thrown away.
        breadth = max(r.top_m_anchors * 4, r.max_facts_in_prompt, 24)
        qvec = self.embedder.encode_one(question)
        best_dense: dict[str, float] = {}
        dense_by_turn: dict[str, float] = {}
        # Every turn, not the top-k: this ranking decides which line of an
        # episode gets emitted first, so a turn scored 0.0 for being outside a
        # top-k window silently falls back to document order and the episode
        # offers its opening line instead of the one that answers the
        # question. Measured cost of getting that wrong: single-hop
        # recall_context 0.858 -> 0.672. It is one matmul over a few hundred
        # rows, so full ranking is cheaper than the bug.
        for turn_id, score in self.turn_index.search(qvec, self._n_turns):
            ep_vid = self._episode_of_turn.get(turn_id)
            if ep_vid is None:
                continue
            dense_by_turn[turn_id] = float(score)
            best_dense[ep_vid] = max(best_dense.get(ep_vid, 0.0), float(score))
        for ep_vid, score in self.episode_index.search(qvec, breadth):
            best_dense[ep_vid] = max(best_dense.get(ep_vid, 0.0), float(score))
        # Episode-level resemblance: "does any line of this exchange match the
        # question". Distinct from the per-turn dense term added in `_emit`,
        # which asks "does THIS line match" -- and the difference between the
        # two is exactly the reply case, where the matching line and the
        # answering line are not the same one. Measured, on 3 conversations at
        # an identical budget: dropping this term cost multi-hop 0.677 ->
        # 0.653, temporal 0.978 -> 0.944 and open-domain 0.524 -> 0.440, and
        # no setting of sibling_frac recovered it.
        for ep_vid, score in best_dense.items():
            touch(ep_vid, sl.dense_weight * score, "")

        # 2. structural channels
        weight_by_kind = {
            KIND_ACTOR: sl.actor_weight,
            KIND_PREDICATE: sl.predicate_weight,
            KIND_CONCEPT: sl.concept_weight,
            KIND_TYPE: sl.type_weight,
            KIND_TIME: sl.time_weight,
        }
        for kind, key, vid in linked:
            degree = self.graph.degree(vid)
            if degree == 0:
                continue
            if kind == KIND_ACTOR:
                continue  # applied as a prior below, not as a summand
            base = weight_by_kind[kind]
            # degree damping: a slot on 200 episodes separates none of them.
            # The exponent is a knob because the right amount is not obvious a
            # priori: damping helps precision (single-hop) and hurts
            # enumeration (multi-hop, where the whole orbit of one moderately
            # common topic IS the answer). Swept, not assumed -- see
            # slots.slot_damping.
            damped = base / (1.0 + math.log(degree)) ** sl.slot_damping
            # per KIND: an actor incident to half the episodes and a concept
            # incident to half the episodes are not the same event, and one
            # absolute cut-off cannot say so. See fgl.memory.calibration.
            hub = degree >= self.calibration.hub_degree(kind)
            for ep_vid in self._orbit_episodes(vid):
                touch(
                    ep_vid,
                    sl.hub_weight if hub else damped,
                    kind if not hub else "",
                    via=vid,
                    label=self.graph.vertices[vid].name,
                )

        # 2b. ENUMERATION. A set question is not answered by the most similar
        # episode -- it is answered by the whole orbit. sigma at a slot vertex
        # is already that orbit, in chronological order, so this needs no
        # ranking and no second retrieval pass: take the RAREST specific slot
        # the question linked (the one that actually discriminates), intersect
        # its orbit with the episodes the named actor owns, and lift every
        # member into the prompt. This is the one place in the design where the
        # ribbon structure does something a dense retriever cannot: the answer
        # is a list, and the rotation IS the list.
        #
        # Rarest, not all: enumerating a common slot's orbit would flood the
        # budget with everything the conversation ever touched, which is the
        # hub mistake this project already made once.
        if slots.is_set:
            enumerable = [
                (self.graph.degree(vid), vid) for kind, _, vid in linked
                if kind in SPECIFIC_KINDS
                and 0 < self.graph.degree(vid) < self.calibration.hub_degree(kind)
            ]
            if enumerable:
                _, vid = min(enumerable)
                owners = [key for kind, key, _ in linked if kind == KIND_ACTOR]
                name = self.graph.vertices[vid].name
                for ep_vid in self._orbit_episodes(vid):
                    if owners and not any(
                        self._actor_weight(k, ep_vid) > 0.0 for k in owners
                    ):
                        continue
                    result.n_enumerated += 1
                    touch(ep_vid, sl.set_orbit_boost, KIND_CONCEPT,
                          via=vid, label=name)

        # 3. the actor is a PARTITION, not a channel. Adding a constant per
        # named actor to every episode that actor touches shifts half the graph
        # by the same amount and discriminates nothing -- the actor's degree is
        # ~half the episodes by construction. What the measurement actually
        # says (98.5-99.7% of questions name one speaker; the evidence is that
        # speaker's 96-100% of the time) is a statement about which episodes
        # are *eligible*, so it applies multiplicatively: an episode the named
        # person barely contributed to is demoted, not merely un-promoted.
        # Applied to the STRUCTURAL score only, which is what this block
        # multiplies -- the dense term is added later, per turn. That split is
        # deliberate: "who is this about" is a claim about the memory's
        # structure, while resemblance to the question is not, and open-domain
        # questions ("would Caroline consider...") are answered as often by
        # the other speaker's turn as by the named one's.
        actor_keys = [key for kind, key, _ in linked if kind == KIND_ACTOR]
        if actor_keys and candidates:
            # floor and full are 1/n_speakers and the median dominant-
            # contributor share when `slots.calibration=derived`, so the prior
            # sharpens on its own as the number of participants grows instead
            # of needing a re-sweep. See fgl.memory.calibration.
            floor = self.calibration.actor_prior_floor
            full = self.calibration.actor_prior_full
            for ep_vid, c in candidates.items():
                own = max(self._actor_weight(k, ep_vid) for k in actor_keys)
                c.score *= floor + (1.0 - floor) * min(own / full, 1.0)

        if not candidates:
            return result

        # 4. rank TURNS, not episodes
        result.tokens_used = self._emit(candidates, dense_by_turn, result)
        result.faces = sorted({f.face_id for f in result.facts})
        result.sigma_vertices = [vid for _, _, vid in linked]
        return result

    # ------------------------------------------------------------ internals --
    def _emit(
        self,
        episodes: dict[str, "_Episode"],
        dense_by_turn: dict[str, float],
        result: RetrievalResult,
    ) -> int:
        """Score every TURN, spend the budget on turns, keep the episode as
        the thing that makes a turn findable.

        This is where the model's central bet is actually cashed, and it took
        three measured attempts to get right:

        * emitting whole episodes bought 18.9 units against L1's 58.7 at an
          identical budget -- everything that needs breadth lost;
        * emitting turns but draining one episode before starting the next
          restored the unit count and still lost multi-hop, because 55 turns
          from 18 consecutive episodes cover 18 regions of a conversation
          where L1's 58 independent turns cover 58;
        * emitting one turn per episode in rounds restored the regions and
          cost single-hop 0.858 -> 0.682, because the turn that *answers* a
          question is very often the partner of the turn that *matches* it --
          which is the entire reason this model has episodes at all.

        So a turn's score is its own similarity plus two things it inherits
        from its exchange: the typed channels that hit the episode (actor,
        predicate, concept, type, time) and ``sibling_frac`` of the best
        similarity among its siblings. That last term is the reply rule stated
        as arithmetic -- "We just did a contemporary piece called 'Finding
        Freedom'" is worth retrieving because the turn next to it matched, not
        because it did. Nothing is grouped, nothing is drained: the budget
        buys the best turns available, and the pair rises together because
        both halves carry the same inherited score.
        """
        r = self.cfg.retrieval
        sl = self.cfg.slots

        scored: list[tuple[float, str, str, "_Episode", HalfEdge]] = []
        for ep_vid, ep in episodes.items():
            halves = self.graph.sigma.get(ep_vid, ())
            if not halves:
                continue
            he = self.graph.H[halves[0]]
            meta = self.graph.vertices[ep_vid].meta
            turn_ids = meta.get("turn_ids") or list(he.turn_ids)
            turn_texts = meta.get("turn_texts") or [he.text]
            dens = [dense_by_turn.get(t, 0.0) for t in turn_ids]
            for i, (turn_id, text) in enumerate(zip(turn_ids, turn_texts)):
                sibling = max((d for j, d in enumerate(dens) if j != i), default=0.0)
                score = (
                    ep.score
                    + sl.dense_weight * dens[i]
                    + sl.sibling_frac * sl.dense_weight * sibling
                )
                scored.append((score, turn_id, text, ep, he))

        scored.sort(key=lambda t: -t[0])
        used = 0
        budget = r.budget_tokens
        for score, turn_id, text, ep, he in scored:
            if len(result.facts) >= r.max_facts_in_prompt or used >= budget:
                break
            cost = self.graph._token_counter(text)  # noqa: SLF001
            if used and used + cost > budget:
                continue
            used += cost
            result.facts.append(self._make_fact(he, ep, turn_id, text, score))

        # regroup by episode so render_context shows each exchange contiguously,
        # best episode first: the ranking above is a budget policy, not a
        # presentation one, and a reader needs the exchange, not the ranking.
        best: dict[str, float] = {}
        for f in result.facts:
            best[f.face_id] = max(best.get(f.face_id, 0.0), f.anchor_score)
        result.facts.sort(key=lambda f: (-best[f.face_id], f.face_id, f.turn_ids[0]))
        return used

    def _orbit_episodes(self, slot_vid: str) -> list[str]:
        """The slot's sigma-orbit, as episode vertex ids, in chronological
        order. This is the ribbon-graph read: no scoring, no ranking -- the
        rotation at a slot vertex *is* the timeline of that slot.
        """
        out: list[str] = []
        for h in self.graph.sigma.get(slot_vid, ()):
            ep_vid = self.graph.H[self.graph.alpha[h]].vertex_id
            if not out or out[-1] != ep_vid:
                out.append(ep_vid)
        return out

    def _actor_weight(self, actor: str, ep_vid: str) -> float:
        """How much of this episode is actually *about* that person, in [0, 1].

        An adjacency pair contains a turn from each speaker by construction, so
        "the episode contains a turn by X" is true for both of them and
        discriminates nothing. What does discriminate is how much content X
        contributed: the person answering "Twisted my knee last Friday" owns
        the episode; the person who said "Wow, great!" does not. Someone merely
        named in it gets ``mention_weight`` -- enough to be findable ("What
        people has Maria met?"), not enough to outrank a speaker.
        """
        meta = self.graph.vertices[ep_vid].meta
        content: dict = meta.get("speaker_content", {})
        total = sum(content.values())
        if total and actor in content:
            return content[actor] / total
        if actor in meta.get("mentioned_actors", []):
            return self.cfg.slots.mention_weight
        return 0.0

    def _corner_support(
        self,
        linked: Sequence[tuple[str, str, str]],
        unlinked: Sequence[tuple[str, str]] = (),
    ) -> tuple[float, str]:
        """Does the question's (actor, specific-slot) corner exist anywhere?

        Returns ``(support, reason)``; ``reason`` is empty when the question is
        supported. Two ways it can fail, and they are different claims:

        ``missing_slot``  the question's whole content vocabulary is absent
                          from the memory -- nobody ever mentioned any of it;
        ``empty_corner``  the content exists, but never in an episode this
                          person owns. "What did Melanie paint?" when it was
                          Caroline who painted.

        The ownership test is a *majority* one (``corner_actor_min``), not
        "contributed anything". An episode is an adjacency pair, so both
        speakers appear in almost all of them by construction; requiring merely
        non-zero contribution would make the corner exist everywhere and the
        test would never fire. Requiring the majority of the episode's content
        is what makes "whose episode is this" a real question.

        Silent by design when the question names no actor, or no specific slot:
        there is no corner to refute, and inventing one would trade the thing
        this test is for (a free true negative) for the thing it must never do
        (delete a correct answer).
        """
        sl = self.cfg.slots
        actors = [vid for kind, _, vid in linked if kind == KIND_ACTOR]
        specific = [
            vid for kind, _, vid in linked
            if kind in SPECIFIC_KINDS
            and self.graph.degree(vid) < self.calibration.hub_degree(kind)
        ]
        asked_specific = specific or [
            k for kind, k in unlinked if kind in SPECIFIC_KINDS
        ]
        if not actors or not asked_specific:
            return 1.0, ""
        if not specific:
            # every content word the question used is absent from the memory
            return 0.0, "missing_slot"

        owned_by: dict[str, set[str]] = {}
        for avid in actors:
            akey = self.graph.vertices[avid].meta.get("key", "")
            owned_by[avid] = {
                ep for ep in self._orbit_episodes(avid)
                if self._actor_weight(akey, ep) >= sl.corner_actor_min
            }

        best = 0.0
        for svid in specific:
            slot_eps = set(self._orbit_episodes(svid))
            if not slot_eps:
                continue
            for owned in owned_by.values():
                if owned:
                    best = max(best, len(slot_eps & owned) / len(slot_eps))
        if best > 0.0:
            return best, ""
        return 0.0, "empty_corner"

    def _make_fact(
        self, he: HalfEdge, c: "_Episode", turn_id: str, text: str, score: float
    ) -> RetrievedFact:
        # face_id is the EPISODE, not the channel: it is what render_context
        # groups by, and the group that means something to a reader is the
        # exchange, not "everything the concept channel happened to find".
        return RetrievedFact(
            edge_id=he.edge_id,
            text=text,
            timestamp=he.timestamp,
            date_raw=self.dates.get(he.session_id, he.timestamp),
            session_id=he.session_id,
            turn_ids=[turn_id],
            state=he.state,
            level=he.level,
            anchor_rank=0,
            anchor_score=score,
            face_id=c.vid,
            position_in_face=0,
            source=c.source,
            via_vertex=c.via_vertex,
            via_entity=c.via_entity,
        )

    # ------------------------------------------------------------- metrics --
    def top_edges(self, question: str, k: int) -> list[str]:
        """Top-k edges by episode relevance, for the recall@k metric.

        Defined the same way L1 defines it: rank by the dense channel alone --
        the one signal every condition has -- and expand each winner into its
        incidences, so the number stays comparable across conditions instead of
        measuring a different thing per retriever.
        """
        qvec = self.embedder.encode_one(question)
        out: list[str] = []
        for ep_vid, _ in self.episode_index.search(qvec, max(k, 8)):
            for h in self.graph.sigma.get(ep_vid, ()):
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
