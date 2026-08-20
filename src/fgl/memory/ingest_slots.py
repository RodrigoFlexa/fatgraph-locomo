"""Typed-slot ingestion over episodes -- condition L2.

Same public contract as :class:`fgl.memory.ingest.Ingestor` and
:class:`fgl.memory.ingest_bipartite.BipartiteIngestor`: ``ingest(conv) ->
(FatGraph, IngestReport)``, and the same hard guarantee as L1 -- **zero LLM
calls** by default. One spaCy pass per turn produces every channel at once
(noun chunks, verbs, PERSON spans, DATE spans); WordNet and a date resolver
do the rest.

The zero-LLM guarantee holds unconditionally through condition L5. Condition
L6 (``cfg.bridges.enabled``) adds one optional pass at the very end of
:meth:`SlotIngestor.ingest`, after this graph is otherwise complete: see
:mod:`fgl.memory.bridges`. It is off by default, so every existing condition
is unaffected by its mere presence in this module.

What the graph looks like
-------------------------
Vertices are of six kinds (:mod:`fgl.memory.slots`)::

    episode      a contiguous, topically cohesive run of turns
    actor        a person: either speaker, or anyone named in a turn
    predicate    a content-verb lemma
    concept      a noun phrase, entity-resolved exactly like L1's entities
    type         a WordNet hypernym of a concept ("chicken" -> food, meat)
    time         a month bucket, from the session date and resolved relatives

Edges are incidences ``episode -- slot``, one per observed (episode, slot)
pair. Nothing is inferred and nothing is generated: an edge means "this was
literally said in these turns", so a slot's degree is a real frequency, not an
artefact of how a model happened to phrase something.

sigma is designed here, not inherited
-------------------------------------
* at a **slot** vertex, ``sigma`` is chronological -- free, because episodes
  are built in session-then-transcript order and incidences are appended;
* at an **episode** vertex, ``sigma`` follows :data:`fgl.memory.slots.SLOT_ORDER`
  and then document order, which is what makes the *corners* of an episode
  the meaningful pairs (who, did-what), (did-what, with-what), (with-what,
  when). That is the one ribbon-graph property this condition actually leans
  on, and it is a deliberate choice of rotation rather than a by-product of
  processing order -- see the ``slots`` module docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from fgl.config import Config
from fgl.core import STATE_EMERGENT, FatGraph
from fgl.data.locomo import Conversation, Session, Turn
from fgl.llm import LLMClient
from fgl.llm.prompts import PromptLibrary
from fgl.logging_utils import JsonlLogger, NullLogger
from fgl.memory.entities import EntityResolver
from fgl.memory.ingest import IngestReport
from fgl.memory.ner import NonGenerativeExtractor
from fgl.memory.slots import (
    KIND_ACTOR,
    KIND_CONCEPT,
    KIND_EPISODE,
    KIND_PREDICATE,
    KIND_TIME,
    KIND_TYPE,
    SLOT_ORDER,
    Episode,
    EpisodeSegmenter,
    actor_key,
    episode_vertex_id,
    lift_types,
    parse_granularities,
    slot_vertex_id,
    time_buckets,
    types_available,
)
from fgl.memory.bridges import synthesize_bridges
from fgl.memory.calibration import calibrate
from fgl.memory.temporal import annotate_text, resolve_all
from fgl.retrieval.embeddings import Embedder


@dataclass
class SlotIncidence:
    """One (episode, slot) incidence -- L2's analogue of ``Fact``.

    ``text`` is the episode's own verbatim turns, speaker-prefixed, with
    resolved relative dates glossed in brackets. Never a summary: what the
    reader eventually sees is what was actually said.
    """

    text: str
    turn_ids: list[str]
    session_id: str = ""
    session_num: int = 0
    timestamp: str = ""
    date_raw: str = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    meta: dict = field(default_factory=dict)
    state: str = STATE_EMERGENT
    level: int = 1


class SlotIngestor:
    """Builds the episode x slot fatgraph."""

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        embedder: Embedder,
        prompts: PromptLibrary,
        logger: JsonlLogger | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm  # unused unless cfg.bridges.enabled -- see synthesize_bridges
        self.embedder = embedder
        self.prompts = prompts  # unused unless cfg.bridges.enabled
        self.log = logger or NullLogger()
        sl = cfg.slots
        self.extractor = NonGenerativeExtractor(
            model_name=sl.ner_model,
            max_chunk_words=sl.max_chunk_words,
            min_chars=sl.min_concept_chars,
            extract_verbs=True,
            split_persons=True,
        )
        self.segmenter = EpisodeSegmenter(
            min_turns=sl.episode_min_turns,
            max_turns=sl.episode_max_turns,
            cohesion_min=sl.episode_cohesion,
        )
        # Parsed once: the time channel is multi-resolution (year/month/day)
        # rather than a chosen grain, so which levels exist is a property of
        # the ingest and has to be recorded in the graph for the retriever to
        # query the same ones. See fgl.memory.slots, "Time".
        self.granularities = parse_granularities(sl.time_granularities)

    # ------------------------------------------------------------------ api --
    def ingest(self, conv: Conversation) -> tuple[FatGraph, IngestReport]:
        graph = FatGraph()
        resolver = EntityResolver(
            graph, self.embedder, self.cfg.entities, llm=None, prompts=None, logger=self.log
        )
        report = IngestReport(sample_id=conv.sample_id, condition=self.cfg.condition)

        speaker_keys = [actor_key(conv.speaker_a), actor_key(conv.speaker_b)]
        speaker_keys = [k for k in speaker_keys if k]

        n_episodes_total = 0
        for session in conv.sessions:
            n_facts_session = 0
            base_dt = _parse_iso(session.timestamp)
            extractions = self.extractor.extract_many(
                [_ner_input(t) for t in session.turns]
            )
            concept_sets = [
                frozenset(c.text for c in e.candidates) for e in extractions
            ]
            groups = self.segmenter.segment(concept_sets)

            episode_texts: list[str] = []
            episode_vids: list[str] = []

            for group in groups:
                episode, slots = self._build_episode(
                    session, group, extractions, base_dt, speaker_keys,
                    index=n_episodes_total,
                )
                if not slots:
                    # An episode nothing links to is a vertex retrieval can
                    # never reach; the dense channel still sees it through
                    # neighbouring episodes, so dropping it costs nothing and
                    # keeps the degree statistics honest.
                    continue
                ep_vid = self._add_episode_vertex(graph, session, episode)
                n_episodes_total += 1

                for kind, key, surface in slots:
                    slot_vid = self._slot_vertex(graph, resolver, kind, key, surface,
                                                 session.id)
                    if slot_vid is None:
                        continue
                    incidence = SlotIncidence(
                        text=episode.text,
                        turn_ids=list(episode.turn_ids),
                        session_id=session.id,
                        session_num=session.num,
                        timestamp=session.timestamp,
                        date_raw=session.date_time_raw,
                        meta={"slot_kind": kind, "slot_key": key, "surface": surface},
                    )
                    # pos1=pos2=None: append. sigma at the episode therefore
                    # follows the order `slots` was built in (SLOT_ORDER, then
                    # document order) and sigma at the slot stays chronological.
                    edge_id = graph.add_edge(ep_vid, slot_vid, incidence)
                    report.n_facts += 1
                    report.n_edges += 1
                    n_facts_session += 1
                    self.log.log(
                        "insert_slot", edge=edge_id, episode=episode.first_turn_id,
                        kind=kind, key=key, surface=surface, session=session.id,
                    )

                episode_texts.append(episode.text)
                episode_vids.append(ep_vid)

            self._embed(graph, episode_texts, episode_vids)
            graph.check_invariants()
            stats = graph.stats()
            report.per_session.append(
                {
                    "session": session.num,
                    "session_id": session.id,
                    "timestamp": session.timestamp,
                    "n_facts": n_facts_session,
                    "n_episodes": len(episode_vids),
                    "n_incongruent_new": 0,
                    "n_collapses": 0,
                    "n_consolidations": 0,
                    **{k: stats[k] for k in ("V", "E", "F", "C", "genus")},
                    "face_length_hist": stats["face_length_hist"],
                }
            )

        report.graph_stats = graph.stats()
        report.graph_stats["n_episodes"] = n_episodes_total
        report.graph_stats["slot_kinds"] = _kind_histogram(graph)
        report.graph_stats["wordnet_types"] = types_available()
        report.graph_stats["time_granularities"] = list(self.granularities)
        # The calibration is a property of the finished graph, so it is
        # measured here and written into the report: a results directory then
        # records "hub_degree=73, derived, 99th percentile of 412 concept
        # degrees" instead of a literal whose sweep nobody can rerun. The
        # question corpus is not available at ingest time, so the framing
        # stoplist is resolved later, by the retriever.
        report.graph_stats["calibration"] = calibrate(
            self.cfg, graph, concept_matrix=_concept_matrix(graph)
        ).as_dict()

        # L6 only: LLM-synthesised bridge episodes between episodes the slot
        # vocabulary above cannot connect at all. Gated on a single flag,
        # default False, so every existing condition (L1-L5) is byte-for-byte
        # unaffected by this module even existing -- see fgl.memory.bridges
        # for what "bridging" means and why it is safe to bolt on at the end.
        if self.cfg.bridges.enabled:
            bridge_report = synthesize_bridges(
                graph, self.cfg, self.llm, self.embedder, self.prompts, logger=self.log
            )
            report.graph_stats["bridges"] = bridge_report.as_dict()
            report.graph_stats["n_episodes"] += bridge_report.n_linked
            report.graph_stats["slot_kinds"] = _kind_histogram(graph)

        report.llm_usage = self.llm.usage.to_dict() if self.llm else {}
        return graph, report

    # ------------------------------------------------------------ internals --
    def _build_episode(
        self,
        session: Session,
        group: list[int],
        extractions: list,
        base_dt: Optional[datetime],
        speaker_keys: list[str],
        index: int,
    ) -> tuple[Episode, list[tuple[str, str, str]]]:
        """One episode plus its slot list, already in ``sigma`` order.

        Returns ``(episode, [(kind, key, surface), ...])``. The list is sorted
        by :data:`SLOT_ORDER` and, inside a kind, by first mention -- so the
        caller can append incidences straight down it and get the intended
        rotation for free.
        """
        sl = self.cfg.slots
        turns = [session.turns[i] for i in group]

        texts: list[str] = []
        concepts: list[str] = []
        predicates: list[str] = []
        mentioned: list[str] = []
        time_keys: list[str] = []
        speaker_content: dict[str, int] = {}

        for i, turn in zip(group, turns):
            ex = extractions[i]
            resolved = (
                resolve_all(ex.date_spans, base_dt)
                if base_dt is not None and sl.resolve_temporal
                else []
            )
            texts.append(annotate_text(turn.rendered, resolved))
            for r in resolved:
                # every level this date supports, not a chosen one: a question
                # asking by day and a memory dated to the month still meet,
                # and the damping term decides which level does the work
                for bucket in time_buckets(r.resolved, self.granularities):
                    if bucket not in time_keys:
                        time_keys.append(bucket)

            key = actor_key(turn.speaker)
            n_content = len(ex.candidates) + len(ex.verbs)
            if key:
                speaker_content[key] = speaker_content.get(key, 0) + n_content

            for cand in ex.candidates:
                if cand.text and cand.text not in concepts:
                    concepts.append(cand.text)
            for verb in ex.verbs:
                if verb.text not in predicates:
                    predicates.append(verb.text)
            for person in ex.persons:
                pkey = actor_key(person.text)
                if pkey and pkey not in mentioned:
                    mentioned.append(pkey)

        # the session's own date, at every level, in front of the resolved
        # relatives: it is the one date every episode certainly has
        for i, bucket in enumerate(time_buckets(base_dt, self.granularities)):
            if bucket in time_keys:
                time_keys.remove(bucket)
            time_keys.insert(i, bucket)

        episode = Episode(
            index=index,
            turn_ids=[t.dia_id for t in turns],
            text="\n".join(texts),
            turn_texts=list(texts),
            speakers=list(dict.fromkeys(t.speaker for t in turns)),
            speaker_content=speaker_content,
            mentioned_actors=mentioned,
        )

        # Actors: everyone who *spoke content* here, then everyone merely
        # named. Both get a vertex -- "What people has Maria met?" is answered
        # by the named ones -- but the retriever weights them differently
        # (see SlotRetriever._actor_weight), because an episode is not equally
        # *about* the person talking and the person greeted.
        actors: list[str] = [k for k, n in speaker_content.items() if n > 0]
        for k in mentioned:
            if k not in actors:
                actors.append(k)

        slots: list[tuple[str, str, str]] = []
        slots += [(KIND_ACTOR, k, k) for k in actors]
        slots += [(KIND_PREDICATE, p, p) for p in predicates[: sl.max_predicates]]
        slots += [(KIND_CONCEPT, c, c) for c in concepts[: sl.max_concepts]]
        if sl.lift_types:
            types: list[str] = []
            for c in concepts[: sl.max_concepts]:
                for t in lift_types(c, sl.max_types_per_concept):
                    if t not in types and t not in concepts:
                        types.append(t)
            slots += [(KIND_TYPE, t, t) for t in types[: sl.max_types]]
        slots += [(KIND_TIME, t, t) for t in time_keys]

        # Defensive, not cosmetic: SLOT_ORDER *is* the rotation, so an
        # accidental reordering above would silently change which corners
        # exist. Sorting by it here makes the invariant hold by construction
        # rather than by the order the blocks happen to be written in.
        rank = {kind: i for i, kind in enumerate(SLOT_ORDER)}
        slots.sort(key=lambda s: rank[s[0]])
        return episode, slots

    def _add_episode_vertex(
        self, graph: FatGraph, session: Session, episode: Episode
    ) -> str:
        vid = episode_vertex_id(episode.first_turn_id)
        if vid in graph.vertices:
            return vid
        graph.add_vertex(
            name=episode.first_turn_id,
            vertex_id=vid,
            meta={
                "kind": KIND_EPISODE,
                "turn_ids": list(episode.turn_ids),
                # kept per turn, not only joined: the retriever scores whole
                # episodes but *renders* single turns, so it needs to price
                # them one at a time (see SlotRetriever._emit).
                "turn_texts": list(episode.turn_texts),
                "speakers": list(episode.speakers),
                "speaker_content": dict(episode.speaker_content),
                "mentioned_actors": list(episode.mentioned_actors),
                "session_id": session.id,
                "session_num": session.num,
                "timestamp": session.timestamp,
                "date_raw": session.date_time_raw,
            },
        )
        return vid

    def _slot_vertex(
        self,
        graph: FatGraph,
        resolver: EntityResolver,
        kind: str,
        key: str,
        surface: str,
        session_id: str,
    ) -> Optional[str]:
        """Vertex for one slot, created on first sight.

        Concepts go through :class:`EntityResolver` (embedding merge, exactly
        as L1 resolves its entities) so "paintings"/"a painting" land on one
        vertex and the comparison against L1 is not confounded by a different
        resolution policy. Every other kind is already a canonical lemma, key
        or bucket, so it is keyed literally -- cheaper, and it keeps two
        ingests of the same conversation vertex-for-vertex comparable.
        """
        if kind == KIND_CONCEPT:
            if not key:
                return None
            res = resolver.resolve(key, session_id)
            vx = graph.vertices[res.vertex_id]
            vx.meta.setdefault("kind", KIND_CONCEPT)
            return res.vertex_id

        vid = slot_vertex_id(kind, key)
        if vid not in graph.vertices:
            graph.add_vertex(name=surface or key, vertex_id=vid,
                             meta={"kind": kind, "key": key})
        return vid

    def _embed(
        self, graph: FatGraph, texts: list[str], vids: list[str]
    ) -> None:
        """One vector per episode, fanned out to its half-edges.

        Mirrors L1: embed each distinct text once, then attach the vector both
        to the episode vertex (so the dense channel is a one-row-per-episode
        index) and to every half-edge that carries that text.
        """
        if not texts:
            return
        vectors = self.embedder.encode(texts)
        by_text = {t: v for t, v in zip(texts, vectors)}
        for vid, vec in zip(vids, vectors):
            graph.vertices[vid].embedding = vec
        for he in graph.H.values():
            if he.embedding is None and he.text in by_text:
                he.embedding = by_text[he.text]


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _ner_input(turn: Turn) -> str:
    """Turn text for the parser, *with* the speaker prefix.

    The opposite of L1's choice, and deliberately so. L1 strips the prefix
    because tagging the speaker as a PERSON on every turn recreates the hub it
    exists to avoid. Here the speaker IS a slot kind of its own, so the prefix
    is signal: it is what lets ``split_persons`` attribute the turn without a
    second pass. The image caption stays in either way -- it is where "sunset"
    comes from on a photo-only turn.
    """
    return f"{turn.speaker}: {turn.text} {turn.img_caption}".strip()


def _concept_matrix(graph: FatGraph) -> Optional[np.ndarray]:
    """Unit-normalised concept embeddings, for the concept-link calibration.

    Built from the graph rather than passed in, so the number recorded in the
    report is the one a retriever loading this graph back would compute.
    """
    rows = [
        vx.embedding
        for vx in graph.vertices.values()
        if vx.meta.get("kind", KIND_CONCEPT) == KIND_CONCEPT
        and vx.embedding is not None
    ]
    if not rows:
        return None
    mat = np.vstack(rows).astype(float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _kind_histogram(graph: FatGraph) -> dict[str, int]:
    out: dict[str, int] = {}
    for vx in graph.vertices.values():
        kind = vx.meta.get("kind", KIND_CONCEPT)
        out[kind] = out.get(kind, 0) + 1
    return dict(sorted(out.items()))


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None
