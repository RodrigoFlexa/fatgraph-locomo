"""Bipartite (turn x entity) ingestion -- condition L1.

Every other condition's memory is built the same way: an LLM reads a session
and emits triples ``(entity_1, relation, entity_2, fact_text)``, and each
triple becomes an edge between two CONTENT vertices. Measured consequence
(``docs/COERENCIA.md``, ``T1_topical.yaml``): 81% of vertices end up with
degree 1 and the two speakers are the two highest-degree vertices in every
conversation, because a generative extractor invents new phrasing for the
"other" entity almost every time, so nothing recurs for entity resolution to
merge. G4/G5/G6/G9/G10 were all tested on that substrate and could not do
anything with it -- there was no real topology to walk.

This module does not extract triples. It builds the graph the way an index
would: a vertex per TURN, a vertex per canonical ENTITY, and an edge for
every "entity E was mentioned in turn T" -- exactly what
:mod:`fgl.memory.ner` finds deterministically, with zero LLM calls. This is
the same move Zero-Mem (arXiv:2607.29377) makes for its entity-context graph,
adapted to ribbon-graph structure: because incidence, not inference, is what
gets recorded, an entity's degree is now a real signal instead of an
artefact of extraction phrasing -- degree 1 means "mentioned once", not
"the extractor never reused a phrase for this".

sigma falls out of processing order, no policy object needed
--------------------------------------------------------------
G1's ``SigmaPolicy`` exists because facts inside a session are not
necessarily processed in a chronologically or texually meaningful order.
Here they are, by construction:

* ``sigma`` at a TURN vertex should be reading order (which entity was
  mentioned first). :mod:`fgl.memory.ner` already returns candidates in
  document order, so appending each incidence as it is found gives reading
  order for free.
* ``sigma`` at an ENTITY vertex should be chronological (when was this
  entity mentioned, across the whole conversation). ``conv.sessions`` is
  already sorted chronologically and ``session.turns`` is transcript order,
  so appending as sessions and turns are visited in their natural order
  gives chronological order for free.

So every ``add_edge`` call below passes ``pos1=None, pos2=None`` (append) and
both the intra-turn and cross-session orderings come out right without any
positional bookkeeping -- unlike ``SigmaTime``, which has to search for where
"now" fits among out-of-order timestamps.

The speaker never becomes a vertex, same invariant as G1/T1: excluded here
by name match against ``conv.speaker_a``/``conv.speaker_b`` rather than by
prompt instruction, so it cannot fail to take the way a generative
extractor's instruction can (measured: T1's prompt asks for the same thing
and does not fully get it). A person's name mentioned in the THIRD person
("Hey Caroline, ...") is excluded the same way -- greetings recur constantly
and would recreate a milder version of the same hub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from fgl.config import Config
from fgl.core import STATE_EMERGENT, FatGraph
from fgl.data.locomo import Conversation, Session, Turn
from fgl.llm import LLMClient
from fgl.llm.prompts import PromptLibrary
from fgl.logging_utils import JsonlLogger, NullLogger
from fgl.memory.entities import EntityResolver, normalize_name
from fgl.memory.ingest import IngestReport
from fgl.memory.ner import NonGenerativeExtractor
from fgl.memory.temporal import annotate_text, resolve_all
from fgl.retrieval.embeddings import Embedder


@dataclass
class Incidence:
    """One (turn, entity) incidence -- the bipartite analogue of ``Fact``.

    ``text`` is the turn's own rendered text (speaker-prefixed, caption
    included, relative dates annotated with their resolution) -- never a
    generated summary. Provenance-preserving by construction: what the reader
    eventually sees is what was actually said, not a paraphrase of it.
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


class BipartiteIngestor:
    """Builds a turn x entity fatgraph. Same public contract as
    :class:`fgl.memory.ingest.Ingestor`: ``ingest(conv) -> (FatGraph,
    IngestReport)``, so :class:`fgl.pipeline.Runner` can dispatch to either
    without a special case beyond the one ``if`` on ``cfg.ingest.mode``.
    """

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        embedder: Embedder,
        prompts: PromptLibrary,
        logger: JsonlLogger | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm  # unused: zero LLM calls in this ingest path
        self.embedder = embedder
        self.prompts = prompts  # unused, kept for interface parity
        self.log = logger or NullLogger()
        bp = cfg.bipartite
        self.extractor = NonGenerativeExtractor(
            model_name=bp.ner_model,
            max_chunk_words=bp.max_chunk_words,
            min_chars=bp.min_entity_chars,
        )

    def ingest(self, conv: Conversation) -> tuple[FatGraph, IngestReport]:
        graph = FatGraph()
        resolver = EntityResolver(
            graph, self.embedder, self.cfg.entities, llm=None, prompts=None, logger=self.log
        )
        report = IngestReport(sample_id=conv.sample_id, condition=self.cfg.condition)

        excluded_exact = {normalize_name(conv.speaker_a), normalize_name(conv.speaker_b)}
        excluded_exact.discard("")

        for session in conv.sessions:
            base_dt = _parse_iso(session.timestamp)
            n_incidences_session = 0
            n_turn_vertices_session = 0

            texts = [_ner_input(t) for t in session.turns]
            extractions = self.extractor.extract_many(texts)

            turn_texts: list[str] = []
            turn_vids: list[str] = []

            for turn, extraction in zip(session.turns, extractions):
                candidates = [
                    c for c in extraction.candidates
                    if not _mentions_speaker(c.text, excluded_exact)
                ]
                if not candidates:
                    # nothing to link -- small talk, greetings, questions.
                    # No turn vertex either: a vertex nothing ever points to
                    # is dead weight, and the extractor already excludes it
                    # from graph structure the same way v1's prompt was told
                    # to ("do NOT extract small talk").
                    continue

                resolved_dates = (
                    resolve_all(extraction.date_spans, base_dt)
                    if base_dt is not None and self.cfg.bipartite.resolve_temporal
                    else []
                )
                turn_text = annotate_text(turn.rendered, resolved_dates)

                turn_vid = _turn_vertex_id(turn)
                if turn_vid not in graph.vertices:
                    graph.add_vertex(
                        name=turn.dia_id,
                        vertex_id=turn_vid,
                        meta={
                            "kind": "turn",
                            "speaker": turn.speaker,
                            "session_id": session.id,
                            "session_num": session.num,
                            "timestamp": session.timestamp,
                            "date_raw": session.date_time_raw,
                            "resolved_dates": [r.resolved_date for r in resolved_dates],
                        },
                    )
                    n_turn_vertices_session += 1

                turn_texts.append(turn_text)
                turn_vids.append(turn_vid)

                for cand in candidates:
                    res = resolver.resolve(cand.text, session.id)
                    incidence = Incidence(
                        text=turn_text,
                        turn_ids=[turn.dia_id],
                        session_id=session.id,
                        session_num=session.num,
                        timestamp=session.timestamp,
                        date_raw=session.date_time_raw,
                        meta={"entity_surface": cand.raw, "entity_label": cand.label},
                    )
                    edge_id = graph.add_edge(turn_vid, res.vertex_id, incidence)
                    n_incidences_session += 1
                    self.log.log(
                        "insert_incidence", edge=edge_id, turn=turn.dia_id,
                        vertex=res.vertex_id, entity=cand.text, label=cand.label,
                        session=session.id, entity_new=res.created,
                    )

            # embed every distinct turn text once, then fan the vector out to
            # every half-edge that shares it (both sides of every incidence
            # from that turn) -- mirrors how Ingestor embeds `fact_text` once
            # per fact rather than once per half-edge.
            if turn_texts:
                vectors = self.embedder.encode(turn_texts)
                text_to_vec = {t: v for t, v in zip(turn_texts, vectors)}
                for hid, he in graph.H.items():
                    if he.embedding is None and he.text in text_to_vec:
                        he.embedding = text_to_vec[he.text]
                # also on the turn VERTEX itself, so retrieval can build a
                # one-row-per-turn dense index without fishing through
                # half-edges (there are up to N of them per turn, one per
                # entity, all carrying the identical vector).
                for vid, vec in zip(turn_vids, vectors):
                    graph.vertices[vid].embedding = vec

            report.n_facts += n_incidences_session
            report.n_edges += n_incidences_session
            graph.check_invariants()
            stats = graph.stats()
            report.per_session.append(
                {
                    "session": session.num,
                    "session_id": session.id,
                    "timestamp": session.timestamp,
                    "n_facts": n_incidences_session,
                    "n_turn_vertices": n_turn_vertices_session,
                    "n_incongruent_new": 0,
                    "n_collapses": 0,
                    "n_consolidations": 0,
                    **{k: stats[k] for k in ("V", "E", "F", "C", "genus")},
                    "face_length_hist": stats["face_length_hist"],
                }
            )

        report.graph_stats = graph.stats()
        report.llm_usage = self.llm.usage.to_dict() if self.llm else {}
        return graph, report


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _is_speaker_mention(candidate_norm: str, speaker_norms: set[str]) -> bool:
    """Exact match, or a short nickname of one of the two speakers.

    Measured gap this closes: excluding only the exact conversation name
    ("Melanie") still let a nickname used constantly as a term of address
    ("Mel") through, and it becomes a degree-58 hub on its own (`fgl ingest
    L1 -n 10`, conv-26) -- a milder replay of the exact problem this design
    exists to avoid, just under a different spelling. A prefix match in
    either direction catches the common case (Mel/Melanie, Cal/Calvin,
    Deb/Deborah) without a hand-maintained nickname table; the length floor
    keeps it from firing on short unrelated words.
    """
    if not candidate_norm or len(candidate_norm) < 3:
        return candidate_norm in speaker_norms
    if candidate_norm in speaker_norms:
        return True
    return any(
        s.startswith(candidate_norm) or candidate_norm.startswith(s)
        for s in speaker_norms
        if len(s) >= 3
    )


def _mentions_speaker(cand_text: str, speaker_norms: set[str]) -> bool:
    """Whole-string check plus a per-word check for multi-token candidates.

    Needed once :func:`fgl.memory.ner._compound_fallback` started recovering
    real multi-word compounds ("adoption agencies"): the same fallback also
    recovers vocative fragments like "Thanks Nate" (an interjection
    mistagged NOUN, compound-attached to the addressee's name) as a single
    two-token span. The whole-string prefix check in `_is_speaker_mention`
    does not fire on "thanks nate" against speaker norm "nate" -- neither
    string is a prefix of the other -- so without this, every "Thanks
    <name>"/"Hey <name>" turn in the conversation would recreate a small
    version of the exact vocative-hub problem the speaker exclusion exists to
    prevent. Checking each word catches it without rejecting genuine
    multi-word topics: none of "adoption"/"agencies", "ice"/"cream",
    "coconut"/"milk" match a speaker name or nickname.
    """
    norm = normalize_name(cand_text)
    if _is_speaker_mention(norm, speaker_norms):
        return True
    words = norm.split()
    if len(words) < 2:
        return False
    return any(_is_speaker_mention(w, speaker_norms) for w in words)


def _turn_vertex_id(turn: Turn) -> str:
    # `dia_id` ("D3:12") is already unique within a conversation and stable
    # across runs, so it doubles as the vertex id -- no separate counter
    # needed, and graphs from two ingests of the same conversation are
    # trivially comparable turn-vertex by turn-vertex.
    return f"turn:{turn.dia_id}"


def _ner_input(turn: Turn) -> str:
    # No speaker prefix here (that would tag the speaker's own name as a
    # PERSON entity on every single turn -- the exact hub this design exists
    # to avoid); the caption IS included, since that is where "sunset" comes
    # from on an image-only turn (see docs/DECISIONS.md L1).
    return f"{turn.text} {turn.img_caption}".strip()


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None
