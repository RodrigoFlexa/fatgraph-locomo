"""Synchronous per-turn ingestion (spec section 6). One LLM call, never
blocks on consolidation -- consolidation is a separate, asynchronous call
(:func:`fgl.clio.consolidate.consolidate`) a caller triggers on its own
schedule (end of session, a pending-count threshold, idle time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from fgl.clio.catalog import Catalog
from fgl.clio.confidence import compute_confidence
from fgl.clio.graph.store import GraphStore
from fgl.clio.index import EntityIndex
from fgl.clio.ingest.context import build_extraction_context
from fgl.clio.ingest.extractor import extract_propositions
from fgl.clio.ingest.validate import validate_and_bind
from fgl.clio.log.mentions import MentionStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.temporal.resolver import resolve_time
from fgl.clio.types import Episode, EvidenceKind, Interval, Operation, Proposition
from fgl.clio.unmapped import UnmappedEntry, UnmappedQueue
from fgl.llm.client import LLMClient
from fgl.llm.prompts import PromptLibrary


@dataclass
class IngestResult:
    episode: Episode
    propositions: list[Proposition] = field(default_factory=list)
    unmapped: list[UnmappedEntry] = field(default_factory=list)


def _mention_surface(ref: str, graph: GraphStore) -> str:
    if ref.startswith("new:"):
        return ref[len("new:") :]
    try:
        return graph.get_entity(ref).canonical_name
    except KeyError:
        return ref


def ingest_turn(
    text: str,
    speaker: str,
    session_id: str,
    ts: datetime,
    log: LogStore,
    graph: GraphStore,
    staging: StagingStore,
    mentions: MentionStore,
    entity_index: EntityIndex,
    catalog: Catalog,
    llm: LLMClient,
    prompts: PromptLibrary,
    unmapped_queue: UnmappedQueue,
    coref_window: int = 3,
    max_candidates: int = 20,
) -> IngestResult:
    """Spec 6.1's eight steps: append the episode, build context, extract,
    validate, resolve time, score confidence, record mentions, stage.

    ``entity_index`` is rebuilt from the graph at the top of every call --
    cheap at this scale (see :class:`fgl.clio.index.EntityIndex`) and
    guarantees the candidates offered to the extractor reflect every fold
    and every write from every earlier turn, including ones from a
    consolidation run that happened since the last call.
    """
    episode = log.append(session_id=session_id, speaker=speaker, text=text, ts_ingest=ts)
    entity_index.rebuild(graph)

    context = build_extraction_context(
        episode, log, entity_index, catalog, coref_window, max_candidates
    )
    raw = extract_propositions(llm, prompts, context)
    valid, unmapped_raw = validate_and_bind(raw, episode, graph, catalog)

    unmapped: list[UnmappedEntry] = []
    for item in unmapped_raw:
        unmapped.append(
            unmapped_queue.append(
                suggested_relation=item.get("suggested_relation"),
                subject_ref=item.get("subject_id", ""),
                object_ref=item.get("object_id", ""),
                span=item.get("span", ""),
                episode_id=episode.id,
            )
        )

    props: list[Proposition] = []
    for i, item in enumerate(valid):
        relation = item["relation"]
        spec = catalog[relation]
        time_expression = item.get("time_expression")
        t_valid, tconf = resolve_time(time_expression, ts, spec)
        evidence_kind = EvidenceKind(item["evidence_kind"])
        confidence = compute_confidence(evidence_kind, tconf)
        props.append(
            Proposition(
                id=f"{episode.id}_p{i}",
                subject_id=item["subject_id"],
                relation=relation,
                object_id=item["object_id"],
                operation=Operation(item["operation"]),
                polarity=bool(item.get("polarity", True)),
                time_expression=time_expression,
                t_valid=t_valid,
                t_tx=Interval(ts, None),
                evidence_kind=evidence_kind,
                confidence=confidence,
                span=item.get("span", ""),
                episode_id=episode.id,
            )
        )

    # Mentions record the OBJECT of each proposition -- "what was talked
    # about" -- not the subject, which on most turns is just the speaker
    # and would otherwise inflate their own count on every single fact
    # they are ever the subject of (spec 3.2's own example counts a
    # TOPIC's mentions, not a person's). Deduplicated by surface WITHIN
    # this turn: "how many times was climbing mentioned" counts OCCASIONS
    # it came up, not how many separate propositions one sentence happened
    # to yield about it -- one turn stating that two different people both
    # like climbing is one mention of climbing, not two.
    seen_surfaces: set[str] = set()
    for p in props:
        surface = _mention_surface(p.object_id, graph)
        if surface in seen_surfaces:
            continue
        seen_surfaces.add(surface)
        mentions.append(episode_id=episode.id, surface=surface, ts=ts)

    staging.insert(props)
    return IngestResult(episode=episode, propositions=props, unmapped=unmapped)
