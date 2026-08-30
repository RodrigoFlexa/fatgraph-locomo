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
from fgl.clio.ingest.validate import Rejection, validate_and_bind
from fgl.clio.log.mentions import MentionStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.temporal.resolver import default_for_volatility, resolve_time
from fgl.clio.types import Episode, EvidenceKind, Interval, Operation, Proposition
from fgl.clio.unmapped import UnmappedEntry, UnmappedQueue
from fgl.llm.client import LLMClient
from fgl.llm.prompts import PromptLibrary


@dataclass
class IngestResult:
    episode: Episode
    propositions: list[Proposition] = field(default_factory=list)
    unmapped: list[UnmappedEntry] = field(default_factory=list)
    #: how many items the model actually produced, before any filtering.
    #: `propositions == []` with `raw_count == 0` means the model said
    #: nothing; with `raw_count > 0` it means everything it said was
    #: refused or unmapped, which is a completely different problem.
    raw_count: int = 0
    #: what was refused and why (spec 6.5's "log, never accept silently")
    rejected: list[Rejection] = field(default_factory=list)
    #: spans that were not verbatim and got downgraded to `contextual`,
    #: i.e. confidence 0.40 -- below tau_promote, needing three
    #: independent episodes to ever reach the graph
    span_downgrades: int = 0

    def summary(self) -> str:
        """One line for a per-turn log: what came in, what survived."""
        parts = [f"raw={self.raw_count}", f"kept={len(self.propositions)}"]
        if self.unmapped:
            parts.append(f"unmapped={len(self.unmapped)}")
        if self.rejected:
            counts: dict[str, int] = {}
            for r in self.rejected:
                counts[r.reason] = counts.get(r.reason, 0) + 1
            parts.append(
                "rejected=" + ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            )
        if self.span_downgrades:
            parts.append(f"span_downgraded={self.span_downgrades}")
        return " ".join(parts)


def _mention_surface(ref: str | None, graph: GraphStore) -> str:
    """The human-readable name behind an entity reference, for the mention
    table. Tolerates a missing reference (returns ""): an UNMAPPED item
    legitimately has no object id -- spec 4.4's own example carries
    ``object_surface`` instead -- and a model is free to send an explicit
    ``null`` for a field it has nothing to put in.
    """
    if not ref:
        return ""
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
    extraction_max_tokens: int = 2000,
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
    raw = extract_propositions(llm, prompts, context, extraction_max_tokens)
    validation = validate_and_bind(raw, episode, graph, catalog)
    valid, unmapped_raw = validation.valid, validation.unmapped

    # `.get(key, "")` is NOT enough here: a key PRESENT with an explicit
    # JSON null returns None, not the default, and an UNMAPPED item is
    # exactly where that happens -- the relation has no catalog entry, so
    # the model often has no id to give and sends null, or names the thing
    # in `object_surface` the way spec 4.4's own example does. Both shapes
    # normalise to a plain string here so nothing downstream has to know.
    unmapped: list[UnmappedEntry] = []
    for item in unmapped_raw:
        unmapped.append(
            unmapped_queue.append(
                suggested_relation=item.get("suggested_relation"),
                subject_ref=item.get("subject_id") or item.get("subject_surface") or "",
                object_ref=item.get("object_id") or item.get("object_surface") or "",
                span=item.get("span") or "",
                episode_id=episode.id,
            )
        )

    props: list[Proposition] = []
    for i, item in enumerate(valid):
        relation = item["relation"]
        spec = catalog[relation]
        time_expression = item.get("time_expression")
        t_valid, tconf = resolve_time(time_expression, ts, spec)
        # An unresolvable time expression is NOT "true for all time".
        # Writing `Interval(None, None)` for it (what the graph used to
        # receive) makes the fact intersect every window a query could
        # ask about, so a trail through it can never die of
        # `empty_temporal_window` -- P4 and false-premise detection both
        # stop working, silently. Spec 5.4 wants the proposition kept and
        # flagged instead, reachable by transaction order but not by a
        # validity restriction. The volatility default gives it a
        # plausible shape; `unanchored` records that no date was actually
        # read from the text. Common English phrasings really do land
        # here -- "each day", "over the weekend", "a couple of months
        # ago" all resolve to None.
        unanchored = t_valid is None
        if unanchored:
            t_valid = default_for_volatility(ts, spec)
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
                unanchored=unanchored,
                subject_ref=item["subject_id"],
                object_ref=item["object_id"],
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
    #
    # UNMAPPED objects are recorded too. They never reach the graph (spec
    # 4.4) and that is correct, but a turn whose every proposition came
    # back UNMAPPED still MENTIONED the things it named -- excluding them
    # made "how many times did X come up" undercount exactly the turns
    # the catalog does not yet cover, which is the opposite of what a raw
    # occurrence table is for.
    seen_surfaces: set[str] = set()
    for p in props:
        surface = _mention_surface(p.object_id, graph)
        if surface in seen_surfaces:
            continue
        seen_surfaces.add(surface)
        mentions.append(episode_id=episode.id, surface=surface, ts=ts, proposition_id=p.id)
    for entry in unmapped:
        surface = _mention_surface(entry.object_ref, graph)
        if not surface or surface in seen_surfaces:
            continue
        seen_surfaces.add(surface)
        mentions.append(episode_id=episode.id, surface=surface, ts=ts)

    staging.insert(props)
    return IngestResult(
        episode=episode,
        propositions=props,
        unmapped=unmapped,
        raw_count=len(raw),
        rejected=validation.rejected,
        span_downgrades=validation.span_downgrades,
    )
