"""Rebuildable fact/event views over CLIO's authoritative stores."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

import numpy as np

from fgl.clio.types import Interval, Operation
from fgl.clio2.model import LedgerFact, MemoryEvent, QueryConstraints
from fgl.retrieval.embeddings import Embedder, cosine

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|i'm|i've|i'd|my|mine|we|we're|we've|we'd|our|ours)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
}


def content_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    }


def normalized_value(value: str) -> str:
    return " ".join(_TOKEN_RE.findall((value or "").casefold()))


def _attributed_subject_id(fact: LedgerFact) -> str:
    """Resolve the actor of a fact without modifying its extracted subject."""

    if not fact.speaker_id:
        return fact.subject_id
    if _FIRST_PERSON_RE.search(fact.span):
        return fact.speaker_id
    subject_name = normalized_value(fact.subject_name)
    span = normalized_value(fact.span)
    if (
        subject_name in {"family", "my family", "our family", "we"}
        and _FIRST_PERSON_RE.search(fact.episode_text)
    ):
        return fact.speaker_id
    if subject_name and subject_name in span:
        return fact.subject_id
    if _FIRST_PERSON_RE.search(fact.episode_text):
        return fact.speaker_id
    return fact.subject_id


def _overlaps(a: Interval, b: Interval | None) -> bool:
    return b is None or a.overlaps(b)


class SemanticLedger:
    """A read-only semantic ledger materialized from CLIO stores.

    It is rebuilt rather than incrementally patched.  This is deliberate:
    the ledger is a view, not a second source of truth, and a few hundred or
    thousand propositions are cheap compared with debugging index drift after
    folds, retractions, and replay.
    """

    EVENT_RELATIONS = frozenset(
        {"attended", "practices", "plans_to", "created", "researches"}
    )

    def __init__(self, memory, include_staged: bool = True):
        self.memory = memory
        self.include_staged = include_staged
        self.facts: list[LedgerFact] = []
        self.events: list[MemoryEvent] = []
        self.by_subject: dict[str, list[LedgerFact]] = defaultdict(list)
        self.by_relation: dict[str, list[LedgerFact]] = defaultdict(list)
        self.by_episode: dict[str, list[LedgerFact]] = defaultdict(list)
        self._build()

    def _entity(self, entity_id: str):
        resolved = self.memory.graph.resolve_entity(entity_id)
        return self.memory.graph.get_entity(resolved)

    def _build(self) -> None:
        episodes = {episode.id: episode for episode in self.memory.log.all()}
        edge_by_proposition = {}
        for edge in self.memory.graph.all_edges():
            for proposition_id in edge.provenance:
                edge_by_proposition[proposition_id] = edge
        for proposition in self.memory.staging.all():
            if proposition.status == "rejected":
                continue
            if proposition.status != "promoted" and not self.include_staged:
                continue
            if proposition.t_valid is None or proposition.episode_id not in episodes:
                continue
            if proposition.operation not in (Operation.ASSERT, Operation.REASSERT):
                # CLOSE and RETRACT mutate the bitemporal row; they are not a
                # second positive fact to retrieve as an answer value.
                continue
            try:
                subject = self._entity(proposition.subject_id)
                object_ = self._entity(proposition.object_id)
            except KeyError:
                continue
            episode = episodes[proposition.episode_id]
            speaker = self.memory.graph.find_entity_by_name_any_type(episode.speaker)
            speaker_id = ""
            speaker_name = episode.speaker
            if speaker is not None:
                speaker_id = self.memory.graph.resolve_entity(speaker.id)
                speaker_name = self.memory.graph.get_entity(speaker_id).canonical_name
            edge = edge_by_proposition.get(proposition.id)
            fact = LedgerFact(
                proposition_id=proposition.id,
                subject_id=subject.id,
                subject_name=subject.canonical_name,
                subject_type=subject.type,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                relation=proposition.relation,
                object_id=object_.id,
                object_name=object_.canonical_name,
                object_type=object_.type,
                t_valid=edge.t_valid if edge is not None else proposition.t_valid,
                t_tx=edge.t_tx if edge is not None else proposition.t_tx,
                episode_id=episode.id,
                episode_ts=episode.ts_ingest,
                span=proposition.span,
                episode_text=episode.text,
                confidence=proposition.confidence,
                polarity=edge.polarity if edge is not None else proposition.polarity,
                promoted=proposition.status == "promoted",
                unanchored=edge.unanchored if edge is not None else proposition.unanchored,
            )
            self.facts.append(fact)
            self.by_subject[fact.subject_id].append(fact)
            self.by_relation[fact.relation].append(fact)
            self.by_episode[fact.episode_id].append(fact)
        self.events = self._build_events(episodes)

    def _build_events(self, episodes) -> list[MemoryEvent]:
        events = []
        for episode_id, facts in self.by_episode.items():
            event_facts = [fact for fact in facts if fact.relation in self.EVENT_RELATIONS]
            if not event_facts:
                continue
            participants = dict.fromkeys(
                fact.subject_id for fact in facts if fact.subject_type == "Person"
            )
            for fact in facts:
                attributed_id = _attributed_subject_id(fact)
                if attributed_id != fact.subject_id:
                    participants.setdefault(attributed_id, None)
            values = dict.fromkeys(fact.object_id for fact in event_facts)
            events.append(
                MemoryEvent(
                    id=f"evt:{episode_id}",
                    episode_id=episode_id,
                    ts=episodes[episode_id].ts_ingest,
                    event_types=tuple(dict.fromkeys(f.relation for f in event_facts)),
                    participant_ids=tuple(participants),
                    participant_names=tuple(
                        self._entity(entity_id).canonical_name for entity_id in participants
                    ),
                    value_ids=tuple(values),
                    value_names=tuple(
                        self._entity(entity_id).canonical_name for entity_id in values
                    ),
                    proposition_ids=tuple(f.proposition_id for f in facts),
                    text=episodes[episode_id].text,
                )
            )
        return events

    def resolve_subjects(self, names: tuple[str, ...]) -> tuple[str, ...]:
        resolved: list[str] = []
        for name in names:
            exact = self.memory.graph.find_entity_by_name_any_type(name)
            if exact is not None:
                entity_id = self.memory.graph.resolve_entity(exact.id)
                if entity_id not in resolved:
                    resolved.append(entity_id)
                continue
            for entity, score in self.memory.entity_index.search_scored(name, k=3):
                if score < 0.35:
                    continue
                entity_id = self.memory.graph.resolve_entity(entity.id)
                if entity_id not in resolved:
                    resolved.append(entity_id)
                break
        return tuple(resolved)


class FactIndex:
    """Hybrid retrieval over ledger facts, with structural filters first."""

    def __init__(self, ledger: SemanticLedger, embedder: Embedder):
        self.ledger = ledger
        self.embedder = embedder
        self._texts = [fact.render() for fact in ledger.facts]
        self._tokens = [content_tokens(text) for text in self._texts]
        self._vectors: np.ndarray | None = (
            embedder.encode(self._texts) if self._texts else None
        )

    def search(
        self,
        query: str,
        *,
        subject_ids: tuple[str, ...] = (),
        relations: tuple[str, ...] = (),
        constraints: QueryConstraints | None = None,
        limit: int | None = 40,
        tx_point: datetime | None = None,
    ) -> list[tuple[LedgerFact, float]]:
        if not self.ledger.facts:
            return []
        constraints = constraints or QueryConstraints()
        tx_point = tx_point or datetime.now()
        q_tokens = content_tokens(query)
        term_tokens = content_tokens(" ".join(constraints.terms))
        q_vec = self.embedder.encode_one(query)
        relation_set = set(relations)
        subject_set = set(subject_ids)
        object_types = set(constraints.object_types)
        scored: list[tuple[LedgerFact, float]] = []
        for index, fact in enumerate(self.ledger.facts):
            if subject_set:
                # First-person evidence is attributed to the speaker, even if
                # extraction produced a collective subject ("my family") or
                # attached the pronoun to the wrong conversational person.
                # Third-person statements keep their explicitly extracted
                # subject.  This is a derived read-time repair: source facts
                # and provenance remain untouched.
                attributed_id = _attributed_subject_id(fact)
                if attributed_id not in subject_set:
                    continue
            if relation_set and fact.relation not in relation_set:
                continue
            if object_types and fact.object_type not in object_types:
                continue
            if constraints.current_only and not fact.t_valid.contains(tx_point):
                continue
            if not _overlaps(fact.t_valid, constraints.interval):
                continue
            if not fact.t_tx.contains(tx_point):
                continue
            lexical = len(q_tokens & self._tokens[index]) / max(1, len(q_tokens))
            term_coverage = len(term_tokens & self._tokens[index]) / max(
                1, len(term_tokens)
            )
            dense = float(cosine(q_vec, self._vectors[index]))
            structural = 0.0
            if subject_set:
                structural += 0.22
            if relation_set:
                structural += 0.28
            if object_types:
                structural += 0.08
            score = (
                0.34 * max(0.0, dense)
                + 0.24 * lexical
                + 0.20 * term_coverage
                + structural
                + (0.04 if fact.promoted else 0.0)
                + 0.04 * fact.confidence
            )
            scored.append((fact, score))
        scored.sort(
            key=lambda pair: (
                -pair[1],
                -pair[0].episode_ts.timestamp(),
                pair[0].object_name,
            )
        )
        return scored if limit is None else scored[:limit]


__all__ = ["FactIndex", "SemanticLedger", "content_tokens", "normalized_value"]
