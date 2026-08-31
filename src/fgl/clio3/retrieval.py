"""Schema-free retrieval and graph expansion for CLIO3."""

from __future__ import annotations

import re

from fgl.clio3.model import MemoryQuery, RetrievalResult
from fgl.retrieval.embeddings import cosine

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) > 1
        and token.casefold()
        not in {"the", "and", "for", "with", "that", "this", "what", "when", "where"}
    }


class OpenGraphRetriever:
    def __init__(self, memory):
        self.memory = memory
        self.store = memory.clio3_store
        self.embedder = memory.entity_index.embedder
        self._entities = [entity for entity in self.store.entities() if not entity.merged_into]
        self._records = self.store.records()
        self._entity_vectors = self.embedder.encode(
            [" ".join([entity.name, entity.type, *entity.aliases]) for entity in self._entities]
        ) if self._entities else None
        self._record_texts = [record.render(self.store._entities) for record in self._records]
        self._record_vectors = self.embedder.encode(self._record_texts) if self._records else None

    def entity_candidates(self, query: str, limit: int = 10):
        if not self._entities:
            return []
        vector = self.embedder.encode_one(query)
        low = query.casefold()
        rows = []
        for index, entity in enumerate(self._entities):
            exact = 1.0 if any(
                name.casefold() in low for name in (entity.name, *entity.aliases) if name
            ) else 0.0
            rows.append((max(exact, float(cosine(vector, self._entity_vectors[index]))), entity))
        rows.sort(key=lambda pair: -pair[0])
        return [entity for score, entity in rows[:limit] if score > 0.1]

    @staticmethod
    def _ranks(scores: list[float]) -> dict[int, int]:
        order = sorted(range(len(scores)), key=lambda index: -scores[index])
        return {index: rank + 1 for rank, index in enumerate(order)}

    def execute(self, question: str, query: MemoryQuery) -> RetrievalResult:
        if not self._records:
            return RetrievalResult(query)
        search_text = " ".join([question, *query.concepts])
        vector = self.embedder.encode_one(search_text)
        query_tokens = _tokens(search_text)
        dense = [float(cosine(vector, row)) for row in self._record_vectors]
        lexical = [
            len(query_tokens & _tokens(text)) / max(1, len(query_tokens))
            for text in self._record_texts
        ]
        dense_rank = self._ranks(dense)
        lexical_rank = self._ranks(lexical)
        focal_ids = {
            entity.id
            for name in query.focal_entities
            for entity in [self.store.find_entity(name)]
            if entity is not None
        }
        fused = []
        for index, record in enumerate(self._records):
            if query.current_only and record.status != "active":
                continue
            if (
                query.start
                and record.valid_time
                and record.valid_time.end
                and record.valid_time.end <= query.start
            ):
                continue
            if (
                query.end
                and record.valid_time
                and record.valid_time.start
                and record.valid_time.start >= query.end
            ):
                continue
            entity_match = any(p.entity_id in focal_ids for p in record.participants)
            if focal_ids and not entity_match and query.max_hops == 0:
                continue
            # Reciprocal-rank fusion avoids benchmark-tuned score weights.
            score = 1 / (60 + dense_rank[index]) + 1 / (60 + lexical_rank[index])
            if entity_match:
                score += 1 / 40
            fused.append((record, score))
        fused.sort(key=lambda pair: (-pair[1], -pair[0].episode_ts.timestamp()))
        seeds = [record.id for record, _ in fused[: self.memory.config.clio3.seed_records]]
        distances = self.store.neighborhood(seeds, query.max_hops)
        score_by_id = {record.id: score for record, score in fused}
        expanded = []
        for record_id, distance in distances.items():
            record = self.store.get_record(record_id)
            base = score_by_id.get(record_id, 0.0)
            expanded.append((record, base + 1 / (100 * (distance + 1))))
        expanded.sort(key=lambda pair: (-pair[1], -pair[0].episode_ts.timestamp()))
        expanded = expanded[: self.memory.config.clio3.answer_record_limit]
        episode_ids = []
        for record, _ in expanded:
            if record.episode_id not in episode_ids:
                episode_ids.append(record.episode_id)
        self.memory.episode_index.rebuild(self.memory.log)
        for episode, _ in self.memory.episode_index.search_scored(
            question, k=self.memory.config.clio3.raw_episode_limit, min_score=0.0
        ):
            if episode.id not in episode_ids:
                episode_ids.append(episode.id)
        return RetrievalResult(
            query=query,
            records=expanded,
            episode_ids=episode_ids[: self.memory.config.clio3.answer_evidence_limit],
        )


__all__ = ["OpenGraphRetriever"]
