"""Hybrid lexical + dense search over entities (spec 6.2b's candidate
search, and the real path for ``anchor``, spec 9.2).

Reuses :mod:`fgl.retrieval.embeddings` -- generic, swappable embedding
backends (hashed n-grams offline, sentence-transformers or Azure for real
runs) with no fatgraph-condition business logic, unlike ``fgl.memory``
(see :mod:`fgl.clio.temporal.resolver` and
:mod:`fgl.clio.consolidate.fold`'s module docstrings for why THOSE avoid
importing it). Reusing this one is the opposite call for the opposite
reason: embedding text and calling an LLM are infrastructure this package
has no reason to reimplement, not fatgraph-specific business logic it
would be wrongly coupled to.
"""

from __future__ import annotations

import numpy as np

from fgl.clio.graph.store import GraphStore
from fgl.clio.log.store import LogStore
from fgl.clio.types import Entity, Episode
from fgl.retrieval.embeddings import Embedder, cosine


def _entity_text(e: Entity) -> str:
    return " ".join([e.canonical_name, *e.aliases])


class EntityIndex:
    """Rebuilt from the graph's current (unfolded-away) entities on
    demand, rather than maintained incrementally: a CLIO memory holds at
    most a few hundred entities, so a full rebuild is cheap, and it can
    never drift from the graph's true state the way an incrementally
    patched index could after a fold migrates or merges vertices.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._entities: list[Entity] = []
        self._vectors: np.ndarray | None = None

    def rebuild(self, graph: GraphStore) -> None:
        self._entities = [e for e in graph.all_entities() if e.merged_into is None]
        self._vectors = (
            self.embedder.encode([_entity_text(e) for e in self._entities])
            if self._entities
            else None
        )

    def search(self, query: str, k: int = 5, min_score: float = 0.15) -> list[Entity]:
        """``min_score`` matters most exactly when it looks least needed:
        with only a handful of entities in the graph, cosine similarity
        against unrelated short text is rarely EXACTLY zero (measured with
        ``HashingEmbedder``: an unrelated name still scores ~0.08), so a
        bare ``> 0`` filter returns every entity in a small memory
        regardless of relevance. 0.15 clears that noise floor while
        staying well under a real lexical or semantic match.
        """
        if not self._entities:
            return []
        query_lower = query.lower()
        q_vec = self.embedder.encode_one(query)
        scored: list[tuple[float, Entity]] = []
        for i, ent in enumerate(self._entities):
            names = [ent.canonical_name, *ent.aliases]
            # An exact surface match is never allowed to lose to a noisy
            # embedding score -- it is the strongest signal there is.
            lexical = 1.0 if any(n.lower() in query_lower for n in names) else 0.0
            dense = float(cosine(q_vec, self._vectors[i]))
            scored.append((max(lexical, dense), ent))
        scored.sort(key=lambda pair: -pair[0])
        return [ent for score, ent in scored[:k] if score >= min_score]


class EpisodeIndex:
    """Same idea, over episode text -- the other half of spec 9.2's
    ``anchor`` ("busca híbrida sobre entidades e episódios")."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._episodes: list[Episode] = []
        self._vectors: np.ndarray | None = None

    def rebuild(self, log: LogStore) -> None:
        self._episodes = log.all()
        self._vectors = (
            self.embedder.encode([e.text for e in self._episodes])
            if self._episodes
            else None
        )

    def search(self, query: str, k: int = 5) -> list[Episode]:
        if not self._episodes:
            return []
        q_vec = self.embedder.encode_one(query)
        scored = [
            (float(cosine(q_vec, self._vectors[i])), ep)
            for i, ep in enumerate(self._episodes)
        ]
        scored.sort(key=lambda pair: -pair[0])
        return [ep for _, ep in scored[:k]]
