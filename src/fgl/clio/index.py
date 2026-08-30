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

import math
import re

import numpy as np

from fgl.clio.graph.store import GraphStore
from fgl.clio.log.store import LogStore
from fgl.clio.types import Entity, Episode
from fgl.retrieval.embeddings import Embedder, cosine


def _entity_text(e: Entity) -> str:
    return " ".join([e.canonical_name, *e.aliases])


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
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


def _search_tokens(text: str) -> set[str]:
    """Stable content tokens for the lexical half of hybrid retrieval."""
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    }


def _episode_text(episode: Episode) -> str:
    # Speaker identity is part of a conversational turn even when the name is
    # naturally absent from its first-person text.
    return f"{episode.speaker}: {episode.text}"


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
        self._tokens: list[set[str]] = []
        self._idf: dict[str, float] = {}

    def rebuild(self, graph: GraphStore) -> None:
        self._entities = [e for e in graph.all_entities() if e.merged_into is None]
        self._vectors = (
            self.embedder.encode([_entity_text(e) for e in self._entities])
            if self._entities
            else None
        )

    def search_scored(
        self, query: str, k: int = 5, min_score: float = 0.15
    ) -> list[tuple[Entity, float]]:
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
        return [(ent, score) for score, ent in scored[:k] if score >= min_score]

    def search(self, query: str, k: int = 5, min_score: float = 0.15) -> list[Entity]:
        return [ent for ent, _ in self.search_scored(query, k=k, min_score=min_score)]

    def exact_person(self, name: str) -> Entity | None:
        """Return the canonical speaker vertex, without dense guesswork."""
        needle = name.strip().casefold()
        for ent in self._entities:
            if ent.type == "Person" and any(
                candidate.strip().casefold() == needle
                for candidate in (ent.canonical_name, *ent.aliases)
            ):
                return ent
        return None


class EpisodeIndex:
    """Same idea, over episode text -- the other half of spec 9.2's
    ``anchor`` ("busca híbrida sobre entidades e episódios")."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._episodes: list[Episode] = []
        self._vectors: np.ndarray | None = None

    def rebuild(self, log: LogStore) -> None:
        self._episodes = log.all()
        self._tokens = [_search_tokens(_episode_text(e)) for e in self._episodes]
        document_frequency: dict[str, int] = {}
        for tokens in self._tokens:
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        n_documents = len(self._episodes)
        self._idf = {
            token: math.log((n_documents + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self._vectors = (
            self.embedder.encode([_episode_text(e) for e in self._episodes])
            if self._episodes
            else None
        )

    def search_scored(
        self, query: str, k: int = 5, min_score: float = 0.20
    ) -> list[tuple[Episode, float]]:
        if not self._episodes:
            return []
        q_vec = self.embedder.encode_one(query)
        query_tokens = _search_tokens(query)
        unseen_idf = math.log(len(self._episodes) + 1) + 1.0
        query_weight = sum(self._idf.get(token, unseen_idf) for token in query_tokens)
        scored = []
        for i, episode in enumerate(self._episodes):
            dense = float(cosine(q_vec, self._vectors[i]))
            lexical = (
                sum(
                    self._idf.get(token, unseen_idf)
                    for token in query_tokens & self._tokens[i]
                )
                / query_weight
                if query_weight
                else 0.0
            )
            scored.append((max(dense, lexical), episode))
        scored.sort(key=lambda pair: -pair[0])
        return [(ep, score) for score, ep in scored[:k] if score >= min_score]

    def search(self, query: str, k: int = 5, min_score: float = 0.20) -> list[Episode]:
        return [ep for ep, _ in self.search_scored(query, k=k, min_score=min_score)]
