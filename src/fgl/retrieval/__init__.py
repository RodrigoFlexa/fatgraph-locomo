"""Retrieval: embedders, vector index, and the face walk that replaces k-NN."""

from fgl.retrieval.embeddings import (
    AzureEmbedder,
    CachedEmbedder,
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    VectorIndex,
    build_embedder,
    build_index,
    cosine,
)
from fgl.retrieval.faces import (
    Answerer,
    FaceRetriever,
    RetrievalResult,
    RetrievedFact,
    clean_answer,
    render_context,
)

__all__ = [
    "AzureEmbedder", "CachedEmbedder", "Embedder", "HashingEmbedder",
    "SentenceTransformerEmbedder", "VectorIndex", "build_embedder", "build_index",
    "cosine", "Answerer", "FaceRetriever", "RetrievalResult", "RetrievedFact",
    "clean_answer", "render_context",
]
