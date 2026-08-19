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
    JOIN_SOURCES,
    SOURCE_COVERAGE,
    SOURCE_FACE,
    SOURCE_FACE_UNIT,
    SOURCE_GEODESIC,
    SOURCE_SIGMA,
    Answerer,
    FaceRetriever,
    QuestionLinker,
    RetrievalResult,
    RetrievedFact,
    clean_answer,
    render_context,
)
from fgl.retrieval.bipartite import (
    BIPARTITE_SOURCES,
    SOURCE_BP_BRIDGE,
    SOURCE_BP_DENSE,
    SOURCE_BP_ENTITY,
    BipartiteRetriever,
)
from fgl.retrieval.slots import (
    SLOT_SOURCES,
    SOURCE_SLOT_ACTOR,
    SOURCE_SLOT_CONCEPT,
    SOURCE_SLOT_DENSE,
    SOURCE_SLOT_PREDICATE,
    SOURCE_SLOT_TIME,
    SOURCE_SLOT_TYPE,
    QuestionSlots,
    SlotRetriever,
)
from fgl.retrieval.propagation import (
    NORMALIZATIONS,
    PropagationRetriever,
    propagate,
    reduces_to_l2,
)
from fgl.retrieval.steiner import (
    NullDistribution,
    SteinerMetric,
    SteinerRead,
    calibrate_null,
)
from fgl.retrieval.unified import UnifiedRetriever

__all__ = [
    "AzureEmbedder", "CachedEmbedder", "Embedder", "HashingEmbedder",
    "SentenceTransformerEmbedder", "VectorIndex", "build_embedder", "build_index",
    "cosine", "Answerer", "FaceRetriever", "RetrievalResult", "RetrievedFact",
    "clean_answer", "render_context", "SOURCE_FACE", "SOURCE_SIGMA",
    "SOURCE_COVERAGE", "SOURCE_GEODESIC", "SOURCE_FACE_UNIT",
    "JOIN_SOURCES", "QuestionLinker",
    "BipartiteRetriever", "SOURCE_BP_ENTITY", "SOURCE_BP_BRIDGE", "SOURCE_BP_DENSE",
    "BIPARTITE_SOURCES",
    "SlotRetriever", "QuestionSlots", "SLOT_SOURCES", "SOURCE_SLOT_ACTOR",
    "SOURCE_SLOT_PREDICATE", "SOURCE_SLOT_CONCEPT", "SOURCE_SLOT_TYPE",
    "SOURCE_SLOT_TIME", "SOURCE_SLOT_DENSE",
    "PropagationRetriever", "propagate", "reduces_to_l2", "NORMALIZATIONS",
    "SteinerMetric", "SteinerRead", "NullDistribution", "calibrate_null",
    "UnifiedRetriever",
]
