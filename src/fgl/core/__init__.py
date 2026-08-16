"""The fatgraph itself: half-edges, alpha, sigma, phi, faces, Euler, curation ops."""

from fgl.core.fatgraph import (
    EDGE_LEVEL_ATTRS,
    STATE_CONSOLIDATED,
    STATE_EMERGENT,
    STATE_INCONGRUENT,
    VALID_STATES,
    ComponentStats,
    EulerStats,
    Face,
    FatGraph,
    FatGraphError,
    HalfEdge,
    InvariantError,
    NotABigonError,
    TopologyViolation,
    Vertex,
    default_token_counter,
    face_id,
)

__all__ = [
    "EDGE_LEVEL_ATTRS", "STATE_CONSOLIDATED", "STATE_EMERGENT", "STATE_INCONGRUENT",
    "VALID_STATES", "ComponentStats", "EulerStats", "Face", "FatGraph",
    "FatGraphError", "HalfEdge", "InvariantError", "NotABigonError",
    "TopologyViolation", "Vertex", "default_token_counter", "face_id",
]
