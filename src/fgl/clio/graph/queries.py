"""Pure address-matching helpers over a flat edge collection.

Kept free of any store so :class:`fgl.clio.graph.store.GraphStore` stays a
thin container and these stay trivially testable.
"""

from __future__ import annotations

from collections.abc import Iterable

from fgl.clio.catalog import Catalog
from fgl.clio.types import Edge, EdgeAddress


class UnknownLabel(ValueError):
    """``follow`` was asked to walk a label that is neither a relation in
    Sigma nor a known inverse of one (spec 9.3)."""


def edges_at(edges: Iterable[Edge], address: EdgeAddress) -> list[Edge]:
    """Every edge ever written at this address, regardless of state."""
    return [e for e in edges if e.src_id == address.src and e.label == address.label]


def live_edges_at(edges: Iterable[Edge], address: EdgeAddress) -> list[Edge]:
    """Edges not retracted (``t_tx`` still open) -- currently believed,
    independent of whether ``t_valid`` has since been closed. A past
    employer is still something the agent currently believes was true.
    """
    return [e for e in edges_at(edges, address) if e.t_tx.end is None]


def out_edges(
    vertex_id: str, label: str, incident: Iterable[Edge], catalog: Catalog
) -> list[tuple[Edge, str]]:
    """Edges ``follow`` can walk from ``vertex_id`` under ``label``, paired
    with the neighbour vertex id each one leads to.

    Three cases, spec 3.3 and 9.3:

    * ``label`` is an ordinary forward relation -- edges where ``vertex_id``
      is the subject, landing on the object.
    * ``label`` is that SAME relation but it is self-inverse (``friend_of``,
      ``family_of``, ``partner_of``, ``works_with`` -- ``inverse_name ==
      name``) -- also include edges where ``vertex_id`` is the OBJECT,
      landing back on the subject, because a symmetric fact recorded from
      either party's turn must be walkable from both.
    * ``label`` is a genuine inverse name (``employs`` for ``works_at``) --
      edges of the FORWARD relation where ``vertex_id`` is the object,
      landing on the subject. No physical ``employs`` edge exists (see this
      module's docstring); this is a live reversal of the ``works_at`` ones.

    ``incident`` should already be narrowed to edges touching ``vertex_id``
    (:meth:`~fgl.clio.graph.store.GraphStore.edges_incident`) -- this
    function does not re-filter by vertex, only by label and direction.
    """
    if label in catalog:
        spec = catalog[label]
        pairs = [
            (e, e.dst_id) for e in incident if e.label == label and e.src_id == vertex_id
        ]
        if spec.invertible and spec.inverse_name == label:
            pairs += [
                (e, e.src_id)
                for e in incident
                if e.label == label and e.dst_id == vertex_id and e.src_id != vertex_id
            ]
        return pairs
    if catalog.is_inverse_label(label):
        forward = catalog.forward_of(label)
        return [
            (e, e.src_id) for e in incident if e.label == forward and e.dst_id == vertex_id
        ]
    raise UnknownLabel(label)
