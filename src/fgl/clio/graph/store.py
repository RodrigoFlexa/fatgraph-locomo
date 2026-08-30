"""The consolidated graph: entities and bitemporal edges (spec section 3).

Rows are never deleted, only narrowed (an interval end gets written) --
that is P2, and it is why this store hands out live references to
:class:`~fgl.clio.types.Edge` objects rather than copies: consolidation
phases narrow them in place and that mutation IS the write.
"""

from __future__ import annotations

from fgl.clio.catalog import Catalog
from fgl.clio.graph.queries import edges_at, live_edges_at
from fgl.clio.graph.queries import out_edges as _out_edges
from fgl.clio.types import Edge, EdgeAddress, Entity, Interval


class GraphStore:
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._edges: dict[str, Edge] = {}
        self._next_entity_seq = 0
        self._next_edge_seq = 0

    # ------------------------------------------------------------ entities --
    def create_entity(
        self,
        canonical_name: str,
        type: str,
        created_from: str = "",
        provisional: bool = False,
        aliases: list[str] | None = None,
    ) -> Entity:
        seq = self._next_entity_seq
        self._next_entity_seq += 1
        ent = Entity(
            id=f"ent_{seq:05d}",
            canonical_name=canonical_name,
            type=type,
            aliases=list(aliases or []),
            created_from=created_from,
            provisional=provisional,
        )
        self._entities[ent.id] = ent
        return ent

    def get_entity(self, entity_id: str) -> Entity:
        return self._entities[entity_id]

    def find_entity_by_name(self, name: str, type: str) -> Entity | None:
        """Exact, case-insensitive match on canonical name or an alias,
        among entities of the same type that are not themselves aliases.

        This is deliberately not :func:`fgl.clio.consolidate.fold.identity_score`
        -- that Jaro-Winkler/structural scorer (spec 8.2) is what milestone
        M6 uses to fold near-duplicates like "Rui" / "Rui Sampaio". Phase 1
        (spec 7.2) only needs to stop the SAME name from creating a second
        vertex every time it is mentioned again; genuine fuzzy resolution
        of *different* surface forms is fold's job, not phase 1's.
        """
        needle = name.strip().lower()
        for ent in self._entities.values():
            if ent.merged_into is not None or ent.type != type:
                continue
            if ent.canonical_name.strip().lower() == needle:
                return ent
            if any(a.strip().lower() == needle for a in ent.aliases):
                return ent
        return None

    def find_entity_by_name_in_types(self, name: str, types: set[str]) -> Entity | None:
        """Exact name/alias match against any entity whose type is in
        ``types`` -- the set a catalog ``type_class`` declares
        interchangeable (see :meth:`fgl.clio.catalog.loader.Catalog.
        type_class`). Prefers a match on the type the caller asked for
        first, so reusing a vertex never depends on dictionary order.
        """
        needle = name.strip().lower()
        matches = [
            ent
            for ent in self._entities.values()
            if ent.merged_into is None
            and ent.type in types
            and (
                ent.canonical_name.strip().lower() == needle
                or any(a.strip().lower() == needle for a in ent.aliases)
            )
        ]
        if not matches:
            return None
        return min(matches, key=lambda e: e.id)

    def find_entity_by_name_any_type(self, name: str) -> Entity | None:
        """Same exact-match rule as :meth:`find_entity_by_name`, without
        requiring the caller to know the type -- what a name-only tool
        argument (``memory_filter``, ``memory_count``) has to work with."""
        needle = name.strip().lower()
        for ent in self._entities.values():
            if ent.merged_into is not None:
                continue
            if ent.canonical_name.strip().lower() == needle:
                return ent
            if any(a.strip().lower() == needle for a in ent.aliases):
                return ent
        return None

    def all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def resolve_entity(self, entity_id: str) -> str:
        """Follows ``merged_into`` to the live vertex a (possibly folded-away)
        id now points to. Safe to call on any id, folded or not."""
        seen = set()
        while True:
            ent = self._entities[entity_id]
            if ent.merged_into is None:
                return entity_id
            if entity_id in seen:  # defensive: a cycle would mean a fold bug
                return entity_id
            seen.add(entity_id)
            entity_id = ent.merged_into

    def mark_alias(self, absorbed_id: str, merged_into: str) -> None:
        """Fold (spec 8.3): ``absorbed_id`` becomes an alias id, resolving
        to ``merged_into`` from now on."""
        self._entities[absorbed_id].merged_into = merged_into

    def unmark_alias(self, entity_id: str) -> None:
        """Unfold (spec 8.4): restores a folded-away vertex to standing on
        its own again."""
        self._entities[entity_id].merged_into = None

    # --------------------------------------------------------------- edges --
    def create_edge(
        self,
        src_id: str,
        label: str,
        dst_id: str,
        t_valid: Interval,
        t_tx: Interval,
        provenance: list[str],
        confidence: float,
        reinforcement: int = 1,
        last_confirmed=None,
        polarity: bool = True,
        unanchored: bool = False,
    ) -> Edge:
        seq = self._next_edge_seq
        self._next_edge_seq += 1
        edge = Edge(
            id=f"edg_{seq:06d}",
            src_id=src_id,
            label=label,
            dst_id=dst_id,
            t_valid=t_valid,
            t_tx=t_tx,
            provenance=list(provenance),
            reinforcement=reinforcement,
            last_confirmed=last_confirmed,
            confidence=confidence,
            polarity=polarity,
            unanchored=unanchored,
        )
        self._edges[edge.id] = edge
        return edge

    def get_edge(self, edge_id: str) -> Edge:
        return self._edges[edge_id]

    def all_edges(self) -> list[Edge]:
        return list(self._edges.values())

    def edges_at(self, address: EdgeAddress) -> list[Edge]:
        return edges_at(self._edges.values(), address)

    def live_edges_at(self, address: EdgeAddress) -> list[Edge]:
        return live_edges_at(self._edges.values(), address)

    def addresses(self) -> set[EdgeAddress]:
        return {EdgeAddress(e.src_id, e.label) for e in self._edges.values()}

    def edges_incident(self, vertex_id: str, live_only: bool = True) -> list[Edge]:
        """Every edge touching ``vertex_id`` on either end. The candidate
        pool for :func:`~fgl.clio.graph.queries.out_edges` (access, M7),
        for fold's neighbourhood overlap (M6), and for migrating a folded
        vertex's edges."""
        return [
            e
            for e in self._edges.values()
            if (e.src_id == vertex_id or e.dst_id == vertex_id)
            and (not live_only or e.t_tx.end is None)
        ]

    def out_edges(
        self, vertex_id: str, label: str, catalog: Catalog
    ) -> list[tuple[Edge, str]]:
        """See :func:`fgl.clio.graph.queries.out_edges`.

        Deliberately passes ``live_only=False``: a retracted edge must
        still reach ``follow``, which is the one that decides whether it
        survives -- against ``state.tx_point``, not against "now". A
        historical-belief query (spec T4: ``restrict(tx, <past point>)``)
        needs exactly the edges a live-only filter would have already
        thrown away.
        """
        return _out_edges(
            vertex_id, label, self.edges_incident(vertex_id, live_only=False), catalog
        )

    def migrate_edges(self, from_id: str, to_id: str) -> None:
        """Fold (spec 8.3): every edge touching ``from_id`` now touches
        ``to_id`` instead. Includes retracted/closed edges -- provenance
        must follow the vertex even into its history."""
        for e in self._edges.values():
            if e.src_id == from_id:
                e.src_id = to_id
            if e.dst_id == from_id:
                e.dst_id = to_id

    def migrate_specific_edges(self, edge_ids: set[str], from_id: str, to_id: str) -> None:
        """Unfold's narrower counterpart to :meth:`migrate_edges`: moves
        back only the edges a specific fold actually migrated, not
        everything ``from_id`` (the kept vertex) happens to hold now --
        which may include edges from a second, later fold onto the same
        vertex, or ordinary new writes since."""
        for eid in edge_ids:
            e = self._edges[eid]
            if e.src_id == from_id:
                e.src_id = to_id
            if e.dst_id == from_id:
                e.dst_id = to_id
