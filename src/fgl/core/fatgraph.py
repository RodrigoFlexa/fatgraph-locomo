"""Core fatgraph (ribbon graph) data structures.

A fatgraph is a graph endowed with a cyclic ordering of the half-edges around
each vertex.  Formally it is a triple ``(H, alpha, sigma)`` where

* ``H``      -- the finite set of half-edges (a.k.a. darts, flags);
* ``alpha``  -- a fixed-point-free involution on ``H``; the orbits of ``alpha``
                are the *edges*;
* ``sigma``  -- a permutation of ``H`` whose orbits are the *vertices*; the
                orbit of ``sigma`` at a vertex is exactly the cyclic order of
                the half-edges glued to that vertex.

The *faces* (boundary components of the ribbon surface) are the orbits of

    phi = sigma o alpha

The surface obtained by thickening the graph satisfies Euler's formula

    V - E + F = 2 * C - 2 * g

where ``C`` is the number of connected components and ``g`` the total genus.
See ``COERENCIA.md`` (item C1) for why the connected-only form ``V-E+F=2-2g``
found in the original specification is not usable on LoCoMo memory graphs.

Design notes
------------
* Half-edges carry the *fact text seen from their vertex*; the two half-edges of
  an edge therefore describe the same memory from two perspectives.
* Attributes that logically belong to the **edge** (``state``, ``level``,
  ``shadowed``, ``children``, ``turn_ids`` ...) are physically stored on both
  half-edges and must only be mutated through the edge-level helpers
  (:meth:`FatGraph.set_edge_attr`) so the two halves never drift apart.
* Clarity beats micro-optimisation (this is research code), but ``faces()`` is
  kept O(|H|) as required, which forces us to maintain an index of the position
  of each half-edge inside its vertex's cyclic list.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

STATE_EMERGENT = "emergente"
STATE_CONSOLIDATED = "consolidada"
STATE_INCONGRUENT = "incongruente"
VALID_STATES = (STATE_EMERGENT, STATE_CONSOLIDATED, STATE_INCONGRUENT)

#: Edge-level attributes: writing them on one half-edge must write the twin too.
EDGE_LEVEL_ATTRS = (
    "state",
    "level",
    "shadowed",
    "children",
    "turn_ids",
    "session_id",
    "timestamp",
    "provenance",
)


# --------------------------------------------------------------------------- #
# Exceptions                                                                   #
# --------------------------------------------------------------------------- #


class FatGraphError(Exception):
    """Base class for every error raised by this module."""


class TopologyViolation(FatGraphError):
    """A curation operation changed a topological invariant it must preserve."""


class NotABigonError(FatGraphError):
    """The face is not a collapsible bigon (two *distinct* parallel edges)."""


class InvariantError(FatGraphError):
    """``check_invariants`` found a structurally inconsistent graph."""


# --------------------------------------------------------------------------- #
# Token accounting                                                             #
# --------------------------------------------------------------------------- #


def default_token_counter(text: str) -> int:
    """Cheap, dependency-free token estimate.

    Uses ``tiktoken`` when available (exact for OpenAI models), otherwise the
    usual ~1.3 tokens/word heuristic.  Injectable everywhere it matters.
    """
    try:  # pragma: no cover - depends on optional dependency
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(round(len(text.split()) * 1.3)))


# --------------------------------------------------------------------------- #
# Dataclasses                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class Vertex:
    """A canonical entity/concept extracted from the conversations."""

    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass
class HalfEdge:
    """One of the two halves of a memory (edge)."""

    id: str
    vertex_id: str
    text: str  # fact text, seen from ``vertex_id``
    embedding: Optional[np.ndarray] = None
    session_id: str = ""
    turn_ids: list[str] = field(default_factory=list)
    timestamp: str = ""  # ISO-8601, see ingest.normalize_timestamp
    state: str = STATE_EMERGENT
    level: int = 1  # 1 = raw fact, 2 = face consolidation

    # --- additions over the original spec (see DECISIONS.md, D2) -----------
    edge_id: str = ""  # id of the alpha-orbit this half-edge belongs to
    shadowed: bool = False  # superseded by a level-2 consolidation
    children: list[str] = field(default_factory=list)  # edge ids summarised
    provenance: list[str] = field(default_factory=list)  # edge ids merged in
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass(frozen=True)
class Face:
    """A phi-orbit: an oriented closed trail of memories."""

    id: str
    half_edges: tuple[str, ...]
    edges: tuple[str, ...]  # edge id per half-edge, in traversal order
    component: int = -1

    @property
    def length(self) -> int:
        """Number of half-edges traversed (the combinatorial face degree)."""
        return len(self.half_edges)

    @property
    def distinct_edges(self) -> tuple[str, ...]:
        """Edges of the face, order preserved, first occurrence kept."""
        seen: set[str] = set()
        out: list[str] = []
        for e in self.edges:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return tuple(out)

    @property
    def is_leaf_face(self) -> bool:
        """True when the face traverses a single edge twice (a leaf/bridge).

        Such faces have length 2 but are **not** redundancy bigons -- see
        ``COERENCIA.md`` item C3.
        """
        return len(set(self.edges)) < len(self.edges) and self.length == 2


@dataclass(frozen=True)
class ComponentStats:
    index: int
    V: int
    E: int
    F: int
    genus: int


@dataclass(frozen=True)
class EulerStats:
    """Result of :meth:`FatGraph.euler`."""

    V: int
    E: int
    F: int
    C: int  # connected components (isolated vertices count as one each)
    genus: int  # sum of the per-component genera
    components: tuple[ComponentStats, ...] = ()

    @property
    def chi(self) -> int:
        """Euler characteristic ``V - E + F``."""
        return self.V - self.E + self.F

    @property
    def genus_connected_formula(self) -> float:
        """The literal ``(2 - V + E - F)/2`` of the specification.

        Only equals :attr:`genus` when ``C == 1``.  Kept for traceability.
        """
        return (2 - self.V + self.E - self.F) / 2

    def as_tuple(self) -> tuple[int, int, int, int]:
        """``(V, E, F, g)`` -- the signature the specification asks for."""
        return (self.V, self.E, self.F, self.genus)

    def to_dict(self) -> dict:
        return {
            "V": self.V,
            "E": self.E,
            "F": self.F,
            "C": self.C,
            "genus": self.genus,
            "chi": self.chi,
            "components": [asdict(c) for c in self.components],
        }


# --------------------------------------------------------------------------- #
# FatGraph                                                                     #
# --------------------------------------------------------------------------- #


class FatGraph:
    """A fatgraph memory.

    Attributes
    ----------
    H : dict[str, HalfEdge]
    alpha : dict[str, str]
        Fixed-point-free involution: ``alpha[alpha[h]] == h`` and
        ``alpha[h] != h``.
    sigma : dict[str, list[str]]
        ``vertex_id -> cyclic ordered list of half-edge ids``.  "Cyclic" means
        the list is read modulo its length; ``sigma(h)`` is the *next* element.
    vertices : dict[str, Vertex]
    """

    def __init__(self, token_counter: Callable[[str], int] | None = None) -> None:
        self.H: dict[str, HalfEdge] = {}
        self.alpha: dict[str, str] = {}
        self.sigma: dict[str, list[str]] = {}
        self.vertices: dict[str, Vertex] = {}

        self._token_counter = token_counter or default_token_counter
        self._he_counter = 0
        self._edge_counter = 0
        self._vertex_counter = 0
        #: position of each half-edge inside ``sigma[vertex_id]`` (O(1) sigma).
        self._sigma_pos: dict[str, int] = {}
        #: ``edge_id -> (h1, h2)``; without it ``edge_half_edges`` would scan the
        #: whole half-edge set and ``faces()`` would be quadratic, not O(|H|).
        self._edge_index: dict[str, list[str]] = {}
        #: cached union-find result, invalidated by any structural change.
        self._components_cache: Optional[dict[str, int]] = None

    # ---------------------------------------------------------------- ids ---
    def _new_vertex_id(self) -> str:
        self._vertex_counter += 1
        return f"v{self._vertex_counter}"

    def _new_edge_id(self) -> str:
        self._edge_counter += 1
        return f"e{self._edge_counter}"

    def _new_half_edge_id(self) -> str:
        self._he_counter += 1
        return f"h{self._he_counter}"

    # ----------------------------------------------------------- vertices ---
    def add_vertex(
        self,
        name: str,
        aliases: Sequence[str] = (),
        embedding: Optional[np.ndarray] = None,
        vertex_id: str | None = None,
        meta: dict | None = None,
    ) -> str:
        """Create a vertex and return its id."""
        vid = vertex_id or self._new_vertex_id()
        if vid in self.vertices:
            raise FatGraphError(f"vertex {vid!r} already exists")
        self.vertices[vid] = Vertex(
            id=vid,
            name=name,
            aliases=list(aliases),
            embedding=embedding,
            meta=dict(meta or {}),
        )
        self.sigma.setdefault(vid, [])
        self._components_cache = None
        return vid

    def degree(self, vertex_id: str) -> int:
        return len(self.sigma.get(vertex_id, ()))

    # -------------------------------------------------------------- edges ---
    def add_edge(
        self,
        v1: str,
        v2: str,
        fact: "FactLike",
        pos1: int | None = None,
        pos2: int | None = None,
    ) -> str:
        """Create the half-edge pair of a memory and return its ``edge_id``.

        Parameters
        ----------
        v1, v2 :
            Vertex ids.  ``v1 == v2`` (a loop) is allowed; ``pos2`` is then
            interpreted **after** ``h1`` has already been inserted.
        fact :
            Anything exposing ``text`` (and optionally ``text_from_v1`` /
            ``text_from_v2``, ``embedding``, ``session_id``, ``turn_ids``,
            ``timestamp``, ``state``, ``level``).  A plain ``dict`` works too.
        pos1, pos2 :
            Insertion index in ``sigma[v1]`` / ``sigma[v2]`` with ``list.insert``
            semantics (the new half-edge ends up *at* that index).  ``None``
            appends at the end of the cyclic list.
        """
        for v in (v1, v2):
            if v not in self.vertices:
                raise FatGraphError(f"unknown vertex {v!r}")

        f = _FactView(fact)
        eid = self._new_edge_id()
        h1_id, h2_id = self._new_half_edge_id(), self._new_half_edge_id()

        common = dict(
            session_id=f.session_id,
            turn_ids=list(f.turn_ids),
            timestamp=f.timestamp,
            state=f.state,
            level=f.level,
            edge_id=eid,
            shadowed=f.shadowed,
            children=list(f.children),
            provenance=list(f.provenance),
            meta=dict(f.meta),
        )
        h1 = HalfEdge(
            id=h1_id,
            vertex_id=v1,
            text=f.text_from(self.vertices[v1].name, 1),
            embedding=f.embedding,
            **common,
        )
        h2 = HalfEdge(
            id=h2_id,
            vertex_id=v2,
            text=f.text_from(self.vertices[v2].name, 2),
            embedding=f.embedding,
            **{**common, "meta": dict(f.meta)},
        )
        self.H[h1_id] = h1
        self.H[h2_id] = h2
        self.alpha[h1_id] = h2_id
        self.alpha[h2_id] = h1_id
        self._edge_index[eid] = [h1_id, h2_id]
        self._components_cache = None

        self._sigma_insert(v1, h1_id, pos1)
        self._sigma_insert(v2, h2_id, pos2)
        return eid

    def edge_half_edges(self, edge_id: str) -> tuple[str, str]:
        """The two half-edge ids of ``edge_id`` (deterministic order).  O(1)."""
        hs = self._edge_index.get(edge_id)
        if hs is None or len(hs) != 2:
            raise FatGraphError(
                f"edge {edge_id!r} has {0 if hs is None else len(hs)} half-edges "
                "(expected 2)"
            )
        return hs[0], hs[1]

    def edges(self) -> list[str]:
        """All edge ids, in creation order."""
        return sorted(self._edge_index, key=lambda e: int(e[1:]))

    def edge_endpoints(self, edge_id: str) -> tuple[str, str]:
        h1, h2 = self.edge_half_edges(edge_id)
        return self.H[h1].vertex_id, self.H[h2].vertex_id

    def set_edge_attr(self, edge_id: str, **attrs) -> None:
        """Set edge-level attributes on **both** half-edges at once."""
        h1, h2 = self.edge_half_edges(edge_id)
        for key, value in attrs.items():
            if key not in EDGE_LEVEL_ATTRS:
                raise FatGraphError(f"{key!r} is not an edge-level attribute")
            if key == "state" and value not in VALID_STATES:
                raise FatGraphError(f"invalid state {value!r}")
            for h in (h1, h2):
                setattr(self.H[h], key, _copy_value(value))

    def get_edge_attr(self, edge_id: str, key: str):
        h1, _ = self.edge_half_edges(edge_id)
        return getattr(self.H[h1], key)

    def remove_edge(self, edge_id: str) -> None:
        """Delete an edge and both of its half-edges (sigma is re-indexed)."""
        h1, h2 = self.edge_half_edges(edge_id)
        for h in (h1, h2):
            v = self.H[h].vertex_id
            self.sigma[v].remove(h)
            del self.H[h]
            del self.alpha[h]
            self._sigma_pos.pop(h, None)
            self._reindex_vertex(v)
        del self._edge_index[edge_id]
        self._components_cache = None

    # -------------------------------------------------------------- sigma ---
    def _sigma_insert(self, vertex_id: str, half_edge_id: str, pos: int | None) -> None:
        lst = self.sigma.setdefault(vertex_id, [])
        if pos is None:
            lst.append(half_edge_id)
        else:
            if not 0 <= pos <= len(lst):
                raise FatGraphError(
                    f"sigma position {pos} out of range for vertex {vertex_id!r} "
                    f"(degree {len(lst)})"
                )
            lst.insert(pos, half_edge_id)
        self._reindex_vertex(vertex_id)

    def _reindex_vertex(self, vertex_id: str) -> None:
        for i, h in enumerate(self.sigma[vertex_id]):
            self._sigma_pos[h] = i

    def sigma_next(self, h: str) -> str:
        """``sigma(h)``: the next half-edge in the cyclic order of its vertex."""
        v = self.H[h].vertex_id
        lst = self.sigma[v]
        return lst[(self._sigma_pos[h] + 1) % len(lst)]

    def sigma_prev(self, h: str) -> str:
        v = self.H[h].vertex_id
        lst = self.sigma[v]
        return lst[(self._sigma_pos[h] - 1) % len(lst)]

    def phi(self, h: str) -> str:
        """``phi(h) = sigma(alpha(h))`` -- the face permutation."""
        return self.sigma_next(self.alpha[h])

    # -------------------------------------------------------------- faces ---
    def faces(self) -> list[Face]:
        """Decompose ``phi`` into cycles.  O(|H|).

        Isolated vertices (degree 0) contribute one *trivial* face each with no
        half-edges; this is the standard ribbon-graph convention (the boundary
        of the vertex disc) and is what keeps Euler's formula integral.
        """
        comp_of_vertex = self._components()
        visited: set[str] = set()
        out: list[Face] = []

        for start in self.H:  # dict preserves insertion order -> determinism
            if start in visited:
                continue
            cycle: list[str] = []
            h = start
            while h not in visited:
                visited.add(h)
                cycle.append(h)
                h = self.phi(h)
            edges = tuple(self.H[x].edge_id for x in cycle)
            comp = comp_of_vertex[self.H[cycle[0]].vertex_id]
            out.append(
                Face(
                    id=face_id(edges),
                    half_edges=tuple(cycle),
                    edges=edges,
                    component=comp,
                )
            )

        for vid, lst in self.sigma.items():
            if not lst:  # isolated vertex -> trivial face
                out.append(
                    Face(
                        id=f"face:isolated:{vid}",
                        half_edges=(),
                        edges=(),
                        component=comp_of_vertex[vid],
                    )
                )
        return out

    def face_of(self, h: str) -> Face:
        """The face through half-edge ``h``."""
        cycle: list[str] = []
        cur = h
        while True:
            cycle.append(cur)
            cur = self.phi(cur)
            if cur == h:
                break
        edges = tuple(self.H[x].edge_id for x in cycle)
        comp = self._components()[self.H[h].vertex_id]
        return Face(id=face_id(edges), half_edges=tuple(cycle), edges=edges, component=comp)

    def faces_through_vertex(self, vertex_id: str) -> list[Face]:
        """Every face incident to ``vertex_id`` (deduplicated by face id)."""
        seen: dict[str, Face] = {}
        for h in self.sigma.get(vertex_id, ()):  # noqa: B007
            f = self.face_of(h)
            seen.setdefault(f.id, f)
        return list(seen.values())

    def walk_face(
        self,
        h: str,
        budget_tokens: int | None = None,
    ) -> list[HalfEdge]:
        """Iterate ``phi`` from ``h`` accumulating texts.

        Stops when the cycle closes or when the token budget would be exceeded.
        Returns the **ordered** sequence of half-edges, one per distinct edge
        (a face may traverse the same edge twice; only the first visit is kept).
        """
        out: list[HalfEdge] = []
        seen_edges: set[str] = set()
        used = 0
        cur = h
        while True:
            he = self.H[cur]
            if he.edge_id not in seen_edges:
                cost = self._token_counter(he.text)
                if budget_tokens is not None and used + cost > budget_tokens and out:
                    break
                used += cost
                seen_edges.add(he.edge_id)
                out.append(he)
                if budget_tokens is not None and used >= budget_tokens:
                    break
            cur = self.phi(cur)
            if cur == h:
                break
        return out

    # -------------------------------------------------------------- euler ---
    def _components(self) -> dict[str, int]:
        """Union-find over vertices; returns ``vertex_id -> component index``.

        Cached: the result only changes when a vertex or an edge is added or
        removed, and both invalidate ``_components_cache``.  Without the cache,
        ``face_of`` -- which ``sigma-agent`` calls once per candidate position --
        would recompute the whole partition every time.
        """
        if self._components_cache is not None:
            return self._components_cache

        parent = {v: v for v in self.vertices}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for eid in self.edges():
            a, b = self.edge_endpoints(eid)
            union(a, b)

        roots: dict[str, int] = {}
        out: dict[str, int] = {}
        for v in sorted(self.vertices):
            r = find(v)
            if r not in roots:
                roots[r] = len(roots)
            out[v] = roots[r]
        self._components_cache = out
        return out

    def euler(self) -> EulerStats:
        """Return ``(V, E, F, C, genus)`` with **per-component** genus.

        ``V - E + F = 2C - 2g``.  Genus is computed per connected component and
        summed, because ``2 - 2g`` is only valid for a connected fatgraph and
        LoCoMo memory graphs are routinely disconnected.
        """
        comp_of_vertex = self._components()
        n_comp = (max(comp_of_vertex.values()) + 1) if comp_of_vertex else 0

        v_count = [0] * n_comp
        e_count = [0] * n_comp
        f_count = [0] * n_comp

        for v, c in comp_of_vertex.items():  # noqa: B007
            v_count[c] += 1
        for eid in self.edges():
            a, _ = self.edge_endpoints(eid)
            e_count[comp_of_vertex[a]] += 1
        for f in self.faces():
            f_count[f.component] += 1

        comps: list[ComponentStats] = []
        total_genus = 0
        for c in range(n_comp):
            two_g = 2 - v_count[c] + e_count[c] - f_count[c]
            if two_g % 2 != 0 or two_g < 0:
                raise TopologyViolation(
                    f"component {c}: 2-V+E-F = {two_g} is not a non-negative even "
                    f"number (V={v_count[c]}, E={e_count[c]}, F={f_count[c]})"
                )
            g = two_g // 2
            total_genus += g
            comps.append(ComponentStats(c, v_count[c], e_count[c], f_count[c], g))

        return EulerStats(
            V=len(self.vertices),
            E=len(self.edges()),
            F=sum(f_count),
            C=n_comp,
            genus=total_genus,
            components=tuple(comps),
        )

    # ----------------------------------------------------------- curation ---
    def collapse_bigon(self, face_id_: str, merged_text: str | None = None) -> str:
        """Collapse a length-2 face made of two **distinct** parallel edges.

        The surviving edge inherits the union of provenance (``turn_ids``,
        ``provenance``) of both.  Euler statistics are recomputed before and
        after; the number of components and every per-component genus must be
        unchanged, otherwise :class:`TopologyViolation` is raised.

        Returns the id of the surviving edge.
        """
        before = self.euler()
        target = next((f for f in self.faces() if f.id == face_id_), None)
        if target is None:
            raise FatGraphError(f"unknown face {face_id_!r}")
        if target.length != 2:
            raise NotABigonError(f"face {face_id_} has length {target.length}, expected 2")
        if target.is_leaf_face:
            raise NotABigonError(
                f"face {face_id_} traverses a single edge twice (leaf/bridge); "
                "it encodes no redundancy and must not be collapsed"
            )

        e_keep, e_drop = target.edges[0], target.edges[1]
        if set(self.edge_endpoints(e_keep)) != set(self.edge_endpoints(e_drop)):
            raise NotABigonError(
                f"edges {e_keep} and {e_drop} do not join the same vertex pair"
            )

        keep_h1, keep_h2 = self.edge_half_edges(e_keep)
        drop_h1, drop_h2 = self.edge_half_edges(e_drop)

        merged_turns = _ordered_union(
            self.H[keep_h1].turn_ids, self.H[drop_h1].turn_ids
        )
        merged_prov = _ordered_union(
            self.H[keep_h1].provenance + [e_keep],
            self.H[drop_h1].provenance + [e_drop],
        )
        self.set_edge_attr(e_keep, turn_ids=merged_turns, provenance=merged_prov)
        if merged_text is not None:
            for h in (keep_h1, keep_h2):
                self.H[h].text = merged_text
        # keep the earliest timestamp/session of the pair (memory birth date)
        ts = min(
            [t for t in (self.H[keep_h1].timestamp, self.H[drop_h1].timestamp) if t]
            or [""]
        )
        if ts:
            self.set_edge_attr(e_keep, timestamp=ts)

        self.remove_edge(e_drop)

        after = self.euler()
        _assert_topology_preserved(before, after, expect_delta_E=-1, expect_delta_F=-1)
        return e_keep

    def whitehead_flip(
        self,
        edge_id: str,
        offset_u: int = 1,
        offset_v: int = 1,
    ) -> str:
        """Whitehead move: contract ``edge_id`` and re-expand on another split.

        The move keeps ``V`` and ``E`` fixed and, being a spine move of the same
        thickened surface, must keep the genus and the number of faces.  Both
        are asserted; a violation raises :class:`TopologyViolation`.

        Disabled by default -- gate it with ``config.curation.whitehead_flip``.

        Returns the id of the newly created edge.
        """
        before = self.euler()
        u, v = self.edge_endpoints(edge_id)
        if u == v:
            raise FatGraphError("cannot flip a loop: contraction is not defined")
        h_u, h_v = self.edge_half_edges(edge_id)
        if self.H[h_u].vertex_id != u:  # normalise orientation
            h_u, h_v = h_v, h_u
        if self.degree(u) < 3 or self.degree(v) < 3:
            raise FatGraphError(
                "Whitehead flip needs both endpoints with degree >= 3 "
                f"(got {self.degree(u)}, {self.degree(v)})"
            )

        # 1. contract: cyclic order of the merged vertex, reading u's list
        #    starting just after h_u then v's list starting just after h_v.
        arc_u = _rotate_after(self.sigma[u], h_u)
        arc_v = _rotate_after(self.sigma[v], h_v)
        template = self.H[h_u]

        # 2. re-expand at a rotated split point ("opposite diagonal").
        new_arc_u = arc_u[offset_u:] + arc_u[:offset_u]
        new_arc_v = arc_v[offset_v:] + arc_v[:offset_v]

        payload = dict(
            text=template.text,
            embedding=template.embedding,
            session_id=template.session_id,
            turn_ids=list(template.turn_ids),
            timestamp=template.timestamp,
            state=template.state,
            level=template.level,
            shadowed=template.shadowed,
            children=list(template.children),
            provenance=_ordered_union(template.provenance, [edge_id]),
            meta=dict(template.meta),
        )
        self.remove_edge(edge_id)

        # rebuild the two cyclic orders, then attach the new edge at the seam
        self.sigma[u] = list(new_arc_u)
        self.sigma[v] = list(new_arc_v)
        self._reindex_vertex(u)
        self._reindex_vertex(v)
        new_edge = self.add_edge(u, v, payload, pos1=0, pos2=0)

        after = self.euler()
        if after.C != before.C or after.genus != before.genus or after.F != before.F:
            raise TopologyViolation(
                f"Whitehead flip changed the surface: before={before.to_dict()} "
                f"after={after.to_dict()}"
            )
        return new_edge

    # ----------------------------------------------------------- identity ----
    def fingerprint(self) -> str:
        """Content hash of the ribbon graph: same memory *and* same rotation.

        Exists so that "these two conditions differ only in retrieval" can be a
        *measured* claim instead of a trusted one.  Sharing a graph directory
        (``paths.graphs_condition``) enforced it by construction, at the price of
        making one condition's results an artefact of another's run.  A
        fingerprint decouples them: every condition builds its own graph, and
        equality of the hash proves the ingest agreed.

        Deliberately content-addressed, not id-addressed: ``V3``/``E17`` depend
        on insertion order, so hashing them would report a difference whenever
        two runs merely numbered the same graph differently.  What is hashed is

        * each vertex by normalised name;
        * each edge by its text and the *names* of its endpoints;
        * the cyclic order of sigma at each vertex, as edge texts, reduced to
          its least rotation -- because a rotation is a cyclic object and any
          starting point denotes the same embedding.

        Embeddings are excluded: they are floats, and a rebuild on another BLAS
        would differ in the last bits without the memory having changed.
        Including sigma is what makes a genus-optimised graph hash *differently*
        from the one it came from, which is correct -- it is a different ribbon
        graph over the same memory.
        """
        name_of = {vid: vx.name for vid, vx in self.vertices.items()}
        parts: list[str] = []

        parts.append("V:" + "|".join(sorted(name_of.values())))

        edges = []
        for eid in self.edges():
            h1, h2 = self.edge_half_edges(eid)
            ends = tuple(sorted((name_of[self.H[h1].vertex_id],
                                 name_of[self.H[h2].vertex_id])))
            edges.append(f"{self.H[h1].text}\x1f{ends[0]}\x1f{ends[1]}")
        parts.append("E:" + "|".join(sorted(edges)))

        rotations = []
        for vid in sorted(self.sigma, key=lambda v: name_of.get(v, v)):
            texts = [self.H[h].text for h in self.sigma[vid]]
            if texts:
                start = _least_rotation(texts + texts, len(texts))
                texts = texts[start:] + texts[:start]
            rotations.append(name_of.get(vid, vid) + "\x1e" + "\x1f".join(texts))
        parts.append("S:" + "|".join(rotations))

        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]

    # ------------------------------------------------- rotation local search --
    def count_faces(self) -> int:
        """Number of ``phi``-orbits.  O(|H|), no ``Face`` objects allocated.

        ``faces()`` builds ids, edge tuples and a union-find partition; the
        local search below evaluates this once per candidate move, so it needs
        the bare count and nothing else.
        """
        visited: set[str] = set()
        n = 0
        for start in self.H:
            if start in visited:
                continue
            n += 1
            h = start
            while h not in visited:
                visited.add(h)
                h = self.phi(h)
        return n

    def transpose_sigma(self, vertex_id: str, i: int, j: int) -> None:
        """Swap two half-edges in the cyclic order at ``vertex_id``.

        This is the move that changes the *surface*.  Whitehead flips do not:
        contracting and re-expanding an edge is a spine move, which is why
        :meth:`whitehead_flip` asserts that genus and ``F`` are unchanged.  To
        alter the embedding you have to alter a rotation, and the smallest such
        alteration is a transposition -- the standard local move on rotation
        systems.
        """
        lst = self.sigma[vertex_id]
        if not (0 <= i < len(lst) and 0 <= j < len(lst)):
            raise FatGraphError(
                f"transposition ({i}, {j}) out of range at {vertex_id!r} "
                f"(degree {len(lst)})"
            )
        lst[i], lst[j] = lst[j], lst[i]
        self._reindex_vertex(vertex_id)

    def maximize_faces(self, max_passes: int = 4, max_degree_scan: int = 48) -> dict:
        """Hill-climb on ``sigma`` to maximise ``F``, i.e. to minimise genus.

        Why this is the principled objective, and not a knob: for a connected
        ribbon graph Euler gives ``F = 2 - 2g + E - V``, so with ``V`` and ``E``
        fixed by the extracted memory, **more faces is exactly less genus**, and
        more faces over the same half-edges means *shorter* faces.

        That matters because the measured failure of face-based retrieval was
        face *length*: ordering sigma by timestamp yields a high-genus embedding
        with a handful of enormous boundary walks (310-348 half-edges on real
        LoCoMo graphs), and a walk that long is a budget sink rather than a
        narrative unit.  Nothing about the memory requires that shape -- it is
        an artefact of having picked the rotation by clock time.  sigma is a
        free parameter of the ribbon structure, and this is the theory's own
        objective for choosing it.

        Vertices of degree <= 2 are skipped: every rotation on them is the same
        cyclic order, so no transposition there can change anything.

        The neighbourhood is *all* pairs within a vertex, not just adjacent
        ones -- adjacent-only converges within a handful of moves on these
        graphs, far short of the Euler ceiling.  ``max_degree_scan`` bounds the
        pair enumeration on the speaker hubs, whose degree runs into the
        hundreds and would otherwise dominate the cost quadratically.

        Returns a report; the graph is mutated in place.
        """
        before = self.count_faces()
        best = before
        moves = 0
        evaluated = 0
        for _ in range(max(1, max_passes)):
            improved = False
            for vid in list(self.sigma):
                deg = len(self.sigma[vid])
                if deg <= 2:
                    continue
                span = min(deg, max_degree_scan) if max_degree_scan else deg
                for i in range(span):
                    for j in range(i + 1, deg):
                        self.transpose_sigma(vid, i, j)
                        evaluated += 1
                        got = self.count_faces()
                        if got > best:
                            best = got
                            moves += 1
                            improved = True
                        else:
                            self.transpose_sigma(vid, i, j)  # revert
            if not improved:
                break
        self._components_cache = None
        return {
            "faces_before": before,
            "faces_after": best,
            "moves_applied": moves,
            "transpositions_evaluated": evaluated,
        }

    # --------------------------------------------------------- invariants ---
    def check_invariants(self) -> None:
        """Raise :class:`InvariantError` if the combinatorial structure broke."""
        for h, a in self.alpha.items():
            if h not in self.H:
                raise InvariantError(f"alpha references unknown half-edge {h!r}")
            if a == h:
                raise InvariantError(f"alpha has a fixed point at {h!r}")
            if self.alpha.get(a) != h:
                raise InvariantError(f"alpha is not an involution at {h!r}")
            if self.H[h].edge_id != self.H[a].edge_id:
                raise InvariantError(f"half-edges {h!r}/{a!r} disagree on edge_id")
            for attr in EDGE_LEVEL_ATTRS:
                if getattr(self.H[h], attr) != getattr(self.H[a], attr):
                    raise InvariantError(
                        f"edge-level attribute {attr!r} out of sync on edge "
                        f"{self.H[h].edge_id}"
                    )
        rebuilt: dict[str, list[str]] = {}
        for hid, he in self.H.items():
            rebuilt.setdefault(he.edge_id, []).append(hid)
        for eid, hs in rebuilt.items():
            hs.sort(key=lambda h: int(h[1:]))
            if self._edge_index.get(eid) != hs:
                raise InvariantError(f"stale edge index for {eid!r}")
        if set(self._edge_index) != set(rebuilt):
            raise InvariantError("edge index and half-edge set disagree")

        seen: set[str] = set()
        for vid, lst in self.sigma.items():
            if vid not in self.vertices:
                raise InvariantError(f"sigma references unknown vertex {vid!r}")
            if len(set(lst)) != len(lst):
                raise InvariantError(f"sigma[{vid!r}] contains duplicates")
            for i, h in enumerate(lst):
                if h not in self.H:
                    raise InvariantError(f"sigma references unknown half-edge {h!r}")
                if self.H[h].vertex_id != vid:
                    raise InvariantError(f"half-edge {h!r} is not glued to {vid!r}")
                if self._sigma_pos.get(h) != i:
                    raise InvariantError(f"stale sigma index for {h!r}")
                seen.add(h)
        if seen != set(self.H):
            raise InvariantError("some half-edges are not present in sigma")
        self.euler()  # raises TopologyViolation on a non-integral genus

    # ------------------------------------------------------------- stats ----
    def stats(self) -> dict:
        """Graph statistics used by the experimental report."""
        fs = self.faces()
        lengths = [f.length for f in fs]
        by_state: dict[str, int] = {s: 0 for s in VALID_STATES}
        n_level2 = 0
        n_shadowed = 0
        for eid in self.edges():
            h1, _ = self.edge_half_edges(eid)
            by_state[self.H[h1].state] = by_state.get(self.H[h1].state, 0) + 1
            n_level2 += int(self.H[h1].level == 2)
            n_shadowed += int(self.H[h1].shadowed)
        e = self.euler()
        return {
            **e.to_dict(),
            "n_faces_nontrivial": sum(1 for f in fs if f.length > 0),
            "face_length_hist": _histogram(lengths),
            "face_length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "face_length_max": max(lengths) if lengths else 0,
            "n_leaf_faces": sum(1 for f in fs if f.is_leaf_face),
            "n_bigon_faces": sum(1 for f in fs if f.length == 2 and not f.is_leaf_face),
            "edges_by_state": by_state,
            "n_level2_edges": n_level2,
            "n_shadowed_edges": n_shadowed,
            "degree_mean": (
                float(np.mean([len(v) for v in self.sigma.values()])) if self.sigma else 0.0
            ),
            **self.star_stats(),
            #: lets "differs only in retrieval" be verified rather than assumed
            "fingerprint": self.fingerprint(),
        }

    def star_stats(self) -> dict:
        """How close this graph is to a star -- the shape that kills multi-hop.

        Both retrieval mechanisms downstream have a topological precondition,
        and it is the *same* one, so it is worth one number rather than a
        post-mortem:

        * sigma expansion (G4) is redundant exactly when the anchor's
          neighbours have degree 1.  There ``sigma(alpha(h)) = alpha(h)``, so
          ``phi`` degenerates into marching along the hub's own orbit and the
          face already delivers every sigma-neighbour, in order.  Nothing to
          expand -- this is a theorem about the shape, not a tuning problem.
        * face coverage (G5) needs faces to *differ* in which entities they
          touch.  A star has a handful of enormous faces that touch nearly
          every vertex, so coverage is ~1 for all of them and ranks nothing.

        ``degree_1_frac`` catches the first, ``hub_share`` (the share of
        half-edges sitting on the two highest-degree vertices) catches the
        second: in a dialogue graph those two are the speakers, and a high
        share means facts are being attached to whoever *said* them instead of
        to what they are *about*.
        """
        degrees = sorted((len(v) for v in self.sigma.values()), reverse=True)
        n_h = len(self.H)
        return {
            "degree_1_frac": (
                round(sum(1 for d in degrees if d == 1) / len(degrees), 4)
                if degrees
                else 0.0
            ),
            "hub_share": round(sum(degrees[:2]) / n_h, 4) if n_h else 0.0,
        }

    # --------------------------------------------------------- persistence --
    def serialize(self) -> dict:
        """JSON-serialisable snapshot (embeddings excluded, see :meth:`save`)."""
        return {
            "version": 1,
            "counters": {
                "half_edge": self._he_counter,
                "edge": self._edge_counter,
                "vertex": self._vertex_counter,
            },
            "vertices": {v: vx.to_dict() for v, vx in self.vertices.items()},
            "half_edges": {h: he.to_dict() for h, he in self.H.items()},
            "alpha": self.alpha,
            "sigma": self.sigma,
        }

    @classmethod
    def deserialize(
        cls,
        payload: dict,
        embeddings: dict[str, np.ndarray] | None = None,
        vertex_embeddings: dict[str, np.ndarray] | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> "FatGraph":
        g = cls(token_counter=token_counter)
        for vid, vd in payload["vertices"].items():
            vx = Vertex(**vd)
            if vertex_embeddings and vid in vertex_embeddings:
                vx.embedding = vertex_embeddings[vid]
            g.vertices[vid] = vx
        for hid, hd in payload["half_edges"].items():
            he = HalfEdge(**hd)
            if embeddings and hid in embeddings:
                he.embedding = embeddings[hid]
            g.H[hid] = he
            g._edge_index.setdefault(he.edge_id, []).append(hid)
        for eid, hs in g._edge_index.items():
            hs.sort(key=lambda h: int(h[1:]))
        g.alpha = dict(payload["alpha"])
        g.sigma = {k: list(v) for k, v in payload["sigma"].items()}
        for vid in g.sigma:
            g._reindex_vertex(vid)
        c = payload.get("counters", {})
        g._he_counter = c.get("half_edge", len(g.H))
        g._edge_counter = c.get("edge", len(g.edges()))
        g._vertex_counter = c.get("vertex", len(g.vertices))
        return g

    def save(self, path: str | "os.PathLike") -> None:
        """Write ``<path>.json`` and ``<path>.npz`` (embeddings)."""
        import os
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.with_suffix(".json").write_text(
            json.dumps(self.serialize(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        arrays = {
            f"he::{h}": he.embedding
            for h, he in self.H.items()
            if he.embedding is not None
        }
        arrays.update(
            {
                f"v::{v}": vx.embedding
                for v, vx in self.vertices.items()
                if vx.embedding is not None
            }
        )
        np.savez_compressed(p.with_suffix(".npz"), **arrays)

    @classmethod
    def load(
        cls, path: str | "os.PathLike", token_counter: Callable[[str], int] | None = None
    ) -> "FatGraph":
        from pathlib import Path

        p = Path(path)
        payload = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        he_emb: dict[str, np.ndarray] = {}
        v_emb: dict[str, np.ndarray] = {}
        npz = p.with_suffix(".npz")
        if npz.exists():
            with np.load(npz) as data:
                for key in data.files:
                    kind, _, ident = key.partition("::")
                    (he_emb if kind == "he" else v_emb)[ident] = data[key]
        return cls.deserialize(payload, he_emb, v_emb, token_counter=token_counter)

    # ----------------------------------------------------------- dunders ----
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        e = self.euler()
        return (
            f"<FatGraph V={e.V} E={e.E} F={e.F} C={e.C} g={e.genus} "
            f"|H|={len(self.H)}>"
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


class _FactView:
    """Uniform read-only access to a fact given as a dataclass or a dict."""

    def __init__(self, obj) -> None:
        self._o = obj
        self._is_dict = isinstance(obj, dict)

    def _get(self, name, default=None):
        if self._is_dict:
            return self._o.get(name, default)
        return getattr(self._o, name, default)

    text = property(lambda self: self._get("text", "") or self._get("fact_text", ""))
    embedding = property(lambda self: self._get("embedding"))
    session_id = property(lambda self: self._get("session_id", "") or "")
    turn_ids = property(lambda self: self._get("turn_ids", []) or [])
    timestamp = property(lambda self: self._get("timestamp", "") or "")
    state = property(lambda self: self._get("state", STATE_EMERGENT) or STATE_EMERGENT)
    level = property(lambda self: int(self._get("level", 1) or 1))
    shadowed = property(lambda self: bool(self._get("shadowed", False)))
    children = property(lambda self: self._get("children", []) or [])
    provenance = property(lambda self: self._get("provenance", []) or [])
    meta = property(lambda self: self._get("meta", {}) or {})

    def text_from(self, vertex_name: str, side: int) -> str:
        """Perspective text for a half-edge glued to ``vertex_name``."""
        explicit = self._get(f"text_from_v{side}")
        if explicit:
            return explicit
        return self.text


def _copy_value(value):
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _ordered_union(*seqs: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for seq in seqs:
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _rotate_after(lst: Sequence[str], pivot: str) -> list[str]:
    """The list read cyclically starting just after ``pivot``, pivot removed."""
    i = list(lst).index(pivot)
    rot = list(lst[i + 1 :]) + list(lst[: i])
    return rot


def _histogram(values: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def face_id(edges: Sequence[str]) -> str:
    """Stable id of a face from its **ordered cyclic** sequence of edge ids.

    The canonical form is the lexicographically smallest rotation of the
    sequence, so the id does not depend on which half-edge started the walk but
    *does* change as soon as an edge is inserted into or removed from the face
    -- which is exactly the signal ``consolidation`` needs to measure stability.
    (Using the *set* of edge ids, as the original spec suggested, collides
    whenever two faces share an edge set; see COERENCIA.md item C6.)
    """
    if not edges:
        return "face:empty"
    n = len(edges)
    doubled = list(edges) * 2
    start = _least_rotation(doubled, n)
    canon = doubled[start : start + n]
    digest = hashlib.sha1("|".join(canon).encode("utf-8")).hexdigest()[:16]
    return f"face:{digest}"


def _least_rotation(doubled: list[str], n: int) -> int:
    """Booth's algorithm: index of the lexicographically least rotation. O(n).

    ``doubled`` must be the sequence concatenated with itself.  The naive
    ``min(range(n), key=lambda i: doubled[i:i+n])`` is O(n^2), which dominates
    ``faces()`` as soon as faces get long -- and on LoCoMo memory graphs they do
    (COERENCIA.md C9: single faces of length 200+).
    """
    f = [-1] * (2 * n)
    k = 0
    for j in range(1, 2 * n):
        sj = doubled[j]
        i = f[j - k - 1]
        while i != -1 and sj != doubled[k + i + 1]:
            if sj < doubled[k + i + 1]:
                k = j - i - 1
            i = f[i]
        if sj != doubled[k + i + 1]:
            if sj < doubled[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k % n


def _assert_topology_preserved(
    before: EulerStats,
    after: EulerStats,
    expect_delta_E: int = 0,
    expect_delta_F: int = 0,
) -> None:
    if after.C != before.C:
        raise TopologyViolation(
            f"component count changed: {before.C} -> {after.C}"
        )
    if after.genus != before.genus:
        raise TopologyViolation(f"genus changed: {before.genus} -> {after.genus}")
    if after.E - before.E != expect_delta_E:
        raise TopologyViolation(
            f"E changed by {after.E - before.E}, expected {expect_delta_E}"
        )
    if after.F - before.F != expect_delta_F:
        raise TopologyViolation(
            f"F changed by {after.F - before.F}, expected {expect_delta_F}"
        )


# Typing alias used above -- kept at the bottom to avoid a forward reference.
FactLike = object
