"""Phase 6 (spec section 8): folding compatible edges together, which
resolves entities as a SIDE EFFECT rather than as its own pass -- there is
no separate "entity resolution module" in CLIO, by design (spec 8's
opening line).

Two deliberate deviations from spec 8.2's literal formula, both because
this package avoids a dependency this repository has no other use for and
because its own precedent already points elsewhere:

* ``name_similarity`` uses :mod:`difflib` (stdlib), not Jaro-Winkler. A
  hand-rolled string-edit-distance algorithm is easy to get subtly wrong,
  and this repository's own entity resolver
  (:mod:`fgl.memory.entities`) already prefers exact match + embedding
  cosine over edit distance for this exact task. ``difflib.SequenceMatcher
  .ratio()`` serves the same purpose (reward near-identical strings)
  without adding a new dependency.
* There is no embedding-similarity term here, even though M5's extraction
  context builder uses one for candidate search: two DIFFERENT surface
  forms with no lexical overlap ("Bob" / "Robert") are a real miss this
  scorer will not catch, and closing that gap by wiring an embedder
  through fold is a reasonable extension -- just not one any test in this
  repository currently forces, so it is left as a documented gap rather
  than an unexercised code path.
"""

from __future__ import annotations

import difflib
import heapq
import itertools
import re
from dataclasses import dataclass
from typing import Any

from fgl.clio.catalog import Catalog
from fgl.clio.consolidate.journal import FoldJournal, FoldRecord
from fgl.clio.graph.store import GraphStore
from fgl.clio.log.store import LogStore
from fgl.clio.types import Entity

# --- weights, spec 8.2 (kept as specified: an exact/alias match already  #
# --- returns 1.0 from name_similarity below, covering what the spec      #
# --- splits into a separate Jaro-Winkler term and an alias-exact term).  #
W_NAME = 0.35
W_CONTAINED = 0.20
W_STRUCT = 0.20
W_ROLE = 0.15
W_TEMPORAL = 0.10
P_DISTINCT = 0.9

_DISTINCTION_RE = re.compile(
    r"\b(the other|a different|another(?:\s+\w+)?\s+named|not that)\b", re.IGNORECASE
)


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def mentions_distinction(text: str, name: str) -> bool:
    """True if ``text`` both carries a disambiguation marker ("the other",
    "a different", ...) and names ``name`` -- the single-name form phase 1
    uses to decide whether to reuse an EXACT name match at all (spec 7.2
    doesn't ask for this, but without it two different people who happen
    to share one exact name are unified before fold ever gets a chance to
    tell them apart, since exact-match reuse runs first and unconditionally
    otherwise)."""
    lowered = text.lower()
    return bool(_DISTINCTION_RE.search(lowered)) and name.strip().lower() in lowered


def name_similarity(a: str, b: str) -> float:
    """Exact/alias match is 1.0. One name being a literal prefix/substring
    of the other ("Rui" in "Rui Sampaio") is scored high but not perfect --
    a strong nickname/partial-name signal, not proof: spec's own worked
    example folds on exactly this pattern, and plain
    ``difflib.SequenceMatcher`` ratio penalises it too heavily for the
    length difference alone to explain (0.43 for "rui" vs "rui sampaio",
    which no combination of the OTHER signals then lifts across
    ``tau_fold``'s default 0.80 -- checked, not assumed).
    """
    a_n, b_n = _normalize(a), _normalize(b)
    if a_n == b_n:
        return 1.0
    if a_n in b_n or b_n in a_n:
        return 0.9
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def contains_as_token(a: str, b: str) -> float:
    """1.0 when the shorter name's tokens are a SUBSET of the longer
    name's ("Rui" subset-of "Rui Sampaio"), 0.0 otherwise. A subset check,
    not Jaccard: partial credit for "close but not quite contained" would
    double-count what ``name_similarity`` already rewards, and containment
    itself is a binary fact, not a matter of degree.
    """
    ta, tb = set(_normalize(a).split()), set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return 1.0 if shorter <= longer else 0.0


def _neighbors(vertex_id: str, graph: GraphStore) -> set[str]:
    out = set()
    for e in graph.edges_incident(vertex_id, live_only=False):
        other = e.dst_id if e.src_id == vertex_id else e.src_id
        if other != vertex_id:
            out.add(other)
    return out


def neighbor_overlap(a_id: str, b_id: str, graph: GraphStore) -> float:
    """Jaccard of the two vertices' neighbourhoods: "connected to the
    same things"."""
    na, nb = _neighbors(a_id, graph), _neighbors(b_id, graph)
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def _roles(vertex_id: str, graph: GraphStore) -> set[tuple[str, str | None]]:
    out: set[tuple[str, str | None]] = set()
    for e in graph.edges_incident(vertex_id, live_only=False):
        if e.dst_id == vertex_id:
            out.add((e.label, e.src_id))
        if e.src_id == vertex_id:
            out.add((e.label, e.dst_id))
    return out


def same_role_context(a_id: str, b_id: str, graph: GraphStore) -> float:
    """1.0 if the two vertices share an incidence -- same (label, other
    endpoint) pair, e.g. both are the object of a ``hired`` proposition
    from the same subject."""
    return 1.0 if (_roles(a_id, graph) & _roles(b_id, graph)) else 0.0


def temporal_compatibility(
    a_id: str, b_id: str, graph: GraphStore, catalog: Catalog
) -> float:
    """0.0 only on a real structural conflict: a functional relation
    where ``a`` and ``b`` each hold a DIFFERENT value over an OVERLAPPING
    window -- the shape phase 4 (cardinality) would itself flag once
    merged. Spec 8.2 only says "intervals don't conflict"; this is the
    concrete, checkable reading of that.
    """
    a_out = [e for e in graph.edges_incident(a_id, live_only=False) if e.src_id == a_id]
    b_out = [e for e in graph.edges_incident(b_id, live_only=False) if e.src_id == b_id]
    for ea in a_out:
        spec = catalog.get(ea.label)
        if spec is None or spec.cardinality != "functional":
            continue
        for eb in b_out:
            if eb.label != ea.label or eb.dst_id == ea.dst_id:
                continue
            if ea.t_valid.overlaps(eb.t_valid):
                return 0.0
    return 1.0


def explicit_distinction(a: Entity, b: Entity, log: LogStore) -> float:
    """1.0 if some episode explicitly marks the two names as different
    people ("the other Rui", "a different Bob") -- spec 8.2: a single
    such marker should be enough to block a fold."""
    names = {
        a.canonical_name,
        b.canonical_name,
        *a.aliases,
        *b.aliases,
    }
    return (
        1.0
        if any(mentions_distinction(ep.text, n) for ep in log.all() for n in names)
        else 0.0
    )


def identity_score(
    a: Entity, b: Entity, graph: GraphStore, log: LogStore, catalog: Catalog
) -> float:
    if a.type != b.type:
        return 0.0
    s = 0.0
    s += W_NAME * name_similarity(a.canonical_name, b.canonical_name)
    s += W_CONTAINED * contains_as_token(a.canonical_name, b.canonical_name)
    s += W_STRUCT * neighbor_overlap(a.id, b.id, graph)
    s += W_ROLE * same_role_context(a.id, b.id, graph)
    s += W_TEMPORAL * temporal_compatibility(a.id, b.id, graph, catalog)
    s -= P_DISTINCT * explicit_distinction(a, b, log)
    return max(0.0, min(1.0, s))


# --------------------------------------------------------------------- #
# C1-C4, spec 8.1 -- C1 (same address) and C3 (temporal adjacency) as a   #
# hard pre-filter are gone; see fold()'s own docstring for why C1 does    #
# not reach spec's own worked example, and temporal_compatibility above  #
# for why C3 is now a scoring term instead of a gate. C2 (type match) and #
# C4 (tau_fold) remain, enforced directly in enqueue_from below.          #
# --------------------------------------------------------------------- #


class _UnionFind:
    def __init__(self, items: Any) -> None:
        self._parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _choose_kept(a: Entity, b: Entity) -> tuple[str, str]:
    """Which vertex survives: a confirmed (non-provisional) vertex over a
    provisional one, and otherwise the one that has existed longer (lower
    id -- ids are assigned in creation order). Arbitrary in the sense that
    either choice preserves the same facts, but it must be DETERMINISTIC:
    spec 17.4's order-invariance test is exactly the property this
    tie-break exists to protect.
    """
    if a.provisional != b.provisional:
        return (b.id, a.id) if a.provisional else (a.id, b.id)
    return (a.id, b.id) if a.id < b.id else (b.id, a.id)


@dataclass(frozen=True)
class FoldConfig:
    tau_fold: float = 0.80


def fold(
    scope: set[str],
    graph: GraphStore,
    log: LogStore,
    catalog: Catalog,
    journal: FoldJournal,
    config: FoldConfig,
    trigger_episode: str,
) -> list[FoldRecord]:
    """Folds compatible vertices reachable from ``scope`` (vertex ids
    touched this consolidation round) to a fixed point.

    Candidate pairs are generated by type -- not, as spec 8.1's literal C1
    has it, by "same origin and same label" (i.e. two DESTINATIONS of the
    same address). That restriction was checked against spec's own worked
    example and does not reach it: "Rui" is the OBJECT of a ``managed_by``
    edge and "Rui Sampaio" is the SUBJECT of a ``hired`` edge, so the two
    never share an address for ``edges_at`` to compare them at all. Every
    other C1-shaped condition here (C2 type match, C3 temporal
    compatibility) still holds -- C3 has moved from a hard gate into
    :func:`temporal_compatibility`'s contribution to the score itself,
    since a real conflict should lower confidence, not veto a merge that
    overwhelming name/structural evidence would otherwise justify.

    Merging two vertices creates new candidate pairs at whatever else they
    were both connected to -- spec 8.3's example: once "Rui" and "Rui
    Sampaio" merge, ``hired`` and ``managed_by`` are suddenly incident on
    the same vertex, and a path that was never observed in any one episode
    becomes walkable. That is why this loop re-enqueues from the merged
    vertex instead of running once over ``scope``.
    """
    uf = _UnionFind(e.id for e in graph.all_entities())
    heap: list[tuple[float, int, str, str]] = []
    counter = itertools.count()

    def enqueue_from(vertex_id: str) -> None:
        v_id = uf.find(vertex_id)
        v = graph.get_entity(v_id)
        for other in graph.all_entities():
            if other.merged_into is not None or other.type != v.type:
                continue  # C2
            o_id = uf.find(other.id)
            if o_id == v_id:
                continue
            score = identity_score(v, graph.get_entity(o_id), graph, log, catalog)
            if score >= config.tau_fold:  # C4
                heapq.heappush(heap, (-score, next(counter), v_id, o_id))

    for vertex_id in scope:
        if graph.get_entity(vertex_id).merged_into is None:
            enqueue_from(vertex_id)

    records: list[FoldRecord] = []
    while heap:
        neg_score, _, v1_id, v2_id = heapq.heappop(heap)
        v1_id, v2_id = uf.find(v1_id), uf.find(v2_id)
        if v1_id == v2_id:
            continue  # already merged via a different pair since being enqueued

        v1, v2 = graph.get_entity(v1_id), graph.get_entity(v2_id)
        kept_id, absorbed_id = _choose_kept(v1, v2)
        kept, absorbed = graph.get_entity(kept_id), graph.get_entity(absorbed_id)

        migrated = [e.id for e in graph.edges_incident(absorbed_id, live_only=False)]
        snapshot = {
            "canonical_name": absorbed.canonical_name,
            "type": absorbed.type,
            "aliases": list(absorbed.aliases),
        }
        rec = journal.append(
            kept=kept_id,
            absorbed=absorbed_id,
            score=-neg_score,
            trigger_episode=trigger_episode,
            migrated_edge_ids=migrated,
            snapshot=snapshot,
        )
        uf.union(kept_id, absorbed_id)
        graph.migrate_edges(absorbed_id, kept_id)
        graph.mark_alias(absorbed_id, kept_id)
        if (
            absorbed.canonical_name not in kept.aliases
            and absorbed.canonical_name != kept.canonical_name
        ):
            kept.aliases.append(absorbed.canonical_name)
        records.append(rec)

        enqueue_from(kept_id)

    return records


def unfold(fold_id: str, journal: FoldJournal, graph: GraphStore) -> None:
    """Reverts one merge, and every later merge that depended on it (spec
    8.4). A structural reversal -- edges and aliases move back exactly as
    recorded -- not a full re-run of consolidation (spec 12.3's
    ``clio.rebuild`` is the tool for that if the underlying propositions
    should be re-evaluated from scratch)."""
    rec = journal.get(fold_id)
    for dependent in reversed(journal.folds_after(fold_id, touching=rec.kept)):
        _revert_single(dependent, graph)
    _revert_single(rec, graph)


def _revert_single(rec: FoldRecord, graph: GraphStore) -> None:
    if rec.reverted:
        return
    graph.migrate_specific_edges(set(rec.migrated_edge_ids), rec.kept, rec.absorbed)
    graph.unmark_alias(rec.absorbed)
    kept = graph.get_entity(rec.kept)
    absorbed_name = rec.snapshot.get("canonical_name")
    if absorbed_name in kept.aliases:
        kept.aliases.remove(absorbed_name)
    rec.reverted = True
