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


#: shortest abbreviation taken seriously. Two characters ("Al", "Jo") are
#: too weak a signal to merge two people on; three is where nicknames
#: actually live ("Mel", "Car", "Ben").
_MIN_ABBREVIATION_CHARS = 3


def _is_abbreviation(a_n: str, b_n: str) -> bool:
    """One single-token name is a shortened form of the other: "mel" ->
    "melanie", "car" -> "caroline".

    This is the dominant identity shape in dialogue and the old scorer
    could not see it at all. ``contains_as_token`` demanded WHOLE-token
    containment, so a truncated first name scored 0 there, and the pair
    topped out at 0.765 even with both structural and role evidence --
    permanently under ``tau_fold``. Measured on conv-26, that left "Mel"
    and "Melanie" as two different people, each holding half her facts.
    """
    if " " in a_n or " " in b_n:
        return False
    short, long_ = sorted((a_n, b_n), key=len)
    if len(short) < _MIN_ABBREVIATION_CHARS or short == long_:
        return False
    return long_.startswith(short)


def _is_name_extension(a_n: str, b_n: str) -> bool:
    """The shorter name's tokens are a leading PREFIX of the longer's:
    "rui" -> "rui sampaio", "melanie" -> "melanie cruz".

    A prefix, deliberately, not a subset. A personal name grows by
    APPENDING (a surname, a middle name); a word sitting in the middle of
    a descriptive phrase is that phrase being ABOUT the thing, not naming
    it. The subset rule could not tell those apart, and on conv-26 it
    merged the vertex "family" into "Our family and moments" at 0.86 --
    scoring 0.9 for substring similarity AND a further 1.0 for token
    containment, counting one weak piece of evidence twice.
    """
    ta, tb = _normalize(a_n).split(), _normalize(b_n).split()
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return len(short) < len(long_) and long_[: len(short)] == short


def name_similarity(a: str, b: str) -> float:
    """Exact/alias match is 1.0. A shortened first name or a name the
    other extends by appending scores high but not perfect -- a strong
    signal, not proof.

    Plain ``difflib.SequenceMatcher`` ratio cannot carry either case (0.43
    for "rui" vs "rui sampaio"), which is why they are recognised
    explicitly; everything else falls through to the ratio, INCLUDING a
    word merely embedded in a longer phrase, which used to be scored as
    though it were a name.
    """
    a_n, b_n = _normalize(a), _normalize(b)
    if a_n == b_n:
        return 1.0
    if _is_abbreviation(a_n, b_n) or _is_name_extension(a_n, b_n):
        return 0.9
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def contains_as_token(a: str, b: str) -> float:
    """1.0 when one name is a morphological narrowing of the other -- a
    leading token prefix ("Rui" -> "Rui Sampaio") or a shortened single
    token ("Mel" -> "Melanie") -- and 0.0 otherwise.

    Binary, not a matter of degree, and NOT the old subset test. A subset
    match fires for any word appearing anywhere in a longer phrase, which
    is how a common noun gets absorbed into a description that merely
    mentions it. Requiring the containment to be positional keeps the
    signal this term was meant to carry (a name extended by a surname)
    and drops the one it was accidentally carrying.
    """
    a_n, b_n = _normalize(a), _normalize(b)
    if not a_n or not b_n or a_n == b_n:
        return 0.0
    return 1.0 if (_is_abbreviation(a_n, b_n) or _is_name_extension(a_n, b_n)) else 0.0


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


def distinction_index(log: LogStore) -> list[str]:
    """The lowercased text of every episode carrying a distinction marker,
    computed ONCE per fold call.

    :func:`explicit_distinction` used to re-run the marker regex over the
    WHOLE log, for every name, for every candidate pair -- and fold
    enumerates pairs quadratically. That was millions of regex searches
    per consolidation call and it dominated the 5.2s measured before this.
    The markers depend only on the log, which fold never modifies, so they
    are a constant for the whole call.

    Kept as a list of texts rather than a set of tokens on purpose: the
    check has to stay a SUBSTRING test, because a name can be several
    words ("the other Rui Sampaio"). This is exactly the old predicate
    with the episodes that cannot possibly match filtered out first --
    marker-bearing episodes are a tiny fraction of any real log.
    """
    return [ep.text.lower() for ep in log.all() if _DISTINCTION_RE.search(ep.text.lower())]


def explicit_distinction(
    a: Entity, b: Entity, log: LogStore, index: list[str] | None = None
) -> float:
    """1.0 if some episode explicitly marks the two names as different
    people ("the other Rui", "a different Bob") -- spec 8.2: a single
    such marker should be enough to block a fold."""
    names = {a.canonical_name, b.canonical_name, *a.aliases, *b.aliases}
    if index is not None:
        return 1.0 if any(_normalize(n) in text for text in index for n in names) else 0.0
    return (
        1.0
        if any(mentions_distinction(ep.text, n) for ep in log.all() for n in names)
        else 0.0
    )


def identity_score(
    a: Entity,
    b: Entity,
    graph: GraphStore,
    log: LogStore,
    catalog: Catalog,
    distinctions: list[str] | None = None,
) -> float:
    # C2, read through the catalog's declared type classes rather than as
    # strict equality: "the charity race" typed Activity by `practices`
    # and Event by `attended` is one thing, and a bare `!=` here made it
    # permanently two.
    if not catalog.types_compatible(a.type, b.type):
        return 0.0
    s = 0.0
    s += W_NAME * name_similarity(a.canonical_name, b.canonical_name)
    s += W_CONTAINED * contains_as_token(a.canonical_name, b.canonical_name)
    s += W_STRUCT * neighbor_overlap(a.id, b.id, graph)
    s += W_ROLE * same_role_context(a.id, b.id, graph)
    s += W_TEMPORAL * temporal_compatibility(a.id, b.id, graph, catalog)
    s -= P_DISTINCT * explicit_distinction(a, b, log, distinctions)
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
    distinctions = distinction_index(log)
    merged_vertices: set[str] = set()

    def enqueue_from(vertex_id: str) -> None:
        v_id = uf.find(vertex_id)
        v = graph.get_entity(v_id)
        for other in graph.all_entities():
            if other.merged_into is not None or not catalog.types_compatible(
                other.type, v.type
            ):
                continue  # C2
            o_id = uf.find(other.id)
            if o_id == v_id:
                continue
            score = identity_score(
                v, graph.get_entity(o_id), graph, log, catalog, distinctions
            )
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
        merged_vertices.add(kept_id)

    for vertex_id in merged_vertices:
        reconcile_duplicate_edges(vertex_id, graph)

    return records


def reconcile_duplicate_edges(vertex_id: str, graph: GraphStore) -> int:
    """Merges edges that became identical because a fold merged their
    endpoints. Returns how many rows were absorbed.

    Folding runs after phase 3, so two edges that were legitimately
    distinct -- different destinations -- can become the same fact once
    those destinations turn out to be one vertex. Spec 8.3 migrates the
    edges and stops there; nothing reconciles them, and phase 4 will not,
    because it skips pairs sharing a destination. Observed on conv-26:
    "Melanie likes Our family and moments" written twice, each claiming
    reinforcement 1, inflating the edge count and emitting the same vertex
    twice from a single ``follow``.

    Only edges still in force AND still believed are merged: a closed or
    retracted interval is history, and history is never collapsed into the
    present (P2).
    """
    groups: dict[tuple[str, str, str, bool], list] = {}
    for e in graph.edges_incident(vertex_id, live_only=False):
        if e.t_tx.end is not None or e.t_valid.end is not None:
            continue
        groups.setdefault((e.src_id, e.label, e.dst_id, e.polarity), []).append(e)

    absorbed = 0
    for edges in groups.values():
        if len(edges) < 2:
            continue
        edges.sort(key=lambda e: e.id)
        kept = edges[0]
        for duplicate in edges[1:]:
            graph.absorb_edge(kept, duplicate)
            absorbed += 1
    return absorbed


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
