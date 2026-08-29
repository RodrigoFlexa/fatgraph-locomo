"""The eight access movements (spec section 9.2). No routing layer: the
agent composes these directly, and ``available_labels`` on the returned
state is what tells it what it can do next (spec 10.2) instead of any
prior classification of the question.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from fgl.clio.access.ppr import personalized_pagerank
from fgl.clio.access.state import AccessState, Trail
from fgl.clio.catalog import Catalog
from fgl.clio.graph.queries import UnknownLabel
from fgl.clio.graph.store import GraphStore
from fgl.clio.log.mentions import MentionStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.types import Episode, Interval

__all__ = [
    "UnknownLabel",
    "anchor",
    "follow",
    "restrict",
    "filter_trails",
    "expand",
    "history",
    "evidence",
    "count",
    "classify_death",
    "available_labels",
]


# --------------------------------------------------------------------- #
# anchor                                                                  #
# --------------------------------------------------------------------- #


def _lexical_entity_search(text: str, graph: GraphStore, k: int) -> list[str]:
    """Zero-dependency fallback anchor when no :class:`~fgl.clio.index.EntityIndex`
    is given: token-overlap over canonical names/aliases. Good enough for
    tests and small memories; the real path (M5's hybrid lexical+dense
    index) is what ``anchor`` should be given in practice."""
    needle = {w for w in text.lower().split() if len(w) > 2}
    scored = []
    for ent in graph.all_entities():
        if ent.merged_into is not None:
            continue
        names = [ent.canonical_name, *ent.aliases]
        hay = {w for name in names for w in name.lower().split()}
        exact = any(name.lower() in text.lower() for name in names)
        overlap = len(needle & hay)
        if exact or overlap:
            scored.append((exact, overlap, ent.id))
    scored.sort(key=lambda s: (-s[0], -s[1]))
    return [eid for _, _, eid in scored[:k]]


def anchor(
    text: str,
    graph: GraphStore,
    index=None,
    k: int = 5,
    tx_point: datetime | None = None,
) -> AccessState:
    """Entry point: finds vertices to start trails from. ``index`` is an
    :class:`fgl.clio.index.EntityIndex` (hybrid lexical+dense, M5) when
    available; without one, a plain token-overlap search over entity names
    still makes this usable standalone.
    """
    if index is not None:
        entity_ids = [e.id for e in index.search(text, k=k)]
    else:
        entity_ids = _lexical_entity_search(text, graph, k)
    trails = [Trail(vertex_id=eid, window=Interval(None, None)) for eid in entity_ids]
    return AccessState(trails=trails, tx_point=tx_point or datetime.now())


# --------------------------------------------------------------------- #
# follow                                                                   #
# --------------------------------------------------------------------- #


def classify_death(trail: Trail, pairs: list[tuple], tx_point: datetime) -> str:
    """Diagnoses why none of ``pairs`` produced a surviving trail.

    Checked in this order, not the reverse: a candidate whose WINDOW never
    fit is irrelevant to "was this retracted" -- it was never a real
    contender. So "all_edges_retracted" is reported when every candidate
    that *would* have fit temporally has since been retracted, even if
    other, temporally-incompatible edges exist at the same address (spec's
    own T3: Rui's edge doesn't overlap the query window at all, and is not
    what makes the trail die -- Bia's retraction is).
    """
    if not pairs:
        return "no_edge_with_label"
    window_compatible = [
        (e, n) for e, n in pairs if trail.window.intersect(e.t_valid) is not None
    ]
    if not window_compatible:
        return "empty_temporal_window"
    if all(not e.t_tx.contains(tx_point) for e, _ in window_compatible):
        return "all_edges_retracted"
    return "unknown"


def follow(
    state: AccessState, label: str, graph: GraphStore, catalog: Catalog
) -> AccessState:
    """The core movement (spec 9.3): applies ``label``, narrows the
    window to the intersection with the traversed edge, and prunes any
    trail whose window would empty -- coherence is structural, not a
    check that can be skipped (invariant I1)."""
    if not catalog.is_known(label):
        raise UnknownLabel(label)

    new_trails: list[Trail] = []
    dead = 0
    cause: str | None = None
    for t in state.trails:
        pairs = graph.out_edges(t.vertex_id, label, catalog)
        matched = False
        for e, neighbor in pairs:
            if not e.t_tx.contains(state.tx_point):
                continue  # retracted in this transaction-time view
            w = t.window.intersect(e.t_valid)
            if w is None:
                continue  # incoherent with the rest of this trail's path
            new_trails.append(
                Trail(
                    vertex_id=neighbor,
                    window=w,
                    path=t.path + tuple(e.provenance),
                    labels=t.labels + (label,),
                )
            )
            matched = True
        if not matched:
            dead += 1
            cause = classify_death(t, pairs, state.tx_point)

    return AccessState(
        trails=new_trails,
        tx_point=state.tx_point,
        dead_count=dead,
        death_cause=cause if not new_trails else None,
        budget_used=state.budget_used + 1,
    )


# --------------------------------------------------------------------- #
# restrict                                                                 #
# --------------------------------------------------------------------- #


def restrict(
    state: AccessState, axis: Literal["valid", "tx"], interval: Interval
) -> AccessState:
    """``axis="valid"`` narrows every trail's window (a real temporal
    filter -- can kill trails). ``axis="tx"`` changes the point of view
    ("what did the agent believe as of ``interval.start``") for whatever
    ``follow`` comes next; it does not retroactively re-validate trails
    already built under the old point of view, matching spec's own T3/T4
    trace, where the belief-history trail is recovered by restricting
    ``tx`` and then following again -- not by rewinding what already
    happened.
    """
    if axis == "tx":
        return AccessState(
            trails=state.trails,
            tx_point=interval.start or state.tx_point,
            budget_used=state.budget_used + 1,
        )
    if axis == "valid":
        new_trails = []
        for t in state.trails:
            w = t.window.intersect(interval)
            if w is not None:
                new_trails.append(Trail(t.vertex_id, w, t.path, t.labels))
        dead = len(state.trails) - len(new_trails)
        return AccessState(
            trails=new_trails,
            tx_point=state.tx_point,
            dead_count=dead,
            death_cause="empty_temporal_window" if dead and not new_trails else None,
            budget_used=state.budget_used + 1,
        )
    raise ValueError(f"restrict: axis must be 'valid' or 'tx', got {axis!r}")


# --------------------------------------------------------------------- #
# filter                                                                   #
# --------------------------------------------------------------------- #


def filter_trails(
    state: AccessState,
    graph: GraphStore,
    name: str | None = None,
    type: str | None = None,
) -> AccessState:
    """Keeps only trails whose vertex matches ``name`` and/or ``type``.
    Named ``filter_trails``, not ``filter``, to avoid shadowing the
    builtin -- the tool-facing name (``memory_filter``) is unaffected."""

    def ok(t: Trail) -> bool:
        ent = graph.get_entity(t.vertex_id)
        if name is not None:
            needle = name.strip().lower()
            names = {
                ent.canonical_name.strip().lower(),
                *(a.strip().lower() for a in ent.aliases),
            }
            if needle not in names:
                return False
        return not (type is not None and ent.type != type)

    kept = [t for t in state.trails if ok(t)]
    dead = len(state.trails) - len(kept)
    return AccessState(
        trails=kept,
        tx_point=state.tx_point,
        dead_count=dead,
        death_cause="filtered_out" if dead and not kept else None,
        budget_used=state.budget_used + 1,
    )


# --------------------------------------------------------------------- #
# expand                                                                   #
# --------------------------------------------------------------------- #


def expand(
    state: AccessState,
    graph: GraphStore,
    k: int = 2,
    expand_k: int = 10,
    alpha: float = 0.15,
) -> AccessState:
    """Associative spreading activation (spec 9.5) for when the question
    does not name a relation. Expansion trails carry an OPEN window and an
    empty path on purpose: PPR does not track which seed reached a vertex
    by which route, so the only honest claim is "associatively nearby" --
    a real, checkable path is what a subsequent ``follow`` has to supply.
    """
    if not state.trails:
        return state
    seeds = {t.vertex_id: 1.0 for t in state.trails}
    scores = personalized_pagerank(graph, seeds, state.tx_point, alpha=alpha, max_hops=k)
    ranked = sorted((v for v in scores if v not in seeds), key=lambda v: -scores[v])
    top = ranked[:expand_k]
    new_trails = [
        Trail(vertex_id=v, window=Interval(None, None), path=(), labels=("expand",))
        for v in top
    ]
    return AccessState(
        trails=new_trails,
        tx_point=state.tx_point,
        dead_count=0 if new_trails else len(state.trails),
        death_cause=None if new_trails else "no_edge_with_label",
        budget_used=state.budget_used + 1,
    )


# --------------------------------------------------------------------- #
# history                                                                  #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class HistoryEntry:
    vertex_id: str
    t_valid: Interval
    edge_id: str


def history(
    state: AccessState, label: str, graph: GraphStore, catalog: Catalog
) -> list[HistoryEntry]:
    """The full time series of ``label`` from every trail's vertex, with
    NO functional collapse (spec 9.2): every value the relation has ever
    held, oldest first, including ones a later fact has since closed.
    """
    if not catalog.is_known(label):
        raise UnknownLabel(label)
    seen: set[str] = set()
    out: list[HistoryEntry] = []
    for t in state.trails:
        for e, neighbor in graph.out_edges(t.vertex_id, label, catalog):
            if e.id in seen or not e.t_tx.contains(state.tx_point):
                continue
            seen.add(e.id)
            out.append(HistoryEntry(neighbor, e.t_valid, e.id))
    out.sort(key=lambda h: h.t_valid.start or datetime.min)
    return out


# --------------------------------------------------------------------- #
# evidence                                                                 #
# --------------------------------------------------------------------- #


def evidence(state: AccessState, staging: StagingStore, log: LogStore) -> list[Episode]:
    """Materialises the raw episode text behind every live trail (spec
    P5): the answer is written from THIS, never from the proposition that
    merely located it."""
    episode_ids: list[str] = []
    seen: set[str] = set()
    for t in state.trails:
        for prop_id in t.path:
            eid = staging.get(prop_id).episode_id
            if eid not in seen:
                seen.add(eid)
                episode_ids.append(eid)
    return [log.get(eid) for eid in episode_ids]


# --------------------------------------------------------------------- #
# count                                                                    #
# --------------------------------------------------------------------- #


def count(
    mentions: MentionStore,
    graph: GraphStore,
    entity: str | None = None,
    surface: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> int:
    """Goes to the log, preserves multiplicity, ignores the graph (spec
    9.2): folding (M6) collapses repeated mentions into one edge, which is
    exactly what a "how many times" question must not read through.

    Matches by canonical name/alias, not by ``Mention.entity_id``:
    :func:`fgl.clio.ingest.pipeline.ingest_turn` records a mention's raw
    surface at ingest time, before consolidation has resolved "new:X" to a
    real vertex, so ``entity_id`` is never populated by the real pipeline
    and a lookup keyed on it would silently return zero for every mention
    a real ingest ever produced.
    """
    if entity is not None:
        ent = graph.find_entity_by_name_any_type(entity)
        if ent is not None:
            names = {ent.canonical_name.lower(), *(a.lower() for a in ent.aliases)}
            return sum(
                1
                for m in mentions.all()
                if m.surface.lower() in names
                and (start is None or m.ts >= start)
                and (end is None or m.ts < end)
            )
        if surface is None:
            surface = entity  # not a linked entity -- try it as a literal surface
    return mentions.count(surface=surface, start=start, end=end)


# --------------------------------------------------------------------- #
# available_labels                                                         #
# --------------------------------------------------------------------- #


def available_labels(state: AccessState, graph: GraphStore, catalog: Catalog) -> list[str]:
    """What ``follow`` could do from here (spec 10.2) -- lets the agent
    see its options instead of requiring any prior classification of the
    question."""
    labels: set[str] = set()
    for t in state.trails:
        for e in graph.edges_incident(t.vertex_id, live_only=False):
            if not e.t_tx.contains(state.tx_point):
                continue
            if e.src_id == t.vertex_id:
                labels.add(e.label)
            spec = catalog.get(e.label)
            if spec and spec.invertible and e.dst_id == t.vertex_id:
                labels.add(spec.inverse_name)
    return sorted(labels)
