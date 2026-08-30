"""Phase 1 (spec 7.2): turn ``"new:Name"`` references into vertex ids.

Full fuzzy identity resolution -- string similarity, structural overlap,
role context, the whole :func:`~fgl.clio.consolidate.fold.identity_score`
machinery -- is what milestone M6 (folding) adds. Phase 1 only needs to
stop the *same* surface name from minting a new vertex every time it
recurs, so an exact, case-insensitive match against existing canonical
names/aliases of the same type is enough here; see
:meth:`fgl.clio.graph.store.GraphStore.find_entity_by_name` for why that
split is deliberate, not a shortcut.

One check goes beyond spec 7.2's own wording: before reusing an exact-name
match, and only when a log is available, the triggering episode's own text
is checked for a disambiguation marker naming it ("the other Rui"). Without
this, two DIFFERENT people who happen to share one EXACT name are unified
right here, unconditionally, before fold (M6) ever gets a chance to tell
them apart -- fold's own ``explicit_distinction`` penalty only ever sees
candidates with DIFFERENT surface forms, because identical ones never reach
it as a pair in the first place.
"""

from __future__ import annotations

from fgl.clio.catalog import Catalog
from fgl.clio.consolidate.fold import mentions_distinction
from fgl.clio.graph.store import GraphStore
from fgl.clio.log.store import LogStore
from fgl.clio.types import Proposition

NEW_PREFIX = "new:"


def _resolve_ref(
    ref: str,
    type_: str,
    graph: GraphStore,
    episode_id: str,
    log: LogStore | None,
    catalog: Catalog | None = None,
) -> str:
    if not ref.startswith(NEW_PREFIX):
        return ref
    name = ref[len(NEW_PREFIX) :]
    # Match across the whole type CLASS, not just the exact signature type
    # (spec 7.2 says "match against the index", not "match on type"). One
    # thing named once is reachable under two signature types -- "the
    # charity race" is an Activity through `practices` and an Event
    # through `attended` -- and typing it from whichever relation happened
    # to mention it first used to mint a SECOND vertex with the same name.
    # Fold could never repair that either: identity_score returns 0.0 on a
    # type mismatch before looking at any other signal, so the two stayed
    # apart forever and every path through that thing was missing.
    accepted = catalog.type_class(type_) if catalog is not None else {type_}
    existing = graph.find_entity_by_name_in_types(name, set(accepted))
    if existing is not None and not (
        log is not None and mentions_distinction(log.get(episode_id).text, name)
    ):
        return existing.id
    ent = graph.create_entity(
        canonical_name=name, type=type_, created_from=episode_id, provisional=True
    )
    return ent.id


def phase_1_resolve_entities(
    props: list[Proposition],
    graph: GraphStore,
    catalog: Catalog,
    log: LogStore | None = None,
) -> None:
    for p in props:
        spec = catalog[p.relation]
        p.subject_id = _resolve_ref(
            p.subject_id, spec.signature[0], graph, p.episode_id, log, catalog
        )
        p.object_id = _resolve_ref(
            p.object_id, spec.signature[1], graph, p.episode_id, log, catalog
        )
