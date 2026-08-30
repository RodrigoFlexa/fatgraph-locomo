"""Canonical form of a consolidated graph (spec 17.4).

The spec calls order-invariance "the most important test for publication",
and it needs one thing this package did not have: a normal form to compare
two memories by. Vertex ids are assignment-order artefacts -- ``ent_00003``
means "the fourth entity created", which shuffling the input changes even
when the resulting memory is identical -- so the normal form is keyed on
canonical NAMES and types instead.

What is deliberately NOT in the form, and why:

* ``reinforcement``, ``confidence`` and ``provenance``. The same body of
  evidence can reach the graph either as a direct assertion later
  reinforced, or as an accumulation promoted by phase 7, depending purely
  on which episode arrived first. Both produce the same fact over the same
  interval, which is what canonicity is a claim about; the bookkeeping of
  how it got there is genuinely order-dependent, and spec 17.4 already
  concedes that staging is not canonical -- only the consolidated graph is.
* ``conflict_flag``. It is a report about the graph, not a fact in it.
"""

from __future__ import annotations

from fgl.clio.graph.store import GraphStore
from fgl.clio.types import Interval

CanonicalRow = tuple[str, str, str, str, str, str, str, str, str, bool, bool]


def _bound(value) -> str:
    return value.isoformat() if value is not None else ""


def _interval(iv: Interval) -> tuple[str, str]:
    return _bound(iv.start), _bound(iv.end)


def canonical_form(graph: GraphStore) -> list[CanonicalRow]:
    """A sorted, id-free serialisation of every edge in ``graph``.

    Two memories built from the same episodes in different session orders
    must produce equal lists. Folded-away vertices are resolved to the
    vertex that absorbed them, so a merge changes the NAMES a row reports
    but never leaves a dangling id behind.
    """
    rows: list[CanonicalRow] = []
    for e in graph.all_edges():
        src = graph.get_entity(graph.resolve_entity(e.src_id))
        dst = graph.get_entity(graph.resolve_entity(e.dst_id))
        vs, ve = _interval(e.t_valid)
        ts, te = _interval(e.t_tx)
        rows.append(
            (
                src.canonical_name.strip().lower(),
                src.type,
                e.label,
                dst.canonical_name.strip().lower(),
                dst.type,
                vs,
                ve,
                ts,
                te,
                e.polarity,
                e.unanchored,
            )
        )
    return sorted(rows)


def canonical_entities(graph: GraphStore) -> list[tuple[str, str, tuple[str, ...]]]:
    """The live vertices, id-free: ``(canonical name, type, aliases)``.

    Separate from :func:`canonical_form` because an entity carrying no
    edges is invisible to the edge list, yet a fold that absorbed it (or
    failed to) is exactly the kind of order-sensitivity worth catching.
    """
    return sorted(
        (
            ent.canonical_name.strip().lower(),
            ent.type,
            tuple(sorted(a.strip().lower() for a in ent.aliases)),
        )
        for ent in graph.all_entities()
        if ent.merged_into is None
    )


__all__ = ["canonical_form", "canonical_entities"]
