"""Renders an :class:`AccessState` to the compact JSON shape spec 10.2
hands the agent. ``available_labels`` is what lets the agent see its own
options from wherever it stands -- there is no prior classification of
the question to route on instead (spec section 9's whole premise).
"""

from __future__ import annotations

from datetime import datetime

from fgl.clio.access.movements import available_labels
from fgl.clio.access.state import AccessState
from fgl.clio.catalog import Catalog
from fgl.clio.graph.store import GraphStore
from fgl.clio.types import Interval


def _format_window(window: Interval) -> str:
    start = window.start.date().isoformat() if window.start else "-inf"
    end = window.end.date().isoformat() if window.end else "now"
    return f"{start}..{end}"


def render_state(
    state: AccessState,
    graph: GraphStore,
    catalog: Catalog,
    budget_total: int,
    sample_size: int = 5,
) -> dict:
    sample = []
    for t in state.trails[:sample_size]:
        ent = graph.get_entity(t.vertex_id)
        sample.append(
            {
                "vertex": ent.canonical_name,
                "type": ent.type,
                "window": _format_window(t.window),
                "hops": len(t.labels),
                "labels": list(t.labels),
            }
        )
    labels: set[str] = set()
    for t in state.trails:
        labels |= set(available_labels(AccessState([t], state.tx_point), graph, catalog))
    return {
        "live_trails": len(state.trails),
        "sample": sample,
        "dead_trails": state.dead_count,
        "death_cause": state.death_cause,
        "available_labels": sorted(labels),
        "tx_point": state.tx_point.date().isoformat()
        if isinstance(state.tx_point, datetime)
        else str(state.tx_point),
        "budget_used": state.budget_used,
        "budget_left": max(0, budget_total - state.budget_used),
    }
