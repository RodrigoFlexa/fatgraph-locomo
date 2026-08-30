"""Validates the extractor's raw output against the schema and Sigma
(spec 6.5). Rejects, never accepts silently -- an extractor is the one
untrusted input in this whole pipeline (P3), so nothing from it reaches
staging without a deterministic check that CODE, not the model, decided.
"""

from __future__ import annotations

from fgl.clio.catalog import Catalog
from fgl.clio.graph.store import GraphStore
from fgl.clio.types import Episode

REQUIRED_FIELDS = (
    "operation",
    "subject_id",
    "relation",
    "object_id",
    "evidence_kind",
    "span",
)


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _type_compatible(
    ref: str | None, expected_type: str, graph: GraphStore, catalog: Catalog
) -> bool:
    """A ``new:`` reference has no type yet -- it gets one from the
    relation's own signature at consolidation's phase 1. Only an EXISTING
    id can violate the signature at this stage.

    Compared through the catalog's type classes, not by equality: an
    entity first created as an Activity (via ``practices``) is the same
    thing when a later turn ``attended`` it, and rejecting that here would
    undo phase 1's cross-type reuse by refusing to let the proposition
    through in the first place.
    """
    if not ref or ref.startswith("new:"):
        return True
    try:
        return catalog.types_compatible(graph.get_entity(ref).type, expected_type)
    except KeyError:
        return False


def validate_and_bind(
    raw: list[dict], episode: Episode, graph: GraphStore, catalog: Catalog
) -> tuple[list[dict], list[dict]]:
    """Returns ``(valid, unmapped)``. ``valid`` entries still need
    temporal resolution and confidence scoring (:mod:`fgl.clio.ingest.
    pipeline`'s job); this only enforces schema + Sigma + span integrity.
    """
    valid: list[dict] = []
    unmapped: list[dict] = []
    for item in raw:
        if not isinstance(item, dict) or any(f not in item for f in REQUIRED_FIELDS):
            continue
        relation = item["relation"]
        if relation == "UNMAPPED":
            unmapped.append(item)
            continue
        if relation not in catalog:
            continue  # outside Sigma and not UNMAPPED -- reject (spec 6.5)
        if item["operation"] not in ("assert", "reassert", "close", "retract"):
            continue
        if item["evidence_kind"] not in (
            "literal",
            "coreference",
            "implicature",
            "contextual",
        ):
            continue

        spec = catalog[relation]
        if not _type_compatible(item.get("subject_id"), spec.signature[0], graph, catalog):
            continue
        if not _type_compatible(item.get("object_id"), spec.signature[1], graph, catalog):
            continue

        span = item.get("span", "")
        if _normalize_ws(span) not in _normalize_ws(episode.text):
            # not a literal excerpt after all -- the evidence is weaker
            # than the model claimed (spec 6.5), not grounds to drop it
            item = {**item, "evidence_kind": "contextual"}
        valid.append(item)
    return valid, unmapped
