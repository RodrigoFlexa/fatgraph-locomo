"""Validates the extractor's raw output against the schema and Sigma
(spec 6.5). Rejects, never accepts silently -- an extractor is the one
untrusted input in this whole pipeline (P3), so nothing from it reaches
staging without a deterministic check that CODE, not the model, decided.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class Rejection:
    """One item the extractor produced and this module refused, with the
    reason. Spec 6.5 says "reject and log, never accept silently" -- the
    logging half was missing, and a bare ``continue`` is what made a turn
    reporting "0 propositions" indistinguishable from a turn the model
    genuinely had nothing to say about. Four different failures wore the
    same face: an empty model response, everything mapped to UNMAPPED,
    a relation outside Sigma, and a signature violation.
    """

    reason: str
    item: dict


#: every value ``Rejection.reason`` can take, so a caller can tabulate
#: them without discovering the vocabulary by running the pipeline.
REJECTION_REASONS = (
    "malformed_item",
    "missing_required_field",
    "relation_not_in_sigma",
    "unknown_operation",
    "unknown_evidence_kind",
    "subject_type_violates_signature",
    "object_type_violates_signature",
    "person_ref_lacks_a_name",
)

#: A `new:` reference cannot violate a signature by TYPE -- it has none
#: until phase 1 gives it one, from that very signature. That is the hole
#: an inverted subject/object falls through: "transgender stories | likes
#: | Caroline" put a topic in the subject slot of a [Person, Topic]
#: relation, and phase 1 dutifully created a vertex of type Person called
#: "transgender stories" (observed on conv-26). Nothing downstream can
#: recover from it -- the vertex is a person as far as the whole graph is
#: concerned.
#:
#: The one property a person's name has in this corpus and a description
#: does not is a capital letter. So a `new:` reference landing in a Person
#: slot must carry at least one capitalised token.
#:
#: The known cost, stated rather than hidden: this also drops unnamed
#: relational nouns -- "husband", "kids", "friends", "family" as the
#: object of family_of/friend_of. That is a defensible loss and arguably a
#: gain: a vertex named "husband" is indistinguishable from every other
#: husband, can never be resolved to a real person, and no question the
#: memory is asked can be answered by reaching it. It is a placeholder,
#: not a person. The count is reported per run so the trade stays visible.
PERSON_TYPE = "Person"


def _looks_like_a_person_name(ref: str | None) -> bool:
    if not ref or not ref.startswith("new:"):
        return True  # an existing id was already type-checked
    name = ref[len("new:") :].strip()
    return any(token[:1].isupper() for token in name.split() if token)


@dataclass
class ValidationResult:
    valid: list[dict] = field(default_factory=list)
    unmapped: list[dict] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    #: entries whose span was not a verbatim excerpt and were therefore
    #: downgraded to ``contextual`` (spec 6.5). Not a rejection -- but a
    #: high rate here means most facts land at confidence 0.40 and need
    #: THREE independent episodes to ever reach the graph, which is worth
    #: seeing before blaming the graph.
    span_downgrades: int = 0

    def __iter__(self):
        """Kept unpackable as ``valid, unmapped`` -- every existing caller
        and test does exactly that."""
        return iter((self.valid, self.unmapped))

    def counts_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rejected:
            counts[r.reason] = counts.get(r.reason, 0) + 1
        return counts


def validate_and_bind(
    raw: list[dict], episode: Episode, graph: GraphStore, catalog: Catalog
) -> ValidationResult:
    """Returns a :class:`ValidationResult`, which still unpacks as
    ``valid, unmapped``. ``valid`` entries need temporal resolution and
    confidence scoring (:mod:`fgl.clio.ingest.pipeline`'s job); this only
    enforces schema + Sigma + span integrity, and records WHY it refused
    whatever it refused.
    """
    result = ValidationResult()
    for item in raw:
        if not isinstance(item, dict):
            result.rejected.append(Rejection("malformed_item", {"raw": repr(item)[:200]}))
            continue
        if any(f not in item for f in REQUIRED_FIELDS):
            result.rejected.append(Rejection("missing_required_field", item))
            continue
        relation = item["relation"]
        if relation == "UNMAPPED":
            result.unmapped.append(item)
            continue
        if relation not in catalog:
            # outside Sigma and not UNMAPPED (spec 6.5). Distinct from
            # UNMAPPED: the model invented a label instead of admitting
            # the catalog had no room for it, so this does NOT reach the
            # queue a human mines to grow Sigma.
            result.rejected.append(Rejection("relation_not_in_sigma", item))
            continue
        if item["operation"] not in ("assert", "reassert", "close", "retract"):
            result.rejected.append(Rejection("unknown_operation", item))
            continue
        if item["evidence_kind"] not in (
            "literal",
            "coreference",
            "implicature",
            "contextual",
        ):
            result.rejected.append(Rejection("unknown_evidence_kind", item))
            continue

        spec = catalog[relation]
        if not _type_compatible(item.get("subject_id"), spec.signature[0], graph, catalog):
            result.rejected.append(Rejection("subject_type_violates_signature", item))
            continue
        if not _type_compatible(item.get("object_id"), spec.signature[1], graph, catalog):
            result.rejected.append(Rejection("object_type_violates_signature", item))
            continue
        person_slots = [
            ref
            for ref, slot_type in (
                (item.get("subject_id"), spec.signature[0]),
                (item.get("object_id"), spec.signature[1]),
            )
            if slot_type == PERSON_TYPE
        ]
        if not all(_looks_like_a_person_name(ref) for ref in person_slots):
            result.rejected.append(Rejection("person_ref_lacks_a_name", item))
            continue

        span = item.get("span") or ""
        if _normalize_ws(span) not in _normalize_ws(episode.text):
            # not a literal excerpt after all -- the evidence is weaker
            # than the model claimed (spec 6.5), not grounds to drop it
            item = {**item, "evidence_kind": "contextual"}
            result.span_downgrades += 1
        result.valid.append(item)
    return result
