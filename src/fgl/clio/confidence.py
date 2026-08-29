"""The fixed confidence table (spec 6.4): confidence is never asked of the
LLM as a number -- it is derived from a linguistic classification the LLM
*is* reliable at (which :class:`~fgl.clio.types.EvidenceKind` a span is),
through a table anyone can audit. Only the classification is a model
output; the number is code.
"""

from __future__ import annotations

from fgl.clio.temporal.resolver import tconf_factor
from fgl.clio.types import EvidenceKind

CONFIDENCE_TABLE: dict[EvidenceKind, float] = {
    EvidenceKind.LITERAL: 0.90,
    EvidenceKind.COREFERENCE: 0.80,
    EvidenceKind.IMPLICATURE: 0.55,
    EvidenceKind.CONTEXTUAL: 0.40,
}


def compute_confidence(evidence_kind: EvidenceKind, tconf: float) -> float:
    return CONFIDENCE_TABLE[evidence_kind] * tconf_factor(tconf)
