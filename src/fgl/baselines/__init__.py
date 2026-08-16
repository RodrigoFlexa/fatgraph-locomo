"""B1 full-context, B2 k-NN over turns, B3 k-NN over the same extracted facts."""

from fgl.baselines.base import Baseline, BaselineAnswer
from fgl.baselines.full_context import FullContextBaseline
from fgl.baselines.rag_facts import RagFactsBaseline
from fgl.baselines.rag_turns import RagTurnsBaseline

REGISTRY: dict[str, type[Baseline]] = {
    "B1-full-context": FullContextBaseline,
    "B2-rag-turns": RagTurnsBaseline,
    "B3-rag-facts": RagFactsBaseline,
}

__all__ = [
    "Baseline", "BaselineAnswer", "FullContextBaseline", "RagFactsBaseline",
    "RagTurnsBaseline", "REGISTRY",
]
