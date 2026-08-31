"""CLIO2: compiled queries over CLIO's bitemporal evidence ledger.

The original CLIO access algebra remains available as an experimental
reader.  CLIO2 keeps the same immutable log and consolidated memory, but
replaces open-ended graph wandering with a typed plan, deterministic
execution, and an evidence-checked answer contract.
"""

from fgl.clio2.engine import run_clio2
from fgl.clio2.model import AnswerType, QueryOperator, QueryPlan

__all__ = ["AnswerType", "QueryOperator", "QueryPlan", "run_clio2"]
