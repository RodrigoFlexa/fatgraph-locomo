"""Scoring and reporting.

`scorer` is the official LoCoMo metric, vendored verbatim plus the two fixes it
needs (see docs/COERENCIA.md C7 and C13.5). `report` turns metrics into tables.
"""

from fgl.evaluation.report import (
    build_report,
    comparison_table,
    cost_table,
    graph_table,
    load_results,
    markdown_table,
    recall_table,
    sanity_banner,
    write_report,
)
from fgl.evaluation.scorer import (
    STEMMER_NAME,
    QAOutcome,
    aggregate,
    evidence_recall,
    f1_multi,
    f1_score,
    is_abstention,
    normalize_answer,
    score_question,
)

__all__ = [
    "STEMMER_NAME", "QAOutcome", "aggregate", "evidence_recall", "f1_multi",
    "f1_score", "is_abstention", "normalize_answer", "score_question",
    "build_report", "comparison_table", "cost_table", "graph_table",
    "load_results", "markdown_table", "recall_table", "sanity_banner", "write_report",
]
