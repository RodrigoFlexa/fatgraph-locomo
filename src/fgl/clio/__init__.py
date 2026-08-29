"""CLIO: Chronologically Layered Interval Ontology.

Bitemporal-graph long-term memory, developed independently of the fatgraph
conditions the rest of :mod:`fgl` benchmarks. See
``CLIO-especificacao-tecnica.md`` at the repository root for the full
design, and that document's own header comments plus this package's
module docstrings for the deliberate deviations from it (temporal
resolution is English, not the spec's own pt-BR; inverse edges are
query-time views, not physically materialised; identity scoring uses
stdlib string similarity, not Jaro-Winkler; fold's candidate search is
type-blocked, not same-address, because the literal same-address rule
does not reach spec's own worked "Rui"/"Rui Sampaio" example).

All nine milestones from the spec's own plan (foundation, catalog,
temporal resolution, consolidation, folding, the access algebra,
extraction, the agent loop) are implemented. :class:`fgl.clio.facade.Clio`
is the entry point that ties them together -- ``.ingest()``,
``.consolidate()``, ``.ask()``. Extraction and the agent loop are the only
two places an LLM is actually called, both through the same
:class:`fgl.llm.client.LLMClient` interface the rest of this repository
uses: ``FakeLLM`` for offline tests, ``AzureLLM`` for the real thing,
credentials from the same ``.env``. Full-dataset LoCoMo benchmarking
(spec's M9) -- wiring ``Clio`` into ``fgl run``/the report tables the
fatgraph conditions use -- is not done: that is a separate integration
decision, not a gap in CLIO itself.
"""

from fgl.clio.access import (
    AccessState,
    HistoryEntry,
    Trail,
    UnknownLabel,
    anchor,
    available_labels,
    count,
    evidence,
    expand,
    filter_trails,
    follow,
    history,
    render_state,
    restrict,
)
from fgl.clio.agent import AgentStep, AgentTrace, generate_answer, run_agent_loop
from fgl.clio.catalog import Catalog, RelationSpec, load_catalog
from fgl.clio.confidence import CONFIDENCE_TABLE, compute_confidence
from fgl.clio.config import ClioConfig
from fgl.clio.consolidate import (
    ConsolidationReport,
    FoldConfig,
    FoldJournal,
    FoldRecord,
    consolidate,
    identity_score,
    run_fold,
    run_unfold,
)
from fgl.clio.facade import Clio
from fgl.clio.graph.store import GraphStore
from fgl.clio.index import EntityIndex, EpisodeIndex
from fgl.clio.ingest import IngestResult, ingest_turn
from fgl.clio.log.mentions import MentionStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.temporal import resolve_time
from fgl.clio.types import (
    Edge,
    EdgeAddress,
    Entity,
    Episode,
    EvidenceKind,
    Interval,
    Mention,
    Operation,
    Proposition,
)
from fgl.clio.unmapped import UnmappedEntry, UnmappedQueue

__all__ = [
    "Clio",
    "Catalog",
    "RelationSpec",
    "load_catalog",
    "CONFIDENCE_TABLE",
    "compute_confidence",
    "ClioConfig",
    "ConsolidationReport",
    "consolidate",
    "FoldConfig",
    "FoldJournal",
    "FoldRecord",
    "identity_score",
    "run_fold",
    "run_unfold",
    "GraphStore",
    "LogStore",
    "MentionStore",
    "StagingStore",
    "EntityIndex",
    "EpisodeIndex",
    "IngestResult",
    "ingest_turn",
    "UnmappedEntry",
    "UnmappedQueue",
    "resolve_time",
    "AccessState",
    "Trail",
    "HistoryEntry",
    "UnknownLabel",
    "anchor",
    "follow",
    "restrict",
    "filter_trails",
    "expand",
    "history",
    "evidence",
    "count",
    "available_labels",
    "render_state",
    "AgentStep",
    "AgentTrace",
    "run_agent_loop",
    "generate_answer",
    "Edge",
    "EdgeAddress",
    "Entity",
    "Episode",
    "EvidenceKind",
    "Interval",
    "Mention",
    "Operation",
    "Proposition",
]
