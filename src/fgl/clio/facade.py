"""``Clio``: one object bundling every store this package needs, plus the
three operations a caller actually performs -- ingest a turn, consolidate,
ask a question. Not a new layer of abstraction over the phases: every
method here is a thin call into the module that already implements it.
This exists because wiring seven stores together by hand on every call is
exactly what ``tests/clio/helpers.py``'s ``HandFedMemory`` was already
doing ad hoc for M1-M4, and a real caller (a demo script, an eventual
LoCoMo integration) needs the same wiring plus the two pieces M1-M4 never
touched an LLM for: extraction and the agent loop.
"""

from __future__ import annotations

from datetime import datetime

from fgl.clio.agent.loop import AgentTrace, run_agent_loop
from fgl.clio.catalog import Catalog, load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.consolidate.journal import FoldJournal
from fgl.clio.consolidate.pipeline import ConsolidationReport, consolidate
from fgl.clio.graph.store import GraphStore
from fgl.clio.index import EntityIndex, EpisodeIndex
from fgl.clio.ingest.pipeline import IngestResult, ingest_turn
from fgl.clio.log.mentions import MentionStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.unmapped import UnmappedQueue
from fgl.llm.client import LLMClient
from fgl.llm.prompts import PromptLibrary
from fgl.retrieval.embeddings import Embedder


class Clio:
    def __init__(
        self,
        catalog: Catalog,
        llm: LLMClient,
        embedder: Embedder,
        prompts: PromptLibrary,
        config: ClioConfig | None = None,
    ):
        self.catalog = catalog
        self.llm = llm
        self.prompts = prompts
        self.config = config or ClioConfig.default()

        self.log = LogStore()
        self.graph = GraphStore()
        self.staging = StagingStore()
        self.mentions = MentionStore()
        self.journal = FoldJournal()
        self.unmapped = UnmappedQueue()
        self.entity_index = EntityIndex(embedder)
        self.episode_index = EpisodeIndex(embedder)

    @classmethod
    def build(
        cls,
        config: ClioConfig | None = None,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        prompts_dir=None,
    ) -> Clio:
        """Builds real backends from ``.env`` / the given ``config`` --
        ``AzureLLM`` and a real embedder unless overridden, which is what
        a live run (or ``test_clio_live.py``, gated on credentials) uses.
        Pass ``llm=FakeLLM(...)`` and ``embedder=HashingEmbedder()`` for
        an offline instance instead.
        """
        from fgl.llm.client import build_llm
        from fgl.retrieval.embeddings import build_embedder

        cfg = config or ClioConfig.default()
        catalog = load_catalog(cfg.catalog_path)
        if llm is None:
            from fgl.config import LLMConfig

            llm = build_llm(LLMConfig())
        if embedder is None:
            from fgl.config import EmbeddingConfig

            embedder = build_embedder(EmbeddingConfig())
        if prompts_dir is None:
            from fgl.paths import Paths, project_root

            prompts_dir = Paths.build(project_root()).prompts
        return cls(catalog, llm, embedder, PromptLibrary(prompts_dir), cfg)

    def ingest(
        self, text: str, speaker: str, session_id: str, ts: datetime
    ) -> IngestResult:
        """Spec section 6: one LLM call, never blocks on consolidation."""
        return ingest_turn(
            text=text,
            speaker=speaker,
            session_id=session_id,
            ts=ts,
            log=self.log,
            graph=self.graph,
            staging=self.staging,
            mentions=self.mentions,
            entity_index=self.entity_index,
            catalog=self.catalog,
            llm=self.llm,
            prompts=self.prompts,
            unmapped_queue=self.unmapped,
            coref_window=self.config.extraction.coref_window,
            max_candidates=self.config.extraction.max_candidates,
        )

    def consolidate(self) -> ConsolidationReport:
        """Spec section 7 + 8: asynchronous, idempotent, includes folding."""
        return consolidate(
            self.catalog,
            self.graph,
            self.staging,
            self.config,
            log=self.log,
            journal=self.journal,
        )

    def ask(self, question: str) -> AgentTrace:
        """Spec sections 9-11: the access algebra, driven by the agent
        loop, ending in an episode-grounded answer."""
        self.entity_index.rebuild(self.graph)
        self.episode_index.rebuild(self.log)
        return run_agent_loop(question, self)
