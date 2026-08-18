"""Session-by-session ingestion of a LoCoMo conversation into a fatgraph.

Per session, in chronological order:

1. **extraction**    -- atomic, self-contained facts (cached on disk and shared
                        by B3 and every G* condition, as the spec demands);
2. **entity resolution** -- :mod:`fatgraph.entities`;
3. **insertion**     -- with a sigma policy (``sigma-time`` or ``sigma-agent``);
4. **incongruence**  -- LLM consistency judgement against sibling memories.

At the end of the session (asynchronous regime) :mod:`fatgraph.curation` runs
bigon collapse and face consolidation when enabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from fgl.config import Config
from fgl.core import STATE_INCONGRUENT, FatGraph
from fgl.memory.curation import Curator, FaceTracker
from fgl.retrieval.embeddings import Embedder
from fgl.memory.entities import EntityResolver
from fgl.llm import LLMClient
from fgl.data.locomo import Conversation, Session
from fgl.logging_utils import JsonlLogger, NullLogger
from fgl.llm.prompts import SYSTEM_EXTRACTOR, SYSTEM_JUDGE, PromptLibrary


# --------------------------------------------------------------------------- #
# Facts                                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class Fact:
    """An atomic memory before it becomes an edge."""

    entity_1: str
    relation: str
    entity_2: str
    fact_text: str
    turn_ids: list[str]
    #: Who said it. Metadata, deliberately NOT a vertex: with the speaker as an
    #: endpoint, 86.6% of edges touch one of the two speakers and the median
    #: bridge vertex between two evidence facts has degree 164 -- so "these two
    #: memories share an entity" degenerates into "both are about Caroline",
    #: which is true of half the graph. The speaker stays in `fact_text`, so the
    #: sentence a retriever embeds is unchanged; only the topology differs.
    speaker: str = ""
    session_id: str = ""
    session_num: int = 0
    timestamp: str = ""
    date_raw: str = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    # consumed by FatGraph.add_edge via _FactView
    @property
    def text(self) -> str:
        return self.fact_text

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FactExtractor:
    """LLM fact extraction with a condition-independent on-disk cache.

    The cache path depends only on ``(sample_id, session, prompt version, llm
    deployment)`` -- never on the experimental condition -- which is what makes
    the B3 vs G1 ablation isolate topology rather than extraction.
    """

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptLibrary,
        cache_dir: str | Path,
        max_facts_per_session: int = 0,
        prompt_name: str = "extract_facts",
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_facts_per_session = max_facts_per_session
        self.prompt_name = prompt_name

    def extract(self, conv: Conversation, session: Session) -> list[Fact]:
        path = self._cache_path(conv, session)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [Fact.from_dict(f) for f in payload["facts"]]

        transcript = "\n".join(
            f"[{t.dia_id}] {t.speaker}: {t.text}"
            + (f" [shares an image: {t.img_caption}]" if t.img_caption else "")
            for t in session.turns
        )
        prompt = self.prompts.render(
            self.prompt_name,
            speaker_a=conv.speaker_a,
            speaker_b=conv.speaker_b,
            session_date=session.date_time_raw,
            transcript=transcript,
        )
        raw = self.llm.complete_json(
            prompt,
            system=SYSTEM_EXTRACTOR,
            purpose="ingest/extract",
            max_tokens=4000,
            default={"facts": []},
        )
        facts = self._normalise(raw, conv, session)
        path.write_text(
            json.dumps({"facts": [f.to_dict() for f in facts]}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        return facts

    def extract_all(self, conv: Conversation) -> list[Fact]:
        out: list[Fact] = []
        for session in conv.sessions:
            out.extend(self.extract(conv, session))
        return out

    # -- helpers -----------------------------------------------------------
    def _cache_path(self, conv: Conversation, session: Session) -> Path:
        # Each prompt keeps its own cache, so switching between them costs
        # nothing already paid for. The default prompt deliberately keeps the
        # ORIGINAL tag shape (deployment-hash): inserting its name would rename
        # every existing directory and silently re-extract ten conversations
        # that were already bought.
        version = self.prompts.version(self.prompt_name)
        tag = (
            f"{self.llm.cfg.deployment}-{version}"
            if self.prompt_name == "extract_facts"
            else f"{self.llm.cfg.deployment}-{self.prompt_name}-{version}"
        )
        d = self.cache_dir / tag / conv.sample_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"session_{session.num:03d}.json"

    def _normalise(self, raw, conv: Conversation, session: Session) -> list[Fact]:
        items = raw.get("facts", raw) if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            items = []
        valid_turns = {t.dia_id for t in session.turns}
        out: list[Fact] = []
        seen: set[tuple] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            e1 = str(item.get("entity_1", "")).strip()
            e2 = str(item.get("entity_2", "")).strip()
            text = str(item.get("fact_text", "")).strip()
            if not (e1 and e2 and text):
                continue
            turn_ids = [t for t in (item.get("turn_ids") or []) if t in valid_turns]
            key = (e1.lower(), e2.lower(), text.lower())
            if key in seen:
                continue
            seen.add(key)
            speaker = str(item.get("speaker", "")).strip()
            if speaker.lower() not in (conv.speaker_a.lower(), conv.speaker_b.lower()):
                speaker = ""  # only the two real speakers, never a hallucination
            out.append(
                Fact(
                    entity_1=e1,
                    relation=str(item.get("relation", "")).strip() or "related to",
                    entity_2=e2,
                    speaker=speaker,
                    fact_text=text,
                    turn_ids=turn_ids or sorted(valid_turns)[:1],
                    session_id=session.id,
                    session_num=session.num,
                    timestamp=session.timestamp,
                    date_raw=session.date_time_raw,
                )
            )
        if self.max_facts_per_session:
            out = out[: self.max_facts_per_session]
        return out


# --------------------------------------------------------------------------- #
# Sigma insertion policies                                                     #
# --------------------------------------------------------------------------- #


class SigmaPolicy:
    """Chooses where a new half-edge enters the cyclic order of a vertex."""

    name = "abstract"

    def position(self, graph: FatGraph, vertex_id: str, fact: Fact) -> int:
        raise NotImplementedError


class SigmaTime(SigmaPolicy):
    """Insert immediately after the most recent half-edge of the vertex."""

    name = "sigma-time"

    def position(self, graph: FatGraph, vertex_id: str, fact: Fact) -> int:
        order = graph.sigma.get(vertex_id, [])
        if not order:
            return 0
        best_i, best_ts = 0, None
        for i, h in enumerate(order):
            ts = graph.H[h].timestamp
            if best_ts is None or ts >= best_ts:  # ties -> the later position wins
                best_i, best_ts = i, ts
        return best_i + 1


class SigmaAgent(SigmaPolicy):
    """Ask the LLM where the memory belongs, showing the existing trails."""

    name = "sigma-agent"

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptLibrary,
        logger: JsonlLogger,
        max_trails: int = 8,
        trail_chars: int = 160,
        fallback: SigmaPolicy | None = None,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.log = logger
        self.max_trails = max_trails
        self.trail_chars = trail_chars
        self.fallback = fallback or SigmaTime()

    def position(self, graph: FatGraph, vertex_id: str, fact: Fact) -> int:
        order = graph.sigma.get(vertex_id, [])
        if len(order) <= 1:
            return len(order)
        if len(order) > self.max_trails:
            # keep the prompt bounded: only the most recent half-edges are shown,
            # and the returned index is mapped back to the full cyclic list.
            window_start = len(order) - self.max_trails
        else:
            window_start = 0
        window = order[window_start:]

        lines = []
        for i, h in enumerate(window):
            he = graph.H[h]
            trail = self._trail_summary(graph, h)
            lines.append(
                f"{i}. [{he.timestamp[:10]}] {he.text}\n"
                f"   trail: {trail}"
            )
        prompt = self.prompts.render(
            "sigma_agent",
            vertex_name=graph.vertices[vertex_id].name,
            trails="\n".join(lines),
            fact_text=fact.fact_text,
            session_date=fact.date_raw or fact.timestamp,
            max_index=len(window) - 1,
        )
        out = self.llm.complete_json(
            prompt, system=SYSTEM_JUDGE, purpose="ingest/sigma_agent",
            default={"after_index": len(window) - 1, "reason": "unparseable output"},
        )
        try:
            after = int(out.get("after_index", len(window) - 1))
        except (TypeError, ValueError):
            after = len(window) - 1
        after = max(-1, min(after, len(window) - 1))
        pos = window_start + after + 1

        self.log.log(
            "sigma_agent_decision",
            vertex=vertex_id,
            vertex_name=graph.vertices[vertex_id].name,
            fact=fact.fact_text,
            session=fact.session_id,
            degree=len(order),
            after_index=after,
            position=pos,
            reason=str(out.get("reason", ""))[:400],
        )
        return pos

    def _trail_summary(self, graph: FatGraph, h: str) -> str:
        face = graph.face_of(h)
        texts = []
        for hid in face.half_edges:
            t = graph.H[hid].text
            if t not in texts:
                texts.append(t)
        joined = " -> ".join(texts)
        return joined[: self.trail_chars] + ("..." if len(joined) > self.trail_chars else "")


def build_sigma_policy(
    cfg: Config, llm: LLMClient, prompts: PromptLibrary, logger: JsonlLogger
) -> SigmaPolicy:
    if cfg.ingest.sigma_policy == "sigma-time":
        return SigmaTime()
    if cfg.ingest.sigma_policy == "sigma-agent":
        return SigmaAgent(
            llm, prompts, logger,
            max_trails=cfg.ingest.sigma_agent_max_trails,
            trail_chars=cfg.ingest.sigma_agent_trail_chars,
        )
    raise ValueError(f"unknown sigma policy {cfg.ingest.sigma_policy!r}")


# --------------------------------------------------------------------------- #
# Ingestor                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class IngestReport:
    sample_id: str
    condition: str
    n_facts: int = 0
    n_edges: int = 0
    n_skipped_self_loops: int = 0
    n_incongruent: int = 0
    n_collapses: int = 0
    n_consolidations: int = 0
    per_session: list[dict] = field(default_factory=list)
    graph_stats: dict = field(default_factory=dict)
    llm_usage: dict = field(default_factory=dict)
    #: faces before/after, and how many transpositions it took (empty when the
    #: rotation search is off)
    genus_optimization: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Ingestor:
    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        embedder: Embedder,
        prompts: PromptLibrary,
        logger: JsonlLogger | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.embedder = embedder
        self.prompts = prompts
        self.log = logger or NullLogger()
        self.extractor = FactExtractor(
            llm, prompts, cfg.paths.facts_cache, cfg.ingest.max_facts_per_session,
            prompt_name=cfg.ingest.extract_prompt,
        )

    # ------------------------------------------------------------------ run --
    def ingest(self, conv: Conversation) -> tuple[FatGraph, IngestReport]:
        graph = FatGraph()
        resolver = EntityResolver(
            graph, self.embedder, self.cfg.entities, self.llm, self.prompts, self.log
        )
        sigma_policy = build_sigma_policy(self.cfg, self.llm, self.prompts, self.log)
        curator = Curator(self.cfg, self.llm, self.prompts, self.log, self.embedder)
        tracker = FaceTracker()
        report = IngestReport(sample_id=conv.sample_id, condition=self.cfg.condition)

        for session in conv.sessions:
            facts = self.extractor.extract(conv, session)
            texts = [f.fact_text for f in facts]
            if texts:
                vectors = self.embedder.encode(texts)
                for f, v in zip(facts, vectors):
                    f.embedding = v

            n_incong_before = report.n_incongruent
            for fact in facts:
                r1 = resolver.resolve(fact.entity_1, session.id)
                r2 = resolver.resolve(fact.entity_2, session.id)
                if r1.vertex_id == r2.vertex_id and not self.cfg.ingest.allow_self_loops:
                    report.n_skipped_self_loops += 1
                    self.log.log(
                        "skip_self_loop", fact=fact.fact_text, vertex=r1.vertex_id,
                        session=session.id,
                    )
                    continue

                pos1 = sigma_policy.position(graph, r1.vertex_id, fact)
                pos2 = sigma_policy.position(graph, r2.vertex_id, fact)
                edge_id = graph.add_edge(r1.vertex_id, r2.vertex_id, fact, pos1, pos2)
                report.n_edges += 1
                self.log.log(
                    "insert_edge", edge=edge_id, session=session.id,
                    v1=r1.vertex_id, v2=r2.vertex_id, pos1=pos1, pos2=pos2,
                    policy=sigma_policy.name, fact=fact.fact_text,
                    turn_ids=fact.turn_ids,
                )
                if self.cfg.ingest.detect_incongruence:
                    if self._check_incongruence(graph, edge_id, fact):
                        report.n_incongruent += 1

            report.n_facts += len(facts)

            # ---- asynchronous end-of-session regime -------------------------
            n_collapse = curator.collapse_redundant_bigons(graph)
            tracker.observe(graph, session.num)
            n_consol = curator.consolidate(graph, tracker, session)
            report.n_collapses += n_collapse
            report.n_consolidations += n_consol

            graph.check_invariants()
            stats = graph.stats()
            report.per_session.append(
                {
                    "session": session.num,
                    "session_id": session.id,
                    "timestamp": session.timestamp,
                    "n_facts": len(facts),
                    "n_incongruent_new": report.n_incongruent - n_incong_before,
                    "n_collapses": n_collapse,
                    "n_consolidations": n_consol,
                    **{k: stats[k] for k in ("V", "E", "F", "C", "genus")},
                    "face_length_hist": stats["face_length_hist"],
                }
            )

        # ---- choose the rotation, instead of inheriting the clock's ----------
        # sigma is a free parameter of the ribbon structure, and ordering it by
        # timestamp is a choice, not a given -- one that produces a high-genus
        # embedding whose boundary walks run to hundreds of half-edges. Euler
        # (F = 2C - 2g + E - V) says maximising the face count IS minimising the
        # genus, and over fixed V and E more faces means shorter ones.
        if self.cfg.curation.maximize_faces:
            report.genus_optimization = graph.maximize_faces(
                max_passes=self.cfg.curation.maximize_faces_passes,
                max_degree_scan=self.cfg.curation.maximize_faces_degree_scan,
            )
            graph.check_invariants()

        report.graph_stats = graph.stats()
        report.llm_usage = self.llm.usage.to_dict()
        return graph, report

    # ---------------------------------------------------------- incongruence -
    def _check_incongruence(self, graph: FatGraph, edge_id: str, fact: Fact) -> bool:
        """Compare the new memory with its siblings over the same vertex pair.

        The spec phrases this as "the new face contradicts another face over the
        same vertices"; a contradiction is a property of the *facts*, and the
        faces sharing those vertices are exactly the ones carrying the sibling
        edges -- so we judge edge against sibling edge.  See DECISIONS.md D7.
        """
        v1, v2 = graph.edge_endpoints(edge_id)
        pair = {v1, v2}
        siblings = [
            e
            for e in graph.edges()
            if e != edge_id and set(graph.edge_endpoints(e)) == pair
        ]
        if not siblings:
            return False

        h_new, _ = graph.edge_half_edges(edge_id)
        new_he = graph.H[h_new]
        for other in siblings[-3:]:  # bound the cost: 3 most recent siblings
            h_old, _ = graph.edge_half_edges(other)
            old_he = graph.H[h_old]
            prompt = self.prompts.render(
                "incongruence",
                old_timestamp=old_he.timestamp or "?",
                old_text=old_he.text,
                new_timestamp=new_he.timestamp or "?",
                new_text=new_he.text,
            )
            out = self.llm.complete_json(
                prompt, system=SYSTEM_JUDGE, purpose="ingest/incongruence",
                default={"contradiction": False, "reason": "unparseable"},
            )
            if bool(out.get("contradiction")):
                graph.set_edge_attr(edge_id, state=STATE_INCONGRUENT)
                meta = dict(new_he.meta)
                meta["conflicts_with"] = sorted(
                    set(meta.get("conflicts_with", [])) | {other}
                )
                for h in graph.edge_half_edges(edge_id):
                    graph.H[h].meta = dict(meta)
                self.log.log(
                    "incongruence", edge=edge_id, conflicts_with=other,
                    new_text=new_he.text, old_text=old_he.text,
                    reason=str(out.get("reason", ""))[:400], session=fact.session_id,
                )
                return True
        return False
