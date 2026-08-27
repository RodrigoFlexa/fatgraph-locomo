"""MECA ingestion: read once, deeply.

The cost structure of ordinary RAG is inverted. Ingestion is cheap and dumb
(chunk, embed); answering is expensive and hard, and pays that price N times --
once per question. Every query re-does the same comprehension under noise and a
token budget: who is "she", when was "last month", did that happen or was it
only a plan.

MECA pays for comprehension once per document, where it can be arbitrarily
expensive, and stores the *result*: attested propositions
(:mod:`fgl.memory.propositions`) rather than pointers into text.

Three stages, each a separate LLM call so each is a separate ablation:

``extract``   what the passage states, with the exact span for each claim;
``infer``     what follows from it that was not said -- marked ``derived``;
``verify``    is each claim entailed by its own cited span? Nothing that fails
              enters the memory.

The third stage is what makes the first two defensible. Extraction asks the
model to do the thing it is worst at (produce structure without inventing);
verification asks it to do the thing it is good at (judge entailment on a short
pair), and only the verified survive. ``NullVerifier`` turns it off, and the
difference between the two IS the measurement of what verification buys.

Nothing in this module knows what a speaker, a session or a dialogue turn is. A
source is a sequence of utterances with an optional author and an optional
timestamp; LoCoMo is one such source and is not the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from fgl.config import Config
from fgl.core import FatGraph
from fgl.data.locomo import Conversation
from fgl.llm.client import LLMClient
from fgl.llm.prompts import PromptLibrary
from fgl.logging_utils import JsonlLogger, NullLogger
from fgl.memory.consolidate import consolidate
from fgl.memory.ingest import IngestReport
from fgl.memory.propositions import (
    Evidence,
    Proposition,
    PropositionStore,
    TimePoint,
    build_graph,
    coerce_proposition,
    normalise,
)
from fgl.retrieval.embeddings import Embedder

# --------------------------------------------------------------------------- #
# Passages                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class Utterance:
    """The generic input unit: text, optionally authored, optionally dated."""

    doc_id: str
    text: str
    author: str = ""
    timestamp: str = ""
    date_raw: str = ""


@dataclass
class Passage:
    """A window of consecutive utterances handed to the extractor."""

    utterances: list[Utterance] = field(default_factory=list)

    @property
    def date_raw(self) -> str:
        for u in self.utterances:
            if u.date_raw:
                return u.date_raw
        return ""

    @property
    def timestamp(self) -> str:
        for u in self.utterances:
            if u.timestamp:
                return u.timestamp
        return ""

    def render(self) -> str:
        lines = []
        for u in self.utterances:
            prefix = f"{u.author}: " if u.author else ""
            lines.append(f"{prefix}{u.text}")
        return "\n".join(lines)

    def locate(self, span: str) -> Optional[Utterance]:
        """Which utterance a quoted span came from.

        Exact containment first, then a normalised containment, then the
        longest shared run. A span the model paraphrased instead of copying
        therefore still lands on the right source rather than being dropped --
        but it lands on *an* utterance, never on a fabricated id.
        """
        if not span:
            return None
        for u in self.utterances:
            if span in u.text:
                return u
        target = normalise(span)
        if not target:
            return None
        for u in self.utterances:
            if target and target in normalise(u.text):
                return u
        best, best_score = None, 0.0
        target_words = set(target.split())
        if not target_words:
            return None
        for u in self.utterances:
            words = set(normalise(u.text).split())
            if not words:
                continue
            score = len(target_words & words) / len(target_words)
            if score > best_score:
                best, best_score = u, score
        return best if best_score >= 0.5 else None


def segment(
    utterances: Sequence[Utterance],
    embedder: Embedder,
    min_size: int = 2,
    max_size: int = 6,
    quantile: float = 0.65,
) -> list[Passage]:
    """Cut a source into passages where its own coherence drops.

    One signal, not five. The cut is the ``quantile`` of the *drop
    distribution of this source*, so a chatty corpus and a terse one get
    different cuts without anyone re-tuning; ``min_size``/``max_size`` are
    declared structural bounds, not swept knobs.

    Worth stating because it is an argument for the whole design: **MECA is far
    less sensitive to this than MEST was.** There the passage was the unit
    indexed and emitted, so a boundary in the wrong place cost recall directly.
    Here the stored unit is the proposition and the passage is only the window
    the extractor reads, so a bad boundary costs the extractor a little context
    and nothing after that.
    """
    n = len(utterances)
    if n == 0:
        return []
    if n <= min_size:
        return [Passage(list(utterances))]

    vectors = np.asarray(
        embedder.encode([u.text or " " for u in utterances]), dtype=np.float32
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    drops = np.array(
        [1.0 - float(vectors[i] @ vectors[i + 1]) for i in range(n - 1)],
        dtype=np.float32,
    )
    cut = float(np.quantile(drops, quantile)) if drops.size else 1.0

    passages: list[Passage] = []
    current: list[Utterance] = []
    for i, u in enumerate(utterances):
        current.append(u)
        if len(current) >= max_size:
            passages.append(Passage(current))
            current = []
            continue
        if (
            len(current) >= min_size
            and i < n - 1
            and float(drops[i]) >= cut
        ):
            passages.append(Passage(current))
            current = []
    if current:
        if passages and len(current) < min_size:
            passages[-1].utterances.extend(current)
        else:
            passages.append(Passage(current))
    return passages


# --------------------------------------------------------------------------- #
# Verification                                                                 #
# --------------------------------------------------------------------------- #


class Verifier:
    """Decides whether a claim is entailed by the span it cites."""

    def verify(self, props: Sequence[Proposition]) -> list[bool]:
        raise NotImplementedError


class NullVerifier(Verifier):
    """Accepts everything. The ablation arm, and the honest name for it."""

    def verify(self, props: Sequence[Proposition]) -> list[bool]:
        return [True] * len(props)


class LLMVerifier(Verifier):
    """One batched call per passage: every claim of that passage at once.

    Batched rather than per-claim because the cost has to stay a small constant
    per passage; a verifier that doubles the ingest bill would be traded away
    the first time the bill was looked at, and it is the piece that must not be.
    """

    def __init__(self, llm: LLMClient, prompts: PromptLibrary, max_tokens: int = 1200):
        self.llm = llm
        self.prompts = prompts
        self.max_tokens = max_tokens

    def verify(self, props: Sequence[Proposition]) -> list[bool]:
        if not props:
            return []
        claims = "\n".join(
            f'{i + 1}. CLAIM: {p.statement()}\n'
            f'   MODALITY: {p.modality}\n'
            f'   EVIDENCE: "{p.evidence[0].span if p.evidence else ""}"'
            for i, p in enumerate(props)
        )
        prompt = self.prompts.render("meca_verify", claims=claims)
        data = self.llm.complete_json(
            prompt, purpose="ingest/meca_verify", max_tokens=self.max_tokens,
            default={"verdicts": []},
        )
        verdicts = {}
        for row in (data or {}).get("verdicts", []) or []:
            try:
                verdicts[int(row.get("id"))] = bool(row.get("supported"))
            except (TypeError, ValueError):
                continue
        # A claim the verifier did not rule on is REJECTED, not accepted. The
        # direction of the default is the whole safety argument: a malformed
        # response must shrink the memory, never fill it with unchecked claims.
        return [verdicts.get(i + 1, False) for i in range(len(props))]


# --------------------------------------------------------------------------- #
# The ingestor                                                                 #
# --------------------------------------------------------------------------- #


class MecaIngestor:
    """Builds the proposition store, consolidates it, and lays it out.

    Both conditions -- M1 (flat) and M2 (ribbon) -- run this class unchanged.
    The fatgraph it returns always carries the rotation; the flat reader simply
    does not look at it. That is what makes "does the ribbon structure pay?" a
    comparison of two readers over one memory instead of two different memories
    wearing the same name.
    """

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
        m = cfg.meca
        self.verifier: Verifier = (
            LLMVerifier(llm, prompts, m.verify_max_tokens)
            if m.verify else NullVerifier()
        )

    # ------------------------------------------------------------------ api --
    def ingest(self, conv: Conversation) -> tuple[FatGraph, IngestReport]:
        m = self.cfg.meca
        report = IngestReport(sample_id=conv.sample_id, condition=self.cfg.condition)
        store = PropositionStore()

        counts = {
            "passages": 0, "extracted": 0, "inferred": 0,
            "rejected_unverified": 0, "rejected_malformed": 0, "no_evidence": 0,
        }

        for session in conv.sessions:
            utterances = [
                Utterance(
                    doc_id=t.dia_id, text=t.text, author=t.speaker,
                    timestamp=session.timestamp, date_raw=session.date_time_raw,
                )
                for t in session.turns
            ]
            passages = segment(
                utterances, self.embedder,
                m.passage_min, m.passage_max, m.passage_quantile,
            )
            counts["passages"] += len(passages)
            n_session = 0
            for passage in passages:
                props = self._comprehend(passage, counts)
                for prop in props:
                    store.add(prop)
                n_session += len(props)
            report.per_session.append(
                {"session_id": session.id, "n_propositions": n_session,
                 "n_passages": len(passages)}
            )

        consolidation = consolidate(
            store, self.embedder,
            entity_quantile=m.entity_quantile, entity_floor=m.entity_floor,
            predicate_quantile=m.predicate_quantile, predicate_floor=m.predicate_floor,
            functional_quantile=m.functional_quantile, functional_floor=m.functional_floor,
            resolve=m.resolve_entities, dedupe=m.deduplicate, timeline=m.timeline,
        )

        graph = build_graph(store, self.embedder)
        report.n_facts = len(store)
        report.n_edges = len(graph.edges())
        report.llm_usage = self.llm.usage.to_dict() if hasattr(self.llm, "usage") else {}
        report.graph_stats = {
            # The CLI's common ingest summary expects the topology metrics
            # (V, E, F, C, genus) regardless of the memory representation.
            # MECA keeps its proposition-specific metrics alongside them.
            **graph.stats(),
            **store.stats(),
            "consolidation": consolidation.as_dict(),
            "comprehension": counts,
        }
        # No sidecar file: every proposition rides in its own vertex's meta,
        # so `store_from_graph` reconstructs the memory exactly from a cached
        # graph and there is no second artifact to drift out of sync with it.
        return graph, report

    # ------------------------------------------------------------ one passage -
    def _comprehend(self, passage: Passage, counts: dict) -> list[Proposition]:
        m = self.cfg.meca
        asserted_at = TimePoint.parse(passage.timestamp)
        text = passage.render()
        if not text.strip():
            return []

        stated = self._extract(passage, text, asserted_at, counts)
        counts["extracted"] += len(stated)

        derived: list[Proposition] = []
        if m.infer and stated:
            derived = self._infer(passage, text, asserted_at, stated, counts)
            counts["inferred"] += len(derived)

        candidates = stated + derived
        if not candidates:
            return []
        verdicts = self.verifier.verify(candidates)
        kept = [p for p, ok in zip(candidates, verdicts, strict=False) if ok]
        counts["rejected_unverified"] += len(candidates) - len(kept)
        return kept

    def _extract(
        self, passage: Passage, text: str, asserted_at: Optional[TimePoint],
        counts: dict,
    ) -> list[Proposition]:
        prompt = self.prompts.render(
            "meca_extract", passage=text, date=passage.date_raw or "unknown"
        )
        data = self.llm.complete_json(
            prompt, purpose="ingest/meca_extract",
            max_tokens=self.cfg.meca.extract_max_tokens,
            default={"propositions": []},
        )
        return self._coerce(data, passage, asserted_at, counts, derived_from=())

    def _infer(
        self, passage: Passage, text: str, asserted_at: Optional[TimePoint],
        stated: Sequence[Proposition], counts: dict,
    ) -> list[Proposition]:
        known = json.dumps(
            [{"subject": p.subject, "predicate": p.predicate, "object": p.object,
              "modality": p.modality} for p in stated],
            ensure_ascii=False,
        )
        prompt = self.prompts.render(
            "meca_infer", passage=text, date=passage.date_raw or "unknown",
            known=known,
        )
        data = self.llm.complete_json(
            prompt, purpose="ingest/meca_infer",
            max_tokens=self.cfg.meca.infer_max_tokens,
            default={"propositions": []},
        )
        parents = tuple(p.make_pid() for p in stated)
        return self._coerce(data, passage, asserted_at, counts, derived_from=parents)

    def _coerce(
        self, data: object, passage: Passage, asserted_at: Optional[TimePoint],
        counts: dict, derived_from: Sequence[str],
    ) -> list[Proposition]:
        rows = (data or {}).get("propositions", []) if isinstance(data, dict) else []
        out: list[Proposition] = []
        for raw in rows or []:
            if not isinstance(raw, dict):
                counts["rejected_malformed"] += 1
                continue
            span = str(raw.get("evidence") or "").strip()
            source = passage.locate(span)
            if source is None:
                # No locatable span means no provenance, and a proposition
                # without provenance is exactly what this design refuses to
                # store. Dropped, and counted so the rate is visible.
                counts["no_evidence"] += 1
                continue
            evidence = [Evidence(
                doc_id=source.doc_id,
                span=span if span in source.text else source.text,
                timestamp=source.timestamp,
                date_raw=source.date_raw,
                author=source.author,
            )]
            prop = coerce_proposition(raw, evidence, asserted_at)
            if prop is None:
                counts["rejected_malformed"] += 1
                continue
            if derived_from:
                prop.derived_from = list(derived_from)
            prop.pid = prop.make_pid()
            out.append(prop)
        return out
