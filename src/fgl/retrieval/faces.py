"""Face-based retrieval and question answering.

For each LoCoMo question:

1. top-``m`` anchor half-edges by cosine similarity to the question, with a
   boost for level-2 (consolidation) edges and a penalty for shadowed ones;
2. ``walk_face`` from every anchor, sharing a single global token budget;
3. the answer prompt lists the facts **in face order**, each prefixed with the
   session date, deduplicated across faces;
4. short extractive answer, or the abstention string when the faces do not
   contain the information (or when the anchors are ``incongruente``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from fgl.config import Config
from fgl.core import STATE_INCONGRUENT, FatGraph, HalfEdge
from fgl.retrieval.embeddings import Embedder, VectorIndex, build_index
from fgl.llm import LLMClient
from fgl.data.locomo import ABSTAIN_ANSWER, Conversation, Question
from fgl.llm.prompts import SYSTEM_ANSWERER, PromptLibrary


@dataclass
class RetrievedFact:
    edge_id: str
    text: str
    timestamp: str
    date_raw: str
    session_id: str
    turn_ids: list[str]
    state: str
    level: int
    anchor_rank: int
    anchor_score: float
    face_id: str
    position_in_face: int


@dataclass
class RetrievalResult:
    facts: list[RetrievedFact] = field(default_factory=list)
    anchors: list[tuple[str, float]] = field(default_factory=list)  # (half_edge, score)
    faces: list[str] = field(default_factory=list)
    all_anchor_ranking: list[tuple[str, float]] = field(default_factory=list)
    tokens_used: int = 0
    any_incongruent: bool = False

    @property
    def turn_ids(self) -> list[str]:
        out: list[str] = []
        for f in self.facts:
            for t in f.turn_ids:
                if t not in out:
                    out.append(t)
        return out


# --------------------------------------------------------------------------- #
# Retriever                                                                    #
# --------------------------------------------------------------------------- #


class FaceRetriever:
    def __init__(
        self,
        graph: FatGraph,
        embedder: Embedder,
        cfg: Config,
        date_by_session: dict[str, str] | None = None,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.cfg = cfg
        self.dates = date_by_session or {}
        self.index: VectorIndex = build_index(cfg.index, embedder.dim)
        ids, vecs = [], []
        for hid, he in graph.H.items():
            if he.embedding is None:
                continue
            ids.append(hid)
            vecs.append(he.embedding)
        if ids:
            self.index.add(ids, np.vstack(vecs))

    # ------------------------------------------------------------------ api --
    def retrieve(self, question: str) -> RetrievalResult:
        r = self.cfg.retrieval
        qvec = self.embedder.encode_one(question)
        raw = self.index.search(qvec, max(r.top_m_anchors * 8, 32))
        scored = [(hid, self._adjust(hid, s)) for hid, s in raw]
        scored.sort(key=lambda kv: -kv[1])

        # one anchor per edge: the two halves of a memory are near-duplicates
        anchors: list[tuple[str, float]] = []
        seen_edges: set[str] = set()
        for hid, score in scored:
            eid = self.graph.H[hid].edge_id
            if eid in seen_edges:
                continue
            seen_edges.add(eid)
            anchors.append((hid, score))
            if len(anchors) >= r.top_m_anchors:
                break

        result = RetrievalResult(anchors=anchors, all_anchor_ranking=scored)
        if not anchors:
            return result

        budget = r.budget_tokens
        used = 0
        seen_facts: set[str] = set()
        for rank, (hid, score) in enumerate(anchors):
            if used >= budget:
                break
            if self.graph.H[hid].state == STATE_INCONGRUENT:
                result.any_incongruent = True
            face = self.graph.face_of(hid)
            walk = self.graph.walk_face(hid, budget_tokens=budget - used)
            result.faces.append(face.id)
            for pos, he in enumerate(walk):
                used += self.graph._token_counter(he.text)  # noqa: SLF001
                if he.edge_id in seen_facts:
                    continue
                seen_facts.add(he.edge_id)
                if he.state == STATE_INCONGRUENT:
                    result.any_incongruent = True
                result.facts.append(
                    RetrievedFact(
                        edge_id=he.edge_id,
                        text=he.text,
                        timestamp=he.timestamp,
                        date_raw=self.dates.get(he.session_id, he.timestamp),
                        session_id=he.session_id,
                        turn_ids=list(he.turn_ids),
                        state=he.state,
                        level=he.level,
                        anchor_rank=rank,
                        anchor_score=score,
                        face_id=face.id,
                        position_in_face=pos,
                    )
                )
        result.tokens_used = used
        if len(result.facts) > r.max_facts_in_prompt:
            result.facts = result.facts[: r.max_facts_in_prompt]
        return result

    def top_edges(self, question: str, k: int) -> list[str]:
        """Top-k *edges* by anchor score -- used for the recall@k metric."""
        qvec = self.embedder.encode_one(question)
        out: list[str] = []
        for hid, _ in self.index.search(qvec, max(k * 4, 16)):
            eid = self.graph.H[hid].edge_id
            if eid not in out:
                out.append(eid)
            if len(out) >= k:
                break
        return out

    def turn_ids_for_edges(self, edge_ids: Sequence[str]) -> list[str]:
        out: list[str] = []
        for e in edge_ids:
            for t in self.graph.get_edge_attr(e, "turn_ids"):
                if t not in out:
                    out.append(t)
        return out

    # ------------------------------------------------------------ internals --
    def _adjust(self, half_edge_id: str, score: float) -> float:
        he = self.graph.H[half_edge_id]
        if he.level == 2:
            score += self.cfg.retrieval.level2_boost
        if he.shadowed:
            score -= self.cfg.retrieval.shadowed_penalty
        return score


# --------------------------------------------------------------------------- #
# Answering                                                                    #
# --------------------------------------------------------------------------- #


def render_context(result: RetrievalResult) -> str:
    """Facts grouped by trail, in face order, each prefixed with its date."""
    lines: list[str] = []
    current_face: Optional[str] = None
    trail_no = 0
    for f in result.facts:
        if f.face_id != current_face:
            current_face = f.face_id
            trail_no += 1
            lines.append(f"--- trail {trail_no} ---")
        marker = " (summary)" if f.level == 2 else ""
        flag = " [INCONSISTENT]" if f.state == STATE_INCONGRUENT else ""
        lines.append(f"[{f.date_raw or f.timestamp}]{marker}{flag} {f.text}")
    return "\n".join(lines) if lines else "(no memories retrieved)"


class Answerer:
    def __init__(self, llm: LLMClient, prompts: PromptLibrary, cfg: Config) -> None:
        self.llm = llm
        self.prompts = prompts
        self.cfg = cfg

    def answer(
        self, conv: Conversation, question: Question, result: RetrievalResult
    ) -> str:
        if not result.facts:
            return ABSTAIN_ANSWER
        if self.cfg.retrieval.incongruent_abstain and result.any_incongruent:
            anchors_incongruent = all(
                f.state == STATE_INCONGRUENT
                for f in result.facts
                if f.anchor_rank == 0
            )
            if anchors_incongruent:
                return ABSTAIN_ANSWER
        prompt = self.prompts.render(
            "answer",
            speaker_a=conv.speaker_a,
            speaker_b=conv.speaker_b,
            context=render_context(result),
            question=question.prompt_question(),
        )
        out = self.llm.complete(
            prompt, system=SYSTEM_ANSWERER, purpose="qa/answer", max_tokens=64
        )
        return clean_answer(out)


def clean_answer(text: str) -> str:
    """Strip the framing models add around short extractive answers."""
    t = (text or "").strip()
    for prefix in ("SHORT ANSWER:", "Short answer:", "Answer:", "A:"):
        if t.startswith(prefix):
            t = t[len(prefix) :].strip()
    t = t.strip().strip('"').strip()
    if not t:
        return ABSTAIN_ANSWER
    return t.split("\n")[0].strip()
