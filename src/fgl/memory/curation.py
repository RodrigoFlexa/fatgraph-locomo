"""End-of-session curation: bigon collapse and face consolidation.

Both operations run in the *asynchronous regime*, i.e. once per session after
every fact of that session has been inserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fgl.config import Config
from fgl.core import (
    STATE_CONSOLIDATED,
    Face,
    FatGraph,
    NotABigonError,
    TopologyViolation,
)
from fgl.retrieval.embeddings import Embedder
from fgl.llm import LLMClient
from fgl.data.locomo import Session
from fgl.logging_utils import JsonlLogger, NullLogger
from fgl.llm.prompts import SYSTEM_JUDGE, PromptLibrary


# --------------------------------------------------------------------------- #
# Face stability tracking                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class FaceRecord:
    first_session: int
    last_session: int
    length: int


class FaceTracker:
    """Remembers how long each face id has existed, in sessions.

    A face id is the hash of the canonical rotation of its edge sequence, so it
    changes the moment an edge is inserted into the face -- exactly the
    "estável há >= k sessões" signal consolidation needs.
    """

    def __init__(self) -> None:
        self.records: dict[str, FaceRecord] = {}

    def observe(self, graph: FatGraph, session_num: int) -> None:
        for f in graph.faces():
            rec = self.records.get(f.id)
            if rec is None:
                self.records[f.id] = FaceRecord(session_num, session_num, f.length)
            else:
                rec.last_session = session_num

    def stable_sessions(self, face_id: str, session_num: int) -> int:
        rec = self.records.get(face_id)
        if rec is None or rec.last_session != session_num:
            return 0
        return session_num - rec.first_session + 1


# --------------------------------------------------------------------------- #
# Curator                                                                      #
# --------------------------------------------------------------------------- #


class Curator:
    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        prompts: PromptLibrary,
        logger: JsonlLogger | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.prompts = prompts
        self.log = logger or NullLogger()
        self.embedder = embedder
        self.consolidated_faces: set[str] = set()
        #: bigon faces the judge already declared non-redundant (avoids re-asking
        #: and guarantees the collapse loop terminates)
        self._rejected: set[str] = set()

    # ---------------------------------------------------------- bigons -------
    def collapse_redundant_bigons(self, graph: FatGraph) -> int:
        """LLM-judged collapse of every *genuine* redundancy bigon.

        Only faces of length 2 made of **two distinct parallel edges** qualify.
        Length-2 faces that traverse a single edge twice are leaves/bridges and
        carry no redundancy (COERENCIA.md C3); they are counted and skipped.
        """
        if not self.cfg.curation.curation:
            return 0

        collapsed = 0
        while True:
            candidate: Optional[Face] = None
            for f in graph.faces():
                if f.length == 2 and not f.is_leaf_face and f.id not in self._rejected:
                    candidate = f
                    break
            if candidate is None:
                break

            e_a, e_b = candidate.edges
            ha, _ = graph.edge_half_edges(e_a)
            hb, _ = graph.edge_half_edges(e_b)
            a, b = graph.H[ha], graph.H[hb]
            out = self.llm.complete_json(
                self.prompts.render(
                    "redundancy",
                    a_timestamp=a.timestamp or "?", a_text=a.text,
                    b_timestamp=b.timestamp or "?", b_text=b.text,
                ),
                system=SYSTEM_JUDGE,
                purpose="curation/redundancy",
                default={"redundant": False, "merged_text": "", "reason": "unparseable"},
            )
            if not bool(out.get("redundant")):
                self._rejected.add(candidate.id)
                self.log.log(
                    "bigon_kept", face=candidate.id, edges=list(candidate.edges),
                    reason=str(out.get("reason", ""))[:400],
                )
                continue

            merged = str(out.get("merged_text") or "").strip() or a.text
            try:
                kept = graph.collapse_bigon(candidate.id, merged_text=merged)
            except (NotABigonError, TopologyViolation) as exc:
                self._rejected.add(candidate.id)
                self.log.log("bigon_collapse_failed", face=candidate.id, error=str(exc))
                continue

            if self.embedder is not None:
                vec = self.embedder.encode_one(merged)
                for h in graph.edge_half_edges(kept):
                    graph.H[h].embedding = vec
            collapsed += 1
            self.log.log(
                "bigon_collapsed", face=candidate.id, kept=kept, dropped=e_b,
                merged_text=merged, reason=str(out.get("reason", ""))[:400],
            )
        return collapsed

    # --------------------------------------------------- consolidation -------
    def consolidate(self, graph: FatGraph, tracker: FaceTracker, session: Session) -> int:
        """Summarise long, stable faces into level-2 edges.

        The summary becomes a *chord* of the face: a new edge between the face's
        two extremal vertices, inserted next to the face's own half-edges. The
        original edges stay in the graph with ``shadowed=True`` and are pointed
        at by ``children``.
        """
        if not self.cfg.curation.consolidation:
            return 0

        L = self.cfg.curation.min_face_len
        k = self.cfg.curation.min_stable_sessions
        made = 0

        for face in list(graph.faces()):
            if face.length < L or face.id in self.consolidated_faces:
                continue
            if tracker.stable_sessions(face.id, session.num) < k:
                continue
            edges = list(face.distinct_edges)
            if len(edges) < 2:
                continue
            if any(graph.get_edge_attr(e, "level") == 2 for e in edges):
                continue  # never consolidate a consolidation

            v_start, v_end, h_start, h_end = _extremes(graph, face)
            if v_start is None:
                continue

            facts_block = "\n".join(
                f"[{graph.H[h].timestamp[:10] or '?'}] {graph.H[h].text}"
                for h in face.half_edges
            )
            out = self.llm.complete_json(
                self.prompts.render(
                    "consolidate",
                    facts=facts_block,
                    v_start=graph.vertices[v_start].name,
                    v_end=graph.vertices[v_end].name,
                ),
                system=SYSTEM_JUDGE,
                purpose="curation/consolidate",
                default={"summary": ""},
            )
            summary = str(out.get("summary") or "").strip()
            if not summary:
                continue

            turn_ids: list[str] = []
            for e in edges:
                for t in graph.get_edge_attr(e, "turn_ids"):
                    if t not in turn_ids:
                        turn_ids.append(t)

            payload = {
                "text": summary,
                "embedding": (
                    self.embedder.encode_one(summary) if self.embedder is not None else None
                ),
                "session_id": session.id,
                "turn_ids": turn_ids,
                "timestamp": session.timestamp,
                "state": STATE_CONSOLIDATED,
                "level": 2,
                "children": edges,
                "meta": {"source_face": face.id, "face_length": face.length},
            }
            before = graph.euler()
            pos1 = graph.sigma[v_start].index(h_start) + 1
            pos2 = graph.sigma[v_end].index(h_end) + 1
            new_edge = graph.add_edge(v_start, v_end, payload, pos1, pos2)
            after = graph.euler()

            for e in edges:
                graph.set_edge_attr(e, shadowed=True)

            self.consolidated_faces.add(face.id)
            made += 1
            self.log.log(
                "consolidation",
                face=face.id, face_length=face.length, new_edge=new_edge,
                v_start=v_start, v_end=v_end, children=edges, summary=summary,
                session=session.id,
                euler_before=before.to_dict(), euler_after=after.to_dict(),
            )
            if after.genus != before.genus:
                # legitimate but worth flagging: a chord joining two *different*
                # faces raises the genus instead of splitting one face.
                self.log.log(
                    "consolidation_genus_change",
                    face=face.id, before=before.genus, after=after.genus,
                )
        return made

    # ------------------------------------------------------ whitehead --------
    def whitehead_pass(self, graph: FatGraph, max_flips: int = 0) -> int:
        """Phase 2, off unless ``curation.whitehead_flip`` is true."""
        if not self.cfg.curation.whitehead_flip:
            return 0
        flips = 0
        for edge_id in graph.edges():
            if max_flips and flips >= max_flips:
                break
            try:
                new_edge = graph.whitehead_flip(edge_id)
            except (TopologyViolation, Exception) as exc:  # noqa: BLE001
                self.log.log("whitehead_skipped", edge=edge_id, error=str(exc))
                continue
            flips += 1
            self.log.log("whitehead_flip", old_edge=edge_id, new_edge=new_edge)
        return flips


def _extremes(
    graph: FatGraph, face: Face
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Pick the two "extremal" vertices of a closed trail.

    A face is a cycle and therefore has no endpoints.  We define the extremes as
    the vertex of the canonical first half-edge and the vertex reached halfway
    around the trail -- the pair maximally separated along the traversal.  If
    they coincide we walk forward until a distinct vertex is found.
    (Documented in DECISIONS.md D8.)
    """
    n = face.length
    if n == 0:
        return None, None, None, None
    h_start = face.half_edges[0]
    v_start = graph.H[h_start].vertex_id
    for offset in range(n // 2, n):
        h_end = face.half_edges[offset]
        v_end = graph.H[h_end].vertex_id
        if v_end != v_start:
            return v_start, v_end, h_start, h_end
    return None, None, None, None
