"""Entity resolution: mention -> canonical vertex.

Cascade (cheapest first):

1. exact match on the normalised surface form or on a known alias;
2. embedding similarity against every existing vertex;
   * ``sim >= entities.match_threshold``  -> same vertex, mention recorded as alias
   * ``entities.llm_threshold <= sim <``  -> LLM tie-break on the best candidates
   * ``sim < entities.llm_threshold``     -> new vertex
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from fgl.config import EntityConfig
from fgl.core import FatGraph
from fgl.retrieval.embeddings import Embedder
from fgl.llm import LLMClient
from fgl.logging_utils import JsonlLogger, NullLogger
from fgl.llm.prompts import SYSTEM_JUDGE, PromptLibrary


def normalize_name(name: str) -> str:
    """Casefold, strip accents/punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", (name or "").strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Resolution:
    vertex_id: str
    created: bool
    similarity: float
    method: str  # "exact" | "embedding" | "llm" | "new"


class EntityResolver:
    """Stateful resolver bound to one :class:`FatGraph`."""

    def __init__(
        self,
        graph: FatGraph,
        embedder: Embedder,
        cfg: EntityConfig,
        llm: LLMClient | None = None,
        prompts: PromptLibrary | None = None,
        logger: JsonlLogger | None = None,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.cfg = cfg
        self.llm = llm
        self.prompts = prompts
        self.log = logger or NullLogger()
        self._by_surface: dict[str, str] = {}  # normalised name/alias -> vertex id
        self._matrix: np.ndarray | None = None
        self._matrix_ids: list[str] = []
        rows: list[np.ndarray] = []
        for vid, vx in graph.vertices.items():  # allow resuming a saved graph
            self._register_surface(vid, vx.name)
            for alias in vx.aliases:
                self._register_surface(vid, alias)
            if vx.embedding is not None:
                rows.append(_unit(vx.embedding))
                self._matrix_ids.append(vid)
        if rows:
            self._matrix = np.vstack(rows)

    # ------------------------------------------------------------------ api --
    def resolve(self, mention: str, session_id: str = "") -> Resolution:
        mention = (mention or "").strip()
        if not mention:
            raise ValueError("empty entity mention")

        key = normalize_name(mention)
        if key in self._by_surface:
            return Resolution(self._by_surface[key], False, 1.0, "exact")

        vec = self.embedder.encode_one(mention)
        best_id, best_sim, ranked = self._nearest(vec)

        if best_id is not None and best_sim >= self.cfg.match_threshold:
            self._add_alias(best_id, mention)
            self.log.log(
                "entity_match", mention=mention, vertex=best_id,
                similarity=round(best_sim, 4), method="embedding", session=session_id,
            )
            return Resolution(best_id, False, best_sim, "embedding")

        if best_id is not None and best_sim >= self.cfg.llm_threshold and self.llm:
            for cand_id, sim in ranked[: self.cfg.max_candidates]:
                if sim < self.cfg.llm_threshold:
                    break
                if self._llm_same(mention, cand_id, session_id, sim):
                    self._add_alias(cand_id, mention)
                    return Resolution(cand_id, False, sim, "llm")

        vid = self._create(mention, vec)
        self.log.log(
            "entity_new", mention=mention, vertex=vid,
            best_similarity=round(best_sim, 4) if best_id else None, session=session_id,
        )
        return Resolution(vid, True, best_sim, "new")

    # -------------------------------------------------------------- internals -
    def _nearest(self, vec: np.ndarray) -> tuple[str | None, float, list[tuple[str, float]]]:
        if self._matrix is None or len(self._matrix_ids) == 0:
            return None, -1.0, []
        sims = self._matrix @ _unit(vec)
        order = np.argsort(-sims)
        ranked = [(self._matrix_ids[int(i)], float(sims[int(i)])) for i in order]
        return ranked[0][0], ranked[0][1], ranked

    def _llm_same(self, mention: str, cand_id: str, session_id: str, sim: float) -> bool:
        assert self.llm is not None and self.prompts is not None
        vx = self.graph.vertices[cand_id]
        context = "\n".join(
            f"- {self.graph.H[h].text}" for h in self.graph.sigma.get(cand_id, [])[:5]
        ) or "- (no facts yet)"
        prompt = self.prompts.render(
            "entity_match",
            mention=mention,
            candidate=vx.name,
            aliases=", ".join(vx.aliases) or "(none)",
            context=context,
        )
        out = self.llm.complete_json(
            prompt, system=SYSTEM_JUDGE, purpose="ingest/entity_match",
            default={"same": False, "reason": "unparseable"},
        )
        same = bool(out.get("same"))
        self.log.log(
            "entity_llm_tiebreak", mention=mention, candidate=vx.name, vertex=cand_id,
            similarity=round(sim, 4), same=same, reason=out.get("reason", ""),
            session=session_id,
        )
        return same

    def _create(self, mention: str, vec: np.ndarray) -> str:
        vid = self.graph.add_vertex(mention, aliases=[], embedding=vec)
        self._register_surface(vid, mention)
        unit = _unit(vec).reshape(1, -1)
        self._matrix = unit if self._matrix is None else np.vstack([self._matrix, unit])
        self._matrix_ids.append(vid)
        return vid

    def _add_alias(self, vertex_id: str, mention: str) -> None:
        vx = self.graph.vertices[vertex_id]
        if mention != vx.name and mention not in vx.aliases:
            vx.aliases.append(mention)
        self._register_surface(vertex_id, mention)

    def _register_surface(self, vertex_id: str, surface: str) -> None:
        key = normalize_name(surface)
        if key:
            self._by_surface.setdefault(key, vertex_id)


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)
