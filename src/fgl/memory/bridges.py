"""LLM-synthesised bridges between episodes that share no slot -- condition L6.

Design: ``docs/L6_DESIGN_bridge_synthesis.md``. Summary of the gap this closes:
L2's typed-slot vocabulary (actor/predicate/concept/type/time,
:mod:`fgl.memory.slots`) and L4/L5's Steiner join channel
(:mod:`fgl.retrieval.steiner`) both connect episodes *through shared
vocabulary* -- the same lemma, the same named entity, the same month. Neither
can connect two episodes that are thematically related but share no surface
form at all (the same underlying event described in different words, a plan
stated in one session and its consequence in another with no repeated noun).
Closing that gap without inventing anything the conversation does not
support, and without ever presupposing what a benchmark conversation looks
like (no speaker-count, turn-count or session-count assumption anywhere in
this module), is the whole job of this file.

Two stages, deliberately split so the expensive one (an LLM call) is spent
only where the cheap one (corpus geometry) says it might pay off:

1. :func:`find_bridge_candidates` -- **zero LLM calls.** Episode-embedding
   cosine similarity, thresholded by the same corpus-derived-quantile recipe
   :func:`fgl.memory.calibration.concept_link_threshold_by_quantile` already
   uses for concept merging, then filtered to drop any pair the existing slot
   graph already connects within a few hops (:func:`fgl.evaluation.hops.episode_hops`)
   -- there is no point paying an LLM call to rediscover a connection Steiner
   already has for free.
2. :func:`synthesize_bridges` -- one LLM call per surviving candidate,
   asking only "is there a concrete connection, and if so what is it and what
   two things does it join" (``prompts/bridge_synthesis.txt``). A "yes"
   materialises a new synthetic episode vertex incident to the two named
   things, exactly like any other episode -- so it is retrieved, rendered and
   scored by the same machinery as everything else, not a special case
   downstream.

Provenance discipline, followed throughout: a bridge vertex/edge is always
flagged (``meta["bridge"]=True`` on the vertex, ``meta["source"]="ingest_bridge"``
on its incidence edges) so it is auditable after the fact, and it is given
turn ids that can never collide with a real dialogue turn id (``"D#:#"``) --
``evidence_recall`` in :mod:`fgl.pipeline` does exact turn-id-set membership,
so a colliding id would silently and spuriously inflate recall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from fgl.config import Config, EntityConfig
from fgl.core import FatGraph
from fgl.evaluation.hops import episode_hops
from fgl.llm import LLMClient
from fgl.llm.prompts import SYSTEM_JUDGE, PromptLibrary
from fgl.logging_utils import JsonlLogger, NullLogger
from fgl.memory.calibration import calibrate, concept_link_threshold_by_quantile
from fgl.memory.entities import normalize_name
from fgl.memory.slots import KIND_ACTOR, KIND_CONCEPT, KIND_EPISODE
from fgl.retrieval.embeddings import Embedder

#: Written into every bridge incidence edge's ``meta["source"]`` -- the one
#: marker that lets anything downstream (a report, a debugger, a test) tell a
#: bridge apart from a fact the ingest actually observed in the transcript.
BRIDGE_SOURCE = "ingest_bridge"


# --------------------------------------------------------------------------- #
# dataclasses                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class BridgeCandidate:
    """One episode pair stage 1 thinks is worth an LLM call."""

    ep_a: str
    ep_b: str
    similarity: float


@dataclass
class BridgeReport:
    """Everything about one ingest's bridging pass -- goes into
    ``report.graph_stats["bridges"]`` so a results directory records *why*
    each bridge exists, not just that it does.
    """

    enabled: bool = False
    n_episodes: int = 0
    threshold: float = 0.0
    threshold_evidence: dict = field(default_factory=dict)
    n_pairs_over_threshold: int = 0
    n_skipped_reachable: int = 0
    n_candidates: int = 0
    n_llm_calls: int = 0
    n_linked: int = 0
    n_rejected: int = 0
    #: A handful of accepted bridges verbatim, capped, for a human to spot-check.
    examples: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# stage 1: candidate pairs, zero LLM                                          #
# --------------------------------------------------------------------------- #


def find_bridge_candidates(
    graph: FatGraph, cfg: Config, logger: JsonlLogger | None = None
) -> tuple[list[BridgeCandidate], dict]:
    """Episode pairs that are semantically close but topologically far.

    "Topologically far" means the existing slot graph does not already
    connect them within ``cfg.bridges.skip_within_hops`` hops -- an LLM call
    is only worth spending on a connection nothing else would have found.
    Returns ``(candidates, meta)``; ``meta`` is folded into
    :class:`BridgeReport` by the caller and is also useful on its own for a
    dry run that never touches an LLM.
    """
    log = logger or NullLogger()
    br = cfg.bridges

    episodes = [
        vid
        for vid, vx in graph.vertices.items()
        if vx.meta.get("kind") == KIND_EPISODE and vx.embedding is not None
    ]
    meta: dict = {"n_episodes": len(episodes)}
    if len(episodes) < 2:
        meta.update(
            threshold=br.floor,
            threshold_evidence={"reason": "fewer than 2 embedded episodes"},
            n_pairs_over_threshold=0,
            n_skipped_reachable=0,
            n_candidates=0,
        )
        return [], meta

    matrix = np.vstack(
        [_unit(graph.vertices[vid].embedding) for vid in episodes]
    ).astype(float)
    threshold, evidence = concept_link_threshold_by_quantile(
        matrix, br.quantile, br.floor
    )
    meta["threshold"] = threshold
    meta["threshold_evidence"] = evidence

    cal = calibrate(cfg, graph)

    def is_hub(vid: str) -> bool:
        kind = graph.vertices[vid].meta.get("kind", KIND_CONCEPT)
        return graph.degree(vid) >= cal.hub_degree(kind)

    # Top-`top_k` neighbours per episode, above `threshold`, deduped into an
    # undirected pair keyed by the lexicographically smaller vertex id so
    # (a, b) and (b, a) collapse to one candidate.
    sims = matrix @ matrix.T
    n = len(episodes)
    seen: dict[tuple[str, str], float] = {}
    for i in range(n):
        order = np.argsort(-sims[i])
        taken = 0
        for j in order:
            j = int(j)
            if j == i:
                continue
            sim = float(sims[i, j])
            if sim < threshold:
                break  # descending order: nothing further clears the bar
            a, b = (episodes[i], episodes[j]) if episodes[i] < episodes[j] else (
                episodes[j], episodes[i]
            )
            key = (a, b)
            if key not in seen or sim > seen[key]:
                seen[key] = sim
            taken += 1
            if taken >= br.top_k:
                break
    meta["n_pairs_over_threshold"] = len(seen)

    # Drop pairs the slot graph already connects: seed the hop search from
    # `a`'s own *non-hub* incident slots -- a hub is a filter, never a
    # bridge (see SlotRetriever.is_hub), so two episodes that share only a
    # hub are not actually connected in any sense the retriever exploits,
    # and must not be treated as "already reachable" here either.
    reached_cache: dict[str, dict[str, int]] = {}
    candidates: list[BridgeCandidate] = []
    n_skipped_reachable = 0
    ordered_pairs = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    for (a, b), sim in ordered_pairs:
        if a not in reached_cache:
            seeds = [s for s in _incident_vertices(graph, a) if not is_hub(s)]
            reached_cache[a] = (
                episode_hops(graph, seeds, is_hub, max_hops=br.skip_within_hops)
                if seeds
                else {}
            )
        if b in reached_cache[a]:
            n_skipped_reachable += 1
            continue
        candidates.append(BridgeCandidate(ep_a=a, ep_b=b, similarity=round(sim, 4)))
        if len(candidates) >= br.max_candidates:
            break

    meta["n_skipped_reachable"] = n_skipped_reachable
    meta["n_candidates"] = len(candidates)
    log.log(
        "bridge_candidates",
        n_episodes=len(episodes),
        threshold=round(float(threshold), 4),
        n_pairs_over_threshold=len(seen),
        n_skipped_reachable=n_skipped_reachable,
        n_candidates=len(candidates),
    )
    return candidates, meta


# --------------------------------------------------------------------------- #
# stage 2: judgment + synthesis, one LLM call per surviving candidate         #
# --------------------------------------------------------------------------- #


def synthesize_bridges(
    graph: FatGraph,
    cfg: Config,
    llm: LLMClient,
    embedder: Embedder,
    prompts: PromptLibrary,
    logger: JsonlLogger | None = None,
) -> BridgeReport:
    """Stage 1 + stage 2, mutating ``graph`` in place. Returns the audit trail."""
    log = logger or NullLogger()
    br = cfg.bridges
    report = BridgeReport(enabled=br.enabled)
    if not br.enabled:
        return report

    candidates, meta = find_bridge_candidates(graph, cfg, logger=log)
    report.n_episodes = meta.get("n_episodes", 0)
    report.threshold = round(float(meta.get("threshold", br.floor)), 4)
    report.threshold_evidence = meta.get("threshold_evidence", {})
    report.n_pairs_over_threshold = meta.get("n_pairs_over_threshold", 0)
    report.n_skipped_reachable = meta.get("n_skipped_reachable", 0)
    report.n_candidates = len(candidates)

    if not candidates:
        return report

    index = _BridgeEntityIndex(graph, embedder, cfg.entities)

    for cand in candidates:
        prompt = prompts.render(
            "bridge_synthesis",
            facts_a=_episode_text(graph, cand.ep_a),
            facts_b=_episode_text(graph, cand.ep_b),
        )
        out = llm.complete_json(
            prompt,
            system=SYSTEM_JUDGE,
            purpose="ingest/bridge_synthesis",
            default={"linked": False},
        )
        report.n_llm_calls += 1

        if not isinstance(out, dict) or not isinstance(out.get("linked"), bool):
            report.n_rejected += 1
            log.log(
                "bridge_rejected", ep_a=cand.ep_a, ep_b=cand.ep_b,
                reason="invalid_llm_schema",
            )
            continue

        if not out["linked"]:
            report.n_rejected += 1
            continue

        bridge_text = out.get("bridge_text")
        entity_1 = out.get("entity_1")
        entity_2 = out.get("entity_2")
        if not all(isinstance(value, str) for value in (bridge_text, entity_1, entity_2)):
            report.n_rejected += 1
            log.log(
                "bridge_rejected", ep_a=cand.ep_a, ep_b=cand.ep_b,
                reason="invalid_llm_schema",
            )
            continue
        bridge_text = bridge_text.strip()
        entity_1 = entity_1.strip()
        entity_2 = entity_2.strip()
        if len(bridge_text) < br.min_bridge_chars or not entity_1 or not entity_2:
            report.n_rejected += 1
            log.log(
                "bridge_rejected", ep_a=cand.ep_a, ep_b=cand.ep_b,
                reason="incomplete_llm_output",
            )
            continue

        bridge_vid = _materialize_bridge(
            graph, index, embedder, cand.ep_a, cand.ep_b,
            bridge_text, entity_1, entity_2,
        )
        if bridge_vid is None:
            report.n_rejected += 1
            log.log(
                "bridge_rejected", ep_a=cand.ep_a, ep_b=cand.ep_b,
                reason="unresolved_or_duplicate_entities",
            )
            continue
        report.n_linked += 1
        if len(report.examples) < 8:
            report.examples.append(
                {
                    "ep_a": cand.ep_a, "ep_b": cand.ep_b,
                    "similarity": cand.similarity, "bridge": bridge_vid,
                    "bridge_text": bridge_text,
                    "entity_1": entity_1, "entity_2": entity_2,
                }
            )
        log.log(
            "bridge_linked", ep_a=cand.ep_a, ep_b=cand.ep_b, bridge=bridge_vid,
            entity_1=entity_1, entity_2=entity_2, similarity=cand.similarity,
        )

    return report


def _materialize_bridge(
    graph: FatGraph,
    index: _BridgeEntityIndex,
    embedder: Embedder,
    ep_a: str,
    ep_b: str,
    bridge_text: str,
    entity_1: str,
    entity_2: str,
) -> str | None:
    """Add the bridge episode vertex and its two incidence edges.

    Idempotent on the vertex id so a repeated (candidate-generation is
    already deduped, but this keeps the invariant true by construction, not
    by trusting the caller): two different LLM calls can never produce two
    vertices for the same (ep_a, ep_b) pair.
    """
    bridge_vid = f"ep:bridge:{ep_a}|{ep_b}"
    turn_id = f"BRIDGE:{ep_a}|{ep_b}"
    if bridge_vid in graph.vertices:
        return bridge_vid

    target_vids = [index.resolve(entity) for entity in (entity_1, entity_2)]
    if any(target_vid is None for target_vid in target_vids):
        return None
    if target_vids[0] == target_vids[1]:
        return None

    source_turn_ids = []
    for episode_vid in (ep_a, ep_b):
        for source_turn_id in graph.vertices[episode_vid].meta.get("turn_ids", []):
            if source_turn_id not in source_turn_ids:
                source_turn_ids.append(source_turn_id)

    vec = embedder.encode_one(bridge_text)
    graph.add_vertex(
        name=bridge_text[:80],
        vertex_id=bridge_vid,
        embedding=vec,
        meta={
            "kind": KIND_EPISODE,
            "turn_ids": [turn_id],
            "turn_texts": [bridge_text],
            "speakers": [],
            "speaker_content": {},
            "mentioned_actors": [],
            "bridge": True,
            "bridge_of": [ep_a, ep_b],
            "bridge_entities": [entity_1, entity_2],
            "bridge_source_turn_ids": source_turn_ids,
        },
    )
    for entity_text, target_vid in zip(
        (entity_1, entity_2), target_vids, strict=True
    ):
        assert target_vid is not None
        # Reflects the vertex actually resolved onto, not always KIND_CONCEPT:
        # "Caroline" can resolve onto a pre-existing KIND_ACTOR vertex, and the
        # edge's own `slot_kind` should say so for anyone auditing it later,
        # even though nothing in retrieval currently reads it back (only the
        # vertex's own `meta["kind"]` is, via `_by_kind`/`touch`).
        target_kind = graph.vertices[target_vid].meta.get("kind", KIND_CONCEPT)
        graph.add_edge(
            bridge_vid,
            target_vid,
            {
                "text": bridge_text,
                "embedding": vec,
                "turn_ids": [turn_id, *source_turn_ids],
                "session_id": "bridge",
                "meta": {
                    "slot_kind": target_kind,
                    "slot_key": normalize_name(entity_text),
                    "surface": entity_text,
                    "source": BRIDGE_SOURCE,
                },
            },
        )
    return bridge_vid


# --------------------------------------------------------------------------- #
# a resolver scoped to concept/actor vertices only                            #
# --------------------------------------------------------------------------- #


class _BridgeEntityIndex:
    """Exact-then-embedding resolver cascade, scoped to concept/actor vertices.

    :class:`fgl.memory.entities.EntityResolver` cannot be reused as-is here:
    its constructor indexes *every* vertex in whatever graph it is given, and
    by the time bridging runs the graph already has episode, predicate, type
    and time vertices in it too. Built on the finished graph, a plain
    ``EntityResolver`` would let a bridge entity mention resolve onto an
    unrelated episode purely by embedding proximity. This mirrors the
    exact-surface-then-embedding half of that same cascade (the LLM
    tie-break tier is skipped, exactly as normal slot ingest already skips it
    for its own concept resolver by passing ``llm=None``), restricted to the
    two kinds a bridge entity can actually mean.
    """

    def __init__(self, graph: FatGraph, embedder: Embedder, cfg: EntityConfig) -> None:
        self.graph = graph
        self.embedder = embedder
        self.cfg = cfg
        self._by_surface: dict[str, str] = {}
        self._matrix: np.ndarray | None = None
        self._matrix_ids: list[str] = []
        rows: list[np.ndarray] = []
        for vid, vx in graph.vertices.items():
            if vx.meta.get("kind") not in (KIND_CONCEPT, KIND_ACTOR):
                continue
            self._register(vid, vx.name)
            for alias in vx.aliases:
                self._register(vid, alias)
            if vx.embedding is not None:
                rows.append(_unit(vx.embedding))
                self._matrix_ids.append(vid)
        if rows:
            self._matrix = np.vstack(rows)

    def resolve(self, mention: str) -> str | None:
        mention = (mention or "").strip()
        key = normalize_name(mention)
        if key and key in self._by_surface:
            return self._by_surface[key]

        vec = self.embedder.encode_one(mention)
        best_id, best_sim = self._nearest(vec)
        if best_id is not None and best_sim >= self.cfg.match_threshold:
            self._add_alias(best_id, mention)
            return best_id
        return None

    def _nearest(self, vec: np.ndarray) -> tuple[str | None, float]:
        if self._matrix is None or not self._matrix_ids:
            return None, -1.0
        sims = self._matrix @ _unit(vec)
        i = int(np.argmax(sims))
        return self._matrix_ids[i], float(sims[i])

    def _register(self, vid: str, surface: str) -> None:
        key = normalize_name(surface)
        if key:
            self._by_surface.setdefault(key, vid)

    def _add_alias(self, vid: str, mention: str) -> None:
        vx = self.graph.vertices[vid]
        if mention != vx.name and mention not in vx.aliases:
            vx.aliases.append(mention)
        self._register(vid, mention)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _incident_vertices(graph: FatGraph, vid: str) -> list[str]:
    """The other endpoint of every half-edge incident to ``vid``, in sigma order."""
    return [graph.H[graph.alpha[hid]].vertex_id for hid in graph.sigma.get(vid, ())]


def _episode_text(graph: FatGraph, ep_vid: str) -> str:
    """The verbatim text an episode vertex stands for -- never a summary.

    Real episodes carry their per-turn texts in ``meta["turn_texts"]``
    (:meth:`fgl.memory.ingest_slots.SlotIngestor._add_episode_vertex`); a
    bridge vertex carries its own single synthesised sentence there under the
    same key, so this function does not need to know which kind it was
    given, and a bridge can itself later be a candidate's endpoint without a
    special case.
    """
    vx = graph.vertices[ep_vid]
    turn_texts = vx.meta.get("turn_texts")
    if turn_texts:
        return "\n".join(turn_texts)
    for hid in graph.sigma.get(ep_vid, ()):
        return graph.H[hid].text
    return vx.name


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)
