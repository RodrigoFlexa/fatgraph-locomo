"""Experiment configuration: YAML files -> dataclasses, with CLI overrides.

Resolution order for any single value, highest priority first:

1. ``--set dotted.key=value`` on the command line;
2. environment / ``.env`` (only for what :class:`fgl.settings.Settings` covers);
3. the condition YAML in ``configs/conditions/``;
4. ``configs/base.yaml``;
5. the dataclass defaults below.

No secret is ever read from here. Credentials live in ``.env`` / the
environment; only the Azure *deployment name* appears in YAML.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from fgl.paths import Paths, project_root


# --------------------------------------------------------------------------- #
# Sections                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class LLMConfig:
    provider: str = "azure"  # azure | fake
    deployment: str = "gpt-4o-mini"  # Azure deployment name, never a secret
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 1234
    request_timeout: float = 60.0
    max_retries: int = 6
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    cache_dir: str = ".cache/llm"
    cache_enabled: bool = True

    # --- health guards ----------------------------------------------------
    #: abort instead of silently turning empty completions into abstentions
    fail_on_empty: bool = True
    #: how many live calls before the sustained-empty check kicks in
    health_check_calls: int = 5
    #: tolerated fraction of empty completions past that point
    max_empty_rate: float = 0.2

    # --- model family ------------------------------------------------------
    #: auto | chat | reasoning. Reasoning deployments (o1/o3/o4-mini, gpt-5)
    #: take max_completion_tokens, reject a custom temperature, and burn the
    #: token budget on internal reasoning before emitting anything.
    api_style: str = "auto"
    #: floor applied to the token budget for reasoning models; 0 omits the cap
    #: entirely, which is what a working notebook against gpt-5 usually does
    reasoning_min_tokens: int = 4000
    #: some gateways reject an explicit temperature even on chat models
    send_temperature: bool = True
    #: reasoning budget hint: minimal | low | medium | high | "" to omit.
    #: LoCoMo answers are short and extractive, so "low" keeps the reasoning
    #: tokens (which are billed) from dominating the cost.
    reasoning_effort: str = "low"


@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers"  # sentence-transformers | azure | hashing
    model: str = "all-MiniLM-L6-v2"
    azure_deployment: str = "text-embedding-3-small"
    dim: int = 384  # hashing provider only
    batch_size: int = 64
    cache_dir: str = ".cache/embeddings"
    normalize: bool = True


@dataclass
class IndexConfig:
    backend: str = "numpy"  # numpy | faiss
    metric: str = "cosine"


@dataclass
class EntityConfig:
    match_threshold: float = 0.85
    llm_threshold: float = 0.70
    max_candidates: int = 5


@dataclass
class IngestConfig:
    sigma_policy: str = "sigma-time"  # sigma-time | sigma-agent
    #: Which extraction prompt to use.
    #:
    #: `extract_facts` (v1) names the speaker as one endpoint of most facts, and
    #: the measured consequence is that 86.6% of edges touch a speaker, the two
    #: speakers are the two highest-degree vertices in every conversation (~200
    #: against 17 for the third), and the median bridge vertex between a pair of
    #: evidence facts has degree 164. Entity sharing then means "both are about
    #: Caroline", so sigma's orbit covers the real bridge in 7.3% of cases and
    #: raising k plateaus at 11%. Every retrieval mechanism G4-G10 was tested on
    #: that substrate.
    #:
    #: `extract_facts_topical` (v2) connects the two *topics* a fact relates and
    #: records the speaker as metadata. `fact_text` is unchanged either way, so
    #: B3 embeds the same sentences and the only variable left between it and
    #: the graph conditions is the vertex assignment.
    extract_prompt: str = "extract_facts"
    detect_incongruence: bool = True
    max_facts_per_session: int = 0
    allow_self_loops: bool = False
    sigma_agent_max_trails: int = 8
    sigma_agent_trail_chars: int = 160


@dataclass
class CurationConfig:
    curation: bool = False
    consolidation: bool = False
    min_face_len: int = 4  # L
    min_stable_sessions: int = 2  # k
    #: Whitehead moves preserve the surface -- `FatGraph.whitehead_flip` asserts
    #: that genus and F are unchanged, because contracting and re-expanding an
    #: edge is a spine move. They therefore cannot reshape faces; the move that
    #: does is a transposition in sigma, below.
    whitehead_flip: bool = False  # phase 2
    #: After ingest, hill-climb on sigma to maximise the face count. By Euler
    #: (F = 2C - 2g + E - V) that is exactly minimising genus, and over fixed V
    #: and E more faces means shorter ones -- which attacks the measured failure
    #: of face retrieval, namely boundary walks of 300+ half-edges produced by
    #: ordering sigma by the clock. Needs its own graphs: a different sigma is a
    #: different ribbon graph, so such a condition cannot borrow G1's.
    maximize_faces: bool = False
    maximize_faces_passes: int = 6
    #: bound on the pair enumeration inside one vertex; the speaker hubs reach
    #: degree in the hundreds and would dominate the cost quadratically
    maximize_faces_degree_scan: int = 48


@dataclass
class RetrievalConfig:
    top_m_anchors: int = 5
    budget_tokens: int = 2000
    level2_boost: float = 0.05
    shadowed_penalty: float = 0.10
    incongruent_abstain: bool = True
    recall_ks: tuple[int, ...] = (5, 10)
    max_facts_in_prompt: int = 40
    #: ceiling on the share of ``max_facts_in_prompt`` the multi-hop mechanisms
    #: (sigma / coverage / geodesic) may occupy when the two compete. Absolute
    #: priority for the joins would let coverage evict *every* anchor fact, and
    #: G5/G6 would then stop being supersets of G1 -- the very comparison the
    #: conditions exist to make. Ignored when both flags are off.
    max_facts_join_frac: float = 0.5
    #: token budget for the answer itself. 64 is plenty for an extractive
    #: phrase, but a *reasoning* deployment spends this budget on internal
    #: reasoning first and then returns an empty string -- give it thousands.
    answer_max_tokens: int = 64

    #: Route LoCoMo category 3 (open-domain) to a prompt that permits
    #: inference. Those questions ask what is *likely*, so under the extractive
    #: instruction the model abstains on a prompt that already holds the
    #: evidence -- measured at F1 0.069 for full-context, whose evidence recall
    #: in that category was 0.97. Applies to the baselines too, so the arms
    #: stay comparable.
    open_domain_inference: bool = True

    #: Shuffle the retrieved facts before rendering the prompt.
    #: The entire distinctive content of a ribbon graph over a plain graph is
    #: *order* -- sigma is a cyclic ordering, phi is the walk it induces. So
    #: this is the load-bearing ablation for the whole thesis: if F1 does not
    #: move when the order is destroyed, then no amount of sigma optimisation
    #: can help, and the honest conclusion is that ordering carries no signal
    #: for the reader. Deterministic, seeded from `seed`.
    shuffle_context: bool = False

    # --- sigma expansion (multi-hop) --------------------------------------
    # A multi-hop question needs two memories that *share an entity*, i.e. two
    # half-edges in the same sigma-orbit. phi = sigma o alpha leaves the vertex
    # at every step, so the face only comes back to that entity after a full
    # lap -- usually past the token budget. Walking sigma directly is the join.
    # Off by default: G1/G2/G3 must keep producing byte-identical numbers.
    #: master switch. False => retrieve() behaves exactly as before.
    sigma_expand: bool = False
    #: how many neighbours to keep per orbit (after reranking)
    sigma_expand_k: int = 4
    #: expand from alpha(h) too, i.e. from *both* entities of the anchor memory
    sigma_expand_both_ends: bool = True
    #: only the top-N anchors get expanded (0 = all of them)
    sigma_expand_max_anchors: int = 2
    #: fraction of budget_tokens reserved for sigma neighbours
    sigma_budget_frac: float = 0.4
    #: rank the whole orbit by similarity to the question instead of taking the
    #: cyclic successors. Under sigma-time the successor is merely the
    #: chronologically adjacent fact, which is rarely the bridge.
    sigma_rerank: bool = True
    #: cap on how many half-edges of one orbit get scored (high-degree hubs).
    #: 0 = no cap, matching sigma_expand_max_anchors and max_facts_per_session.
    sigma_max_orbit_scan: int = 64

    #: Skip vertices of degree >= this when expanding sigma. 0 disables.
    #:
    #: The speaker is the `"the"` of this graph. Measured: 86% of edges touch
    #: one of the two speakers, they are the two highest-degree vertices in
    #: every conversation (~200 against 9-18 for the third), and the median
    #: vertex shared by a pair of evidence facts has degree 115. So "these two
    #: memories share an entity" degenerates into "both are about Caroline",
    #: true of half the graph, and sigma's orbit contains the real bridge in
    #: 7.7% of cases -- raising k plateaus at 11% because degrees reach 229.
    #:
    #: Information retrieval settled this in the 1960s: you do not index a
    #: stopword. A hub vertex carries no discriminative information, so the
    #: orbit through it is noise, and skipping it forces the bridge to be a
    #: real entity or to not exist. This is the same idea applied to vertices
    #: instead of terms -- and it is the test that decides whether semantic
    #: bridges exist in this data at all, because it needs no re-ingest.
    #:
    #: The threshold is not a guess: degrees are bimodal with an empty band
    #: between them. Across the ten conversations the speakers never fall below
    #: 95 and the third-ranked vertex never rises above 50, so a cut at 60
    #: removes exactly the 20 speaker vertices and nothing else -- 100%
    #: precision, measured, not tuned.
    sigma_skip_hub_degree: int = 0

    # --- face as the unit of retrieval (G10) -------------------------------
    # One line of difference from the k-NN baseline: B3 returns the top-k
    # facts, this returns the *faces containing* them, whole. What a face adds
    # is the memory that does not match the question but belongs to the same
    # narrative unit as one that does -- corroboration, which k independent
    # matches cannot produce.
    #
    # A face is a SET here, not a path. Three measurements forced that: walking
    # phi lost 0.21 of multi-hop recall; picking faces by entity coverage was
    # null (coverage saturated at 0.955); and permuting the prompt was null in
    # multi-hop, so sequence never carried the signal. Membership did.
    #
    # PRECONDITION: curation.maximize_faces. Under the clock-ordered rotation
    # 19 faces hold 75% of the memory and the median half-edge sits in a face
    # of 263 -- there is no "unit" to retrieve. After the genus search the
    # distribution is unimodal with median 36. The unit only exists there.
    #
    # Ranking is max member similarity, on purpose: a face containing a
    # relevant fact IS the coverage signal, with no linker, threshold,
    # aggregation mode or weight to tune.
    face_units: bool = False

    # --- face-first retrieval by entity coverage (multi-hop) ---------------
    # The anchor ranking asks "which fact looks like the question?", which is
    # what every RAG does and what makes single-hop easy. A multi-hop question
    # names two entities and the answer lies on a trail *between* them, so the
    # useful question is "which face covers the entities the question names?".
    # Coverage is a structural signal: a face through both vertices is a
    # candidate bridge even when none of its facts resembles the question --
    # precisely what cosine cannot express.
    # Off by default, for the same reason as sigma_expand.
    face_coverage: bool = False
    #: weight of the coverage term against the similarity term
    coverage_weight: float = 1.0
    #: how to aggregate member similarity into a face score: max | top2 | mean.
    #: mean penalises long faces, which are exactly the cross-session ones.
    coverage_sim_aggregate: str = "top2"
    #: fraction of budget_tokens reserved for the covering faces
    coverage_budget_frac: float = 0.4
    #: at most this many entities are linked from the question
    coverage_max_entities: int = 4
    #: at most this many candidate faces get scored, *in total*. The candidates
    #: are drawn round-robin across the question's entities: taking them entity
    #: by entity would let the first one exhaust the cap, and a bridge face is
    #: by definition one that shows up under more than one of them.
    coverage_max_faces: int = 24
    #: at most this many memories taken from *each* covering face. Faces are
    #: long (200+ is normal, see COERENCIA C9), so without this cap a single
    #: trail fills max_facts_in_prompt on its own and starves both the anchor
    #: walk and the sigma expansion -- measured, not hypothesised.
    coverage_max_facts_per_face: int = 20
    #: minimum cosine to accept a vertex as "named by the question" when the
    #: match is not a literal surface match
    coverage_entity_threshold: float = 0.75
    #: when no face covers 2+ of the question's entities, retrieve the shortest
    #: path between them instead: a length-2 path *is* the bridging memory pair
    coverage_geodesic_fallback: bool = True
    coverage_geodesic_max_depth: int = 3


@dataclass
class BaselineConfig:
    rag_top_k: int = 10
    full_context_max_tokens: int = 110_000
    full_context_truncate: str = "head"


@dataclass
class PathsConfig:
    locomo_repo: str = "data/external/locomo"
    data_file: str = "data/external/locomo/data/locomo10.json"
    prompts_dir: str = "prompts"
    results_dir: str = "results"
    graphs_dir: str = "artifacts/graphs"
    facts_cache: str = "artifacts/facts"
    logs_dir: str = "artifacts/logs"
    #: read the memory graphs from *another* condition's directory instead of
    #: ``graphs_dir/<condition>``. Empty = use this condition's own name.
    #: Lets a retrieval-only ablation (G4) run on G1's byte-identical graphs,
    #: so the delta isolates retrieval and costs no LLM calls.
    graphs_condition: str = ""


SECTIONS: dict[str, type] = {
    "llm": LLMConfig,
    "embeddings": EmbeddingConfig,
    "index": IndexConfig,
    "entities": EntityConfig,
    "ingest": IngestConfig,
    "curation": CurationConfig,
    "retrieval": RetrievalConfig,
    "baselines": BaselineConfig,
    "paths": PathsConfig,
}


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    condition: str = "G1-fatgraph-min"
    seed: int = 1234
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    entities: EntityConfig = field(default_factory=EntityConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    #: filled in by :meth:`load`, for provenance in the results manifest
    source: Optional[str] = None

    # ------------------------------------------------------------- loading --
    @classmethod
    def load(
        cls,
        condition: str | Path,
        overrides: Iterable[str] | None = None,
        root: Path | None = None,
        settings=None,
    ) -> "Config":
        """Load by condition name (``G1``), file stem or explicit path.

        ``overrides`` are ``dotted.key=value`` strings from ``--set``.
        ``settings`` is an optional :class:`fgl.settings.Settings` whose model
        choices are overlaid *before* the CLI overrides, so ``--set`` always wins.
        """
        path = resolve_condition(condition, root)
        cfg = cls.from_yaml(path)
        cfg.source = str(path)
        if settings is not None:
            settings.apply_to(cfg)
        if overrides:
            cfg.apply_overrides(overrides)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "Config":
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        base_name = raw.pop("extends", None)
        if base_name:
            base_path = (p.parent / base_name).resolve()
            if not base_path.exists():  # allow `extends: base.yaml` from conditions/
                base_path = (p.parent.parent / base_name).resolve()
            merged = deep_merge(cls.from_yaml(base_path).to_dict(), raw)
            merged.pop("source", None)
        else:
            merged = raw
        if overrides:
            merged = deep_merge(merged, overrides)
        return cls.from_dict(merged)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = copy.deepcopy(d)
        d.pop("source", None)
        kwargs: dict[str, Any] = {}
        for key, klass in SECTIONS.items():
            payload = d.pop(key, {}) or {}
            unknown = set(payload) - set(klass.__dataclass_fields__)
            if unknown:
                raise ConfigError(
                    f"unknown keys in config.{key}: {sorted(unknown)} "
                    f"(valid: {sorted(klass.__dataclass_fields__)})"
                )
            if key == "retrieval" and "recall_ks" in payload:
                payload["recall_ks"] = tuple(payload["recall_ks"])
            kwargs[key] = klass(**payload)
        unknown_top = set(d) - {"condition", "seed"}
        if unknown_top:
            raise ConfigError(f"unknown top-level config keys: {sorted(unknown_top)}")
        kwargs.update(d)
        return cls(**kwargs)

    # ----------------------------------------------------------- overrides --
    def apply_overrides(self, overrides: Iterable[str]) -> "Config":
        """Apply ``--set dotted.key=value`` strings, typed from the dataclass."""
        for item in overrides:
            if "=" not in item:
                raise ConfigError(
                    f"malformed override {item!r}; expected section.key=value"
                )
            dotted, _, raw = item.partition("=")
            self.set(dotted.strip(), raw.strip())
        return self

    def set(self, dotted: str, raw: str) -> None:
        parts = dotted.split(".")
        target: Any = self
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ConfigError(f"unknown config section {part!r} in {dotted!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not is_dataclass(target) or leaf not in target.__dataclass_fields__:
            raise ConfigError(
                f"unknown config key {dotted!r}. Try: fgl config keys"
            )
        current = getattr(target, leaf)
        setattr(target, leaf, coerce(raw, current))

    def get(self, dotted: str) -> Any:
        target: Any = self
        for part in dotted.split("."):
            target = getattr(target, part)
        return target

    # ---------------------------------------------------------- validation --
    def validate(self) -> "Config":
        if self.llm.provider not in ("azure", "fake"):
            raise ConfigError(f"llm.provider must be azure|fake, got {self.llm.provider!r}")
        if self.embeddings.provider not in ("sentence-transformers", "azure", "hashing"):
            raise ConfigError(f"unknown embeddings.provider {self.embeddings.provider!r}")
        if self.llm.api_style not in ("auto", "chat", "reasoning"):
            raise ConfigError(
                f"llm.api_style must be auto|chat|reasoning, got {self.llm.api_style!r}"
            )
        if self.index.backend not in ("numpy", "faiss"):
            raise ConfigError(f"index.backend must be numpy|faiss")
        if self.ingest.extract_prompt not in ("extract_facts", "extract_facts_topical"):
            raise ConfigError(
                "ingest.extract_prompt must be extract_facts|extract_facts_topical, "
                f"got {self.ingest.extract_prompt!r}"
            )
        if self.ingest.sigma_policy not in ("sigma-time", "sigma-agent"):
            raise ConfigError(
                f"ingest.sigma_policy must be sigma-time|sigma-agent, "
                f"got {self.ingest.sigma_policy!r}"
            )
        if not 0 <= self.entities.llm_threshold <= self.entities.match_threshold <= 1:
            raise ConfigError(
                "require 0 <= entities.llm_threshold <= entities.match_threshold <= 1"
            )
        if self.retrieval.top_m_anchors < 1:
            raise ConfigError("retrieval.top_m_anchors must be >= 1")
        if self.retrieval.budget_tokens < 1:
            raise ConfigError("retrieval.budget_tokens must be >= 1")
        if self.curation.min_face_len < 2:
            raise ConfigError("curation.min_face_len must be >= 2")
        if self.curation.maximize_faces:
            if self.curation.maximize_faces_passes < 1:
                raise ConfigError("curation.maximize_faces_passes must be >= 1")
            if self.curation.maximize_faces_degree_scan < 0:
                raise ConfigError(
                    "curation.maximize_faces_degree_scan must be >= 0 (0 = no bound)"
                )
            if self.paths.graphs_condition:
                raise ConfigError(
                    "curation.maximize_faces rewrites sigma, so the graph is a "
                    "different ribbon graph and cannot be borrowed: leave "
                    f"paths.graphs_condition empty (got "
                    f"{self.paths.graphs_condition!r})"
                )
        if self.retrieval.sigma_expand:
            if self.retrieval.sigma_expand_k < 1:
                raise ConfigError("retrieval.sigma_expand_k must be >= 1")
            if not 0.0 <= self.retrieval.sigma_budget_frac < 1.0:
                raise ConfigError(
                    "retrieval.sigma_budget_frac must be in [0, 1) -- the face "
                    "walk needs what is left of the budget"
                )
            if self.retrieval.sigma_expand_max_anchors < 0:
                raise ConfigError(
                    "retrieval.sigma_expand_max_anchors must be >= 0 (0 = all anchors)"
                )
            if self.retrieval.sigma_max_orbit_scan < 0:
                raise ConfigError(
                    "retrieval.sigma_max_orbit_scan must be >= 0 (0 = no cap)"
                )
            if self.retrieval.sigma_skip_hub_degree < 0:
                raise ConfigError(
                    "retrieval.sigma_skip_hub_degree must be >= 0 (0 = disabled)"
                )
            if 0 < self.retrieval.sigma_skip_hub_degree <= 2:
                raise ConfigError(
                    "retrieval.sigma_skip_hub_degree <= 2 would skip almost every "
                    "vertex and disable the expansion instead of focusing it"
                )
        if self.retrieval.face_units:
            for flag in ("sigma_expand", "face_coverage"):
                if getattr(self.retrieval, flag):
                    raise ConfigError(
                        f"retrieval.face_units retrieves whole faces; "
                        f"retrieval.{flag} is a different retrieval policy and "
                        "combining them would measure neither"
                    )
            if not self.curation.maximize_faces:
                raise ConfigError(
                    "retrieval.face_units requires curation.maximize_faces: "
                    "under the clock-ordered rotation a handful of faces hold "
                    "most of the memory (median half-edge in a face of 263), "
                    "so there is no unit to retrieve"
                )
        if self.retrieval.face_coverage:
            if self.retrieval.coverage_sim_aggregate not in ("max", "top2", "mean"):
                raise ConfigError(
                    "retrieval.coverage_sim_aggregate must be max|top2|mean, got "
                    f"{self.retrieval.coverage_sim_aggregate!r}"
                )
            if not 0.0 <= self.retrieval.coverage_budget_frac < 1.0:
                raise ConfigError(
                    "retrieval.coverage_budget_frac must be in [0, 1)"
                )
            if self.retrieval.coverage_max_entities < 1:
                raise ConfigError("retrieval.coverage_max_entities must be >= 1")
            if self.retrieval.coverage_max_faces < 1:
                raise ConfigError("retrieval.coverage_max_faces must be >= 1")
            if self.retrieval.coverage_geodesic_max_depth < 1:
                raise ConfigError("retrieval.coverage_geodesic_max_depth must be >= 1")
        if self.retrieval.sigma_expand or self.retrieval.face_coverage:
            if not 0.0 < self.retrieval.max_facts_join_frac <= 1.0:
                raise ConfigError(
                    "retrieval.max_facts_join_frac must be in (0, 1]"
                )
            # The anchor index L2-normalises internally, so anchors are cosine
            # either way; the coverage and sigma scorers dot the raw vectors.
            # With normalisation off, `sim` leaves [-1, 1] and the
            # `sim + coverage_weight * coverage` sum stops meaning anything.
            if not self.embeddings.normalize:
                raise ConfigError(
                    "retrieval.sigma_expand / face_coverage require "
                    "embeddings.normalize=true: they score half-edges by raw "
                    "dot product, which is only cosine on unit vectors"
                )
        reserved = 0.0
        if self.retrieval.sigma_expand:
            reserved += self.retrieval.sigma_budget_frac
        if self.retrieval.face_coverage:
            reserved += self.retrieval.coverage_budget_frac
        if reserved >= 1.0:
            raise ConfigError(
                f"sigma_budget_frac + coverage_budget_frac = {reserved:.2f} >= 1: "
                "nothing would be left for the anchor face walk"
            )
        return self

    def requires_azure(self) -> bool:
        return self.llm.provider == "azure" or self.embeddings.provider == "azure"

    # -------------------------------------------------------------- output --
    def to_dict(self) -> dict:
        d = asdict(self)
        d["retrieval"]["recall_ks"] = list(self.retrieval.recall_ks)
        return d

    def to_yaml(self) -> str:
        d = self.to_dict()
        d.pop("source", None)
        return yaml.safe_dump(d, sort_keys=False, allow_unicode=True)

    def flat(self) -> dict[str, Any]:
        """``{"llm.deployment": "gpt-4o-mini", ...}`` — what ``--set`` accepts."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value):
                for sub in fields(value):
                    out[f"{f.name}.{sub.name}"] = getattr(value, sub.name)
            elif f.name != "source":
                out[f.name] = value
        return out

    def diff(self, other: "Config") -> dict[str, tuple[Any, Any]]:
        a, b = self.flat(), other.flat()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    def resolved_paths(self, root: Path | None = None) -> dict[str, Path]:
        p = Paths.build(root or project_root())
        return {f.name: p.resolve(getattr(self.paths, f.name)) for f in fields(self.paths)}


# --------------------------------------------------------------------------- #
# Condition discovery                                                          #
# --------------------------------------------------------------------------- #


def conditions_dir(root: Path | None = None) -> Path:
    return Paths.build(root or project_root()).conditions


def list_conditions(root: Path | None = None) -> list[tuple[str, str, Path]]:
    """``[(name, condition_id, path), ...]`` for every YAML in conditions/."""
    out = []
    for p in sorted(conditions_dir(root).glob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cond = raw.get("condition", p.stem)
        except Exception:
            cond = p.stem
        out.append((p.stem, cond, p))
    return out


def resolve_condition(condition: str | Path, root: Path | None = None) -> Path:
    """Accept a path, a file stem (``G1_fatgraph_min``), an id (``G1-fatgraph-min``)
    or a short prefix (``G1``)."""
    p = Path(condition)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()

    key = str(condition).strip().lower().replace("-", "_")
    entries = list_conditions(root)
    if not entries:
        raise ConfigError(f"no conditions found in {conditions_dir(root)}")

    exact = [e for e in entries if e[0].lower() == key or e[1].lower().replace("-", "_") == key]
    if len(exact) == 1:
        return exact[0][2]

    # The leading id segment, matched whole: `G1` names G1_fatgraph_min and not
    # G10_face_units. Without this a plain prefix search makes every short id
    # ambiguous the moment a tenth condition exists -- which is a property of
    # the numbering, not of the user's request.
    ident = [e for e in entries if e[0].split("_")[0].lower() == key]
    if len(ident) == 1:
        return ident[0][2]

    prefix = [e for e in entries if e[0].lower().startswith(key) or
              e[1].lower().replace("-", "_").startswith(key)]
    if len(prefix) == 1:
        return prefix[0][2]
    if len(prefix) > 1:
        raise ConfigError(
            f"{condition!r} is ambiguous: {sorted(e[1] for e in prefix)}"
        )
    raise ConfigError(
        f"unknown condition {condition!r}. Available: "
        + ", ".join(sorted(e[1] for e in entries))
    )


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def coerce(raw: str, current: Any) -> Any:
    """Cast a CLI string to the type of the value it replaces.

    Every failure surfaces as :class:`ConfigError` so the CLI can report it as a
    configuration problem (exit 2) instead of crashing with a traceback.
    """
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ConfigError(f"expected a boolean (true/false), got {raw!r}")
    if isinstance(current, tuple):
        try:
            return tuple(int(x) for x in raw.strip("[]() ").split(",") if x.strip())
        except ValueError as exc:
            raise ConfigError(
                f"expected a comma-separated list of integers, got {raw!r}"
            ) from exc
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"expected an integer, got {raw!r}") from exc
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"expected a number, got {raw!r}") from exc
    if current is None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw
