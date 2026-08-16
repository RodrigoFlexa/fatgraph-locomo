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
    whitehead_flip: bool = False  # phase 2


@dataclass
class RetrievalConfig:
    top_m_anchors: int = 5
    budget_tokens: int = 2000
    level2_boost: float = 0.05
    shadowed_penalty: float = 0.10
    incongruent_abstain: bool = True
    recall_ks: tuple[int, ...] = (5, 10)
    max_facts_in_prompt: int = 40
    #: token budget for the answer itself. 64 is plenty for an extractive
    #: phrase, but a *reasoning* deployment spends this budget on internal
    #: reasoning first and then returns an empty string -- give it thousands.
    answer_max_tokens: int = 64


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
        if self.index.backend not in ("numpy", "faiss"):
            raise ConfigError(f"index.backend must be numpy|faiss")
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
