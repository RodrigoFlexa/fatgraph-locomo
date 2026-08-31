"""CLIO configuration (spec section 14).

Deliberately narrower than the spec document's own YAML in a few places:
no ``retrieval`` section (the hybrid-search weighting lives in
:class:`fgl.clio.index.EntityIndex` as a fixed, checked constant --
``min_score``'s docstring explains the number), and no ``tau_entity``
(phase 1's entity reuse is an exact-name match, not a threshold -- see
:mod:`fgl.clio.consolidate.entities`). A config field nothing reads is
dead weight, not forward-compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TemporalConfig:
    #: width of the default window for a "fast" relation with no explicit
    #: time expression (spec 5.3).
    fast_window_days: int = 1
    #: LoCoMo, the target corpus, is English dialogue -- see
    #: fgl.clio.temporal.patterns_en, the only locale implemented so far.
    locale: str = "en_US"


@dataclass
class ThresholdsConfig:
    tau_promote: float = 0.70  # min confidence to reach the graph directly
    tau_fold: float = 0.80  # min identity confidence to fold two vertices (M6)


@dataclass
class ExtractionConfig:
    #: prior turns shown to the extractor for coreference only (spec 6.2a)
    coref_window: int = 3
    #: entity candidates offered to the extractor (spec 6.2b)
    max_candidates: int = 20
    #: completion budget for ONE extraction call. The repository-wide
    #: default (``LLMConfig.max_tokens = 512``) is sized for short answers
    #: and is far too small here: one proposition serialises to roughly 90
    #: tokens, so a turn yielding six of them is cut off mid-object and
    #: the whole response fails to parse -- silently becoming "0
    #: propositions" for that turn. Measured on conv-26: 3 of 58 turns
    #: (5%) were lost exactly this way, and it is the RICH turns that get
    #: lost, which is the worst possible bias. 2000 matches what
    #: fgl.config's own MECA extractor already uses for the same job.
    max_tokens: int = 2000


@dataclass
class AccessConfig:
    #: max movements the agent loop may spend on one question (spec 10.3)
    movement_budget: int = 8
    #: candidates kept per `expand` call (spec 9.5)
    expand_k: int = 10
    #: personalized-pagerank restart probability (spec 9.5)
    ppr_alpha: float = 0.15
    #: hop radius `expand`'s spreading activation is restricted to
    expand_max_hops: int = 2
    #: hard cap after relevance ranking; prevents a high-degree speaker from
    #: flooding both the next decision prompt and the final answer.
    trail_limit: int = 20
    #: direct log candidates contributed by the episodic half of anchor.
    anchor_episode_k: int = 5
    #: source episodes the answer writer may see after deterministic ranking.
    answer_evidence_limit: int = 12


@dataclass
class Clio2Config:
    #: Staged facts remain lower-confidence ledger candidates. The immutable
    #: episode is still required before they may support an answer.
    include_staged_facts: bool = True
    #: Actual raw episodes passed to the typed answerer after value coverage
    #: ranking. Unlike CLIO1's metric, this is also the set reported as recall.
    answer_evidence_limit: int = 16


@dataclass
class ClioConfig:
    catalog_path: str = str(Path(__file__).parent / "catalog" / "personal_dialogue.yaml")
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    access: AccessConfig = field(default_factory=AccessConfig)
    clio2: Clio2Config = field(default_factory=Clio2Config)
    #: ``agent`` preserves the original experimental reader; ``clio2`` uses
    #: the compiled query engine.
    reader: str = "agent"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClioConfig:
        p = Path(path)
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw = raw.get("clio", raw)  # allow either a bare doc or a `clio:` root key
        kwargs: dict[str, Any] = {}
        if "catalog_path" in raw:
            kwargs["catalog_path"] = raw["catalog_path"]
        if "temporal" in raw:
            kwargs["temporal"] = TemporalConfig(**raw["temporal"])
        if "thresholds" in raw:
            kwargs["thresholds"] = ThresholdsConfig(**raw["thresholds"])
        if "extraction" in raw:
            kwargs["extraction"] = ExtractionConfig(**raw["extraction"])
        if "access" in raw:
            kwargs["access"] = AccessConfig(**raw["access"])
        if "clio2" in raw:
            kwargs["clio2"] = Clio2Config(**raw["clio2"])
        if "reader" in raw:
            reader = str(raw["reader"])
            if reader not in ("agent", "clio2"):
                raise ValueError("clio.reader must be 'agent' or 'clio2'")
            kwargs["reader"] = reader
        return cls(**kwargs)

    @classmethod
    def default(cls) -> ClioConfig:
        return cls()
