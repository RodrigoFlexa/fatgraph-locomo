"""Shared test scaffolding: a hand-fed CLIO instance built from
``tests/fixtures/melanie.yaml`` without going through extraction (M5) --
exactly the M4 milestone's "consolidation fed by hand-written
propositions" mode. Used by the M4 consolidation fixture test and by the
M7 access-algebra tests, which both need the same consolidated graph.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fgl.clio.catalog import Catalog
from fgl.clio.confidence import compute_confidence
from fgl.clio.config import ClioConfig
from fgl.clio.consolidate.fold import FoldRecord
from fgl.clio.consolidate.journal import FoldJournal
from fgl.clio.consolidate.pipeline import consolidate
from fgl.clio.graph.store import GraphStore
from fgl.clio.log.mentions import MentionStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.temporal.resolver import default_for_volatility, resolve_time
from fgl.clio.types import Edge, EvidenceKind, Interval, Operation, Proposition

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "melanie.yaml"


class HandFedMemory:
    """Everything one CLIO instance needs, bundled for a test -- fed with
    hand-classified propositions rather than a real extractor.

    ``with_fold=True`` runs phase 6 (folding, M6) on every
    :meth:`ingest_episode` call, exactly like a caller who passes ``log``
    and ``journal`` to :func:`consolidate` would; the M4 tests need it off
    (their whole point is pinning consolidation WITHOUT folding), so it
    defaults to off rather than being unconditional.
    """

    def __init__(self, catalog: Catalog, with_fold: bool = False):
        self.catalog = catalog
        self.config = ClioConfig.default()
        self.log = LogStore()
        self.mentions = MentionStore()
        self.graph = GraphStore()
        self.staging = StagingStore()
        self.journal = FoldJournal() if with_fold else None
        self.fold_records: list[FoldRecord] = []

    def ingest_episode(self, raw: dict[str, Any], defer: bool = False) -> None:
        """Stands in for :func:`fgl.clio.ingest.pipeline.ingest_turn`:
        resolves time and confidence with the real code, then hands the
        result to real consolidation. What is hand-authored is only the
        *classification* (relation/operation/evidence_kind) an extractor
        would otherwise have produced.
        """
        ts = datetime.strptime(raw["date"], "%Y-%m-%d")
        ep = self.log.append(
            session_id="melanie",
            speaker=raw["speaker"],
            text=raw["text"],
            ts_ingest=ts,
            episode_id=raw["id"],
        )
        for surface in raw.get("mentions", []):
            self.mentions.append(episode_id=ep.id, surface=surface, ts=ts)

        props: list[Proposition] = []
        for i, raw_p in enumerate(raw["propositions"]):
            relation = raw_p["relation"]
            spec = self.catalog[relation]
            t_valid, tconf = resolve_time(raw_p.get("time_expression"), ts, spec)
            # same unanchored fallback the real pipeline applies
            unanchored = t_valid is None
            if unanchored:
                t_valid = default_for_volatility(ts, spec)
            evidence_kind = EvidenceKind(raw_p["evidence_kind"])
            confidence = compute_confidence(evidence_kind, tconf)
            props.append(
                Proposition(
                    id=f"{ep.id}_p{i}",
                    subject_id=raw_p["subject"],
                    relation=relation,
                    object_id=raw_p["object"],
                    operation=Operation(raw_p["operation"]),
                    polarity=raw_p.get("polarity", True),
                    time_expression=raw_p.get("time_expression"),
                    t_valid=t_valid,
                    t_tx=Interval(ts, None),
                    evidence_kind=evidence_kind,
                    confidence=confidence,
                    span=raw_p["span"],
                    episode_id=ep.id,
                    unanchored=unanchored,
                    subject_ref=raw_p["subject"],
                    object_ref=raw_p["object"],
                )
            )
        self.staging.insert(props)
        if defer:
            return
        self.consolidate()

    def consolidate(self) -> None:
        report = consolidate(
            self.catalog,
            self.graph,
            self.staging,
            self.config,
            log=self.log if self.journal is not None else None,
            journal=self.journal,
            mentions=self.mentions,
        )
        self.fold_records.extend(report.folded)

    def entity(self, name: str):
        return next(
            e
            for e in self.graph.all_entities()
            if e.canonical_name == name and e.merged_into is None
        )

    def edge_to(self, name: str) -> Edge:
        ent = self.entity(name)
        return next(e for e in self.graph.all_edges() if e.dst_id == ent.id)


def load_melanie(catalog: Catalog, with_fold: bool = False) -> HandFedMemory:
    mem = HandFedMemory(catalog, with_fold=with_fold)
    doc = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    for episode in doc["episodes"]:
        mem.ingest_episode(episode)
    return mem
