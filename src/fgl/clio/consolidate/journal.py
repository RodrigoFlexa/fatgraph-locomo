"""The fold journal (spec 8.4): every merge is reversible. Without this,
one wrong "Rui" == "Rui Sampaio" contaminates a region of the graph
permanently, because folding is a congruence and it propagates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FoldRecord:
    id: str
    kept: str
    absorbed: str
    score: float
    trigger_episode: str
    #: ids of every edge that was touching ``absorbed`` at fold time and
    #: got migrated to ``kept`` -- exactly what ``unfold`` moves back, not
    #: "everything currently on kept" (which may include unrelated later
    #: writes or a second, later fold).
    migrated_edge_ids: list[str] = field(default_factory=list)
    #: the absorbed entity's state before the merge (canonical_name, type,
    #: aliases) -- enough to restore it as a standalone vertex again.
    snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    reverted: bool = False


class FoldJournal:
    def __init__(self) -> None:
        self._records: list[FoldRecord] = []
        self._next_seq = 0

    def append(
        self,
        kept: str,
        absorbed: str,
        score: float,
        trigger_episode: str,
        migrated_edge_ids: list[str],
        snapshot: dict[str, Any],
    ) -> FoldRecord:
        rec = FoldRecord(
            id=f"fold_{self._next_seq:04d}",
            kept=kept,
            absorbed=absorbed,
            score=score,
            trigger_episode=trigger_episode,
            migrated_edge_ids=list(migrated_edge_ids),
            snapshot=snapshot,
        )
        self._next_seq += 1
        self._records.append(rec)
        return rec

    def get(self, fold_id: str) -> FoldRecord:
        return next(r for r in self._records if r.id == fold_id)

    def all(self) -> list[FoldRecord]:
        return list(self._records)

    def folds_after(self, fold_id: str, touching: str) -> list[FoldRecord]:
        """Folds recorded after ``fold_id`` that also touch ``touching``
        (as either side) -- these depend on it and must be reverted first
        (spec 8.4's ``unfold``: "reverte também fusões posteriores que
        dependam dela")."""
        idx = next(i for i, r in enumerate(self._records) if r.id == fold_id)
        return [r for r in self._records[idx + 1 :] if touching in (r.kept, r.absorbed)]
