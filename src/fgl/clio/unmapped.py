"""The unmapped queue (spec 4.4): what the extractor could not map to a
Sigma relation. Never written to the graph -- growing the catalog is a
human, versioned action (``clio.catalog.mine_unmapped`` + a review of this
queue), never the extractor widening its own vocabulary.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class UnmappedEntry:
    id: str
    suggested_relation: str | None
    subject_ref: str
    object_ref: str
    span: str
    episode_id: str


class UnmappedQueue:
    def __init__(self) -> None:
        self._entries: list[UnmappedEntry] = []
        self._next_seq = 0

    def append(
        self,
        suggested_relation: str | None,
        subject_ref: str,
        object_ref: str,
        span: str,
        episode_id: str,
    ) -> UnmappedEntry:
        entry = UnmappedEntry(
            id=f"unm_{self._next_seq:05d}",
            suggested_relation=suggested_relation,
            subject_ref=subject_ref,
            object_ref=object_ref,
            span=span,
            episode_id=episode_id,
        )
        self._next_seq += 1
        self._entries.append(entry)
        return entry

    def all(self) -> list[UnmappedEntry]:
        return list(self._entries)

    def grouped_by_suggestion(self) -> dict[str, list[UnmappedEntry]]:
        """The report a human reviews before promoting a candidate to the
        catalog (spec 4.4): every suggestion, with every occurrence, so
        the decision is "is this a real recurring gap" rather than a
        guess from one example."""
        groups: dict[str, list[UnmappedEntry]] = defaultdict(list)
        for entry in self._entries:
            groups[entry.suggested_relation or "(unnamed)"].append(entry)
        return dict(groups)
