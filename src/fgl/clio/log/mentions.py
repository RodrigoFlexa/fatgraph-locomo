"""Raw mention counts (spec 3.2). Never consolidated, never folded --
counting questions ("how many times did she mention X") read this table
directly and must not go through the graph, where folding has already
collapsed repeated occurrences into one edge.
"""

from __future__ import annotations

from datetime import datetime

from fgl.clio.types import Mention


class MentionStore:
    def __init__(self) -> None:
        self._mentions: list[Mention] = []
        self._next_seq = 0

    def append(
        self,
        episode_id: str,
        surface: str,
        ts: datetime,
        entity_id: str | None = None,
    ) -> Mention:
        seq = self._next_seq
        self._next_seq += 1
        m = Mention(
            id=f"mnt_{seq:06d}",
            episode_id=episode_id,
            entity_id=entity_id,
            surface=surface,
            ts=ts,
        )
        self._mentions.append(m)
        return m

    def all(self) -> list[Mention]:
        return list(self._mentions)

    def count(
        self,
        entity_id: str | None = None,
        surface: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """Multiplicity, never cardinality. See :class:`Mention`'s docstring:
        this is what lets a count-shaped question bypass the folded graph.
        """
        n = 0
        for m in self._mentions:
            if entity_id is not None and m.entity_id != entity_id:
                continue
            if surface is not None and m.surface != surface:
                continue
            if start is not None and m.ts < start:
                continue
            if end is not None and m.ts >= end:
                continue
            n += 1
        return n
