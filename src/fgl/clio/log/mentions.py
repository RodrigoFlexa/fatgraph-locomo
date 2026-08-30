"""Raw mention counts (spec 3.2). Never consolidated, never folded --
counting questions ("how many times did she mention X") read this table
directly and must not go through the graph, where folding has already
collapsed repeated occurrences into one edge.
"""

from __future__ import annotations

from collections.abc import Iterable
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
        proposition_id: str | None = None,
    ) -> Mention:
        seq = self._next_seq
        self._next_seq += 1
        m = Mention(
            id=f"mnt_{seq:06d}",
            episode_id=episode_id,
            entity_id=entity_id,
            surface=surface,
            ts=ts,
            proposition_id=proposition_id,
        )
        self._mentions.append(m)
        return m

    def relink(self, propositions: Iterable) -> int:
        """Fills in ``entity_id`` for mentions whose proposition has since
        been resolved to a real vertex, and returns how many were linked.

        A mention is recorded at INGEST time, when its object is still a
        ``"new:X"`` reference -- so ``entity_id`` cannot be known yet, and
        before this existed it stayed ``None`` for every mention a real
        ingest ever produced. That silently reduced
        :func:`fgl.clio.access.movements.count` to exact surface-string
        matching, which misses every mention recorded under a surface form
        that later folded into a different canonical name.

        Called from consolidation, after phase 1 (which is what rewrites
        the proposition's ``object_id``), so it is idempotent: a mention
        already linked is skipped.
        """
        resolved = {p.id: p.object_id for p in propositions}
        linked = 0
        for m in self._mentions:
            if m.entity_id is not None or m.proposition_id is None:
                continue
            target = resolved.get(m.proposition_id)
            if target and not target.startswith("new:"):
                m.entity_id = target
                linked += 1
        return linked

    def unlink(self) -> None:
        """Drops every ``entity_id`` link, so a rebuild can redo them
        against the fresh graph. Only links this store INFERRED are
        dropped -- a mention recorded with an explicit ``entity_id`` at
        append time keeps it, because nothing about a rebuild makes it
        wrong. In practice the real pipeline records none that way, which
        is exactly why :meth:`relink` exists.
        """
        for m in self._mentions:
            if m.proposition_id is not None:
                m.entity_id = None

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
