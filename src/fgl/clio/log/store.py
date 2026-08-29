"""The episodic log: append-only, total order. This is P1's "the log is
truth" (spec section 1) -- nothing here is ever edited or removed, and
every other store is in principle rebuildable from it (spec 12.3).
"""

from __future__ import annotations

from datetime import datetime

from fgl.clio.types import Episode


class LogStore:
    def __init__(self) -> None:
        self._episodes: list[Episode] = []
        self._by_id: dict[str, Episode] = {}
        self._next_seq = 0

    def append(
        self,
        session_id: str,
        speaker: str,
        text: str,
        ts_ingest: datetime,
        meta: dict | None = None,
        episode_id: str | None = None,
    ) -> Episode:
        seq = self._next_seq
        self._next_seq += 1
        ep = Episode(
            id=episode_id or f"ep_{seq:06d}",
            session_id=session_id,
            speaker=speaker,
            text=text,
            ts_ingest=ts_ingest,
            seq=seq,
            meta=dict(meta or {}),
        )
        self._episodes.append(ep)
        self._by_id[ep.id] = ep
        return ep

    def get(self, episode_id: str) -> Episode:
        return self._by_id[episode_id]

    def all(self) -> list[Episode]:
        """All episodes, in log order (append order == total order)."""
        return list(self._episodes)

    def by_session(self, session_id: str) -> list[Episode]:
        return [e for e in self._episodes if e.session_id == session_id]

    def previous_turns(self, episode: Episode, n: int) -> list[Episode]:
        """Up to ``n`` episodes immediately before ``episode`` in the same
        session, oldest first. Used only to resolve coreference at
        extraction time (spec 6.2a) -- never to extract new facts from.
        """
        same_session = self.by_session(episode.session_id)
        idx = next(i for i, e in enumerate(same_session) if e.id == episode.id)
        return same_session[max(0, idx - n) : idx]
