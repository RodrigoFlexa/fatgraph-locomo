"""Append-oriented store for CLIO3's entity/event hypergraph."""

from __future__ import annotations

from collections import defaultdict, deque

from fgl.clio3.model import MemoryRecord, OpenEntity, RecordLink


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


class EventGraphStore:
    """Entities are vertices; records are event/state hyperedges with roles.

    A record is never overwritten. Lifecycle operations append a new record
    and only change the prior record's status, retaining its evidence and
    transaction history for audit and replay.
    """

    def __init__(self) -> None:
        self._entities: dict[str, OpenEntity] = {}
        self._records: dict[str, MemoryRecord] = {}
        self._links: list[RecordLink] = []
        self._entity_records: dict[str, set[str]] = defaultdict(set)
        self._record_links: dict[str, list[RecordLink]] = defaultdict(list)
        self._next_entity_seq = 0
        self._next_record_seq = 0

    def entities(self) -> list[OpenEntity]:
        return list(self._entities.values())

    def records(self, active_only: bool = False) -> list[MemoryRecord]:
        rows = list(self._records.values())
        if active_only:
            rows = [row for row in rows if row.status == "active"]
        return rows

    def links(self) -> list[RecordLink]:
        return list(self._links)

    def get_entity(self, entity_id: str) -> OpenEntity:
        entity = self._entities[entity_id]
        while entity.merged_into:
            entity = self._entities[entity.merged_into]
        return entity

    def get_record(self, record_id: str) -> MemoryRecord:
        return self._records[record_id]

    def find_entity(self, name: str, type: str | None = None) -> OpenEntity | None:
        needle = _norm(name)
        for entity in self._entities.values():
            if entity.merged_into or type and _norm(entity.type) != _norm(type):
                continue
            if needle in {_norm(entity.name), *(_norm(a) for a in entity.aliases)}:
                return entity
        return None

    def add_entity(
        self, name: str, type: str, episode_id: str, aliases: list[str] | None = None
    ) -> OpenEntity:
        normalized_type = (type or "entity").strip().casefold()
        existing = self.find_entity(name, normalized_type)
        if existing is None and normalized_type == "entity":
            existing = self.find_entity(name)
        if existing is not None:
            if episode_id and episode_id not in existing.episode_ids:
                existing.episode_ids.append(episode_id)
            for alias in aliases or ():
                if _norm(alias) != _norm(existing.name) and alias not in existing.aliases:
                    existing.aliases.append(alias)
            return existing
        entity = OpenEntity(
            id=f"c3e_{self._next_entity_seq:06d}",
            name=name.strip(),
            type=normalized_type,
            aliases=list(dict.fromkeys(aliases or ())),
            episode_ids=[episode_id] if episode_id else [],
        )
        self._next_entity_seq += 1
        self._entities[entity.id] = entity
        return entity

    def merge_entity(self, alias_id: str, canonical_id: str) -> None:
        if alias_id == canonical_id:
            return
        alias = self.get_entity(alias_id)
        canonical = self.get_entity(canonical_id)
        if alias.id == canonical.id:
            return
        if alias.name not in canonical.aliases:
            canonical.aliases.append(alias.name)
        canonical.aliases.extend(a for a in alias.aliases if a not in canonical.aliases)
        canonical.episode_ids.extend(
            e for e in alias.episode_ids if e not in canonical.episode_ids
        )
        alias.merged_into = canonical.id
        for record_id in list(self._entity_records.get(alias.id, ())):
            record = self._records[record_id]
            record.participants = [
                type(p)(canonical.id if p.entity_id == alias.id else p.entity_id, p.role)
                for p in record.participants
            ]
            self._entity_records[canonical.id].add(record_id)

    def add_record(self, record: MemoryRecord) -> MemoryRecord:
        if not record.id:
            record.id = f"c3r_{self._next_record_seq:07d}"
        self._next_record_seq = max(
            self._next_record_seq,
            int(record.id.rsplit("_", 1)[-1]) + 1 if record.id.startswith("c3r_") else 0,
        )
        self._records[record.id] = record
        for participant in record.participants:
            self._entity_records[participant.entity_id].add(record.id)
        for prior_id in record.supersedes:
            prior = self._records.get(prior_id)
            if prior is None:
                continue
            if record.operation == "retract":
                prior.status = "retracted"
            elif record.operation == "close":
                prior.status = "closed"
            else:
                prior.status = "superseded"
        return record

    def new_record_id(self) -> str:
        value = f"c3r_{self._next_record_seq:07d}"
        self._next_record_seq += 1
        return value

    def add_link(self, link: RecordLink) -> None:
        if link.source_id not in self._records or link.target_id not in self._records:
            return
        key = (link.source_id, _norm(link.relation), link.target_id)
        if any((x.source_id, _norm(x.relation), x.target_id) == key for x in self._links):
            return
        self._links.append(link)
        self._record_links[link.source_id].append(link)
        self._record_links[link.target_id].append(link)

    def records_for_entity(self, entity_id: str) -> list[MemoryRecord]:
        entity = self.get_entity(entity_id)
        return [self._records[rid] for rid in self._entity_records.get(entity.id, ())]

    def neighborhood(self, record_ids: list[str], max_hops: int) -> dict[str, int]:
        """Return record ids and shortest event-graph distance from the seeds."""

        distances = {record_id: 0 for record_id in record_ids if record_id in self._records}
        queue = deque(distances)
        while queue:
            record_id = queue.popleft()
            distance = distances[record_id]
            if distance >= max_hops:
                continue
            record = self._records[record_id]
            neighbors: set[str] = set()
            for participant in record.participants:
                neighbors.update(self._entity_records.get(participant.entity_id, ()))
            for link in self._record_links.get(record_id, ()):
                neighbors.add(
                    link.target_id if link.source_id == record_id else link.source_id
                )
            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
        return distances

    def edge_count(self) -> int:
        return sum(len(record.participants) for record in self._records.values()) + len(
            self._links
        )

    def consolidate(self) -> None:
        """The write path is already append-safe; retained as a session hook."""


__all__ = ["EventGraphStore"]
