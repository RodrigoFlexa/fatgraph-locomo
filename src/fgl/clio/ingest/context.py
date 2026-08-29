"""Builds the extraction context (spec 6.2): the piece that makes
extraction work at all, because without linking to what the memory
already knows, an extractor produces floating strings that never connect
to anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from fgl.clio.catalog import Catalog, RelationSpec
from fgl.clio.index import EntityIndex
from fgl.clio.log.store import LogStore
from fgl.clio.types import Entity, Episode


@dataclass
class ExtractionContext:
    episode: Episode
    previous_turns: list[Episode]
    candidates: list[Entity]
    relations: list[RelationSpec]


def build_extraction_context(
    episode: Episode,
    log: LogStore,
    entity_index: EntityIndex,
    catalog: Catalog,
    coref_window: int = 3,
    max_candidates: int = 20,
) -> ExtractionContext:
    """``entity_index`` is searched with the turn's own raw text as the
    query (spec 6.2b's noun-phrase extraction step is skipped on purpose:
    :class:`~fgl.clio.index.EntityIndex` already does a literal-substring
    check against the query in addition to the dense one, which is the
    practical realisation of "is this entity mentioned here" without
    pulling in a POS tagger this package otherwise has no use for)."""
    previous = log.previous_turns(episode, coref_window)
    candidates = entity_index.search(episode.text, k=max_candidates)
    types_present = {c.type for c in candidates} | {
        "Person"
    }  # the speaker is always a Person
    relations = catalog.filter_by_types(types_present)
    return ExtractionContext(episode, previous, candidates, relations)
