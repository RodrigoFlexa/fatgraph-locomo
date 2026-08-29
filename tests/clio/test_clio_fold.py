"""Folding (spec section 8, milestone M6): entity resolution as a side
effect of merging compatible edges, not a separate pass.

The canonical case is spec's own "Rui" / "Rui Sampaio" (melanie.yaml's
E7). Checked empirically before trusting it (see git history / session
notes): spec's own C1 condition ("same origin and same label") does NOT
reach this pair at all -- "Rui" is the OBJECT of a ``managed_by`` edge and
"Rui Sampaio" is the SUBJECT of a ``hired`` edge, so they never share an
address for spec's literal same-address comparison to find. Candidate
generation here is type-blocked instead (see ``fold()``'s docstring for
the full reasoning); the identity-scoring formula itself still follows
spec 8.2's weights.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from tests.clio.helpers import HandFedMemory, load_melanie

from fgl.clio import access
from fgl.clio.catalog import load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.consolidate.fold import identity_score, unfold
from fgl.clio.consolidate.journal import FoldJournal


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(ClioConfig.default().catalog_path)


@pytest.fixture(scope="module")
def folded_memory(catalog) -> HandFedMemory:
    return load_melanie(catalog, with_fold=True)


# --------------------------------------------------------------------- #
# The canonical merge                                                      #
# --------------------------------------------------------------------- #
def test_rui_sampaio_folds_into_rui(folded_memory: HandFedMemory):
    assert len(folded_memory.fold_records) == 1
    rec = folded_memory.fold_records[0]
    assert rec.score >= 0.80
    assert rec.trigger_episode == "E7"


def test_absorbed_entity_resolves_to_kept(folded_memory: HandFedMemory):
    kept = folded_memory.entity("Rui")
    absorbed_id = folded_memory.fold_records[0].absorbed
    assert folded_memory.graph.resolve_entity(absorbed_id) == kept.id
    assert "Rui Sampaio" in kept.aliases


def test_edges_migrate_to_the_kept_vertex(folded_memory: HandFedMemory):
    kept = folded_memory.entity("Rui")
    melanie = folded_memory.entity("Melanie")
    labels = {
        e.label
        for e in folded_memory.graph.edges_incident(kept.id, live_only=False)
        if melanie.id in (e.src_id, e.dst_id)
    }
    assert labels == {"managed_by", "hired"}


def test_t8_fold_created_path_is_now_walkable(folded_memory: HandFedMemory):
    """Spec 17.3's T8: "who hired me was my boss?" -- a path that existed
    in no single episode, made walkable only because folding identified
    the two mentions as the same vertex."""
    state = access.anchor("Melanie", folded_memory.graph, tx_point=datetime(2024, 1, 21))
    state = access.follow(state, "hired_by", folded_memory.graph, folded_memory.catalog)
    state = access.follow(state, "manages", folded_memory.graph, folded_memory.catalog)
    state = access.filter_trails(state, folded_memory.graph, name="Melanie")
    assert len(state.trails) == 1


def test_no_fold_without_opting_in():
    """The M4 tests' own scope cut, checked directly: without a log and a
    journal, phase 1's exact-name matching is ALL the entity resolution
    that happens -- "Rui" and "Rui Sampaio" stay two separate vertices."""
    catalog = load_catalog(ClioConfig.default().catalog_path)
    mem = load_melanie(catalog, with_fold=False)
    names = {e.canonical_name for e in mem.graph.all_entities()}
    assert {"Rui", "Rui Sampaio"} <= names


# --------------------------------------------------------------------- #
# identity_score sanity: unrelated same-type entities score low            #
# --------------------------------------------------------------------- #
def test_unrelated_entities_score_far_below_threshold(folded_memory: HandFedMemory):
    melanie = folded_memory.entity("Melanie")
    bia = folded_memory.entity("Bia")
    score = identity_score(
        melanie, bia, folded_memory.graph, folded_memory.log, folded_memory.catalog
    )
    assert score < 0.5


# --------------------------------------------------------------------- #
# Explicit distinction blocks a fold that would otherwise clear tau_fold  #
# --------------------------------------------------------------------- #
def test_explicit_distinction_blocks_an_otherwise_qualifying_fold(catalog):
    """The distinguishing episode has to be logged BEFORE the fold-
    triggering one: ``explicit_distinction`` scans everything in the log
    at the moment fold runs, and fold runs as soon as the second name is
    introduced (see ``fold()``'s docstring on why -- the same call that
    creates "Rui Sampaio" is the one that scores it against "Rui")."""
    mem = HandFedMemory(catalog, with_fold=True)
    mem.ingest_episode(
        {
            "id": "D1",
            "date": "2023-01-01",
            "speaker": "Melanie",
            "text": "My boss Rui approved my trip",
            "propositions": [
                {
                    "subject": "new:Melanie",
                    "relation": "managed_by",
                    "object": "new:Rui",
                    "operation": "assert",
                    "evidence_kind": "literal",
                    "time_expression": None,
                    "span": "My boss Rui approved my trip",
                }
            ],
        }
    )
    mem.ingest_episode(
        {
            "id": "D2",
            "date": "2023-02-01",
            "speaker": "Melanie",
            "text": "That Rui Sampaio I mentioned is a different Rui, not my manager",
            "propositions": [],
        }
    )
    mem.ingest_episode(
        {
            "id": "D3",
            "date": "2023-03-01",
            "speaker": "Melanie",
            "text": "Rui Sampaio was the one who hired me",
            "propositions": [
                {
                    "subject": "new:Rui Sampaio",
                    "relation": "hired",
                    "object": "new:Melanie",
                    "operation": "assert",
                    "evidence_kind": "literal",
                    "time_expression": None,
                    "span": "Rui Sampaio was the one who hired me",
                }
            ],
        }
    )
    assert mem.fold_records == []
    names = {e.canonical_name for e in mem.graph.all_entities()}
    assert {"Rui", "Rui Sampaio"} <= names


def test_same_exact_name_distinction_blocks_phase_1_reuse_too(catalog):
    """Without this, two DIFFERENT people who happen to share one EXACT
    name would be unified at phase 1, unconditionally, before fold (which
    only ever compares DIFFERENT surface forms as a pair) gets a chance to
    tell them apart."""
    mem = HandFedMemory(catalog, with_fold=True)
    mem.ingest_episode(
        {
            "id": "D1",
            "date": "2023-01-01",
            "speaker": "Melanie",
            "text": "My boss Rui approved my trip",
            "propositions": [
                {
                    "subject": "new:Melanie",
                    "relation": "managed_by",
                    "object": "new:Rui",
                    "operation": "assert",
                    "evidence_kind": "literal",
                    "time_expression": None,
                    "span": "My boss Rui approved my trip",
                }
            ],
        }
    )
    mem.ingest_episode(
        {
            "id": "D2",
            "date": "2023-02-01",
            "speaker": "Melanie",
            "text": "I met a different Rui at the conference",
            "propositions": [
                {
                    "subject": "new:Melanie",
                    "relation": "friend_of",
                    "object": "new:Rui",
                    "operation": "assert",
                    "evidence_kind": "literal",
                    "time_expression": None,
                    "span": "I met a different Rui at the conference",
                }
            ],
        }
    )
    ruis = [e for e in mem.graph.all_entities() if e.canonical_name == "Rui"]
    assert len(ruis) == 2


# --------------------------------------------------------------------- #
# unfold reverses a merge structurally                                    #
# --------------------------------------------------------------------- #
def test_unfold_restores_the_absorbed_vertex(catalog):
    mem = HandFedMemory(catalog, with_fold=True)
    mem.ingest_episode(
        {
            "id": "D1",
            "date": "2023-01-01",
            "speaker": "Melanie",
            "text": "My boss Rui approved my trip",
            "propositions": [
                {
                    "subject": "new:Melanie",
                    "relation": "managed_by",
                    "object": "new:Rui",
                    "operation": "assert",
                    "evidence_kind": "literal",
                    "time_expression": None,
                    "span": "My boss Rui approved my trip",
                }
            ],
        }
    )
    mem.ingest_episode(
        {
            "id": "D2",
            "date": "2023-02-01",
            "speaker": "Melanie",
            "text": "Rui Sampaio was the one who hired me",
            "propositions": [
                {
                    "subject": "new:Rui Sampaio",
                    "relation": "hired",
                    "object": "new:Melanie",
                    "operation": "assert",
                    "evidence_kind": "literal",
                    "time_expression": None,
                    "span": "Rui Sampaio was the one who hired me",
                }
            ],
        }
    )
    assert len(mem.fold_records) == 1
    rec = mem.fold_records[0]
    kept_id, absorbed_id = rec.kept, rec.absorbed

    hired_edge = next(e for e in mem.graph.all_edges() if e.label == "hired")
    assert hired_edge.src_id == kept_id  # migrated onto the kept vertex

    unfold(rec.id, mem.journal, mem.graph)

    assert mem.graph.get_entity(absorbed_id).merged_into is None
    assert "Rui Sampaio" not in mem.graph.get_entity(kept_id).aliases
    hired_edge = next(e for e in mem.graph.all_edges() if e.label == "hired")
    assert hired_edge.src_id == absorbed_id  # moved back


def test_unfold_reverts_dependent_later_folds_too(catalog):
    """spec 8.4: reverting a fold also reverts any LATER fold that
    depended on it (touched the same kept vertex)."""
    journal = FoldJournal()
    rec1 = journal.append(
        kept="v1",
        absorbed="v2",
        score=0.9,
        trigger_episode="E1",
        migrated_edge_ids=[],
        snapshot={"canonical_name": "V2"},
    )
    rec2 = journal.append(
        kept="v1",
        absorbed="v3",
        score=0.9,
        trigger_episode="E2",
        migrated_edge_ids=[],
        snapshot={"canonical_name": "V3"},
    )
    dependents = journal.folds_after(rec1.id, touching="v1")
    assert dependents == [rec2]
