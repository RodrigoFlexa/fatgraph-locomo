"""Unit tests for consolidation phases 1-5, 7-8 (spec 17.1), fed by
hand-written propositions -- no LLM, no extraction, exactly milestone
M4's scope. The full multi-episode scenario lives in
``test_melanie_fixture.py``; these isolate one mechanism each.
"""

from __future__ import annotations

import itertools
from datetime import datetime

import pytest

from fgl.clio.catalog import Catalog, RelationSpec, load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.consolidate.promote import combine_confidence
from fgl.clio.graph.store import GraphStore
from fgl.clio.staging import StagingStore
from fgl.clio.types import EvidenceKind, Interval, Operation, Proposition

D = datetime
CATALOG = load_catalog(ClioConfig.default().catalog_path)
_ID_COUNTER = itertools.count()


def _prop(
    subject_id: str,
    relation: str,
    object_id: str,
    operation: Operation,
    t_valid: Interval | None,
    t_tx_start: datetime,
    episode_id: str,
    confidence: float = 0.90,
    polarity: bool = True,
    evidence_kind: EvidenceKind = EvidenceKind.LITERAL,
) -> Proposition:
    return Proposition(
        id=f"p{next(_ID_COUNTER)}",
        subject_id=subject_id,
        relation=relation,
        object_id=object_id,
        operation=operation,
        polarity=polarity,
        t_valid=t_valid,
        t_tx=Interval(t_tx_start, None),
        evidence_kind=evidence_kind,
        confidence=confidence,
        episode_id=episode_id,
    )


def _consolidate(
    catalog: Catalog, graph: GraphStore, staging: StagingStore, props: list[Proposition]
):
    from fgl.clio.consolidate.pipeline import consolidate

    staging.insert(props)
    config = ClioConfig.default()
    return consolidate(catalog, graph, staging, config)


# --------------------------------------------------------------------- #
# Phase 4: cardinality closes at the NEW fact's validity date            #
# --------------------------------------------------------------------- #
def test_cardinality_closes_at_validity_date_not_episode_date():
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    recife = graph.create_entity("Recife", "Place")
    salvador = graph.create_entity("Salvador", "Place")

    p1 = _prop(
        melanie.id,
        "lives_in",
        recife.id,
        Operation.ASSERT,
        Interval(D(2023, 1, 14), None),
        D(2023, 1, 14),
        "E1",
    )
    # reported in June, but valid from May -- the close must land in May.
    p2 = _prop(
        melanie.id,
        "lives_in",
        salvador.id,
        Operation.ASSERT,
        Interval(D(2023, 5, 1), None),
        D(2023, 6, 20),
        "E3",
    )

    _consolidate(CATALOG, graph, staging, [p1, p2])

    recife_edge = next(e for e in graph.all_edges() if e.dst_id == recife.id)
    assert recife_edge.t_valid.end == D(2023, 5, 1)
    assert recife_edge.t_valid.end != D(2023, 6, 20)


def test_cardinality_ignores_a_repeated_same_value():
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    vertex = graph.create_entity("Vertex", "Organization")

    p1 = _prop(
        melanie.id,
        "works_at",
        vertex.id,
        Operation.ASSERT,
        Interval(D(2023, 1, 14), None),
        D(2023, 1, 14),
        "E1",
    )
    p2 = _prop(
        melanie.id,
        "works_at",
        vertex.id,
        Operation.ASSERT,
        Interval(D(2023, 6, 20), None),
        D(2023, 6, 20),
        "E3",
    )

    _consolidate(CATALOG, graph, staging, [p1, p2])

    edges = [e for e in graph.all_edges() if e.dst_id == vertex.id]
    # same destination twice is not a conflict to resolve -- both live,
    # neither closed by the other.
    assert all(e.t_valid.end is None for e in edges)


# --------------------------------------------------------------------- #
# Phase 5: dependents cascade to depth 2                                #
# --------------------------------------------------------------------- #
def _chain_catalog() -> Catalog:
    """a -> depends on -> b -> depends on -> c, all functional/slow. A
    dedicated catalog because the melanie fixture only exercises depth 1
    (works_at -> managed_by)."""
    specs = {
        "a": RelationSpec(
            "a", ("Person", "Thing"), "functional", "slow", dependents=("b",)
        ),
        "b": RelationSpec(
            "b", ("Person", "Thing"), "functional", "slow", dependents=("c",)
        ),
        "c": RelationSpec("c", ("Person", "Thing"), "functional", "slow"),
    }
    return Catalog(["Person", "Thing"], specs)


def test_dependents_cascade_to_depth_two():
    catalog = _chain_catalog()
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    x = graph.create_entity("X", "Thing")
    y = graph.create_entity("Y", "Thing")
    z = graph.create_entity("Z", "Thing")

    start = D(2023, 1, 1)
    p_a = _prop(melanie.id, "a", x.id, Operation.ASSERT, Interval(start, None), start, "E1")
    p_b = _prop(melanie.id, "b", y.id, Operation.ASSERT, Interval(start, None), start, "E1")
    p_c = _prop(melanie.id, "c", z.id, Operation.ASSERT, Interval(start, None), start, "E1")
    _consolidate(catalog, graph, staging, [p_a, p_b, p_c])

    close_at = D(2023, 9, 5)
    p_close = _prop(
        melanie.id, "a", x.id, Operation.CLOSE, Interval(close_at, None), close_at, "E4"
    )
    _consolidate(catalog, graph, staging, [p_close])

    edge_a = next(e for e in graph.all_edges() if e.dst_id == x.id)
    edge_b = next(e for e in graph.all_edges() if e.dst_id == y.id)
    edge_c = next(e for e in graph.all_edges() if e.dst_id == z.id)
    assert edge_a.t_valid.end == close_at
    assert edge_b.t_valid.end == close_at  # depth 1
    assert edge_c.t_valid.end == close_at  # depth 2


def test_dependents_propagation_does_not_close_a_same_batch_replacement():
    """A new manager asserted in the SAME batch as the job change must not
    be born already closed -- it is the replacement, not the predecessor
    the closure is meant to end (regression: this is exactly what
    tests/clio/test_melanie_fixture.py's E4 exercises for real)."""
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    old_org = graph.create_entity("OldOrg", "Organization")
    new_org = graph.create_entity("NewOrg", "Organization")
    old_boss = graph.create_entity("OldBoss", "Person")
    new_boss = graph.create_entity("NewBoss", "Person")

    start = D(2023, 1, 1)
    _consolidate(
        CATALOG,
        graph,
        staging,
        [
            _prop(
                melanie.id,
                "works_at",
                old_org.id,
                Operation.ASSERT,
                Interval(start, None),
                start,
                "E1",
            ),
            _prop(
                melanie.id,
                "managed_by",
                old_boss.id,
                Operation.ASSERT,
                Interval(start, None),
                start,
                "E1",
            ),
        ],
    )

    switch = D(2023, 9, 5)
    _consolidate(
        CATALOG,
        graph,
        staging,
        [
            _prop(
                melanie.id,
                "works_at",
                old_org.id,
                Operation.CLOSE,
                Interval(switch, None),
                switch,
                "E4",
            ),
            _prop(
                melanie.id,
                "works_at",
                new_org.id,
                Operation.ASSERT,
                Interval(switch, None),
                switch,
                "E4",
            ),
            _prop(
                melanie.id,
                "managed_by",
                new_boss.id,
                Operation.ASSERT,
                Interval(switch, None),
                switch,
                "E4",
            ),
        ],
    )

    new_boss_edge = next(e for e in graph.all_edges() if e.dst_id == new_boss.id)
    old_boss_edge = next(e for e in graph.all_edges() if e.dst_id == old_boss.id)
    assert new_boss_edge.t_valid.end is None  # still open, not born pre-closed
    assert old_boss_edge.t_valid.end == switch


# --------------------------------------------------------------------- #
# CLOSE touches only t_valid; RETRACT touches only t_tx                 #
# --------------------------------------------------------------------- #
def test_close_narrows_t_valid_and_leaves_t_tx_alone():
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    vertex = graph.create_entity("Vertex", "Organization")
    p1 = _prop(
        melanie.id,
        "works_at",
        vertex.id,
        Operation.ASSERT,
        Interval(D(2023, 1, 1), None),
        D(2023, 1, 1),
        "E1",
    )
    _consolidate(CATALOG, graph, staging, [p1])
    edge = graph.all_edges()[0]
    original_t_tx = edge.t_tx

    close_at = D(2023, 9, 5)
    p2 = _prop(
        melanie.id,
        "works_at",
        vertex.id,
        Operation.CLOSE,
        Interval(close_at, None),
        close_at,
        "E4",
    )
    _consolidate(CATALOG, graph, staging, [p2])

    assert edge.t_valid.end == close_at
    assert edge.t_tx == original_t_tx  # untouched


def test_retract_narrows_t_tx_and_leaves_t_valid_alone():
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    bia = graph.create_entity("Bia", "Person")
    p1 = _prop(
        melanie.id,
        "managed_by",
        bia.id,
        Operation.ASSERT,
        Interval(D(2023, 3, 2), None),
        D(2023, 3, 2),
        "E2",
    )
    _consolidate(CATALOG, graph, staging, [p1])
    edge = graph.all_edges()[0]
    original_t_valid = edge.t_valid

    retract_at = D(2023, 12, 1)
    p2 = _prop(melanie.id, "managed_by", bia.id, Operation.RETRACT, None, retract_at, "E6")
    _consolidate(CATALOG, graph, staging, [p2])

    assert edge.t_valid == original_t_valid  # untouched
    assert edge.t_tx.end == retract_at


# --------------------------------------------------------------------- #
# Phase 7: noisy-OR combination                                          #
# --------------------------------------------------------------------- #
def test_combine_confidence_is_noisy_or_not_average():
    combined = combine_confidence([0.55, 0.55])
    assert combined == pytest.approx(1 - 0.45 * 0.45)
    assert combined > max(0.55, 0.55)  # two weak signals outrank either alone


def test_single_low_confidence_never_promotes_alone():
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    escalada = graph.create_entity("escalada", "Activity")
    p = _prop(
        melanie.id,
        "practices",
        escalada.id,
        Operation.ASSERT,
        Interval(D(2023, 3, 2), None),
        D(2023, 3, 2),
        "E2",
        confidence=0.55,
        evidence_kind=EvidenceKind.IMPLICATURE,
    )
    _consolidate(CATALOG, graph, staging, [p])
    assert graph.all_edges() == []
    assert p.status == "staged"


# --------------------------------------------------------------------- #
# Phase 1: exact-name entity reuse (fuzzy identity is milestone M6)      #
# --------------------------------------------------------------------- #
def test_same_new_name_resolves_to_the_same_entity():
    graph = GraphStore()
    staging = StagingStore()
    p1 = _prop(
        "new:Melanie",
        "works_at",
        "new:Vertex",
        Operation.ASSERT,
        Interval(D(2023, 1, 1), None),
        D(2023, 1, 1),
        "E1",
    )
    p2 = _prop(
        "new:melanie",
        "lives_in",
        "new:Recife",
        Operation.ASSERT,
        Interval(D(2023, 1, 1), None),
        D(2023, 1, 1),
        "E1",
    )
    _consolidate(CATALOG, graph, staging, [p1, p2])

    persons = [e for e in graph.all_entities() if e.type == "Person"]
    assert len(persons) == 1  # "Melanie" and "melanie" are the same exact match


def test_different_new_names_create_different_entities():
    graph = GraphStore()
    staging = StagingStore()
    p1 = _prop(
        "new:Melanie",
        "friend_of",
        "new:Bob",
        Operation.ASSERT,
        Interval(D(2023, 1, 1), None),
        D(2023, 1, 1),
        "E1",
    )
    _consolidate(CATALOG, graph, staging, [p1])

    persons = {e.canonical_name for e in graph.all_entities() if e.type == "Person"}
    assert persons == {"Melanie", "Bob"}


# --------------------------------------------------------------------- #
# Phase 8: opposite polarity, overlapping window -> conflict, not fix   #
# --------------------------------------------------------------------- #
def test_opposite_polarity_same_window_flags_conflict():
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    jazz = graph.create_entity("jazz", "Topic")
    window = Interval(D(2023, 1, 1), D(2023, 6, 1))
    p_likes = _prop(
        melanie.id,
        "likes",
        jazz.id,
        Operation.ASSERT,
        window,
        D(2023, 1, 1),
        "E1",
        polarity=True,
    )
    p_dislikes = _prop(
        melanie.id,
        "likes",
        jazz.id,
        Operation.ASSERT,
        window,
        D(2023, 2, 1),
        "E2",
        polarity=False,
    )
    _consolidate(CATALOG, graph, staging, [p_likes, p_dislikes])

    edges = [e for e in graph.all_edges() if e.dst_id == jazz.id]
    assert edges  # both propositions cleared tau_promote and wrote an edge
    assert all(e.conflict_flag for e in edges)


def test_agreeing_polarity_does_not_flag_a_conflict():
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    jazz = graph.create_entity("jazz", "Topic")
    window = Interval(D(2023, 1, 1), D(2023, 6, 1))
    p1 = _prop(melanie.id, "likes", jazz.id, Operation.ASSERT, window, D(2023, 1, 1), "E1")
    p2 = _prop(melanie.id, "likes", jazz.id, Operation.ASSERT, window, D(2023, 2, 1), "E2")
    _consolidate(CATALOG, graph, staging, [p1, p2])

    edges = [e for e in graph.all_edges() if e.dst_id == jazz.id]
    assert not any(e.conflict_flag for e in edges)


# --------------------------------------------------------------------- #
# Reprocessing the same staging backlog is idempotent                    #
# --------------------------------------------------------------------- #
def test_reconsolidating_does_not_duplicate_already_promoted_edges():
    graph = GraphStore()
    staging = StagingStore()
    melanie = graph.create_entity("Melanie", "Person")
    vertex = graph.create_entity("Vertex", "Organization")
    p1 = _prop(
        melanie.id,
        "works_at",
        vertex.id,
        Operation.ASSERT,
        Interval(D(2023, 1, 1), None),
        D(2023, 1, 1),
        "E1",
    )
    staging.insert([p1])
    config = ClioConfig.default()

    from fgl.clio.consolidate.pipeline import consolidate

    consolidate(CATALOG, graph, staging, config)
    assert len(graph.all_edges()) == 1
    assert p1.status == "promoted"

    consolidate(CATALOG, graph, staging, config)  # nothing new inserted
    assert len(graph.all_edges()) == 1  # not duplicated
