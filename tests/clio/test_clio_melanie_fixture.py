"""Integration test for spec section 17.2's worked fixture.

Loads ``tests/fixtures/melanie.yaml`` (episodes E1-E6 -- E7's Rui/Rui
Sampaio fold is milestone M6 and is not in this fixture, see that file's
header) and feeds it through the real M1-M3 pieces
(:func:`resolve_time`, :func:`compute_confidence`) plus the M4
consolidation pipeline, one episode at a time -- the realistic case where
each turn's propositions are consolidated before the next episode's
CLOSE/RETRACT has to find them (see ``tests/clio/helpers.py``).

Then it checks the five things spec 17.2 says no other test checks, plus
the full expected graph state table (fold aside).
"""

from __future__ import annotations

from datetime import datetime

import pytest
from tests.clio.helpers import HandFedMemory, load_melanie

from fgl.clio.catalog import load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.types import Interval


@pytest.fixture(scope="module")
def memory() -> HandFedMemory:
    catalog = load_catalog(ClioConfig.default().catalog_path)
    return load_melanie(catalog)


# --------------------------------------------------------------------- #
# The five numbered assertions from spec 17.2 (E7's fold is milestone M6) #
# --------------------------------------------------------------------- #
def test_1_lives_in_recife_closes_in_may_not_june(memory: HandFedMemory):
    edge = memory.edge_to("Recife")
    assert edge.t_valid.end == datetime(2023, 5, 1)


def test_2_managed_by_bia_closes_t_valid_in_september_via_dependency(memory: HandFedMemory):
    edge = memory.edge_to("Bia")
    assert edge.t_valid.end == datetime(2023, 9, 5)


def test_3_managed_by_bia_closes_t_tx_at_e6_without_moving_t_valid(memory: HandFedMemory):
    edge = memory.edge_to("Bia")
    assert edge.t_tx.end == datetime(2023, 12, 1)
    assert edge.t_valid.end == datetime(2023, 9, 5)  # unchanged by the retraction


def test_4_practices_climbing_inherits_march_start_from_e2_implicature(
    memory: HandFedMemory,
):
    melanie = memory.entity("Melanie")
    climbing = memory.entity("climbing")
    edge = next(
        e
        for e in memory.graph.all_edges()
        if e.src_id == melanie.id and e.label == "practices" and e.dst_id == climbing.id
    )
    assert edge.t_valid.start == datetime(2023, 3, 2)  # E2, not the promotion date
    assert edge.t_tx.start == datetime(2023, 11, 11)  # E5, when it crossed tau_promote


# --------------------------------------------------------------------- #
# Full state table (spec 17.2), fold aside                               #
# --------------------------------------------------------------------- #
def test_full_graph_state(memory: HandFedMemory):
    vertex = memory.edge_to("Vertex")
    assert vertex.t_valid.start == datetime(2023, 1, 8)  # "this week", Sun-start week
    assert vertex.t_valid.end == datetime(2023, 9, 5)
    assert vertex.t_tx.start == datetime(2023, 1, 14)
    assert vertex.t_tx.end is None
    assert vertex.reinforcement == 2  # E1 assert + E3 reassert

    kaia = memory.edge_to("Kaia")
    assert kaia.t_valid == Interval(datetime(2023, 9, 5), None)

    salvador = memory.edge_to("Salvador")
    assert salvador.t_valid.start == datetime(2023, 5, 1)
    assert salvador.t_valid.end is None

    rui = memory.edge_to("Rui")
    assert rui.t_valid.start == datetime(2023, 9, 5)
    assert rui.t_valid.end is None

    bia_practices = next(
        e
        for e in memory.graph.all_edges()
        if e.dst_id == memory.entity("climbing").id and e.src_id == memory.entity("Bia").id
    )
    assert bia_practices.t_valid.start == datetime(2023, 3, 2)
    assert bia_practices.t_valid.end is None
    assert bia_practices.t_tx.start == datetime(2023, 3, 2)


def test_mention_count_preserves_multiplicity_the_graph_would_lose(memory: HandFedMemory):
    """The graph folds E2's and E5's climbing evidence into ONE edge
    (assertion 4) -- a count question must not read that edge."""
    assert memory.mentions.count(surface="climbing") == 2
