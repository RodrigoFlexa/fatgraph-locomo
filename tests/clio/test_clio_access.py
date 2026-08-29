"""The access algebra (spec section 9, milestone M7), run without any
agent or LLM in the loop -- movements are called directly, exactly as
spec 17.3's traces do. Built on the same consolidated melanie graph as
``test_clio_melanie_fixture.py`` (see ``tests/clio/helpers.py``); T8 (the
fold-created path) is not here because it needs the Rui/Rui Sampaio fold,
which is milestone M6 -- see ``test_clio_fold.py``.

Two of spec's own traces needed a real fix, not just a translation:

* T3/T4 need ``lives_in`` to be invertible to walk back from a place to
  its resident -- the catalog now declares it so (see
  ``personal_dialogue.yaml``'s note).
* T5's death-cause label is reproduced by ``restrict`` directly rather
  than through a ``follow`` + ``filter`` chain: ``filter_trails`` reports
  a plain ``"filtered_out"`` when a named vertex is not among the
  survivors, not a *reason* a temporally-incoherent candidate was never a
  contender in the first place -- carrying that reason across a filter
  step is a real diagnostic feature this implementation does not attempt,
  a deliberate scope cut over a marginal-value mechanism, not a bug.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from tests.clio.helpers import HandFedMemory, load_melanie

from fgl.clio import access
from fgl.clio.catalog import load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.graph.queries import UnknownLabel
from fgl.clio.types import Interval

TODAY = datetime(2024, 1, 20)  # after every episode in the fixture


@pytest.fixture(scope="module")
def memory() -> HandFedMemory:
    catalog = load_catalog(ClioConfig.default().catalog_path)
    return load_melanie(catalog)


def _anchor_melanie(memory: HandFedMemory, tx_point: datetime = TODAY):
    return access.anchor("Melanie", memory.graph, tx_point=tx_point)


# --------------------------------------------------------------------- #
# T1. Current fact                                                        #
# --------------------------------------------------------------------- #
def test_t1_current_employer(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    state = access.restrict(state, "valid", Interval(TODAY, TODAY + timedelta(days=1)))
    state = access.follow(state, "works_at", memory.graph, memory.catalog)
    assert len(state.trails) == 1
    assert state.trails[0].vertex_id == memory.entity("Kaia").id


# --------------------------------------------------------------------- #
# T2. Past fact -- same composition, different argument (spec's point:    #
#     there is no separate "temporal engine")                             #
# --------------------------------------------------------------------- #
def test_t2_past_employer(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    state = access.restrict(
        state, "valid", Interval(datetime(2023, 2, 1), datetime(2023, 2, 2))
    )
    state = access.follow(state, "works_at", memory.graph, memory.catalog)
    assert len(state.trails) == 1
    assert state.trails[0].vertex_id == memory.entity("Vertex").id


# --------------------------------------------------------------------- #
# T3. Multi-hop with retraction: the naive answer would be wrong          #
# --------------------------------------------------------------------- #
def test_t3_boss_while_in_recife_is_dead_because_retracted(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    state = access.follow(state, "lives_in", memory.graph, memory.catalog)
    state = access.filter_trails(state, memory.graph, name="Recife")
    assert len(state.trails) == 1
    state = access.follow(state, "resided_by", memory.graph, memory.catalog)
    assert state.trails[0].vertex_id == memory.entity("Melanie").id

    state = access.follow(state, "managed_by", memory.graph, memory.catalog)
    assert state.trails == []
    assert state.death_cause == "all_edges_retracted"


# --------------------------------------------------------------------- #
# T4. Historical belief: change the transaction point of view, not the    #
#     the world -- and get Bia back, with the window as the INTERSECTION #
#     of both edges' intervals (invariant I1), not either one alone.      #
# --------------------------------------------------------------------- #
def test_t4_historical_belief_recovers_bia_with_intersected_window(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    state = access.follow(state, "lives_in", memory.graph, memory.catalog)
    state = access.filter_trails(state, memory.graph, name="Recife")
    state = access.follow(state, "resided_by", memory.graph, memory.catalog)

    state = access.restrict(state, "tx", Interval(datetime(2023, 11, 1), None))
    state = access.follow(state, "managed_by", memory.graph, memory.catalog)

    assert len(state.trails) == 1
    assert state.trails[0].vertex_id == memory.entity("Bia").id
    assert state.trails[0].window == Interval(datetime(2023, 3, 2), datetime(2023, 5, 1))


# --------------------------------------------------------------------- #
# T5. False premise: the intersection that empties is diagnosable         #
# --------------------------------------------------------------------- #
def test_t5_impossible_premise_is_an_empty_window_not_a_guess(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    state = access.follow(state, "lives_in", memory.graph, memory.catalog)
    state = access.filter_trails(state, memory.graph, name="Recife")
    state = access.follow(state, "resided_by", memory.graph, memory.catalog)
    assert state.trails[0].window == Interval(datetime(2023, 1, 14), datetime(2023, 5, 1))

    kaia_window = memory.edge_to("Kaia").t_valid
    state = access.restrict(state, "valid", kaia_window)

    assert state.trails == []
    assert state.death_cause == "empty_temporal_window"


# --------------------------------------------------------------------- #
# T6. Count: multiplicity the folded graph has already lost               #
# --------------------------------------------------------------------- #
def test_t6_count_reads_the_log_not_the_graph(memory: HandFedMemory):
    n = access.count(memory.mentions, memory.graph, surface="climbing")
    assert n == 2


# --------------------------------------------------------------------- #
# T7. Evolution: history, no functional collapse                          #
# --------------------------------------------------------------------- #
def test_t7_employment_history_in_order_uncollapsed(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    entries = access.history(state, "works_at", memory.graph, memory.catalog)
    names = [memory.graph.get_entity(e.vertex_id).canonical_name for e in entries]
    assert names == ["Vertex", "Kaia"]
    assert entries[0].t_valid == Interval(datetime(2023, 1, 8), datetime(2023, 9, 5))
    assert entries[1].t_valid == Interval(datetime(2023, 9, 5), None)


# --------------------------------------------------------------------- #
# evidence(): materialises real episode text, not the proposition         #
# --------------------------------------------------------------------- #
def test_evidence_returns_the_source_episode_text(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    state = access.follow(state, "works_at", memory.graph, memory.catalog)
    state = access.filter_trails(state, memory.graph, name="Kaia")
    episodes = access.evidence(state, memory.staging, memory.log)
    assert len(episodes) == 1
    assert "Kaia" in episodes[0].text


# --------------------------------------------------------------------- #
# Movement-level edge cases                                                #
# --------------------------------------------------------------------- #
def test_follow_unknown_label_raises(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    with pytest.raises(UnknownLabel):
        access.follow(state, "not_a_real_relation", memory.graph, memory.catalog)


def test_follow_no_edge_at_all_reports_that_cause(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    state = access.follow(state, "owns", memory.graph, memory.catalog)  # never asserted
    assert state.trails == []
    assert state.death_cause == "no_edge_with_label"


def test_anchor_returns_no_trails_for_an_unknown_name(memory: HandFedMemory):
    state = access.anchor("someone who was never mentioned", memory.graph, tx_point=TODAY)
    assert state.trails == []


def test_available_labels_reflects_real_out_edges(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    labels = access.available_labels(state, memory.graph, memory.catalog)
    assert "works_at" in labels
    assert "lives_in" in labels
    assert "managed_by" in labels


def test_expand_finds_a_neighbour_within_two_hops(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    expanded = access.expand(state, memory.graph, k=2)
    reached = {t.vertex_id for t in expanded.trails}
    assert memory.entity("Kaia").id in reached or memory.entity("Bia").id in reached


def test_expand_never_reintroduces_the_seed_itself(memory: HandFedMemory):
    state = _anchor_melanie(memory)
    expanded = access.expand(state, memory.graph, k=2)
    assert memory.entity("Melanie").id not in {t.vertex_id for t in expanded.trails}
