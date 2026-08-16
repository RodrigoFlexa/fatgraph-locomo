"""Unit tests for the fatgraph core: alpha, sigma, phi, faces, Euler, curation."""

from __future__ import annotations

import pytest
from conftest import build_torus, build_triangle

from fgl.core import (
    FatGraph,
    NotABigonError,
    TopologyViolation,
    face_id,
)


# --------------------------------------------------------------------------- #
# alpha / sigma / phi                                                          #
# --------------------------------------------------------------------------- #


def test_alpha_is_a_fixed_point_free_involution():
    g, _ = build_triangle()
    assert len(g.alpha) == len(g.H) == 6
    for h, a in g.alpha.items():
        assert a != h, "alpha must have no fixed point"
        assert g.alpha[a] == h, "alpha^2 must be the identity"


def test_alpha_pairs_share_one_edge_id():
    g, _ = build_triangle()
    for h, a in g.alpha.items():
        assert g.H[h].edge_id == g.H[a].edge_id
    assert len(g.edges()) == 3


def test_sigma_is_a_consistent_cyclic_order():
    g, _ = build_triangle()
    for vid, order in g.sigma.items():
        assert len(set(order)) == len(order)
        for h in order:
            assert g.H[h].vertex_id == vid
        # walking sigma from any half-edge returns to it after deg(v) steps
        for h in order:
            cur = h
            for _ in range(len(order)):
                cur = g.sigma_next(cur)
            assert cur == h
        # sigma_prev inverts sigma_next
        for h in order:
            assert g.sigma_prev(g.sigma_next(h)) == h


def test_phi_is_sigma_after_alpha():
    g, _ = build_triangle()
    for h in g.H:
        assert g.phi(h) == g.sigma_next(g.alpha[h])


def test_phi_is_a_permutation():
    g, _ = build_triangle()
    images = [g.phi(h) for h in g.H]
    assert sorted(images) == sorted(g.H)


# --------------------------------------------------------------------------- #
# Faces                                                                        #
# --------------------------------------------------------------------------- #


def test_triangle_has_two_faces_of_length_three():
    g, _ = build_triangle()
    faces = g.faces()
    assert len(faces) == 2
    assert sorted(f.length for f in faces) == [3, 3]
    assert sum(f.length for f in faces) == len(g.H)


def test_faces_partition_the_half_edges():
    g, _ = build_triangle()
    covered = [h for f in g.faces() for h in f.half_edges]
    assert sorted(covered) == sorted(g.H)


def test_face_id_is_rotation_invariant_and_order_sensitive():
    assert face_id(["e1", "e2", "e3"]) == face_id(["e2", "e3", "e1"])
    assert face_id(["e1", "e2", "e3"]) != face_id(["e1", "e3", "e2"])
    # a *set* hash would collide here; the sequence hash must not
    assert face_id(["e1", "e2", "e1", "e3"]) != face_id(["e1", "e1", "e2", "e3"])


def test_face_of_agrees_with_faces():
    g, _ = build_triangle()
    by_id = {f.id: f for f in g.faces()}
    for h in g.H:
        f = g.face_of(h)
        assert f.id in by_id
        assert set(f.half_edges) == set(by_id[f.id].half_edges)


def test_leaf_face_traverses_the_same_edge_twice():
    g = FatGraph()
    a, b = g.add_vertex("a"), g.add_vertex("b")
    g.add_edge(a, b, {"text": "only fact"})
    (face,) = g.faces()
    assert face.length == 2
    assert face.is_leaf_face
    assert face.distinct_edges == ("e1",)


# --------------------------------------------------------------------------- #
# Euler                                                                        #
# --------------------------------------------------------------------------- #


def test_triangle_euler_is_a_sphere():
    g, _ = build_triangle()
    assert g.euler().as_tuple() == (3, 3, 2, 0)
    assert g.euler().C == 1
    assert g.euler().chi == 2


def test_torus_has_genus_one():
    g = build_torus()
    e = g.euler()
    assert (e.V, e.E, e.F, e.genus) == (1, 2, 1, 1)
    assert e.chi == 0


def test_disconnected_graph_needs_the_per_component_formula():
    g = FatGraph()
    p, q, r, s = (g.add_vertex(n) for n in "pqrs")
    g.add_edge(p, q, {"text": "1"})
    g.add_edge(r, s, {"text": "2"})
    e = g.euler()
    assert e.C == 2
    assert e.genus == 0, "each component is a sphere"
    # the literal 2-2g formula of the spec would report a negative genus
    assert e.genus_connected_formula == -1.0
    assert e.chi == 2 * e.C - 2 * e.genus


def test_isolated_vertex_contributes_a_trivial_face():
    g = FatGraph()
    g.add_vertex("lonely")
    e = g.euler()
    assert (e.V, e.E, e.F, e.C, e.genus) == (1, 0, 1, 1, 0)


def test_euler_holds_after_every_insertion():
    g = FatGraph()
    names = [g.add_vertex(f"v{i}") for i in range(6)]
    for i in range(5):
        g.add_edge(names[i], names[i + 1], {"text": f"fact {i}"})
        e = g.euler()
        assert e.chi == 2 * e.C - 2 * e.genus
        g.check_invariants()


# --------------------------------------------------------------------------- #
# collapse_bigon                                                               #
# --------------------------------------------------------------------------- #


def _parallel_pair() -> FatGraph:
    g = FatGraph()
    u, v = g.add_vertex("Caroline"), g.add_vertex("photography")
    g.add_edge(u, v, {"text": "Caroline took up photography.",
                      "turn_ids": ["D1:4"], "timestamp": "2023-05-08"})
    g.add_edge(u, v, {"text": "Caroline started photography.",
                      "turn_ids": ["D3:2"], "timestamp": "2023-06-10"})
    return g


def test_bigon_collapse_preserves_genus_and_merges_provenance():
    g = _parallel_pair()
    before = g.euler()
    bigons = [f for f in g.faces() if f.length == 2 and not f.is_leaf_face]
    assert len(bigons) == 2, "a parallel pair bounds two bigons"

    kept = g.collapse_bigon(bigons[0].id, merged_text="Caroline took up photography.")
    after = g.euler()

    assert after.genus == before.genus == 0
    assert (after.V, after.E, after.F) == (before.V, before.E - 1, before.F - 1)
    assert g.get_edge_attr(kept, "turn_ids") == ["D1:4", "D3:2"]
    assert g.get_edge_attr(kept, "provenance") == ["e1", "e2"]
    assert g.get_edge_attr(kept, "timestamp") == "2023-05-08"
    g.check_invariants()


def test_collapse_refuses_a_leaf_face():
    g = FatGraph()
    a, b = g.add_vertex("a"), g.add_vertex("b")
    g.add_edge(a, b, {"text": "solo"})
    (face,) = g.faces()
    with pytest.raises(NotABigonError):
        g.collapse_bigon(face.id)


def test_collapse_refuses_a_longer_face():
    g, _ = build_triangle()
    face = g.faces()[0]
    with pytest.raises(NotABigonError):
        g.collapse_bigon(face.id)


def test_collapse_is_idempotent_on_the_graph_invariants():
    g = _parallel_pair()
    bigons = [f for f in g.faces() if f.length == 2 and not f.is_leaf_face]
    g.collapse_bigon(bigons[0].id)
    g.check_invariants()
    assert [f.is_leaf_face for f in g.faces()] == [True]


# --------------------------------------------------------------------------- #
# walk_face                                                                    #
# --------------------------------------------------------------------------- #


def test_walk_face_returns_the_face_in_order():
    g, ctx = build_triangle()
    h = g.sigma[ctx["vertices"]["Caroline"]][0]
    walk = g.walk_face(h)
    assert [he.edge_id for he in walk] == list(g.face_of(h).distinct_edges)
    assert len(walk) == 3


def test_walk_face_respects_the_token_budget():
    g, ctx = build_triangle()
    h = g.sigma[ctx["vertices"]["Caroline"]][0]
    assert len(g.walk_face(h, budget_tokens=1)) == 1
    assert len(g.walk_face(h, budget_tokens=100000)) == 3


def test_walk_face_deduplicates_a_repeated_edge():
    g = FatGraph()
    a, b = g.add_vertex("a"), g.add_vertex("b")
    g.add_edge(a, b, {"text": "one memory"})
    h = g.sigma[a][0]
    assert len(g.walk_face(h)) == 1


# --------------------------------------------------------------------------- #
# whitehead_flip (phase 2)                                                     #
# --------------------------------------------------------------------------- #


def _theta_with_degree_three() -> tuple[FatGraph, str]:
    """Two degree-3 vertices joined by three parallel edges (theta graph)."""
    g = FatGraph()
    u, v = g.add_vertex("u"), g.add_vertex("v")
    e = [g.add_edge(u, v, {"text": f"fact {i}"}) for i in range(3)]
    return g, e[1]


def test_whitehead_flip_preserves_the_surface():
    g, edge = _theta_with_degree_three()
    before = g.euler()
    g.whitehead_flip(edge)
    after = g.euler()
    assert (after.V, after.E, after.F, after.genus) == (
        before.V, before.E, before.F, before.genus,
    )
    g.check_invariants()


def test_whitehead_flip_refuses_a_loop():
    g = FatGraph()
    v = g.add_vertex("v")
    e = g.add_edge(v, v, {"text": "loop"})
    with pytest.raises(Exception):
        g.whitehead_flip(e)


# --------------------------------------------------------------------------- #
# invariants and persistence                                                   #
# --------------------------------------------------------------------------- #


def test_check_invariants_detects_a_broken_alpha():
    g, _ = build_triangle()
    h = next(iter(g.H))
    g.alpha[h] = h
    with pytest.raises(Exception):
        g.check_invariants()


def test_edge_level_attributes_stay_in_sync():
    g, ctx = build_triangle()
    e = ctx["edges"][0]
    g.set_edge_attr(e, state="incongruente", shadowed=True)
    h1, h2 = g.edge_half_edges(e)
    assert g.H[h1].state == g.H[h2].state == "incongruente"
    assert g.H[h1].shadowed and g.H[h2].shadowed
    g.check_invariants()


def test_set_edge_attr_rejects_a_non_edge_attribute():
    g, ctx = build_triangle()
    with pytest.raises(Exception):
        g.set_edge_attr(ctx["edges"][0], text="not an edge-level attribute")


def test_serialize_round_trip(tmp_path):
    import numpy as np

    g, ctx = build_triangle()
    for h in g.H.values():
        h.embedding = np.arange(4, dtype=np.float32)
    path = tmp_path / "graph"
    g.save(path)
    back = FatGraph.load(path)

    assert back.euler().as_tuple() == g.euler().as_tuple()
    assert back.sigma == g.sigma
    assert back.alpha == g.alpha
    assert {f.id for f in back.faces()} == {f.id for f in g.faces()}
    assert all(back.H[h].embedding is not None for h in back.H)
    back.check_invariants()
    # counters survive, so new ids do not collide
    new_v = back.add_vertex("new")
    assert new_v not in ctx["vertices"].values()


def test_faces_is_linear_in_the_number_of_half_edges():
    """Regression guard: an O(|H|^2) faces() makes ingestion unusable."""
    import random
    import time

    def build(n_edges: int) -> FatGraph:
        random.seed(0)
        g = FatGraph()
        vs = [g.add_vertex(f"v{i}") for i in range(40)]
        while len(g.edges()) < n_edges:
            a, b = random.choice(vs), random.choice(vs)
            if a != b:
                g.add_edge(a, b, {"text": "x"})
        return g

    def timed(g: FatGraph) -> float:
        best = float("inf")
        for _ in range(5):
            g._components_cache = None
            t = time.perf_counter()
            g.faces()
            best = min(best, time.perf_counter() - t)
        return best

    small, large = build(400), build(1600)
    ratio = timed(large) / max(timed(small), 1e-6)
    assert ratio < 8, f"faces() scaled by {ratio:.1f}x for 4x the edges (quadratic?)"


def test_edge_index_survives_removal_and_reload(tmp_path):
    g, ctx = build_triangle()
    g.remove_edge(ctx["edges"][1])
    g.check_invariants()
    assert set(g.edges()) == {ctx["edges"][0], ctx["edges"][2]}
    with pytest.raises(Exception):
        g.edge_half_edges(ctx["edges"][1])

    g.save(tmp_path / "g")
    back = FatGraph.load(tmp_path / "g")
    back.check_invariants()
    assert back.edges() == g.edges()


def test_stats_reports_face_histogram():
    g, _ = build_triangle()
    s = g.stats()
    assert s["face_length_hist"] == {"3": 2}
    assert s["n_leaf_faces"] == 0
    assert s["edges_by_state"]["emergente"] == 3
