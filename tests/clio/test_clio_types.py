"""Interval algebra, exhaustively (spec 17.1 item 1): coincident bounds,
half-open semantics, ``None`` on each side, and empty intersection by
exact adjacency.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from fgl.clio.types import Interval

D = datetime


def test_open_on_both_sides_contains_everything():
    i = Interval(None, None)
    assert i.contains(D(1900, 1, 1))
    assert i.contains(D(2100, 1, 1))
    assert i.is_open()


def test_half_open_end_excludes_the_boundary_instant():
    i = Interval(D(2023, 1, 1), D(2023, 2, 1))
    assert i.contains(D(2023, 1, 1))  # start is inclusive
    assert i.contains(D(2023, 1, 31, 23, 59, 59))
    assert not i.contains(D(2023, 2, 1))  # end is exclusive


def test_intersect_coincident_bounds():
    a = Interval(D(2023, 1, 1), D(2023, 3, 1))
    b = Interval(D(2023, 1, 1), D(2023, 3, 1))
    assert a.intersect(b) == Interval(D(2023, 1, 1), D(2023, 3, 1))


def test_intersect_exact_adjacency_is_empty():
    """[Jan, Mar) and [Mar, May) share no instant -- half-open, not closed."""
    a = Interval(D(2023, 1, 1), D(2023, 3, 1))
    b = Interval(D(2023, 3, 1), D(2023, 5, 1))
    assert a.intersect(b) is None
    assert not a.overlaps(b)


def test_intersect_one_sided_open_start():
    a = Interval(None, D(2023, 6, 1))
    b = Interval(D(2023, 1, 1), None)
    result = a.intersect(b)
    assert result == Interval(D(2023, 1, 1), D(2023, 6, 1))


def test_intersect_one_sided_open_end():
    a = Interval(D(2023, 1, 1), None)
    b = Interval(D(2023, 3, 1), None)
    result = a.intersect(b)
    assert result == Interval(D(2023, 3, 1), None)


def test_intersect_both_fully_open():
    a = Interval(None, None)
    b = Interval(None, None)
    assert a.intersect(b) == Interval(None, None)


def test_intersect_disjoint_real_gap():
    a = Interval(D(2023, 1, 1), D(2023, 2, 1))
    b = Interval(D(2023, 3, 1), D(2023, 4, 1))
    assert a.intersect(b) is None


def test_intersect_nested():
    outer = Interval(D(2020, 1, 1), D(2025, 1, 1))
    inner = Interval(D(2022, 1, 1), D(2022, 6, 1))
    assert outer.intersect(inner) == inner
    assert inner.intersect(outer) == inner


def test_overlaps_is_symmetric():
    a = Interval(D(2023, 1, 1), D(2023, 6, 1))
    b = Interval(D(2023, 5, 1), D(2023, 8, 1))
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_construction_rejects_start_after_end():
    with pytest.raises(ValueError):
        Interval(D(2023, 6, 1), D(2023, 1, 1))


def test_construction_allows_start_equal_end_as_empty_but_valid():
    # a zero-width interval is legal to construct; it simply contains nothing
    i = Interval(D(2023, 1, 1), D(2023, 1, 1))
    assert not i.contains(D(2023, 1, 1))
