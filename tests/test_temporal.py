"""Unit tests for :mod:`fgl.memory.temporal` -- the deterministic relative-date
resolver. Each test pins down one of the empirically-verified quirks
documented in the module docstring, rather than re-testing dateparser itself:
this project does not own dateparser's correctness, only the two-stage
fallback and the direction/no-op-rejection logic wrapped around it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("dateparser")

from fgl.memory.temporal import (
    annotate_text,
    resolve_all,
    resolve_relative_date,
)

# Wednesday, 17 May 2023 -- a session timestamp fixed so every test is
# reproducible regardless of what day it actually runs (and not itself a
# Saturday, which would make "resolves to the most recent Saturday" and
# "resolves to today" indistinguishable).
BASE = datetime(2023, 5, 17, 13, 56, 0)


def test_bare_weekday_resolves_to_the_most_recent_occurrence():
    # No past/future cue in "Saturday" alone -> _direction() defaults to
    # "past" (see temporal.py's _direction docstring), so this resolves
    # backward to the Saturday before BASE, not the following one.
    r = resolve_relative_date("Saturday", BASE)
    assert r is not None
    assert r.resolved.date().isoformat() == "2023-05-13"


def test_last_weekday_resolves_backward():
    # The exact case dateparser.parse() returns None on outright -- this is
    # what the search_dates fallback exists for.
    r = resolve_relative_date("last Saturday", BASE)
    assert r is not None
    assert r.resolved <= BASE
    assert r.resolved.date().isoformat() == "2023-05-13"


def test_next_weekday_resolves_forward():
    r = resolve_relative_date("next Saturday", BASE)
    assert r is not None
    assert r.resolved >= BASE


def test_last_week_resolves():
    r = resolve_relative_date("last week", BASE)
    assert r is not None
    assert r.resolved < BASE


def test_next_month_resolves_forward():
    r = resolve_relative_date("next month", BASE)
    assert r is not None
    assert r.resolved > BASE


def test_unresolvable_span_returns_none():
    assert resolve_relative_date("", BASE) is None
    assert resolve_relative_date("   ", BASE) is None


def test_no_op_resolution_is_rejected():
    # "this week" is a real DATE span (spaCy tags it as one -- see
    # test_ner.py) that dateparser resolves to exactly BASE itself: not a
    # fabricated date, but not an actual movement either, so the guard
    # rejects it rather than reporting a resolution that adds no information.
    r = resolve_relative_date("this week", BASE)
    assert r is None


def test_today_is_accepted_even_though_it_equals_base():
    r = resolve_relative_date("today", BASE)
    assert r is not None
    assert r.resolved_date == BASE.date().isoformat()


def test_resolve_all_deduplicates_repeated_spans():
    out = resolve_all(["last Saturday", "last Saturday", "next week"], BASE)
    assert len(out) == 2
    assert {r.raw for r in out} == {"last Saturday", "next week"}


def test_resolve_all_skips_unresolvable_without_raising():
    out = resolve_all(["last Saturday", "gibberish not a date"], BASE)
    assert len(out) == 1
    assert out[0].raw == "last Saturday"


def test_annotate_text_appends_gloss_without_touching_original_phrase():
    resolved = resolve_all(["last Saturday"], BASE)
    out = annotate_text("I went hiking last Saturday.", resolved)
    assert out.startswith("I went hiking last Saturday.")
    # natural-language format ("13 May 2023"), not ISO -- see render()'s
    # docstring for the measured F1 regression ISO caused on the real run.
    assert "13 May 2023" in out
    assert "last Saturday" in out  # original phrase preserved, not replaced


def test_render_uses_natural_language_not_iso():
    r = resolve_relative_date("last Saturday", BASE)
    assert r is not None
    assert "2023-05-13" not in r.render()
    assert "13 May 2023" in r.render()
    # resolved_date itself stays ISO -- it is machine-facing (vertex meta),
    # only render()'s reader-facing gloss changed.
    assert r.resolved_date == "2023-05-13"


def test_annotate_text_is_a_no_op_with_nothing_resolved():
    text = "Nothing date-related here."
    assert annotate_text(text, []) == text
