"""Deterministic English temporal resolution (spec section 5, milestone
M3). English, not the spec document's own pt-BR examples, because the
target corpus (LoCoMo) is English dialogue -- see
:mod:`fgl.clio.temporal.patterns_en`.

Table-driven over every row of spec table 5.2, at least 40 distinct
expressions as the milestone plan requires, plus the volatility-based
"absent" defaults (5.3) and the marker-vs-collapse interaction that spec's
own worked fixture (melanie.yaml) forced into the open (see the module
docstring of :mod:`fgl.clio.temporal.resolver`).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fgl.clio.catalog import load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.temporal.resolver import _subtract_months, _subtract_years, resolve_time
from fgl.clio.types import Interval

CATALOG = load_catalog(ClioConfig.default().catalog_path)
SLOW = CATALOG["lives_in"]  # functional, slow
STATIC = CATALOG["born_in"]  # functional, static
FAST = CATALOG["attended"]  # multi, fast

ANCHOR = datetime(2023, 6, 15)  # a Thursday


# --------------------------------------------------------------------- #
# 1. Absolute dates -- confidence 1.00                                    #
# --------------------------------------------------------------------- #
ABSOLUTE_CASES = [
    ("14 January 2023", datetime(2023, 1, 14), datetime(2023, 1, 15), "day"),
    ("December 1, 2020", datetime(2020, 12, 1), datetime(2020, 12, 2), "day"),
    ("25 March 2019", datetime(2019, 3, 25), datetime(2019, 3, 26), "day"),
    ("July 3, 2022", datetime(2022, 7, 3), datetime(2022, 7, 4), "day"),
    ("in May 2021", datetime(2021, 5, 1), datetime(2021, 6, 1), "month"),
    ("December 2018", datetime(2018, 12, 1), datetime(2019, 1, 1), "month"),
    ("in 2019", datetime(2019, 1, 1), datetime(2020, 1, 1), "year"),
    ("2005", datetime(2005, 1, 1), datetime(2006, 1, 1), "year"),
]


@pytest.mark.parametrize("expr,start,end,gran", ABSOLUTE_CASES)
def test_absolute_dates(expr, start, end, gran):
    interval, tconf = resolve_time(expr, ANCHOR, FAST)
    assert interval.start == start
    assert interval.end == end
    assert interval.granularity == gran
    assert tconf == 1.00


# --------------------------------------------------------------------- #
# 2. Day deixis -- confidence 0.95                                        #
# --------------------------------------------------------------------- #
DAY_DEIXIS_CASES = ["today", "yesterday", "the day before yesterday"]
DAY_OFFSETS = {"today": 0, "yesterday": -1, "the day before yesterday": -2}


@pytest.mark.parametrize("expr", DAY_DEIXIS_CASES)
def test_day_deixis(expr):
    interval, tconf = resolve_time(expr, ANCHOR, FAST)
    expected_start = ANCHOR + timedelta(days=DAY_OFFSETS[expr])
    assert interval.start == expected_start
    assert interval.end == expected_start + timedelta(days=1)
    assert interval.granularity == "day"
    assert tconf == 0.95


# --------------------------------------------------------------------- #
# 3. Week deixis -- confidence 0.85. Sunday-start week (see resolver's    #
#    module docstring for why -- matches the spec's own worked fixture). #
# --------------------------------------------------------------------- #
def _week_start(d: datetime) -> datetime:
    return d - timedelta(days=(d.weekday() + 1) % 7)


def test_this_week_on_fast_relation_keeps_full_window():
    interval, tconf = resolve_time("this week", ANCHOR, FAST)
    start = _week_start(ANCHOR)
    assert interval.start == start
    assert interval.end == start + timedelta(days=7)
    assert interval.granularity == "week"
    assert tconf == 0.85


def test_last_week_on_fast_relation():
    interval, _ = resolve_time("last week", ANCHOR, FAST)
    start = _week_start(ANCHOR) - timedelta(days=7)
    assert interval.start == start
    assert interval.end == start + timedelta(days=7)


@pytest.mark.parametrize("expr", ["last weekend", "this past weekend"])
def test_weekend_deixis_is_resolved_without_dateparser(expr):
    interval, tconf = resolve_time(expr, ANCHOR, FAST)
    start = _week_start(ANCHOR) - timedelta(days=7)
    assert interval.start == start
    assert interval.end == start + timedelta(days=7)
    assert interval.granularity == "week"
    assert tconf == 0.85


def test_this_week_on_slow_relation_collapses_to_open_start():
    """The melanie fixture's works_at edge starts at the week's Sunday, not
    at the episode date, and never closes on its own (assertion basis for
    E1's works_at)."""
    interval, _ = resolve_time("this week", ANCHOR, SLOW)
    assert interval.start == _week_start(ANCHOR)
    assert interval.end is None


# --------------------------------------------------------------------- #
# 4. Month deixis -- confidence 0.85                                      #
# --------------------------------------------------------------------- #
def test_last_month_on_fast_relation():
    interval, tconf = resolve_time("last month", ANCHOR, FAST)
    assert interval.start == datetime(2023, 5, 1)
    assert interval.end == datetime(2023, 6, 1)
    assert interval.granularity == "month"
    assert tconf == 0.85


def test_next_month_preserves_month_granularity():
    interval, tconf = resolve_time("next month", ANCHOR, FAST)
    assert interval.start == datetime(2023, 7, 1)
    assert interval.end == datetime(2023, 8, 1)
    assert interval.granularity == "month"
    assert tconf == 0.85


def test_last_month_on_slow_relation_collapses():
    interval, _ = resolve_time("last month", ANCHOR, SLOW)
    assert interval.start == datetime(2023, 5, 1)
    assert interval.end is None


def test_bare_month_name_before_anchor_uses_this_year():
    interval, _ = resolve_time("in January", ANCHOR, FAST)  # Jan <= June
    assert interval.start == datetime(2023, 1, 1)
    assert interval.end == datetime(2023, 2, 1)


def test_bare_month_name_after_anchor_uses_last_year():
    interval, _ = resolve_time("in December", ANCHOR, FAST)  # Dec > June
    assert interval.start == datetime(2022, 12, 1)
    assert interval.end == datetime(2023, 1, 1)


def test_bare_month_name_same_as_anchor_month_uses_this_year():
    interval, _ = resolve_time("in June", ANCHOR, FAST)
    assert interval.start == datetime(2023, 6, 1)


# --------------------------------------------------------------------- #
# 5. Year deixis -- confidence 0.90                                       #
# --------------------------------------------------------------------- #
def test_last_year_on_fast_relation():
    interval, tconf = resolve_time("last year", ANCHOR, FAST)
    assert interval.start == datetime(2022, 1, 1)
    assert interval.end == datetime(2023, 1, 1)
    assert interval.granularity == "year"
    assert tconf == 0.90


def test_last_year_on_slow_relation_collapses():
    interval, _ = resolve_time("last year", ANCHOR, SLOW)
    assert interval.start == datetime(2022, 1, 1)
    assert interval.end is None


# --------------------------------------------------------------------- #
# 6. Retroactive duration -- always open, confidence 0.80                #
# --------------------------------------------------------------------- #
DURATION_CASES = [
    ("two years ago", 2, "years"),
    ("three months ago", 3, "months"),
    ("5 days ago", 5, "days"),
    ("a year ago", 1, "years"),
    ("one week ago", 1, "weeks"),
    ("ten years ago", 10, "years"),
    ("3 weeks ago", 3, "weeks"),
    ("6 months ago", 6, "months"),
    ("one month ago", 1, "months"),
    ("two days ago", 2, "days"),
    ("two weekends ago", 2, "weeks"),
]


def _expected_duration_start(amount: int, unit: str) -> datetime:
    if unit == "days":
        return ANCHOR - timedelta(days=amount)
    if unit == "weeks":
        return ANCHOR - timedelta(weeks=amount)
    if unit == "months":
        return _subtract_months(ANCHOR, amount)
    return _subtract_years(ANCHOR, amount)


@pytest.mark.parametrize("expr,amount,unit", DURATION_CASES)
def test_retroactive_duration_always_open(expr, amount, unit):
    interval, tconf = resolve_time(expr, ANCHOR, FAST)
    assert interval.end is None  # open regardless of the relation's volatility
    assert interval.start == _expected_duration_start(amount, unit)
    assert tconf == 0.80


# --------------------------------------------------------------------- #
# 7/8. Start / end markers force open-ended, confidence inherited        #
# --------------------------------------------------------------------- #
MARKER_CASES = [
    "since 2019",
    "started in May 2021",
    "since yesterday",
    "until yesterday",
    "left in May 2021",
    "stopped last month",
    "joined in 2019",
]


@pytest.mark.parametrize("expr", MARKER_CASES)
def test_markers_force_open_even_on_fast_relation(expr):
    interval, _ = resolve_time(expr, ANCHOR, FAST)
    assert interval.end is None


# --------------------------------------------------------------------- #
# 9. Vague -- unresolved, confidence 0.0                                  #
# --------------------------------------------------------------------- #
VAGUE_CASES = ["recently", "a while ago", "lately", "some time ago"]


@pytest.mark.parametrize("expr", VAGUE_CASES)
def test_vague_resolves_to_nothing(expr):
    interval, tconf = resolve_time(expr, ANCHOR, SLOW)
    assert interval is None
    assert tconf == 0.0


# --------------------------------------------------------------------- #
# 10. Weekday-qualified fallback (not a spec 5.2 row, but the dominant     #
#     shape of LoCoMo's own temporal questions -- see resolver.py's       #
#     module docstring). Confidence 0.90, exact day granularity.          #
# --------------------------------------------------------------------- #
WEEKDAY_CASES = ["last Saturday", "next Tuesday", "this Saturday", "on Monday"]


@pytest.mark.parametrize("expr", WEEKDAY_CASES)
def test_weekday_qualified_phrases_resolve_via_the_dateparser_fallback(expr):
    interval, tconf = resolve_time(expr, ANCHOR, FAST)
    assert interval is not None, f"{expr!r} should not be left unresolved"
    assert interval.granularity == "day"
    assert interval.end == interval.start + timedelta(days=1)
    assert tconf == 0.90


def test_weekday_fallback_still_defers_to_the_hand_rolled_ladder_first():
    """ "Yesterday" must keep its 0.95/day-deixis identity, not fall through
    to the 0.90 dateparser fallback -- the fallback only fires when nothing
    above it in the ladder matched."""
    interval, tconf = resolve_time("yesterday", ANCHOR, FAST)
    assert tconf == 0.95


def test_weekday_fallback_on_slow_relation_collapses_like_any_other_match():
    interval, _ = resolve_time("last Saturday", ANCHOR, SLOW)
    assert interval.start == datetime(2023, 6, 10)
    assert interval.end is None  # same slow-volatility collapse as any other row


def test_marker_plus_weekday_forces_open_ended():
    interval, _ = resolve_time("since last Saturday", ANCHOR, FAST)
    assert interval.start == datetime(2023, 6, 10)
    assert interval.end is None


# --------------------------------------------------------------------- #
# Out-of-vocabulary phrases: honestly unresolved, not guessed (spec 5.4)  #
# --------------------------------------------------------------------- #
OOV_CASES = ["over the weekend", "sometime soon", "one of these days"]


@pytest.mark.parametrize("expr", OOV_CASES)
def test_unmatched_phrase_is_unresolved_not_guessed(expr):
    interval, tconf = resolve_time(expr, ANCHOR, SLOW)
    assert interval is None
    assert tconf == 0.0


# --------------------------------------------------------------------- #
# Absent expression: pure volatility default (spec 5.3)                  #
# --------------------------------------------------------------------- #
def test_absent_expression_static_is_fully_open():
    interval, tconf = resolve_time(None, ANCHOR, STATIC)
    assert interval == Interval(None, None)
    assert tconf == 1.0


def test_absent_expression_slow_opens_at_anchor():
    interval, _ = resolve_time(None, ANCHOR, SLOW)
    assert interval.start == ANCHOR
    assert interval.end is None


def test_absent_expression_fast_defaults_to_one_day_window():
    interval, _ = resolve_time("", ANCHOR, FAST)
    assert interval.start == ANCHOR
    assert interval.end == ANCHOR + timedelta(days=1)


def test_unsupported_locale_raises():
    with pytest.raises(NotImplementedError):
        resolve_time("yesterday", ANCHOR, FAST, locale="pt_BR")


def test_expression_count_meets_milestone_m3_requirement():
    """Spec 16, M3: 'a test suite with at least 40 expressions' -- in
    English here, not the spec document's own pt-BR, since the target
    corpus (LoCoMo) is English. Counted across every parametrized table
    above."""
    all_expressions = (
        [c[0] for c in ABSOLUTE_CASES]
        + DAY_DEIXIS_CASES
        + ["this week", "last week"]
        + ["last month", "in January", "in December", "in June"]
        + ["last year"]
        + [c[0] for c in DURATION_CASES]
        + MARKER_CASES
        + VAGUE_CASES
        + WEEKDAY_CASES
        + OOV_CASES
    )
    assert len(all_expressions) >= 40
