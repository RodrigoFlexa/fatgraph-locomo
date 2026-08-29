"""Deterministic temporal resolution (spec section 5). The LLM never
computes a date -- it copies the literal span (spec 6.3 rule 1), and this
module does the arithmetic once, here, so it is auditable and testable
without any model in the loop.

English, not the spec document's own pt-BR examples: the target corpus
(LoCoMo) is English dialogue, so the surface patterns this resolver
actually needs to recognise are English ones -- see
:mod:`fgl.clio.temporal.patterns_en`.

Resolution order matches spec table 5.2 exactly: absolute date, then day /
week / month / year deixis, then retroactive duration, then "vague" (which
resolves to nothing on purpose -- spec 5.4 treats an honestly unresolved
date as a capability, not a failure). Start/end markers ("started",
"since", "left", "until") are checked independently of that ladder and,
when present, force the result open-ended.

One rule is *not* spelled out as its own table row and is instead implied
by spec's own worked fixture (``tests/fixtures/melanie.yaml``, assertion 1
and 4 of section 17.2): a deictic window that has both a start and an end
(e.g. "last month" -> the whole of May) describes *when the changing
event happened*, not the shape of the resulting state. For a "slow" or
"static" relation -- an ongoing state, not a one-off occurrence -- the
window collapses to ``[window.start, None)``: the state begins somewhere
in that window and, absent a closing fact, persists. A "fast" relation (an
event) keeps the full window, because there the window IS the answer.
Without this collapse, "Melanie moved to Salvador last month" would open a
residency that closes itself a month later for no reason -- confirmed
against the fixture, not guessed.

Weekday-qualified phrases ("last Saturday", "next Tuesday", "on Monday")
are not one of table 5.2's own rows either, and they matter far more than
anything the table does name: measured on the real LoCoMo dataset (see
:mod:`fgl.memory.temporal`, condition L1's relative-date resolver), 69.5%
of category-2/temporal questions have an evidence turn containing a
relative-date phrase, and a bare weekday qualifier is the dominant shape
of it. This ladder resolves them with the same ``dateparser``-based
approach that module already uses -- reimplemented locally
(:func:`~fgl.clio.temporal.patterns_en.resolve_via_dateparser`) rather than
imported, because ``fgl.memory`` eagerly imports ``fgl.config`` and
``fgl.core`` (the fatgraph-condition experiment machinery) as a side
effect of importing anything in that package at all, which this package is
built to stay independent of (see ``fgl/clio/__init__.py``). The fallback
runs only after :func:`~fgl.clio.temporal.patterns_en.is_vague` has had a
chance to say no: otherwise a genuinely vague phrase this module
intentionally leaves unresolved (spec 5.4) could get a plausible-looking
date handed to it by a more permissive parser tuned for a different job.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta

from fgl.clio.catalog.spec import RelationSpec
from fgl.clio.temporal import patterns_en as pt
from fgl.clio.types import Interval

#: fallback width for a "fast" relation with no explicit time expression and
#: no per-relation override (RelationSpec.default_duration_days), matching
#: the spec's own config default (``temporal.fast_window_days: 1``).
_DEFAULT_FAST_WINDOW_DAYS = 1

# --- confidence, spec 5.2's rightmost column --------------------------------- #
_CONF_ABSOLUTE = 1.00
_CONF_DAY_DEIXIS = 0.95
_CONF_WEEK_DEIXIS = 0.85
_CONF_MONTH_DEIXIS = 0.85
_CONF_YEAR_DEIXIS = 0.90
_CONF_DURATION = 0.80
#: below the hand-rolled patterns' 0.85+ (those match a fixed, closed
#: vocabulary; this runs a general-purpose parser) but above duration's
#: 0.80 (a resolved weekday is an exact day, not an open guess).
_CONF_WEEKDAY_FALLBACK = 0.90
_CONF_ABSENT = 1.00  # a code-computed default carries no parse ambiguity
_CONF_VAGUE = 0.0


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_range(year: int, month: int) -> Interval:
    start = datetime(year, month, 1)
    end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return Interval(start, end, granularity="month")


def _year_range(year: int) -> Interval:
    return Interval(datetime(year, 1, 1), datetime(year + 1, 1, 1), granularity="year")


def _week_start(anchor: datetime) -> datetime:
    """Sunday-Saturday week containing ``anchor`` (see module docstring's
    fixture note: the melanie fixture's expected dates only work out under
    a Sunday-start week, not an ISO Monday-start one)."""
    day = _start_of_day(anchor)
    days_since_sunday = (day.weekday() + 1) % 7  # Monday=0 .. Sunday=6
    return day - timedelta(days=days_since_sunday)


def _subtract_months(dt: datetime, n: int) -> datetime:
    total = dt.month - 1 - n
    year = dt.year + total // 12
    month = total % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _subtract_years(dt: datetime, n: int) -> datetime:
    try:
        return dt.replace(year=dt.year - n)
    except ValueError:  # Feb 29 with no leap target
        return dt.replace(year=dt.year - n, day=28)


def _match_ladder(span: str, anchor: datetime) -> tuple[Interval | None, float]:
    """Table 5.2, rows 1-6, then "vague", then the weekday-qualified
    fallback (see module docstring). Does NOT apply the marker override or
    the slow/static collapse -- those are the caller's job."""
    m = pt.match_absolute(span)
    if m is not None:
        if m.day is not None:
            start = datetime(m.year, m.month, m.day)
            return Interval(
                start, start + timedelta(days=1), granularity="day"
            ), _CONF_ABSOLUTE
        if m.granularity == "month":
            return _month_range(m.year, m.month), _CONF_ABSOLUTE
        return _year_range(m.year), _CONF_ABSOLUTE

    offset = pt.match_day_deixis(span)
    if offset is not None:
        start = _start_of_day(anchor) + timedelta(days=offset)
        return Interval(
            start, start + timedelta(days=1), granularity="day"
        ), _CONF_DAY_DEIXIS

    week = pt.match_week_deixis(span)
    if week is not None:
        this_week_start = _week_start(anchor)
        start = this_week_start if week == "this" else this_week_start - timedelta(days=7)
        return Interval(
            start, start + timedelta(days=7), granularity="week"
        ), _CONF_WEEK_DEIXIS

    month = pt.match_month_deixis(span)
    if month is not None:
        kind, month_number = month
        if kind == "last":
            prev = _subtract_months(anchor, 1)
            return _month_range(prev.year, prev.month), _CONF_MONTH_DEIXIS
        year = anchor.year if month_number <= anchor.month else anchor.year - 1
        return _month_range(year, month_number), _CONF_MONTH_DEIXIS

    if pt.match_year_deixis(span):
        return _year_range(anchor.year - 1), _CONF_YEAR_DEIXIS

    duration = pt.match_retroactive_duration(span)
    if duration is not None:
        if duration.granularity == "day":
            start = _start_of_day(anchor) - timedelta(days=duration.amount)
        elif duration.granularity == "week":
            start = _start_of_day(anchor) - timedelta(weeks=duration.amount)
        elif duration.granularity == "month":
            start = _subtract_months(anchor, duration.amount)
        else:
            start = _subtract_years(anchor, duration.amount)
        return Interval(start, None, granularity=duration.granularity), _CONF_DURATION

    if pt.is_vague(span):
        return None, _CONF_VAGUE

    resolved_dt = pt.resolve_via_dateparser(span, anchor)
    if resolved_dt is not None:
        start = _start_of_day(resolved_dt)
        return Interval(
            start, start + timedelta(days=1), granularity="day"
        ), _CONF_WEEKDAY_FALLBACK

    return None, _CONF_VAGUE


def _default_for_volatility(anchor: datetime, relation: RelationSpec) -> Interval:
    """Spec 5.3: what a relation defaults to when no time expression is
    given at all, keyed purely by volatility."""
    if relation.volatility == "static":
        return Interval(None, None)
    if relation.volatility == "slow":
        return Interval(anchor, None)
    window_days = relation.default_duration_days or _DEFAULT_FAST_WINDOW_DAYS
    return Interval(anchor, anchor + timedelta(days=window_days))


def resolve_time(
    expression: str | None,
    anchor: datetime,
    relation: RelationSpec,
    locale: str = "en_US",
) -> tuple[Interval | None, float]:
    """Returns ``(interval, resolution_confidence)``. ``interval`` is
    ``None`` when the expression could not be resolved at all (spec 5.4):
    the proposition still exists, just unanchored, findable only through
    the log's partial order until something resolves it.
    """
    if locale != "en_US":
        raise NotImplementedError(f"resolve_time: unsupported locale {locale!r}")

    if not expression or not expression.strip():
        return _default_for_volatility(anchor, relation), _CONF_ABSENT

    base, tconf = _match_ladder(expression, anchor)
    if base is None:
        return None, tconf

    if pt.has_start_marker(expression) or pt.has_end_marker(expression):
        return Interval(base.start, None, granularity=base.granularity), tconf

    if relation.volatility in ("slow", "static") and base.end is not None:
        return Interval(base.start, None, granularity=base.granularity), tconf

    return base, tconf


def tconf_factor(tconf: float) -> float:
    """Spec 6.4: poor temporal resolution discounts the proposition's
    overall confidence -- never boosts it, never zeroes it outright (a
    proposition with an unresolved date is still evidence, just weaker)."""
    if tconf >= 0.85:
        return 1.0
    if tconf > 0:
        return 0.9
    return 0.85
