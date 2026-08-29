"""English surface patterns for the temporal resolver (spec table 5.2).

English, not Portuguese, because the target corpus (LoCoMo) is English
dialogue -- the spec's own worked examples happen to be pt-BR, but the
patterns a resolver actually needs are whatever language the memory is
built from.

Every function here takes the already-isolated ``time_expression`` span
(the LLM never sees a whole turn at this stage, spec 6.3 rule 1) and
either returns a match or ``None``. ``resolve_time`` in
:mod:`fgl.clio.temporal.resolver` tries them in the table's order and
takes the first hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Granularity = Literal["day", "week", "month", "year"]

MONTHS_EN: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_ALTERNATION = "|".join(sorted(MONTHS_EN, key=len, reverse=True))

NUMBER_WORDS_EN: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_NUMBER_ALTERNATION = "|".join(sorted(NUMBER_WORDS_EN, key=len, reverse=True))


def _number(text: str) -> int | None:
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS_EN.get(text)


@dataclass(frozen=True)
class DateMatch:
    year: int
    month: int
    day: int | None  # None = whole month
    granularity: Granularity


# --- 1. absolute dates --------------------------------------------------- #
# Both orders LoCoMo's own gold answers and session headers use interchange-
# ably: "14 January 2023" and "January 14, 2023" (see fgl.memory.temporal,
# which renders resolved dates the same "D Month YYYY" way for the same
# reason -- token-overlap scoring punishes an ISO gloss the gold answer
# never uses).

_ABS_FULL_DMY = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALTERNATION})\s*,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
_ABS_FULL_MDY = re.compile(
    rf"\b({_MONTH_ALTERNATION})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
_ABS_MONTH_YEAR = re.compile(
    rf"\b(?:in\s+)?({_MONTH_ALTERNATION})\s+(\d{{4}})\b", re.IGNORECASE
)
#: not anchored to "in" -- the span is already isolated to a temporal
#: expression (spec 6.3 rule 1), so a bare 4-digit token is unambiguously a
#: year even after a marker word with no preposition ("since 2019").
_ABS_YEAR = re.compile(r"\b(\d{4})\b")


def match_absolute(span: str) -> DateMatch | None:
    m = _ABS_FULL_DMY.search(span)
    if m:
        day, month_name, year = m.groups()
        return DateMatch(int(year), MONTHS_EN[month_name.lower()], int(day), "day")
    m = _ABS_FULL_MDY.search(span)
    if m:
        month_name, day, year = m.groups()
        return DateMatch(int(year), MONTHS_EN[month_name.lower()], int(day), "day")
    m = _ABS_MONTH_YEAR.search(span)
    if m:
        month_name, year = m.groups()
        return DateMatch(int(year), MONTHS_EN[month_name.lower()], None, "month")
    m = _ABS_YEAR.search(span)
    if m:
        return DateMatch(int(m.group(1)), 1, None, "year")
    return None


# --- 2. day deixis --------------------------------------------------------- #
# Longest phrase first: "the day before yesterday" must not be short-
# circuited by a bare "yesterday" match inside it.

_DAY_BEFORE_YESTERDAY_RE = re.compile(r"\bthe\s+day\s+before\s+yesterday\b", re.IGNORECASE)
_YESTERDAY_RE = re.compile(r"\byesterday\b", re.IGNORECASE)
_TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)


def match_day_deixis(span: str) -> int | None:
    """Returns the day offset from the anchor, or None."""
    if _DAY_BEFORE_YESTERDAY_RE.search(span):
        return -2
    if _YESTERDAY_RE.search(span):
        return -1
    if _TODAY_RE.search(span):
        return 0
    return None


# --- 3. week deixis --------------------------------------------------------- #

_WEEK_LAST_RE = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_WEEK_THIS_RE = re.compile(r"\bthis\s+week\b", re.IGNORECASE)


def match_week_deixis(span: str) -> Literal["last", "this"] | None:
    if _WEEK_LAST_RE.search(span):
        return "last"
    if _WEEK_THIS_RE.search(span):
        return "this"
    return None


# --- 4. month deixis --------------------------------------------------------- #

_MONTH_LAST_RE = re.compile(r"\blast\s+month\b", re.IGNORECASE)
_MONTH_BARE_RE = re.compile(rf"\bin\s+({_MONTH_ALTERNATION})\b", re.IGNORECASE)


def match_month_deixis(span: str) -> tuple[str, int | None] | None:
    """Returns ("last", None) for "last month", or ("named", month_number)
    for a bare month name with no year attached."""
    if _MONTH_LAST_RE.search(span):
        return ("last", None)
    m = _MONTH_BARE_RE.search(span)
    if m:
        return ("named", MONTHS_EN[m.group(1).lower()])
    return None


# --- 5. year deixis --------------------------------------------------------- #

_YEAR_LAST_RE = re.compile(r"\blast\s+year\b", re.IGNORECASE)


def match_year_deixis(span: str) -> bool:
    return bool(_YEAR_LAST_RE.search(span))


# --- 6. retroactive duration --------------------------------------------- #

_DURATION_RE = re.compile(
    rf"\b({_NUMBER_ALTERNATION}|\d+)\s+"
    r"(days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE,
)

_UNIT_TO_GRANULARITY: dict[str, Granularity] = {
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
}


@dataclass(frozen=True)
class DurationMatch:
    amount: int
    granularity: Granularity


def match_retroactive_duration(span: str) -> DurationMatch | None:
    m = _DURATION_RE.search(span)
    if not m:
        return None
    amount = _number(m.group(1))
    if amount is None:
        return None
    unit = m.group(2).lower()
    return DurationMatch(amount, _UNIT_TO_GRANULARITY[unit])


# --- 7/8. start/end markers ------------------------------------------------ #
# Force an open-ended interval regardless of what the base pattern above
# produced (spec table rows "start marker"/"end marker"). They fire on top
# of a resolved base match, not as a resolution path of their own -- a bare
# "since" with no date anywhere in the span resolves nothing.

_START_MARKER_RE = re.compile(r"\b(started|since|joined)\b", re.IGNORECASE)
_END_MARKER_RE = re.compile(r"\b(left|quit|stopped|resigned|until)\b", re.IGNORECASE)


def has_start_marker(span: str) -> bool:
    return bool(_START_MARKER_RE.search(span))


def has_end_marker(span: str) -> bool:
    return bool(_END_MARKER_RE.search(span))


# --- 9. vague ---------------------------------------------------------------- #

_VAGUE_RE = re.compile(
    r"\b(recently|a\s+while\s+ago|lately|some\s+time\s+ago)\b", re.IGNORECASE
)


def is_vague(span: str) -> bool:
    return bool(_VAGUE_RE.search(span))


# --- 10. weekday-qualified fallback (not a spec 5.2 row) -------------------- #
# "last Saturday" / "next Tuesday" / "on Monday": the dominant shape of
# LoCoMo's own relative-date questions (69.5% of category-2/temporal
# questions have one, measured against the real dataset -- see
# fgl.memory.temporal, which resolves the same pattern for condition L1).
# Reimplemented here, not imported from there: importing anything under
# fgl.memory eagerly pulls in fgl.config/fgl.core (the fatgraph-condition
# experiment machinery), which this package is built to stay independent
# of. dateparser.parse() alone fails outright on "last/next + weekday"
# (returns None) even though it resolves everything else in this fallback
# fine; dateparser.search.search_dates() on the same isolated span
# succeeds, which is why both are tried.

_PAST_CUES_RE = re.compile(
    r"\b(last|yesterday|ago|before|prior|earlier|past)\b", re.IGNORECASE
)
_FUTURE_CUES_RE = re.compile(
    r"\b(next|tomorrow|coming|upcoming|after|later|from now|in the future)\b",
    re.IGNORECASE,
)


def resolve_via_dateparser(span: str, anchor: datetime) -> datetime | None:
    """Best-effort resolution for anything the hand-rolled patterns above
    miss. Returns ``None`` (never a guess) when ``dateparser`` is not
    installed, when it cannot parse the span at all, or when it silently
    reflects the anchor back -- its documented failure mode, and
    indistinguishable from a real "today" only by checking the words.
    """
    try:
        import dateparser
        from dateparser.search import search_dates
    except ImportError:
        return None

    direction = (
        "future"
        if _FUTURE_CUES_RE.search(span) and not _PAST_CUES_RE.search(span)
        else "past"
    )
    settings = {"RELATIVE_BASE": anchor, "PREFER_DATES_FROM": direction}

    dt = dateparser.parse(span, settings=settings)
    if dt is None:
        hits = search_dates(span, settings=settings)
        if hits:
            dt = hits[0][1]
    if dt is None:
        return None
    if dt.date() == anchor.date() and span.strip().lower() not in ("today", "this day"):
        return None
    return dt
