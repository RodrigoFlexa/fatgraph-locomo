"""Deterministic resolution of relative dates ("last Saturday") to absolute
ones, anchored to the session timestamp the phrase was said in. No LLM.

Motivation, measured on the real LoCoMo dataset (all 10 conversations, all
321 category-2/temporal questions): 223 of them (69.5%) have their evidence
turn contain a relative-date phrase, and the current pipeline has nothing
that resolves it -- ``ingest.extract_facts`` never asks the extractor to do
date arithmetic, so whatever the reader gets is whatever `fact_text`
happened to preserve verbatim. This module does the arithmetic once, at
ingest time, deterministically, and hands the reader the resolved date
instead of asking it to compute "last Saturday" relative to a date it may not
even see in the same prompt fragment.

``dateparser`` alone is not reliable enough to point at raw turn text and
trust the result -- verified empirically before writing this module:
``dateparser.search.search_dates`` on a full sentence produces false
positives on short common words ("an", "a"), and ``dateparser.parse`` on an
isolated span fails outright on "last/next + weekday" combinations
(``dateparser.parse("last Saturday")`` returns ``None``) even though it
handles a bare weekday name or "last/next + week/month" correctly. So this
module only ever resolves a span spaCy's NER already isolated as DATE/TIME
(see :mod:`fgl.memory.ner`), tries the fast path first, falls back to
``search_dates`` on that same isolated span for the weekday-qualifier case,
and never guesses a direction it cannot infer from the words themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

#: words that flip the search direction. Checked against the lower-cased
#: span text. "before"/"ago"/"last" -> look backward from the session date;
#: "next"/"coming"/"after" -> look forward. Ambiguous or absent -> backward,
#: which matches the dominant pattern in the corpus (a recap of what someone
#: did recently) and is also the safer default for an extractive reader.
_PAST_CUES = re.compile(r"\b(last|yesterday|ago|before|prior|earlier|past)\b", re.IGNORECASE)
_FUTURE_CUES = re.compile(
    r"\b(next|tomorrow|coming|upcoming|after|later|from now|in the future)\b",
    re.IGNORECASE,
)

#: "last/next/this (coming) <weekday>" -- the pattern dateparser.parse cannot
#: resolve directly but dateparser.search.search_dates can, once isolated.
_WEEKDAY_QUALIFIED = re.compile(
    r"\b(last|next|this( coming)?|coming)\s+"
    r"(mon|tues?|wednes|thurs?|fri|satur|sun)day\b",
    re.IGNORECASE,
)


@dataclass
class ResolvedDate:
    raw: str  # the span exactly as it appeared ("last Saturday")
    resolved: datetime  # absolute datetime it was resolved to
    resolved_date: str  # ISO date -- machine-facing (logs, vertex meta), NOT
    # what gets shown to the reader; see render().

    def render(self) -> str:
        # Natural-language format ("7 May 2023"), matching how LoCoMo's own
        # gold answers and session dates are written. Measured regression
        # (first real run of L1, all 10 conversations, real LLM): the ISO
        # gloss this used to emit ("... = 2023-05-07") was arithmetically
        # exact but scored F1=0.0 against gold answers like "7 May 2023" --
        # zero token overlap after the official scorer's tokenisation --
        # even on questions where recall_context was 1.0 and the resolved
        # date was the literal correct day. Temporal was the worst-scoring
        # category (F1 0.178) despite the best non-adversarial recall_context
        # (0.768); this format mismatch, not a retrieval miss, was the cause
        # for a large share of those. `%-d` (no leading zero) is avoided
        # deliberately -- it is a glibc strftime extension, not portable.
        day = self.resolved.day
        return f"'{self.raw}' = {day} {self.resolved.strftime('%B %Y')}"


def _direction(span: str) -> str:
    has_past = bool(_PAST_CUES.search(span))
    has_future = bool(_FUTURE_CUES.search(span))
    if has_future and not has_past:
        return "future"
    return "past"  # default, and the case where both/neither cue fires


def resolve_relative_date(span: str, base: datetime) -> Optional[ResolvedDate]:
    """Resolve one DATE/TIME span (already isolated by spaCy) against ``base``.

    Returns ``None`` rather than guessing when nothing resolves -- an
    unresolved phrase is left exactly as the turn text already has it, which
    is always at least as good as a fabricated wrong date.
    """
    import dateparser
    from dateparser.search import search_dates

    span = (span or "").strip()
    if not span:
        return None
    direction = _direction(span)
    settings = {"RELATIVE_BASE": base, "PREFER_DATES_FROM": direction}

    dt = dateparser.parse(span, settings=settings)
    if dt is None and _WEEKDAY_QUALIFIED.search(span):
        hits = search_dates(span, settings=settings)
        if hits:
            dt = hits[0][1]
    if dt is None:
        return None
    # dateparser can return the base itself when it fails to move at all
    # (matched only stray punctuation/short tokens); that is not a real
    # resolution, so it is rejected rather than reported as one.
    if dt.date() == base.date() and span.lower() not in ("today", "this day"):
        return None
    return ResolvedDate(raw=span, resolved=dt, resolved_date=dt.date().isoformat())


def resolve_all(spans: list[str], base: datetime) -> list[ResolvedDate]:
    out: list[ResolvedDate] = []
    seen: set[str] = set()
    for span in spans:
        if span in seen:
            continue
        seen.add(span)
        r = resolve_relative_date(span, base)
        if r is not None:
            out.append(r)
    return out


def annotate_text(text: str, resolved: list[ResolvedDate]) -> str:
    """Append resolved dates to a turn's text, so the reader sees the
    arithmetic already done instead of having to infer it from a session
    date it may not have in the same context fragment.

    Deliberately appended, not substituted in-place: the original phrase
    stays exactly as spoken (provenance-preserving), and the resolution
    reads as a gloss on it, the same way a human annotator would add
    "(= 20 May 2023)" in brackets after "last Saturday".
    """
    if not resolved:
        return text
    gloss = "; ".join(r.render() for r in resolved)
    return f"{text} [{gloss}]"
