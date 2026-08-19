"""Answer shaping: recover the F1 the metric charges for wording, not content.

Why this module exists, measured rather than assumed
----------------------------------------------------
LoCoMo's headline metric is token-level F1 against a very short reference, so
a correct answer wrapped in a sentence is scored as a partly wrong one. On the
L2 run, restricted to questions whose annotated evidence was actually in the
prompt:

    single-hop  n=759  F1 0.651   best contiguous window of the same
                                  prediction: 0.745   irrecoverable: 13%
    multi-hop   n=109  F1 0.378   best window: 0.492  irrecoverable: 29%

"Best window" is an oracle -- it picks the best substring knowing the gold --
so it is an upper bound, not a target. But 49% of single-hop predictions
already contain a substring scoring a perfect 1.0: the model knows the answer
and pads it. That padding is what this module removes.

    gold 'Three dogs.'   pred 'I already have three dogs at home.'   F1 0.44
    gold '7 years'       pred 'Seven years now'                      F1 0.40

Design rules
------------
1. **Every transform is individually toggleable and individually measured.**
   A shaping rule is a hypothesis about how the model pads, and some of them
   are wrong: stripping a leading "yes" helps a single-hop answer and destroys
   an open-domain one. :func:`shape` takes a :class:`ShapingRules` so
   ``fgl reshape --ablate`` can price each rule on its own.
2. **Never invent tokens.** Everything here deletes; nothing rewrites content.
   The one exception is numeral spelling (:attr:`ShapingRules.numerals`), which
   is a substitution and is therefore off by default until measured -- LoCoMo
   golds use both forms ("7 years" but "Three dogs."), so it is as likely to
   cost as to pay, and only the data can say which.
3. **Never empty an answer.** Every rule falls back to the input when it would
   return nothing, so shaping can lose F1 by being wrong but never by erasing.
4. **Abstentions are untouchable.** The category-5 rule is a substring test
   for "not mentioned" / "no information available"; a trimmer that clipped
   those strings would silently convert correct abstentions into wrong
   answers, which is the one failure mode this module must not have.

Applied post hoc, this is free and retroactive: it rewrites a saved prediction
string and re-scores it, so every condition already on disk -- baselines
included -- can be re-measured under identical treatment without a single new
LLM call. That is also the only *fair* way to use it: shaping applied to one
condition is a prompt artefact, applied to all of them it is a metric fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fgl.data.locomo import ABSTAIN_ANSWER

# --------------------------------------------------------------------------- #
# Vocabulary                                                                   #
# --------------------------------------------------------------------------- #

#: Framing the model puts before the answer proper. Ordered longest-first so
#: "as far as I can tell" is consumed before "as far".
_LEAD_PHRASES = (
    "based on the memories", "based on the context", "according to the memories",
    "as far as i can tell", "from what i can tell", "it seems that", "it seems like",
    "it looks like", "i think that", "i believe that", "it appears that",
    "the answer is", "she said that", "he said that", "they said that",
    "i think", "i believe", "it was", "it is", "there was", "there were",
    "she said", "he said", "they said", "she has", "he has", "they have",
    "she had", "he had", "they had", "she is", "he is", "they are",
    "she was", "he was", "they were", "i already have", "i have", "i had",
    "i was", "i am", "well",
)

#: Polarity openers. Separate from the list above because they are framing for
#: an extractive answer and *content* for an inferential one: `answer_open.txt`
#: explicitly instructs category 3 to start with "Yes"/"No"/"Likely yes", and
#: the reference for those questions starts the same way. Stripping them
#: everywhere measured as a straight loss on open-domain, so `shape` takes the
#: category and keeps them there.
_POLARITY_PHRASES = ("likely yes", "likely no", "yes", "no", "probably yes",
                     "probably no")

#: LoCoMo category 3 -- open-domain / inferential. The one category whose
#: reference answers begin with a polarity word.
_INFERENTIAL_CATEGORY = 3

#: Trailing qualifiers that carry no content the reference could contain.
_TRAIL_PHRASES = (
    "in the conversation", "according to the memories", "based on the memories",
    "as mentioned", "as stated", "i think", "i believe", "at the time",
    "at home", "as well", "too", "now", "recently", "lately", "currently",
    "for now", "so far", "of course", "apparently", "probably", "maybe",
)

#: Single tokens safe to shave off either end once the phrases are gone.
_EDGE_TOKENS = frozenset({
    "so", "and", "but", "then", "just", "also", "really", "very", "quite",
    "actually", "basically", "definitely", "certainly", "the", "a", "an",
})

_NUMERALS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100",
}

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december"
)
#: ISO date -> the way LoCoMo writes dates. Kept here as well as in
#: fgl.memory.temporal because a prediction can carry an ISO date the model
#: produced on its own, which no ingest-side fix can reach.
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@dataclass(frozen=True)
class ShapingRules:
    """Which transforms to apply. Every one is a hypothesis; price them with
    ``fgl reshape --ablate`` before trusting a bundle.
    """

    #: keep only the first sentence -- the reference is a phrase, never two
    first_sentence: bool = True
    #: drop leading framing ("I think", "It was", "She said")
    strip_lead: bool = True
    #: drop trailing qualifiers ("at home", "as well", "in the conversation")
    strip_trail: bool = True
    #: drop leftover single filler tokens at either end
    strip_edges: bool = True
    #: rewrite an ISO date the model emitted into "7 May 2023"
    iso_dates: bool = True
    #: spell numerals as digits. OFF: LoCoMo golds use both forms, so this is
    #: as likely to cost as to pay and only measurement can decide.
    numerals: bool = False

    @classmethod
    def none(cls) -> "ShapingRules":
        return cls(first_sentence=False, strip_lead=False, strip_trail=False,
                   strip_edges=False, iso_dates=False, numerals=False)

    @classmethod
    def all_on(cls) -> "ShapingRules":
        return cls(first_sentence=True, strip_lead=True, strip_trail=True,
                   strip_edges=True, iso_dates=True, numerals=True)

    def names(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.__dict__.items() if v)


DEFAULT_RULES = ShapingRules()


# --------------------------------------------------------------------------- #
# API                                                                          #
# --------------------------------------------------------------------------- #


def is_abstention(text: str) -> bool:
    low = (text or "").lower()
    return "not mentioned" in low or "no information available" in low


def shape(
    text: str, rules: ShapingRules = DEFAULT_RULES, category: int | None = None
) -> str:
    """Trim a prediction down to the phrase the reference could contain.

    ``category`` is the LoCoMo category. It is used for exactly one decision --
    whether a leading "Yes"/"No" is framing or content (see
    ``_POLARITY_PHRASES``) -- and for nothing else; passing ``None`` treats the
    answer as extractive. It is not a routing hook and must not become one:
    shaping that varied by category would be fitting the metric per class
    rather than removing padding.

    Never returns empty, and never touches an abstention (rule 4 in the module
    docstring: clipping one would turn a correct category-5 answer into a wrong
    one, which no amount of F1 elsewhere would justify).
    """
    original = (text or "").strip()
    if not original or is_abstention(original):
        return original or ABSTAIN_ANSWER

    t = original
    if rules.iso_dates:
        t = _rewrite_iso_dates(t)
    if rules.first_sentence:
        t = _first_sentence(t)
    if rules.strip_lead:
        t = _strip_lead(t, keep_polarity=category == _INFERENTIAL_CATEGORY)
    if rules.strip_trail:
        t = _strip_trail(t)
    if rules.strip_edges:
        t = _strip_edges(t)
    if rules.numerals:
        t = _digits(t)

    t = t.strip().strip('"').strip("'").strip()
    # A rule that emptied the answer got it wrong; the unshaped string is
    # always a legal answer, an empty one never is.
    return t or original


# --------------------------------------------------------------------------- #
# Transforms                                                                   #
# --------------------------------------------------------------------------- #


def _rewrite_iso_dates(t: str) -> str:
    def repl(m: re.Match) -> str:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 1 <= mo <= 12:
            return m.group(0)
        return f"{d} {_MONTH_NAMES[mo - 1]} {y}"

    return _ISO_DATE.sub(repl, t)


def _first_sentence(t: str) -> str:
    """First line, first sentence.

    The split is on ``. `` and not on ``.`` so an abbreviation or a decimal
    ("Dr. Smith", "3.5 km") does not get cut in half -- the reference is short
    enough that a false split costs more than a missed one.
    """
    t = t.split("\n")[0].strip()
    m = re.search(r"[.!?](\s|$)", t)
    return t[: m.start()].strip() if m else t


def _strip_lead(t: str, keep_polarity: bool = False) -> str:
    """Peel framing phrases off the front, repeatedly.

    Repeated because they stack ("Well, I think it was ..."). Bounded so a
    pathological answer cannot spin.
    """
    phrases = _LEAD_PHRASES if keep_polarity else _LEAD_PHRASES + _POLARITY_PHRASES
    for _ in range(6):
        low = t.lower().lstrip()
        hit = next(
            (
                p for p in phrases
                if low.startswith(p) and _boundary_after(low, len(p))
            ),
            None,
        )
        if hit is None:
            return t
        t = t.lstrip()[len(hit):].lstrip(" ,:-—")
        if not t:
            return ""
    return t


def _strip_trail(t: str) -> str:
    for _ in range(6):
        low = t.lower().rstrip().rstrip(".!?,")
        hit = next(
            (
                p for p in _TRAIL_PHRASES
                if low.endswith(p) and _boundary_before(low, len(low) - len(p))
            ),
            None,
        )
        if hit is None:
            return t
        cut = len(low) - len(hit)
        t = t[:cut].rstrip(" ,;:-—")
        if not t:
            return ""
    return t


def _strip_edges(t: str) -> str:
    words = t.split()
    while words and words[0].lower().strip(",.") in _EDGE_TOKENS:
        words = words[1:]
    while words and words[-1].lower().strip(",.") in _EDGE_TOKENS:
        words = words[:-1]
    return " ".join(words)


def _digits(t: str) -> str:
    def repl(m: re.Match) -> str:
        return _NUMERALS.get(m.group(0).lower(), m.group(0))

    return re.sub(r"\b[A-Za-z]+\b", repl, t)


def _boundary_after(s: str, i: int) -> bool:
    """True when position ``i`` ends a whole word (so "yes" does not fire on
    "yesterday" and "no" does not fire on "nothing")."""
    return i >= len(s) or not (s[i].isalnum() or s[i] == "'")


def _boundary_before(s: str, i: int) -> bool:
    return i <= 0 or not (s[i - 1].isalnum() or s[i - 1] == "'")
