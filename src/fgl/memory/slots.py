"""The slot model -- condition L2.

Why a new vocabulary of vertices
--------------------------------
L1 asks "which turns mention the same nouns as the question?". Measured on
L1's own predictions and graphs (all 10 conversations, 1986 questions), of
the annotated evidence turns L1 *failed* to retrieve, the fraction that
shared at least one entity vertex with the question is:

    single-hop 13%   multi-hop 7%   temporal 10%   open-domain 5%

and only 0.5% of evidence turns were missing from the graph at all. So the
recall gap is not coverage and not ranking: 87-95% of the misses are turns
the entity-incidence graph offers *no path to*. Inspecting them gives four
recurring bridges, none of which L1 models:

1. **the reply.** "What kind of dance piece did Gina's team perform?" -- the
   evidence turn is `"We just did a contemporary piece called 'Finding
   Freedom.'"`. It carries the value and none of the topic; the topic is in
   the turn above it. A dialogue's atomic semantic unit is the adjacency
   pair, not the turn. -> :class:`EpisodeSegmenter`.
2. **the predicate.** "What did James *adopt* in April 2022?" -> "I *adopted*
   a pup". Questions are framed by a verb; L1 indexes only nouns. -> slot
   kind ``predicate``.
3. **the type.** "What *foods* does Audrey like?" -> "*Roasted Chicken* is
   one of my favorites". The question asks by category, the turn answers with
   an instance. -> slot kind ``type``, WordNet hypernym lift.
4. **the person.** 98.5-99.7% of LoCoMo questions name exactly one of the two
   speakers, and when they do, the evidence turn belongs to that speaker in
   96-100% of cases (multi-hop 244/244, open-domain 72/72). L1 deletes the
   speaker from the graph on purpose, to avoid a hub -- and pays ~24% of
   every context in wrong-speaker turns. A high-degree vertex that is
   *typed* is not a hub, it is a partition. -> slot kind ``actor``.

So a LoCoMo question is a tuple with a hole::

    "What did James adopt in April 2022?"   (James, adopt, ?,       2022-04)
    "When did Caroline go to the group?"    (Caroline, go-to, group, ?     )
    "What foods does Audrey like?"          (Audrey, like, ?:food,   *     )
    adversarial                             (Melanie, paint, ?, ?) -> no support

This module defines that vocabulary. :mod:`fgl.memory.ingest_slots` builds
the graph from it and :mod:`fgl.retrieval.slots` queries it.

Where the ribbon structure earns its place
------------------------------------------
Two objects, and deliberately *not* the face:

* the **orbit** ``sigma(slot)`` is chronological by construction (episodes are
  visited in order, incidences are appended), so "everything this person /
  predicate / concept ever touched, in order" is a list lookup, not a ranking.
* the **corner** -- two consecutive half-edges in ``sigma(episode)`` -- is the
  query. ``sigma`` at an episode vertex is ordered by :data:`SLOT_ORDER`
  (actor, predicate, concept, type, time) and then by document order, so the
  consecutive pairs *are* (who, did-what), (did-what, with-what), (with-what,
  when). A question names two of those and leaves one blank; an adversarial
  question names a pair that has no corner anywhere, which is a deterministic
  abstention signal and needs no LLM (see
  :meth:`fgl.retrieval.slots.SlotRetriever._corner_support`).

Faces are left alone here. On L1's measured graphs one face already swallows
2954 of the half-edges, and for "everything X did" the right object was always
the orbit -- which is the vertex, not the face. Typing the vertices is what
would give a face a chance of meaning something; that is a later experiment,
not the load-bearing claim of this condition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Iterable, Sequence

from fgl.memory.entities import normalize_name

# --------------------------------------------------------------------------- #
# Slot kinds                                                                   #
# --------------------------------------------------------------------------- #

KIND_EPISODE = "episode"
KIND_ACTOR = "actor"
KIND_PREDICATE = "predicate"
KIND_CONCEPT = "concept"
KIND_TYPE = "type"
KIND_TIME = "time"

#: The rotation order of ``sigma`` at an episode vertex, and therefore which
#: pairs of incidences end up adjacent -- i.e. which corners exist. Read the
#: consecutive pairs off this list: (actor, predicate) is "who did what",
#: (predicate, concept) is "did what to what", (type, time) is "what sort of
#: thing, when". Changing this tuple changes the ribbon graph, not just a
#: tie-break, which is why it lives here as data instead of inside a sort key.
SLOT_ORDER: tuple[str, ...] = (
    KIND_ACTOR,
    KIND_PREDICATE,
    KIND_CONCEPT,
    KIND_TYPE,
    KIND_TIME,
)

#: Slot kinds a question can be *answered* from, in decreasing specificity.
#: Used by the corner test to pick which linked slot to demand support for.
SPECIFIC_KINDS: tuple[str, ...] = (KIND_CONCEPT, KIND_PREDICATE, KIND_TYPE)


def slot_vertex_id(kind: str, key: str) -> str:
    """Deterministic vertex id for a slot.

    Deterministic on purpose: two ingests of the same conversation produce
    byte-comparable graphs, and a slot's identity never depends on the order
    it was first seen (unlike ``v17``-style counters).
    """
    return f"{kind}:{key}"


def episode_vertex_id(first_dia_id: str) -> str:
    return f"ep:{first_dia_id}"


# --------------------------------------------------------------------------- #
# Predicates                                                                   #
# --------------------------------------------------------------------------- #

#: Verbs that are grammar or discourse rather than content. spaCy already tags
#: true auxiliaries ``AUX`` (so "have", "be", "will" mostly never reach here);
#: what is left are the dialogue verbs that occur in nearly every turn of a
#: casual conversation and would be a hub with zero discrimination -- the
#: predicate-side analogue of ``ner._GENERIC_NOUNS``. Retrieval *also* applies
#: a degree cut-off, so this list only has to catch the obvious cases.
PREDICATE_STOP = frozenset({
    "be", "have", "do", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "let", "get", "know", "think", "guess", "mean",
    "seem", "sound", "say", "tell", "ask", "mention", "happen", "gon", "wan",
    "'s", "s", "re", "ve", "m", "ll", "d",
})

#: Verbs a *question* uses to frame the query rather than to name the event
#: ("what did X do", "can you tell me"). Dropped from the question side only:
#: on the turn side they are already rare enough not to matter.
QUESTION_PREDICATE_STOP = PREDICATE_STOP | frozenset({
    "use", "answer", "describe", "list", "name", "give", "provide",
})


# --------------------------------------------------------------------------- #
# Concepts                                                                     #
# --------------------------------------------------------------------------- #

#: Nouns that belong to the *question's* framing, not to its content.
#: Measured on L1's own audit column (``question_entities`` over 1986
#: questions): "conversation" was linked to a graph vertex in 194 questions,
#: "date" in 118, "answer" in 33, "type" in 46 -- none of which is a topic.
#: They come from the question template and the temporal suffix, so they are
#: filtered on the question side rather than added to ``ner._GENERIC_NOUNS``
#: (where they would also delete real turn content: a turn saying "our
#: conversation" is at least about something).
#:
#: LEGACY, and named so on purpose. This list is the clearest piece of
#: calibration debt in the condition: it does not encode a property of English
#: questions, it encodes the *template* of one benchmark's question generator,
#: and on any other corpus it would neither help nor hurt -- it would simply be
#: unrelated. :func:`fgl.memory.calibration.derive_question_stop` replaces it
#: with an estimator that contrasts question frequency against memory
#: frequency, and ``tests/test_calibration.py`` pins that the estimator
#: recovers these words on LoCoMo. The list is kept so ``question_stop=
#: "literal"`` reproduces the swept numbers exactly and so the recovery test
#: has something to compare against.
LEGACY_QUESTION_NOUN_STOP = frozenset({
    "conversation", "date", "answer", "question", "type", "kind", "sort",
    "one", "ones", "some", "any", "example", "examples", "reason", "reasons",
    "detail", "details", "information", "activity", "activities", "thing",
    "things", "stuff", "way", "ways", "favorite", "favourite", "plan", "plans",
    "approximate date", "use date",
})

#: Backwards-compatible alias. Retrieval no longer reads this: the active
#: stoplist arrives through :class:`fgl.memory.calibration.Calibration`, whose
#: provenance field records whether it was this list or a derived one.
QUESTION_NOUN_STOP = LEGACY_QUESTION_NOUN_STOP


# --------------------------------------------------------------------------- #
# Types (WordNet hypernym lift)                                                #
# --------------------------------------------------------------------------- #

#: Hypernyms too abstract to discriminate anything. The depth band below
#: already removes ``entity``/``physical_entity``/``abstraction``/``matter``/
#: ``whole``/``object``; these are the ones that survive it and still mean
#: nothing ("artifact" would otherwise sit on half the graph).
TYPE_STOP = frozenset({
    "artifact", "artefact", "instrumentality", "instrumentation", "whole",
    "object", "entity", "physical_entity", "abstraction", "psychological_feature",
    "attribute", "relation", "measure", "unit", "part", "thing", "matter",
    "substance", "causal_agent", "living_thing", "organism", "group",
    "grouping", "state", "event", "act", "action", "activity", "cognition",
    "content", "creation", "representation", "device", "structure",
    "commodity", "consumer_goods", "covering", "solid", "phenomenon",
    "process", "condition", "quality", "property", "person", "individual",
})

#: WordNet ``min_depth`` band kept when lifting a concept to its types.
#: Below 4 everything is ``entity``/``matter``/``whole``; above 12 the
#: "hypernym" is essentially a synonym of the word itself and adds no reach.
#: Verified on real LoCoMo vocabulary: chicken -> food(4), meat(5), bird(6),
#: poultry(7); knee -> body_part(4), joint(5); muffin -> food(4); shoe ->
#: footwear(6); dog -> animal(6).
TYPE_MIN_DEPTH = 4
TYPE_MAX_DEPTH = 12


class _WordNetUnavailable(Exception):
    pass


@lru_cache(maxsize=1)
def _wordnet():
    """WordNet, or ``None`` if the corpus is not installed.

    Never raises: the type channel is additive, so an environment without
    WordNet degrades L2 to "L1 plus episodes, actors and predicates" instead
    of failing the run. :func:`types_available` reports which happened, and
    the ingest records it in the graph so a results directory can be audited
    for it after the fact rather than trusted.
    """
    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("dog", pos=wn.NOUN)  # forces the lazy corpus load
        return wn
    except Exception:  # pragma: no cover - environment-dependent
        return None


def types_available() -> bool:
    return _wordnet() is not None


@lru_cache(maxsize=20_000)
def _lift_one(word: str, max_types: int) -> tuple[str, ...]:
    wn = _wordnet()
    if wn is None or not word:
        return ()
    try:
        synsets = wn.synsets(word, pos=wn.NOUN)
    except Exception:  # pragma: no cover
        return ()
    if not synsets:
        return ()
    # first sense only: WordNet orders senses by frequency, and averaging over
    # every sense of a common word ("turtle" -> reptile AND turtleneck) makes
    # the channel noisier without making it broader in any useful direction.
    out: list[str] = []
    for anc in synsets[0].closure(lambda s: s.hypernyms()):
        depth = anc.min_depth()
        if not (TYPE_MIN_DEPTH <= depth <= TYPE_MAX_DEPTH):
            continue
        lemma = anc.lemmas()[0].name().lower()
        if lemma in TYPE_STOP or lemma == word:
            continue
        if lemma not in out:
            out.append(lemma)
        if len(out) >= max_types:
            break
    return tuple(out)


def lift_types(concept: str, max_types: int = 6) -> list[str]:
    """Hypernyms of ``concept``, most specific first, generic ones removed.

    Only the *head* (last word) of a multi-word concept is lifted: "roasted
    chicken" is a kind of chicken, so its types are chicken's. Lifting every
    token would put "roast" and "chicken" in unrelated branches and dilute the
    channel.
    """
    head = (concept or "").strip().split()
    if not head:
        return []
    return list(_lift_one(head[-1], max_types))


# --------------------------------------------------------------------------- #
# Time                                                                         #
# --------------------------------------------------------------------------- #

_MONTH_NAMES = {
    m.lower(): i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"],
        start=1,
    )
}
_MONTH_ABBR = {name[:3]: num for name, num in _MONTH_NAMES.items()}

_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\w*\.?,?\s+(\d{4})\b", re.IGNORECASE
)
_YEAR_RE = re.compile(r"\b(19|20)(\d{2})\b")

#: "7 May 2023" / "7th of May, 2023" -- day first, the format LoCoMo's own
#: session dates and gold answers use (see fgl.memory.temporal.ResolvedDate).
_DAY_MONTH_YEAR_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?("
    + "|".join(_MONTH_NAMES)
    + r")\w*\.?,?\s+(\d{4})\b",
    re.IGNORECASE,
)
#: "May 7, 2023" / "May 7th 2023" -- month first.
_MONTH_DAY_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\w*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    re.IGNORECASE,
)
#: "2023-05-07"
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# --------------------------------------------------------------------------- #
# Time: a multi-resolution index, not a chosen granularity                     #
# --------------------------------------------------------------------------- #
#
# This used to be one vertex per month, and the justification in the code was
# "month is the granularity LoCoMo questions actually use". That is a true
# observation about one benchmark's question generator and a bad reason for a
# parameter: a productivity assistant asks by day, a legal corpus by year, and
# a per-corpus grain would have to be re-measured every time -- from the
# questions, which is exactly the dependence this method should not have.
#
# The parameter is removed rather than retuned. Every date is indexed at every
# granularity it supports (year, month, day), a question emits every level it
# names, and the level that ends up carrying the match is decided by the
# degree damping that was already there: a year vertex is incident to most of
# the corpus, so ``1/(1+log(deg))`` all but erases it, while a day vertex is
# incident to a handful of episodes and scores nearly full weight. Multi-
# resolution time therefore needs no granularity knob AND no new weighting
# rule -- the existing damping term is exactly the mechanism that picks the
# level, which is the argument for having made damping degree-based in the
# first place.
#
# Cost is bounded: at most three time slots per distinct date instead of one.

#: Coarse to fine. Order matters: it is the order slots are emitted in, hence
#: the order they sit in ``sigma`` at an episode.
TIME_GRANULARITIES: tuple[str, ...] = ("year", "month", "day")

#: Character length of each granularity's key, which is what makes the level
#: recoverable from a bare vertex key ("2023" / "2023-05" / "2023-05-07").
_GRANULARITY_BY_LEN = {4: "year", 7: "month", 10: "day"}


def parse_granularities(spec: str | Sequence[str] | None) -> tuple[str, ...]:
    """``"year,month,day"`` -> ``("year", "month", "day")``, coarse first.

    A string rather than a list because :func:`fgl.config.coerce` types a
    ``--set`` override from the value it replaces, and it has no list case --
    so a comma-separated string is the form that survives the CLI unchanged.
    Unknown names raise here rather than being ignored: silently indexing
    fewer levels than asked for would look like a retrieval regression.
    """
    if spec is None:
        return TIME_GRANULARITIES
    parts = (
        [p.strip().lower() for p in spec.split(",")]
        if isinstance(spec, str)
        else [str(p).strip().lower() for p in spec]
    )
    parts = [p for p in parts if p]
    if not parts:
        return ()
    unknown = [p for p in parts if p not in TIME_GRANULARITIES]
    if unknown:
        raise ValueError(
            f"unknown time granularity {unknown}; valid: {list(TIME_GRANULARITIES)}"
        )
    return tuple(g for g in TIME_GRANULARITIES if g in parts)


def granularity_of(key: str) -> str:
    """Which level a time vertex key belongs to, from the key alone."""
    return _GRANULARITY_BY_LEN.get(len(key or ""), "")


def time_key(dt: datetime | None, granularity: str) -> str:
    """One bucket key: ``2023`` / ``2023-05`` / ``2023-05-07``."""
    if dt is None:
        return ""
    if granularity == "year":
        return f"{dt.year:04d}"
    if granularity == "month":
        return f"{dt.year:04d}-{dt.month:02d}"
    if granularity == "day":
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    return ""


def time_buckets(
    dt: datetime | None, granularities: Sequence[str] = TIME_GRANULARITIES
) -> list[str]:
    """Every bucket key one date belongs to, coarse to fine."""
    if dt is None:
        return []
    out: list[str] = []
    for g in granularities:
        key = time_key(dt, g)
        if key and key not in out:
            out.append(key)
    return out


def month_bucket(dt: datetime | None) -> str:
    """``2023-05``. Kept as the month-only view of :func:`time_buckets`, for
    callers (and tests) that mean the month specifically.
    """
    return time_key(dt, "month")


def question_time_buckets(question: str) -> list[str]:
    """Month/year buckets named literally in a question -- the single-resolution
    view, kept so ``slots.time_granularities=month`` behaves exactly as before.

    Deliberately does *not* try to resolve relative phrases: the question has
    no session date to resolve against, so "last summer" is unanswerable from
    the question text alone -- that resolution belongs on the turn side
    (:mod:`fgl.memory.temporal`), where a base date exists.
    """
    out: list[str] = []
    text = question or ""
    for mon, year in _MONTH_YEAR_RE.findall(text):
        bucket = f"{int(year):04d}-{_MONTH_NAMES[mon.lower()]:02d}"
        if bucket not in out:
            out.append(bucket)
    if not out:
        for c, yy in _YEAR_RE.findall(text):
            year = f"{c}{yy}"
            if year not in out:
                out.append(year)  # year-only prefix, matched by str.startswith
    return out


def question_time_slots(
    question: str, granularities: Sequence[str] = TIME_GRANULARITIES
) -> list[str]:
    """Every time bucket a question names, **finest first**, with backoff.

    "What did James adopt on 7 May 2023?" yields ``["2023-05-07", "2023-05",
    "2023"]``: the day if the question is that precise, and the coarser levels
    behind it so a turn the memory only dated to the month is still reachable.
    Emitting all of them is safe precisely because scoring is degree-damped --
    the year vertex sits on most of the corpus and contributes almost nothing,
    so the backoff costs recall nothing and buys the case where the memory is
    vaguer than the question.

    With ``granularities=("month",)`` this reduces to
    :func:`question_time_buckets`, which is what makes the multi-resolution
    change a strict generalisation rather than a different retriever.
    """
    gran = tuple(granularities)
    if gran == ("month",):
        return question_time_buckets(question)

    text = question or ""
    days: list[tuple[int, int, int]] = []
    months: list[tuple[int, int]] = []
    years: list[int] = []

    for y, m, d in _ISO_DATE_RE.findall(text):
        days.append((int(y), int(m), int(d)))
    for d, mon, y in _DAY_MONTH_YEAR_RE.findall(text):
        days.append((int(y), _MONTH_NAMES[mon.lower()], int(d)))
    for mon, d, y in _MONTH_DAY_YEAR_RE.findall(text):
        days.append((int(y), _MONTH_NAMES[mon.lower()], int(d)))
    for mon, y in _MONTH_YEAR_RE.findall(text):
        months.append((int(y), _MONTH_NAMES[mon.lower()]))
    for c, yy in _YEAR_RE.findall(text):
        years.append(int(f"{c}{yy}"))

    # a day implies its month and year, a month implies its year -- that IS the
    # backoff, and it is generated rather than special-cased per level
    for y, m, _d in days:
        if (y, m) not in months:
            months.append((y, m))
    for y, _m in months:
        if y not in years:
            years.append(y)

    out: list[str] = []
    if "day" in gran:
        for y, m, d in days:
            key = f"{y:04d}-{m:02d}-{d:02d}"
            if key not in out:
                out.append(key)
    if "month" in gran:
        for y, m in months:
            key = f"{y:04d}-{m:02d}"
            if key not in out:
                out.append(key)
    if "year" in gran:
        for y in years:
            key = f"{y:04d}"
            if key not in out:
                out.append(key)
    return out


# --------------------------------------------------------------------------- #
# Episodes                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class Episode:
    """A contiguous run of turns inside one session -- the atomic memory.

    The unit exists because of failure mode (1) in the module docstring: a
    reply turn carries the value and not the topic, so indexing turns
    separately guarantees the topic and its answer are never reachable from
    each other. ``turn_ids`` lists every turn folded in, which is what makes
    the evidence-recall metric see the whole pair when the episode is
    retrieved.
    """

    index: int
    turn_ids: list[str]
    text: str
    speakers: list[str]
    #: the same turns, unjoined and aligned with ``turn_ids``
    turn_texts: list[str] = field(default_factory=list)
    #: content slots contributed per speaker -- the weight of an actor's claim
    #: on this episode (see ``SlotRetriever``: an episode both speakers appear
    #: in is not equally *about* both of them).
    speaker_content: dict[str, int] = field(default_factory=dict)
    #: actor keys named inside the episode without speaking in it ("Cindy",
    #: "Max"). They get a vertex too -- "What people has Maria met?" is
    #: answered by exactly these -- but a smaller weight than a speaker.
    mentioned_actors: list[str] = field(default_factory=list)

    @property
    def first_turn_id(self) -> str:
        return self.turn_ids[0]


@dataclass
class EpisodeSegmenter:
    """Deterministic topic segmentation. No LLM, no model download.

    Three rules, in order:

    * a session boundary always ends an episode (the caller segments one
      session at a time);
    * the first ``min_turns`` turns are glued unconditionally -- that is the
      adjacency pair, the whole point of the unit, and a reply has no content
      overlap with its own question precisely when it is answering it;
    * after that, a turn joins while it shares content with the episode so far
      (Jaccard-style overlap over concept strings), up to ``max_turns``.

    A turn that yields no concepts at all ("Thanks!", "Haha, right?") joins the
    current episode rather than starting one: a contentless episode is a vertex
    nothing can ever link to, and gluing it costs a handful of tokens.
    """

    min_turns: int = 2
    max_turns: int = 6
    cohesion_min: float = 0.10

    def segment(
        self, concept_sets: Sequence[frozenset[str]]
    ) -> list[list[int]]:
        """Group turn indices into episodes given each turn's concept set."""
        episodes: list[list[int]] = []
        current: list[int] = []
        current_concepts: set[str] = set()

        for i, concepts in enumerate(concept_sets):
            if not current:
                current = [i]
                current_concepts = set(concepts)
                continue
            if len(current) >= self.max_turns:
                episodes.append(current)
                current, current_concepts = [i], set(concepts)
                continue
            if len(current) < self.min_turns or not concepts or not current_concepts:
                current.append(i)
                current_concepts |= concepts
                continue
            overlap = len(concepts & current_concepts) / min(
                len(concepts), len(current_concepts)
            )
            if overlap >= self.cohesion_min:
                current.append(i)
                current_concepts |= concepts
            else:
                episodes.append(current)
                current, current_concepts = [i], set(concepts)

        if current:
            episodes.append(current)
        return episodes


# --------------------------------------------------------------------------- #
# Set questions                                                                #
# --------------------------------------------------------------------------- #

#: Words that make a question ask for a *list* rather than a value. Measured
#: motivation: LoCoMo category 1 is scored by ``f1_multi``, which splits the
#: gold on commas and averages the best match per gold ITEM -- so a gold with
#: four items caps a one-item answer at ~0.25 by construction, however correct
#: that one item is. Inspecting the run, most multi-hop predictions return one
#: item: only 20% of them contain a substring scoring 1.0 against the whole
#: gold, against 49% for single-hop. They are incomplete, not wrong.
_SET_CUES = re.compile(
    r"\b(some|all|any|several|various|different|"
    r"kinds?\s+of|types?\s+of|sorts?\s+of|examples?\s+of|list)\b",
    re.IGNORECASE,
)

#: Plural POS tags. Kept as tags rather than a morphology lookup so the check
#: works on any spaCy English pipeline without the `morphologizer`.
_PLURAL_TAGS = frozenset({"NNS", "NNPS"})


def is_set_question(question: str, doc=None) -> bool:
    """Does this question ask for a set of items?

    Deterministic and derived from the question text ALONE -- never from the
    LoCoMo category. That matters: routing on the gold category would make the
    mechanism unusable outside this benchmark and would be reading the label at
    inference time. The cost of that discipline is a few misses on
    category-1 questions phrased in the singular, which is the right trade.

    Two independent signals, either sufficient:

    * an explicit quantifier or enumeration cue ("some", "all", "kinds of");
    * a plural noun inside the interrogative phrase ("What **foods**...",
      "What **books** has Melanie read?"), which is what a "give me the list"
      question looks like when it does not say so.
    """
    q = (question or "").strip()
    if not q:
        return False
    if _SET_CUES.search(q):
        return True
    if doc is None:
        return False
    # plural inside the wh-phrase: the head of the question, not any plural
    # anywhere ("What did Melanie paint for her friends?" is not a set question
    # just because "friends" is plural)
    head = list(doc)[:6]
    if not head or head[0].text.lower() not in ("what", "which", "who", "name"):
        return False
    return any(t.tag_ in _PLURAL_TAGS for t in head)


# --------------------------------------------------------------------------- #
# Actors                                                                       #
# --------------------------------------------------------------------------- #


def actor_key(name: str) -> str:
    """Canonical key for a person. First name only, normalised.

    LoCoMo speakers are referred to by first name throughout ("Melanie",
    "Mel", "Hey Caroline"), so the first token is the identity. Nicknames are
    matched by :func:`match_actor` rather than folded in here, so the vertex
    keeps the name the dataset uses.
    """
    norm = normalize_name(name)
    return norm.split()[0] if norm else ""


def match_actor(text: str, actor_keys: Iterable[str]) -> list[str]:
    """Which of ``actor_keys`` a piece of text names.

    Whole-word match on the canonical key, plus the nickname rule L1 uses:
    a word in the text may be a *prefix* of the canonical name (Mel/Melanie,
    Cal/Calvin, Deb/Deborah).

    Only that direction. The symmetric rule -- a text word that *starts with*
    the key -- reads "sam" out of "the same meal" and "evan" out of "evening",
    and a false actor is not a ranking nuisance here: the actor is applied as
    a multiplicative prior and drives the corner test, so it would demote the
    right episodes and abstain on a supported question.
    """
    words = normalize_name(text).split()
    if not words:
        return []
    found: list[str] = []
    for key in actor_keys:
        if not key:
            continue
        for w in dict.fromkeys(words):
            if w == key or (len(w) >= 3 and key.startswith(w)):
                found.append(key)
                break
    return found
