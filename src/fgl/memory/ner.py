"""Non-generative entity/keyword extraction for the bipartite ingest (L1).

No LLM anywhere in this module. A single spaCy model (``en_core_web_sm`` by
default) does two jobs at once, from one parse per turn:

1. **candidate entities** -- named entities plus noun chunks, normalised and
   de-overlapped, in document order. Order matters: the ingest inserts a
   turn's candidates into ``sigma(turn)`` in the order this module returns
   them, so this function *is* choosing sigma-rel (see ``ingest_bipartite.py``).
2. **date/time spans** -- handed to :mod:`fgl.memory.temporal`, kept out of
   the entity candidates entirely. A relative date ("last Saturday") is not a
   bridge between two memories the way "sunset" or "pottery class" is; it is
   context about ONE turn, resolved against that turn's own session date. Two
   turns essentially never share a literal relative-date phrase in a way that
   would make it useful graph structure, so folding dates into the same
   candidate list would only manufacture near-duplicate leaf vertices.

Why noun chunks and not just named entities: spaCy's statistical NER tags
PERSON/GPE/ORG/DATE and the like, but most of what LoCoMo's multi-hop
questions bridge on are ordinary common nouns -- "sunset", "painting",
"pottery class", "charity race". None of those are named entities. Noun-chunk
extraction (which spaCy derives from the dependency parse, so it needs a
model with a parser -- ``en_core_web_sm`` has one, a bare NER-only model would
not) is what actually recovers those.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from fgl.memory.entities import normalize_name

if TYPE_CHECKING:  # pragma: no cover
    import spacy
    from spacy.tokens import Doc, Span

#: spaCy entity labels that are dates/times, handled by the temporal
#: resolver instead of becoming graph vertices.
_DATE_LABELS = frozenset({"DATE", "TIME"})

#: labels that are neither a useful bridge entity nor a date: numbers,
#: money and the like carry no recurring identity across turns.
_DROP_LABELS = frozenset({"CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"})

#: pronouns and determiner-only chunks spaCy's noun_chunks includes ("we",
#: "it", "this", "something") -- degree-1 noise that never recurs as the same
#: entity, so filtering them at the source is cheaper than filtering vertices
#: after the fact.
_PRONOUN_LEMMAS = frozenset({
    "-PRON-", "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "something", "someone", "anything", "anyone",
    "everything", "everyone", "nothing", "no one", "who", "what", "which",
})

_LEADING_DET = re.compile(r"^\s*(a|an|the|my|your|his|her|our|their|its)\s+", re.IGNORECASE)

#: noun chunks that are generic enough to carry no recurring topical identity
#: ("a lot of things happened" is not evidence that two turns are about the
#: same thing). The same "do not index a stopword" reasoning the speaker
#: exclusion and G11's hub skip already apply -- generic nouns are the
#: content-word equivalent of a hub: high potential degree, zero
#: discrimination. Filtered at the source rather than by a degree threshold
#: at retrieval time, because a threshold cannot tell "sunset" (legitimately
#: common) from "thing" (never means anything on its own).
_GENERIC_NOUNS = frozenset({
    "thing", "things", "lot", "lots", "stuff", "way", "ways", "bit", "bits",
    "part", "parts", "time", "times", "something", "someone", "everything",
    "anything", "kind", "kinds", "sort", "sorts",
    # measured on the real LoCoMo dataset (`fgl ingest L1 -n 10`): these are
    # the top-degree vertex in EVERY conversation (74-115 mentions), because
    # LoCoMo turns constantly attach an image and say thanks -- a
    # conversational/meta-communicative act, not a topic, same failure mode
    # as "thing"/"lot" above and caught the same way, after being measured
    # rather than guessed up front.
    "thank", "thanks", "photo", "photos", "picture", "pictures", "pic", "pics",
})

#: relative-date phrases spaCy's NER occasionally misses as DATE depending on
#: surrounding context (measured: "last weekend" tagged DATE in isolation but
#: not always inside a longer turn). Filtered here as a second net so a missed
#: tag does not leak into the entity graph as a spurious low-value vertex --
#: it already failed to resolve via the temporal path in that case anyway.
_DATE_LIKE = re.compile(
    r"^(last|next|this( coming)?|coming)\s+"
    r"(weekend|week|month|year|summer|winter|spring|fall|autumn|"
    r"morning|afternoon|evening|night|"
    r"mon|tues?|wednes|thurs?|fri|satur|sun)(day)?$",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    """One normalised entity candidate, in the order it was mentioned."""

    text: str  # normalised surface form, fed to EntityResolver.resolve()
    raw: str  # original span text, kept for logging/debugging
    label: str  # spaCy ent label, or "NOUN_CHUNK"


@dataclass
class ExtractionResult:
    candidates: list[Candidate] = field(default_factory=list)
    #: raw DATE/TIME span texts, in document order, for the temporal resolver
    date_spans: list[str] = field(default_factory=list)
    #: content-verb lemmas, in document order. Empty unless the extractor was
    #: built with ``extract_verbs=True`` (condition L2), so L1's graphs are
    #: byte-identical with or without this field existing.
    verbs: list[Candidate] = field(default_factory=list)
    #: PERSON spans, in document order. Empty unless ``split_persons=True``;
    #: when it is on they are removed from ``candidates`` instead, so a person
    #: is never both an actor and a topic.
    persons: list[Candidate] = field(default_factory=list)


@lru_cache(maxsize=4)
def _load_model(name: str) -> "spacy.language.Language":
    import spacy

    try:
        return spacy.load(name)
    except OSError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            f"spaCy model {name!r} is not installed. Run: "
            f"python -m spacy download {name}"
        ) from exc


class NonGenerativeExtractor:
    """Wraps one spaCy pipeline; extracts candidates and date spans together.

    Stateless past construction -- safe to share across turns and threads.
    """

    def __init__(self, model_name: str = "en_core_web_sm", max_chunk_words: int = 6,
                 min_chars: int = 3, extract_verbs: bool = False,
                 split_persons: bool = False):
        self.model_name = model_name
        self.max_chunk_words = max_chunk_words
        self.min_chars = min_chars
        #: collect content-verb lemmas as a separate channel (L2's ``predicate``
        #: slot). Off by default: L1 indexes nouns only and must keep doing so
        #: for its numbers to stay comparable with the ones already measured.
        self.extract_verbs = extract_verbs
        #: route PERSON entities to ``persons`` instead of ``candidates`` (L2's
        #: ``actor`` slot). Off by default for the same reason.
        self.split_persons = split_persons
        self._nlp = _load_model(model_name)

    def extract(self, text: str) -> ExtractionResult:
        text = (text or "").strip()
        if not text:
            return ExtractionResult()
        doc = self._nlp(text)
        return self._extract_from_doc(doc)

    def extract_many(self, texts: list[str]) -> list[ExtractionResult]:
        """Batched via ``nlp.pipe`` -- one call, not N Python-level calls."""
        cleaned = [(t or "").strip() for t in texts]
        out: list[ExtractionResult] = [ExtractionResult() for _ in cleaned]
        idx = [i for i, t in enumerate(cleaned) if t]
        if not idx:
            return out
        for i, doc in zip(idx, self._nlp.pipe([cleaned[i] for i in idx])):
            out[i] = self._extract_from_doc(doc)
        return out

    # ------------------------------------------------------------ internal --
    def _extract_from_doc(self, doc: "Doc") -> ExtractionResult:
        date_spans: list[str] = []
        chosen: list["Span"] = []
        persons: list[Candidate] = []
        claimed_tokens: set[int] = set()

        # Named entities first (excluding dates and numeric labels), longest
        # spans win ties over noun chunks below because they are usually the
        # more precise unit ("LGBTQ support group" as an ORG/EVENT-ish span
        # beats the noun-chunk tokenisation of the same words).
        for ent in doc.ents:
            if ent.label_ in _DATE_LABELS:
                date_spans.append(ent.text)
                continue
            if ent.label_ in _DROP_LABELS:
                continue
            if self.split_persons and ent.label_ == "PERSON":
                # claimed, so the noun-chunk pass below cannot re-add the same
                # tokens as a topic: a person is an actor or nothing.
                persons.append(Candidate(text=normalize_name(ent.text),
                                         raw=ent.text, label="PERSON"))
                claimed_tokens.update(range(ent.start, ent.end))
                continue
            chosen.append(ent)
            claimed_tokens.update(range(ent.start, ent.end))

        for chunk in doc.noun_chunks:
            if any(i in claimed_tokens for i in range(chunk.start, chunk.end)):
                continue
            chosen.append(chunk)
            claimed_tokens.update(range(chunk.start, chunk.end))

        for span in _compound_fallback(doc, claimed_tokens):
            chosen.append(span)
            claimed_tokens.update(range(span.start, span.end))

        # document order, not entity-then-chunk order
        chosen.sort(key=lambda s: s.start)

        candidates: list[Candidate] = []
        seen_norm: set[str] = set()
        for span in chosen:
            norm = self._normalise_span(span)
            if not norm or norm in seen_norm:
                continue
            if len(norm) < self.min_chars:
                continue
            if norm in _GENERIC_NOUNS or _DATE_LIKE.match(span.text.strip()):
                continue
            seen_norm.add(norm)
            label = span.label_ if hasattr(span, "label_") and span.label_ else "NOUN_CHUNK"
            candidates.append(Candidate(text=norm, raw=span.text, label=label))

        verbs = self._extract_verbs(doc) if self.extract_verbs else []
        return ExtractionResult(
            candidates=candidates, date_spans=date_spans,
            verbs=verbs, persons=persons,
        )

    def _extract_verbs(self, doc: "Doc") -> list[Candidate]:
        """Content-verb lemmas, de-duplicated, in document order.

        ``pos_ == "VERB"`` already excludes what spaCy tags ``AUX`` ("is",
        "was", "will"), so the stop list only has to remove the dialogue verbs
        that survive that ("know", "think", "say") -- see
        ``fgl.memory.slots.PREDICATE_STOP`` for why they are removed at all.
        """
        from fgl.memory.slots import PREDICATE_STOP

        out: list[Candidate] = []
        seen: set[str] = set()
        for tok in doc:
            if tok.pos_ != "VERB":
                continue
            lemma = tok.lemma_.lower().strip()
            if not lemma or lemma in seen or lemma in PREDICATE_STOP:
                continue
            if len(lemma) < 2 or not lemma.isalpha():
                continue
            seen.add(lemma)
            out.append(Candidate(text=lemma, raw=tok.text, label="VERB"))
        return out

    def _normalise_span(self, span: "Span") -> str:
        tokens = list(span)
        # drop a leading determiner/possessive token spaCy's parser attaches
        # to the chunk ("a painting" -> "painting")
        while tokens and tokens[0].pos_ in ("DET", "PRON") and tokens[0].dep_ in (
            "det", "poss", "predet",
        ):
            tokens = tokens[1:]
        if not tokens:
            return ""
        if len(tokens) > self.max_chunk_words:
            tokens = tokens[: self.max_chunk_words]

        # a bare pronoun chunk ("it", "something") slips through the det/poss
        # strip above when the pronoun IS the head
        if len(tokens) == 1 and (
            tokens[0].lemma_.lower() in _PRONOUN_LEMMAS
            or tokens[0].pos_ == "PRON"
        ):
            return ""

        parts = []
        for tok in tokens:
            if tok.is_punct or tok.is_space:
                continue
            # lemmatise common nouns (merges "paintings"/"painting"); leave
            # proper nouns, adjectives etc. as written, so names are not
            # mangled by the lemmatiser ("Caroline" must stay "Caroline")
            if tok.pos_ == "NOUN":
                parts.append(tok.lemma_.lower())
            else:
                parts.append(tok.text.lower())
        norm = " ".join(p for p in parts if p)
        norm = _strip_accents(norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm


def _compound_span(tok: "Span") -> tuple[int, int]:
    """Token index range for ``tok`` plus every ``compound``-dependent child,
    walking the dependency tree rather than assuming linear adjacency alone.
    Returns a half-open range; non-contiguous results collapse to the head
    token by itself (caller then requires a real multi-token result).
    """
    idxs = {tok.i}
    stack = [tok]
    while stack:
        t = stack.pop()
        for c in t.children:
            if c.dep_ == "compound" and c.i not in idxs:
                idxs.add(c.i)
                stack.append(c)
    lo, hi = min(idxs), max(idxs)
    if hi - lo + 1 != len(idxs):
        return tok.i, tok.i + 1
    return lo, hi + 1


def _compound_fallback(doc: "Doc", claimed_tokens: set[int]) -> list["Span"]:
    """Recovers noun compounds that ``doc.noun_chunks`` silently drops.

    Measured gap (turn D2:8, conv-1: "Researching adoption agencies -- it's
    been a dream..."): spaCy's noun_chunks generator only yields a span for a
    token whose dep_ label is in its fixed extraction-pattern set (nsubj,
    dobj, pobj, attr, appos, ...). LoCoMo's turns are casual, subject-dropped
    fragments ("Researching X", not "I am researching X"), and the parser
    regularly produces a degenerate dep_ for the sentence's real object under
    that register ("agencies" came back dep_="dep", attached to an unrelated
    ROOT) -- noun_chunks then drops it silently, taking the entire phrase
    with it. The dependency EDGES between the noun and its own compound
    modifiers stay correct even when the higher-level attachment is garbage,
    so walking `compound` children directly recovers the phrase without
    depending on the parse being clean above the noun phrase itself.

    Restricted to genuine multi-token results (a head noun with at least one
    compound child): measured on 2760 real LoCoMo turns, allowing single-word
    catches through this path gave 251/1451 turns (17.3%) a spurious hit
    (mistagged interjections -- "Glad", "Aww", "Awesome" -- and vocatives
    already handled by the speaker-mention filter). Requiring a real compound
    edge drops that to 13/2760 (0.5%), and what remains is real ("adoption
    agencies", "ice cream", "coconut milk") plus a handful of
    interjection+name vocatives ("Thanks Nate") left for the caller's
    speaker-mention filter to catch per-word, not per-phrase.
    """
    out: list["Span"] = []
    for tok in doc:
        if tok.i in claimed_tokens:
            continue
        if tok.pos_ not in ("NOUN", "PROPN") or tok.dep_ == "compound":
            continue
        lo, hi = _compound_span(tok)
        if hi - lo < 2:
            continue  # single token: not what this fallback is for
        if any(i in claimed_tokens for i in range(lo, hi)):
            continue
        out.append(doc[lo:hi])
        claimed_tokens.update(range(lo, hi))
    return out


def _strip_accents(s: str) -> str:
    n = unicodedata.normalize("NFKD", s)
    return "".join(c for c in n if not unicodedata.combining(c))
