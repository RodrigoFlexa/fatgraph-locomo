"""Unit tests for :mod:`fgl.memory.ner` -- the zero-LLM extractor L1 ingest
runs on every turn. Covers the three things the module docstring and the
in-code comments claim, each traced back to a measured failure this project
actually hit:

* noun chunks recover ordinary topical nouns spaCy's NER alone would miss
  ("sunset", "pottery class") -- the whole reason noun_chunks is used at all.
* the compound-fallback pass recovers "adoption agencies" from a
  subject-dropped fragment ("Researching adoption agencies -- ...") that
  `doc.noun_chunks` silently drops (turn D2:8, conv-1 -- the exact miss that
  motivated adding the fallback), restricted to genuine multi-token results
  so it does not also readmit the single-word noise (`_GENERIC_NOUNS`
  already blocks some of it) that a looser version of this fallback was
  measured to produce.
* dates are pulled out into ``date_spans`` and never become entity candidates.
"""

from __future__ import annotations

import pytest

from fgl.memory.ner import NonGenerativeExtractor

pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def extractor() -> NonGenerativeExtractor:
    return NonGenerativeExtractor()


def _texts(extractor, text):
    return [c.text for c in extractor.extract(text).candidates]


def test_ordinary_noun_recovered_as_a_candidate(extractor):
    # "sunset" is not a named entity -- only noun-chunk extraction finds it
    # (the modifier "beautiful" stays attached: normalise_span only strips
    # leading determiners/possessives, not adjectives).
    cands = _texts(extractor, "Melanie painted a beautiful sunset yesterday.")
    assert any(c.endswith("sunset") for c in cands), cands


def test_person_name_recovered_via_ner(extractor):
    # Label is whatever spaCy's statistical tagger assigns (occasionally ORG
    # rather than PERSON for a name in an unusual position) -- not something
    # this module controls or depends on beyond the DATE/DROP label filters,
    # so the test only pins down that the span itself survives.
    cands = extractor.extract("Caroline joined a support group last year.").candidates
    texts = [c.text for c in cands]
    assert "caroline" in texts


def test_compound_fallback_recovers_the_measured_miss(extractor):
    # The actual conv-1/D2:8 sentence. Without the fallback, doc.noun_chunks
    # drops "adoption agencies" entirely because spaCy's parser gives
    # "agencies" a degenerate dep_ ("dep") on this subject-dropped fragment,
    # which is outside noun_chunks' fixed extraction-pattern set.
    text = (
        "Researching adoption agencies — it's been a dream to have a "
        "family and give a loving home to kids who need it."
    )
    cands = _texts(extractor, text)
    assert any("adoption" in c and "agenc" in c for c in cands), cands


def test_compound_fallback_does_not_fire_on_single_stray_nouns():
    # Unit-tests _compound_fallback directly rather than end-to-end: a full
    # sentence pulls in unrelated spaCy NER quirks (e.g. it mistags some
    # capitalised interjections as PERSON *before* the fallback ever runs),
    # which would make this test about the wrong thing. Isolated case: strip
    # "adoption" out of the measured D2:8 sentence so "agencies" is a lone
    # noun with no compound child -- the fallback must leave it alone (the
    # hi-lo>=2 restriction is exactly what keeps single stray nouns like this
    # out, see the function's own docstring for the measured 17.3%->0.5% gap
    # that restriction closes).
    import spacy

    from fgl.memory.ner import _compound_fallback

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Researching agencies — it's been a dream to have a family.")
    claimed: set[int] = set()
    for c in doc.noun_chunks:
        claimed.update(range(c.start, c.end))
    for e in doc.ents:
        claimed.update(range(e.start, e.end))
    assert _compound_fallback(doc, claimed) == []


def test_generic_communicative_nouns_are_filtered(extractor):
    # Measured on the real dataset: "thank"/"photo" are the top-degree
    # vertex in every LoCoMo conversation before this filter was added.
    cands = _texts(extractor, "Thanks so much for the photo, it's lovely!")
    assert "thank" not in cands
    assert "photo" not in cands


def test_pronouns_are_never_candidates(extractor):
    cands = _texts(extractor, "I did it because it mattered to me and them.")
    assert cands == [] or all(c not in ("it", "i", "them", "me") for c in cands)


def test_dates_are_isolated_not_mixed_into_candidates(extractor):
    result = extractor.extract("I went hiking last Saturday with my dog.")
    cand_texts = [c.text for c in result.candidates]
    assert "dog" in cand_texts
    assert not any("saturday" in c for c in cand_texts)
    assert any("saturday" in s.lower() for s in result.date_spans)


def test_numeric_labels_are_dropped(extractor):
    cands = extractor.extract("I spent twenty dollars on three books.").candidates
    assert not any(c.label in ("CARDINAL", "MONEY", "QUANTITY") for c in cands)


def test_plural_nouns_are_lemmatised_for_recurrence(extractor):
    # Two turns that should merge onto the same entity vertex at resolution
    # time need matching normalised text first.
    singular = _texts(extractor, "I love my painting.")
    plural = _texts(extractor, "I love my paintings.")
    assert "painting" in singular
    assert "painting" in plural


def test_extract_many_matches_extract_one_at_a_time(extractor):
    texts = [
        "Melanie painted a sunset.",
        "",
        "Caroline researched adoption agencies.",
    ]
    batched = extractor.extract_many(texts)
    assert len(batched) == 3
    assert batched[1].candidates == []
    singles = [extractor.extract(t) for t in texts]
    for b, s in zip(batched, singles):
        assert [c.text for c in b.candidates] == [c.text for c in s.candidates]


def test_empty_text_yields_no_candidates(extractor):
    result = extractor.extract("   ")
    assert result.candidates == []
    assert result.date_spans == []
