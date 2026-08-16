"""Dataset loading, against the real LoCoMo file when it is available."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import PATHS, needs_dataset

from fgl.data.locomo import (
    ABSTAIN_ANSWER,
    TEMPORAL_QUESTION_SUFFIX,
    load_conversations,
    normalize_timestamp,
)

DATA = PATHS.locomo_file
needs_data = needs_dataset


def test_timestamp_parsing():
    assert normalize_timestamp("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00"
    assert normalize_timestamp("12:05 am on 1 January, 2024") == "2024-01-01T00:05:00"
    assert normalize_timestamp("12:30 pm on 3 December, 2022") == "2022-12-03T12:30:00"
    assert normalize_timestamp("garbage") == "garbage"


def test_temporal_questions_get_the_official_suffix():
    from fgl.data.locomo import Question

    q = Question("When did she go?", "May 2023", 2, [])
    assert q.prompt_question().endswith(TEMPORAL_QUESTION_SUFFIX)
    assert Question("Who?", "x", 4, []).prompt_question() == "Who?"


@needs_data
def test_the_whole_official_dataset_loads():
    convs = load_conversations(DATA)
    assert len(convs) == 10
    assert sum(len(c.questions) for c in convs) == 1986, "no question may be dropped"
    assert all(c.sessions for c in convs)
    assert all(c.n_turns > 300 for c in convs)


@needs_data
def test_sessions_are_chronological_and_dated():
    for conv in load_conversations(DATA):
        stamps = [s.timestamp for s in conv.sessions]
        assert stamps == sorted(stamps)
        assert all(s.timestamp and s.timestamp[0].isdigit() for s in conv.sessions)


@needs_data
def test_every_category_is_present_and_adversarial_gold_is_normalised():
    convs = load_conversations(DATA)
    cats = {q.category for c in convs for q in c.questions}
    assert cats == {1, 2, 3, 4, 5}
    adversarial = [q for c in convs for q in c.questions if q.category == 5]
    assert len(adversarial) == 446
    assert all(q.answer == ABSTAIN_ANSWER for q in adversarial), (
        "upstream stores adversarial_answer and usually omits 'answer' entirely"
    )


@needs_data
def test_evidence_ids_resolve_to_real_turns():
    convs = load_conversations(DATA)
    missing = 0
    total = 0
    for conv in convs:
        for q in conv.questions:
            for e in q.evidence:
                total += 1
                if conv.turn_by_id(e) is None:
                    missing += 1
    assert total > 0
    # a handful of upstream annotations point at turns that do not exist;
    # the loader must not crash on them, and they must stay a small minority
    assert missing / total < 0.05, f"{missing}/{total} evidence ids unresolved"


@needs_data
def test_image_captions_are_rendered_like_upstream():
    conv = load_conversations(DATA)[0]
    with_img = [t for t in conv.turns() if t.img_caption]
    assert with_img, "conv-26 shares images"
    assert "[shares " in with_img[0].rendered
