"""Official LoCoMo scoring, vendored verbatim + the two fixes it needs.

The scoring functions below are copied from ``task_eval/evaluation.py`` of
``snap-research/locomo`` (branch ``code``) so that our numbers are produced by
*their* metric, not a re-implementation:

* ``normalize_answer``  -- lowercase, strip punctuation/articles/commas;
* ``f1_score``          -- token-level F1 over Porter stems;
* ``f1``                -- multi-answer variant used for category 1;
* category rules        -- cat 3 keeps only the part before the first ``;``;
                           cats 2/3/4 use ``f1_score``; cat 1 uses ``f1``;
                           cat 5 is correct iff the prediction contains
                           ``"not mentioned"`` or ``"no information available"``.

Two deviations, both forced and both documented in COERENCIA.md:

* **C7** -- the upstream function reads ``line['answer']`` for *every* category,
  but adversarial items store ``adversarial_answer`` and 444 of the 446 have no
  ``answer`` key at all, so the upstream code raises ``KeyError``.  We fall back
  to the abstention string, which does not change the category-5 rule.
* **C8** -- ``nltk``'s ``PorterStemmer`` is an optional dependency here.  When it
  is unavailable we fall back to an identity stemmer and *record it* in the
  output (``stemmer: "identity"``), because it slightly changes F1.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from fgl.data.locomo import ABSTAIN_ANSWER, CATEGORY_NAMES, Question

# --------------------------------------------------------------------------- #
# Stemming (optional dependency)                                               #
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - depends on optional dependency
    from nltk.stem import PorterStemmer

    _PS = PorterStemmer()
    STEMMER_NAME = "porter"

    def _stem(word: str) -> str:
        return _PS.stem(word)

except Exception:  # pragma: no cover
    STEMMER_NAME = "identity"

    def _stem(word: str) -> str:
        return word


# --------------------------------------------------------------------------- #
# Upstream scoring functions                                                   #
# --------------------------------------------------------------------------- #


def normalize_answer(s: str) -> str:
    """Verbatim from ``task_eval/evaluation.py``."""
    s = (s or "").replace(",", "")

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 over stems -- the LoCoMo headline metric."""
    prediction_tokens = [_stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [_stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi(prediction: str, ground_truth: str) -> float:
    """Multi-answer F1 (category 1): both sides split on commas."""
    predictions = [p.strip() for p in (prediction or "").split(",")]
    ground_truths = [g.strip() for g in (ground_truth or "").split(",")]
    return float(
        np.mean([max(f1_score(p, gt) for p in predictions) for gt in ground_truths])
    )


def is_abstention(prediction: str) -> bool:
    """The upstream category-5 rule."""
    low = (prediction or "").lower()
    return "no information available" in low or "not mentioned" in low


def score_question(question: Question, prediction: str) -> float:
    """Per-question score, following the upstream category dispatch exactly."""
    answer = question.answer
    if question.category == 3:
        answer = answer.split(";")[0].strip()

    if question.category in (2, 3, 4):
        return f1_score(prediction, answer)
    if question.category == 1:
        return f1_multi(prediction, answer)
    if question.category == 5:
        return 1.0 if is_abstention(prediction) else 0.0
    raise ValueError(f"unknown LoCoMo category {question.category!r}")


# --------------------------------------------------------------------------- #
# Retrieval recall                                                             #
# --------------------------------------------------------------------------- #


def evidence_recall(evidence: Sequence[str], retrieved_turn_ids: Iterable[str]) -> float:
    """Fraction of annotated evidence turns present in the retrieved context.

    Same definition as the upstream ``recall_acc`` for turn-level contexts.
    """
    ev = [e for e in evidence if e]
    if not ev:
        return 1.0
    got = set(retrieved_turn_ids)
    return sum(1 for e in ev if e in got) / len(ev)


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class QAOutcome:
    question: str
    category: int
    gold: str
    prediction: str
    f1: float
    evidence: list[str] = field(default_factory=list)
    retrieved_turn_ids: list[str] = field(default_factory=list)
    recall: dict[str, float] = field(default_factory=dict)
    n_facts: int = 0
    n_faces: int = 0
    tokens_context: int = 0
    abstained: bool = False

    # --- sigma expansion audit (see fgl.retrieval.faces) -------------------
    # These columns exist so a results directory can be *proved* to have used
    # (or not used) the expansion, instead of being trusted. On every run that
    # predates it, and on every condition with the flag off, they are
    # false/0/[] -- which is itself the check.
    #: retrieval.sigma_expand was on for this question
    sigma_expand: bool = False
    #: facts in the prompt that came from a sigma orbit
    n_sigma_facts: int = 0
    #: bridging entities whose orbit contributed
    sigma_vertices: list[str] = field(default_factory=list)
    #: tokens spent on those facts
    sigma_tokens: int = 0
    #: evidence turns reached *only* via sigma -- the marginal contribution
    sigma_only_turn_ids: list[str] = field(default_factory=list)
    #: orbit candidates examined, and why they were dropped
    sigma_scanned: int = 0
    sigma_dup: int = 0
    sigma_over_budget: int = 0

    # --- coverage retrieval audit -----------------------------------------
    #: retrieval.face_coverage was on for this question
    face_coverage: bool = False
    #: entities the question was linked to (names, for eyeballing)
    question_entities: list[str] = field(default_factory=list)
    #: best fraction of those entities covered by a single face
    coverage_best: float = 0.0
    #: faces covering 2+ of them -- the actual bridges
    coverage_faces_multi: int = 0
    #: facts retrieved because their trail covered the question's entities
    n_coverage_facts: int = 0
    #: of those, how many came from the geodesic fallback
    n_geodesic_facts: int = 0
    geodesic_len: int = 0
    coverage_tokens: int = 0
    #: evidence turns reached only via coverage/geodesic
    coverage_only_turn_ids: list[str] = field(default_factory=list)

    # --- face-as-a-unit audit (G10) ---------------------------------------
    face_units: bool = False
    #: whole faces that fitted in the budget; 1 means the memory did not split
    face_units_used: int = 0
    #: facts a k-NN over the same facts would not have returned -- the method's
    #: claim in one number
    corroborating_facts: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "category": self.category,
            "category_name": CATEGORY_NAMES.get(self.category, str(self.category)),
            "gold": self.gold,
            "prediction": self.prediction,
            "f1": round(self.f1, 4),
            "evidence": self.evidence,
            "retrieved_turn_ids": self.retrieved_turn_ids,
            "recall": {k: round(v, 4) for k, v in self.recall.items()},
            "n_facts": self.n_facts,
            "n_faces": self.n_faces,
            "tokens_context": self.tokens_context,
            "abstained": self.abstained,
            "sigma_expand": self.sigma_expand,
            "n_sigma_facts": self.n_sigma_facts,
            "sigma_vertices": self.sigma_vertices,
            "sigma_tokens": self.sigma_tokens,
            "sigma_only_turn_ids": self.sigma_only_turn_ids,
            "sigma_scanned": self.sigma_scanned,
            "sigma_dup": self.sigma_dup,
            "sigma_over_budget": self.sigma_over_budget,
            "face_coverage": self.face_coverage,
            "question_entities": self.question_entities,
            "coverage_best": round(self.coverage_best, 4),
            "coverage_faces_multi": self.coverage_faces_multi,
            "n_coverage_facts": self.n_coverage_facts,
            "n_geodesic_facts": self.n_geodesic_facts,
            "geodesic_len": self.geodesic_len,
            "coverage_tokens": self.coverage_tokens,
            "coverage_only_turn_ids": self.coverage_only_turn_ids,
            "face_units": self.face_units,
            "face_units_used": self.face_units_used,
            "corroborating_facts": self.corroborating_facts,
        }


def _per_point(
    outcomes: Sequence[QAOutcome], substantive: Sequence[QAOutcome]
) -> float:
    """Mean context tokens per point of substantive F1 (0.0 if unscored)."""
    if not outcomes or not substantive:
        return 0.0
    f1 = float(np.mean([o.f1 for o in substantive]))
    if f1 < 1e-6:
        return 0.0
    return round(float(np.mean([o.tokens_context for o in outcomes])) / f1, 1)


def aggregate(outcomes: Sequence[QAOutcome]) -> dict:
    """F1 per category + macro/micro aggregates + recall@k per category."""
    by_cat: dict[int, list[QAOutcome]] = {}
    for o in outcomes:
        by_cat.setdefault(o.category, []).append(o)

    per_category: dict[str, dict] = {}
    for cat in sorted(by_cat):
        items = by_cat[cat]
        entry = {
            "n": len(items),
            "f1": round(float(np.mean([o.f1 for o in items])), 4),
            "abstention_rate": round(
                float(np.mean([o.abstained for o in items])), 4
            ),
        }
        recall_keys = sorted({k for o in items for k in o.recall})
        for key in recall_keys:
            vals = [o.recall[key] for o in items if key in o.recall]
            if vals:
                entry[key] = round(float(np.mean(vals)), 4)
        entry.update(_sigma_stats(items))
        entry.update(_coverage_stats(items))
        entry.update(_face_unit_stats(items))
        per_category[CATEGORY_NAMES.get(cat, str(cat))] = entry

    # Category 5 is scored 1.0 exactly when the model abstained, so its "F1" is
    # a pure measure of timidity -- and it is 22% of LoCoMo. A condition that
    # abstains more scores higher overall while answering nothing better, which
    # is why the headline f1_micro must never be quoted on its own. Reported
    # here rather than left for the reader to derive.
    substantive = [o for o in outcomes if o.category != 5]
    overall = {
        "n": len(outcomes),
        "f1_micro": round(float(np.mean([o.f1 for o in outcomes])), 4)
        if outcomes
        else 0.0,
        "f1_macro": round(
            float(np.mean([v["f1"] for v in per_category.values()])), 4
        )
        if per_category
        else 0.0,
        #: f1_micro over everything EXCEPT adversarial: the comparison that
        #: reflects retrieval quality rather than abstention propensity
        "f1_substantive": round(float(np.mean([o.f1 for o in substantive])), 4)
        if substantive
        else 0.0,
        "n_substantive": len(substantive),
        "abstention_rate": round(
            float(np.mean([o.abstained for o in outcomes])), 4
        )
        if outcomes
        else 0.0,
        #: abstention outside category 5, where abstaining is simply a miss
        "abstention_rate_substantive": round(
            float(np.mean([o.abstained for o in substantive])), 4
        )
        if substantive
        else 0.0,
        "mean_context_tokens": round(
            float(np.mean([o.tokens_context for o in outcomes])), 1
        )
        if outcomes
        else 0.0,
        #: context tokens spent per point of substantive F1 -- the axis on which
        #: a memory method competes once quality alone stops separating arms.
        #: 0.0 when the run scored nothing: a ratio over a zero denominator
        #: would print an astronomical number and read like a real measurement.
        "tokens_per_f1_point": _per_point(outcomes, substantive),
    }
    overall.update(_sigma_stats(outcomes))
    overall.update(_coverage_stats(outcomes))
    overall.update(_face_unit_stats(outcomes))
    return {
        "overall": overall,
        "per_category": per_category,
        "stemmer": STEMMER_NAME,
    }


def _sigma_stats(outcomes: Sequence[QAOutcome]) -> dict:
    """Audit block for the sigma expansion.

    Returns ``{}`` when the expansion was off for every question, so metrics
    files of runs that predate it stay shape-compatible.
    """
    on = [o for o in outcomes if o.sigma_expand]
    if not on:
        return {}
    used = [o for o in on if o.n_sigma_facts > 0]
    return {
        "sigma_expand": True,
        # questions where the expansion actually contributed a fact
        "sigma_use_rate": round(len(used) / len(on), 4),
        "sigma_facts_mean": round(float(np.mean([o.n_sigma_facts for o in on])), 2),
        "sigma_tokens_mean": round(float(np.mean([o.sigma_tokens for o in on])), 1),
        # questions where sigma reached an evidence turn nothing else reached:
        # the marginal contribution of the join, not just its activity
        "sigma_evidence_rate": round(
            float(
                np.mean(
                    [
                        any(t in set(o.evidence) for t in o.sigma_only_turn_ids)
                        for o in on
                    ]
                )
            ),
            4,
        ),
        "sigma_bridges_mean": round(
            float(np.mean([len(o.sigma_vertices) for o in on])), 2
        ),
        # diagnóstico de uma expansão inerte: 'scanned' alto com 'dup' alto =
        # a face já cobria a órbita; 'scanned' baixo = as órbitas estão vazias
        "sigma_scanned_mean": round(float(np.mean([o.sigma_scanned for o in on])), 2),
        "sigma_dup_rate": round(
            float(
                np.sum([o.sigma_dup for o in on])
                / max(1, np.sum([o.sigma_scanned for o in on]))
            ),
            4,
        ),
        "sigma_over_budget_rate": round(
            float(
                np.sum([o.sigma_over_budget for o in on])
                / max(1, np.sum([o.sigma_scanned for o in on]))
            ),
            4,
        ),
    }


def _face_unit_stats(outcomes: Sequence[QAOutcome]) -> dict:
    """Audit block for face-as-a-unit retrieval.  ``{}`` when it never ran."""
    on = [o for o in outcomes if o.face_units]
    if not on:
        return {}
    return {
        "face_units": True,
        #: mean whole faces per prompt. Near 1 means the memory never split
        #: into units and the method degenerated into "one big face" -- the
        #: check G5's saturated coverage needed and did not have.
        "face_units_mean": round(float(np.mean([o.face_units_used for o in on])), 2),
        "face_units_single_rate": round(
            float(np.mean([o.face_units_used <= 1 for o in on])), 4
        ),
        #: facts a k-NN over the same facts would not have returned. If this is
        #: ~0 the faces contribute nothing and G10 is B3 with extra steps,
        #: whatever the F1 happens to say.
        "corroborating_facts_mean": round(
            float(np.mean([o.corroborating_facts for o in on])), 2
        ),
        "corroboration_rate": round(
            float(np.mean([o.corroborating_facts / max(o.n_facts, 1) for o in on])), 4
        ),
    }


def _coverage_stats(outcomes: Sequence[QAOutcome]) -> dict:
    """Audit block for the coverage retrieval.  ``{}`` when it never ran."""
    on = [o for o in outcomes if o.face_coverage]
    if not on:
        return {}
    linked = [o for o in on if o.question_entities]
    return {
        "face_coverage": True,
        # the linker is the precondition for everything else: no entities
        # linked, no coverage signal, and the condition degrades to G1
        "coverage_link_rate": round(len(linked) / len(on), 4),
        "coverage_entities_mean": round(
            float(np.mean([len(o.question_entities) for o in on])), 2
        ),
        # how often a single trail covered *all* the entities named
        "coverage_best_mean": round(float(np.mean([o.coverage_best for o in on])), 4),
        "coverage_bridge_rate": round(
            float(np.mean([o.coverage_faces_multi > 0 for o in on])), 4
        ),
        "coverage_use_rate": round(
            float(np.mean([o.n_coverage_facts > 0 for o in on])), 4
        ),
        "coverage_facts_mean": round(
            float(np.mean([o.n_coverage_facts for o in on])), 2
        ),
        "geodesic_rate": round(float(np.mean([o.n_geodesic_facts > 0 for o in on])), 4),
        # marginal contribution: an evidence turn no other mechanism reached
        "coverage_evidence_rate": round(
            float(
                np.mean(
                    [
                        any(t in set(o.evidence) for t in o.coverage_only_turn_ids)
                        for o in on
                    ]
                )
            ),
            4,
        ),
    }
