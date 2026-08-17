"""LLM-as-judge scoring, as a *separate pass* over saved predictions.

Why this exists
---------------
The token-overlap F1 this project reports punishes paraphrase, and the
measurement that made that undeniable is the ceiling analysis: on the 52.5% of
substantive questions whose retrieval already put **every** annotated evidence
turn into the prompt, F1 is only 0.515.  Retrieval cannot explain that.  Either
the answerer genuinely fails, or the metric is scoring correct answers as
wrong -- and the two call for opposite work.

The observed case that decides it is in the run's own log::

    gold : "Psychology, counseling certification"
    pred : "counseling or working in mental health"
    f1   : 0.222

A second reason is comparability.  Published LoCoMo numbers are not measured on
one scale: groups use different judges and different instructions (one
explicitly tells its judge to "be generous"), which is why a trivial
full-context baseline here scores above a published state of the art.  A
strict, documented judge does not fix the literature, but it gives a number
whose provenance is stated.

Design
------
* The judge sees **question, reference, candidate** -- never the conversation.
  Give it the context and it starts answering the question itself, which
  measures the judge instead of the system.
* Category 5 (adversarial) never reaches the judge: the reference is "no
  information available", so correctness *is* "did it abstain", which is
  decidable by rule.  Sending 22% of the benchmark to an LLM to re-derive a
  regex would only add variance.
* Strict, not generous.  A lenient judge inflates every arm equally and buys
  nothing; a strict one gives a defensible floor.  Both scores are always
  reported side by side -- ``f1`` never gets deleted.
* A separate pass over ``predictions.jsonl`` rather than a step inside the run,
  so twelve conditions already on disk can be re-scored without re-answering a
  single question, and a change to the judge does not invalidate the answers.
* Deterministic and cached by ``(question, gold, prediction)``, so re-running is
  free and two conditions that produced the same answer are graded once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from fgl.evaluation.scorer import is_abstention
from fgl.llm import LLMClient
from fgl.llm.prompts import SYSTEM_JUDGE, PromptLibrary

#: LoCoMo's adversarial category, decided by rule and never by the judge
ADVERSARIAL = 5


@dataclass
class JudgedRow:
    """One prediction, with both scores attached."""

    row: dict
    correct: bool
    reason: str = ""
    by_rule: bool = False

    @property
    def category(self) -> int:
        return int(self.row.get("category", 0))

    @property
    def f1(self) -> float:
        return float(self.row.get("f1", 0.0))


class Judge:
    """Grades saved predictions.  One LLM call per distinct triple, cached."""

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptLibrary,
        max_tokens: int = 200,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.max_tokens = max_tokens

    def judge_row(self, row: dict) -> JudgedRow:
        category = int(row.get("category", 0))
        prediction = row.get("prediction", "") or ""
        gold = row.get("gold", "") or ""

        # adversarial: correctness is abstention, decidable without a model
        if category == ADVERSARIAL:
            return JudgedRow(row, is_abstention(prediction), "regra: abstenção", True)
        # an empty answer is never right, and asking costs a call
        if not prediction.strip():
            return JudgedRow(row, False, "resposta vazia", True)

        payload = self.llm.complete_json(
            self.prompts.render(
                "judge",
                question=row.get("question", ""),
                gold=gold,
                prediction=prediction,
            ),
            system=SYSTEM_JUDGE,
            purpose="eval/judge",
            max_tokens=self.max_tokens,
            default={"correct": False, "reason": "JSON inválido"},
        )
        correct = bool(payload.get("correct", False)) if isinstance(payload, dict) else False
        reason = str(payload.get("reason", ""))[:120] if isinstance(payload, dict) else ""
        return JudgedRow(row, correct, reason)

    def judge_all(
        self,
        rows: Sequence[dict],
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[JudgedRow]:
        out: list[JudgedRow] = []
        for i, row in enumerate(rows):
            if progress and i % 25 == 0:
                progress(i, len(rows))
            out.append(self.judge_row(row))
        if progress:
            progress(len(rows), len(rows))
        return out


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #


def judge_metrics(judged: Sequence[JudgedRow]) -> dict:
    """``judge_*`` counterparts of the F1 block, plus the agreement audit."""
    if not judged:
        return {}
    from fgl.data.locomo import CATEGORY_NAMES

    substantive = [j for j in judged if j.category != ADVERSARIAL]
    per_category: dict[str, dict] = {}
    for cat in sorted({j.category for j in judged}):
        items = [j for j in judged if j.category == cat]
        per_category[CATEGORY_NAMES.get(cat, str(cat))] = {
            "n": len(items),
            "judge": round(float(np.mean([j.correct for j in items])), 4),
            "f1": round(float(np.mean([j.f1 for j in items])), 4),
        }

    block = {
        "judge_micro": round(float(np.mean([j.correct for j in judged])), 4),
        "judge_substantive": round(
            float(np.mean([j.correct for j in substantive])), 4
        )
        if substantive
        else 0.0,
        "judge_macro": round(
            float(np.mean([v["judge"] for v in per_category.values()])), 4
        ),
        "judge_per_category": per_category,
        "judge_calls": sum(1 for j in judged if not j.by_rule),
    }
    block.update(agreement(judged))
    return block


def agreement(judged: Sequence[JudgedRow], threshold: float = 0.5) -> dict:
    """How the judge and token-F1 disagree -- the audit that makes it usable.

    A judge reported without this is just another number of unknown provenance.
    The asymmetry is the interesting part: ``judge_yes_f1_low`` counts answers
    the judge accepts and F1 nearly rejects, i.e. paraphrase, which is the
    hypothesis under test.  ``judge_no_f1_high`` is the opposite failure and
    should stay small; if it does not, the judge is too strict and its number
    should not be trusted before the disagreements are read by hand.
    """
    scored = [j for j in judged if not j.by_rule]
    if not scored:
        return {}
    yes_low = [j for j in scored if j.correct and j.f1 < threshold]
    no_high = [j for j in scored if not j.correct and j.f1 >= threshold]
    same = [j for j in scored if j.correct == (j.f1 >= threshold)]
    return {
        "judge_f1_agreement": round(len(same) / len(scored), 4),
        #: judge accepts, F1 rejects -> paraphrase the token metric was losing
        "judge_yes_f1_low": len(yes_low),
        #: judge rejects, F1 accepts -> the judge is being harsh; read these
        "judge_no_f1_high": len(no_high),
        "judge_n_scored": len(scored),
    }


def disagreements(
    judged: Sequence[JudgedRow], limit: int = 40, threshold: float = 0.5
) -> list[dict]:
    """Cases to read by hand before quoting the judge anywhere.

    Sorted by how far apart the two verdicts are, so the sample is the strongest
    evidence for or against the judge rather than an arbitrary slice.
    """
    scored = [j for j in judged if not j.by_rule]
    gap = lambda j: abs(float(j.correct) - j.f1)  # noqa: E731
    worst = sorted(
        (j for j in scored if j.correct != (j.f1 >= threshold)), key=gap, reverse=True
    )
    return [
        {
            "question": j.row.get("question", ""),
            "gold": j.row.get("gold", ""),
            "prediction": j.row.get("prediction", ""),
            "f1": round(j.f1, 3),
            "judge": j.correct,
            "reason": j.reason,
            "category": j.row.get("category_name", j.category),
        }
        for j in worst[:limit]
    ]


# --------------------------------------------------------------------------- #
# Disk                                                                         #
# --------------------------------------------------------------------------- #


def load_predictions(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_judged(path: str | Path, judged: Sequence[JudgedRow]) -> None:
    """Rewrite ``predictions.jsonl`` with the verdict attached to each row.

    Additive: ``f1`` stays untouched, so the token metric remains reportable and
    the two can always be compared after the fact.
    """
    with Path(path).open("w", encoding="utf-8") as fh:
        for j in judged:
            row = dict(j.row)
            row["judge"] = j.correct
            row["judge_reason"] = j.reason
            row["judge_by_rule"] = j.by_rule
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
