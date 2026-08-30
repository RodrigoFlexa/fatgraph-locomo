"""Gate 1: extraction fidelity (the spec's own first milestone check, and
the gate the MECA post-mortem insisted comes BEFORE any F1 number).

The question this answers is not "how good are the answers" but "does the
memory even contain the turns the questions are about". Every downstream
metric is bounded by it: a question whose evidence turns produced no
proposition cannot be answered from the graph no matter how good the
access algebra is, and measuring F1 first tells you a number without
telling you which half of the system to fix.

Cost is one extraction call per turn and ZERO question-answering calls --
about a third of what benchmarking the same conversation costs, and it is
the number that decides whether benchmarking is worth paying for.

Definition, stated before measuring so it cannot be tuned afterwards:

* an evidence turn is COVERED when at least one proposition was kept from
  it (validated and staged), regardless of whether consolidation later
  promoted it -- promotion is a separate gate, and conflating the two
  would hide which one failed;
* a question is FULLY covered when every one of its evidence turns is,
  and PARTIALLY covered when some but not all are. Multi-hop questions
  need all their hops, so partial coverage is a failure for them and is
  reported separately rather than averaged away.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from fgl.clio.catalog import Catalog
from fgl.clio.config import ClioConfig
from fgl.clio.facade import Clio
from fgl.data.locomo import CATEGORY_NAMES, Conversation
from fgl.llm.client import LLMClient
from fgl.llm.prompts import PromptLibrary
from fgl.retrieval.embeddings import Embedder


@dataclass
class CategoryCoverage:
    category: int
    n_questions: int = 0
    fully_covered: int = 0
    partially_covered: int = 0
    uncovered: int = 0
    evidence_turns: int = 0
    evidence_turns_covered: int = 0

    @property
    def name(self) -> str:
        return CATEGORY_NAMES.get(self.category, str(self.category))

    @property
    def question_coverage(self) -> float:
        return self.fully_covered / self.n_questions if self.n_questions else 0.0

    @property
    def turn_coverage(self) -> float:
        return (
            self.evidence_turns_covered / self.evidence_turns
            if self.evidence_turns
            else 0.0
        )


@dataclass
class GateReport:
    sample_id: str
    n_turns: int = 0
    n_questions: int = 0
    #: evidence turn ids a question cited that the conversation does not
    #: contain. Not a CLIO failure -- a dataset one -- but it has to be
    #: separated out or it silently depresses the coverage number.
    dangling_evidence: list[str] = field(default_factory=list)
    per_category: dict[int, CategoryCoverage] = field(default_factory=dict)
    #: turns that produced nothing, with their text, so the failure is
    #: inspectable instead of just counted
    uncovered_examples: list[tuple[str, str]] = field(default_factory=list)
    turns_with_propositions: int = 0
    raw_items: int = 0
    kept_items: int = 0
    unmapped_items: int = 0
    rejected_items: int = 0
    #: rejections by reason. Reporting only the TOTAL was a real gap: the
    #: coverage number says the memory is missing turns, and this is what
    #: says whether the pipeline threw them away on purpose and under
    #: which rule -- so a rule costing more recall than it buys precision
    #: can be found and reverted instead of suspected.
    rejections_by_reason: dict[str, int] = field(default_factory=dict)
    #: turns where the model DID propose something and none of it
    #: survived. Distinct from a silent turn in the way that matters: the
    #: model saw a fact worth stating and the pipeline refused it.
    turns_fully_suppressed: int = 0
    #: uncovered evidence turns, split by cause
    evidence_turns_fully_suppressed: list[str] = field(default_factory=list)
    evidence_turns_silent: list[str] = field(default_factory=list)
    span_downgrades: int = 0

    @property
    def evidence_turns(self) -> int:
        return sum(c.evidence_turns for c in self.per_category.values())

    @property
    def evidence_turns_covered(self) -> int:
        return sum(c.evidence_turns_covered for c in self.per_category.values())

    @property
    def turn_coverage(self) -> float:
        return (
            self.evidence_turns_covered / self.evidence_turns
            if self.evidence_turns
            else 0.0
        )

    @property
    def fully_covered_questions(self) -> int:
        return sum(c.fully_covered for c in self.per_category.values())

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "n_turns": self.n_turns,
            "n_questions": self.n_questions,
            "evidence_turns": self.evidence_turns,
            "evidence_turns_covered": self.evidence_turns_covered,
            "turn_coverage": round(self.turn_coverage, 4),
            "questions_fully_covered": self.fully_covered_questions,
            "question_coverage": round(self.fully_covered_questions / self.n_questions, 4)
            if self.n_questions
            else 0.0,
            "dangling_evidence": len(self.dangling_evidence),
            "extraction": {
                "raw": self.raw_items,
                "kept": self.kept_items,
                "unmapped": self.unmapped_items,
                "rejected": self.rejected_items,
                "rejections_by_reason": dict(
                    sorted(self.rejections_by_reason.items(), key=lambda kv: -kv[1])
                ),
                "span_downgrades": self.span_downgrades,
                "turns_with_propositions": self.turns_with_propositions,
                "turns_fully_suppressed": self.turns_fully_suppressed,
            },
            "uncovered_evidence": {
                "silent": sorted(self.evidence_turns_silent),
                "suppressed": sorted(self.evidence_turns_fully_suppressed),
            },
            "per_category": {
                c.name: {
                    "n": c.n_questions,
                    "fully_covered": c.fully_covered,
                    "partially_covered": c.partially_covered,
                    "uncovered": c.uncovered,
                    "question_coverage": round(c.question_coverage, 4),
                    "turn_coverage": round(c.turn_coverage, 4),
                }
                for c in sorted(self.per_category.values(), key=lambda c: c.category)
            },
        }


def run_gate1(
    conv: Conversation,
    catalog: Catalog,
    llm: LLMClient,
    embedder: Embedder,
    prompts: PromptLibrary,
    config: ClioConfig | None = None,
    on_turn=None,
    max_examples: int = 12,
) -> tuple[GateReport, Clio]:
    """Ingests every turn of ``conv`` (no question answering at all) and
    reports which evidence turns produced propositions.

    Episodes are keyed by the dataset's own ``dia_id`` so an evidence
    reference resolves to an episode directly -- that is the whole reason
    this can be measured without heuristics.
    """
    clio = Clio(catalog, llm, embedder, prompts, config or ClioConfig.default())
    report = GateReport(sample_id=conv.sample_id)

    covered: dict[str, bool] = {}
    outcomes: dict[str, object] = {}
    for session in conv.sessions:
        ts = datetime.fromisoformat(session.timestamp)
        for turn in session.turns:
            text = turn.text
            if turn.img_caption:
                text = f"{text} [shares {turn.img_caption}]"
            result = clio.ingest(
                text, speaker=turn.speaker, session_id=conv.sample_id, ts=ts
            )
            # the log assigns its own episode id; key coverage by dia_id,
            # which is what the questions cite
            covered[turn.dia_id] = bool(result.propositions)
            outcomes[turn.dia_id] = result
            report.n_turns += 1
            report.raw_items += result.raw_count
            report.kept_items += len(result.propositions)
            report.unmapped_items += len(result.unmapped)
            report.rejected_items += len(result.rejected)
            report.span_downgrades += result.span_downgrades
            for rejection in result.rejected:
                report.rejections_by_reason[rejection.reason] = (
                    report.rejections_by_reason.get(rejection.reason, 0) + 1
                )
            if result.propositions:
                report.turns_with_propositions += 1
            elif result.raw_count > 0:
                report.turns_fully_suppressed += 1
            if on_turn is not None:
                on_turn(turn, result)
        clio.consolidate()

    per_question_uncovered: dict[str, int] = defaultdict(int)
    for q in conv.questions:
        # An adversarial question (category 5) has no evidence to cover by
        # construction -- its gold answer is an abstention. Scoring it here
        # would credit or punish coverage for a question that has none.
        if q.is_adversarial or not q.evidence:
            continue
        cat = report.per_category.setdefault(q.category, CategoryCoverage(q.category))
        cat.n_questions += 1
        report.n_questions += 1
        hits = 0
        total = 0
        for dia_id in q.evidence:
            if dia_id not in covered:
                report.dangling_evidence.append(dia_id)
                continue
            total += 1
            cat.evidence_turns += 1
            if covered[dia_id]:
                hits += 1
                cat.evidence_turns_covered += 1
            else:
                per_question_uncovered[dia_id] += 1
        if total == 0:
            cat.uncovered += 1
        elif hits == total:
            cat.fully_covered += 1
        elif hits == 0:
            cat.uncovered += 1
        else:
            cat.partially_covered += 1

    # Split the uncovered evidence by CAUSE. "The model said nothing" and
    # "the model spoke and the pipeline refused it" are different failures
    # with different fixes -- a prompt problem versus a validation-rule
    # problem -- and one coverage number hides which one you are looking at.
    for dia_id in per_question_uncovered:
        result = outcomes.get(dia_id)
        if result is None:
            continue
        if getattr(result, "raw_count", 0) > 0:
            report.evidence_turns_fully_suppressed.append(dia_id)
        else:
            report.evidence_turns_silent.append(dia_id)

    for dia_id, _ in sorted(per_question_uncovered.items(), key=lambda kv: -kv[1])[
        :max_examples
    ]:
        turn = conv.turn_by_id(dia_id)
        if turn is None:
            continue
        result = outcomes.get(dia_id)
        cause = "silent"
        if result is not None and getattr(result, "raw_count", 0) > 0:
            reasons = sorted({r.reason for r in result.rejected})
            if result.unmapped:
                reasons.append("unmapped")
            cause = "+".join(reasons) or "suppressed"
        # parentheses, not brackets: this line is printed through rich,
        # which reads "[silent]" as a style tag and swallows it whole
        report.uncovered_examples.append((dia_id, f"({cause}) {turn.speaker}: {turn.text}"))

    return report, clio


__all__ = ["GateReport", "CategoryCoverage", "run_gate1"]
