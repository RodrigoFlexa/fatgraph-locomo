"""Retrieval-only comparison between two memory models -- the kill test.

Answering a question costs an LLM call; *retrieving* for it costs nothing. So
before spending a run on a new memory model, measure the only thing the model
can change on its own: whether the annotated evidence ends up in the context,
at an identical token budget.

This is deliberately not a second scorer. It reports one number per category,
``recall_context``, computed exactly as :mod:`fgl.pipeline` computes it during
a real run -- same retrievers, same budget, same evidence definition -- plus
the two things that decide whether the comparison is fair (how many units each
model fits in the budget, and how many tokens it actually spends) and the
false-positive rate of L2's deterministic abstention.

Why ``recall_context`` and not F1: measured on L1's first full run,
conditional on the evidence being in the context, L1 already matches the
full-context baseline on the substantive categories -- single-hop 0.641 vs
0.653, multi-hop 0.360 vs 0.392, adversarial 0.639 vs 0.630 -- at 5.4% of the
tokens. There is no answering headroom left to find; every remaining point is
retrieval. So retrieval is what this measures, and it measures it without
paying for a single completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from fgl.config import Config
from fgl.data.locomo import CATEGORY_NAMES, Conversation
from fgl.evaluation.scorer import evidence_recall
from fgl.llm import build_llm
from fgl.pipeline import Runner, _RETRIEVERS, _build_retriever

#: Thresholds the new model has to clear for it to be worth an LLM run, set
#: from L1's measured recall_context (0.768 / 0.498 / 0.768 / 0.449). Not a
#: pass mark for the paper -- a stopping rule, so a model that does not move
#: retrieval is abandoned before it costs anything.
TARGETS = {
    "single-hop": 0.90,
    "multi-hop": 0.70,
    "temporal": 0.85,
    "open-domain": 0.65,
}


@dataclass
class _Acc:
    n: int = 0
    recall: float = 0.0
    tokens: float = 0.0
    units: float = 0.0
    empty: int = 0

    def add(self, recall: float, tokens: int, units: int) -> None:
        self.n += 1
        self.recall += recall
        self.tokens += tokens
        self.units += units
        if units == 0:
            self.empty += 1

    def as_dict(self) -> dict:
        n = max(self.n, 1)
        return {
            "n": self.n,
            "recall_context": round(self.recall / n, 4),
            "mean_tokens": round(self.tokens / n, 1),
            "mean_units": round(self.units / n, 2),
            "empty_contexts": self.empty,
        }


@dataclass
class _CornerAcc:
    """Confusion counts for the deterministic abstention signal.

    ``fired`` on an adversarial question is a true positive (it *should*
    abstain); on any other category it is a false positive, and a false
    positive here costs a correct answer -- which is why the flag that acts on
    this signal ships off and is turned on from these two numbers.
    """

    adversarial_total: int = 0
    adversarial_fired: int = 0
    substantive_total: int = 0
    substantive_fired: int = 0
    reasons: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "adversarial_total": self.adversarial_total,
            "adversarial_fired": self.adversarial_fired,
            "true_positive_rate": round(
                self.adversarial_fired / max(self.adversarial_total, 1), 4
            ),
            "substantive_total": self.substantive_total,
            "substantive_fired": self.substantive_fired,
            "false_positive_rate": round(
                self.substantive_fired / max(self.substantive_total, 1), 4
            ),
            "reasons": dict(sorted(self.reasons.items())),
        }


def run_oracle(
    conditions: Sequence[str],
    conversations: Sequence[Conversation],
    root=None,
    force_ingest: bool = False,
    progress=None,
) -> dict:
    """Retrieve for every question under every ``condition``; answer none.

    Graphs are built through :class:`fgl.pipeline.Runner`, so they land in the
    same ``artifacts/graphs/<condition>/`` a real run would use and the run
    that follows reuses them instead of rebuilding.
    """
    report: dict = {"conditions": {}, "targets": TARGETS}
    say = progress or (lambda *a: None)

    for condition in conditions:
        cfg = Config.load(condition)
        # Pinned, not inherited: this command retrieves and never answers, so a
        # misconfigured condition must not be able to open a billable client.
        cfg.llm.provider = "fake"
        cfg.llm.cache_enabled = False
        runner = Runner(cfg, root=root, llm=build_llm(cfg.llm))
        retriever_cls = _RETRIEVERS[cfg.retrieval.mode]

        per_cat: dict[str, _Acc] = {name: _Acc() for name in CATEGORY_NAMES.values()}
        corner = _CornerAcc()
        calibration: dict = {}

        for i, conv in enumerate(conversations):
            say(condition, i, len(conversations), conv.sample_id)
            graph, _ = runner._ingest(conv, force=force_ingest)  # noqa: SLF001
            retriever = _build_retriever(
                retriever_cls,
                graph,
                runner.embedder,
                cfg,
                {s.id: s.date_time_raw for s in conv.sessions},
                conv,
            )
            # Provenance of every threshold this condition used, from the
            # first conversation: a report that quotes recall without saying
            # whether its cut-offs were swept or derived is not auditable.
            if not calibration and hasattr(retriever, "calibration"):
                calibration = retriever.calibration.as_dict()
            for q in conv.questions:
                result = retriever.retrieve(q.prompt_question())
                per_cat[q.category_name].add(
                    evidence_recall(q.evidence, result.turn_ids),
                    result.tokens_used,
                    len(result.facts),
                )
                reason = getattr(result, "abstain_reason", "")
                if q.category == 5:
                    corner.adversarial_total += 1
                    corner.adversarial_fired += bool(reason)
                else:
                    corner.substantive_total += 1
                    corner.substantive_fired += bool(reason)
                if reason:
                    corner.reasons[reason] = corner.reasons.get(reason, 0) + 1

        report["conditions"][cfg.condition] = {
            "retrieval_mode": cfg.retrieval.mode,
            "budget_tokens": cfg.retrieval.budget_tokens,
            "max_facts_in_prompt": cfg.retrieval.max_facts_in_prompt,
            "per_category": {k: v.as_dict() for k, v in per_cat.items() if v.n},
            "corner_test": corner.as_dict(),
            "calibration": calibration,
        }
        say(condition, len(conversations), len(conversations), "done")

    return report


def format_oracle(report: dict) -> str:
    """The report as a plain-text table -- no Rich, so it is also loggable."""
    cats = ["single-hop", "multi-hop", "temporal", "open-domain", "adversarial"]
    conds = list(report["conditions"])
    lines: list[str] = []

    lines.append("recall_context (evidence reaching the prompt), by category")
    lines.append(f"{'condition':<14}" + "".join(f"{c[:11]:>13}" for c in cats))
    for cond in conds:
        pc = report["conditions"][cond]["per_category"]
        row = f"{cond:<14}"
        for c in cats:
            row += f"{pc[c]['recall_context']:>13.3f}" if c in pc else f"{'-':>13}"
        lines.append(row)

    tgt = report.get("targets", {})
    if tgt and conds:
        row = f"{'target':<14}"
        for c in cats:
            row += f"{tgt[c]:>13.2f}" if c in tgt else f"{'-':>13}"
        lines.append(row)

    lines.append("")
    lines.append("cost of the context (must be comparable for the above to mean anything)")
    lines.append(f"{'condition':<14}{'mean tokens':>13}{'mean units':>13}{'empty ctx':>12}")
    for cond in conds:
        pc = report["conditions"][cond]["per_category"]
        n = sum(v["n"] for v in pc.values()) or 1
        tokens = sum(v["mean_tokens"] * v["n"] for v in pc.values()) / n
        units = sum(v["mean_units"] * v["n"] for v in pc.values()) / n
        empty = sum(v["empty_contexts"] for v in pc.values())
        lines.append(f"{cond:<14}{tokens:>13.0f}{units:>13.1f}{empty:>12}")

    lines.append("")
    lines.append("threshold provenance (derived = estimated from the corpus, "
                 "no gold labels)")
    for cond in conds:
        cal = report["conditions"][cond].get("calibration") or {}
        src = cal.get("source") or {}
        if not src:
            continue
        lines.append(
            f"{cond:<14} " + "  ".join(f"{k}={v}" for k, v in sorted(src.items()))
        )
        hub = cal.get("hub_degree_by_kind") or {}
        if hub:
            lines.append(
                f"{'':<14} hub_degree " + ", ".join(f"{k}={v}" for k, v in hub.items())
                + f"   concept_link={cal.get('concept_link_threshold')}"
                + f"   actor_prior={cal.get('actor_prior_floor')}"
                f"/{cal.get('actor_prior_full')}"
                + f"   question_stop={cal.get('n_question_noun_stop')} words"
            )

    lines.append("")
    lines.append("deterministic abstention (corner test)")
    for cond in conds:
        ct = report["conditions"][cond]["corner_test"]
        if not ct["adversarial_fired"] and not ct["substantive_fired"]:
            continue
        lines.append(
            f"{cond:<14} adversarial caught {ct['adversarial_fired']}/"
            f"{ct['adversarial_total']} ({ct['true_positive_rate']:.1%})   "
            f"false positives {ct['substantive_fired']}/{ct['substantive_total']} "
            f"({ct['false_positive_rate']:.1%})   {ct['reasons']}"
        )
    return "\n".join(lines)
