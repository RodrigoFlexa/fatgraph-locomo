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


@dataclass
class _SupportAcc:
    """Support scores split by the only distinction that matters for the cut.

    Not a second scorer either: the oracle never *acts* on a threshold, it
    reports what every threshold would do. That is the whole reform. The old
    objective was ``recall_context`` alone, and on 22.5% of this benchmark
    recall is anticorrelated with the correct behaviour -- retrieving plausible
    context for a question that has no answer is what produces the
    hallucination. A knob that raises recall and lowers the separation below is
    a trade, not an improvement, and this is the instrument that can say so
    before an LLM call is spent.
    """

    substantive: list = field(default_factory=list)
    adversarial: list = field(default_factory=list)
    shapes: dict = field(default_factory=dict)
    shapes_adversarial: dict = field(default_factory=dict)

    def add(self, score: float, shape: str, is_adversarial: bool) -> None:
        (self.adversarial if is_adversarial else self.substantive).append(score)
        self.shapes[shape] = self.shapes.get(shape, 0) + 1
        if is_adversarial:
            self.shapes_adversarial[shape] = self.shapes_adversarial.get(shape, 0) + 1

    def as_dict(self, cfg) -> dict:
        from fgl.retrieval.support import auc, calibrate_threshold, operating_curve

        if not self.substantive and not self.adversarial:
            return {}
        sp = cfg.support
        threshold, source = calibrate_threshold(
            self.substantive + self.adversarial,
            method=sp.method, quantile=sp.quantile, floor=sp.floor, bins=sp.bins,
        )
        curve = operating_curve(self.substantive, self.adversarial)
        at = operating_curve(self.substantive, self.adversarial, [threshold])[0]
        best = max(curve, key=lambda r: r["net_questions"]) if curve else {}
        return {
            "n_substantive": len(self.substantive),
            "n_adversarial": len(self.adversarial),
            # P(an unanswerable question scores below an answerable one). 0.5
            # is a coin flip and means the mechanism has no chain at all.
            "separation_auc": round(
                auc(self.substantive, self.adversarial), 4
            ),
            "threshold": round(threshold, 4),
            "threshold_source": source,
            "at_threshold": at,
            "best_achievable": best,
            "shapes": dict(sorted(self.shapes.items())),
            "shapes_adversarial": dict(sorted(self.shapes_adversarial.items())),
            "curve": curve,
        }


def run_oracle(
    conditions: Sequence[str],
    conversations: Sequence[Conversation],
    root=None,
    force_ingest: bool = False,
    progress=None,
    overrides: Sequence[str] = (),
) -> dict:
    """Retrieve for every question under every ``condition``; answer none.

    Graphs are built through :class:`fgl.pipeline.Runner`, so they land in the
    same ``artifacts/graphs/<condition>/`` a real run would use and the run
    that follows reuses them instead of rebuilding.

    ``overrides`` are ``dotted.key=value`` strings applied to EVERY condition,
    which is what makes an ablation a one-liner instead of a new config file:
    ``-C L3 --set propagation.bridge_hubs=true`` asks "what does the hub rule
    buy?" and answers it in two minutes. They are recorded in the report so a
    number produced under an override cannot later be mistaken for the
    condition's own.

    A caveat that matters: an override touching an INGEST knob changes the
    graph, and graphs are cached per condition. Pass ``force_ingest=True``
    alongside those, or the run silently reads a graph built under a different
    setting.
    """
    report: dict = {
        "conditions": {}, "targets": TARGETS, "overrides": list(overrides),
    }
    say = progress or (lambda *a: None)

    for condition in conditions:
        cfg = Config.load(condition, overrides=overrides)
        # Pinned, not inherited: this command retrieves and never answers, so a
        # misconfigured condition must not be able to open a billable client.
        cfg.llm.provider = "fake"
        cfg.llm.cache_enabled = False
        # The attestation is computed for EVERY slots condition and is never
        # allowed to act here: `abstain=False` keeps the early return off, so
        # recall_context is byte-identical to a run without it and the two
        # halves of the objective are measured on the same retrieval.
        if cfg.ingest.mode == "slots":
            cfg.support.enabled = True
            cfg.support.abstain = False
        runner = Runner(cfg, root=root, llm=build_llm(cfg.llm))
        retriever_cls = _RETRIEVERS[cfg.retrieval.mode]

        per_cat: dict[str, _Acc] = {name: _Acc() for name in CATEGORY_NAMES.values()}
        corner = _CornerAcc()
        support = _SupportAcc()
        calibration: dict = {}
        read_stats: dict = {}

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
            if not read_stats:
                # what the READ looks like on this graph. `bridgeable_frac`
                # near zero means the walk has nowhere to go and the condition
                # has quietly collapsed back to L2 whatever `hops` says -- the
                # kind of silent degeneration that otherwise shows up only as
                # "the numbers did not move".
                for attr in ("connection_stats", "walk_stats"):
                    fn = getattr(retriever, attr, None)
                    if fn is not None:
                        read_stats = fn()
                        break
            for q in conv.questions:
                result = retriever.retrieve(q.prompt_question())
                per_cat[q.category_name].add(
                    evidence_recall(q.evidence, result.turn_ids),
                    result.tokens_used,
                    len(result.facts),
                )
                if cfg.support.enabled and getattr(result, "support_shape", ""):
                    support.add(
                        float(result.support_score),
                        result.support_shape,
                        q.category == 5,
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
            "support": support.as_dict(cfg),
            "calibration": calibration,
            "read": read_stats,
        }
        say(condition, len(conversations), len(conversations), "done")

    return report


def format_oracle(report: dict) -> str:
    """The report as a plain-text table -- no Rich, so it is also loggable."""
    cats = ["single-hop", "multi-hop", "temporal", "open-domain", "adversarial"]
    conds = list(report["conditions"])
    lines: list[str] = []

    if report.get("overrides"):
        lines.append(
            "OVERRIDES APPLIED TO EVERY CONDITION: "
            + ", ".join(report["overrides"])
        )
        lines.append("")
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
    lines.append("how the graph is read (L3/L4: the walk and the connection)")
    for cond in conds:
        rd = report["conditions"][cond].get("read") or {}
        if not rd:
            continue
        lines.append(
            f"{cond:<14} hops={rd.get('hops')} norm={rd.get('normalization')} "
            f"nb={rd.get('non_backtracking')} dense_seed={rd.get('dense_seed')} "
            f"bridgeable={rd.get('bridgeable_frac')} of {rd.get('n_slots')} slots"
        )
        st = rd.get("steiner")
        if st:
            lines.append(
                f"{'':<14} steiner w={st.get('weight')} "
                f"terminals<={st.get('max_terminals')} "
                f"abstain={st.get('abstain')} "
                f"null={(st.get('null') or {}).get('threshold_by_k')}"
            )

    lines.append("")
    lines.append("deterministic abstention (corner test / connection cost)")
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

    lines.extend(_format_support(report, conds))
    return "\n".join(lines)


def _format_support(report: dict, conds: Sequence[str]) -> list[str]:
    """The other half of the objective: does support separate the two halves?

    Printed as a curve rather than a number on purpose. A system that always
    abstains scores 1.000 on adversarial and zero on everything else, so no
    single operating point can be read as a result -- the shape of the trade is
    the result. `net_questions` does the arithmetic that the per-category
    tables invite you to skip: adversarial is 446 questions and multi-hop is
    282, so a mechanism is judged on both columns or on neither.
    """
    lines: list[str] = []
    have = [c for c in conds if report["conditions"][c].get("support")]
    if not have:
        return lines

    lines.append("")
    lines.append("support attestation -- separation between answerable and not")
    lines.append(
        f"{'condition':<14} {'AUC':>6} {'cut':>7} {'source':>9} "
        f"{'caught':>7} {'deleted':>8} {'net Q':>7} {'net micro':>10}"
    )
    for cond in have:
        sp = report["conditions"][cond]["support"]
        at = sp["at_threshold"]
        lines.append(
            f"{cond:<14} {sp['separation_auc']:>6.3f} {sp['threshold']:>7.3f} "
            f"{sp['threshold_source']:>9} {at['adversarial_caught']:>7.1%} "
            f"{at['substantive_deleted']:>8.1%} {at['net_questions']:>7.1f} "
            f"{at['net_micro']:>+10.4f}"
        )

    lines.append("")
    lines.append(
        "  AUC 0.5 = coin flip (the mechanism has no chain). `net Q` = "
        "questions won minus destroyed,"
    )
    lines.append(
        "  against the reference F1 of results/L2d-derived (substantive "
        "0.5263, adversarial 0.5762)."
    )

    for cond in have:
        sp = report["conditions"][cond]["support"]
        best = sp.get("best_achievable") or {}
        lines.append("")
        lines.append(f"{cond} -- best cut on this curve")
        lines.append(
            f"{'':<4}threshold {best.get('threshold')}  caught "
            f"{best.get('adversarial_caught', 0):.1%}  deleted "
            f"{best.get('substantive_deleted', 0):.1%}  net "
            f"{best.get('net_questions')} questions "
            f"({best.get('net_micro', 0):+.4f} micro)"
        )
        lines.append(f"{'':<4}shapes, all questions: {sp['shapes']}")
        lines.append(f"{'':<4}shapes, adversarial:   {sp['shapes_adversarial']}")
        lines.append(
            f"{'':<4}{'cut':>7} {'caught':>8} {'deleted':>8} {'net Q':>8}"
        )
        curve = sp.get("curve") or []
        step = max(1, len(curve) // 12)
        for row in curve[::step]:
            lines.append(
                f"{'':<4}{row['threshold']:>7.3f} "
                f"{row['adversarial_caught']:>8.1%} "
                f"{row['substantive_deleted']:>8.1%} "
                f"{row['net_questions']:>8.1f}"
            )
    return lines
