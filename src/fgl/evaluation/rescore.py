"""Re-score saved predictions under answer shaping -- offline, no LLM.

A shaping rule rewrites the prediction *string*, so it can be applied to a
``predictions.jsonl`` that is already on disk and re-scored with the official
scorer. That makes the whole question cheap and, more importantly, **fair**:
every condition already measured -- baselines included -- gets the identical
treatment in the same pass, so the resulting table is a metric fix rather than
a prompt advantage handed to one arm.

Two things this deliberately does *not* do:

* it never rewrites the original ``predictions.jsonl``. Shaped output goes to
  ``predictions_shaped.jsonl`` / ``metrics_shaped.json``, so the run that cost
  6M tokens stays exactly as the model produced it and the comparison can
  always be redone from the raw strings.
* it never scores category 5 differently. Shaping preserves abstention strings
  by construction (:func:`fgl.evaluation.shaping.shape`), and :func:`rescore`
  asserts the adversarial score is unchanged -- if a future rule ever clips a
  "not mentioned", the assertion fires instead of the number silently moving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from fgl.data.locomo import CATEGORY_NAMES
from fgl.evaluation.scorer import score_question
from fgl.evaluation.shaping import DEFAULT_RULES, ShapingRules, shape


@dataclass
class _Q:
    """Minimal stand-in for :class:`fgl.data.locomo.Question`.

    ``score_question`` needs exactly ``category`` and ``answer``; rebuilding it
    from the saved row keeps the category dispatch (multi-answer F1 for
    category 1, the ``;`` split for category 3, the abstention rule for 5)
    byte-identical to a real run instead of re-implementing it here.
    """

    category: int
    answer: str


def _score(row: dict, prediction: str) -> float:
    return score_question(_Q(int(row["category"]), row.get("gold") or ""), prediction)


def rescore_rows(rows: Sequence[dict], rules: ShapingRules) -> dict:
    """Per-category and micro F1 for ``rows`` under ``rules``."""
    per_cat: dict[str, list[float]] = {name: [] for name in CATEGORY_NAMES.values()}
    before: dict[str, list[float]] = {name: [] for name in CATEGORY_NAMES.values()}
    changed = 0
    for row in rows:
        name = CATEGORY_NAMES.get(int(row["category"]), str(row["category"]))
        shaped = shape(row.get("prediction", ""), rules, category=int(row["category"]))
        if shaped != (row.get("prediction") or "").strip():
            changed += 1
        per_cat[name].append(_score(row, shaped))
        before[name].append(float(row.get("f1", 0.0)))

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    adv = CATEGORY_NAMES[5]
    if per_cat[adv] and abs(mean(per_cat[adv]) - mean(before[adv])) > 1e-9:
        raise AssertionError(
            "answer shaping moved the adversarial score: a rule is clipping an "
            "abstention string, which silently turns correct abstentions into "
            "wrong answers. Fix the rule before trusting any other number here."
        )

    flat = [x for xs in per_cat.values() for x in xs]
    flat_before = [x for xs in before.values() for x in xs]
    return {
        "n": len(flat),
        "changed": changed,
        "per_category": {
            k: {"n": len(v), "before": round(mean(before[k]), 4),
                "after": round(mean(v), 4),
                "delta": round(mean(v) - mean(before[k]), 4)}
            for k, v in per_cat.items() if v
        },
        "micro_before": round(mean(flat_before), 4),
        "micro_after": round(mean(flat), 4),
        "micro_delta": round(mean(flat) - mean(flat_before), 4),
    }


def load_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_conditions(results_dir: Path) -> list[tuple[str, Path]]:
    out = []
    for child in sorted(results_dir.iterdir()):
        pred = child / "predictions.jsonl"
        if child.is_dir() and pred.exists():
            out.append((child.name, pred))
    return out


def rescore_dir(
    results_dir: str | Path,
    rules: ShapingRules = DEFAULT_RULES,
    write: bool = False,
) -> dict:
    """Re-score every condition under ``results_dir``."""
    root = Path(results_dir)
    report: dict = {"rules": list(rules.names()), "conditions": {}}
    for name, pred_path in find_conditions(root):
        rows = load_predictions(pred_path)
        result = rescore_rows(rows, rules)
        report["conditions"][name] = result
        if write:
            shaped_rows = []
            for row in rows:
                new = dict(row)
                new["prediction_raw"] = row.get("prediction", "")
                new["prediction"] = shape(
                    row.get("prediction", ""), rules, category=int(row["category"])
                )
                new["f1"] = _score(row, new["prediction"])
                shaped_rows.append(new)
            (pred_path.parent / "predictions_shaped.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in shaped_rows) + "\n",
                encoding="utf-8",
            )
            (pred_path.parent / "metrics_shaped.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return report


def ablate(
    results_dir: str | Path, base: ShapingRules = DEFAULT_RULES
) -> dict:
    """Price each rule twice: alone, and by its absence from the full bundle.

    Both directions, because they answer different questions. "Alone" says
    what a rule can do by itself; "leave-one-out" says what it still adds once
    the others have already fired -- and a rule that scores well alone and zero
    leave-one-out is redundant, not good.
    """
    root = Path(results_dir)
    rows_by_cond = {name: load_predictions(p) for name, p in find_conditions(root)}
    names = [k for k in ShapingRules().__dict__]
    out: dict = {"conditions": {}}
    for cond, rows in rows_by_cond.items():
        entry: dict = {
            "none": rescore_rows(rows, ShapingRules.none())["micro_after"],
            "full": rescore_rows(rows, base)["micro_after"],
            "rules": {},
        }
        for rule in names:
            alone = replace(ShapingRules.none(), **{rule: True})
            without = replace(base, **{rule: False})
            entry["rules"][rule] = {
                "alone": round(
                    rescore_rows(rows, alone)["micro_after"] - entry["none"], 4
                ),
                "leave_one_out": round(
                    entry["full"] - rescore_rows(rows, without)["micro_after"], 4
                ),
            }
        out["conditions"][cond] = entry
    return out


def format_rescore(report: dict) -> str:
    cats = ["single-hop", "multi-hop", "temporal", "open-domain", "adversarial"]
    lines = [f"answer shaping: {', '.join(report['rules']) or '(none)'}", ""]
    lines.append(
        f"{'condition':<20}" + "".join(f"{c[:11]:>13}" for c in cats)
        + f"{'micro':>9}{'delta':>8}"
    )
    for cond, r in report["conditions"].items():
        pc = r["per_category"]
        row = f"{cond:<20}"
        for c in cats:
            row += f"{pc[c]['after']:>13.3f}" if c in pc else f"{'-':>13}"
        row += f"{r['micro_after']:>9.3f}{r['micro_delta']:>+8.3f}"
        lines.append(row)
        deltas = "".join(
            f"{pc[c]['delta']:>+13.3f}" if c in pc else f"{'-':>13}" for c in cats
        )
        lines.append(f"{'  (delta)':<20}{deltas}")
    return "\n".join(lines)


def format_ablation(report: dict) -> str:
    lines = []
    for cond, entry in report["conditions"].items():
        lines.append(f"\n{cond}  raw={entry['none']:.3f}  shaped={entry['full']:.3f}")
        lines.append(f"  {'rule':<16}{'alone':>10}{'leave-one-out':>16}")
        for rule, v in entry["rules"].items():
            lines.append(
                f"  {rule:<16}{v['alone']:>+10.4f}{v['leave_one_out']:>+16.4f}"
            )
    return "\n".join(lines)
