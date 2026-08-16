"""Shared helpers for the analysis notebooks.

One import gets a notebook everything it needs::

    from nbutils import *
    ctx = setup()                 # paths, results, dataframes, plot style
    ctx.f1                        # tidy DataFrame: condition x category x F1

The loading path is the same code the CLI uses (``fgl.evaluation.report``), so a
number in a plot and the same number in ``fgl report`` cannot drift apart.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- make `import fgl` work whether or not the package was pip-installed ------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
os.environ.setdefault("FGL_PROJECT_ROOT", str(_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fgl.evaluation.report import (  # noqa: E402
    CATEGORY_ORDER,
    KEY_PAIRS,
    build_report,
    categories_in,
    load_results,
    order_conditions,
)
from fgl.paths import Paths  # noqa: E402

__all__ = [
    "Context", "setup", "PALETTE", "CATEGORY_ORDER", "KEY_PAIRS", "colors",
    "plot_f1_by_category", "plot_face_length_distribution", "plot_graph_growth",
    "plot_recall", "plot_cost", "plot_deltas", "show", "build_report",
    "load_results", "pd", "np", "plt", "Path",
]

#: Colour-blind-safe (Okabe–Ito), stable per condition across every figure.
PALETTE = {
    "B1-full-context": "#999999",
    "B2-rag-turns": "#E69F00",
    "B3-rag-facts": "#D55E00",
    "G1-fatgraph-min": "#56B4E9",
    "G2-fatgraph-cur": "#0072B2",
    "G3-fatgraph-agent": "#009E73",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "font.size": 9,
            "legend.frameon": False,
        }
    )


def colors(conditions) -> list[str]:
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [
        PALETTE.get(c, fallback[i % len(fallback)]) for i, c in enumerate(conditions)
    ]


# --------------------------------------------------------------------------- #
# Context                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class Context:
    """Everything a notebook needs, already loaded and tidied."""

    paths: Paths
    results_dir: Path
    raw: dict[str, dict] = field(repr=False, default_factory=dict)
    conditions: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    f1: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    overall: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    recall: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    graph: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    faces: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    growth: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    cost: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def ok(self) -> bool:
        return bool(self.raw)

    def predictions(self, condition: str) -> pd.DataFrame:
        """Per-question rows for one condition (from ``predictions.jsonl``)."""
        p = self.results_dir / condition / "predictions.jsonl"
        if not p.exists():
            return pd.DataFrame()
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
        df = pd.json_normalize(rows)
        df.insert(0, "condition", condition)
        return df

    def all_predictions(self) -> pd.DataFrame:
        frames = [self.predictions(c) for c in self.conditions]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def report(self) -> str:
        return build_report(self.raw)


def setup(results_dir: str | Path | None = None, dry: bool = False) -> Context:
    """Load the results and configure the plot style. Call once per notebook."""
    _style()
    paths = Paths.build()
    rd = Path(results_dir) if results_dir else paths.root / ("results-dry" if dry else "results")
    raw = load_results(rd)

    ctx = Context(paths=paths, results_dir=rd, raw=raw)
    if not raw:
        print(
            f"⚠️  no metrics.json under {rd}\n"
            f"   run:  fgl run-all{' --dry-run' if dry else ''}\n"
            f"   or:   setup(dry=True) to look at a smoke run"
        )
        return ctx

    ctx.conditions = order_conditions(raw)
    ctx.categories = categories_in(raw)
    ctx.f1 = _f1_frame(raw, ctx)
    ctx.overall = _overall_frame(raw, ctx)
    ctx.recall = _recall_frame(raw, ctx)
    ctx.graph = _graph_frame(raw, ctx)
    ctx.faces = _faces_frame(raw, ctx)
    ctx.growth = _growth_frame(raw, ctx)
    ctx.cost = _cost_frame(raw, ctx)

    stemmers = {r.get("stemmer") for r in raw.values()}
    print(f"✓ {len(raw)} condition(s) from {rd}: {', '.join(ctx.conditions)}")
    if len(stemmers) > 1:
        print(f"⚠️  mixed stemmers {sorted(s for s in stemmers if s)} — not comparable")
    return ctx


# --------------------------------------------------------------------------- #
# Frames                                                                       #
# --------------------------------------------------------------------------- #


def _f1_frame(raw, ctx) -> pd.DataFrame:
    rows = []
    for c in ctx.conditions:
        for cat, vals in raw[c].get("per_category", {}).items():
            rows.append(
                {"condition": c, "category": cat, "f1": vals.get("f1"),
                 "n": vals.get("n"), "abstention_rate": vals.get("abstention_rate")}
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["category"] = pd.Categorical(df.category, ctx.categories, ordered=True)
        df["condition"] = pd.Categorical(df.condition, ctx.conditions, ordered=True)
    return df


def _overall_frame(raw, ctx) -> pd.DataFrame:
    return pd.DataFrame(
        [{"condition": c, **raw[c].get("overall", {})} for c in ctx.conditions]
    ).set_index("condition")


def _recall_frame(raw, ctx) -> pd.DataFrame:
    rows = []
    for c in ctx.conditions:
        for cat, vals in raw[c].get("per_category", {}).items():
            for k, v in vals.items():
                if k.startswith("recall"):
                    rows.append({"condition": c, "category": cat, "metric": k, "value": v})
    return pd.DataFrame(rows)


def _graph_frame(raw, ctx) -> pd.DataFrame:
    rows = []
    for c in ctx.conditions:
        for conv in raw[c].get("per_conversation", []):
            if "graph" not in conv:
                continue
            rows.append(
                {"condition": c, "sample_id": conv["sample_id"],
                 "n_turns": conv.get("n_turns"), "f1": conv.get("f1"),
                 **{k: v for k, v in conv["graph"].items()
                    if not isinstance(v, (dict, list))},
                 **{f"ingest_{k}": v for k, v in (conv.get("ingest") or {}).items()}}
            )
    return pd.DataFrame(rows)


def _faces_frame(raw, ctx) -> pd.DataFrame:
    rows = []
    for c in ctx.conditions:
        for conv in raw[c].get("per_conversation", []):
            for length, n in (conv.get("graph", {}).get("face_length_hist") or {}).items():
                rows.append(
                    {"condition": c, "sample_id": conv["sample_id"],
                     "length": int(length), "count": int(n)}
                )
    return pd.DataFrame(rows)


def _growth_frame(raw, ctx) -> pd.DataFrame:
    rows = []
    for c in ctx.conditions:
        for conv in raw[c].get("per_conversation", []):
            for s in conv.get("per_session", []):
                rows.append(
                    {"condition": c, "sample_id": conv["sample_id"],
                     **{k: v for k, v in s.items() if not isinstance(v, (dict, list))}}
                )
    return pd.DataFrame(rows)


def _cost_frame(raw, ctx) -> pd.DataFrame:
    rows = []
    for c in ctx.conditions:
        r = raw[c]
        per = r.get("per_conversation", [])
        rows.append(
            {
                "condition": c,
                "calls": r.get("cost", {}).get("calls", 0),
                "cached": r.get("cost", {}).get("cached_calls", 0),
                "tokens_ingest": sum(
                    (x.get("cost_ingest") or {}).get("total_tokens", 0) for x in per
                ),
                "tokens_qa": sum(
                    (x.get("cost_qa") or {}).get("total_tokens", 0) for x in per
                ),
                "tokens_total": r.get("cost", {}).get("total_tokens", 0),
                "wall_seconds": r.get("wall_seconds", 0),
                "mean_context_tokens": r.get("overall", {}).get("mean_context_tokens", 0),
            }
        )
    return pd.DataFrame(rows).set_index("condition")


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #


def _empty(ax, msg="sem dados — rode `fgl run-all`"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_axis_off()
    return ax


def plot_f1_by_category(ctx: Context, ax=None):
    """Grouped bars: F1 per category, one colour per condition."""
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    if ctx.f1.empty:
        return _empty(ax)
    piv = ctx.f1.pivot_table(index="category", columns="condition",
                             values="f1", observed=True)
    piv = piv[[c for c in ctx.conditions if c in piv.columns]]
    piv.plot(kind="bar", ax=ax, color=colors(piv.columns), width=0.82)
    ax.set_ylabel("F1 (métrica oficial LoCoMo)")
    ax.set_xlabel("")
    ax.set_title("F1 por categoria × condição")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return ax


def plot_deltas(ctx: Context, ax=None):
    """The three key comparisons, as signed F1 deltas per category."""
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    if ctx.f1.empty:
        return _empty(ax)
    piv = ctx.f1.pivot_table(index="category", columns="condition",
                             values="f1", observed=True)
    rows = {
        f"{b.split('-')[0]} − {a.split('-')[0]}\n{label}": piv[b] - piv[a]
        for a, b, label in KEY_PAIRS
        if a in piv.columns and b in piv.columns
    }
    if not rows:
        return _empty(ax, "rode os dois lados de cada comparação")
    d = pd.DataFrame(rows)
    d.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Δ F1")
    ax.set_xlabel("")
    ax.set_title("Comparações-chave: o que cada ingrediente adiciona")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return ax


def plot_face_length_distribution(ctx: Context, ax=None, log=True):
    """Face-length histogram — the evidence for COERENCIA.md C9."""
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    if ctx.faces.empty:
        return _empty(ax, "sem condições de fatgraph nos resultados")
    for cond, grp in ctx.faces.groupby("condition", observed=True):
        agg = grp.groupby("length")["count"].sum().sort_index()
        ax.step(agg.index, agg.values, where="mid", label=cond,
                color=PALETTE.get(cond), lw=1.8)
        ax.scatter(agg.index, agg.values, s=12, color=PALETTE.get(cond))
    if log:
        ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("comprimento da face (nº de meias-arestas)")
    ax.set_ylabel("nº de faces")
    ax.set_title("Distribuição de comprimento de faces (todas as conversas)")
    ax.legend(fontsize=8)
    return ax


def plot_graph_growth(ctx: Context, metrics=("V", "E", "F", "genus"), axes=None):
    """V, E, F and genus session by session — how the memory surface evolves."""
    if axes is None:
        _, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2))
    axes = np.atleast_1d(axes)
    if ctx.growth.empty:
        for ax in axes:
            _empty(ax)
        return axes
    for ax, metric in zip(axes, metrics):
        for cond, grp in ctx.growth.groupby("condition", observed=True):
            m = grp.groupby("session")[metric].mean()
            ax.plot(m.index, m.values, marker="o", ms=3, label=cond,
                    color=PALETTE.get(cond))
        ax.set_title(metric)
        ax.set_xlabel("sessão")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    return axes


def plot_recall(ctx: Context, ax=None):
    """recall@k of the annotated evidence turns, averaged over categories."""
    ax = ax or plt.subplots(figsize=(9, 3.8))[1]
    if ctx.recall.empty:
        return _empty(ax)
    ks = sorted(m for m in ctx.recall.metric.unique() if m.startswith("recall@"))
    piv = (
        ctx.recall[ctx.recall.metric.isin(ks)]
        .pivot_table(index="condition", columns="metric", values="value", observed=True)
        .reindex([c for c in ctx.conditions if c in set(ctx.recall.condition)])
    )
    piv.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("recall das evidências")
    ax.set_xlabel("")
    ax.set_title("Recall@k da recuperação")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(fontsize=8)
    return ax


def plot_cost(ctx: Context, ax=None):
    """Token cost split between memory construction and question answering."""
    ax = ax or plt.subplots(figsize=(9, 3.6))[1]
    if ctx.cost.empty or ctx.cost[["tokens_ingest", "tokens_qa"]].values.sum() == 0:
        return _empty(ax)
    ctx.cost[["tokens_ingest", "tokens_qa"]].plot(
        kind="barh", stacked=True, ax=ax, color=["#0072B2", "#E69F00"]
    )
    ax.set_xlabel("tokens de LLM")
    ax.set_ylabel("")
    ax.set_title("Custo por fase: ingestão vs QA")
    ax.legend(["ingestão", "QA"], fontsize=8)
    return ax


def show(*_):
    plt.tight_layout()
    plt.show()
