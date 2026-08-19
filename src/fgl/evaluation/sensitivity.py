"""One-at-a-time sensitivity sweep over the zero-LLM oracle.

The question this answers
-------------------------
``hub_degree: 60``. ``sibling_frac: 0.2``. ``concept_link_threshold: 0.75``.
Reporting the value that won a sweep, and only that value, tells a reader
nothing about whether the method depends on it. Two very different situations
produce the same config line:

* the metric is **flat** across the swept range and 60 was picked off a
  plateau. Then the number is not load-bearing, the tuning bought nothing, and
  saying so is the strongest defence the method has;
* the metric has a **peak** at 60 and falls away on both sides. Then the number
  IS the result, it was found by looking at annotated data, and it is a
  fragility that has to be declared -- and re-measured on any other corpus.

Nobody can tell which from a config file. This module measures it: it re-runs
``fgl slots-oracle`` (retrieval only, no completion, no cost) across a grid of
values for one knob at a time and reports, per knob, the curve plus three
numbers that summarise it:

``sensitivity``   ``(best - worst) / best`` over the swept range -- how much the
                  knob can move the metric at all;
``plateau_frac``  share of swept values within ``tol`` of the best -- how wide
                  the good region is;
``tuning_gain``   ``shipped - median(curve)`` -- how much the shipped value beats
                  a value picked blind from the same range. **This is the
                  calibration debt of that knob, in points of recall.** Summed
                  over the grid it is the honest answer to "how much of this
                  number came from having the annotations?".

A knob whose ``tuning_gain`` is ~0 was not really tuned, whatever the sweep
history says. A knob with a large ``tuning_gain`` and a narrow plateau is a
finding to report, not a line to quietly keep.

Cost
----
Zero LLM calls, like the oracle it wraps. Retrieval-only knobs reuse one built
graph and one built retriever per conversation and only recalibrate between
values, so a 7-point sweep costs 7 retrieval passes and 1 ingest, not 7 of
each. Knobs that change the graph (:data:`INGEST_KNOBS`) rebuild it in memory
per value and never touch the condition's cached graph directory -- a sweep
must not be able to leave a differently-built graph behind for the next real
run to load.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from html import escape
from typing import Any, Mapping, Optional, Sequence

from fgl.config import Config
from fgl.data.locomo import Conversation
from fgl.evaluation.scorer import evidence_recall
from fgl.llm import build_llm
from fgl.pipeline import Runner, _RETRIEVERS, _build_retriever

#: Knobs whose value changes the GRAPH, not just how it is read. They need a
#: fresh ingest per value; everything else can reuse one build. Getting this
#: set wrong in the safe direction costs time; getting it wrong in the unsafe
#: direction silently sweeps a knob that never took effect, so the default for
#: an unlisted knob is "retrieval-only" and this list is checked against the
#: ingest path in ``tests/test_sensitivity.py``.
INGEST_KNOBS: frozenset[str] = frozenset({
    "slots.episode_min_turns",
    "slots.episode_max_turns",
    "slots.episode_cohesion",
    "slots.max_concepts",
    "slots.max_predicates",
    "slots.max_types",
    "slots.max_types_per_concept",
    "slots.lift_types",
    "slots.resolve_temporal",
    "slots.time_granularities",
    "slots.ner_model",
    "slots.max_chunk_words",
    "slots.min_concept_chars",
    "entities.match_threshold",
    "entities.llm_threshold",
})

#: The knobs the calibration critique actually names, with a range wide enough
#: that a plateau is distinguishable from a peak. Deliberately symmetric around
#: the shipped value where that is meaningful: a grid that only explores one
#: side of the current setting cannot tell you that you are on a cliff.
DEFAULT_GRID: dict[str, list[Any]] = {
    "slots.slot_damping": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "slots.hub_degree": [15, 30, 45, 60, 90, 150, 300],
    "slots.sibling_frac": [0.0, 0.1, 0.2, 0.3, 0.5, 0.8],
    "slots.concept_link_threshold": [0.60, 0.675, 0.75, 0.80, 0.85, 0.90],
    "slots.actor_prior_floor": [0.0, 0.15, 0.35, 0.5, 0.7, 1.0],
    "slots.actor_prior_full": [0.25, 0.35, 0.5, 0.65, 0.8],
    "slots.set_orbit_boost": [0.0, 1.0, 2.0, 3.0, 5.0],
    "slots.concept_weight": [0.75, 1.0, 1.5, 2.0, 3.0],
    "slots.predicate_weight": [0.0, 0.6, 1.2, 1.8, 2.4],
    "slots.type_weight": [0.0, 0.3, 0.6, 1.2, 2.0],
    "slots.time_weight": [0.0, 0.4, 0.8, 1.6],
    "slots.mention_weight": [0.0, 0.25, 0.5, 1.0],
    "slots.episode_max_turns": [2, 3, 4, 6, 8],
    "slots.episode_cohesion": [0.0, 0.05, 0.15, 0.3, 0.5],
    # --- L3 / L4. Inert on a condition that does not use them, and the
    # sweep reports them as perfectly flat, which is the correct answer.
    #
    # `propagation.hops` is the one curve in this grid whose leftmost point is
    # a PUBLISHED NUMBER: at hops=1 with normalization=none the operator is
    # L2's structural read exactly. Sweeping it is therefore not "tuning a new
    # knob", it is measuring the size of the generalisation.
    "propagation.hops": [1, 2, 3],
    "propagation.decay": [0.25, 0.5, 0.75, 1.0],
    "propagation.normalization": ["none", "rw", "sym"],
    "propagation.non_backtracking": [False, True],
    "propagation.dense_seed": [0.0, 0.25, 0.5, 1.0],
    "propagation.bridge_hubs": [False, True],
    "steiner.weight": [0.0, 0.75, 1.5, 3.0],
    "steiner.max_terminals": [2, 3, 4, 6],
    "steiner.max_cost": [6.0, 9.0, 12.0, 18.0],
    "steiner.abstain_quantile": [0.80, 0.90, 0.95, 0.99],
}

#: Knobs that only exist for L3/L4. Sweeping them on L2 is a no-op, so
#: `fgl slots-sweep` drops them unless the condition can read them -- a flat
#: curve for a knob that was never consulted is the most misleading output
#: this tool can produce.
MODE_KNOBS: dict[str, tuple[str, ...]] = {
    "propagation": ("propagation", "unified"),
    "steiner": ("unified",),
}

#: Categories, in the order they are reported and coloured.
CATEGORIES: tuple[str, ...] = (
    "single-hop", "multi-hop", "temporal", "open-domain", "adversarial",
)

#: Relative tolerance defining the plateau: a value counts as "as good as the
#: best" when it is within this fraction of it. 1% of a recall around 0.7 is
#: ~0.007, which is below the resolution anyone would act on.
PLATEAU_TOL = 0.01


# --------------------------------------------------------------------------- #
# Result types                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class Point:
    """One swept value and what retrieval did at it."""

    value: Any
    overall: float = 0.0
    per_category: dict[str, float] = field(default_factory=dict)
    mean_tokens: float = 0.0
    mean_units: float = 0.0
    is_shipped: bool = False

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "overall": round(self.overall, 4),
            "per_category": {k: round(v, 4) for k, v in self.per_category.items()},
            "mean_tokens": round(self.mean_tokens, 1),
            "mean_units": round(self.mean_units, 2),
            "is_shipped": self.is_shipped,
        }


@dataclass
class Curve:
    """The sweep of one knob, plus the three numbers that summarise it."""

    knob: str
    points: list[Point] = field(default_factory=list)
    shipped_value: Any = None
    rebuilt_graph: bool = False

    # ------------------------------------------------------------ summary --
    @property
    def overall(self) -> list[float]:
        return [p.overall for p in self.points]

    @property
    def best(self) -> float:
        return max(self.overall) if self.points else 0.0

    @property
    def worst(self) -> float:
        return min(self.overall) if self.points else 0.0

    @property
    def best_value(self) -> Any:
        return max(self.points, key=lambda p: p.overall).value if self.points else None

    @property
    def shipped(self) -> float:
        for p in self.points:
            if p.is_shipped:
                return p.overall
        return 0.0

    @property
    def sensitivity(self) -> float:
        """How much the knob can move the metric at all, as a share of the best."""
        return (self.best - self.worst) / self.best if self.best else 0.0

    def plateau_frac(self, tol: float = PLATEAU_TOL) -> float:
        """Share of swept values within ``tol`` (relative) of the best."""
        if not self.points:
            return 0.0
        cut = self.best * (1.0 - tol)
        return sum(1 for v in self.overall if v >= cut) / len(self.points)

    @property
    def tuning_gain(self) -> float:
        """Shipped value minus the median of the swept range.

        The calibration debt of this knob in points of recall: what having the
        annotations bought over picking a value blind from the same range. It
        can be negative, which is worth knowing too -- it means the sweep
        settled somewhere a coin flip would have beaten.
        """
        if not self.points:
            return 0.0
        return self.shipped - statistics.median(self.overall)

    @property
    def regret(self) -> float:
        """How much the shipped value leaves on the table against the best here."""
        return self.best - self.shipped

    def verdict(self, tol: float = PLATEAU_TOL) -> str:
        """``flat`` / ``shallow`` / ``peaked`` / ``cliff``.

        ``flat``     the knob barely moves the metric; reporting its optimum is
                     not overfitting because there is nothing to overfit to.
        ``shallow``  it moves it a little and the good region is wide.
        ``peaked``   it moves it a lot and the good region is narrow: the value
                     IS a result obtained from annotated data and has to be
                     declared as such.
        ``cliff``    peaked AND the shipped value sits at the edge of the
                     plateau, so a small corpus shift can fall off it.
        """
        if self.sensitivity < 0.02:
            return "flat"
        if self.plateau_frac(tol) >= 0.5:
            return "shallow"
        # inside the plateau but adjacent to a value outside it -> a cliff
        cut = self.best * (1.0 - tol)
        idx = next((i for i, p in enumerate(self.points) if p.is_shipped), None)
        if idx is not None:
            neighbours = [
                self.points[j].overall
                for j in (idx - 1, idx + 1)
                if 0 <= j < len(self.points)
            ]
            if self.shipped >= cut and any(v < self.best * 0.95 for v in neighbours):
                return "cliff"
        return "peaked"

    def as_dict(self, tol: float = PLATEAU_TOL) -> dict:
        return {
            "knob": self.knob,
            "shipped_value": self.shipped_value,
            "rebuilt_graph": self.rebuilt_graph,
            "points": [p.as_dict() for p in self.points],
            "best": round(self.best, 4),
            "best_value": self.best_value,
            "worst": round(self.worst, 4),
            "shipped": round(self.shipped, 4),
            "sensitivity": round(self.sensitivity, 4),
            "plateau_frac": round(self.plateau_frac(tol), 4),
            "tuning_gain": round(self.tuning_gain, 4),
            "regret": round(self.regret, 4),
            "verdict": self.verdict(tol),
        }


# --------------------------------------------------------------------------- #
# The sweep                                                                    #
# --------------------------------------------------------------------------- #


def _evaluate(retriever, conv: Conversation) -> tuple[dict[str, list[float]], list[int], list[int]]:
    """Retrieve for every question of one conversation; score nothing else."""
    per_cat: dict[str, list[float]] = {}
    tokens: list[int] = []
    units: list[int] = []
    for q in conv.questions:
        result = retriever.retrieve(q.prompt_question())
        per_cat.setdefault(q.category_name, []).append(
            evidence_recall(q.evidence, result.turn_ids)
        )
        tokens.append(result.tokens_used)
        units.append(len(result.facts))
    return per_cat, tokens, units


def _fold(
    acc_cat: dict[str, list[float]],
    acc_tokens: list[int],
    acc_units: list[int],
    point: Point,
) -> None:
    """Turn accumulated per-question numbers into one :class:`Point`."""
    point.per_category = {
        cat: sum(vals) / len(vals) for cat, vals in acc_cat.items() if vals
    }
    n = sum(len(v) for v in acc_cat.values())
    point.overall = (
        sum(sum(v) for v in acc_cat.values()) / n if n else 0.0
    )
    point.mean_tokens = sum(acc_tokens) / len(acc_tokens) if acc_tokens else 0.0
    point.mean_units = sum(acc_units) / len(acc_units) if acc_units else 0.0


def sweep(
    condition: str,
    conversations: Sequence[Conversation],
    grid: Optional[Mapping[str, Sequence[Any]]] = None,
    root=None,
    force_ingest: bool = False,
    tol: float = PLATEAU_TOL,
    progress=None,
) -> dict:
    """Sweep each knob in ``grid`` one at a time, holding the rest at their
    condition values. Returns a JSON-serialisable report.

    The shipped value of each knob is inserted into its grid if the grid does
    not already contain it, so ``tuning_gain`` and ``regret`` always have a
    reference point and the reader can see where the config actually sits on
    the curve.
    """
    grid = dict(grid or DEFAULT_GRID)
    say = progress or (lambda *a: None)

    base = Config.load(condition, root=root)
    # Drop knobs this condition's retriever never reads: a knob that was not
    # consulted produces a perfectly flat curve, and a flat curve is this
    # tool's way of saying "not a result" -- so reporting one for a knob that
    # simply does not apply would be a lie in the tool's own vocabulary.
    dropped = [
        k for k in grid
        if k.split(".")[0] in MODE_KNOBS
        and base.retrieval.mode not in MODE_KNOBS[k.split(".")[0]]
    ]
    for k in dropped:
        grid.pop(k)
    if dropped:
        say("sweep", 0, 1,
            f"skipping {len(dropped)} knob(s) {base.retrieval.mode!r} does not read")
    base.llm.provider = "fake"
    base.llm.cache_enabled = False

    # Build (and cache to disk) the condition's own graphs once, through the
    # normal Runner, so a sweep never invents a graph the real run would not use.
    runner = Runner(base, root=root, llm=build_llm(base.llm))
    retriever_cls = _RETRIEVERS[base.retrieval.mode]
    graphs: dict[str, Any] = {}
    for conv in conversations:
        graph, _ = runner._ingest(conv, force=force_ingest)  # noqa: SLF001
        graphs[conv.sample_id] = graph

    baseline_point = Point(value="(condition)", is_shipped=True)
    acc: dict[str, list[float]] = {}
    tok: list[int] = []
    uni: list[int] = []
    for conv in conversations:
        r = _build_retriever(
            retriever_cls, graphs[conv.sample_id], runner.embedder, base,
            {s.id: s.date_time_raw for s in conv.sessions}, conv,
        )
        pc, t, u = _evaluate(r, conv)
        for k, v in pc.items():
            acc.setdefault(k, []).extend(v)
        tok += t
        uni += u
    _fold(acc, tok, uni, baseline_point)

    curves: dict[str, Curve] = {}
    total = sum(len(list(v)) for v in grid.values())
    done = 0

    for knob, raw_values in grid.items():
        shipped = base.get(knob)
        values = list(raw_values)
        if shipped not in values:
            values.append(shipped)
        values = sorted(values, key=_sort_key)

        curve = Curve(knob=knob, shipped_value=shipped,
                      rebuilt_graph=knob in INGEST_KNOBS)
        for value in values:
            done += 1
            say("sweep", done, total, f"{knob}={value}")
            cfg = Config.load(condition, root=root)
            cfg.llm.provider = "fake"
            cfg.llm.cache_enabled = False
            cfg.set(knob, str(value))
            cfg.validate()

            point = Point(value=value, is_shipped=(value == shipped))
            acc, tok, uni = {}, [], []
            for conv in conversations:
                if knob in INGEST_KNOBS:
                    graph = _ingest_in_memory(cfg, runner, conv)
                else:
                    graph = graphs[conv.sample_id]
                r = _build_retriever(
                    retriever_cls, graph, runner.embedder, cfg,
                    {s.id: s.date_time_raw for s in conv.sessions}, conv,
                )
                pc, t, u = _evaluate(r, conv)
                for k, v in pc.items():
                    acc.setdefault(k, []).extend(v)
                tok += t
                uni += u
            _fold(acc, tok, uni, point)
            curve.points.append(point)
        curves[knob] = curve

    debt = sum(c.tuning_gain for c in curves.values())
    return {
        "condition": base.condition,
        "n_conversations": len(conversations),
        "n_questions": sum(len(c.questions) for c in conversations),
        "budget_tokens": base.retrieval.budget_tokens,
        "plateau_tol": tol,
        "baseline": baseline_point.as_dict(),
        "curves": {k: c.as_dict(tol) for k, c in curves.items()},
        # The headline. Summing the per-knob tuning gains double-counts any
        # interaction between knobs, so it is an ESTIMATE and labelled one --
        # but it is an estimate of the right quantity, and it is the number a
        # reader deserves next to a reported score.
        "estimated_calibration_debt": round(debt, 4),
        "note": (
            "estimated_calibration_debt is the sum of per-knob tuning gains "
            "(shipped minus the median of that knob's swept range) under a "
            "one-at-a-time sweep. It ignores interactions between knobs, so "
            "read it as the order of magnitude of how much of the reported "
            "recall came from having the annotations -- not as an exact "
            "decomposition."
        ),
    }


def _sort_key(v: Any):
    """Numbers sort numerically, everything else lexically, and the two never
    meet in one grid (a knob is one type)."""
    return (0, v, "") if isinstance(v, (int, float)) and not isinstance(v, bool) \
        else (1, 0.0, str(v))


def _ingest_in_memory(cfg: Config, runner: Runner, conv: Conversation):
    """Build a graph for a swept ingest knob WITHOUT touching the cache.

    A sweep that wrote through ``Runner._ingest`` would leave the last swept
    value's graph in ``artifacts/graphs/<condition>/`` for the next real run to
    load, silently changing a condition nobody edited.
    """
    from fgl.logging_utils import NullLogger
    from fgl.pipeline import _INGESTORS

    ingestor_cls = _INGESTORS[cfg.ingest.mode]
    graph, _ = ingestor_cls(
        cfg, runner.llm, runner.embedder, runner.prompts, NullLogger()
    ).ingest(conv)
    return graph


# --------------------------------------------------------------------------- #
# Plain-text report                                                            #
# --------------------------------------------------------------------------- #


_VERDICT_GLOSS = {
    "flat": "knob does not move the metric -- its value is not a result",
    "shallow": "wide good region; the exact value is not load-bearing",
    "peaked": "narrow good region; the value IS a tuned result, declare it",
    "cliff": "shipped value sits on the edge of the plateau -- fragile",
}


def format_sweep(report: dict) -> str:
    lines: list[str] = []
    b = report["baseline"]
    lines.append(
        f"sensitivity sweep · {report['condition']} · {report['n_questions']} "
        f"questions · {report['n_conversations']} conversation(s) · zero LLM calls"
    )
    lines.append(
        f"baseline recall_context {b['overall']:.4f} at "
        f"{b['mean_tokens']:.0f} tokens / {b['mean_units']:.1f} units"
    )
    lines.append("")
    lines.append(
        f"{'knob':<34}{'shipped':>9}{'best':>8}{'worst':>8}"
        f"{'sens':>8}{'plateau':>9}{'tuning':>9}  verdict"
    )
    for knob, c in sorted(
        report["curves"].items(), key=lambda kv: -abs(kv[1]["tuning_gain"])
    ):
        lines.append(
            f"{knob:<34}{c['shipped']:>9.4f}{c['best']:>8.4f}{c['worst']:>8.4f}"
            f"{c['sensitivity']:>8.3f}{c['plateau_frac']:>9.2f}"
            f"{c['tuning_gain']:>+9.4f}  {c['verdict']}"
        )
    lines.append("")
    lines.append("curves (recall_context by swept value; * = the shipped value)")
    for knob, c in report["curves"].items():
        cells = " ".join(
            f"{p['value']}{'*' if p['is_shipped'] else ''}={p['overall']:.3f}"
            for p in c["points"]
        )
        lines.append(f"  {knob}")
        lines.append(f"    {cells}")
    lines.append("")
    lines.append(
        f"estimated calibration debt: {report['estimated_calibration_debt']:+.4f} "
        "recall_context"
    )
    lines.append("  " + report["note"])
    lines.append("")
    for v, gloss in _VERDICT_GLOSS.items():
        lines.append(f"  {v:<9} {gloss}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML report                                                                  #
# --------------------------------------------------------------------------- #

# Categorical slots 1-5 of the reference palette, in its fixed order, light and
# dark. Validated as a set in both modes (worst adjacent CVD dE 9.1 light /
# 8.4 dark; worst adjacent normal-vision dE 19.6 / 19.3). Three light slots sit
# below 3:1 on the light surface, so the relief rule applies and every series
# is direct-labelled at its line end AND repeated in a table below the charts --
# identity is never carried by colour alone.
_SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
_SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181")


def _svg_curve(curve: dict, width: int = 360, height: int = 190) -> str:
    """One small multiple: recall_context against the swept value, per category.

    A line chart because the job is change across an ordered parameter. One
    y-axis only, shared 0..1 domain across every panel so the panels are
    comparable at a glance, and the x-axis is the swept value's RANK rather
    than its magnitude -- the grids are not evenly spaced and a magnitude axis
    would compress half of every curve into the left margin.
    """
    pts = curve["points"]
    if not pts:
        return ""
    pad_l, pad_r, pad_t, pad_b = 34, 58, 12, 26
    iw = width - pad_l - pad_r
    ih = height - pad_t - pad_b
    n = len(pts)

    lo = min(
        min(p["per_category"].values(), default=0.0) for p in pts
    )
    hi = max(
        max(p["per_category"].values(), default=1.0) for p in pts
    )
    span = max(hi - lo, 0.05)
    lo = max(0.0, lo - span * 0.12)
    hi = min(1.0, hi + span * 0.12)

    def x(i: int) -> float:
        return pad_l + (iw * i / max(n - 1, 1))

    def y(v: float) -> float:
        return pad_t + ih * (1.0 - (v - lo) / (hi - lo))

    out: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(curve["knob"])} sensitivity curve">'
    ]
    # gridlines + y ticks, recessive
    for frac in (0.0, 0.5, 1.0):
        v = lo + (hi - lo) * frac
        yy = y(v)
        out.append(
            f'<line class="grid" x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l+iw}" '
            f'y2="{yy:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{pad_l-6}" y="{yy+3.5:.1f}" '
            f'text-anchor="end">{v:.2f}</text>'
        )
    # the shipped value, marked on the axis rather than by colour
    for i, p in enumerate(pts):
        if p["is_shipped"]:
            out.append(
                f'<line class="shipped" x1="{x(i):.1f}" y1="{pad_t}" '
                f'x2="{x(i):.1f}" y2="{pad_t+ih}"/>'
            )
    # x ticks
    for i, p in enumerate(pts):
        label = f'{p["value"]}'
        out.append(
            f'<text class="tick" x="{x(i):.1f}" y="{pad_t+ih+15}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )
    # one line per category, 2px, direct-labelled at the end
    for s, cat in enumerate(CATEGORIES):
        vals = [p["per_category"].get(cat) for p in pts]
        if any(v is None for v in vals):
            continue
        d = " ".join(
            f'{"M" if i == 0 else "L"}{x(i):.1f},{y(v):.1f}'
            for i, v in enumerate(vals)
        )
        out.append(f'<path class="s{s+1} line" d="{d}"/>')
        for i, v in enumerate(vals):
            out.append(f'<circle class="s{s+1} dot" cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.6"/>')
        out.append(
            f'<text class="s{s+1} lbl" x="{pad_l+iw+6}" y="{y(vals[-1])+3.5:.1f}">'
            f'{escape(cat)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def sweep_to_html(report: dict) -> str:
    """A self-contained page: one small multiple per knob, plus the table.

    The table is not decoration -- three of the five light-mode series colours
    sit below 3:1 on the light surface, so the palette's relief rule requires
    either visible labels or a table view. This ships both.
    """
    curves = sorted(
        report["curves"].items(), key=lambda kv: -abs(kv[1]["tuning_gain"])
    )
    panels: list[str] = []
    for knob, c in curves:
        badge = c["verdict"]
        panels.append(
            f'<figure class="panel">'
            f'<figcaption><span class="knob">{escape(knob)}</span>'
            f'<span class="badge b-{badge}">{badge}</span></figcaption>'
            f'<p class="sub">shipped <b>{escape(str(c["shipped_value"]))}</b> · '
            f'sensitivity {c["sensitivity"]:.3f} · plateau {c["plateau_frac"]:.0%} · '
            f'tuning gain {c["tuning_gain"]:+.4f}</p>'
            f'{_svg_curve(c)}</figure>'
        )

    rows: list[str] = []
    for knob, c in curves:
        cells = ", ".join(
            f'{p["value"]}{"*" if p["is_shipped"] else ""}: {p["overall"]:.3f}'
            for p in c["points"]
        )
        rows.append(
            f"<tr><td>{escape(knob)}</td><td>{escape(str(c['shipped_value']))}</td>"
            f"<td>{c['shipped']:.4f}</td><td>{c['best']:.4f}</td>"
            f"<td>{c['worst']:.4f}</td><td>{c['sensitivity']:.3f}</td>"
            f"<td>{c['plateau_frac']:.2f}</td><td>{c['tuning_gain']:+.4f}</td>"
            f"<td>{escape(badge_of(c))}</td><td class='cells'>{escape(cells)}</td></tr>"
        )

    b = report["baseline"]
    series_css_light = "".join(
        f".viz-root .s{i+1}{{--c:{hexv};}}" for i, hexv in enumerate(_SERIES_LIGHT)
    )
    series_css_dark = "".join(
        f".viz-root .s{i+1}{{--c:{hexv};}}" for i, hexv in enumerate(_SERIES_DARK)
    )
    debt = report["estimated_calibration_debt"]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sensitivity sweep · {escape(report['condition'])}</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1:#fcfcfb; --plane:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
    {series_css_light}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1:#1a1a19; --plane:#0d0d0d;
      --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
      {series_css_dark}
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    {series_css_dark}
  }}
  html,body{{margin:0;background:var(--plane);}}
  .viz-root{{
    background:var(--plane); color:var(--text-primary);
    font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
    padding:28px clamp(16px,4vw,48px) 64px; max-width:1240px; margin:0 auto;
  }}
  h1{{font-size:20px;margin:0 0 4px;}}
  .lede{{color:var(--text-secondary);margin:0 0 6px;max-width:70ch;}}
  .hero{{display:flex;gap:32px;flex-wrap:wrap;margin:20px 0 8px;
        padding:16px 18px;background:var(--surface-1);
        border:1px solid var(--border);border-radius:10px;}}
  .hero div span{{display:block;color:var(--muted);font-size:12px;}}
  .hero div b{{font-size:24px;font-weight:650;}}
  .grid-panels{{display:grid;gap:14px;margin-top:22px;
        grid-template-columns:repeat(auto-fill,minmax(380px,1fr));}}
  .panel{{margin:0;padding:12px 12px 6px;background:var(--surface-1);
        border:1px solid var(--border);border-radius:10px;}}
  figcaption{{display:flex;align-items:center;justify-content:space-between;gap:8px;}}
  .knob{{font-weight:600;font-size:13px;}}
  .sub{{color:var(--text-secondary);font-size:12px;margin:2px 0 4px;}}
  .badge{{font-size:11px;padding:2px 7px;border-radius:999px;
        border:1px solid var(--border);color:var(--text-secondary);white-space:nowrap;}}
  .b-peaked,.b-cliff{{color:#d03b3b;border-color:#d03b3b;}}
  .b-flat{{color:#0ca30c;border-color:#0ca30c;}}
  svg{{width:100%;height:auto;display:block;}}
  .grid{{stroke:var(--grid);stroke-width:1;}}
  .shipped{{stroke:var(--axis);stroke-width:1;stroke-dasharray:3 3;}}
  .tick{{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums;}}
  .line{{fill:none;stroke:var(--c);stroke-width:2;stroke-linejoin:round;
        stroke-linecap:round;}}
  .dot{{fill:var(--c);stroke:var(--surface-1);stroke-width:2;}}
  .lbl{{fill:var(--text-secondary);font-size:10px;}}
  table{{border-collapse:collapse;width:100%;margin-top:26px;font-size:12.5px;
        background:var(--surface-1);border:1px solid var(--border);border-radius:10px;}}
  th,td{{text-align:right;padding:7px 10px;border-bottom:1px solid var(--grid);
        font-variant-numeric:tabular-nums;}}
  th:first-child,td:first-child,.cells{{text-align:left;font-variant-numeric:normal;}}
  th{{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
        letter-spacing:.04em;}}
  .cells{{color:var(--text-secondary);}}
  footer{{color:var(--muted);font-size:12px;margin-top:20px;max-width:80ch;}}
</style></head>
<body><div class="viz-root">
<h1>Sensitivity sweep · {escape(report['condition'])}</h1>
<p class="lede">Each knob swept one at a time over the zero-LLM retrieval oracle,
every other knob held at its condition value. A <b>flat</b> curve means the
shipped number is not a result; a <b>peaked</b> one means it is, and was obtained
by looking at annotated data.</p>
<div class="hero">
  <div><span>baseline recall_context</span><b>{b['overall']:.4f}</b></div>
  <div><span>estimated calibration debt</span><b>{debt:+.4f}</b></div>
  <div><span>questions</span><b>{report['n_questions']}</b></div>
  <div><span>token budget</span><b>{report['budget_tokens']}</b></div>
</div>
<div class="grid-panels">{''.join(panels)}</div>
<table><caption class="sub" style="text-align:left;padding:8px 10px">
Same data as the charts, for the colour-independent read. * marks the shipped value.
</caption>
<thead><tr><th>knob</th><th>shipped value</th><th>shipped</th><th>best</th>
<th>worst</th><th>sens</th><th>plateau</th><th>tuning gain</th><th>verdict</th>
<th>curve</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<footer>{escape(report['note'])}</footer>
</div></body></html>
"""


def badge_of(curve_dict: dict) -> str:
    return curve_dict["verdict"]


def write_sweep(report: dict, json_path=None, html_path=None) -> None:
    from pathlib import Path

    if json_path:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if html_path:
        p = Path(html_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(sweep_to_html(report), encoding="utf-8")
