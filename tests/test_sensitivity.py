"""Tests for the sensitivity sweep and the scope check.

The sweep's job is to tell a flat curve from a peaked one, so these tests feed
it synthetic curves of each shape and pin the verdict, rather than running a
real sweep (which needs the dataset). The one end-to-end test is marked
``needs_dataset``.
"""

from __future__ import annotations

import pytest

from fgl.config import Config
from fgl.evaluation.sensitivity import (
    DEFAULT_GRID,
    INGEST_KNOBS,
    PLATEAU_TOL,
    Curve,
    Point,
    format_sweep,
    sweep_to_html,
)

from conftest import PATHS, needs_dataset


def _curve(knob: str, values, overalls, shipped) -> Curve:
    c = Curve(knob=knob, shipped_value=shipped)
    for v, o in zip(values, overalls):
        c.points.append(
            Point(value=v, overall=o, per_category={"single-hop": o},
                  is_shipped=(v == shipped))
        )
    return c


# --------------------------------------------------------------------------- #
# The three summary numbers                                                    #
# --------------------------------------------------------------------------- #


def test_a_flat_curve_is_reported_as_not_a_result():
    """The good case, and the one worth defending.

    If the metric does not move across the swept range, the shipped value was
    picked off a plateau and the tuning bought nothing -- which means reporting
    its optimum is not overfitting, because there was nothing to overfit to.
    """
    c = _curve("slots.type_weight", [0.0, 0.3, 0.6, 1.2], [0.700, 0.702, 0.701, 0.699], 0.6)
    assert c.verdict() == "flat"
    assert c.sensitivity < 0.02
    assert abs(c.tuning_gain) < 0.005


def test_a_peaked_curve_is_reported_as_a_tuned_result():
    """The case that has to be declared: the number IS the result."""
    c = _curve("slots.hub_degree", [15, 30, 60, 120, 300],
               [0.50, 0.58, 0.72, 0.60, 0.51], 60)
    assert c.verdict() in ("peaked", "cliff")
    assert c.sensitivity > 0.2
    assert c.tuning_gain > 0.1, "this is calibration debt, in points of recall"
    assert c.regret == pytest.approx(0.0), "the sweep found the peak, as expected"


def test_a_cliff_is_distinguished_from_a_plateau():
    """Sitting ON the best value is not the same as sitting SAFELY on it. A
    shipped value whose neighbour falls off a cliff is fragile to any corpus
    shift, and that is a different warning."""
    cliff = _curve("slots.sibling_frac", [0.0, 0.1, 0.2, 0.3],
                   [0.40, 0.41, 0.72, 0.71], 0.2)
    assert cliff.verdict() == "cliff"

    safe = _curve("slots.sibling_frac", [0.0, 0.1, 0.2, 0.3],
                  [0.60, 0.715, 0.72, 0.719], 0.2)
    assert safe.verdict() == "shallow"


def test_tuning_gain_can_be_negative():
    """Worth knowing: a sweep that settled somewhere a blind pick would have
    beaten. Hiding that behind an abs() would make the debt look smaller than
    it is."""
    c = _curve("slots.concept_weight", [0.75, 1.0, 1.5, 2.0, 3.0],
               [0.70, 0.71, 0.66, 0.72, 0.73], 1.5)
    assert c.tuning_gain < 0


def test_plateau_frac_counts_the_good_region():
    c = _curve("k", [1, 2, 3, 4], [0.70, 0.699, 0.50, 0.40], 1)
    assert c.plateau_frac(PLATEAU_TOL) == pytest.approx(0.5)


def test_shipped_value_is_always_on_the_curve():
    """`tuning_gain` and `regret` are defined against the shipped value, so a
    grid that does not contain it would silently report zero for both."""
    c = _curve("k", [1, 2, 3], [0.5, 0.6, 0.7], 2)
    assert c.shipped == pytest.approx(0.6)
    assert c.best_value == 3


# --------------------------------------------------------------------------- #
# The grid and the ingest/retrieval split                                      #
# --------------------------------------------------------------------------- #


def test_every_default_grid_knob_exists_in_the_config():
    """A grid entry naming a knob that does not exist would sweep nothing and
    report a perfectly flat curve -- the most misleading possible output."""
    cfg = Config.load("L2")
    for knob in DEFAULT_GRID:
        cfg.get(knob)  # raises AttributeError if the path is wrong


def test_every_default_grid_value_is_accepted_by_the_config():
    """A value the validator rejects would abort the sweep halfway."""
    for knob, values in DEFAULT_GRID.items():
        for v in values:
            cfg = Config.load("L2")
            cfg.set(knob, str(v))
            cfg.validate()


def test_ingest_knobs_are_the_ones_the_ingestor_actually_reads():
    """Mislabelling a graph-changing knob as retrieval-only silently sweeps a
    parameter that never took effect. This pins the list against the ingest
    path's own reads."""
    import inspect

    from fgl.memory import ingest_slots

    source = inspect.getsource(ingest_slots)
    for knob in INGEST_KNOBS:
        section, _, name = knob.partition(".")
        if section != "slots":
            continue
        assert f"sl.{name}" in source or f"slots.{name}" in source, (
            f"{knob} is listed as an ingest knob but the ingestor never reads it"
        )


def test_retrieval_only_grid_knobs_are_not_read_by_the_ingestor():
    import inspect

    from fgl.memory import ingest_slots

    source = inspect.getsource(ingest_slots)
    for knob in DEFAULT_GRID:
        if knob in INGEST_KNOBS:
            continue
        name = knob.split(".")[-1]
        assert f"sl.{name}" not in source, (
            f"{knob} is swept without a rebuild but the ingestor reads it"
        )


# --------------------------------------------------------------------------- #
# Reports                                                                      #
# --------------------------------------------------------------------------- #


def _report() -> dict:
    flat = _curve("slots.type_weight", [0.0, 0.6, 1.2], [0.70, 0.701, 0.699], 0.6)
    peak = _curve("slots.hub_degree", [15, 60, 300], [0.50, 0.72, 0.51], 60)
    return {
        "condition": "L2-slots",
        "n_conversations": 2,
        "n_questions": 400,
        "budget_tokens": 2000,
        "plateau_tol": PLATEAU_TOL,
        "baseline": Point(value="(condition)", overall=0.72, mean_tokens=1989.0,
                          mean_units=58.2, is_shipped=True).as_dict(),
        "curves": {c.knob: c.as_dict() for c in (peak, flat)},
        "estimated_calibration_debt": round(
            peak.tuning_gain + flat.tuning_gain, 4
        ),
        "note": "one-at-a-time; ignores interactions",
    }


def test_text_report_leads_with_the_biggest_debt():
    text = format_sweep(_report())
    assert "estimated calibration debt" in text
    lines = [l for l in text.splitlines() if l.startswith("slots.")]
    assert lines[0].startswith("slots.hub_degree"), "sorted by |tuning_gain|"


def test_html_report_is_self_contained_and_has_a_table_view():
    """Three of the five light-mode series colours sit below 3:1 on the light
    surface, so the palette's relief rule requires visible labels or a table.
    This ships both -- identity is never carried by colour alone."""
    html = sweep_to_html(_report())
    assert html.startswith("<!doctype html>")
    assert "<table" in html and "<svg" in html
    assert "http://" not in html and "https://" not in html, "no external assets"
    assert "prefers-color-scheme: dark" in html
    # every series is direct-labelled at its line end
    assert 'class="s1 lbl"' in html


# --------------------------------------------------------------------------- #
# End to end                                                                   #
# --------------------------------------------------------------------------- #


@needs_dataset
@pytest.mark.slow
def test_sweep_runs_end_to_end_without_an_llm():
    """One real conversation, one knob, no completion.

    Guards the wiring the synthetic tests above cannot: that `sweep` can build
    a condition's graphs through the Runner, rebuild a retriever per swept
    value, and come back with a curve. Skipped without the dataset -- which is
    also why this file's loader name has to match `fgl.data.locomo` exactly and
    not be assumed.
    """
    from fgl.data.locomo import load_conversations
    from fgl.evaluation.sensitivity import sweep

    convs = load_conversations(PATHS.locomo_file)[:1]
    convs[0].questions = convs[0].questions[:12]
    report = sweep(
        "L2", convs,
        grid={"slots.sibling_frac": [0.0, 0.2, 0.5]},
    )
    curve = report["curves"]["slots.sibling_frac"]
    assert len(curve["points"]) == 3
    assert any(p["is_shipped"] for p in curve["points"])
    assert curve["verdict"] in ("flat", "shallow", "peaked", "cliff")
    assert report["baseline"]["overall"] >= 0.0
