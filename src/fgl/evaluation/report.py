"""Turn ``results/*/metrics.json`` into Markdown tables.

Every table here is also what ``fgl report`` prints and what the notebooks
render, so the numbers in the terminal, in ``results/report.md`` and in the
plots come from a single code path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

CATEGORY_ORDER = ["multi-hop", "temporal", "open-domain", "single-hop", "adversarial"]
CONDITION_ORDER = [
    "B1-full-context",
    "B2-rag-turns",
    "B3-rag-facts",
    "G1-fatgraph-min",
    "G2-fatgraph-cur",
    "G3-fatgraph-agent",
]
#: The three comparisons the study is built around.
KEY_PAIRS = [
    ("B3-rag-facts", "G1-fatgraph-min", "valor das faces (mesmos fatos)"),
    ("G1-fatgraph-min", "G2-fatgraph-cur", "valor da curadoria + consolidação"),
    ("G2-fatgraph-cur", "G3-fatgraph-agent", "valor do sigma-agent"),
]


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #


def load_results(results_dir: str | Path) -> dict[str, dict]:
    """Read every ``<condition>/metrics.json`` under ``results_dir``."""
    out: dict[str, dict] = {}
    for p in sorted(Path(results_dir).glob("*/metrics.json")):
        try:
            out[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return out


def order_conditions(results: Mapping[str, dict]) -> list[str]:
    known = [c for c in CONDITION_ORDER if c in results]
    return known + sorted(c for c in results if c not in CONDITION_ORDER)


def categories_in(results: Mapping[str, dict]) -> list[str]:
    present = {c for r in results.values() for c in r.get("per_category", {})}
    known = [c for c in CATEGORY_ORDER if c in present]
    return known + sorted(present - set(known))


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def markdown_table(results: Mapping[str, dict], metric: str = "f1") -> str:
    """F1 per category x condition, plus macro/micro."""
    cats = categories_in(results)
    rows = []
    for cond in order_conditions(results):
        r = results[cond]
        cells = [cond]
        cells += [
            f"{r['per_category'][c][metric]:.3f}" if c in r.get("per_category", {}) else "-"
            for c in cats
        ]
        cells += [
            f"{r['overall'].get('f1_macro', float('nan')):.3f}",
            f"{r['overall'].get('f1_micro', float('nan')):.3f}",
        ]
        rows.append(cells)
    return _table(["condition", *cats, "macro", "micro"], rows)


def recall_table(results: Mapping[str, dict]) -> str:
    ks = sorted(
        {
            k
            for r in results.values()
            for cat in r.get("per_category", {}).values()
            for k in cat
            if k.startswith("recall")
        }
    )
    if not ks:
        return "_(sem métricas de recall)_"
    rows = []
    for cond in order_conditions(results):
        cats = results[cond].get("per_category", {}).values()
        cells = [cond]
        for k in ks:
            vals = [c[k] for c in cats if k in c]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "-")
        rows.append(cells)
    return _table(["condition", *ks], rows)


def cost_table(results: Mapping[str, dict]) -> str:
    rows = []
    for cond in order_conditions(results):
        r = results[cond]
        c = r.get("cost", {})
        per = r.get("per_conversation", [])
        ing = sum((x.get("cost_ingest") or {}).get("total_tokens", 0) for x in per)
        qa = sum((x.get("cost_qa") or {}).get("total_tokens", 0) for x in per)
        rows.append(
            [
                cond,
                str(c.get("calls", 0)),
                str(c.get("cached_calls", 0)),
                f"{ing:,}",
                f"{qa:,}",
                f"{c.get('total_tokens', 0):,}",
                f"{r.get('wall_seconds', 0):.0f}s",
            ]
        )
    return _table(
        ["condition", "calls", "cached", "tokens ingest", "tokens QA", "total", "wall"],
        rows,
    )


def graph_table(results: Mapping[str, dict]) -> str:
    rows = []
    for cond in order_conditions(results):
        per = [p for p in results[cond].get("per_conversation", []) if "graph" in p]
        if not per:
            continue

        def s(key: str, sub: str = "graph") -> int:
            return sum((p.get(sub) or {}).get(key, 0) or 0 for p in per)

        rows.append(
            [
                cond,
                str(s("V")), str(s("E")), str(s("F")), str(s("C")), str(s("genus")),
                str(max((p["graph"].get("face_length_max", 0)) for p in per)),
                str(s("n_bigon_faces")), str(s("n_leaf_faces")),
                str(s("n_collapses", "ingest")), str(s("n_consolidations", "ingest")),
                str(s("n_incongruent", "ingest")),
            ]
        )
    if not rows:
        return "_(sem condições de fatgraph nos resultados)_"
    return _table(
        ["condition", "V", "E", "F", "C", "genus", "max face", "bigons",
         "leaf faces", "collapses", "consolid.", "incongr."],
        rows,
    )


def comparison_table(results: Mapping[str, dict]) -> str:
    rows = []
    for a, b, label in KEY_PAIRS:
        if a in results and b in results:
            fa = results[a]["overall"]["f1_micro"]
            fb = results[b]["overall"]["f1_micro"]
            rows.append(
                [f"{b} − {a}", label, f"{fa:.3f}", f"{fb:.3f}",
                 f"{fb - fa:+.3f}", f"{(fb - fa) / fa * 100:+.1f}%" if fa else "-"]
            )
    if not rows:
        return "_(rode as duas condições de cada par para ver os deltas)_"
    return _table(["comparação", "isola", "F1 base", "F1 novo", "Δ", "Δ%"], rows)


# --------------------------------------------------------------------------- #
# Full report                                                                  #
# --------------------------------------------------------------------------- #


def build_report(results: Mapping[str, dict]) -> str:
    stemmers = {r.get("stemmer") for r in results.values()}
    warn = (
        "\n> ⚠️ Resultados produzidos com stemmers diferentes "
        f"({sorted(s for s in stemmers if s)}) — não são comparáveis entre si.\n"
        if len(stemmers) > 1
        else ""
    )
    return "\n".join(
        [
            "# Resultados — memória fatgraph no LoCoMo",
            "",
            "F1 é a métrica oficial do LoCoMo (token-level, `task_eval/evaluation.py`), "
            "reportada para **todas** as categorias, inclusive adversarial. "
            "Nada foi filtrado nem subamostrado.",
            warn,
            "## F1 por categoria",
            "",
            markdown_table(results),
            "",
            "## Comparações-chave",
            "",
            comparison_table(results),
            "",
            "## Recall da recuperação (evidências anotadas)",
            "",
            recall_table(results),
            "",
            "## Estatísticas do grafo",
            "",
            graph_table(results),
            "",
            "## Custo em tokens de LLM",
            "",
            cost_table(results),
            "",
        ]
    )


def write_report(results: Mapping[str, dict], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_report(results), encoding="utf-8")
    return p
