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
    "G4-fatgraph-sigma",
    "G5-fatgraph-coverage",
    "G6-fatgraph-join",
    "G7-rag-sigma",
    "G8-shuffled",
    "G9-genus",
    "G10-face-units",
]
#: The comparisons the study is built around.
KEY_PAIRS = [
    ("B3-rag-facts", "G1-fatgraph-min", "valor das faces (mesmos fatos)"),
    ("G1-fatgraph-min", "G2-fatgraph-cur", "valor da curadoria + consolidação"),
    ("G2-fatgraph-cur", "G3-fatgraph-agent", "valor do sigma-agent"),
    ("G1-fatgraph-min", "G4-fatgraph-sigma", "valor da expansão por sigma (mesmo grafo)"),
    ("G1-fatgraph-min", "G5-fatgraph-coverage", "valor da cobertura de entidades"),
    ("G4-fatgraph-sigma", "G6-fatgraph-join", "cobertura ADICIONADA a sigma"),
    ("G5-fatgraph-coverage", "G6-fatgraph-join", "sigma ADICIONADO à cobertura"),
    # as três que os resultados motivaram, em ordem de poder de decisão
    ("G4-fatgraph-sigma", "G8-shuffled", "A ORDEM IMPORTA? (mesmo conteúdo, permutado)"),
    ("B3-rag-facts", "G7-rag-sigma", "sigma ACRESCENTA ao k-NN puro?"),
    ("G1-fatgraph-min", "G9-genus", "sigma escolhida por gênero mínimo vs pelo relógio"),
    # a proposta: a face como UNIDADE, contra o alvo e contra a mesma
    # superfície percorrida como caminho
    ("B3-rag-facts", "G10-face-units", "FACE COMO UNIDADE vs k-NN puro (o alvo)"),
    ("G9-genus", "G10-face-units", "face como conjunto vs face como caminho"),
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


def sigma_table(results: Mapping[str, dict]) -> str:
    """Was the sigma expansion actually used, and did it bring new evidence?

    A condition that claims the expansion but shows ``uso 0.000`` never joined
    anything, and its F1 is just G1's with extra steps. This table is the
    audit: it reads the per-question columns, not the config.
    """
    rows = []
    for cond in order_conditions(results):
        r = results[cond]
        cats = r.get("per_category", {})
        overall = r.get("overall", {})
        if not overall.get("sigma_expand"):
            continue
        mh = cats.get("multi-hop", {})
        rows.append(
            [
                cond,
                f"{overall.get('sigma_use_rate', 0):.3f}",
                f"{overall.get('sigma_facts_mean', 0):.2f}",
                f"{overall.get('sigma_bridges_mean', 0):.2f}",
                f"{overall.get('sigma_tokens_mean', 0):.0f}",
                f"{overall.get('sigma_evidence_rate', 0):.3f}",
                f"{mh.get('sigma_use_rate', float('nan')):.3f}" if mh else "-",
                f"{mh.get('recall_context_no_sigma', float('nan')):.3f}" if mh else "-",
                f"{mh.get('recall_context', float('nan')):.3f}" if mh else "-",
            ]
        )
    if not rows:
        return (
            "_(nenhuma condição rodou com `retrieval.sigma_expand` — as colunas "
            "de auditoria estão ausentes/zeradas, como esperado para G1–G3)_"
        )
    return _table(
        [
            "condition", "uso", "fatos σ", "pontes", "tokens σ", "evidência só via σ",
            "uso (multi-hop)", "recall MH sem σ", "recall MH com σ",
        ],
        rows,
    )


def coverage_table(results: Mapping[str, dict]) -> str:
    """A recuperação por cobertura foi usada, e trouxe evidência nova?

    `ligadas` é a pré-condição de tudo: sem entidade ligada não há sinal de
    cobertura e a condição degenera em G1.
    """
    rows = []
    for cond in order_conditions(results):
        overall = results[cond].get("overall", {})
        if not overall.get("face_coverage"):
            continue
        mh = results[cond].get("per_category", {}).get("multi-hop", {})
        rows.append(
            [
                cond,
                f"{overall.get('coverage_link_rate', 0):.3f}",
                f"{overall.get('coverage_entities_mean', 0):.2f}",
                f"{overall.get('coverage_best_mean', 0):.3f}",
                f"{overall.get('coverage_bridge_rate', 0):.3f}",
                f"{overall.get('coverage_use_rate', 0):.3f}",
                f"{overall.get('geodesic_rate', 0):.3f}",
                f"{overall.get('coverage_evidence_rate', 0):.3f}",
                f"{mh.get('recall_context_no_coverage', float('nan')):.3f}" if mh else "-",
                f"{mh.get('recall_context', float('nan')):.3f}" if mh else "-",
            ]
        )
    if not rows:
        return "_(nenhuma condição rodou com `retrieval.face_coverage`)_"
    return _table(
        ["condition", "ligadas", "entidades", "cobertura máx", "faces-ponte",
         "uso", "geodésica", "evidência só via cobertura",
         "recall MH sem cob.", "recall MH com cob."],
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


#: Conditions whose ingest is byte-for-byte G1's, so their graphs MUST hash
#: equal to G1's. They differ from it only in how the memory is queried.
RETRIEVAL_ONLY = [
    "G4-fatgraph-sigma",
    "G5-fatgraph-coverage",
    "G6-fatgraph-join",
    "G7-rag-sigma",
    "G8-shuffled",
]


def _fingerprints(metrics: dict) -> dict[str, str]:
    """``sample_id -> graph fingerprint`` for one condition's run."""
    out = {}
    for row in metrics.get("per_conversation", []):
        fp = (row.get("graph") or {}).get("fingerprint")
        if fp:
            out[row.get("sample_id", "?")] = fp
    return out


def graph_identity_table(results: Mapping[str, dict]) -> str:
    """Do the retrieval-only conditions really share G1's memory?

    They used to guarantee it by reading G1's graph directory, which made their
    numbers an artefact of G1's run.  Each now builds its own, so the guarantee
    has to be *checked*: identical ingest over the same fact cache must produce
    the same ribbon graph, and `FatGraph.fingerprint` is content-addressed, so
    equality here is equality of memory and rotation both.

    A mismatch invalidates the corresponding delta -- it would no longer isolate
    retrieval, because the two arms would be remembering different things.
    """
    base = _fingerprints(results.get("G1-fatgraph-min", {}))
    if not base:
        return "_(rode a G1 para comparar as impressões digitais dos grafos)_"

    rows = []
    for cond in RETRIEVAL_ONLY:
        if cond not in results:
            continue
        got = _fingerprints(results[cond])
        shared = sorted(set(base) & set(got))
        if not shared:
            rows.append([cond, "-", "-", "sem conversas em comum"])
            continue
        same = [s for s in shared if got[s] == base[s]]
        verdict = (
            "idêntico à G1"
            if len(same) == len(shared)
            else f"**DIVERGE** em {len(shared) - len(same)} — o delta não isola recuperação"
        )
        rows.append([cond, f"{len(same)}/{len(shared)}", base[shared[0]][:12], verdict])

    if not rows:
        return "_(nenhuma condição de recuperação-apenas nos resultados)_"
    note = (
        "\nCada condição constrói o próprio grafo; a igualdade abaixo é medida, "
        "não imposta por compartilhamento de diretório.\n\n"
    )
    return note + _table(
        ["condição", "grafos iguais", "fingerprint G1", "veredito"], rows
    )


def judge_table(results: Mapping[str, dict]) -> str:
    """Token-F1 next to the LLM judge, plus how far apart they are.

    Never replaces the F1 column.  The two measure different things -- one asks
    whether the words overlap, the other whether the claim matches -- and the
    gap between them is itself a finding, so both are quoted or neither is.
    """
    rows = []
    for cond in CONDITION_ORDER:
        r = results.get(cond)
        if not r or "judge_micro" not in (r.get("overall") or {}):
            continue
        o = r["overall"]
        rows.append([
            cond,
            f"{o.get('f1_micro', 0):.4f}",
            f"{o.get('judge_micro', 0):.4f}",
            f"{o.get('f1_substantive', 0):.4f}",
            f"{o.get('judge_substantive', 0):.4f}",
            f"{o.get('judge_f1_agreement', 0):.1%}",
            str(o.get("judge_yes_f1_low", 0)),
            str(o.get("judge_no_f1_high", 0)),
        ])
    if not rows:
        return "_(rode `fgl judge` para pontuar as predições com o juiz LLM)_"
    note = (
        "\n`juiz aceita / F1 rejeita` é paráfrase que a métrica de tokens perdia. "
        "`juiz rejeita / F1 aceita` é o erro oposto e tem de ser pequeno — se não "
        "for, o juiz está severo demais e o número não deve ser citado antes de "
        "ler as discordâncias à mão.\n\n"
    )
    return note + _table(
        ["condição", "F1", "juiz", "F1 subst.", "juiz subst.",
         "concord.", "juiz aceita / F1 rejeita", "juiz rejeita / F1 aceita"],
        rows,
    )


def sanity_banner(results: Mapping[str, dict]) -> str:
    """A run whose answers are all identical is not a result."""
    bad = {c: (r.get("sanity") or {}) for c, r in results.items()}
    bad = {c: s for c, s in bad.items() if s and not s.get("ok", True)}
    if not bad:
        return ""
    lines = ["", "> ## ⚠️ Corrida suspeita — não interprete estes números", ">"]
    for cond, s in bad.items():
        lines.append(f"> **{cond}**")
        lines += [f">   - {w}" for w in s.get("warnings", [])]
    lines += [">", "> Diagnostique com `fgl doctor`.", ""]
    return "\n".join(lines)


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
            sanity_banner(results),
            "## F1 por categoria",
            "",
            markdown_table(results),
            "",
            "## Métrica: sobreposição de tokens vs juiz LLM",
            "",
            judge_table(results),
            "",
            "## Comparações-chave",
            "",
            comparison_table(results),
            "",
            "## Identidade dos grafos (as ablations isolam o que dizem isolar?)",
            "",
            graph_identity_table(results),
            "",
            "## Recall da recuperação (evidências anotadas)",
            "",
            recall_table(results),
            "",
            "## Expansão por sigma (auditoria)",
            "",
            "`uso` = fração de perguntas em que a órbita de sigma contribuiu com "
            "pelo menos um fato. `evidência só via σ` = fração em que sigma "
            "alcançou um turno de evidência que nenhuma face alcançou — a "
            "contribuição marginal do salto, não só sua atividade.",
            "",
            sigma_table(results),
            "",
            "## Recuperação por cobertura de entidades (auditoria)",
            "",
            coverage_table(results),
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
