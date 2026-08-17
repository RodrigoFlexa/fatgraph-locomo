"""Orchestration: build a memory, answer every question, write metrics.

Enforced protocol (spec section 5):

* memory construction reads **only** the dialogues, never the questions;
* the QA phase reads **only** the memory -- the graph is reloaded from disk --
  except in B1/B2, which are defined in terms of raw turns;
* every question of every category is answered; nothing is filtered.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from fgl.config import Config
from fgl.core import FatGraph
from fgl.data.locomo import Conversation, load_conversations
from fgl.evaluation import QAOutcome, aggregate, evidence_recall, score_question
from fgl.llm import LLMClient, PromptLibrary, build_llm
from fgl.logging_utils import JsonlLogger
from fgl.memory.ingest import Ingestor
from fgl.paths import Paths, project_root
from fgl.retrieval import (
    JOIN_SOURCES,
    SOURCE_COVERAGE,
    SOURCE_GEODESIC,
    SOURCE_SIGMA,
    Answerer,
    CachedEmbedder,
    Embedder,
    FaceRetriever,
    build_embedder,
)

FATGRAPH_CONDITIONS = (
    "G1-fatgraph-min",
    "G2-fatgraph-cur",
    "G3-fatgraph-agent",
    "G4-fatgraph-sigma",
    "G5-fatgraph-coverage",
    "G6-fatgraph-join",
)
BASELINE_CONDITIONS = ("B1-full-context", "B2-rag-turns", "B3-rag-facts")

#: Above these the memory graph is a star and multi-hop retrieval is redundant
#: by construction -- see `Runner._topology_warnings`. A long tail of degree-1
#: entities is normal (plenty are named once), which is why the line sits well
#: above half rather than at it; `hub_share` is the sharper of the two, since
#: two vertices holding a third of all half-edges means the speakers.
STAR_DEGREE1_FRAC = 0.6
STAR_HUB_SHARE = 0.35

#: called as ``progress(stage, done, total, detail)``; the CLI wires a Rich bar.
ProgressFn = Callable[[str, int, int, str], None]


class Runner:
    """Runs one experimental condition end to end."""

    def __init__(
        self,
        cfg: Config,
        root: str | Path | None = None,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        prompts: PromptLibrary | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self.cfg = cfg
        self.paths = Paths.build(root or project_root())
        self.root = self.paths.root
        self.prompts = prompts or PromptLibrary(self.paths.resolve(cfg.paths.prompts_dir))
        self.llm = llm or build_llm(self._llm_cfg())
        self.embedder = embedder or build_embedder(self._embedding_cfg())
        self.progress = progress or (lambda *a: None)

    # -- absolute-path variants of the two cfg sections that write to disk ----
    def _llm_cfg(self):
        c = self.cfg.llm
        c.cache_dir = str(self.paths.resolve(c.cache_dir))
        return c

    def _embedding_cfg(self):
        c = self.cfg.embeddings
        c.cache_dir = str(self.paths.resolve(c.cache_dir))
        return c

    # ------------------------------------------------------------------ api --
    def run(
        self,
        conversations: Sequence[Conversation],
        limit_questions: int = 0,
    ) -> dict:
        started = time.time()
        is_baseline = self.cfg.condition in BASELINE_CONDITIONS
        outcomes: list[QAOutcome] = []
        per_conversation: list[dict] = []
        total = len(conversations)

        for i, conv in enumerate(conversations):
            self.progress("conversation", i, total, conv.sample_id)
            if is_baseline:
                conv_outcomes, extra = self._run_baseline(conv, limit_questions)
            else:
                conv_outcomes, extra = self._run_fatgraph(conv, limit_questions)
            outcomes.extend(conv_outcomes)
            per_conversation.append(
                {
                    "sample_id": conv.sample_id,
                    "n_sessions": len(conv.sessions),
                    "n_turns": conv.n_turns,
                    "n_questions": len(conv_outcomes),
                    "f1": aggregate(conv_outcomes)["overall"]["f1_micro"],
                    **extra,
                }
            )
        self.progress("conversation", total, total, "done")

        metrics = {
            "condition": self.cfg.condition,
            "manifest": self._manifest(),
            **aggregate(outcomes),
            "per_conversation": per_conversation,
            "cost": self.llm.usage.to_dict(),
            "wall_seconds": round(time.time() - started, 1),
        }
        metrics["sanity"] = self._sanity(outcomes, per_conversation)
        self._write(metrics, outcomes)
        if isinstance(self.embedder, CachedEmbedder):
            self.embedder.flush()
        return metrics

    def ingest_only(self, conversations: Sequence[Conversation], force: bool = False) -> list[dict]:
        """Build (or rebuild) the memory graphs without answering anything."""
        reports = []
        for i, conv in enumerate(conversations):
            self.progress("ingest", i, len(conversations), conv.sample_id)
            reports.append(self._ingest(conv, force=force)[1])
        self.progress("ingest", len(conversations), len(conversations), "done")
        if isinstance(self.embedder, CachedEmbedder):
            self.embedder.flush()
        return reports

    # ------------------------------------------------------------ fatgraph --
    def graphs_condition(self) -> str:
        """Which condition's graphs this run reads.

        ``paths.graphs_condition`` lets a *retrieval-only* ablation reuse the
        graphs of another condition instead of rebuilding an identical memory:
        G4 differs from G1 only in how the memory is queried, so sharing G1's
        graphs makes the delta attributable to retrieval alone -- and costs no
        LLM calls.
        """
        return self.cfg.paths.graphs_condition or self.cfg.condition

    def _graph_path(self, conv: Conversation) -> Path:
        return (
            self.paths.resolve(self.cfg.paths.graphs_dir)
            / self.graphs_condition()
            / conv.sample_id
        )

    def _ingest(self, conv: Conversation, force: bool = False) -> tuple[FatGraph, dict]:
        graph_path = self._graph_path(conv)
        report_path = graph_path.with_name(graph_path.name + ".report.json")
        if graph_path.with_suffix(".json").exists() and report_path.exists() and not force:
            graph = FatGraph.load(graph_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            return graph, _refresh_graph_stats(graph, report)

        borrowed = self.graphs_condition()
        if borrowed != self.cfg.condition:
            # Never *build* into somebody else's directory: the graph would be
            # made with this condition's ingest settings and silently poison
            # the condition it was borrowed from.
            raise FileNotFoundError(
                f"{self.cfg.condition} is configured to reuse the graphs of "
                f"{borrowed} (paths.graphs_condition), but {graph_path.name} is "
                f"missing from {graph_path.parent}.\n"
                f"Build them first:  fgl ingest {borrowed}"
            )

        log_path = (
            self.paths.resolve(self.cfg.paths.logs_dir)
            / self.cfg.condition
            / f"{conv.sample_id}.jsonl"
        )
        with JsonlLogger(log_path) as logger:
            cfg = self._ingest_cfg()
            graph, report_obj = Ingestor(
                cfg, self.llm, self.embedder, self.prompts, logger
            ).ingest(conv)
        report = report_obj.to_dict()
        graph.save(graph_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return graph, report

    def _ingest_cfg(self) -> Config:
        self.cfg.paths.facts_cache = str(self.paths.resolve(self.cfg.paths.facts_cache))
        return self.cfg

    def _run_fatgraph(
        self, conv: Conversation, limit_questions: int
    ) -> tuple[list[QAOutcome], dict]:
        usage_before = self.llm.usage.to_dict()
        graph, report = self._ingest(conv)
        ingest_usage = _usage_delta(usage_before, self.llm.usage.to_dict())

        dates = {s.id: s.date_time_raw for s in conv.sessions}
        retriever = FaceRetriever(graph, self.embedder, self.cfg, dates)
        answerer = Answerer(self.llm, self.prompts, self.cfg)

        outcomes: list[QAOutcome] = []
        questions = conv.questions[:limit_questions] if limit_questions else conv.questions
        for i, q in enumerate(questions):
            if i % 25 == 0:
                self.progress("qa", i, len(questions), conv.sample_id)
            result = retriever.retrieve(q.prompt_question())
            prediction = answerer.answer(conv, q, result)
            recall = {
                f"recall@{k}": evidence_recall(
                    q.evidence,
                    retriever.turn_ids_for_edges(
                        retriever.top_edges(q.prompt_question(), k)
                    ),
                )
                for k in self.cfg.retrieval.recall_ks
            }
            recall["recall_context"] = evidence_recall(q.evidence, result.turn_ids)
            # the same recall computed as if each mechanism had not run: the
            # difference against recall_context IS the effect of that join
            if result.sigma_expand:
                recall["recall_context_no_sigma"] = evidence_recall(
                    q.evidence, result.turn_ids_excluding(SOURCE_SIGMA)
                )
            if result.face_coverage:
                recall["recall_context_no_coverage"] = evidence_recall(
                    q.evidence,
                    result.turn_ids_excluding(SOURCE_COVERAGE, SOURCE_GEODESIC),
                )
            if result.sigma_expand and result.face_coverage:
                recall["recall_context_anchors_only"] = evidence_recall(
                    q.evidence, result.turn_ids_excluding(*JOIN_SOURCES)
                )
            outcomes.append(
                QAOutcome(
                    question=q.question,
                    category=q.category,
                    gold=q.answer,
                    prediction=prediction,
                    f1=score_question(q, prediction),
                    evidence=q.evidence,
                    retrieved_turn_ids=result.turn_ids,
                    recall=recall,
                    n_facts=len(result.facts),
                    n_faces=len(set(result.faces)),
                    tokens_context=result.tokens_used,
                    abstained=prediction.strip().lower().startswith("not mentioned"),
                    sigma_expand=result.sigma_expand,
                    n_sigma_facts=result.n_sigma_facts,
                    sigma_vertices=list(result.sigma_vertices),
                    sigma_tokens=result.sigma_tokens,
                    sigma_only_turn_ids=result.sigma_turn_ids,
                    sigma_scanned=result.sigma_scanned,
                    sigma_dup=result.sigma_dup,
                    sigma_over_budget=result.sigma_over_budget,
                    face_coverage=result.face_coverage,
                    question_entities=list(result.question_entities),
                    coverage_best=result.coverage_best,
                    coverage_faces_multi=result.coverage_faces_multi,
                    n_coverage_facts=result.n_coverage_facts,
                    n_geodesic_facts=result.n_geodesic_facts,
                    geodesic_len=result.geodesic_len,
                    coverage_tokens=result.coverage_tokens,
                    coverage_only_turn_ids=result.coverage_turn_ids,
                )
            )
        qa_usage = _usage_delta(_add(usage_before, ingest_usage), self.llm.usage.to_dict())
        return outcomes, {
            "graph": report.get("graph_stats", {}),
            "ingest": {
                k: report.get(k)
                for k in (
                    "n_facts", "n_edges", "n_incongruent", "n_collapses",
                    "n_consolidations", "n_skipped_self_loops",
                )
            },
            "per_session": report.get("per_session", []),
            "cost_ingest": ingest_usage,
            "cost_qa": qa_usage,
        }

    # ------------------------------------------------------------ baselines --
    def _run_baseline(
        self, conv: Conversation, limit_questions: int
    ) -> tuple[list[QAOutcome], dict]:
        from fgl.baselines import REGISTRY

        usage_before = self.llm.usage.to_dict()
        cfg = self._ingest_cfg()  # B3 needs the resolved facts cache
        baseline = REGISTRY[cfg.condition](cfg, self.llm, self.embedder, self.prompts)
        baseline.prepare(conv)
        prep_usage = _usage_delta(usage_before, self.llm.usage.to_dict())

        outcomes: list[QAOutcome] = []
        questions = conv.questions[:limit_questions] if limit_questions else conv.questions
        for i, q in enumerate(questions):
            if i % 25 == 0:
                self.progress("qa", i, len(questions), conv.sample_id)
            ans = baseline.answer(conv, q)
            ranked = ans.ranked_turn_ids or ans.retrieved_turn_ids
            recall = {
                f"recall@{k}": evidence_recall(q.evidence, ranked[:k])
                for k in self.cfg.retrieval.recall_ks
            }
            recall["recall_context"] = evidence_recall(q.evidence, ans.retrieved_turn_ids)
            outcomes.append(
                QAOutcome(
                    question=q.question,
                    category=q.category,
                    gold=q.answer,
                    prediction=ans.prediction,
                    f1=score_question(q, ans.prediction),
                    evidence=q.evidence,
                    retrieved_turn_ids=ans.retrieved_turn_ids[:50],
                    recall=recall,
                    n_facts=ans.n_items,
                    tokens_context=ans.tokens_context,
                    abstained=ans.prediction.strip().lower().startswith("not mentioned"),
                )
            )
        qa_usage = _usage_delta(_add(usage_before, prep_usage), self.llm.usage.to_dict())
        return outcomes, {"cost_ingest": prep_usage, "cost_qa": qa_usage}

    # ---------------------------------------------------------------- io -----
    def results_dir(self) -> Path:
        return self.paths.resolve(self.cfg.paths.results_dir) / self.cfg.condition

    def _write(self, metrics: dict, outcomes: list[QAOutcome]) -> None:
        out_dir = self.results_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
            for o in outcomes:
                fh.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")

    def _topology_warnings(self, per_conversation: Sequence[dict]) -> list[str]:
        """Is the memory shaped so that multi-hop retrieval *can* work at all?

        Both G4 and G5 have the same topological precondition, and when it
        fails they do not fail loudly -- they quietly reproduce G1, and the
        table shows three conditions agreeing to four decimals as though that
        were a finding.  The cause is then upstream, in what the extractor
        chose to call an entity, and no amount of `sigma_expand_k` fixes it.

        In a star, ``sigma(alpha(h)) = alpha(h)`` for every degree-1 neighbour,
        so ``phi`` collapses into a walk along the hub's own orbit: the face
        already yields exactly what sigma would propose (hence a `sigma_dup`
        rate near 1), and every face is enormous and touches nearly every
        vertex, so coverage cannot rank them apart.  Both mechanisms are then
        redundant *by construction*, which is a fact about the graph and not
        about the retriever.
        """
        stats = [c.get("graph") or {} for c in per_conversation]
        stats = [s for s in stats if s.get("degree_1_frac") is not None]
        if not stats:
            return []

        deg1 = float(np.mean([s["degree_1_frac"] for s in stats]))
        hub = float(np.mean([s.get("hub_share", 0.0) for s in stats]))
        faces = float(np.mean([s.get("n_faces_nontrivial", 0) for s in stats]))
        if deg1 < STAR_DEGREE1_FRAC and hub < STAR_HUB_SHARE:
            return []

        uses_join = self.cfg.retrieval.sigma_expand or self.cfg.retrieval.face_coverage
        consequence = (
            "sigma é redundante com phi e a cobertura por face não discrimina: "
            "esta condição vai reproduzir a G1 e o delta que ela deveria medir "
            "não existe neste grafo"
            if uses_join
            else "G4/G5/G6 sobre estes grafos vão reproduzir a G1"
        )
        return [
            f"o grafo é quase uma ESTRELA ({deg1:.0%} dos vértices têm grau 1, "
            f"{hub:.0%} das meias-arestas estão nos dois maiores vértices, "
            f"{faces:.1f} faces não triviais em média) — {consequence}. "
            "A causa está no INGEST, não na recuperação: os fatos estão sendo "
            "ancorados em quem falou, não no que foi dito. Confira a extração e "
            "a resolução de entidades antes de interpretar qualquer número."
        ]

    def _sanity(
        self,
        outcomes: Sequence[QAOutcome],
        per_conversation: Sequence[dict] = (),
    ) -> dict:
        """Detect the failure modes that still produce a plausible-looking table.

        The one that actually bit us: a broken backend makes every answer the
        abstention string, which scores 1.0 on adversarial and ~0.01 elsewhere --
        identical across all six conditions. Numbers alone look like a result;
        they are not.
        """
        checks: list[str] = []
        n = len(outcomes)
        if not n:
            return {"ok": False, "warnings": ["nenhuma pergunta foi respondida"]}

        checks.extend(self.llm.usage.warnings())
        checks.extend(self._topology_warnings(per_conversation))

        abstained = sum(o.abstained for o in outcomes)
        non_adv = [o for o in outcomes if o.category != 5]
        abstained_non_adv = sum(o.abstained for o in non_adv)
        if non_adv and abstained_non_adv / len(non_adv) > 0.95:
            checks.append(
                f"{abstained_non_adv}/{len(non_adv)} perguntas NÃO adversariais "
                "foram respondidas com a string de abstenção — provável backend "
                "quebrado ou recuperação vazia (rode: fgl doctor)"
            )

        distinct = {o.prediction.strip().lower() for o in outcomes}
        if len(distinct) <= 2 and n > 20:
            checks.append(
                f"apenas {len(distinct)} resposta(s) distinta(s) em {n} perguntas — "
                "o modelo não está de fato respondendo"
            )

        if self.cfg.condition not in BASELINE_CONDITIONS:
            empty_ctx = sum(1 for o in outcomes if o.n_facts == 0)
            if empty_ctx / n > 0.5:
                checks.append(
                    f"{empty_ctx}/{n} perguntas recuperaram ZERO fatos — o grafo de "
                    "memória provavelmente está vazio (extração falhou?)"
                )

        # A condition that advertises the sigma expansion but never joins
        # anything is G1 with extra steps -- and its F1 would be reported as if
        # it were a result. Catch it here, not three tables later.
        # (The *why* usually lives in the topology, which `_topology_warnings`
        # has already reported above; these two read together.)
        if self.cfg.retrieval.sigma_expand:
            with_sigma = sum(1 for o in outcomes if o.n_sigma_facts > 0)
            if with_sigma == 0:
                checks.append(
                    "retrieval.sigma_expand está LIGADO mas nenhuma pergunta "
                    "recebeu fato algum pela órbita de sigma — a expansão está "
                    "inerte (vértices de grau 1? sigma_budget_frac pequeno "
                    "demais?) e estes números são os da G1"
                )
            elif with_sigma / n < 0.1:
                scanned = sum(o.sigma_scanned for o in outcomes)
                dup = sum(o.sigma_dup for o in outcomes)
                over = sum(o.sigma_over_budget for o in outcomes)
                if scanned and dup / scanned > 0.8:
                    why = (
                        f"{dup}/{scanned} candidatos já tinham sido trazidos pela "
                        "face — as órbitas são redundantes com phi (vizinhos de "
                        "grau 1?), e nesse grafo sigma não tem o que acrescentar"
                    )
                elif scanned and over / scanned > 0.5:
                    why = (
                        f"{over}/{scanned} candidatos foram cortados por orçamento "
                        "— suba sigma_budget_frac"
                    )
                elif scanned == 0:
                    why = (
                        "nenhum candidato sequer foi examinado — as órbitas estão "
                        "vazias, isto é, as entidades não estão sendo compartilhadas "
                        "entre memórias (problema de ingest, não de recuperação)"
                    )
                else:
                    why = "confira sigma_expand_k e sigma_budget_frac"
                checks.append(
                    f"apenas {with_sigma}/{n} perguntas usaram a expansão por "
                    f"sigma: {why}"
                )

        if self.cfg.retrieval.face_coverage:
            linked = sum(1 for o in outcomes if o.question_entities)
            with_cov = sum(1 for o in outcomes if o.n_coverage_facts > 0)
            if linked == 0:
                checks.append(
                    "face_coverage está LIGADO mas NENHUMA pergunta foi ligada a "
                    "um vértice — o linker não casa com os nomes do grafo (as "
                    "entidades do ingest saíram com outra grafia?) e a condição "
                    "está rodando como G1"
                )
            elif with_cov == 0:
                checks.append(
                    f"{linked}/{n} perguntas ligadas a entidades, mas nenhuma face "
                    "de cobertura entrou no contexto — confira coverage_budget_frac"
                )
            elif with_cov / n < 0.1:
                checks.append(
                    f"apenas {with_cov}/{n} perguntas usaram faces de cobertura "
                    f"(ligadas: {linked}/{n})"
                )

        return {
            "ok": not checks,
            "warnings": checks,
            "abstention_rate": round(abstained / n, 4),
            "distinct_predictions": len(distinct),
        }

    def _manifest(self) -> dict:
        from fgl import __version__
        from fgl.settings import load_settings

        return {
            "fgl_version": __version__,
            "config": self.cfg.to_dict(),
            "config_source": self.cfg.source,
            "prompts": self.prompts.manifest(),
            "environment": load_settings().redacted(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(self.root),
        }


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _refresh_graph_stats(graph: FatGraph, report: dict) -> dict:
    """Backfill statistics a *cached* report predates.

    Reports are written once, next to the graph, and never revisited -- so a
    graph built before a metric existed keeps a report without it, and any
    check reading that metric silently finds nothing to complain about.  That
    is the worst possible failure for a guard rail: it is quietest exactly on
    the old artefacts, which are the ones nobody re-examined.

    Recomputing is free and safe: ``stats()`` is a pure function of the graph
    that was just loaded, so this cannot disagree with a fresh ingest.  Only
    missing keys are filled -- whatever the ingest recorded stays untouched,
    since counters like ``n_collapses`` describe the *run*, not the artefact.
    """
    stats = report.get("graph_stats")
    if not isinstance(stats, dict):
        return report
    missing = {k: v for k, v in graph.stats().items() if k not in stats}
    if missing:
        report["graph_stats"] = {**stats, **missing}
    return report


def _usage_delta(before: dict, after: dict) -> dict:
    keys = ("calls", "cached_calls", "prompt_tokens", "completion_tokens")
    out = {k: after.get(k, 0) - before.get(k, 0) for k in keys}
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    out["by_purpose"] = {
        p: {k: v.get(k, 0) - before.get("by_purpose", {}).get(p, {}).get(k, 0) for k in keys}
        for p, v in after.get("by_purpose", {}).items()
    }
    return out


def _add(a: dict, b: dict) -> dict:
    keys = ("calls", "cached_calls", "prompt_tokens", "completion_tokens")
    out = {k: a.get(k, 0) + b.get(k, 0) for k in keys}
    out["by_purpose"] = a.get("by_purpose", {})
    return out


def _git_commit(root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_dataset(cfg: Config, root: str | Path | None = None) -> list[Conversation]:
    paths = Paths.build(root or project_root())
    path = paths.resolve(cfg.paths.data_file)
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo data file not found at {path}.\nRun `fgl setup` to fetch it."
        )
    return load_conversations(path)


def select_conversations(
    conversations: Sequence[Conversation],
    sample_ids: Sequence[str] | None = None,
    limit: int = 0,
) -> list[Conversation]:
    out = list(conversations)
    if sample_ids:
        wanted = {s.strip() for s in sample_ids if s.strip()}
        out = [c for c in out if c.sample_id in wanted]
        missing = wanted - {c.sample_id for c in out}
        if missing:
            raise ValueError(f"unknown sample_id(s): {sorted(missing)}")
    if limit:
        out = out[:limit]
    return out
