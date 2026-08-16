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

from fgl.config import Config
from fgl.core import FatGraph
from fgl.data.locomo import Conversation, load_conversations
from fgl.evaluation import QAOutcome, aggregate, evidence_recall, score_question
from fgl.llm import LLMClient, PromptLibrary, build_llm
from fgl.logging_utils import JsonlLogger
from fgl.memory.ingest import Ingestor
from fgl.paths import Paths, project_root
from fgl.retrieval import Answerer, CachedEmbedder, Embedder, FaceRetriever, build_embedder

FATGRAPH_CONDITIONS = ("G1-fatgraph-min", "G2-fatgraph-cur", "G3-fatgraph-agent")
BASELINE_CONDITIONS = ("B1-full-context", "B2-rag-turns", "B3-rag-facts")

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
        metrics["sanity"] = self._sanity(outcomes)
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
    def _graph_path(self, conv: Conversation) -> Path:
        return (
            self.paths.resolve(self.cfg.paths.graphs_dir)
            / self.cfg.condition
            / conv.sample_id
        )

    def _ingest(self, conv: Conversation, force: bool = False) -> tuple[FatGraph, dict]:
        graph_path = self._graph_path(conv)
        report_path = graph_path.with_name(graph_path.name + ".report.json")
        if graph_path.with_suffix(".json").exists() and report_path.exists() and not force:
            return FatGraph.load(graph_path), json.loads(
                report_path.read_text(encoding="utf-8")
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

    def _sanity(self, outcomes: Sequence[QAOutcome]) -> dict:
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
