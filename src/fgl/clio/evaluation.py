"""LoCoMo evaluation for CLIO (spec's own M9 milestone): ingest every
session of a conversation, answer every official question, score with the
SAME scorer every other condition in this repository uses
(:mod:`fgl.evaluation.scorer`), and write
``results/<condition>/{metrics.json,predictions.jsonl}`` in the exact
shape ``fgl report`` already reads.

No registration anywhere else is needed: ``fgl.evaluation.report.
load_results`` is a pure directory glob over ``results/*/metrics.json``,
and the directory name IS the condition name (verified by reading that
function directly, not assumed) -- nothing in ``fgl.config``'s condition
YAML list or ``_MODEL_PAIRS`` validation has to know CLIO exists.

One fresh :class:`~fgl.clio.facade.Clio` memory per conversation, matching
every other condition in this repository: different conversations have
different people, and folding "Melanie" from one into an unrelated
"Melanie" from another would be a real error, not a feature.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from fgl.clio.access.movements import evidence as clio_evidence
from fgl.clio.catalog import Catalog, load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.facade import Clio
from fgl.data.locomo import Conversation, load_conversations
from fgl.evaluation.scorer import (
    QAOutcome,
    aggregate,
    evidence_recall,
    is_abstention,
    score_question,
)
from fgl.llm.client import LLMClient
from fgl.llm.prompts import PromptLibrary
from fgl.retrieval.embeddings import Embedder


@dataclass
class ConversationResult:
    sample_id: str
    n_turns: int
    n_questions: int
    outcomes: list[QAOutcome] = field(default_factory=list)
    #: one record per question of what the agent actually DID -- the
    #: movements it chose, why, and what died. `AgentTrace` was being
    #: built and thrown away, so a bad F1 could not be attributed to a
    #: movement: "0.26" said the reader failed without saying where. This
    #: is what makes the next fix measurable instead of guessed.
    traces: list[dict] = field(default_factory=list)
    ingest_seconds: float = 0.0
    qa_seconds: float = 0.0
    n_entities: int = 0
    n_edges: int = 0
    n_folds: int = 0


def _turn_text(turn) -> str:
    """Matches the official rendering's caption handling (``Turn.rendered``)
    without duplicating the speaker prefix CLIO's own ``ingest_turn``
    already adds from the ``speaker=`` argument."""
    text = turn.text
    if turn.img_caption:
        text = f"{text} [shares {turn.img_caption}]"
    return text


def run_conversation(
    conv: Conversation,
    catalog: Catalog,
    llm: LLMClient,
    embedder: Embedder,
    prompts: PromptLibrary,
    config: ClioConfig,
    limit_questions: int | None = None,
) -> tuple[ConversationResult, Clio]:
    clio = Clio(catalog, llm, embedder, prompts, config)

    t0 = time.monotonic()
    n_turns = 0
    for session in conv.sessions:
        ts = datetime.fromisoformat(session.timestamp)
        for turn in session.turns:
            clio.ingest(
                _turn_text(turn),
                speaker=turn.speaker,
                session_id=conv.sample_id,
                ts=ts,
                episode_id=turn.dia_id,
            )
            n_turns += 1
        clio.consolidate()  # spec's own "end of session" trigger (section 7)
    ingest_seconds = time.monotonic() - t0

    questions = conv.questions[:limit_questions] if limit_questions else conv.questions
    t0 = time.monotonic()
    outcomes: list[QAOutcome] = []
    traces: list[dict] = []
    for q in questions:
        calls_before = llm.usage.calls
        prompt_tokens_before = llm.usage.prompt_tokens
        completion_tokens_before = llm.usage.completion_tokens
        trace = clio.ask(q.prompt_question())
        prediction = trace.answer
        state = trace.final_state
        evidence_episodes = clio_evidence(state, clio.staging, clio.log)
        retrieved_turn_ids = [episode.id for episode in evidence_episodes]
        prompt_tokens = llm.usage.prompt_tokens - prompt_tokens_before
        completion_tokens = llm.usage.completion_tokens - completion_tokens_before
        outcomes_trace = {
            "question": q.question,
            "category": q.category_name,
            "gold": q.answer,
            "prediction": prediction,
            "evidence": list(q.evidence),
            "movements": [
                {"action": s.action, "args": s.args, "reason": s.reason}
                for s in trace.steps
            ],
            "n_movements": len(trace.steps),
            "budget_used": state.budget_used,
            "live_trails": len(state.trails),
            "dead_trails": state.dead_count,
            "death_cause": state.death_cause,
            "valid_restricted": state.valid_restricted,
            # the episodes the answer was actually written from (P5) --
            # an empty list with a non-abstaining answer means the model
            # answered from nothing
            "evidence_episodes": retrieved_turn_ids,
            "count_result": trace.count_result,
            "llm_calls": llm.usage.calls - calls_before,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        traces.append(outcomes_trace)
        outcomes.append(
            QAOutcome(
                question=q.question,
                category=q.category,
                gold=q.answer,
                prediction=prediction,
                f1=score_question(q, prediction),
                evidence=q.evidence,
                retrieved_turn_ids=retrieved_turn_ids,
                recall={"recall_context": evidence_recall(q.evidence, retrieved_turn_ids)},
                n_facts=len(clio.graph.all_edges()),
                # For an agentic reader the comparable finite resource is
                # every prompt token spent deciding and answering this
                # question, not a fictitious single RAG context.
                tokens_context=prompt_tokens,
                abstained=is_abstention(prediction),
            )
        )
    qa_seconds = time.monotonic() - t0

    result = ConversationResult(
        sample_id=conv.sample_id,
        n_turns=n_turns,
        n_questions=len(questions),
        outcomes=outcomes,
        traces=traces,
        ingest_seconds=ingest_seconds,
        qa_seconds=qa_seconds,
        n_entities=len(clio.graph.all_entities()),
        n_edges=len(clio.graph.all_edges()),
        n_folds=len(clio.journal.all()),
    )
    return result, clio


def run_benchmark(
    data_file: str | Path,
    llm: LLMClient,
    embedder: Embedder,
    config: ClioConfig | None = None,
    limit_conversations: int | None = None,
    limit_questions: int | None = None,
    results_dir: str | Path = "results",
    condition_name: str = "CLIO",
    prompts_dir: str | Path | None = None,
    on_conversation_done: Callable[[Conversation, ConversationResult], None] | None = None,
    save_memory_snapshots: bool = True,
) -> dict:
    """Runs every (or the first ``limit_conversations``) LoCoMo
    conversation through CLIO end to end and writes the results directory
    ``fgl report`` reads. Returns the ``metrics.json`` dict.
    """
    if prompts_dir is None:
        from fgl.paths import Paths, project_root

        prompts_dir = Paths.build(project_root()).prompts
    prompts = PromptLibrary(prompts_dir)

    cfg = config or ClioConfig.default()
    catalog = load_catalog(cfg.catalog_path)
    conversations: Sequence[Conversation] = load_conversations(data_file)
    if limit_conversations:
        conversations = conversations[:limit_conversations]

    out_dir = Path(results_dir) / condition_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_outcomes: list[QAOutcome] = []
    all_traces: list[dict] = []
    per_conversation: list[dict] = []
    t0 = time.monotonic()
    for conv in conversations:
        result, clio = run_conversation(
            conv, catalog, llm, embedder, prompts, cfg, limit_questions
        )
        all_outcomes.extend(result.outcomes)
        all_traces.extend(result.traces)
        conv_f1 = (
            sum(o.f1 for o in result.outcomes) / len(result.outcomes)
            if result.outcomes
            else 0.0
        )
        per_conversation.append(
            {
                "sample_id": result.sample_id,
                "n_turns": result.n_turns,
                "n_questions": result.n_questions,
                "f1": round(conv_f1, 4),
                "n_entities": result.n_entities,
                "n_edges": result.n_edges,
                "n_folds": result.n_folds,
                "ingest_seconds": round(result.ingest_seconds, 1),
                "qa_seconds": round(result.qa_seconds, 1),
            }
        )
        if save_memory_snapshots:
            from fgl.clio.persist import save_memory

            save_memory(clio, out_dir / f"memory_{conv.sample_id}.json")
        if on_conversation_done:
            on_conversation_done(conv, result)

    wall_seconds = time.monotonic() - t0
    metrics = {
        "condition": condition_name,
        **aggregate(all_outcomes),
        "per_conversation": per_conversation,
        "cost": llm.usage.to_dict(),
        "clio_run": {
            "config": asdict(cfg),
            "llm_provider": getattr(llm.cfg, "provider", type(llm).__name__),
            "llm_deployment": getattr(llm.cfg, "deployment", ""),
            "embedder": type(embedder).__name__,
            "embedder_tag": getattr(embedder, "tag", ""),
            "memory_snapshots": save_memory_snapshots,
        },
        "wall_seconds": round(wall_seconds, 1),
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for o in all_outcomes:
            f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")
    # separate file, not extra columns in predictions.jsonl: that file's
    # schema is shared with every other condition in this repository and
    # `fgl report` reads it, so widening it here would be a cross-cutting
    # change to serve one condition's debugging
    with (out_dir / "traces.jsonl").open("w", encoding="utf-8") as f:
        for tr in all_traces:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")

    return metrics
