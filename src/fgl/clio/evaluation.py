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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fgl.clio.catalog import Catalog, load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.facade import Clio
from fgl.data.locomo import Conversation, load_conversations
from fgl.evaluation.scorer import QAOutcome, aggregate, is_abstention, score_question
from fgl.llm.client import LLMClient
from fgl.llm.prompts import PromptLibrary
from fgl.retrieval.embeddings import Embedder


@dataclass
class ConversationResult:
    sample_id: str
    n_turns: int
    n_questions: int
    outcomes: list[QAOutcome] = field(default_factory=list)
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
                _turn_text(turn), speaker=turn.speaker, session_id=conv.sample_id, ts=ts
            )
            n_turns += 1
        clio.consolidate()  # spec's own "end of session" trigger (section 7)
    ingest_seconds = time.monotonic() - t0

    questions = conv.questions[:limit_questions] if limit_questions else conv.questions
    t0 = time.monotonic()
    outcomes: list[QAOutcome] = []
    for q in questions:
        prediction = clio.ask(q.prompt_question()).answer
        outcomes.append(
            QAOutcome(
                question=q.question,
                category=q.category,
                gold=q.answer,
                prediction=prediction,
                f1=score_question(q, prediction),
                evidence=q.evidence,
                n_facts=len(clio.graph.all_edges()),
                # Not measured: CLIO's answer isn't one retrieved "context"
                # the way a RAG condition's is -- it is the agent loop's
                # own trajectory of several prompts. 0 here is honest
                # (spec's own tokens_per_f1_point column degrades to 0.0
                # on an all-zero tokens_context, per aggregate()), not a
                # placeholder pretending to be a measurement.
                tokens_context=0,
                abstained=is_abstention(prediction),
            )
        )
    qa_seconds = time.monotonic() - t0

    result = ConversationResult(
        sample_id=conv.sample_id,
        n_turns=n_turns,
        n_questions=len(questions),
        outcomes=outcomes,
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

    all_outcomes: list[QAOutcome] = []
    per_conversation: list[dict] = []
    t0 = time.monotonic()
    for conv in conversations:
        result, clio = run_conversation(
            conv, catalog, llm, embedder, prompts, cfg, limit_questions
        )
        all_outcomes.extend(result.outcomes)
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
        if on_conversation_done:
            on_conversation_done(conv, result)

    wall_seconds = time.monotonic() - t0
    metrics = {
        "condition": condition_name,
        **aggregate(all_outcomes),
        "per_conversation": per_conversation,
        "cost": llm.usage.to_dict(),
        "wall_seconds": round(wall_seconds, 1),
    }

    out_dir = Path(results_dir) / condition_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for o in all_outcomes:
            f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")

    return metrics
