"""End-to-end CLIO3 read path."""

from __future__ import annotations

from datetime import datetime

from fgl.clio.access.state import AccessState
from fgl.clio.agent.loop import AgentStep, AgentTrace
from fgl.clio3.answer import answer_query
from fgl.clio3.planner import compile_question
from fgl.clio3.retrieval import OpenGraphRetriever


class Clio3Runtime:
    def __init__(self, memory):
        self.signature = self.signature_for(memory)
        self.retriever = OpenGraphRetriever(memory)

    @staticmethod
    def signature_for(memory):
        return (
            len(memory.log.all()),
            len(memory.clio3_store.entities()),
            len(memory.clio3_store.records()),
            len(memory.clio3_store.links()),
        )


def _runtime(memory) -> Clio3Runtime:
    runtime = getattr(memory, "_clio3_runtime", None)
    if runtime is None or runtime.signature != Clio3Runtime.signature_for(memory):
        runtime = Clio3Runtime(memory)
        memory._clio3_runtime = runtime
    memory.clio3_retriever = runtime.retriever
    return runtime


def run_clio3(question: str, memory) -> AgentTrace:
    runtime = _runtime(memory)
    query = compile_question(question, memory)
    result = runtime.retriever.execute(question, query)
    answer, structured = answer_query(question, result, memory)
    state = AccessState(
        trails=[],
        tx_point=datetime.now(),
        budget_used=3,
        candidate_episode_ids=tuple(result.episode_ids),
        evidence_ids=tuple(structured.support),
        query=question,
    )
    return AgentTrace(
        question=question,
        steps=[
            AgentStep(
                "compile",
                {
                    "operation": query.operation,
                    "answer_type": query.answer_type,
                    "focal_entities": list(query.focal_entities),
                    "concepts": list(query.concepts),
                    "max_hops": query.max_hops,
                },
                query.rationale,
            ),
            AgentStep(
                "retrieve",
                {
                    "records": [record.id for record, _ in result.records],
                    "candidate_episodes": result.episode_ids,
                },
                "reciprocal-rank retrieval plus event-graph expansion",
            ),
            AgentStep(
                "verify",
                {
                    "answer_type": structured.answer_type,
                    "support": list(structured.support),
                    "abstain": structured.abstain,
                },
                "episode provenance contract",
            ),
        ],
        final_state=state,
        answer=answer,
    )


__all__ = ["run_clio3"]
