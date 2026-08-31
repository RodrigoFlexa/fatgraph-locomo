"""End-to-end CLIO2 read path."""

from __future__ import annotations

from datetime import datetime

from fgl.clio.access.state import AccessState
from fgl.clio.agent.loop import AgentStep, AgentTrace
from fgl.clio2.answer import answer_query
from fgl.clio2.executor import QueryExecutor
from fgl.clio2.ledger import FactIndex, SemanticLedger
from fgl.clio2.planner import compile_question


class Clio2Runtime:
    def __init__(self, memory):
        self.signature = self.signature_for(memory)
        self.ledger = SemanticLedger(
            memory, include_staged=memory.config.clio2.include_staged_facts
        )
        self.fact_index = FactIndex(self.ledger, memory.entity_index.embedder)
        self.executor = QueryExecutor(memory, self.ledger, self.fact_index)

    @staticmethod
    def signature_for(memory) -> tuple[int, int, int]:
        return (
            len(memory.log.all()),
            len(memory.staging.all()),
            len(memory.graph.all_edges()),
        )


def _runtime(memory) -> Clio2Runtime:
    runtime = getattr(memory, "_clio2_runtime", None)
    if runtime is None or runtime.signature != Clio2Runtime.signature_for(memory):
        runtime = Clio2Runtime(memory)
        memory._clio2_runtime = runtime
    return runtime


def run_clio2(question: str, memory) -> AgentTrace:
    runtime = _runtime(memory)
    plan = compile_question(question, memory)
    result = runtime.executor.execute(question, plan)
    answer, structured = answer_query(question, result, memory)
    selected = tuple(structured.support)
    state = AccessState(
        trails=[],
        tx_point=datetime.now(),
        budget_used=3,
        candidate_episode_ids=tuple(result.candidate_episode_ids),
        evidence_ids=selected,
        query=question,
    )
    return AgentTrace(
        question=question,
        steps=[
            AgentStep(
                action="compile",
                args={
                    "operator": plan.operator.value,
                    "subjects": list(plan.subjects),
                    "relations": list(plan.relations),
                    "answer_type": plan.answer_type.value,
                    "terms": list(plan.constraints.terms),
                    "start": plan.constraints.start.isoformat()
                    if plan.constraints.start
                    else None,
                    "end": plan.constraints.end.isoformat()
                    if plan.constraints.end
                    else None,
                },
                reason=plan.rationale,
            ),
            AgentStep(
                action="execute",
                args={
                    "n_facts": len(result.candidate_facts),
                    "n_values": len(result.items),
                    "diagnostics": result.diagnostics,
                    "candidate_episodes": result.candidate_episode_ids,
                },
                reason="deterministic query algebra",
            ),
            AgentStep(
                action="verify",
                args={
                    "answer_type": structured.answer_type.value,
                    "support": list(structured.support),
                    "abstain": structured.abstain,
                },
                reason="evidence contract",
            ),
        ],
        final_state=state,
        count_result=result.scalar if type(result.scalar) is int else None,
        answer=answer,
    )


__all__ = ["Clio2Runtime", "run_clio2"]
