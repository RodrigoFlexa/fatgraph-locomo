"""The agent loop (spec 10.3): no router. The LLM composes movements
itself, one at a time, seeing only ``available_labels`` and
``death_cause`` at each step -- never a prior classification of the
question. Every decision is a small JSON object; the movement it names is
plain code (P3, applied to reading the same way M4 applies it to
writing).

``memory`` is typed loosely (``Any``) rather than imported as
:class:`fgl.clio.facade.Clio`: the facade's ``.ask()`` method IS this
loop, so importing it back here would cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fgl.clio.access.movements import (
    anchor,
    count,
    expand,
    filter_trails,
    follow,
    history,
    restrict,
    select_evidence,
)
from fgl.clio.access.render import render_state
from fgl.clio.access.state import AccessState, Trail
from fgl.clio.agent.answer import generate_answer
from fgl.clio.graph.queries import UnknownLabel
from fgl.clio.types import Interval

SYSTEM_AGENT = (
    "You control a small set of memory-access movements over an explicit "
    "state. You respond only with valid JSON, never with prose."
)

_ACTIONS = (
    "anchor",
    "follow",
    "restrict",
    "filter",
    "expand",
    "history",
    "evidence",
    "count",
    "answer",
)


@dataclass
class AgentStep:
    action: str
    args: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class AgentTrace:
    question: str
    steps: list[AgentStep] = field(default_factory=list)
    final_state: AccessState = field(default_factory=lambda: AccessState(trails=[]))
    #: set only when the terminal action was `count` -- a count answer is
    #: the number itself (spec T6), never synthesised by an LLM call that
    #: could round it, hedge it, or simply get it wrong.
    count_result: int | None = None
    answer: str = ""


def _parse_date(s: str | None) -> datetime | None:
    return datetime.strptime(s, "%Y-%m-%d") if s else None


def _decide(memory: Any, question: str, state: AccessState) -> dict:
    rendered = render_state(
        state,
        memory.graph,
        memory.catalog,
        memory.config.access.movement_budget,
        log=memory.log,
    )
    prompt = memory.prompts.render(
        "clio_agent", question=question, state_json=json.dumps(rendered, indent=2)
    )
    decision = memory.llm.complete_json(
        prompt,
        system=SYSTEM_AGENT,
        purpose="clio_agent",
        default={"action": "answer", "args": {}, "reason": "fallback: unparsable decision"},
    )
    if not isinstance(decision, dict) or decision.get("action") not in _ACTIONS:
        return {"action": "answer", "args": {}, "reason": "fallback: malformed decision"}
    return decision


def _apply_movement(
    action: str, args: dict, state: AccessState, memory: Any
) -> AccessState:
    if action == "anchor":
        anchored = anchor(
            args.get("text", ""),
            memory.graph,
            index=memory.entity_index,
            episode_index=memory.episode_index,
            episode_k=memory.config.access.anchor_episode_k,
            tx_point=state.tx_point,
        )
        if not state.query:
            return anchored
        # A later anchor adds a second entry point; it must not erase the
        # path, temporal restriction, evidence, or budget already spent.
        by_vertex = {t.vertex_id: t for t in state.trails}
        for trail in anchored.trails:
            previous = by_vertex.get(trail.vertex_id)
            if previous is None or trail.score > previous.score:
                by_vertex[trail.vertex_id] = trail
        candidate_episode_ids = tuple(
            dict.fromkeys((*state.candidate_episode_ids, *anchored.candidate_episode_ids))
        )
        return AccessState(
            trails=list(by_vertex.values()),
            tx_point=state.tx_point,
            dead_count=state.dead_count,
            death_cause=state.death_cause,
            budget_used=state.budget_used + 1,
            valid_restricted=state.valid_restricted,
            candidate_episode_ids=candidate_episode_ids,
            evidence_ids=state.evidence_ids,
            query=state.query,
        )
    if action == "follow":
        return follow(state, args["label"], memory.graph, memory.catalog)
    if action == "restrict":
        axis = args.get("axis", "valid")
        interval = Interval(_parse_date(args.get("start")), _parse_date(args.get("end")))
        return restrict(state, axis, interval)
    if action == "filter":
        return filter_trails(
            state, memory.graph, name=args.get("name"), type=args.get("type")
        )
    if action == "expand":
        cfg = memory.config.access
        return expand(
            state,
            memory.graph,
            k=cfg.expand_max_hops,
            expand_k=cfg.expand_k,
            alpha=cfg.ppr_alpha,
        )
    if action == "history":
        entries = history(state, args["label"], memory.graph, memory.catalog)
        trails = [
            Trail(
                vertex_id=e.vertex_id,
                window=e.t_valid,
                path=e.provenance,
                labels=(args["label"],),
            )
            for e in entries
        ]
        return AccessState(
            trails=trails,
            tx_point=state.tx_point,
            budget_used=state.budget_used + 1,
            valid_restricted=state.valid_restricted,
            candidate_episode_ids=state.candidate_episode_ids,
            evidence_ids=state.evidence_ids,
            query=state.query,
        )
    if action == "evidence":
        episode_ids = args.get("episode_ids", [])
        if not isinstance(episode_ids, list) or not all(
            isinstance(episode_id, str) for episode_id in episode_ids
        ):
            raise ValueError("evidence.episode_ids must be a list of strings")
        return select_evidence(state, episode_ids)
    raise ValueError(f"{action!r} is not a state-producing movement")


def _prune_state(
    state: AccessState, memory: Any, episode_scores: dict[str, float]
) -> AccessState:
    """Rank valid trails by question relevance, then enforce a finite fan-out."""
    ranked: list[Trail] = []
    for trail in state.trails:
        scores = []
        for proposition_id in trail.path:
            try:
                episode_id = memory.staging.get(proposition_id).episode_id
            except KeyError:
                continue
            scores.append(episode_scores.get(episode_id, 0.0))
        score = max([trail.score, *scores])
        ranked.append(
            Trail(
                trail.vertex_id,
                trail.window,
                trail.path,
                trail.labels,
                score,
            )
        )
    ranked.sort(key=lambda trail: (-trail.score, len(trail.path), trail.vertex_id))
    candidate_episode_ids = tuple(
        sorted(
            dict.fromkeys(state.candidate_episode_ids),
            key=lambda episode_id: -episode_scores.get(episode_id, 0.0),
        )
    )
    return AccessState(
        trails=ranked[: memory.config.access.trail_limit],
        tx_point=state.tx_point,
        dead_count=state.dead_count,
        death_cause=state.death_cause,
        budget_used=state.budget_used,
        valid_restricted=state.valid_restricted,
        candidate_episode_ids=candidate_episode_ids,
        evidence_ids=state.evidence_ids,
        query=state.query,
    )


def run_agent_loop(question: str, memory: Any) -> AgentTrace:
    trace = AgentTrace(question=question)
    state = AccessState(trails=[], tx_point=datetime.now())
    budget = memory.config.access.movement_budget
    episode_scores = {
        episode.id: score
        for episode, score in memory.episode_index.search_scored(
            question, k=min(50, len(memory.log.all())), min_score=-1.0
        )
    }

    for _ in range(budget):
        decision = _decide(memory, question, state)
        action = decision["action"]
        args = decision.get("args") or {}
        trace.steps.append(
            AgentStep(action=action, args=args, reason=decision.get("reason", ""))
        )

        if action == "answer":
            break
        if action == "count":
            trace.count_result = count(
                memory.mentions,
                memory.graph,
                entity=args.get("entity"),
                surface=args.get("surface"),
                start=_parse_date(args.get("start")),
                end=_parse_date(args.get("end")),
            )
            break
        try:
            state = _apply_movement(action, args, state, memory)
            state = _prune_state(state, memory, episode_scores)
        except (UnknownLabel, KeyError, ValueError):
            # A bad tool call (unknown label, missing arg) stops movement
            # rather than crashing the whole question -- the agent answers
            # from whatever it already has, same as running out of budget.
            break
        if state.budget_used >= budget:
            break

    trace.final_state = state
    if trace.count_result is not None:
        trace.answer = str(trace.count_result)
    else:
        trace.answer = generate_answer(
            memory.llm,
            memory.prompts,
            question,
            state,
            memory.graph,
            memory.staging,
            memory.log,
            max_episodes=memory.config.access.answer_evidence_limit,
        )
    return trace
