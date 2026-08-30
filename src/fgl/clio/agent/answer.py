"""Answer generation (spec section 11). Three blocks, in priority order:
episode text (the source of truth), structured facts (only there to say
WHEN to trust the text), and a diagnosis (only relevant when there is no
live evidence at all) -- P5: the answer is written from the episode, never
from the proposition that merely located it.
"""

from __future__ import annotations

from fgl.clio.access.movements import evidence
from fgl.clio.access.state import AccessState
from fgl.clio.graph.store import GraphStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.types import Interval
from fgl.llm.client import LLMClient
from fgl.llm.prompts import SYSTEM_ANSWERER, PromptLibrary


def _render_episodes(episodes) -> str:
    if not episodes:
        return "(no evidence retrieved)"
    return "\n".join(
        f"[{e.id}; {e.ts_ingest.strftime('%d %B %Y')}] {e.speaker}: {e.text}"
        for e in episodes
    )


def _format_window(window: Interval) -> str:
    start = window.start.strftime("%d %B %Y") if window.start else "the beginning"
    end = window.end.strftime("%d %B %Y") if window.end else "now"
    hedge = " (approximate)" if window.granularity in ("month", "year") else ""
    return f"{start} -> {end}{hedge}"


def _render_facts(state: AccessState, graph: GraphStore, staging: StagingStore) -> str:
    if not state.trails:
        return "(no live facts)"
    lines = []
    seen: set[str] = set()
    for t in state.trails:
        for proposition_id in t.path:
            if proposition_id in seen:
                continue
            seen.add(proposition_id)
            try:
                proposition = staging.get(proposition_id)
                subject = graph.get_entity(proposition.subject_id).canonical_name
                object_ = graph.get_entity(proposition.object_id).canonical_name
            except KeyError:
                continue
            polarity = "" if proposition.polarity else "NOT "
            lines.append(
                f"- {subject} --{polarity}{proposition.relation}--> {object_}; "
                f"valid {_format_window(proposition.t_valid)}; source {proposition.episode_id}"
            )
    if not lines:
        for trail in state.trails:
            entity = graph.get_entity(trail.vertex_id)
            via = "/".join(trail.labels) or "anchor"
            lines.append(
                f"- {entity.canonical_name} ({entity.type}), "
                f"valid {_format_window(trail.window)}, via {via}"
            )
    return "\n".join(lines)


def _render_diagnosis(state: AccessState) -> str:
    if state.is_alive:
        return "(not needed -- live evidence above)"
    cause = state.death_cause or "no movement has been made yet"
    return f"No live trail survived. Cause: {cause}."


def generate_answer(
    llm: LLMClient,
    prompts: PromptLibrary,
    question: str,
    state: AccessState,
    graph: GraphStore,
    staging: StagingStore,
    log: LogStore,
    max_episodes: int = 12,
) -> str:
    episodes = evidence(state, staging, log)[:max_episodes]
    prompt = prompts.render(
        "clio_answer",
        question=question,
        episodes=_render_episodes(episodes),
        facts=_render_facts(state, graph, staging),
        diagnosis=_render_diagnosis(state),
    )
    return llm.complete(
        prompt, system=SYSTEM_ANSWERER, purpose="clio_answer", max_tokens=64
    ).strip()
