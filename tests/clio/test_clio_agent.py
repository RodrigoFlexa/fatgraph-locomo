"""Milestone M8 (the agent loop + answer generation), offline: a scripted
:class:`FakeLLM` plays both roles an LLM has in this package -- the
extractor (M5) and the agent (M8) -- so the whole path from raw turn text
to a synthesised answer runs and is checked without any real API call.

The script is keyed on the ``# TASK: <marker>`` line every prompt in this
repository carries (see :func:`fgl.llm.client._task_marker`), not on
prompt content, so it cannot accidentally cross-match the way a bare
substring check on turn text did during development (see
``test_clio_ingest.py``'s ``_scripted_responder`` for that lesson).
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import pytest

from fgl.clio.config import ClioConfig
from fgl.clio.facade import Clio
from fgl.config import LLMConfig
from fgl.llm.client import FakeLLM
from fgl.retrieval.embeddings import HashingEmbedder


def _task(prompt: str) -> str:
    m = re.search(r"^#\s*TASK:\s*([a-z_]+)\s*$", prompt, flags=re.MULTILINE)
    return m.group(1) if m else ""


_EXTRACT_SCRIPT: dict[str, list[dict]] = {
    "I started at Vertex this week, I'm living in Recife": [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "works_at",
            "object_id": "new:Vertex",
            "polarity": True,
            "time_expression": "this week",
            "evidence_kind": "literal",
            "span": "I started at Vertex this week",
        },
    ],
    "I left Vertex, joined Kaia": [
        {
            "operation": "close",
            "subject_id": "new:Melanie",
            "relation": "works_at",
            "object_id": "new:Vertex",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "I left Vertex",
        },
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "works_at",
            "object_id": "new:Kaia",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "joined Kaia",
        },
    ],
}


def _extract_responder(prompt: str, system):
    for turn_text, facts in _EXTRACT_SCRIPT.items():
        if f'THIS TURN:\n"{turn_text}"' in prompt:
            return json.dumps(facts)
    return "[]"


class ScriptedAgent:
    """Plays both the extractor and the agent role for one Clio instance,
    dispatched by TASK marker. The agent script is a fixed, ordered
    sequence -- this test is about the LOOP's plumbing (does each action
    reach the right movement, does the answer step see the right facts),
    not about whether an LLM makes good choices."""

    def __init__(self, agent_steps: list[dict], final_answer: str):
        self.agent_steps = list(agent_steps)
        self.final_answer = final_answer
        self._agent_call = 0
        self.answer_prompts: list[str] = []

    def __call__(self, prompt: str, system):
        task = _task(prompt)
        if task == "clio_extract":
            return _extract_responder(prompt, system)
        if task == "clio_agent":
            step = self.agent_steps[min(self._agent_call, len(self.agent_steps) - 1)]
            self._agent_call += 1
            return json.dumps(step)
        if task == "clio_answer":
            self.answer_prompts.append(prompt)
            return self.final_answer
        return "[]"


def _build_clio(scripted: ScriptedAgent) -> Clio:
    llm = FakeLLM(LLMConfig(provider="fake", cache_enabled=False), responder=scripted)
    return Clio.build(
        config=ClioConfig.default(), llm=llm, embedder=HashingEmbedder(dim=64)
    )


def test_agent_loop_anchors_restricts_follows_and_answers():
    """Mirrors spec T1's own trace (anchor -> restrict(valid, today) ->
    follow): a naive anchor -> follow with no temporal restriction would
    correctly surface BOTH the current and the former employer, since
    nothing has told the trail which instant to care about yet -- that is
    ``follow``'s invariant working as intended, not a bug this test should
    paper over by only asking for one relation hop."""
    scripted = ScriptedAgent(
        agent_steps=[
            {"action": "anchor", "args": {"text": "Melanie"}, "reason": "start"},
            {
                "action": "restrict",
                "args": {"axis": "valid", "start": "2024-01-01", "end": "2024-01-02"},
                "reason": "the question asks about now",
            },
            {
                "action": "follow",
                "args": {"label": "works_at"},
                "reason": "the question asks about work",
            },
            {"action": "answer", "args": {}, "reason": "have a live trail"},
        ],
        final_answer="Kaia",
    )
    clio = _build_clio(scripted)
    clio.ingest(
        "I started at Vertex this week, I'm living in Recife",
        "Melanie",
        "s1",
        datetime(2023, 1, 14),
    )
    clio.consolidate()
    clio.ingest("I left Vertex, joined Kaia", "Melanie", "s1", datetime(2023, 9, 5))
    clio.consolidate()

    trace = clio.ask("Where does Melanie work now?")

    assert [s.action for s in trace.steps] == ["anchor", "restrict", "follow", "answer"]
    assert trace.answer == "Kaia"
    assert len(trace.final_state.trails) == 1
    kaia = next(e for e in clio.graph.all_entities() if e.canonical_name == "Kaia")
    assert trace.final_state.trails[0].vertex_id == kaia.id
    # the answer prompt actually carried the live fact, not an empty state
    assert "Kaia" in scripted.answer_prompts[0]


def test_agent_loop_stops_on_an_unknown_label_and_still_answers():
    scripted = ScriptedAgent(
        agent_steps=[
            {"action": "anchor", "args": {"text": "Melanie"}, "reason": "start"},
            {"action": "follow", "args": {"label": "not_a_real_relation"}, "reason": "bad"},
        ],
        final_answer="Not mentioned in the conversation",
    )
    clio = _build_clio(scripted)
    clio.ingest(
        "I started at Vertex this week, I'm living in Recife",
        "Melanie",
        "s1",
        datetime(2023, 1, 14),
    )
    clio.consolidate()

    trace = clio.ask("What does Melanie eat for breakfast?")
    # the bad follow() call stops movement rather than crashing the question
    assert trace.answer == "Not mentioned in the conversation"


def test_count_action_returns_the_number_directly_without_an_llm_call():
    """spec T6: a count answer IS the number -- never synthesised, never
    at risk of the LLM rounding or hedging it."""
    scripted = ScriptedAgent(
        agent_steps=[
            {"action": "count", "args": {"entity": "climbing"}, "reason": "how many times"}
        ],
        final_answer="should never be called",
    )
    clio = _build_clio(scripted)
    clio.mentions.append(episode_id="e1", surface="climbing", ts=datetime(2023, 3, 2))
    clio.mentions.append(episode_id="e2", surface="climbing", ts=datetime(2023, 11, 11))

    trace = clio.ask("How many times did she mention climbing?")
    assert trace.answer == "2"
    assert trace.count_result == 2
    assert scripted.answer_prompts == []  # the answer LLM call never happened


@pytest.mark.parametrize(
    "action",
    ["anchor", "follow", "restrict", "filter", "expand", "history", "count", "answer"],
)
def test_every_documented_action_is_recognised(action):
    """A decision naming any of spec 10.1's eight movements must not be
    treated as malformed -- checked directly against the loop's own
    action whitelist rather than trusting the prompt text to stay in
    sync with the code."""
    from fgl.clio.agent.loop import _ACTIONS

    assert action in _ACTIONS
