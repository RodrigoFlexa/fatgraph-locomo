"""A scripted :class:`~fgl.llm.client.FakeLLM` responder for ``fgl clio
demo --fake``. Not a real extractor or a real agent -- a canned replay of
the exact facts :mod:`tests.fixtures.melanie` already proves this
pipeline handles correctly, plus a generic best-effort fallback for
questions/turns the script does not recognise (an arbitrary ``--question``,
or a ``--locomo`` conversation). The point of ``--fake`` is to prove the
PLUMBING end to end for zero cost, not to demonstrate extraction quality
-- that needs a real deployment (``fgl clio demo``, no ``--fake``).
"""

from __future__ import annotations

import json
import re

_TASK_RE = re.compile(r"^#\s*TASK:\s*([a-z_]+)\s*$", re.MULTILINE)

# Mirrors tests/fixtures/melanie.yaml's hand-authored propositions for the
# exact wording of cli.py's `_DEMO_CONVERSATION` -- already proven correct
# by test_clio_melanie_fixture.py, reused here rather than re-derived.
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
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "lives_in",
            "object_id": "new:Recife",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "I'm living in Recife",
        },
    ],
    "My manager here is Bia, she also likes climbing": [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "managed_by",
            "object_id": "new:Bia",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "My manager here is Bia",
        },
        {
            "operation": "assert",
            "subject_id": "new:Bia",
            "relation": "practices",
            "object_id": "new:climbing",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "she also likes climbing",
        },
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "practices",
            "object_id": "new:climbing",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "implicature",
            "span": "she also likes climbing",
        },
    ],
    "I moved to Salvador last month, I'm still at Vertex remotely": [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "lives_in",
            "object_id": "new:Salvador",
            "polarity": True,
            "time_expression": "last month",
            "evidence_kind": "literal",
            "span": "I moved to Salvador last month",
        },
        {
            "operation": "reassert",
            "subject_id": "new:Melanie",
            "relation": "works_at",
            "object_id": "new:Vertex",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "I'm still at Vertex remotely",
        },
    ],
    "I left Vertex, joined Kaia. My boss now is Rui": [
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
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "managed_by",
            "object_id": "new:Rui",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "My boss now is Rui",
        },
    ],
    "Went climbing again over the weekend": [
        {
            "operation": "assert",
            "subject_id": "new:Melanie",
            "relation": "practices",
            "object_id": "new:climbing",
            "polarity": True,
            "time_expression": "over the weekend",
            "evidence_kind": "implicature",
            "span": "Went climbing again over the weekend",
        },
    ],
    "Actually Bia was never my manager, she was on another team": [
        {
            "operation": "retract",
            "subject_id": "new:Melanie",
            "relation": "managed_by",
            "object_id": "new:Bia",
            "polarity": True,
            "time_expression": None,
            "evidence_kind": "literal",
            "span": "Bia was never my manager",
        },
    ],
}

# Canned agent trajectories for the CLI's own built-in demo questions
# (fgl/clio/cli.py's _DEMO_QUESTIONS). Anything else falls back to
# `_generic_agent_step`, a shallow heuristic that exercises the loop
# without pretending to be a real decision-maker.
_AGENT_SCRIPTS: dict[str, list[dict]] = {
    "Where does Melanie work now?": [
        {"action": "anchor", "args": {"text": "Melanie"}, "reason": "start"},
        {
            "action": "restrict",
            "args": {"axis": "valid", "start": "2024-06-01", "end": "2024-06-02"},
            "reason": "asking about now",
        },
        {
            "action": "follow",
            "args": {"label": "works_at"},
            "reason": "the question names the relation",
        },
        {"action": "answer", "args": {}, "reason": "one live trail"},
    ],
    "Where did Melanie live in February 2023?": [
        {"action": "anchor", "args": {"text": "Melanie"}, "reason": "start"},
        {
            "action": "restrict",
            "args": {"axis": "valid", "start": "2023-02-01", "end": "2023-02-28"},
            "reason": "the question names the period",
        },
        {
            "action": "follow",
            "args": {"label": "lives_in"},
            "reason": "the question names the relation",
        },
        {"action": "answer", "args": {}, "reason": "one live trail"},
    ],
    "How many times has Melanie mentioned climbing?": [
        {
            "action": "count",
            "args": {"entity": "climbing"},
            "reason": "a counting question",
        },
    ],
}


def _task(prompt: str) -> str:
    m = _TASK_RE.search(prompt)
    return m.group(1) if m else ""


def _generic_agent_step(prompt: str, call_index: int) -> dict:
    """Anchor once on the question text, follow the first available label
    if there is one, then answer -- a shallow, honest fallback for a
    question the script above does not recognise."""
    if call_index == 0:
        m = re.search(r"^QUESTION:\s*(.+)$", prompt, flags=re.MULTILINE)
        return {
            "action": "anchor",
            "args": {"text": m.group(1) if m else ""},
            "reason": "generic anchor",
        }
    m = re.search(r'"available_labels":\s*\[(.*?)\]', prompt, flags=re.DOTALL)
    labels = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    if labels and call_index == 1:
        return {
            "action": "follow",
            "args": {"label": labels[0]},
            "reason": "generic follow",
        }
    return {"action": "answer", "args": {}, "reason": "generic: out of ideas"}


def demo_fake_responder():
    """Returns a fresh ``responder(prompt, system) -> str`` closure with
    its own per-question call counter (a new :class:`FakeLLM` should get
    a fresh one per ``Clio`` instance, not a shared global)."""
    agent_calls: dict[str, int] = {}

    def responder(prompt: str, system) -> str:
        task = _task(prompt)
        if task == "clio_extract":
            for turn_text, facts in _EXTRACT_SCRIPT.items():
                if f'THIS TURN:\n"{turn_text}"' in prompt:
                    return json.dumps({"propositions": facts})
            return "[]"
        if task == "clio_agent":
            m = re.search(r"^QUESTION:\s*(.+)$", prompt, flags=re.MULTILINE)
            question = m.group(1) if m else ""
            if question in _AGENT_SCRIPTS:
                steps = _AGENT_SCRIPTS[question]
                idx = agent_calls.get(question, 0)
                agent_calls[question] = idx + 1
                return json.dumps(steps[min(idx, len(steps) - 1)])
            idx = agent_calls.get(question, 0)
            agent_calls[question] = idx + 1
            return json.dumps(_generic_agent_step(prompt, idx))
        if task == "clio_answer":
            # The facts block sits between the (two-line) "STRUCTURED
            # FACTS (...)" header and "DIAGNOSIS" -- anchored on the
            # header's own closing text, not "^", since it is never at
            # the very start of the prompt.
            m = re.search(r"resolved\):\n(.*?)\n\nDIAGNOSIS", prompt, flags=re.DOTALL)
            facts = (m.group(1) if m else "").strip()
            if facts and facts != "(no live facts)":
                first = facts.splitlines()[0].lstrip("- ")
                return first.split(",")[0].split("(")[0].strip()
            return "Not mentioned in the conversation"
        return "[]"

    return responder
