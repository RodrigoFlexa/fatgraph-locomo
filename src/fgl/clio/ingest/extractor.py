"""Calls the LLM to extract candidate propositions (spec 6.3): the ONLY
place besides answer generation (M8) this package calls a model at all
(P3 -- the LLM proposes, code decides everything else).

Reuses :class:`fgl.llm.client.LLMClient` and :class:`fgl.llm.prompts.
PromptLibrary` -- the same generic LLM-calling and prompt-templating
infrastructure the rest of this repository uses, so ``FakeLLM`` (offline
tests) and ``AzureLLM`` (the real thing, same ``.env``) both work here for
free. Reusing this is not the same call as avoiding ``fgl.memory``
elsewhere in this package: an LLM client and a prompt loader carry no
fatgraph-condition business logic to be wrongly coupled to.
"""

from __future__ import annotations

from fgl.clio.catalog import RelationSpec
from fgl.clio.ingest.context import ExtractionContext
from fgl.clio.types import Entity, Episode
from fgl.llm.client import LLMClient
from fgl.llm.prompts import SYSTEM_EXTRACTOR, PromptLibrary


def _render_candidates(candidates: list[Entity]) -> str:
    if not candidates:
        return "(none yet -- everything in this turn is new)"
    return "\n".join(f"- {e.id} ({e.type}): {e.canonical_name}" for e in candidates)


def _render_relations(relations: list[RelationSpec]) -> str:
    if not relations:
        return "(none)"
    return "\n".join(
        f"- {r.name} ({r.signature[0]} -> {r.signature[1]})" for r in relations
    )


def _render_previous_turns(previous: list[Episode]) -> str:
    if not previous:
        return "(none -- this is the first turn)"
    return "\n".join(f"{e.speaker}: {e.text}" for e in previous)


def build_prompt(prompts: PromptLibrary, context: ExtractionContext) -> str:
    return prompts.render(
        "clio_extract",
        speaker=context.episode.speaker,
        turn_date=context.episode.ts_ingest.strftime("%d %B %Y"),
        previous_turns=_render_previous_turns(context.previous_turns),
        turn_text=context.episode.text,
        candidates_table=_render_candidates(context.candidates),
        relations_table=_render_relations(context.relations),
    )


def extract_propositions(
    llm: LLMClient, prompts: PromptLibrary, context: ExtractionContext
) -> list[dict]:
    """Returns the raw, not-yet-validated JSON objects the model produced.
    Never raises on a malformed response -- ``complete_json``'s ``default``
    turns a parse failure into "nothing extracted this turn", counted in
    ``llm.usage.json_failures`` rather than aborting the whole ingest."""
    prompt = build_prompt(prompts, context)
    raw = llm.complete_json(
        prompt, system=SYSTEM_EXTRACTOR, purpose="clio_extract", default=[]
    )
    return raw if isinstance(raw, list) else []
