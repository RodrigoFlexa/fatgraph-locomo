"""Shared interface for the three baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

from fgl.config import Config
from fgl.retrieval.embeddings import Embedder
from fgl.llm import LLMClient
from fgl.data.locomo import Conversation, Question
from fgl.llm.prompts import (
    SYSTEM_ANSWERER,
    SYSTEM_ANSWERER_OPEN,
    PromptLibrary,
)


@dataclass
class BaselineAnswer:
    prediction: str
    retrieved_turn_ids: list[str] = field(default_factory=list)
    n_items: int = 0
    tokens_context: int = 0
    ranked_turn_ids: list[str] = field(default_factory=list)  # for recall@k


class Baseline:
    """Every baseline builds its own state from a conversation, then answers."""

    name = "abstract"

    #: The baselines' historical system message, kept verbatim on purpose. It
    #: differs from SYSTEM_ANSWERER only in mentioning "the memories you are
    #: given", but `system` is part of the LLM cache key -- swapping it for the
    #: sake of tidiness would invalidate every cached baseline answer and make
    #: B1 re-issue 1986 prompts of ~23k tokens for no measurable gain. The
    #: open-domain route below is a real change and pays for its own misses;
    #: this one would not.
    system_answerer = (
        "You answer questions about a conversation with a short "
        "extractive phrase and nothing else."
    )

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        embedder: Embedder,
        prompts: PromptLibrary,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.embedder = embedder
        self.prompts = prompts

    def prepare(self, conv: Conversation) -> None:
        raise NotImplementedError

    def answer(self, conv: Conversation, question: Question) -> BaselineAnswer:
        raise NotImplementedError

    # ------------------------------------------------------------------------
    def complete_answer(
        self, conv: Conversation, question: Question, context: str
    ) -> str:
        """Render, call, and clean -- shared so every arm answers identically.

        The three baselines each had their own copy of this.  That was harmless
        until open-domain routing arrived: a fix applied to the fatgraph path
        and not to the baselines would show up as a method effect, when all it
        would be is two arms answering under different instructions.  One code
        path, one comparison.
        """
        from fgl.retrieval.faces import clean_answer

        open_domain = (
            self.cfg.retrieval.open_domain_inference and question.category == 3
        )
        prompt = self.prompts.render(
            "answer_open" if open_domain else "answer",
            speaker_a=conv.speaker_a,
            speaker_b=conv.speaker_b,
            context=context,
            question=question.prompt_question(),
        )
        out = self.llm.complete(
            prompt,
            system=SYSTEM_ANSWERER_OPEN if open_domain else self.system_answerer,
            purpose="qa/answer_open" if open_domain else "qa/answer",
            max_tokens=self.cfg.retrieval.answer_max_tokens,
        )
        return clean_answer(out)
