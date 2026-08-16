"""Shared interface for the three baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

from fgl.config import Config
from fgl.retrieval.embeddings import Embedder
from fgl.llm import LLMClient
from fgl.data.locomo import Conversation, Question
from fgl.llm.prompts import PromptLibrary


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
