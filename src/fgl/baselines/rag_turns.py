"""B2 -- k-NN over raw dialogue turns, top-10, ordered by score."""

from __future__ import annotations

import numpy as np

from fgl.core import default_token_counter
from fgl.retrieval.embeddings import build_index
from fgl.data.locomo import Conversation, Question
from fgl.retrieval.faces import clean_answer

from fgl.baselines.base import Baseline, BaselineAnswer


class RagTurnsBaseline(Baseline):
    name = "B2-rag-turns"

    def prepare(self, conv: Conversation) -> None:
        turns = list(conv.turns())
        self._turns = {t.dia_id: t for t in turns}
        self._date = {
            t.dia_id: s.date_time_raw for s in conv.sessions for t in s.turns
        }
        self._index = build_index(self.cfg.index, self.embedder.dim)
        if turns:
            vecs = self.embedder.encode([t.rendered for t in turns])
            self._index.add([t.dia_id for t in turns], np.vstack(vecs))

    def answer(self, conv: Conversation, question: Question) -> BaselineAnswer:
        k = self.cfg.baselines.rag_top_k
        hits = self._index.search(
            self.embedder.encode_one(question.prompt_question()), max(k, 10)
        )
        ranked = [dia for dia, _ in hits]
        chosen = ranked[:k]
        context = "\n".join(
            f"[{self._date.get(d, '')}] {self._turns[d].rendered}" for d in chosen
        ) or "(no turns retrieved)"
        prompt = self.prompts.render(
            "answer",
            speaker_a=conv.speaker_a,
            speaker_b=conv.speaker_b,
            context=context,
            question=question.prompt_question(),
        )
        out = self.llm.complete(
            prompt,
            system="You answer questions about a conversation with a short "
            "extractive phrase and nothing else.",
            purpose="qa/answer",
            max_tokens=64,
        )
        return BaselineAnswer(
            prediction=clean_answer(out),
            retrieved_turn_ids=chosen,
            ranked_turn_ids=ranked,
            n_items=len(chosen),
            tokens_context=default_token_counter(context),
        )
