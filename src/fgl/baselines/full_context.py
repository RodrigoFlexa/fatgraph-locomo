"""B1 -- the whole conversation in the prompt.

Truncation policy (spec asks for it to be documented): with gpt-4o-mini's 128k
window, a full LoCoMo conversation (~690 turns, ~25-35k tokens) fits entirely,
so **no truncation happens in practice**.  The guard is kept for smaller
deployments; when it fires it drops the OLDEST turns first
(``baselines.full_context_truncate: head``), which is the upstream behaviour --
and it biases B1 downward on multi-hop questions whose evidence lives in the
first sessions.  The number of truncated conversations is reported.
"""

from __future__ import annotations

from fgl.core import default_token_counter
from fgl.data.locomo import Conversation, Question
from fgl.retrieval.faces import clean_answer

from fgl.baselines.base import Baseline, BaselineAnswer


class FullContextBaseline(Baseline):
    name = "B1-full-context"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.truncated_conversations = 0
        self._context = ""
        self._n_turns = 0

    def prepare(self, conv: Conversation) -> None:
        budget = self.cfg.baselines.full_context_max_tokens
        sessions = list(conv.sessions)
        while True:
            text = self._render(conv, sessions)
            if default_token_counter(text) <= budget or len(sessions) <= 1:
                break
            self.truncated_conversations += 1
            if self.cfg.baselines.full_context_truncate == "head":
                sessions = sessions[1:]  # drop the oldest session
            else:
                sessions = sessions[:-1]
        self._context = text
        self._n_turns = sum(len(s.turns) for s in sessions)

    def answer(self, conv: Conversation, question: Question) -> BaselineAnswer:
        return BaselineAnswer(
            prediction=self.complete_answer(conv, question, self._context),
            retrieved_turn_ids=[t.dia_id for t in conv.turns()],
            n_items=self._n_turns,
            tokens_context=default_token_counter(self._context),
        )

    @staticmethod
    def _render(conv: Conversation, sessions) -> str:
        parts = [
            f"Below is a conversation between two people: {conv.speaker_a} and "
            f"{conv.speaker_b}. The conversation takes place over multiple days "
            f"and the date of each conversation is written at the beginning of "
            f"the conversation.\n"
        ]
        for s in sessions:
            parts.append(f"DATE: {s.date_time_raw}\nCONVERSATION:")
            parts.extend(f"[{t.dia_id}] {t.rendered}" for t in s.turns)
            parts.append("")
        return "\n".join(parts)
