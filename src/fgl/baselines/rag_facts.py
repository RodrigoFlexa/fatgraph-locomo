"""B3 -- the critical ablation: k-NN over the SAME extracted facts as G1.

B3 and G1 must consume byte-identical facts, so this baseline reads them from
the shared :class:`fatgraph.ingest.FactExtractor` cache.  The only difference
between B3 and G1 is then the presence of the fatgraph topology.
"""

from __future__ import annotations

import numpy as np

from fgl.core import default_token_counter
from fgl.retrieval.embeddings import build_index
from fgl.memory.ingest import FactExtractor
from fgl.data.locomo import Conversation, Question
from fgl.retrieval.faces import clean_answer

from fgl.baselines.base import Baseline, BaselineAnswer


class RagFactsBaseline(Baseline):
    name = "B3-rag-facts"

    def prepare(self, conv: Conversation) -> None:
        extractor = FactExtractor(
            self.llm,
            self.prompts,
            self.cfg.paths.facts_cache,
            self.cfg.ingest.max_facts_per_session,
            prompt_name=self.cfg.ingest.extract_prompt,
        )
        self._facts = extractor.extract_all(conv)
        self._index = build_index(self.cfg.index, self.embedder.dim)
        if self._facts:
            vecs = self.embedder.encode([f.fact_text for f in self._facts])
            self._index.add([str(i) for i in range(len(self._facts))], np.vstack(vecs))

    def answer(self, conv: Conversation, question: Question) -> BaselineAnswer:
        k = self.cfg.baselines.rag_top_k
        hits = self._index.search(
            self.embedder.encode_one(question.prompt_question()), max(k, 10)
        )
        ranked_idx = [int(i) for i, _ in hits]
        chosen = ranked_idx[:k]
        context = "\n".join(
            f"[{self._facts[i].date_raw}] {self._facts[i].fact_text}" for i in chosen
        ) or "(no facts retrieved)"
        return BaselineAnswer(
            prediction=self.complete_answer(conv, question, context),
            retrieved_turn_ids=_turns(self._facts, chosen),
            ranked_turn_ids=_turns(self._facts, ranked_idx),
            n_items=len(chosen),
            tokens_context=default_token_counter(context),
        )


def _turns(facts, idxs) -> list[str]:
    out: list[str] = []
    for i in idxs:
        for t in facts[i].turn_ids:
            if t not in out:
                out.append(t)
    return out
