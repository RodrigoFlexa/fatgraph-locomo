"""A cascata de tetos: onde a resposta deixa de ser alcançável.

G1 a G10 variaram política de recuperação assumindo que o grafo codifica a
resposta e que ela é alcançável. Isso nunca foi testado. Se a extração perdeu um
turno de evidência, ou se os dois fatos de uma multi-hop caem em componentes
diferentes, nenhuma política recupera — e o experimento inteiro mede ruído
abaixo de um teto que ninguém olhou.

Só o último degrau (rank por cosseno) é consertável por recuperação. Uma queda
antes dele é problema de ingest fantasiado de problema de recuperação.
"""

from __future__ import annotations

import pytest

from fgl.core import FatGraph
from fgl.data.locomo import Question
from fgl.evaluation.diagnose import (
    Diagnostician, by_category, failing_cases, waterfall,
)


def _q(question="q?", answer="a", category=1, evidence=("D1:1",)):
    return Question(question=question, answer=answer, category=category,
                    evidence=list(evidence))


@pytest.fixture
def chain_graph(embedder):
    """A--m1--B--m2--C: dois fatos de evidência a distância 1, via B."""
    g = FatGraph()
    v = {n: g.add_vertex(n, embedding=embedder.encode_one(n))
         for n in ("Melanie", "agency", "Bangkok", "solo")}
    g.add_edge(v["Melanie"], v["agency"],
               {"text": "Melanie ligou para a agência", "turn_ids": ["D1:1"],
                "embedding": embedder.encode_one("Melanie ligou para a agência")})
    g.add_edge(v["agency"], v["Bangkok"],
               {"text": "a agência reservou Bangkok", "turn_ids": ["D1:2"],
                "embedding": embedder.encode_one("a agência reservou Bangkok")})
    g.add_edge(v["solo"], v["solo"] if False else v["Bangkok"],
               {"text": "fato solto", "turn_ids": ["D9:9"],
                "embedding": embedder.encode_one("fato solto")})
    return g


# --------------------------------------------------------------------------- #
# Teto 1: a extração                                                           #
# --------------------------------------------------------------------------- #


def test_evidence_never_extracted_stops_at_the_first_rung(chain_graph, embedder):
    """O teto que domina tudo: se o fato não existe, não há o que recuperar."""
    t = Diagnostician(chain_graph, embedder).trace(_q(evidence=["D7:77"]))
    assert not t.fully_extracted
    assert t.covered == []
    # e nenhum degrau seguinte é sequer avaliado
    assert t.same_component is None and t.distance is None


def test_partial_extraction_also_fails_the_rung(chain_graph, embedder):
    t = Diagnostician(chain_graph, embedder).trace(_q(evidence=["D1:1", "D7:77"]))
    assert not t.fully_extracted
    assert t.covered == ["D1:1"]


def test_fully_extracted_evidence_passes(chain_graph, embedder):
    t = Diagnostician(chain_graph, embedder).trace(_q(evidence=["D1:1", "D1:2"]))
    assert t.fully_extracted and len(t.edges) == 2


# --------------------------------------------------------------------------- #
# Tetos 2-4: alcançabilidade                                                   #
# --------------------------------------------------------------------------- #


def test_measures_distance_between_the_evidence_facts(chain_graph, embedder):
    """D1:1 e D1:2 compartilham o vértice `agency`: distância 0."""
    t = Diagnostician(chain_graph, embedder).trace(_q(evidence=["D1:1", "D1:2"]))
    assert t.same_component is True
    assert t.shares_vertex is True
    assert t.distance == 0


def test_disconnected_evidence_is_flagged(embedder):
    """Dois fatos em componentes diferentes: multi-hop impossível, por topologia."""
    g = FatGraph()
    a, b = g.add_vertex("a"), g.add_vertex("b")
    c, d = g.add_vertex("c"), g.add_vertex("d")
    g.add_edge(a, b, {"text": "um", "turn_ids": ["D1:1"]})
    g.add_edge(c, d, {"text": "dois", "turn_ids": ["D1:2"]})
    t = Diagnostician(g, embedder).trace(_q(evidence=["D1:1", "D1:2"]))
    assert t.fully_extracted
    assert t.same_component is False
    assert t.shares_vertex is False


def test_single_fact_question_is_trivially_reachable(chain_graph, embedder):
    t = Diagnostician(chain_graph, embedder).trace(_q(evidence=["D1:1"]))
    assert t.same_component and t.shares_vertex and t.same_face
    assert t.distance == 0


def test_common_face_is_detected(chain_graph, embedder):
    t = Diagnostician(chain_graph, embedder).trace(_q(evidence=["D1:1", "D1:2"]))
    assert t.same_face is True


# --------------------------------------------------------------------------- #
# Teto 5: o ranking — o único que a recuperação conserta                       #
# --------------------------------------------------------------------------- #


def test_rank_is_none_without_an_index(chain_graph, embedder):
    assert Diagnostician(chain_graph, embedder).trace(_q()).worst_rank is None


def test_rank_reports_the_hardest_evidence_fact(chain_graph, embedder):
    """Rank 12 diz 'aumente k'; rank 900 diz que nenhum k resolve."""
    import numpy as np

    from fgl.config import Config
    from fgl.retrieval.embeddings import build_index

    idx = build_index(Config.load("G1").index, embedder.dim)
    ids = [h for h, he in chain_graph.H.items() if he.embedding is not None]
    idx.add(ids, np.vstack([chain_graph.H[h].embedding for h in ids]))
    t = Diagnostician(chain_graph, embedder, idx).trace(
        _q(question="a agência reservou Bangkok", evidence=["D1:1", "D1:2"])
    )
    assert isinstance(t.worst_rank, int) and t.worst_rank >= 0


# --------------------------------------------------------------------------- #
# Agregação                                                                    #
# --------------------------------------------------------------------------- #


def test_waterfall_conditions_each_rung_on_the_previous(chain_graph, embedder):
    d = Diagnostician(chain_graph, embedder)
    traces = [
        d.trace(_q(evidence=["D1:1", "D1:2"])),   # passa tudo
        d.trace(_q(evidence=["D7:77"])),          # morre no degrau 1
    ]
    w = waterfall(traces)
    assert w["n_questions"] == 2
    assert w["questions_fully_extracted"] == 0.5
    # só a que sobreviveu ao degrau 1 conta nos degraus seguintes
    assert w["n_multi_fact"] == 1
    assert w["same_component"] == 1.0


def test_waterfall_reports_the_extraction_ceiling(chain_graph, embedder):
    d = Diagnostician(chain_graph, embedder)
    w = waterfall([d.trace(_q(evidence=["D1:1", "D7:77"]))])
    assert w["evidence_turns_extracted"] == 0.5


def test_by_category_splits_the_waterfall(chain_graph, embedder):
    d = Diagnostician(chain_graph, embedder)
    out = by_category([
        d.trace(_q(category=1, evidence=["D1:1"])),
        d.trace(_q(category=2, evidence=["D1:2"])),
    ])
    assert set(out) == {"multi-hop", "temporal"}


def test_failing_cases_surface_the_lost_evidence(chain_graph, embedder):
    d = Diagnostician(chain_graph, embedder)
    cases = failing_cases([
        d.trace(_q(evidence=["D7:77"])),
        d.trace(_q(evidence=["D1:1"])),
    ], limit=5)
    assert any(not c.fully_extracted for c in cases)


def test_empty_input_is_not_a_diagnosis():
    assert waterfall([]) == {}
    assert by_category([]) == {}


def test_costs_no_llm_calls(chain_graph, embedder, llm):
    """Lê grafos que já existem: um diagnóstico não pode custar uma rodada."""
    d = Diagnostician(chain_graph, embedder)
    for _ in range(20):
        d.trace(_q(evidence=["D1:1", "D1:2"]))
    assert llm.usage.calls == 0
