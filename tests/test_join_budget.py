"""Repartição do prompt entre âncoras e joins, e a regra de abstenção (G4/G5/G6).

Três defeitos concretos, cada um fixado aqui como regressão:

1. ``_truncate`` dava prioridade ABSOLUTA aos fatos de join. Uma trilha de
   cobertura longa estoura ``max_facts_in_prompt`` sozinha e evacuava *todos*
   os fatos de âncora -- e aí G5/G6 deixavam de ser superconjuntos da G1, que é
   exatamente a comparação para a qual as condições existem.
2. ``Answerer`` abstinha quando NENHUM fato de âncora rank-0 sobrevivia, porque
   ``all([])`` é ``True``. Sob G5/G6 isso acontece legitimamente (a cobertura
   gastou o orçamento), e a resposta era descartada mesmo com todo fato
   sobrevivente congruente.
3. ``score_faces`` drenava as faces da PRIMEIRA entidade antes de olhar a
   segunda, estourando ``coverage_max_faces`` e enviesando o candidato --
   quando uma face-ponte é, por definição, a que aparece sob mais de uma.
"""

from __future__ import annotations

import pytest

from fgl.config import Config, ConfigError
from fgl.core import STATE_INCONGRUENT, FatGraph
from fgl.data.locomo import ABSTAIN_ANSWER
from fgl.retrieval import SOURCE_COVERAGE, SOURCE_FACE, FaceRetriever
from fgl.retrieval.faces import Answerer, RetrievalResult, RetrievedFact


def _fact(source: str, i: int, rank: int = 0, state: str = "emergente") -> RetrievedFact:
    return RetrievedFact(
        edge_id=f"e{i}", text="t", timestamp="", date_raw="", session_id="S1",
        turn_ids=[f"D1:{i}"], state=state, level=1, anchor_rank=rank,
        anchor_score=0.0, face_id="f", position_in_face=i, source=source,
    )


@pytest.fixture
def retriever(embedder):
    g = FatGraph()
    a = g.add_vertex("a", embedding=embedder.encode_one("a"))
    b = g.add_vertex("b", embedding=embedder.encode_one("b"))
    g.add_edge(a, b, {"text": "x", "embedding": embedder.encode_one("x")})
    return FaceRetriever(g, embedder, Config.load("G6"))


# --------------------------------------------------------------------------- #
# 1. Truncamento                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_join,n_face", [(60, 30), (100, 5), (3, 100), (40, 40)])
def test_truncation_never_starves_the_anchor_walk(retriever, n_join, n_face):
    res = RetrievalResult(sigma_expand=True, face_coverage=True)
    res.facts = [_fact(SOURCE_COVERAGE, i) for i in range(n_join)]
    res.facts += [_fact(SOURCE_FACE, 1000 + i) for i in range(n_face)]
    retriever._truncate(res, 40)

    assert len(res.facts) <= 40
    assert sum(1 for f in res.facts if f.source == SOURCE_FACE) > 0
    assert sum(1 for f in res.facts if f.source == SOURCE_COVERAGE) > 0


def test_truncation_lends_an_unused_share_to_the_other_pool(retriever):
    """Cota é piso, não teto: o que uma parte não usa a outra aproveita."""
    res = RetrievalResult(sigma_expand=True, face_coverage=True)
    res.facts = [_fact(SOURCE_COVERAGE, i) for i in range(100)]
    res.facts += [_fact(SOURCE_FACE, 1000 + i) for i in range(5)]
    retriever._truncate(res, 40)
    assert len(res.facts) == 40
    # as 5 âncoras cabem inteiras e o resto do prompt vai para os joins
    assert sum(1 for f in res.facts if f.source == SOURCE_FACE) == 5
    assert sum(1 for f in res.facts if f.source == SOURCE_COVERAGE) == 35


def test_truncation_keeps_the_blunt_slice_when_both_flags_are_off(retriever):
    """G1/G2/G3 têm números guardados: o caminho antigo não pode mudar."""
    res = RetrievalResult()  # sigma_expand=False, face_coverage=False
    res.facts = [_fact(SOURCE_FACE, i) for i in range(60)]
    retriever._truncate(res, 40)
    assert [f.edge_id for f in res.facts] == [f"e{i}" for i in range(40)]


def test_truncation_preserves_prompt_order(retriever):
    res = RetrievalResult(sigma_expand=True, face_coverage=True)
    res.facts = [_fact(SOURCE_COVERAGE, i) for i in range(30)]
    res.facts += [_fact(SOURCE_FACE, 1000 + i) for i in range(30)]
    retriever._truncate(res, 40)
    order = [f.edge_id for f in res.facts]
    assert order == sorted(order, key=lambda e: int(e[1:]))


# --------------------------------------------------------------------------- #
# 2. Abstenção por incongruência                                               #
# --------------------------------------------------------------------------- #


class _SpyLLM:
    def complete(self, *a, **k):  # noqa: D102
        return "uma resposta concreta"


class _Prompts:
    def render(self, *a, **k):  # noqa: D102
        return "prompt"


class _Question:
    question = "q?"
    category = 1  # multi-hop: fora da rota de inferência de open-domain

    def prompt_question(self):  # noqa: D102
        return "q?"


class _Conv:
    speaker_a = "A"
    speaker_b = "B"


def _answer(facts):
    cfg = Config.load("G6")
    res = RetrievalResult(sigma_expand=True, face_coverage=True)
    res.facts = facts
    res.any_incongruent = True
    return Answerer(_SpyLLM(), _Prompts(), cfg).answer(_Conv(), _Question(), res)


def test_no_abstention_when_only_a_join_fact_is_incongruent():
    """O bug do ``all([])``: sem fato de âncora rank-0 não há do que abster."""
    out = _answer([_fact(SOURCE_COVERAGE, 1, rank=-1, state=STATE_INCONGRUENT)])
    assert out != ABSTAIN_ANSWER


def test_abstains_when_the_whole_rank0_anchor_is_incongruent():
    out = _answer([
        _fact(SOURCE_FACE, 1, rank=0, state=STATE_INCONGRUENT),
        _fact(SOURCE_FACE, 2, rank=0, state=STATE_INCONGRUENT),
    ])
    assert out == ABSTAIN_ANSWER


def test_does_not_abstain_when_the_rank0_anchor_is_only_partly_incongruent():
    out = _answer([
        _fact(SOURCE_FACE, 1, rank=0, state=STATE_INCONGRUENT),
        _fact(SOURCE_FACE, 2, rank=0),
    ])
    assert out != ABSTAIN_ANSWER


def test_a_lower_ranked_anchor_does_not_trigger_abstention():
    out = _answer([
        _fact(SOURCE_FACE, 1, rank=0),
        _fact(SOURCE_FACE, 2, rank=1, state=STATE_INCONGRUENT),
    ])
    assert out != ABSTAIN_ANSWER


# --------------------------------------------------------------------------- #
# 3. Teto de faces candidatas                                                  #
# --------------------------------------------------------------------------- #


@pytest.fixture
def four_components(embedder):
    """Quatro componentes desconexas: cada hub tem faces só suas."""
    g = FatGraph()
    hubs = []
    for c in range(4):
        h = g.add_vertex(f"hub{c}", embedding=embedder.encode_one(f"hub{c}"))
        hubs.append(h)
        for i in range(10):
            a = g.add_vertex(f"c{c}a{i}", embedding=embedder.encode_one(f"c{c}a{i}"))
            b = g.add_vertex(f"c{c}b{i}", embedding=embedder.encode_one(f"c{c}b{i}"))
            for x, y, tag in ((h, a, "x"), (a, b, "y"), (b, h, "z")):
                g.add_edge(x, y, {"text": f"t{c}{i}{tag}",
                                  "embedding": embedder.encode_one(f"t{c}{i}{tag}")})
    return g, hubs


@pytest.mark.parametrize("cap", [2, 3, 5, 24])
def test_coverage_max_faces_is_a_hard_cap(four_components, embedder, cap):
    graph, hubs = four_components
    cfg = Config.load("G5")
    cfg.retrieval.coverage_max_faces = cap
    ranked = FaceRetriever(graph, embedder, cfg).score_faces(
        hubs, embedder.encode_one("hub0 hub1 hub2 hub3")
    )
    assert len(ranked) <= cap


def test_every_question_entity_contributes_a_candidate(four_components, embedder):
    """Round-robin: a primeira entidade não pode drenar o teto sozinha."""
    graph, hubs = four_components
    cfg = Config.load("G5")
    cfg.retrieval.coverage_max_faces = 4
    r = FaceRetriever(graph, embedder, cfg)
    got = {f.id for f, _, _ in r.score_faces(hubs, embedder.encode_one("hubs"))}
    for hub in hubs:
        assert got & {f.id for f in r.faces_through_vertex(hub)}, (
            "uma das entidades da pergunta não contribuiu com nenhuma face"
        )


# --------------------------------------------------------------------------- #
# Cache de faces                                                               #
# --------------------------------------------------------------------------- #


def test_face_cache_agrees_with_the_graph(four_components, embedder):
    """O memo só é aceitável se devolver a MESMA face que o passeio faria."""
    graph, hubs = four_components
    r = FaceRetriever(graph, embedder, Config.load("G5"))
    for h in graph.H:
        assert r.face_of(h).id == graph.face_of(h).id
    for hub in hubs:
        assert ({f.id for f in r.faces_through_vertex(hub)}
                == {f.id for f in graph.faces_through_vertex(hub)})


# --------------------------------------------------------------------------- #
# Configuração                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key,value", [
    ("max_facts_join_frac", 0.0),
    ("max_facts_join_frac", 1.5),
    ("coverage_max_faces", 0),
    ("sigma_max_orbit_scan", -1),
])
def test_invalid_budget_config_is_rejected(key, value):
    cfg = Config.load("G6")
    setattr(cfg.retrieval, key, value)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_join_scoring_requires_normalised_embeddings():
    """`sim + coverage_weight * coverage` só faz sentido com cosseno."""
    cfg = Config.load("G6")
    cfg.embeddings.normalize = False
    with pytest.raises(ConfigError):
        cfg.validate()
    # sem os mecanismos de join o índice normaliza por dentro: segue válido
    plain = Config.load("G1")
    plain.embeddings.normalize = False
    plain.validate()


def test_orbit_cap_zero_means_no_cap(embedder):
    """0 = sem teto, como em sigma_expand_max_anchors e max_facts_per_session."""
    g = FatGraph()
    hub = g.add_vertex("hub", embedding=embedder.encode_one("hub"))
    for i in range(12):
        leaf = g.add_vertex(f"n{i}", embedding=embedder.encode_one(f"n{i}"))
        g.add_edge(hub, leaf, {"text": f"m{i}", "embedding": embedder.encode_one(f"m{i}")})

    cfg = Config.load("G4")
    cfg.retrieval.sigma_max_orbit_scan = 0
    r = FaceRetriever(g, embedder, cfg)
    start = g.sigma[hub][0]
    assert len(r._orbit(start, 0)) == g.degree(hub) - 1
    assert len(r._orbit(start, 4)) == 4
