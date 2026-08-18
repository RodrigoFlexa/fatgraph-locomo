"""O vértice-hub como stopword do grafo (G11).

Medido nos grafos reais: 86% das arestas tocam um dos dois falantes, eles são os
dois maiores vértices de toda conversa, e o vértice compartilhado por um par de
fatos de evidência tem grau mediano 115. Logo "estas duas memórias compartilham
uma entidade" quer dizer "as duas são sobre a Caroline" — verdade para metade do
grafo — e a órbita de sigma contém a ponte real em 7.7% dos casos.

Recuperação de informação resolveu isso nos anos 60: não se indexa stopword.

O limiar 60 não é chute. Os graus são bimodais com banda vazia: os falantes
nunca ficam abaixo de 95 e o terceiro vértice nunca passa de 50.
"""

from __future__ import annotations

import pytest

from fgl.config import Config, ConfigError
from fgl.core import FatGraph
from fgl.retrieval import SOURCE_SIGMA, FaceRetriever

QUESTION = "What did Melanie say about Bangkok?"


@pytest.fixture
def hub_and_bridge(embedder):
    """Um falante-hub e uma ponte tópica de verdade, no mesmo grafo.

    `speaker` liga tudo (grau alto e sem informação). `Bangkok` liga apenas duas
    memórias, e é a ponte que a pergunta precisa.
    """
    g = FatGraph()
    v = {n: g.add_vertex(n, embedding=embedder.encode_one(n))
         for n in ["speaker", "Bangkok", "hotel", "agency"]}
    for i in range(40):  # o hub: 40 memórias penduradas no falante
        leaf = g.add_vertex(f"t{i}", embedding=embedder.encode_one(f"t{i}"))
        t = f"speaker mencionou o assunto {i}"
        g.add_edge(v["speaker"], leaf,
                   {"text": t, "turn_ids": [f"D1:{i}"], "session_id": "S1",
                    "embedding": embedder.encode_one(t)})
    for a, b, t, turn in [
        ("Bangkok", "hotel", "o hotel em Bangkok tinha vista para o rio", "D2:1"),
        ("Bangkok", "agency", "a agência reservou Bangkok em junho", "D2:2"),
    ]:
        g.add_edge(v[a], v[b], {"text": t, "turn_ids": [turn], "session_id": "S2",
                                "embedding": embedder.encode_one(t)})
    return g, v


def _cfg(skip: int):
    c = Config.load("G4")
    c.retrieval.sigma_skip_hub_degree = skip
    return c.validate()


#: uma pergunta cujo melhor âncora é uma memória PENDURADA no hub, para o
#: caminho do filtro ser exercitado; com QUESTION os âncoras caem em Bangkok
#: (grau 2) e não há hub algum a pular
HUB_QUESTION = "o que speaker mencionou sobre o assunto 7"


def test_the_hub_orbit_is_skipped(hub_and_bridge, embedder):
    graph, v = hub_and_bridge
    assert graph.degree(v["speaker"]) == 40
    res = FaceRetriever(graph, embedder, _cfg(20)).retrieve(HUB_QUESTION)
    assert res.sigma_hubs_skipped > 0


def test_nothing_is_skipped_when_no_anchor_touches_a_hub(hub_and_bridge, embedder):
    """O contador mede hubs encontrados, não perguntas feitas."""
    graph, v = hub_and_bridge
    anchor = graph.sigma[v["Bangkok"]][0]
    r = FaceRetriever(graph, embedder, _cfg(20))
    assert r.sigma_neighborhood(anchor, None)
    assert r._hubs_skipped == 0


def test_a_real_bridge_still_survives_the_filter(hub_and_bridge, embedder):
    """O ponto todo: a ponte tópica passa, o hub não.

    Se o filtro matasse tudo, ele não estaria focando a junção, estaria
    desligando-a — e o resultado não distinguiria uma hipótese da outra.
    """
    graph, v = hub_and_bridge
    r = FaceRetriever(graph, embedder, _cfg(20))
    nb = r.sigma_neighborhood(graph.sigma[v["Bangkok"]][0], None)
    assert nb, "a órbita de Bangkok (grau 2) não pode ser pulada"
    assert all(vid != v["speaker"] for _h, vid in nb)


def test_disabled_by_default_so_g1_g10_are_untouched(hub_and_bridge, embedder):
    graph, _ = hub_and_bridge
    res = FaceRetriever(graph, embedder, _cfg(0)).retrieve(QUESTION)
    assert res.sigma_hubs_skipped == 0
    for name in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"):
        assert Config.load(name).retrieval.sigma_skip_hub_degree == 0


def test_skipping_reduces_sigma_facts_but_need_not_zero_them(hub_and_bridge, embedder):
    graph, _ = hub_and_bridge
    loose = FaceRetriever(graph, embedder, _cfg(0)).retrieve(QUESTION)
    tight = FaceRetriever(graph, embedder, _cfg(20)).retrieve(QUESTION)
    assert tight.n_sigma_facts <= loose.n_sigma_facts


def test_telemetry_is_per_question_not_cumulative(hub_and_bridge, embedder):
    """Contador acumulado inflaria com o número de perguntas e não com o hub."""
    graph, _ = hub_and_bridge
    r = FaceRetriever(graph, embedder, _cfg(20))
    first = r.retrieve(QUESTION).sigma_hubs_skipped
    second = r.retrieve(QUESTION).sigma_hubs_skipped
    assert first == second


def test_is_deterministic(hub_and_bridge, embedder):
    graph, _ = hub_and_bridge
    a = FaceRetriever(graph, embedder, _cfg(20)).retrieve(QUESTION)
    b = FaceRetriever(graph, embedder, _cfg(20)).retrieve(QUESTION)
    assert [f.edge_id for f in a.facts] == [f.edge_id for f in b.facts]


# --------------------------------------------------------------------------- #
# Configuração                                                                 #
# --------------------------------------------------------------------------- #


def test_g11_differs_from_g4_only_in_the_hub_rule():
    assert set(Config.load("G4").diff(Config.load("G11"))) == {
        "condition", "retrieval.sigma_skip_hub_degree",
    }


def test_g11_threshold_sits_in_the_empty_band():
    """Falantes >= 95, terceiro vértice <= 50: o corte tem de cair entre eles."""
    assert 50 < Config.load("G11").retrieval.sigma_skip_hub_degree < 95


def test_g11_needs_no_reingest():
    """É filtro de recuperação: roda sobre grafos que já existem."""
    g11, g4 = Config.load("G11"), Config.load("G4")
    assert g11.ingest == g4.ingest and g11.curation == g4.curation


@pytest.mark.parametrize("bad", [-1, 1, 2])
def test_degenerate_thresholds_are_rejected(bad):
    cfg = Config.load("G11")
    cfg.retrieval.sigma_skip_hub_degree = bad
    with pytest.raises(ConfigError):
        cfg.validate()
