"""Expansão por sigma (condição G4).

O teste central monta o regime que a hipótese descreve e verifica que ele se
comporta como afirmado -- e, tão importante quanto, que com o flag desligado
NADA muda, porque G1/G2/G3 já têm resultados guardados.

Nota de topologia, aprendida na marra ao escrever isto: numa ESTRELA a
expansão por sigma é inútil. Vértices de grau 1 devolvem sigma a si mesmos,
então phi = sigma∘alpha degenera em marchar pela órbita do próprio hub, e a
face já entrega os vizinhos de sigma em ordem. O ganho aparece quando os
vizinhos têm grau > 1 -- que é o caso num grafo de memória real, onde as
entidades se repetem -- porque aí phi sai passeando e só volta ao hub depois
de uma volta na superfície.
"""

from __future__ import annotations

import pytest

from fgl.config import Config, ConfigError
from fgl.core import FatGraph
from fgl.retrieval import SOURCE_SIGMA, FaceRetriever, render_context

BRIDGE_TEXT = "Bangkok rooftop bars stayed open until dawn that whole June."
BRIDGE_TURN = "D1:99"
ANCHOR_TEXT = "Caroline asked Melanie about her travel plans."
QUESTION = "What did Caroline ask about her travel plans?"
N_TOPICS = 60


@pytest.fixture
def hub_graph(embedder):
    """Hub `Melanie` de grau alto, vizinhos encadeados, ponte na órbita.

    Devolve ``(graph, vertices, anchor_edge, bridge_edge)``. A ponte não
    compartilha vocabulário algum com a pergunta -- é essa a definição de um
    segundo salto: ela se liga à âncora apenas por compartilhar a entidade.
    """
    g = FatGraph()
    names = ["Melanie", "Caroline", "support group", "photography", "Bangkok"]
    names += [f"topic {i}" for i in range(N_TOPICS)]
    v = {n: g.add_vertex(n) for n in names}

    def add(a, b, text, turn, pos1=None, pos2=None):
        return g.add_edge(
            v[a], v[b],
            {"text": text, "turn_ids": [turn], "session_id": "S1",
             "timestamp": "2023-05-08T13:56:00",
             "embedding": embedder.encode_one(text)},
            pos1=pos1, pos2=pos2,
        )

    anchor_edge = add("Caroline", "Melanie", ANCHOR_TEXT, "D1:1")
    for i in range(N_TOPICS):
        add("Melanie", f"topic {i}",
            f"Melanie mentioned topic {i} during a long conversation about "
            f"unrelated everyday matters number {i}.", f"D1:{10 + i}")
    for i in range(N_TOPICS - 1):  # grau > 1 nos vizinhos: ver docstring
        add(f"topic {i}", f"topic {i + 1}",
            f"Topics {i} and {i + 1} came up in the same session.", f"D1:{200 + i}")
    add("Caroline", "support group", "Caroline attended the support group.", "D1:5")
    add("support group", "photography", "The support group organised a photo walk.", "D1:6")

    anchor_h = next(
        h for h in g.edge_half_edges(anchor_edge) if g.H[h].vertex_id == v["Melanie"]
    )
    pos = g.sigma[v["Melanie"]].index(anchor_h) + 4
    bridge_edge = add("Melanie", "Bangkok", BRIDGE_TEXT, BRIDGE_TURN, pos1=pos)
    return g, v, anchor_edge, bridge_edge


def _retrieve(condition: str, graph, embedder, **overrides):
    cfg = Config.load(condition)
    # Um âncora só: o HashingEmbedder rankeia por colisão de hash, então com
    # cinco âncoras a ponte entraria por acidente e o teste mediria o k-NN.
    cfg.retrieval.top_m_anchors = 1
    cfg.retrieval.budget_tokens = 2000
    for k, val in overrides.items():
        setattr(cfg.retrieval, k, val)
    cfg.validate()
    return FaceRetriever(graph, embedder, cfg).retrieve(QUESTION)


def _has_bridge(res) -> bool:
    return any(BRIDGE_TEXT in f.text for f in res.facts)


# --------------------------------------------------------------------------- #
# O comportamento afirmado                                                     #
# --------------------------------------------------------------------------- #


def test_bridge_is_invisible_to_the_face_and_to_cosine(hub_graph, embedder):
    """Pré-condição do experimento: sem sigma, a ponte é inalcançável.

    Se este teste falhar, todos os outros passam a medir outra coisa.
    """
    graph, _, anchor_edge, bridge_edge = hub_graph
    res = _retrieve("G1", graph, embedder)

    assert graph.H[res.anchors[0][0]].edge_id == anchor_edge
    assert bridge_edge not in {graph.H[h].edge_id for h, _ in res.anchors}
    assert not _has_bridge(res), "a face alcançou a ponte — o teste não isola nada"


def test_sigma_expansion_retrieves_the_bridge(hub_graph, embedder):
    graph, _, _, _ = hub_graph
    res = _retrieve("G4", graph, embedder, sigma_rerank=False)

    assert res.sigma_expand
    assert res.n_sigma_facts > 0
    assert _has_bridge(res)
    bridge = next(f for f in res.facts if BRIDGE_TEXT in f.text)
    assert bridge.source == SOURCE_SIGMA
    assert bridge.via_entity == "Melanie"
    # o turno da ponte é creditado como alcançado SÓ por sigma
    assert BRIDGE_TURN in res.sigma_turn_ids


def test_sigma_facts_are_labelled_with_the_bridging_entity(hub_graph, embedder):
    """O prompt tem de dizer ONDE as duas trilhas se encontram."""
    graph, _, _, _ = hub_graph
    ctx = render_context(_retrieve("G4", graph, embedder, sigma_rerank=False))
    assert "other memories about Melanie" in ctx
    assert "--- trail 1 ---" in ctx


def test_expansion_respects_the_token_budget(hub_graph, embedder):
    graph, _, _, _ = hub_graph
    res = _retrieve("G4", graph, embedder, budget_tokens=300, sigma_rerank=False)
    assert res.tokens_used <= 300
    assert res.sigma_tokens <= 300 * Config.load("G4").retrieval.sigma_budget_frac + 1


def test_truncation_never_drops_the_sigma_facts(hub_graph, embedder):
    """`facts[:max]` cortaria justo o salto, que entra por último."""
    graph, _, _, _ = hub_graph
    res = _retrieve("G4", graph, embedder, max_facts_in_prompt=3, sigma_rerank=False)
    assert len(res.facts) <= 3
    assert res.n_sigma_facts > 0


def test_both_ends_of_the_anchor_are_scanned(hub_graph, embedder):
    """A ponte pode estar em qualquer uma das duas entidades da âncora."""
    graph, _, _, _ = hub_graph
    both = _retrieve("G4", graph, embedder, sigma_rerank=False)
    one = _retrieve("G4", graph, embedder, sigma_rerank=False,
                    sigma_expand_both_ends=False)
    assert len(both.sigma_vertices) >= len(one.sigma_vertices)


# --------------------------------------------------------------------------- #
# Não-regressão: com o flag desligado, nada muda                               #
# --------------------------------------------------------------------------- #


def test_flag_off_is_bit_identical_to_the_old_path(hub_graph, embedder):
    graph, _, _, _ = hub_graph
    a = _retrieve("G1", graph, embedder)
    b = _retrieve("G1", graph, embedder)

    assert [f.edge_id for f in a.facts] == [f.edge_id for f in b.facts]
    assert a.sigma_expand is False
    assert (a.n_sigma_facts, a.sigma_tokens, a.sigma_scanned) == (0, 0, 0)
    assert a.sigma_vertices == [] and a.sigma_turn_ids == []
    assert all(f.source != SOURCE_SIGMA for f in a.facts)


def test_retrieval_is_deterministic(hub_graph, embedder):
    graph, _, _, _ = hub_graph
    a = _retrieve("G4", graph, embedder, sigma_rerank=False)
    b = _retrieve("G4", graph, embedder, sigma_rerank=False)
    assert [f.edge_id for f in a.facts] == [f.edge_id for f in b.facts]
    assert [f.source for f in a.facts] == [f.source for f in b.facts]


def test_star_graph_gains_nothing_from_sigma(embedder):
    """A degenerescência documentada, fixada como teste.

    Sem o encadeamento entre vizinhos o grafo é uma estrela e phi já percorre
    a órbita do hub: tudo que sigma propõe já veio pela face. É o cenário que
    ``sigma_dup_rate`` acusa nos resultados.
    """
    g = FatGraph()
    v = {n: g.add_vertex(n) for n in ["Melanie", "Caroline"] + [f"t{i}" for i in range(8)]}
    g.add_edge(v["Caroline"], v["Melanie"],
               {"text": ANCHOR_TEXT, "turn_ids": ["D1:1"],
                "embedding": embedder.encode_one(ANCHOR_TEXT)})
    for i in range(8):
        t = f"Melanie mentioned t{i}."
        g.add_edge(v["Melanie"], v[f"t{i}"],
                   {"text": t, "turn_ids": [f"D1:{i}"], "embedding": embedder.encode_one(t)})

    res = _retrieve("G4", g, embedder, sigma_rerank=False)
    assert res.sigma_scanned > 0, "as órbitas não estão vazias"
    assert res.n_sigma_facts == 0, "numa estrela phi já cobre a órbita"
    assert res.sigma_dup == res.sigma_scanned


# --------------------------------------------------------------------------- #
# Configuração                                                                 #
# --------------------------------------------------------------------------- #


def test_g4_differs_from_g1_only_in_retrieval():
    """O delta G4 − G1 tem de ser atribuível só à recuperação.

    A G4 constrói o próprio grafo (nada de `graphs_condition`), então esta
    igualdade de configuração é metade da garantia; a outra metade é o
    fingerprint do grafo, verificado em ``test_graph_identity.py``.
    """
    diff = Config.load("G1").diff(Config.load("G4"))
    assert set(diff) == {"condition", "retrieval.sigma_expand"}
    assert Config.load("G4").paths.graphs_condition == ""


def test_sigma_is_off_in_every_other_condition():
    """G1–G3 e as baselines têm de continuar reproduzindo os números antigos."""
    for name in ("G1", "G2", "G3", "B1", "B2", "B3", "test_offline"):
        assert Config.load(name).retrieval.sigma_expand is False


@pytest.mark.parametrize(
    "key,value",
    [("sigma_expand_k", 0), ("sigma_budget_frac", 1.0), ("sigma_expand_max_anchors", -1)],
)
def test_invalid_sigma_config_is_rejected(key, value):
    cfg = Config.load("G4")
    setattr(cfg.retrieval, key, value)
    with pytest.raises(ConfigError):
        cfg.validate()
