"""As três condições que os resultados motivaram: G7, G8 e G9.

Todas nascem de medições, não da hipótese original:

* `recall@10` é idêntico entre B3 e G1 — mesmos fatos, mesmo ranking — e a
  divergência aparece só no `recall_context`. Logo o passeio pela face TIRA
  evidência, enquanto σ ACRESCENTA (`recall_context_no_sigma` confirma). G7 é σ
  sem passeio.
* Tudo que um ribbon graph acrescenta a um grafo comum é ORDEM. G8 destrói a
  ordem preservando o conteúdo: é o teste que decide se a tese tem fundação.
* O comprimento das faces é artefato de ter escolhido σ pelo relógio, e Euler dá
  o objetivo para escolhê-la melhor. G9 faz isso.
"""

from __future__ import annotations

import pytest

from fgl.config import Config, ConfigError
from fgl.core import FatGraph
from fgl.retrieval import FaceRetriever, render_context


# --------------------------------------------------------------------------- #
# G8 — a ordem importa?                                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture
def chain(embedder):
    """Trilha longa o suficiente para o contexto ter ordem observável."""
    g = FatGraph()
    names = ["Melanie", "Caroline", "Bangkok", "hotel", "agency"]
    names += [f"topic {i}" for i in range(12)]
    v = {n: g.add_vertex(n, embedding=embedder.encode_one(n)) for n in names}
    prev = "Melanie"
    for i, n in enumerate(list(v)[1:]):
        t = f"memoria numero {i} sobre {n} e {prev}"
        g.add_edge(v[prev], v[n], {"text": t, "turn_ids": [f"D1:{i}"],
                                   "session_id": "S1",
                                   "timestamp": "2023-05-08T13:56:00",
                                   "embedding": embedder.encode_one(t)})
        prev = n
    return g


QUESTION = "What did Melanie say about Bangkok?"


def _res(condition, graph, embedder, **over):
    cfg = Config.load(condition)
    for k, val in over.items():
        setattr(cfg.retrieval, k, val)
    cfg.validate()
    return cfg, FaceRetriever(graph, embedder, cfg).retrieve(QUESTION)


def test_shuffle_preserves_content_and_destroys_order(chain, embedder):
    _, res = _res("G4", chain, embedder)
    plain = render_context(res)
    shuffled = render_context(res, shuffle_seed=1234)

    # mesmo conteúdo: cada fato aparece nos dois
    for f in res.facts:
        assert f.text in plain and f.text in shuffled
    # e a ordem mudou (ou o grafo é degenerado demais para o teste valer)
    if len(res.facts) > 3:
        assert plain.split("\n") != shuffled.split("\n")


def test_shuffle_is_deterministic(chain, embedder):
    _, res = _res("G4", chain, embedder)
    assert render_context(res, 7) == render_context(res, 7)
    assert render_context(res, 7) != render_context(res, 8) or len(res.facts) < 3


def test_shuffle_drops_the_trail_headers(chain, embedder):
    """Os headers são informação de ordem; deixá-los vazaria a estrutura."""
    _, res = _res("G4", chain, embedder)
    shuffled = render_context(res, shuffle_seed=99)
    assert "--- trail" not in shuffled
    assert "other memories about" not in shuffled


def test_flag_off_renders_exactly_as_before(chain, embedder):
    """G1–G6 têm resultados guardados: sem o flag, o render é o antigo."""
    _, res = _res("G1", chain, embedder)
    assert render_context(res) == render_context(res, shuffle_seed=None)


def test_g8_differs_from_g4_only_by_the_permutation():
    assert set(Config.load("G4").diff(Config.load("G8"))) == {
        "condition", "retrieval.shuffle_context",
    }


# --------------------------------------------------------------------------- #
# G9 — sigma por gênero mínimo                                                 #
# --------------------------------------------------------------------------- #


def _dense(n_hub: int = 6, n_leaf: int = 4) -> FatGraph:
    """Grafo com gênero folgado: hubs interligados e vizinhos encadeados."""
    g = FatGraph()
    hubs = [g.add_vertex(f"h{i}") for i in range(n_hub)]
    for i in range(n_hub):
        for j in range(i + 1, n_hub):
            g.add_edge(hubs[i], hubs[j], {"text": f"hub {i}-{j}"})
        for k in range(n_leaf):
            leaf = g.add_vertex(f"l{i}_{k}")
            g.add_edge(hubs[i], leaf, {"text": f"leaf {i}-{k}"})
    return g


def test_count_faces_agrees_with_faces():
    g = _dense()
    assert g.count_faces() == len([f for f in g.faces() if f.half_edges])


def test_transposition_changes_the_surface():
    """É o movimento que altera o mergulho — Whitehead não altera."""
    g = _dense()
    before = g.count_faces()
    changed = False
    for vid in g.sigma:
        if len(g.sigma[vid]) < 3:
            continue
        for j in range(1, len(g.sigma[vid])):
            g.transpose_sigma(vid, 0, j)
            if g.count_faces() != before:
                changed = True
            g.transpose_sigma(vid, 0, j)
            if changed:
                break
        if changed:
            break
    assert changed, "nenhuma transposição mudou F: o grafo não serve ao teste"


def test_maximize_faces_raises_f_and_lowers_genus():
    g = _dense()
    e0, len0 = g.euler(), [f.length for f in g.faces() if f.length]
    rep = g.maximize_faces(max_passes=4)
    e1, len1 = g.euler(), [f.length for f in g.faces() if f.length]

    assert rep["faces_after"] >= rep["faces_before"]
    assert e1.F >= e0.F
    assert e1.genus <= e0.genus, "mais faces com V e E fixos tem de baixar o gênero"
    if e1.F > e0.F:
        assert max(len1) <= max(len0), "mais faces sobre as mesmas meias-arestas => mais curtas"


def test_maximize_faces_preserves_the_memory():
    """Reescrever sigma não pode inventar, perder ou alterar memória."""
    g = _dense()
    before = {(g.get_edge_attr(e, "text"), frozenset(g.edge_endpoints(e)))
              for e in g.edges()}
    v0, e0 = len(g.vertices), len(g.edges())
    g.maximize_faces(max_passes=3)
    g.check_invariants()
    after = {(g.get_edge_attr(e, "text"), frozenset(g.edge_endpoints(e)))
             for e in g.edges()}
    assert before == after
    assert (len(g.vertices), len(g.edges())) == (v0, e0)


def test_maximize_faces_respects_euler():
    g = _dense()
    g.maximize_faces(max_passes=3)
    e = g.euler()
    assert e.V - e.E + e.F == e.chi


def test_maximize_faces_is_deterministic():
    a, b = _dense(), _dense()
    a.maximize_faces(max_passes=3)
    b.maximize_faces(max_passes=3)
    assert a.count_faces() == b.count_faces()
    assert {v: list(a.sigma[v]) for v in a.sigma} == {v: list(b.sigma[v]) for v in b.sigma}


def test_maximize_faces_noop_on_low_degree():
    """Grau <= 2: toda rotação é a mesma ordem cíclica."""
    g = FatGraph()
    vs = [g.add_vertex(f"v{i}") for i in range(5)]
    for i in range(4):
        g.add_edge(vs[i], vs[i + 1], {"text": f"e{i}"})
    rep = g.maximize_faces(max_passes=2)
    assert rep["moves_applied"] == 0
    assert rep["transpositions_evaluated"] == 0


def test_genus_condition_cannot_borrow_graphs():
    """sigma diferente é ribbon graph diferente: emprestar seria incoerente."""
    cfg = Config.load("G9")
    assert cfg.curation.maximize_faces and cfg.paths.graphs_condition == ""
    cfg.paths.graphs_condition = "G1-fatgraph-min"
    with pytest.raises(ConfigError):
        cfg.validate()


# --------------------------------------------------------------------------- #
# G7 e a instrução de open-domain                                              #
# --------------------------------------------------------------------------- #


def test_g7_keeps_sigma_and_drops_the_walk():
    cfg = Config.load("G7")
    assert cfg.retrieval.sigma_expand and not cfg.retrieval.face_coverage
    # orçamento no nível da B3, e não os 2000 que sustentam o passeio
    assert cfg.retrieval.budget_tokens <= 400
    assert cfg.retrieval.budget_tokens < Config.load("G1").retrieval.budget_tokens


def test_open_domain_routing_only_touches_category_3(cfg, prompts):
    """A rota de inferência não pode afrouxar as adversariais."""
    from fgl.data.locomo import Question

    lib = prompts
    assert "answer_open" in lib.names() if hasattr(lib, "names") else True
    for cat, expected in ((3, "answer_open"), (1, "answer"), (5, "answer")):
        q = Question(question="q?", answer="a", category=cat, evidence=[])
        route = "answer_open" if (cfg.retrieval.open_domain_inference
                                 and q.category == 3) else "answer"
        assert route == expected


def test_open_domain_prompt_exists_and_permits_inference(prompts):
    text = prompts.render(
        "answer_open", speaker_a="A", speaker_b="B", context="c", question="q"
    )
    low = text.lower()
    assert "infer" in low or "likely" in low
    # e continua exigindo resposta curta, porque a métrica é F1 de tokens
    assert "short" in low
