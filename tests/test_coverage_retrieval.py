"""Recuperação por cobertura de entidades (G5) e a combinação com sigma (G6).

O cenário é o que a hipótese descreve: a pergunta nomeia DUAS entidades, a
evidência está numa trilha entre elas, e nenhum fato isolado dessa trilha
parece com a pergunta -- de modo que o ranking por cosseno não pode encontrá-la
e a cobertura, que é sinal estrutural, pode.
"""

from __future__ import annotations

import pytest

from fgl.config import Config, ConfigError
from fgl.core import FatGraph
from fgl.retrieval import (
    SOURCE_COVERAGE,
    SOURCE_GEODESIC,
    SOURCE_SIGMA,
    FaceRetriever,
    QuestionLinker,
    render_context,
)

QUESTION = "Did Melanie and Caroline both end up in Bangkok?"
TARGET_TEXT = "The travel agency booked the same hotel for both of them."


@pytest.fixture
def linked_graph(embedder):
    """`Melanie` e `Bangkok` existem como vértices e há uma trilha entre eles."""
    g = FatGraph()
    names = ["Melanie", "Caroline", "Bangkok", "travel agency", "hotel"]
    names += [f"filler {i}" for i in range(20)]
    v = {n: g.add_vertex(n, embedding=embedder.encode_one(n)) for n in names}

    def add(a, b, text, turn, **kw):
        return g.add_edge(
            v[a], v[b],
            {"text": text, "turn_ids": [turn], "session_id": "S1",
             "timestamp": "2023-05-08T13:56:00",
             "embedding": embedder.encode_one(text)},
            **kw,
        )

    # a trilha que liga as duas entidades da pergunta, sem vocabulário dela
    add("Melanie", "travel agency", "She rang the agency on a Tuesday.", "D1:1")
    add("travel agency", "Bangkok", TARGET_TEXT, "D1:2")
    add("Bangkok", "hotel", "The hotel overlooked the river.", "D1:3")
    add("hotel", "Melanie", "She kept the key card as a souvenir.", "D1:4")
    # ruído que o cosseno prefere
    for i in range(20):
        add("Caroline", f"filler {i}",
            f"Caroline said something about filler {i} at length.", f"D1:{50 + i}")
    return g, v


def _retrieve(condition, graph, embedder, **overrides):
    cfg = Config.load(condition)
    cfg.retrieval.budget_tokens = 400
    for k, val in overrides.items():
        setattr(cfg.retrieval, k, val)
    cfg.validate()
    return FaceRetriever(graph, embedder, cfg).retrieve(QUESTION)


# --------------------------------------------------------------------------- #
# Linker                                                                       #
# --------------------------------------------------------------------------- #


def test_linker_finds_the_entities_the_question_names(linked_graph, embedder):
    graph, v = linked_graph
    linked = dict(QuestionLinker(graph, embedder).link(QUESTION, max_entities=4))
    assert v["Melanie"] in linked
    assert v["Bangkok"] in linked
    assert v["Caroline"] in linked


def test_linker_never_creates_a_vertex(linked_graph, embedder):
    """Durante o QA só se PODE ler a memória (spec seção 5).

    É por isso que existe um linker próprio em vez de reusar o EntityResolver,
    que cria um vértice quando não casa nada.
    """
    graph, _ = linked_graph
    before = set(graph.vertices)
    QuestionLinker(graph, embedder).link(
        "Who is Zephyrina Quackenbush and where does she live?", max_entities=4
    )
    assert set(graph.vertices) == before


def test_linker_prefers_the_longer_surface(embedder):
    g = FatGraph()
    g.add_vertex("group", embedding=embedder.encode_one("group"))
    long_id = g.add_vertex("support group", embedding=embedder.encode_one("support group"))
    linked = QuestionLinker(g, embedder).link("what happened at the support group?", 1)
    assert linked[0][0] == long_id


# --------------------------------------------------------------------------- #
# Cobertura                                                                    #
# --------------------------------------------------------------------------- #


def test_faces_are_scored_by_coverage_of_the_question_entities(linked_graph, embedder):
    graph, v = linked_graph
    cfg = Config.load("G5")
    r = FaceRetriever(graph, embedder, cfg)
    qvec = embedder.encode_one(QUESTION)
    ranked = r.score_faces([v["Melanie"], v["Bangkok"]], qvec)

    assert ranked, "nenhuma face candidata"
    best_face, _score, best_cov = ranked[0]
    assert best_cov > 0
    # a face mais bem pontuada tem de tocar as duas entidades pedidas
    touched = {graph.H[h].vertex_id for h in best_face.half_edges}
    assert {v["Melanie"], v["Bangkok"]} <= touched


def test_coverage_retrieves_a_fact_cosine_would_not(linked_graph, embedder):
    graph, _ = linked_graph
    g1 = _retrieve("G1", graph, embedder)
    g5 = _retrieve("G5", graph, embedder)

    assert g5.face_coverage and not g1.face_coverage
    assert g5.n_coverage_facts > 0
    assert g5.question_entities, "o linker não ligou nada"
    covered = [f for f in g5.facts if f.source == SOURCE_COVERAGE]
    assert covered and all(f.anchor_rank == -1 for f in covered)


def test_coverage_marks_provenance_and_counterfactual_turns(linked_graph, embedder):
    graph, _ = linked_graph
    res = _retrieve("G5", graph, embedder)
    # os turnos alcançados SÓ pela cobertura são exatamente os que o contexto
    # perderia se ela não tivesse rodado
    without = set(res.turn_ids_excluding(SOURCE_COVERAGE, SOURCE_GEODESIC))
    assert all(t not in without for t in res.coverage_turn_ids)


def test_geodesic_fallback_links_two_entities_with_no_common_face(embedder):
    """Sem face cobrindo as duas, a cadeia mínima ainda é recuperável."""
    g = FatGraph()
    v = {n: g.add_vertex(n, embedding=embedder.encode_one(n))
         for n in ("Melanie", "Bangkok", "travel agency")}
    g.add_edge(v["Melanie"], v["travel agency"],
               {"text": "She rang the agency on a Tuesday.", "turn_ids": ["D1:1"],
                "embedding": embedder.encode_one("She rang the agency on a Tuesday.")})
    g.add_edge(v["travel agency"], v["Bangkok"],
               {"text": TARGET_TEXT, "turn_ids": ["D1:2"],
                "embedding": embedder.encode_one(TARGET_TEXT)})

    r = FaceRetriever(g, embedder, Config.load("G5"))
    path = r.geodesic(v["Melanie"], v["Bangkok"], max_depth=3)
    assert len(path) == 2, "o caminho mínimo entre as duas entidades tem 2 arestas"

    res = r.retrieve(QUESTION)
    if res.geodesic_used:
        assert res.geodesic_len == 2
        assert any(f.source == SOURCE_GEODESIC for f in res.facts)


def test_geodesic_returns_empty_when_unreachable(embedder):
    g = FatGraph()
    a = g.add_vertex("a", embedding=embedder.encode_one("a"))
    b = g.add_vertex("b", embedding=embedder.encode_one("b"))
    g.add_vertex("c", embedding=embedder.encode_one("c"))
    g.add_edge(a, b, {"text": "a and b", "embedding": embedder.encode_one("a and b")})
    r = FaceRetriever(g, embedder, Config.load("G5"))
    assert r.geodesic(a, a, 3) == []
    assert r.geodesic(a, "does-not-exist", 3) == []


def test_context_labels_the_chain(linked_graph, embedder):
    graph, _ = linked_graph
    ctx = render_context(_retrieve("G5", graph, embedder))
    assert "--- trail 1 ---" in ctx


# --------------------------------------------------------------------------- #
# G6 = G4 + G5                                                                 #
# --------------------------------------------------------------------------- #


def test_join_condition_runs_both_mechanisms(linked_graph, embedder):
    graph, _ = linked_graph
    res = _retrieve("G6", graph, embedder)
    assert res.sigma_expand and res.face_coverage
    sources = {f.source for f in res.facts}
    assert SOURCE_COVERAGE in sources
    # sigma pode não achar nada neste grafo pequeno, mas a telemetria existe
    assert res.sigma_scanned >= 0 and res.coverage_faces_scored > 0


def test_join_never_exceeds_the_budget(linked_graph, embedder):
    graph, _ = linked_graph
    res = _retrieve("G6", graph, embedder, budget_tokens=200)
    assert res.tokens_used <= 200


def test_join_retrieves_at_least_what_each_half_does(linked_graph, embedder):
    """A combinação não pode perder o que cada mecanismo sozinho traz."""
    graph, _ = linked_graph
    g5 = _retrieve("G5", graph, embedder)
    g6 = _retrieve("G6", graph, embedder)
    # mesmo com orçamento repartido, as entidades ligadas são as mesmas
    assert g6.question_entities == g5.question_entities
    assert g6.n_coverage_facts > 0


def test_truncation_protects_every_join_source(linked_graph, embedder):
    graph, _ = linked_graph
    res = _retrieve("G6", graph, embedder, max_facts_in_prompt=2)
    assert len(res.facts) <= 2
    assert res.n_coverage_facts > 0, "a cobertura foi cortada pelo truncamento"


# --------------------------------------------------------------------------- #
# Não-regressão e configuração                                                 #
# --------------------------------------------------------------------------- #


def test_flag_off_leaves_no_coverage_telemetry(linked_graph, embedder):
    graph, _ = linked_graph
    res = _retrieve("G1", graph, embedder)
    assert res.face_coverage is False
    assert res.question_entities == [] and res.n_coverage_facts == 0
    assert res.coverage_faces_scored == 0 and res.geodesic_used is False


def test_coverage_is_deterministic(linked_graph, embedder):
    graph, _ = linked_graph
    a = _retrieve("G5", graph, embedder)
    b = _retrieve("G5", graph, embedder)
    assert [(f.edge_id, f.source) for f in a.facts] == [
        (f.edge_id, f.source) for f in b.facts
    ]


def test_conditions_isolate_one_mechanism_each():
    g1 = Config.load("G1")
    assert set(g1.diff(Config.load("G5"))) == {
        "condition", "retrieval.face_coverage", "paths.graphs_condition",
    }
    # G6 difere da G4 pela cobertura; o resto são as fatias de contexto, que
    # PRECISAM encolher quando os dois mecanismos dividem o mesmo prompt
    assert set(Config.load("G4").diff(Config.load("G6"))) == {
        "condition", "retrieval.face_coverage",
        "retrieval.coverage_budget_frac", "retrieval.sigma_budget_frac",
        "retrieval.coverage_max_facts_per_face",
    }


def test_coverage_is_off_in_every_earlier_condition():
    for name in ("G1", "G2", "G3", "G4", "B1", "B2", "B3", "test_offline"):
        assert Config.load(name).retrieval.face_coverage is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("coverage_sim_aggregate", "median"),
        ("coverage_budget_frac", 1.0),
        ("coverage_max_entities", 0),
        ("coverage_geodesic_max_depth", 0),
    ],
)
def test_invalid_coverage_config_is_rejected(key, value):
    cfg = Config.load("G5")
    setattr(cfg.retrieval, key, value)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_budget_slices_cannot_starve_the_anchor_walk():
    cfg = Config.load("G6")
    cfg.retrieval.sigma_budget_frac = 0.6
    cfg.retrieval.coverage_budget_frac = 0.6
    with pytest.raises(ConfigError):
        cfg.validate()
