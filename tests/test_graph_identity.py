"""Autossuficiência das condições, e o fingerprint que a torna verificável.

Antes, G4/G5/G6 liam o diretório de grafos da G1 (`paths.graphs_condition`).
Isso garantia por construção que o delta isolasse a recuperação, mas ao preço de
tornar os resultados de uma condição um artefato da rodada de outra: a G4 não
podia ser reproduzida sem antes reproduzir a G1, e um grafo gerado por uma versão
antiga do código entrava silenciosamente na rodada nova.

Agora cada condição constrói o seu, e a igualdade passou de imposta a MEDIDA:
ingest idêntico sobre o mesmo cache de fatos tem de produzir o mesmo ribbon
graph, e `FatGraph.fingerprint` é endereçado por conteúdo, então a igualdade do
hash é igualdade de memória E de rotação.
"""

from __future__ import annotations

import pytest

from fgl.config import Config
from fgl.core import FatGraph
from fgl.evaluation import graph_identity_table


# --------------------------------------------------------------------------- #
# Propriedades do fingerprint                                                  #
# --------------------------------------------------------------------------- #


def _triangle(names=("Caroline", "Melanie", "support group"), texts=None):
    g = FatGraph()
    v = [g.add_vertex(n) for n in names]
    texts = texts or ["a e b", "b e c", "c e a"]
    for i, t in enumerate(texts):
        g.add_edge(v[i], v[(i + 1) % len(v)], {"text": t})
    return g, v


def test_same_memory_same_fingerprint():
    assert _triangle()[0].fingerprint() == _triangle()[0].fingerprint()


def test_fingerprint_ignores_id_assignment():
    """`V3`/`E17` dependem da ordem de inserção; a memória não."""
    a, _ = _triangle()
    b = FatGraph()
    # mesmos vértices e arestas, inseridos em ordem diferente
    vb = {n: b.add_vertex(n) for n in ("support group", "Melanie", "Caroline")}
    b.add_edge(vb["Melanie"], vb["support group"], {"text": "b e c"})
    b.add_edge(vb["support group"], vb["Caroline"], {"text": "c e a"})
    b.add_edge(vb["Caroline"], vb["Melanie"], {"text": "a e b"})
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_ignores_embeddings():
    """Floats de um BLAS diferente não podem contar como memória diferente."""
    import numpy as np

    a, va = _triangle()
    b, vb = _triangle()
    for vid in vb:
        b.vertices[vid].embedding = np.random.rand(8).astype(np.float32)
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_when_a_fact_changes():
    a, _ = _triangle()
    b, _ = _triangle(texts=["a e b", "b e c", "OUTRO FATO"])
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_changes_when_an_entity_changes():
    a, _ = _triangle()
    b, _ = _triangle(names=("Caroline", "Melanie", "book club"))
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_is_sensitive_to_sigma():
    """É ribbon graph, não grafo: a rotação faz parte da identidade.

    É isto que faz a G9 (sigma por gênero mínimo) hashear diferente da G1 sobre
    a mesma memória — corretamente, porque é outra superfície.
    """
    g = FatGraph()
    hub = g.add_vertex("hub")
    for i in range(4):
        g.add_edge(hub, g.add_vertex(f"n{i}"), {"text": f"fato {i}"})
    before = g.fingerprint()
    g.transpose_sigma(hub, 0, 2)
    assert g.fingerprint() != before


def test_fingerprint_is_invariant_to_rotating_sigma():
    """Uma ordem cíclica não tem começo: girá-la denota o mesmo mergulho."""
    g = FatGraph()
    hub = g.add_vertex("hub")
    for i in range(4):
        g.add_edge(hub, g.add_vertex(f"n{i}"), {"text": f"fato {i}"})
    before = g.fingerprint()
    g.sigma[hub] = g.sigma[hub][1:] + g.sigma[hub][:1]
    g._reindex_vertex(hub)
    assert g.fingerprint() == before, "girar sigma mudou o fingerprint"


def test_fingerprint_is_reported_in_stats():
    assert "fingerprint" in _triangle()[0].stats()


def test_genus_optimisation_changes_the_fingerprint():
    g = FatGraph()
    hubs = [g.add_vertex(f"h{i}") for i in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(hubs[i], hubs[j], {"text": f"hub {i}-{j}"})
        for k in range(3):
            g.add_edge(hubs[i], g.add_vertex(f"l{i}_{k}"), {"text": f"leaf {i}-{k}"})
    before = g.fingerprint()
    rep = g.maximize_faces(max_passes=3)
    if rep["moves_applied"]:
        assert g.fingerprint() != before


# --------------------------------------------------------------------------- #
# Autossuficiência das condições                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name", ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "B1", "B2", "B3"]
)
def test_no_condition_borrows_another_s_graphs(name):
    """Nenhuma condição pode depender do artefato de uma rodada anterior."""
    assert Config.load(name).paths.graphs_condition == "", (
        f"{name} empresta grafos e não é reproduzível isoladamente"
    )


def test_retrieval_only_conditions_still_differ_from_g1_only_in_retrieval():
    """Sem o compartilhamento, o ingest tem de continuar idêntico ao da G1."""
    g1 = Config.load("G1")
    ingest_keys = {"ingest.", "curation."}
    for name in ("G4", "G5", "G6", "G7", "G8"):
        diff = set(g1.diff(Config.load(name)))
        offenders = {
            d for d in diff if any(d.startswith(p) for p in ingest_keys)
        }
        assert not offenders, f"{name} difere da G1 no ingest: {offenders}"


# --------------------------------------------------------------------------- #
# A tabela de verificação                                                      #
# --------------------------------------------------------------------------- #


def _metrics(fp_by_sample: dict) -> dict:
    return {
        "per_conversation": [
            {"sample_id": s, "graph": {"fingerprint": fp}}
            for s, fp in fp_by_sample.items()
        ]
    }


def test_identity_table_confirms_matching_graphs():
    res = {
        "G1-fatgraph-min": _metrics({"conv-26": "abc", "conv-30": "def"}),
        "G4-fatgraph-sigma": _metrics({"conv-26": "abc", "conv-30": "def"}),
    }
    out = graph_identity_table(res)
    assert "idêntico à G1" in out
    assert "DIVERGE" not in out


def test_identity_table_flags_a_mismatch():
    """Se divergir, o delta deixa de isolar recuperação — e tem de gritar."""
    res = {
        "G1-fatgraph-min": _metrics({"conv-26": "abc", "conv-30": "def"}),
        "G4-fatgraph-sigma": _metrics({"conv-26": "abc", "conv-30": "OUTRO"}),
    }
    out = graph_identity_table(res)
    assert "DIVERGE" in out
    assert "não isola recuperação" in out


def test_identity_table_without_g1():
    assert "rode a G1" in graph_identity_table({"G4-fatgraph-sigma": _metrics({})})
