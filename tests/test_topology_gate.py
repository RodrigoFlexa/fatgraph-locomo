"""Topologia da memória: o pré-requisito que G4/G5/G6 têm em comum.

Quando o grafo é uma estrela, os dois mecanismos de multi-hop são redundantes
POR CONSTRUÇÃO -- num vértice de grau 1, ``sigma(alpha(h)) = alpha(h)``, então
``phi`` degenera em marchar pela órbita do próprio hub e a face já entrega o que
sigma proporia; e as faces ficam enormes e tocam quase todo vértice, de modo que
a cobertura não consegue ordená-las. As condições não falham alto nesse regime:
elas reproduzem a G1 em silêncio e a tabela mostra três números iguais como se
fossem resultado.

Estes testes fixam (a) as métricas que detectam a forma e (b) o extrator fake,
cuja versão anterior fabricava exatamente essa degenerescência e portanto fazia
o ``--dry-run`` reportar uma patologia do dublê de teste como se fosse da LoCoMo.
"""

from __future__ import annotations

import json

import pytest

from fgl.core import FatGraph
from fgl.llm.client import _fake_extract, default_fake_responder
from fgl.pipeline import STAR_DEGREE1_FRAC, STAR_HUB_SHARE, Runner


# --------------------------------------------------------------------------- #
# star_stats                                                                   #
# --------------------------------------------------------------------------- #


def _star(n_leaves: int) -> FatGraph:
    g = FatGraph()
    hub = g.add_vertex("hub")
    for i in range(n_leaves):
        leaf = g.add_vertex(f"leaf{i}")
        g.add_edge(hub, leaf, {"text": f"fato {i}"})
    return g


def _cycle(n: int) -> FatGraph:
    g = FatGraph()
    vs = [g.add_vertex(f"v{i}") for i in range(n)]
    for i in range(n):
        g.add_edge(vs[i], vs[(i + 1) % n], {"text": f"fato {i}"})
    return g


def test_star_stats_flags_a_star():
    s = _star(30).star_stats()
    assert s["degree_1_frac"] > STAR_DEGREE1_FRAC
    assert s["hub_share"] > STAR_HUB_SHARE


def test_star_stats_clears_a_cycle():
    """Num ciclo todo vértice tem grau 2 e nenhum hub concentra nada."""
    s = _cycle(30).star_stats()
    assert s["degree_1_frac"] == 0.0
    assert s["hub_share"] < STAR_HUB_SHARE


def test_star_stats_is_reported_by_stats():
    assert {"degree_1_frac", "hub_share"} <= set(_cycle(6).stats())


def test_star_stats_on_an_empty_graph():
    assert FatGraph().star_stats() == {"degree_1_frac": 0.0, "hub_share": 0.0}


# --------------------------------------------------------------------------- #
# O aviso                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def runner(cfg, llm, embedder, prompts):
    return Runner(cfg, llm=llm, embedder=embedder, prompts=prompts)


def test_warns_when_the_graph_is_a_star(runner):
    warnings = runner._topology_warnings([{"graph": _star(30).stats()}])
    assert len(warnings) == 1
    msg = warnings[0]
    assert "ESTRELA" in msg
    # o aviso tem de apontar a causa a montante, senão manda ajustar o knob errado
    assert "INGEST" in msg


def test_stays_quiet_on_a_healthy_graph(runner):
    assert runner._topology_warnings([{"graph": _cycle(30).stats()}]) == []


def test_no_graph_stats_means_no_opinion(runner):
    """Baselines não constroem grafo: nada a dizer, e nada a inventar."""
    assert runner._topology_warnings([]) == []
    assert runner._topology_warnings([{"graph": {}}]) == []


def test_the_warning_reaches_the_sanity_block(runner):
    from fgl.evaluation import QAOutcome

    outcomes = [
        QAOutcome(question=f"q{i}", category=1, gold="x", prediction="x",
                  f1=1.0, evidence=[], retrieved_turn_ids=[], recall={},
                  n_facts=3, n_faces=1, tokens_context=10, abstained=False)
        for i in range(30)
    ]
    sanity = runner._sanity(outcomes, [{"graph": _star(30).stats()}])
    assert sanity["ok"] is False
    assert any("ESTRELA" in w for w in sanity["warnings"])


# --------------------------------------------------------------------------- #
# O extrator fake                                                              #
# --------------------------------------------------------------------------- #


PROMPT = """# TASK: extract_facts
[D1:1] Caroline: I finally joined the support group downtown last Tuesday.
[D1:2] Melanie: The support group in Bangkok changed everything for my family.
[D1:3] Caroline: Bangkok was where my family spent that whole summer.
[D1:4] Melanie: Painting kept my family together through the worst of it.
[D1:5] Caroline: Painting is what the support group does every Thursday.
"""


def test_fake_extract_pairs_entities_not_speakers():
    """A versão antiga emitia (falante, primeira-palavra-longa): estrela dupla."""
    facts = _fake_extract(PROMPT)
    assert facts
    speakers = {"Caroline", "Melanie"}
    assert any(
        f["entity_1"] not in speakers and f["entity_2"] not in speakers
        for f in facts
    ), "nenhum fato liga duas entidades que não sejam os falantes"


def test_fake_extract_reuses_entities_across_turns():
    """Recorrência é o que dá grau > 1 nas duas pontas -- e faces de verdade."""
    facts = _fake_extract(PROMPT)
    ends = [e for f in facts for e in (f["entity_1"], f["entity_2"])]
    shared = {e for e in ends if ends.count(e) > 1}
    assert len(shared) >= 2, f"entidades não recorrem: {ends}"


def test_fake_extract_drops_contractions_and_stopwords():
    prompt = "# TASK: extract_facts\n[D1:1] Caroline: That's really about what I've been doing.\n"
    for f in _fake_extract(prompt):
        for end in (f["entity_1"], f["entity_2"]):
            assert "'" in end or end == "Caroline" or end.isalpha()
            assert end.lower() not in {"thats", "that", "really", "about", "been"}


def test_fake_extract_is_deterministic():
    assert _fake_extract(PROMPT) == _fake_extract(PROMPT)


def test_fake_extract_survives_a_prompt_with_no_turns():
    assert _fake_extract("# TASK: extract_facts\n(nada aqui)\n") == []


def test_the_default_responder_still_returns_valid_json():
    out = json.loads(default_fake_responder(PROMPT, None))
    assert "facts" in out
    for f in out["facts"]:
        assert {"entity_1", "entity_2", "fact_text", "turn_ids"} <= set(f)


def test_a_cached_report_gets_the_new_metrics_backfilled():
    """Grafos construídos antes da métrica existir não podem calar o gate.

    É o caso real: os grafos da G1 no servidor são anteriores a `star_stats`,
    e sem isto a G4/G5/G6 rodariam sobre eles com o aviso desligado -- mudo
    justamente nos artefatos que ninguém reexaminou.
    """
    from fgl.pipeline import _refresh_graph_stats

    graph = _star(30)
    old_report = {"n_facts": 30, "graph_stats": {"V": 31, "E": 30}}
    fresh = _refresh_graph_stats(graph, old_report)

    assert fresh["graph_stats"]["degree_1_frac"] > STAR_DEGREE1_FRAC
    assert fresh["n_facts"] == 30, "o resto do relatório não pode ser tocado"
    # o que o ingest registrou sobre a RODADA prevalece sobre o recálculo
    assert fresh["graph_stats"]["V"] == 31


def test_refresh_leaves_a_report_without_graph_stats_alone():
    from fgl.pipeline import _refresh_graph_stats

    assert _refresh_graph_stats(_star(3), {"n_facts": 3}) == {"n_facts": 3}


def test_the_fake_graph_is_no_longer_a_star(cfg, llm, embedder, prompts):
    """O teste que fecha o buraco: o dry-run tem de produzir topologia honesta.

    Sem isto, ``--dry-run`` e a pytest exercitam os caminhos de código de
    G4/G5/G6 sobre um grafo em que eles não podem funcionar, e os números
    resultantes descrevem o dublê, não o método.
    """
    from fgl.data.locomo import Conversation, Session, Turn
    from fgl.memory.ingest import Ingestor

    topics = ["support group", "Bangkok", "family", "painting", "community"]
    turns = [
        Turn(
            dia_id=f"D1:{i + 1}",
            speaker="Caroline" if i % 2 else "Melanie",
            text=(
                f"The {topics[i % len(topics)]} and the "
                f"{topics[(i + 1) % len(topics)]} came up again that week."
            ),
            session_num=1,
        )
        for i in range(24)
    ]
    conv = Conversation(
        sample_id="star-check", speaker_a="Caroline", speaker_b="Melanie",
        sessions=[Session(num=1, date_time_raw="1:56 pm on 8 May, 2023",
                          timestamp="2023-05-08T13:56:00", turns=turns)],
        questions=[],
    )
    graph, _ = Ingestor(cfg, llm, embedder, prompts).ingest(conv)
    s = graph.star_stats()
    assert not (s["degree_1_frac"] > STAR_DEGREE1_FRAC
                and s["hub_share"] > STAR_HUB_SHARE), (
        f"o extrator fake voltou a fabricar uma estrela: {s}"
    )
