"""A face como UNIDADE (G10) e o juiz LLM.

G10 nasce de quatro medições, não da hipótese original: percorrer φ perdeu 0.21
de recall multi-hop; escolher face por cobertura deu nulo com `coverage_best`
saturado em 0.955; permutar o prompt deu nulo em multi-hop (p=0.499), então a
sequência nunca foi o sinal; e a busca de gênero transformou 19 faces monstro
segurando 75% da memória numa distribuição unimodal. Sobrou a PERTENÇA, e ela só
existe depois do gênero — daí a pré-condição na validação.

O juiz existe porque nas perguntas cuja recuperação já colocou TODA a evidência
no prompt o F1 é 0.515: ou o respondedor falha, ou a métrica pune paráfrase, e
os dois pedem trabalho oposto.
"""

from __future__ import annotations

import json

import pytest

from fgl.config import Config, ConfigError
from fgl.core import FatGraph
from fgl.evaluation.judge import (
    Judge, agreement, disagreements, judge_metrics, load_predictions, write_judged,
)
from fgl.llm import FakeLLM, PromptLibrary
from fgl.paths import Paths, project_root
from fgl.retrieval import SOURCE_FACE_UNIT, FaceRetriever, render_context

QUESTION = "What did Melanie say about Bangkok?"


@pytest.fixture
def units_graph(embedder):
    """Vários blocos densos: gênero mínimo separa a memória em unidades."""
    g = FatGraph()
    for b in range(4):
        hubs = [g.add_vertex(f"b{b}h{i}", embedding=embedder.encode_one(f"b{b}h{i}"))
                for i in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                t = f"bloco {b}: {i} e {j} conversaram sobre Bangkok"
                g.add_edge(hubs[i], hubs[j],
                           {"text": t, "turn_ids": [f"D{b}:{i}{j}"], "session_id": f"S{b}",
                            "timestamp": "2023-05-08T13:56:00",
                            "embedding": embedder.encode_one(t)})
    g.maximize_faces(max_passes=4)
    return g


def _cfg(name="G10", **over):
    c = Config.load(name)
    for k, v in over.items():
        setattr(c.retrieval, k, v)
    return c.validate()


# --------------------------------------------------------------------------- #
# G10                                                                          #
# --------------------------------------------------------------------------- #


def test_returns_whole_faces_not_a_walk(units_graph, embedder):
    res = FaceRetriever(units_graph, embedder, _cfg()).retrieve(QUESTION)
    assert res.face_units is True
    assert res.facts and all(f.source == SOURCE_FACE_UNIT for f in res.facts)
    assert res.face_units_used >= 1


def test_the_memory_actually_splits_into_units(units_graph, embedder):
    """`face_units_used` = 1 sempre significaria "uma face gigante", que é o
    regime em que a G5 falhou. É a checagem que ela não tinha."""
    res = FaceRetriever(units_graph, embedder, _cfg()).retrieve(QUESTION)
    assert res.face_units_used > 1


def test_retrieves_facts_a_knn_would_not(units_graph, embedder):
    """A afirmação inteira do método: corroboração."""
    res = FaceRetriever(units_graph, embedder, _cfg()).retrieve(QUESTION)
    assert res.corroborating_facts > 0


def test_respects_the_budget(units_graph, embedder):
    res = FaceRetriever(units_graph, embedder, _cfg(budget_tokens=120)).retrieve(QUESTION)
    assert res.tokens_used <= 120


def test_faces_come_in_descending_relevance(units_graph, embedder):
    res = FaceRetriever(units_graph, embedder, _cfg()).retrieve(QUESTION)
    assert res.face_unit_scores == sorted(res.face_unit_scores, reverse=True)


def test_is_deterministic(units_graph, embedder):
    a = FaceRetriever(units_graph, embedder, _cfg()).retrieve(QUESTION)
    b = FaceRetriever(units_graph, embedder, _cfg()).retrieve(QUESTION)
    assert [f.edge_id for f in a.facts] == [f.edge_id for f in b.facts]
    assert a.face_unit_scores == b.face_unit_scores


def test_context_labels_groups_not_trails(units_graph, embedder):
    ctx = render_context(FaceRetriever(units_graph, embedder, _cfg()).retrieve(QUESTION))
    assert "related memories, group 1" in ctx
    assert "--- trail" not in ctx


def test_no_duplicate_facts(units_graph, embedder):
    res = FaceRetriever(units_graph, embedder, _cfg()).retrieve(QUESTION)
    ids = [f.edge_id for f in res.facts]
    assert len(ids) == len(set(ids))


def test_leaves_the_other_conditions_untouched(units_graph, embedder):
    """G10 é um caminho separado: os flags antigos não podem mudar de rota."""
    res = FaceRetriever(units_graph, embedder, Config.load("G1")).retrieve(QUESTION)
    assert res.face_units is False and res.face_units_used == 0
    assert all(f.source != SOURCE_FACE_UNIT for f in res.facts)


def test_requires_the_genus_search():
    """Sem gênero mínimo não há unidade a recuperar — a config recusa."""
    cfg = Config.load("G10")
    cfg.curation.maximize_faces = False
    with pytest.raises(ConfigError):
        cfg.validate()


@pytest.mark.parametrize("flag", ["sigma_expand", "face_coverage"])
def test_refuses_to_be_an_ensemble(flag):
    cfg = Config.load("G10")
    setattr(cfg.retrieval, flag, True)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_g10_isolates_the_change_of_unit_against_g9():
    """G9 e G10 partilham a superfície; o que muda é como ela é lida."""
    diff = set(Config.load("G9").diff(Config.load("G10")))
    assert "retrieval.face_units" in diff
    assert not {d for d in diff if d.startswith(("ingest.", "curation."))}


# --------------------------------------------------------------------------- #
# Juiz                                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture
def judge(cfg):
    return Judge(FakeLLM(cfg.llm), PromptLibrary(Paths.build(project_root()).prompts))


def _row(cat, gold, pred, f1=0.0):
    return {"question": "q?", "category": cat, "category_name": str(cat),
            "gold": gold, "prediction": pred, "f1": f1}


def test_adversarial_is_decided_by_rule_never_by_the_llm(judge):
    """Categoria 5 é 22% do benchmark e é decidível por regra."""
    yes = judge.judge_row(_row(5, "Not mentioned", "Not mentioned in the conversation"))
    no = judge.judge_row(_row(5, "Not mentioned", "Bangkok"))
    assert yes.correct and yes.by_rule
    assert not no.correct and no.by_rule
    assert judge.llm.usage.calls == 0, "não pode gastar chamada para reexecutar uma regex"


def test_empty_answer_is_wrong_without_asking(judge):
    j = judge.judge_row(_row(1, "Bangkok", "   "))
    assert not j.correct and j.by_rule
    assert judge.llm.usage.calls == 0


def test_judge_accepts_a_paraphrase(judge):
    """O caso que motivou tudo: certo em outras palavras, F1 baixo."""
    j = judge.judge_row(_row(3, "Psychology, counseling certification",
                             "counseling certification in psychology", f1=0.2))
    assert j.correct and not j.by_rule


def test_judge_rejects_a_different_fact(judge):
    assert not judge.judge_row(_row(2, "8 May 2023", "unrelated painting hobby")).correct


def test_judge_never_sees_the_conversation(judge, prompts):
    """Com o contexto, o juiz passa a responder a pergunta e mede a si mesmo."""
    text = prompts.render("judge", question="q", gold="g", prediction="p")
    low = text.lower()
    assert "{context}" not in text
    assert "memories" not in low and "conversation history" not in low


def test_metrics_report_both_scores_and_the_gap(judge):
    rows = [_row(1, "Bangkok", "Bangkok", 1.0), _row(1, "Bangkok", "Paris", 0.0),
            _row(5, "Not mentioned", "Not mentioned in the conversation", 1.0)]
    m = judge_metrics(judge.judge_all(rows))
    assert {"judge_micro", "judge_substantive", "judge_macro",
            "judge_f1_agreement", "judge_yes_f1_low", "judge_no_f1_high"} <= set(m)
    assert m["judge_calls"] == 2, "só as substantivas custam chamada"


def test_agreement_counts_both_directions_of_disagreement(judge):
    rows = [_row(1, "counseling psychology", "psychology counseling work", 0.1),
            _row(1, "Bangkok", "Bangkok", 1.0)]
    judged = judge.judge_all(rows)
    a = agreement(judged)
    assert a["judge_n_scored"] == 2
    assert a["judge_yes_f1_low"] + a["judge_no_f1_high"] <= 2


def test_disagreements_are_readable_for_hand_checking(judge):
    rows = [_row(1, "counseling psychology", "psychology counseling work", 0.1)]
    out = disagreements(judge.judge_all(rows), limit=5)
    for d in out:
        assert {"gold", "prediction", "f1", "judge", "reason"} <= set(d)


def test_writing_is_additive_and_keeps_f1(judge, tmp_path):
    """`f1` nunca some: as duas métricas têm de continuar comparáveis."""
    path = tmp_path / "predictions.jsonl"
    rows = [_row(1, "Bangkok", "Bangkok", 1.0), _row(5, "Not mentioned", "Not mentioned", 1.0)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    judged = judge.judge_all(load_predictions(path))
    write_judged(path, judged)
    back = load_predictions(path)
    assert len(back) == 2
    for r in back:
        assert {"f1", "judge", "judge_reason", "judge_by_rule"} <= set(r)
    assert back[0]["f1"] == 1.0


def test_judging_is_deterministic(judge):
    rows = [_row(1, "Bangkok in June", "June, in Bangkok", 0.3)]
    a = judge.judge_all(rows)[0].correct
    b = judge.judge_all(rows)[0].correct
    assert a == b


def test_empty_input_is_not_a_score(judge):
    assert judge_metrics([]) == {}
    assert agreement([]) == {}
