"""Extração tópica: o falante sai da topologia e fica no texto.

O diagnóstico que motivou isto, medido sobre os grafos reais:

    turnos de evidência extraídos      92.5%   a extração está boa
    fatos no mesmo componente          95.8%   a conectividade está boa
    compartilham um vértice            88.4%   a "ponte" existe...
    grau MEDIANO do vértice-ponte        164   ...e é o falante
    86.6% das arestas tocam um falante

Em cada conversa os dois maiores vértices são os falantes (~200) e o terceiro
tem 17. "Duas memórias compartilham uma entidade" quer dizer "as duas são sobre
a Caroline", o que vale para metade do grafo — e a órbita de sigma cobre a ponte
real em 7.3% dos casos. G4 a G10 foram testadas sobre esse substrato.

O `fact_text` NÃO muda de propósito: a B3 embute as mesmas frases, então a única
variável entre ela e as condições de grafo passa a ser a atribuição de vértices.
"""

from __future__ import annotations

import json

import pytest

from fgl.config import Config, ConfigError
from fgl.data.locomo import Conversation, Session, Turn
from fgl.llm import FakeLLM, PromptLibrary
from fgl.memory.ingest import Fact, FactExtractor
from fgl.paths import Paths, project_root


@pytest.fixture
def prompts_lib():
    return PromptLibrary(Paths.build(project_root()).prompts)


def _conv():
    turns = [
        Turn(dia_id="D1:1", speaker="Caroline", session_num=1,
             text="I found the transgender stories at the support group inspiring."),
        Turn(dia_id="D1:2", speaker="Melanie", session_num=1,
             text="The camping trip with my children is next month."),
    ]
    return Conversation(
        sample_id="t1", speaker_a="Caroline", speaker_b="Melanie",
        sessions=[Session(num=1, date_time_raw="1:56 pm on 8 May, 2023",
                          timestamp="2023-05-08T13:56:00", turns=turns)],
        questions=[],
    )


# --------------------------------------------------------------------------- #
# O prompt                                                                     #
# --------------------------------------------------------------------------- #


def test_topical_prompt_exists_and_teaches_the_choice(prompts_lib):
    text = prompts_lib.render(
        "extract_facts_topical", speaker_a="Caroline", speaker_b="Melanie",
        session_date="8 May 2023", transcript="[D1:1] Caroline: oi",
    )
    low = text.lower()
    # o exemplo errado/certo é o coração do prompt: sem ele o modelo volta a
    # ancorar tudo no falante
    assert "wrong" in low and "right" in low
    assert "speaker" in low
    # e continua exigindo o nome próprio na frase, senão a B3 perde semântica
    assert "proper names" in low


def test_topical_prompt_still_asks_for_the_speaker_in_the_sentence(prompts_lib):
    text = prompts_lib.render(
        "extract_facts_topical", speaker_a="A", speaker_b="B",
        session_date="d", transcript="t",
    )
    assert "fact_text" in text
    assert '"speaker"' in text or "`speaker`" in text


def test_both_prompts_share_the_task_marker(prompts_lib):
    """O responder fake despacha pelo marcador; os dois têm de casar."""
    for name in ("extract_facts", "extract_facts_topical"):
        text = prompts_lib.render(
            name, speaker_a="A", speaker_b="B", session_date="d", transcript="t"
        )
        assert "# TASK: extract_facts" in text


# --------------------------------------------------------------------------- #
# O campo speaker                                                              #
# --------------------------------------------------------------------------- #


def test_speaker_is_carried_as_metadata(cfg, prompts_lib, tmp_path):
    fx = FactExtractor(FakeLLM(cfg.llm), prompts_lib, tmp_path,
                       prompt_name="extract_facts_topical")
    facts = fx.extract_all(_conv())
    assert facts
    assert all(f.speaker in ("Caroline", "Melanie") for f in facts)


def test_a_hallucinated_speaker_is_dropped(cfg, prompts_lib, tmp_path):
    """Só os dois falantes reais; qualquer outro nome vira vazio."""
    def responder(prompt, system):
        return json.dumps({"facts": [{
            "entity_1": "a", "entity_2": "b", "relation": "r",
            "speaker": "Napoleão", "fact_text": "a e b", "turn_ids": ["D1:1"],
        }]})

    fx = FactExtractor(FakeLLM(cfg.llm, responder=responder), prompts_lib, tmp_path,
                       prompt_name="extract_facts_topical")
    assert fx.extract_all(_conv())[0].speaker == ""


def test_speaker_survives_the_cache_round_trip(cfg, prompts_lib, tmp_path):
    fx = FactExtractor(FakeLLM(cfg.llm), prompts_lib, tmp_path,
                       prompt_name="extract_facts_topical")
    first = fx.extract_all(_conv())
    second = fx.extract_all(_conv())  # agora vem do disco
    assert [f.speaker for f in first] == [f.speaker for f in second]


def test_speaker_is_not_an_edge_attribute():
    """O falante é metadado do fato, não atributo de aresta -- é o ponto todo."""
    from fgl.core.fatgraph import EDGE_LEVEL_ATTRS

    assert "speaker" not in EDGE_LEVEL_ATTRS


# --------------------------------------------------------------------------- #
# Cache: nada do que já foi pago pode ser perdido                              #
# --------------------------------------------------------------------------- #


def test_each_prompt_gets_its_own_cache(cfg, prompts_lib, tmp_path):
    conv = _conv()
    paths = {
        name: FactExtractor(FakeLLM(cfg.llm), prompts_lib, tmp_path, prompt_name=name)
        ._cache_path(conv, conv.sessions[0])
        for name in ("extract_facts", "extract_facts_topical")
    }
    assert paths["extract_facts"] != paths["extract_facts_topical"]


def test_the_default_prompt_keeps_its_historical_cache_tag(cfg, prompts_lib, tmp_path):
    """Inserir o nome na tag renomearia tudo e re-extrairia 10 conversas pagas."""
    conv = _conv()
    fx = FactExtractor(FakeLLM(cfg.llm), prompts_lib, tmp_path)
    tag = fx._cache_path(conv, conv.sessions[0]).parent.parent.name
    assert tag == f"{cfg.llm.deployment}-{prompts_lib.version('extract_facts')}"
    assert "extract_facts" not in tag.replace(cfg.llm.deployment, "")


# --------------------------------------------------------------------------- #
# Configuração                                                                 #
# --------------------------------------------------------------------------- #


def test_t1_differs_from_g1_only_in_the_extraction():
    """O delta T1 - G1 tem de ser atribuível só ao prompt de extração."""
    assert set(Config.load("G1").diff(Config.load("T1"))) == {
        "condition", "ingest.extract_prompt",
    }


def test_t1_leaves_every_retrieval_mechanism_off():
    r = Config.load("T1").retrieval
    assert not (r.sigma_expand or r.face_coverage or r.face_units)


def test_unknown_extraction_prompt_is_rejected():
    cfg = Config.load("T1")
    cfg.ingest.extract_prompt = "não existe"
    with pytest.raises(ConfigError):
        cfg.validate()


def test_every_other_condition_still_uses_v1():
    for name in ("G1", "G2", "G3", "G4", "G9", "G10", "B1", "B2", "B3"):
        assert Config.load(name).ingest.extract_prompt == "extract_facts"


# --------------------------------------------------------------------------- #
# Fact                                                                         #
# --------------------------------------------------------------------------- #


def test_fact_round_trips_the_speaker():
    f = Fact(entity_1="a", relation="r", entity_2="b", fact_text="a e b",
             turn_ids=["D1:1"], speaker="Caroline")
    assert Fact.from_dict(f.to_dict()).speaker == "Caroline"


def test_fact_without_a_speaker_is_valid():
    """Fatos v1 no cache não têm o campo; carregá-los não pode quebrar."""
    assert Fact.from_dict({
        "entity_1": "a", "relation": "r", "entity_2": "b",
        "fact_text": "t", "turn_ids": [],
    }).speaker == ""
