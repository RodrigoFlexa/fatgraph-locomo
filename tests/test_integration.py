"""End-to-end integration test on a synthetic 3-session mini-conversation.

The expected fatgraph is known by hand: **3 vertices, 3 edges, 2 faces, g = 0**
(the triangle of ``conftest.build_triangle``).  The whole pipeline runs --
extraction, entity resolution, sigma-time insertion, incongruence check,
retrieval by faces, answering and official scoring -- with a scripted offline
LLM, so the test is deterministic and free.
"""

from __future__ import annotations

import json

import pytest
from conftest import PATHS

from fgl.config import Config
from fgl.retrieval import HashingEmbedder
from fgl.evaluation import aggregate, score_question
from fgl.memory.ingest import Ingestor
from fgl.llm import FakeLLM
from fgl.data.locomo import Conversation, Question, Session, Turn
from fgl.pipeline import Runner
from fgl.llm.prompts import PromptLibrary
from fgl.retrieval import Answerer, FaceRetriever

# --------------------------------------------------------------------------- #
# The mini-conversation                                                        #
# --------------------------------------------------------------------------- #

SESSIONS = [
    (1, "1:56 pm on 8 May, 2023", [
        ("D1:1", "Caroline", "Hey Mel! We have known each other for so long."),
        ("D1:2", "Melanie", "We really are close friends, Caroline."),
    ]),
    (2, "10:10 am on 20 May, 2023", [
        ("D2:1", "Melanie", "You should try the support group I go to."),
        ("D2:2", "Caroline", "That sounds good, thanks."),
    ]),
    (3, "6:30 pm on 7 June, 2023", [
        ("D3:1", "Caroline", "I finally went to the support group last month."),
        ("D3:2", "Melanie", "I am so glad you did."),
    ]),
]

#: One atomic fact per session -- exactly what closes the expected triangle.
SCRIPT = {
    1: [{
        "entity_1": "Caroline", "relation": "is a close friend of",
        "entity_2": "Melanie",
        "fact_text": "Caroline and Melanie are close friends.",
        "turn_ids": ["D1:2"],
    }],
    2: [{
        "entity_1": "Melanie", "relation": "recommended",
        "entity_2": "support group",
        "fact_text": "Melanie recommended the support group to Caroline.",
        "turn_ids": ["D2:1"],
    }],
    3: [{
        "entity_1": "support group", "relation": "was attended by",
        "entity_2": "Caroline",
        "fact_text": "Caroline went to the support group in May 2023.",
        "turn_ids": ["D3:1"],
    }],
}

QUESTIONS = [
    Question("When did Caroline go to the support group?", "May 2023", 2, ["D3:1"]),
    Question("Who recommended the support group to Caroline?", "Melanie", 4, ["D2:1"]),
    Question("What car does Melanie drive?", "Not mentioned in the conversation", 5,
             ["D2:2"]),
]


def build_conversation() -> Conversation:
    from fgl.data.locomo import normalize_timestamp

    sessions = []
    for num, date, turns in SESSIONS:
        s = Session(num=num, date_time_raw=date, timestamp=normalize_timestamp(date))
        s.turns = [
            Turn(dia_id=d, speaker=sp, text=tx, session_num=num) for d, sp, tx in turns
        ]
        sessions.append(s)
    return Conversation(
        sample_id="synthetic-3", speaker_a="Caroline", speaker_b="Melanie",
        sessions=sessions, questions=list(QUESTIONS),
    )


def scripted_responder(prompt: str, system: str | None) -> str:
    """Deterministic stand-in that follows SCRIPT and answers extractively."""
    if "# TASK: extract_facts" in prompt:
        for num, _date, turns in SESSIONS:
            if turns[0][0] in prompt:
                return json.dumps({"facts": SCRIPT[num]})
        return json.dumps({"facts": []})
    if "# TASK: incongruence" in prompt:
        return json.dumps({"contradiction": False, "reason": "no conflict"})
    if "# TASK: entity_match" in prompt:
        return json.dumps({"same": False, "reason": "distinct"})
    if "# TASK: redundancy" in prompt:
        return json.dumps({"redundant": False, "merged_text": "", "reason": "distinct"})
    if "# TASK: consolidate" in prompt:
        return json.dumps({"summary": "Caroline, Melanie and the support group."})
    if "# TASK: sigma_agent" in prompt:
        return json.dumps({"after_index": -1, "reason": "front"})
    if "# TASK: answer" in prompt:
        q = prompt.split("QUESTION:")[-1].splitlines()[0].strip().lower()
        if "when" in q and "support group" in prompt.lower():
            return "May 2023"
        if "who recommended" in q:
            return "Melanie"
        return "Not mentioned in the conversation"
    return "Not mentioned in the conversation"


@pytest.fixture
def offline(cfg, tmp_path):
    cfg.condition = "TEST-offline"
    llm = FakeLLM(cfg.llm, responder=scripted_responder)
    embedder = HashingEmbedder(cfg.embeddings.dim)
    prompts = PromptLibrary(PATHS.prompts)
    return cfg, llm, embedder, prompts


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def test_ingestion_builds_the_expected_fatgraph(offline):
    cfg, llm, embedder, prompts = offline
    graph, report = Ingestor(cfg, llm, embedder, prompts).ingest(build_conversation())

    v, e, f, g = graph.euler().as_tuple()
    assert (v, e, f, g) == (3, 3, 2, 0), "expected the hand-built triangle"
    assert graph.euler().C == 1
    assert report.n_facts == 3
    assert report.n_edges == 3
    assert report.n_incongruent == 0
    graph.check_invariants()

    names = sorted(vx.name for vx in graph.vertices.values())
    assert names == ["Caroline", "Melanie", "support group"]
    assert sorted(f.length for f in graph.faces()) == [3, 3]


def test_entities_are_reused_across_sessions(offline):
    cfg, llm, embedder, prompts = offline
    graph, _ = Ingestor(cfg, llm, embedder, prompts).ingest(build_conversation())
    # "Caroline" appears in sessions 1 and 3 and must be a single vertex
    caroline = [vid for vid, vx in graph.vertices.items() if vx.name == "Caroline"]
    assert len(caroline) == 1
    assert graph.degree(caroline[0]) == 2


def test_per_session_report_tracks_growth(offline):
    cfg, llm, embedder, prompts = offline
    _, report = Ingestor(cfg, llm, embedder, prompts).ingest(build_conversation())
    assert [s["E"] for s in report.per_session] == [1, 2, 3]
    assert [s["V"] for s in report.per_session] == [2, 3, 3]
    assert [s["F"] for s in report.per_session] == [1, 1, 2]
    for s in report.per_session:
        assert s["genus"] == 0


def test_provenance_points_back_to_the_right_turns(offline):
    cfg, llm, embedder, prompts = offline
    graph, _ = Ingestor(cfg, llm, embedder, prompts).ingest(build_conversation())
    turns = {t for e in graph.edges() for t in graph.get_edge_attr(e, "turn_ids")}
    assert turns == {"D1:2", "D2:1", "D3:1"}


def test_retrieval_walks_a_face_and_answers(offline):
    cfg, llm, embedder, prompts = offline
    conv = build_conversation()
    graph, _ = Ingestor(cfg, llm, embedder, prompts).ingest(conv)

    retriever = FaceRetriever(
        graph, embedder, cfg, {s.id: s.date_time_raw for s in conv.sessions}
    )
    answerer = Answerer(llm, prompts, cfg)

    result = retriever.retrieve(conv.questions[1].prompt_question())
    assert result.facts, "at least one fact must be retrieved"
    assert len({f.edge_id for f in result.facts}) == len(result.facts), "no duplicates"
    assert result.tokens_used <= cfg.retrieval.budget_tokens + 64

    prediction = answerer.answer(conv, conv.questions[1], result)
    assert score_question(conv.questions[1], prediction) == pytest.approx(1.0)


def test_adversarial_question_is_answered_with_the_abstention_string(offline):
    cfg, llm, embedder, prompts = offline
    conv = build_conversation()
    graph, _ = Ingestor(cfg, llm, embedder, prompts).ingest(conv)
    retriever = FaceRetriever(graph, embedder, cfg)
    answerer = Answerer(llm, prompts, cfg)

    q = conv.questions[2]
    prediction = answerer.answer(conv, q, retriever.retrieve(q.prompt_question()))
    assert "not mentioned" in prediction.lower()
    assert score_question(q, prediction) == 1.0


def test_full_runner_writes_metrics_and_predictions(offline, tmp_path):
    cfg, llm, embedder, prompts = offline
    # every backend is injected, so the Runner touches neither network nor models
    runner = Runner(cfg, root=tmp_path, llm=llm, embedder=embedder, prompts=prompts)

    metrics = runner.run([build_conversation()])

    assert metrics["overall"]["n"] == 3
    assert set(metrics["per_category"]) == {"temporal", "single-hop", "adversarial"}
    assert metrics["per_category"]["adversarial"]["f1"] == 1.0
    assert metrics["cost"]["calls"] > 0

    out = tmp_path / cfg.paths.results_dir / cfg.condition
    assert (out / "metrics.json").exists()
    rows = (out / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3
    assert all("recall@5" in json.loads(r)["recall"] for r in rows)


def test_facts_cache_is_condition_independent(offline, tmp_path):
    """B3 and G1 must read byte-identical facts (spec section 6)."""
    cfg, llm, embedder, prompts = offline
    conv = build_conversation()

    from fgl.memory.ingest import FactExtractor

    a = FactExtractor(llm, prompts, cfg.paths.facts_cache).extract_all(conv)
    cfg2 = Config.load("test_offline")
    cfg2.condition = "SOMETHING-ELSE"
    cfg2.paths.facts_cache = cfg.paths.facts_cache
    b = FactExtractor(llm, prompts, cfg2.paths.facts_cache).extract_all(conv)

    assert [f.to_dict() for f in a] == [f.to_dict() for f in b]


def test_curation_and_consolidation_do_not_break_invariants(offline):
    cfg, llm, embedder, prompts = offline
    cfg.curation.curation = True
    cfg.curation.consolidation = True
    cfg.curation.min_face_len = 3
    cfg.curation.min_stable_sessions = 1

    graph, report = Ingestor(cfg, llm, embedder, prompts).ingest(build_conversation())
    graph.check_invariants()
    assert report.n_consolidations >= 1
    level2 = [e for e in graph.edges() if graph.get_edge_attr(e, "level") == 2]
    assert level2, "a level-2 consolidation edge must exist"
    children = graph.get_edge_attr(level2[0], "children")
    assert children
    assert all(graph.get_edge_attr(c, "shadowed") for c in children)


def test_sigma_agent_is_a_no_op_when_there_is_no_choice(offline, tmp_path):
    """A vertex of degree <= 1 admits a single cyclic order: never ask the LLM."""
    from fgl.logging_utils import JsonlLogger

    cfg, llm, embedder, prompts = offline
    cfg.ingest.sigma_policy = "sigma-agent"
    with JsonlLogger(tmp_path / "a.jsonl") as logger:
        graph, _ = Ingestor(cfg, llm, embedder, prompts, logger).ingest(
            build_conversation()
        )
        assert not [r for r in logger.records if r["event"] == "sigma_agent_decision"]
    assert graph.euler().as_tuple() == (3, 3, 2, 0)


def test_sigma_agent_logs_its_reasoning(offline, tmp_path, monkeypatch):
    """With a vertex of degree >= 2 the agent must decide, and log why."""
    from fgl.logging_utils import JsonlLogger

    cfg, llm, embedder, prompts = offline
    cfg.ingest.sigma_policy = "sigma-agent"
    monkeypatch.setitem(
        SCRIPT,
        3,
        SCRIPT[3]
        + [{
            "entity_1": "Caroline", "relation": "thanked",
            "entity_2": "Melanie",
            "fact_text": "Caroline thanked Melanie for the recommendation.",
            "turn_ids": ["D3:2"],
        }],
    )

    log_path = tmp_path / "decisions.jsonl"
    with JsonlLogger(log_path) as logger:
        graph, report = Ingestor(cfg, llm, embedder, prompts, logger).ingest(
            build_conversation()
        )
        decisions = [r for r in logger.records if r["event"] == "sigma_agent_decision"]

    assert report.n_edges == 4
    assert decisions, "sigma-agent must log every insertion decision it makes"
    assert all(d["reason"] for d in decisions)
    assert all(-1 <= d["after_index"] < d["degree"] for d in decisions)
    assert log_path.exists() and log_path.read_text(encoding="utf-8").strip()
    graph.check_invariants()
