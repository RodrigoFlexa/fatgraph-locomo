"""Guards against the failure mode that produced a full but meaningless table.

Symptom seen in the wild: a custom Azure gateway returned empty completions.
Every question fell back to the abstention string, so adversarial scored exactly
1.000 and every other category ~0.01 -- identical across all six conditions.
Numbers existed, and none of them meant anything.

Everything below exists so that can never silently happen again.
"""

from __future__ import annotations

import pytest

from fgl.data.locomo import ABSTAIN_ANSWER
from fgl.evaluation import QAOutcome, sanity_banner, score_question
from fgl.llm import FakeLLM, LLMError
from fgl.llm.client import LLMUnhealthy, Usage


def empty_responder(prompt: str, system: str | None) -> str:
    return ""


def blank_json_responder(prompt: str, system: str | None) -> str:
    return "   \n  "


# --------------------------------------------------------------------------- #
# The client refuses to keep going                                             #
# --------------------------------------------------------------------------- #


def test_first_empty_completion_aborts_immediately(cfg):
    """Fail before spending a whole run's budget on nothing."""
    llm = FakeLLM(cfg.llm, responder=empty_responder)
    with pytest.raises(LLMUnhealthy) as exc:
        llm.complete("anything", purpose="qa/answer")
    msg = str(exc.value)
    assert "VAZIA" in msg
    assert "fgl doctor" in msg, "the error must say how to diagnose"
    assert cfg.llm.deployment in msg, "and which deployment it tried"


def test_empty_responses_are_counted(cfg):
    cfg.llm.fail_on_empty = False
    llm = FakeLLM(cfg.llm, responder=empty_responder)
    for _ in range(4):
        llm.complete("x", purpose="qa/answer")
    u = llm.usage
    assert u.empty_responses == 4
    assert u.healthy is False
    assert any("VAZIAS" in w for w in u.warnings())
    assert u.to_dict()["empty_responses"] == 4
    assert u.by_purpose["qa/answer"]["empty_responses"] == 4


def test_sustained_emptiness_aborts_even_if_the_first_call_was_fine(cfg):
    cfg.llm.health_check_calls = 4
    cfg.llm.max_empty_rate = 0.5
    calls = {"n": 0}

    def flaky(prompt, system):
        calls["n"] += 1
        return "fine" if calls["n"] == 1 else ""

    llm = FakeLLM(cfg.llm, responder=flaky)
    llm.complete("x", purpose="qa/answer")  # ok
    for _ in range(2):
        llm.complete(f"x{calls['n']}", purpose="qa/answer")  # empty, still tolerated
    with pytest.raises(LLMUnhealthy):
        for i in range(5):
            llm.complete(f"y{i}", purpose="qa/answer")


def test_opting_out_is_possible_but_still_recorded(cfg):
    cfg.llm.fail_on_empty = False
    llm = FakeLLM(cfg.llm, responder=empty_responder)
    assert llm.complete("x", purpose="misc") == ""
    assert llm.usage.empty_responses == 1
    assert llm.usage.healthy is False


def test_empty_responses_are_never_cached(cfg, tmp_path):
    """Caching an empty answer would poison every later run."""
    cfg.llm.fail_on_empty = False
    cfg.llm.cache_enabled = True
    cfg.llm.cache_dir = str(tmp_path / "llmcache")
    llm = FakeLLM(cfg.llm, responder=empty_responder)
    llm.cfg.cache_enabled = True
    llm._cache_dir = tmp_path / "llmcache"
    llm._cache_dir.mkdir(parents=True, exist_ok=True)

    llm.complete("same prompt", purpose="misc")
    assert not list(llm._cache_dir.rglob("*.json")), "an empty answer must not be cached"


def test_unparseable_json_is_counted(cfg):
    cfg.llm.fail_on_empty = False
    llm = FakeLLM(cfg.llm, responder=lambda p, s: "desculpe, não posso ajudar")
    out = llm.complete_json("x", purpose="ingest/extract", default={"facts": []})
    assert out == {"facts": []}
    assert llm.usage.json_failures == 1
    assert any("JSON" in w for w in llm.usage.warnings())


# --------------------------------------------------------------------------- #
# The scorer would otherwise make it look like a result                        #
# --------------------------------------------------------------------------- #


def test_constant_abstention_is_the_observed_degenerate_signature():
    """Reproduce the exact numbers the broken run produced."""
    from fgl.data.locomo import Question

    gold = [
        Question("q1", "7 May 2023", 2, []),
        Question("q2", "Melanie", 4, []),
        Question("q3", ABSTAIN_ANSWER, 5, []),
    ]
    scores = [score_question(q, ABSTAIN_ANSWER) for q in gold]
    assert scores[2] == 1.0, "adversarial scores a perfect 1.0 for free"
    assert all(s < 0.2 for s in scores[:2]), "everything else is near zero"


def test_sanity_check_flags_a_run_where_everything_abstained():
    from fgl.config import Config
    from fgl.pipeline import Runner

    runner = Runner.__new__(Runner)
    runner.cfg = Config.load("B1")  # baseline: skips the graph-specific check
    runner.llm = type("L", (), {"usage": Usage()})()

    outcomes = [
        QAOutcome(f"q{i}", 4, "gold", ABSTAIN_ANSWER, 0.01, abstained=True)
        for i in range(30)
    ]
    sanity = runner._sanity(outcomes)

    assert sanity["ok"] is False
    assert sanity["distinct_predictions"] == 1
    assert any("abstenção" in w for w in sanity["warnings"])
    assert any("distinta" in w for w in sanity["warnings"])


def test_sanity_check_passes_a_healthy_run():
    from fgl.config import Config
    from fgl.pipeline import Runner

    runner = Runner.__new__(Runner)
    runner.cfg = Config.load("B1")
    runner.llm = type("L", (), {"usage": Usage()})()

    outcomes = [
        QAOutcome(f"q{i}", 4, "gold", f"resposta {i}", 0.5, n_facts=5) for i in range(30)
    ]
    assert runner._sanity(outcomes)["ok"] is True


def test_report_carries_the_warning_into_the_markdown():
    results = {
        "G1-fatgraph-min": {
            "overall": {"f1_macro": 0.01, "f1_micro": 0.01},
            "per_category": {"single-hop": {"f1": 0.01, "n": 10}},
            "sanity": {"ok": False, "warnings": ["tudo virou abstenção"]},
        }
    }
    banner = sanity_banner(results)
    assert "Corrida suspeita" in banner
    assert "tudo virou abstenção" in banner
    assert "fgl doctor" in banner

    assert sanity_banner({"X": {"sanity": {"ok": True}}}) == ""
