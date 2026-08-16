"""The CLI surface: exit codes, overrides reaching the run, and safe failures."""

from __future__ import annotations

import json

import pytest
from conftest import needs_dataset
from typer.testing import CliRunner

from fgl.cli import app

runner = CliRunner()


def invoke(*args, env=None):
    return runner.invoke(app, list(args), env=env or {})


# --------------------------------------------------------------------------- #
# Smoke                                                                        #
# --------------------------------------------------------------------------- #


def test_version_and_help():
    r = invoke("--version")
    assert r.exit_code == 0 and "fgl" in r.stdout
    assert invoke("--help").exit_code == 0
    for cmd in ("info", "setup", "config", "ingest", "qa", "run", "run-all", "report"):
        assert cmd in invoke("--help").stdout


def test_info_runs_without_credentials(monkeypatch):
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"):
        monkeypatch.delenv(k, raising=False)
    r = invoke("info")
    assert r.exit_code == 0
    assert "conditions" in r.stdout


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #


def test_config_list_shows_every_condition():
    out = invoke("config", "list").stdout
    for cond in ("G1-fatgraph-min", "G2-fatgraph-cur", "B3-rag-facts"):
        assert cond in out


def test_config_show_json_reflects_overrides():
    r = invoke("config", "show", "G1", "--json", "-d",
               "--set", "retrieval.top_m_anchors=9")
    assert r.exit_code == 0
    payload = json.loads(r.stdout[r.stdout.index("{"): r.stdout.rindex("}") + 1])
    assert payload["retrieval"]["top_m_anchors"] == 9


def test_config_show_rejects_a_bad_override():
    r = invoke("config", "show", "G1", "-d", "--set", "nope.key=1")
    assert r.exit_code == 2
    assert "config error" in r.stdout or "config error" in (r.stderr or "")


def test_config_keys_lists_settable_paths():
    out = invoke("config", "keys", "retrieval").stdout
    assert "retrieval.top_m_anchors" in out
    assert "llm.deployment" not in out, "the grep filter must apply"


def test_config_diff_isolates_the_intended_knobs():
    out = invoke("config", "diff", "G2", "G3").stdout
    assert "ingest.sigma_policy" in out
    assert "retrieval.budget_tokens" not in out


def test_config_validate_passes_for_the_shipped_conditions():
    assert invoke("config", "validate").exit_code == 0


# --------------------------------------------------------------------------- #
# Guard rails                                                                  #
# --------------------------------------------------------------------------- #


def test_a_real_run_without_credentials_fails_loudly(monkeypatch):
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("fgl.cli.load_settings", lambda *a, **k: _no_credentials())

    r = invoke("qa", "G1")
    assert r.exit_code == 3, "must refuse before spending a single token"


def _no_credentials():
    from fgl.settings import Settings

    return Settings(dotenv_path=None, dotenv_found=False)


def test_unknown_condition_exits_with_a_config_error():
    assert invoke("qa", "not-a-condition", "-d").exit_code == 2


@pytest.mark.parametrize(
    "override",
    ["retrieval.top_m_anchors=zero", "curation.curation=maybe", "nope.key=1"],
)
def test_bad_overrides_exit_cleanly_with_code_2(override):
    r = invoke("qa", "G1", "-d", "--set", override)
    assert r.exit_code == 2, "a bad --set must not crash with a traceback"


# --------------------------------------------------------------------------- #
# End to end (needs the dataset)                                               #
# --------------------------------------------------------------------------- #


@needs_dataset
def test_dry_run_qa_writes_metrics(tmp_path):
    r = invoke(
        "qa", "G1", "-d", "-n", "1", "-q", "3",
        "--set", f"paths.results_dir={tmp_path / 'res'}",
        "--set", f"paths.graphs_dir={tmp_path / 'graphs'}",
        "--set", f"paths.facts_cache={tmp_path / 'facts'}",
        "--set", f"paths.logs_dir={tmp_path / 'logs'}",
    )
    assert r.exit_code == 0, r.stdout
    metrics = json.loads(
        (tmp_path / "res" / "G1-fatgraph-min" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["overall"]["n"] == 3
    assert metrics["manifest"]["config"]["condition"] == "G1-fatgraph-min"
    assert "azure_api_key" in metrics["manifest"]["environment"]
    assert "sk-" not in json.dumps(metrics["manifest"]), "no secret may reach the manifest"


@needs_dataset
def test_set_wins_over_dry_run(tmp_path):
    """--dry-run must not silently discard an explicit --set."""
    r = invoke(
        "config", "show", "G1", "--json", "-d",
        "--set", f"paths.results_dir={tmp_path / 'explicit'}",
    )
    assert r.exit_code == 0
    payload = json.loads(r.stdout[r.stdout.index("{"): r.stdout.rindex("}") + 1])
    assert payload["paths"]["results_dir"] == str(tmp_path / "explicit")
    assert payload["llm"]["provider"] == "fake", "--dry-run still applies elsewhere"


@needs_dataset
def test_report_rebuilds_tables_from_results(tmp_path):
    res = tmp_path / "res" / "G1-fatgraph-min"
    res.mkdir(parents=True)
    (res / "metrics.json").write_text(
        json.dumps(
            {
                "condition": "G1-fatgraph-min",
                "overall": {"f1_macro": 0.4, "f1_micro": 0.35, "n": 10},
                "per_category": {"multi-hop": {"f1": 0.4, "n": 5, "recall@5": 0.6}},
                "cost": {"calls": 10, "cached_calls": 0, "total_tokens": 100},
                "per_conversation": [],
                "stemmer": "porter",
            }
        ),
        encoding="utf-8",
    )
    r = invoke("report", "-r", str(tmp_path / "res"))
    assert r.exit_code == 0
    assert "G1-fatgraph-min" in r.stdout
    assert (tmp_path / "res" / "report.md").exists()
