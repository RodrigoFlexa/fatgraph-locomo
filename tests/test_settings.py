"""`.env` loading, precedence and redaction."""

from __future__ import annotations

import pytest

from fgl.config import Config
from conftest import PATHS
from fgl.settings import Settings, load_dotenv, parse_dotenv


def test_parse_dotenv_handles_the_usual_shapes():
    parsed = parse_dotenv(
        """
        # a comment
        AZURE_OPENAI_ENDPOINT=https://example.openai.azure.com/
        export AZURE_OPENAI_API_KEY="sk-secret"
        FGL_LLM_DEPLOYMENT = gpt-4o   # trailing comment
        QUOTED='single quoted'
        EMPTY=
        not a variable line
        """
    )
    assert parsed["AZURE_OPENAI_ENDPOINT"] == "https://example.openai.azure.com/"
    assert parsed["AZURE_OPENAI_API_KEY"] == "sk-secret"
    assert parsed["FGL_LLM_DEPLOYMENT"] == "gpt-4o"
    assert parsed["QUOTED"] == "single quoted"
    assert parsed["EMPTY"] == ""
    assert "not" not in parsed


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FGL_LLM_DEPLOYMENT=from-file\n", encoding="utf-8")
    monkeypatch.setenv("FGL_LLM_DEPLOYMENT", "from-environment")

    load_dotenv(env)
    assert Settings.load(env).llm_deployment == "from-environment"

    load_dotenv(env, override=True)
    assert Settings.load(env, override=True).llm_deployment == "from-file"


def test_missing_file_is_not_an_error(tmp_path):
    s = Settings.load(tmp_path / "nope.env")
    assert s.dotenv_found is False
    assert s.azure_ready is False
    assert set(s.missing_azure()) == {
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
    }


def test_require_azure_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    s = Settings.load(tmp_path / "nope.env")
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
        s.require_azure()


# --------------------------------------------------------------------------- #
# Placeholder guard                                                            #
# --------------------------------------------------------------------------- #


def test_untouched_env_example_is_not_mistaken_for_credentials(tmp_path):
    """`fgl setup` copies .env.example verbatim; that must not count as ready."""
    example = PATHS.root / ".env.example"
    env = tmp_path / ".env"
    env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    s = Settings.load(env)
    assert s.dotenv_found is True
    assert s.azure_ready is False, "the template endpoint must not pass as configured"
    assert "AZURE_OPENAI_ENDPOINT" in s.placeholders
    assert "exemplo" in s.explain_missing()
    assert "AZURE_OPENAI_API_KEY" in s.explain_missing()


def test_placeholder_endpoint_is_treated_as_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "a-real-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    s = Settings.load(tmp_path / "nope.env")
    assert s.azure_endpoint is None
    assert s.azure_ready is False
    assert s.placeholders == ("AZURE_OPENAI_ENDPOINT",)


def test_real_values_are_not_flagged_as_placeholders(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://aiims-prod.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "abc123")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    s = Settings.load(tmp_path / "nope.env")
    assert s.azure_ready is True
    assert s.placeholders == ()
    assert s.explain_missing() == "credenciais incompletas"


def test_redacted_never_leaks_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "super-secret-value")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-tenant.openai.azure.com/")
    red = Settings.load(tmp_path / "nope.env").redacted()

    blob = str(red)
    assert "super-secret-value" not in blob
    assert red["azure_api_key"].startswith("<set:")
    assert "my-tenant" not in blob, "the resource name is masked too"
    assert red["azure_endpoint"].endswith("openai.azure.com/")


def test_settings_repr_does_not_show_the_key(monkeypatch):
    s = Settings(azure_api_key="super-secret-value")
    assert "super-secret-value" not in repr(s)


def test_settings_override_the_yaml_but_not_the_cli(monkeypatch):
    monkeypatch.setenv("FGL_LLM_DEPLOYMENT", "gpt-4o-from-env")
    monkeypatch.setenv("FGL_EMBEDDING_PROVIDER", "hashing")
    settings = Settings.load(dotenv_path="/nonexistent")

    cfg = Config.load("G1", settings=settings)
    assert cfg.llm.deployment == "gpt-4o-from-env"
    assert cfg.embeddings.provider == "hashing"

    cfg = Config.load("G1", overrides=["llm.deployment=gpt-5-from-cli"], settings=settings)
    assert cfg.llm.deployment == "gpt-5-from-cli", "--set must win over .env"


def test_unset_settings_leave_the_yaml_alone(monkeypatch):
    for k in ("FGL_LLM_DEPLOYMENT", "FGL_EMBEDDING_PROVIDER", "FGL_EMBEDDING_MODEL"):
        monkeypatch.delenv(k, raising=False)
    cfg = Config.load("G1", settings=Settings.load(dotenv_path="/nonexistent"))
    assert cfg.llm.deployment == "gpt-4o-mini"
