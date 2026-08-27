"""Corporate-gateway and reasoning-model support.

Modelled on a real deployment: a private CA bundle, a ``base_url`` that already
carries the routing path, and a ``gpt-5-mini`` deployment -- a *reasoning*
model, which takes ``max_completion_tokens``, rejects a custom ``temperature``,
and returns an empty string when the budget is spent on internal reasoning.

``.env`` is the only configuration file: the ``FGL_AZURE_CONFIG_INI`` path was
removed, so the tests below pin that credentials come from the environment and
from nowhere else.
"""

from __future__ import annotations

import pytest

from fgl.config import Config, ConfigError
from fgl.llm.azure import is_reasoning_deployment
from fgl.settings import Settings


# --------------------------------------------------------------------------- #
# Detecting the model family                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "deployment,expected",
    [
        ("gpt-5-mini-petrobras", True),
        ("gpt-5", True),
        ("o1-preview", True),
        ("o3-mini", True),
        ("o4-mini-prod", True),
        ("gpt-4o-petrobras", False),
        ("gpt-4o-mini", False),
        ("gpt-35-turbo", False),
        ("", False),
    ],
)
def test_reasoning_deployments_are_recognised(deployment, expected):
    assert is_reasoning_deployment(deployment) is expected


# --------------------------------------------------------------------------- #
# One source of truth: .env / the environment                                  #
# --------------------------------------------------------------------------- #


def test_credentials_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "abc-123-secret")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT", "https://gateway.corp.example.com/openai/v1"
    )
    s = Settings.load(tmp_path / "no.env")
    assert s.azure_ready is True
    assert s.azure_api_key == "abc-123-secret"
    assert s.use_base_url is True


def test_credentials_come_from_a_dotenv_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "AZURE_OPENAI_API_KEY=from-dotenv\n"
        "AZURE_OPENAI_API_VERSION=2024-10-21\n"
        "AZURE_OPENAI_ENDPOINT=https://gateway.corp.example.com/openai/v1\n",
        encoding="utf-8",
    )
    s = Settings.load(env)
    assert s.dotenv_found is True
    assert s.azure_api_key == "from-dotenv"


def test_there_is_no_second_config_path(tmp_path, monkeypatch):
    """The .ini loader is gone: setting it must not resurrect credentials.

    Regression guard for the thing the removal was for -- two places to look
    meant "where did this key come from?" had two possible answers.
    """
    ini = tmp_path / "config-v1.x.ini"
    ini.write_text(
        "[OPENAI]\nOPENAI_API_KEY = from-ini\nOPENAI_API_VERSION = v\n"
        "AZURE_OPENAI_BASE_URL = https://gateway.corp.example.com/openai/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FGL_AZURE_CONFIG_INI", str(ini))

    s = Settings.load(tmp_path / "no.env")
    assert s.azure_ready is False
    assert not hasattr(s, "ini_path")
    assert "from-ini" not in str(s.redacted())


def test_the_key_never_reaches_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "abc-123-secret")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "v")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com/")

    red = Settings.load(tmp_path / "no.env").redacted()
    assert "abc-123-secret" not in str(red)
    assert red["azure_api_key"].startswith("<set:")


# --------------------------------------------------------------------------- #
# base_url vs azure_endpoint                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "endpoint,use_base_url",
    [
        ("https://gateway.corp.example.com/openai/v1", True),
        ("https://gateway.corp.example.com/some/path", True),
        ("https://my-resource.openai.azure.com/", False),
        ("https://my-resource.openai.azure.com", False),
    ],
)
def test_endpoint_shape_decides_base_url(tmp_path, monkeypatch, endpoint, use_base_url):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "v")
    assert Settings.load(tmp_path / "no.env").use_base_url is use_base_url


def test_base_url_can_be_forced(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "v")
    monkeypatch.setenv("FGL_AZURE_USE_BASE_URL", "true")
    assert Settings.load(tmp_path / "no.env").use_base_url is True


# --------------------------------------------------------------------------- #
# CA bundle                                                                    #
# --------------------------------------------------------------------------- #


def test_ca_bundle_is_resolved_relative_to_the_project(tmp_path, monkeypatch):
    from fgl.paths import project_root

    pem = project_root() / "test-ca.pem"
    pem.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    try:
        monkeypatch.setenv("FGL_CA_BUNDLE", "test-ca.pem")
        s = Settings.load(tmp_path / "no.env")
        assert s.ca_bundle == str(pem), "must not depend on the current directory"
    finally:
        pem.unlink()


def test_a_missing_ca_bundle_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("FGL_CA_BUNDLE", "definitely-not-here.pem")
    with pytest.raises(RuntimeError, match="FGL_CA_BUNDLE"):
        Settings.load(tmp_path / "no.env")


# --------------------------------------------------------------------------- #
# Request shape                                                                #
# --------------------------------------------------------------------------- #


class _FakeAzure:
    """Records the kwargs the client sends, and replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen: list[dict] = []
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        self.seen.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _response(content="ok", finish="stop", reasoning=0):
    msg = type("M", (), {"content": content})()
    choice = type("Ch", (), {"message": msg, "finish_reason": finish})()
    details = type("D", (), {"reasoning_tokens": reasoning})()
    usage = type("U", (), {
        "prompt_tokens": 10, "completion_tokens": 5,
        "completion_tokens_details": details,
    })()
    return type("R", (), {"choices": [choice], "usage": usage})()


def _client(monkeypatch, deployment, responses, **overrides):
    from fgl.llm import azure as azmod

    cfg = Config.load("G1").llm
    cfg.deployment = deployment
    cfg.cache_enabled = False
    for k, v in overrides.items():
        setattr(cfg, k, v)

    client = azmod.AzureLLM.__new__(azmod.AzureLLM)
    from fgl.llm.client import LLMClient

    LLMClient.__init__(client, cfg)
    client._client = _FakeAzure(responses)
    client.reasoning = (
        azmod.is_reasoning_deployment(deployment)
        if cfg.api_style == "auto"
        else cfg.api_style == "reasoning"
    )
    client._unsupported = set()
    client.last_finish_reason = None
    client.last_reasoning_tokens = 0
    return client


def test_reasoning_model_gets_max_completion_tokens_and_no_temperature(monkeypatch):
    c = _client(monkeypatch, "gpt-5-mini-petrobras", [_response()])
    c.complete("q", purpose="qa/answer", max_tokens=64)
    sent = c._client.seen[0]

    assert "max_completion_tokens" in sent, "reasoning models use the new parameter"
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] >= 4000, "64 would be eaten by the reasoning"
    assert "temperature" not in sent, "reasoning models only accept the default"
    assert "seed" not in sent


def test_chat_model_keeps_the_classic_parameters(monkeypatch):
    c = _client(monkeypatch, "gpt-4o-petrobras", [_response()])
    c.complete("q", purpose="qa/answer", max_tokens=64)
    sent = c._client.seen[0]

    assert sent["max_tokens"] == 64
    assert "max_completion_tokens" not in sent
    assert "temperature" in sent
    assert "seed" in sent


def test_reasoning_min_tokens_zero_omits_the_cap(monkeypatch):
    """Matches a working notebook that simply does not pass a token cap."""
    c = _client(monkeypatch, "gpt-5-mini-petrobras", [_response()], reasoning_min_tokens=0)
    c.complete("q", purpose="qa/answer", max_tokens=64)
    sent = c._client.seen[0]

    assert "max_completion_tokens" not in sent
    assert "max_tokens" not in sent


def test_api_style_can_be_forced(monkeypatch):
    c = _client(monkeypatch, "some-custom-name", [_response()], api_style="reasoning")
    c.complete("q", purpose="misc", max_tokens=64)
    assert "max_completion_tokens" in c._client.seen[0]


def test_unsupported_parameters_are_dropped_and_the_call_retried(monkeypatch):
    """A gateway that rejects `seed` must not fail the whole run."""

    class BadRequest(Exception):
        status_code = 400

        def __str__(self):
            return "Unrecognized request argument supplied: seed"

    c = _client(monkeypatch, "gpt-4o-petrobras", [BadRequest(), _response("funcionou")])
    text = c.complete("q", purpose="misc", max_tokens=64)

    assert text == "funcionou"
    assert "seed" in c._client.seen[0]
    assert "seed" not in c._client.seen[1], "the rejected parameter is dropped"
    assert "seed" in c._unsupported, "and remembered, so it is not sent again"


def test_finish_reason_length_is_reported_in_the_empty_error(monkeypatch):
    from fgl.llm.client import LLMUnhealthy

    c = _client(
        monkeypatch, "gpt-5-mini-petrobras",
        [_response(content="", finish="length", reasoning=64)],
    )
    with pytest.raises(LLMUnhealthy) as exc:
        c.complete("q", purpose="qa/answer", max_tokens=64)

    msg = str(exc.value)
    assert "finish_reason='length'" in msg
    assert "reasoning" in msg.lower()
    assert "answer_max_tokens" in msg, "must name the knob that fixes it"


def test_config_rejects_a_bogus_api_style():
    with pytest.raises(ConfigError, match="api_style"):
        Config.load("G1", overrides=["llm.api_style=magic"])
