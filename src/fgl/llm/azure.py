"""Azure OpenAI backend, including corporate-gateway and reasoning-model support.

Three things make a real deployment differ from the textbook example, and all
three are handled here rather than by hand-patching the client:

**Corporate gateways.** The endpoint may be a full ``base_url`` instead of an
``azure_endpoint`` and TLS may need a private CA bundle. Both come from
:class:`fgl.settings.Settings` (a single ``.env``) through
:func:`build_azure_client`, which the chat backend and the embedding backend
share so they cannot drift apart.

**Reasoning models** (``o1``/``o3``/``o4-mini``, the ``gpt-5`` family) take
``max_completion_tokens`` instead of ``max_tokens``, reject a custom
``temperature``, and -- critically -- spend that token budget on *internal
reasoning before* emitting anything. A budget sized for a short extractive
answer (64 tokens) is consumed entirely by reasoning and the API returns
``content=""`` with ``finish_reason="length"``. We detect the model family and
apply a floor; ``reasoning_min_tokens: 0`` omits the cap altogether, which is
what a working notebook against such a deployment usually does.

**Parameter drift.** Gateways and model families disagree about which optional
parameters exist. Rather than fail the run, an unsupported-parameter error is
parsed, the offending key dropped, and the call retried -- once per key.
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Any

from fgl.config import LLMConfig
from fgl.llm.client import LLMClient, LLMError

#: Deployment-name fragments that indicate a reasoning model. Matched on the
#: *deployment* name, which is why a suffix like `-petrobras` is harmless.
REASONING_MARKERS = ("gpt-5", "gpt5", "o1-", "o3-", "o4-", "-o1", "-o3", "-o4")

#: Parameters we send optimistically and drop if the endpoint rejects them.
OPTIONAL_PARAMS = (
    "seed", "temperature", "frequency_penalty", "presence_penalty",
    "response_format", "max_tokens", "max_completion_tokens", "reasoning_effort",
)


def is_reasoning_deployment(name: str) -> bool:
    n = (name or "").lower()
    return any(m in n for m in REASONING_MARKERS)


def http_client_for(ca_bundle: str | None):
    """An ``httpx.Client`` pinned to a private CA, or ``None`` for the default."""
    if not ca_bundle:
        return None
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise LLMError("FGL_CA_BUNDLE exige httpx: pip install httpx") from exc
    return httpx.Client(verify=ca_bundle)


def build_azure_client(settings=None):
    """Build one ``AzureOpenAI`` client from the environment. Shared on purpose.

    The chat backend and the embedding backend must agree about all three
    things a corporate gateway changes -- the private CA bundle, whether the
    endpoint is a bare ``azure_endpoint`` or an already-routed ``base_url``,
    and which credentials are in play. They used to disagree: the embedder read
    ``os.environ`` directly and therefore ignored ``FGL_CA_BUNDLE`` and the
    gateway URL entirely, so ``provider: azure`` could only ever have worked
    against a plain Azure resource. One constructor removes the whole class of
    bug.

    Returns ``(client, settings)``.
    """
    try:
        from openai import AzureOpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMError(
            "o pacote 'openai' (>=1.x) é necessário para o backend azure: pip install openai"
        ) from exc

    from fgl.settings import load_settings

    settings = settings or load_settings()
    settings.require_azure()

    kwargs: dict[str, Any] = dict(
        api_key=settings.azure_api_key,
        api_version=settings.azure_api_version,
    )
    http_client = http_client_for(settings.ca_bundle)
    if http_client is not None:
        kwargs["http_client"] = http_client

    # A gateway URL already contains the routing path, so it must be passed as
    # base_url; a bare resource host is an azure_endpoint. The SDK appends
    # `/deployments/<model>/<verb>` either way, which is why the same client
    # serves /chat/completions and /embeddings with no special casing.
    endpoint = settings.azure_endpoint or ""
    if settings.use_base_url:
        kwargs["base_url"] = endpoint
    else:
        kwargs["azure_endpoint"] = endpoint
    return AzureOpenAI(**kwargs), settings


class AzureLLM(LLMClient):
    """Azure OpenAI chat completions with backoff and parameter negotiation."""

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self._client, _settings = build_azure_client()

        self.reasoning = (
            is_reasoning_deployment(cfg.deployment)
            if cfg.api_style == "auto"
            else cfg.api_style == "reasoning"
        )
        #: parameters this endpoint has already rejected once
        self._unsupported: set[str] = set()
        self.last_finish_reason: str | None = None
        self.last_reasoning_tokens: int = 0

    # ------------------------------------------------------------------ io --
    #: kept as an alias so existing callers/tests keep working
    _http_client = staticmethod(http_client_for)

    # -------------------------------------------------------------- params --
    def _budget(self, max_tokens: int) -> int:
        """Token cap to request. Reasoning models need room to think first."""
        if not self.reasoning:
            return max_tokens
        floor = self.cfg.reasoning_min_tokens
        if floor <= 0:
            return 0  # omit the cap entirely, like the working notebook does
        return max(max_tokens, floor)

    def _build_kwargs(self, messages, json_mode, max_tokens, temperature) -> dict:
        kwargs: dict[str, Any] = {"model": self.cfg.deployment, "messages": messages}

        budget = self._budget(max_tokens)
        if budget > 0:
            key = "max_completion_tokens" if self.reasoning else "max_tokens"
            kwargs[key] = budget

        # Reasoning deployments only accept the default temperature.
        if not self.reasoning and self.cfg.send_temperature:
            kwargs["temperature"] = temperature
        if self.cfg.seed is not None and not self.reasoning:
            kwargs["seed"] = self.cfg.seed
        if self.reasoning and self.cfg.reasoning_effort:
            kwargs["reasoning_effort"] = self.cfg.reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        return {k: v for k, v in kwargs.items() if k not in self._unsupported}

    # ---------------------------------------------------------------- call --
    def _call(self, prompt, system, json_mode, max_tokens, temperature):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            kwargs = self._build_kwargs(messages, json_mode, max_tokens, temperature)
            try:
                resp = self._client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                usage = getattr(resp, "usage", None)
                self.last_finish_reason = getattr(choice, "finish_reason", None)
                details = getattr(usage, "completion_tokens_details", None)
                self.last_reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                return (
                    text,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                )
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                dropped = self._maybe_drop_parameter(exc, kwargs)
                if dropped:
                    continue  # retry immediately without the rejected parameter
                if not _is_retryable(exc) or attempt == self.cfg.max_retries - 1:
                    break
                delay = min(self.cfg.backoff_max, self.cfg.backoff_base**attempt) * (
                    0.5 + random.random()
                )
                time.sleep(max(delay, _retry_after_seconds(exc)))

        raise LLMError(
            f"chamada ao Azure falhou (deployment={self.cfg.deployment!r}, "
            f"reasoning={self.reasoning}): {last_exc}"
        ) from last_exc

    def _maybe_drop_parameter(self, exc: Exception, kwargs: dict) -> bool:
        """Learn which optional parameters this endpoint rejects, and stop sending them."""
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status not in (400, 422):
            return False
        message = str(exc)
        for name in OPTIONAL_PARAMS:
            if name in self._unsupported or name not in kwargs:
                continue
            if re.search(rf"\b{re.escape(name)}\b", message):
                self._unsupported.add(name)
                return True
        return False


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (408, 409, 429, 500, 502, 503, 504):
        return True
    name = type(exc).__name__
    return any(
        tok in name
        for tok in (
            "RateLimit", "Timeout", "APIConnection", "InternalServer",
            "ServiceUnavailable",
        )
    )


def _retry_after_seconds(exc: Exception) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after-ms", "Retry-After-Ms"):
        if key in headers:
            try:
                return float(headers[key]) / 1000.0
            except (TypeError, ValueError):
                pass
    for key in ("retry-after", "Retry-After"):
        if key in headers:
            try:
                return float(headers[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def resolve_ca_bundle(name: str | None) -> str | None:
    """Find a CA bundle without depending on the current working directory."""
    if not name:
        return None
    from fgl.paths import project_root

    p = Path(name).expanduser()
    if p.is_absolute():
        return str(p) if p.exists() else None
    for base in (project_root(), Path.cwd(), Path(__file__).resolve().parent):
        candidate = (base / p).resolve()
        if candidate.exists():
            return str(candidate)
    return None
