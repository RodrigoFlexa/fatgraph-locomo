"""LLM abstraction layer.

A single interface -- :meth:`LLMClient.complete` -- so the deployment can be
swapped for a bigger model later without touching any other module.

Backends
--------
``AzureLLM``   Azure OpenAI via the official ``openai`` SDK (>=1.x).  Credentials
               strictly from the environment; the *deployment* name comes from
               the YAML config.  Exponential backoff on 429/timeout/5xx.
``FakeLLM``    Deterministic, offline, scripted -- used by the test suite and by
               ``--dry-run`` so the whole pipeline is exercisable without spend.

Every call goes through an aggressive on-disk cache keyed by
``sha256(model | temperature | max_tokens | seed | system | prompt | json_mode)``
which makes runs reproducible and cheap to resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from fgl.config import LLMConfig
import base64
from openai import AzureOpenAI
from configparser import ConfigParser, ExtendedInterpolation
import httpx
import numpy as np

# --------------------------------------------------------------------------- #
# Usage accounting                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class Usage:
    """Token/latency accounting, split by pipeline phase."""

    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_purpose: dict = field(default_factory=dict)

    def add(
        self,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool = False,
    ) -> None:
        self.calls += 1
        if cached:
            self.cached_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        slot = self.by_purpose.setdefault(
            purpose,
            {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        )
        slot["calls"] += 1
        slot["cached_calls"] += int(cached)
        slot["prompt_tokens"] += prompt_tokens
        slot["completion_tokens"] += completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "by_purpose": self.by_purpose,
        }


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Base client                                                                  #
# --------------------------------------------------------------------------- #


class LLMClient:
    """Interface every backend implements."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.usage = Usage()
        self._cache_dir = Path(cfg.cache_dir)
        if cfg.cache_enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public ------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        purpose: str = "misc",
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the model's text completion (cached)."""
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens
        key = self._cache_key(prompt, system, json_mode, max_tokens, temperature)

        cached = self._cache_read(key)
        if cached is not None:
            self.usage.add(
                purpose, cached.get("prompt_tokens", 0), cached.get("completion_tokens", 0),
                cached=True,
            )
            return cached["text"]

        text, ptok, ctok = self._call(
            prompt, system, json_mode, max_tokens, temperature
        )
        self.usage.add(purpose, ptok, ctok, cached=False)
        self._cache_write(
            key,
            {
                "text": text,
                "prompt_tokens": ptok,
                "completion_tokens": ctok,
                "purpose": purpose,
                "model": self.cfg.deployment,
            },
        )
        return text

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        purpose: str = "misc",
        max_tokens: int | None = None,
        default: Any = None,
    ) -> Any:
        """:meth:`complete` in JSON mode, tolerant of fenced/notated output."""
        raw = self.complete(
            prompt, system=system, purpose=purpose, json_mode=True, max_tokens=max_tokens
        )
        try:
            return parse_json_loose(raw)
        except ValueError:
            if default is not None:
                return default
            raise

    # -- to implement ------------------------------------------------------
    def _call(
        self,
        prompt: str,
        system: str | None,
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        raise NotImplementedError

    # -- cache -------------------------------------------------------------
    def _cache_key(
        self,
        prompt: str,
        system: str | None,
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = "\x1f".join(
            [
                self.cfg.provider,
                self.cfg.deployment,
                f"{temperature:.4f}",
                str(max_tokens),
                str(self.cfg.seed),
                str(bool(json_mode)),
                system or "",
                prompt,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / key[:2] / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[dict]:
        if not self.cfg.cache_enabled:
            return None
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_write(self, key: str, payload: dict) -> None:
        if not self.cfg.cache_enabled:
            return
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)


# --------------------------------------------------------------------------- #
# Azure OpenAI                                                                 #
# --------------------------------------------------------------------------- #


class AzureLLM(LLMClient):
    """Azure OpenAI chat completions with exponential backoff."""

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "the 'openai' package (>=1.x) is required for the azure backend"
            ) from exc

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        missing = [
            name
            for name, val in [
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_API_KEY", api_key),
                ("AZURE_OPENAI_API_VERSION", api_version),
            ]
            if not val
        ]
        if missing:
            raise LLMError(f"missing environment variables: {', '.join(missing)}")


        http_client = httpx.Client(verify='petrobras-ca-root.pem')
        self._client = AzureOpenAI(
            base_url=endpoint,
            api_key=api_key,
            api_version=api_version,
            http_client=http_client

        )

    def _call(self, prompt, system, json_mode, max_tokens, temperature):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = dict(
            model=self.cfg.deployment,
            messages=messages,
            # temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        if self.cfg.seed is not None:
            kwargs["seed"] = self.cfg.seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                return (
                    text,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                )
            except Exception as exc:  # noqa: BLE001 - we classify below
                last_exc = exc
                if not _is_retryable(exc) or attempt == self.cfg.max_retries - 1:
                    break
                delay = min(
                    self.cfg.backoff_max, self.cfg.backoff_base ** attempt
                ) * (0.5 + random.random())
                retry_after = _retry_after_seconds(exc)
                time.sleep(max(delay, retry_after))
        raise LLMError(f"Azure OpenAI call failed: {last_exc}") from last_exc


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (408, 409, 429, 500, 502, 503, 504):
        return True
    name = type(exc).__name__
    return any(
        tok in name
        for tok in ("RateLimit", "Timeout", "APIConnection", "InternalServer", "ServiceUnavailable")
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


# --------------------------------------------------------------------------- #
# Fake backend (tests / dry runs)                                              #
# --------------------------------------------------------------------------- #


class FakeLLM(LLMClient):
    """Deterministic offline backend.

    ``responder`` receives ``(prompt, system, purpose_hint)`` and returns the
    completion.  The default responder covers every prompt this repository
    issues with sensible, hand-written behaviour, so ``pytest`` and
    ``--dry-run`` exercise the real code paths end to end.
    """

    def __init__(
        self,
        cfg: LLMConfig,
        responder: Callable[[str, Optional[str]], str] | None = None,
    ) -> None:
        cfg = LLMConfig(**{**cfg.__dict__, "cache_enabled": False})
        super().__init__(cfg)
        self.responder = responder or default_fake_responder
        self.prompts: list[tuple[str, str]] = []  # (purpose, prompt) audit trail

    def complete(self, prompt, *, system=None, purpose="misc", **kw):  # type: ignore[override]
        self.prompts.append((purpose, prompt))
        return super().complete(prompt, system=system, purpose=purpose, **kw)

    def _call(self, prompt, system, json_mode, max_tokens, temperature):
        text = self.responder(prompt, system)
        return text, len(prompt) // 4, len(text) // 4


def default_fake_responder(prompt: str, system: str | None) -> str:
    """Rule-based stand-in keyed on the task marker each prompt carries."""
    marker = _task_marker(prompt)
    if marker == "extract_facts":
        return json.dumps({"facts": _fake_extract(prompt)})
    if marker == "entity_match":
        return json.dumps({"same": False, "reason": "fake backend defaults to 'new entity'"})
    if marker == "sigma_agent":
        return json.dumps({"after_index": 0, "reason": "fake backend: front of the trail"})
    if marker == "incongruence":
        return json.dumps({"contradiction": False, "reason": "fake backend: no contradiction"})
    if marker == "redundancy":
        return json.dumps(
            {"redundant": False, "merged_text": "", "reason": "fake backend: keep both"}
        )
    if marker == "consolidate":
        return json.dumps({"summary": "fake consolidation of the face"})
    if marker == "answer":
        return _fake_answer(prompt)
    return "Not mentioned in the conversation"


def _task_marker(prompt: str) -> str:
    m = re.search(r"^#\s*TASK:\s*([a-z_]+)\s*$", prompt, flags=re.MULTILINE)
    return m.group(1) if m else ""


def _fake_extract(prompt: str) -> list[dict]:
    """Turn each ``D<i>:<j> <speaker>: <text>`` line into one toy triple."""
    facts = []
    for line in prompt.splitlines():
        m = re.match(r"^\[(D\d+:\d+)\]\s*([^:]+):\s*(.+)$", line.strip())
        if not m:
            continue
        turn, speaker, text = m.groups()
        words = [w.strip(".,!?'\"") for w in text.split() if len(w) > 4]
        if not words:
            continue
        facts.append(
            {
                "entity_1": speaker.strip(),
                "relation": "mentions",
                "entity_2": words[0],
                "fact_text": f"{speaker.strip()} mentioned {words[0]}.",
                "turn_ids": [turn],
            }
        )
    return facts[:8]


def _fake_answer(prompt: str) -> str:
    m = re.search(r"^QUESTION:\s*(.+)$", prompt, flags=re.MULTILINE)
    question = m.group(1) if m else ""
    for line in prompt.splitlines():
        if line.startswith("[") and "]" in line:
            body = line.split("]", 1)[1].strip()
            overlap = set(body.lower().split()) & set(question.lower().split())
            if len(overlap) >= 2:
                return body[:80]
    return "Not mentioned in the conversation"


# --------------------------------------------------------------------------- #
# Factory + JSON helpers                                                       #
# --------------------------------------------------------------------------- #


def build_llm(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "azure":
        return AzureLLM(cfg)
    if cfg.provider == "fake":
        return FakeLLM(cfg)
    raise LLMError(f"unknown llm provider {cfg.provider!r}")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_loose(text: str) -> Any:
    """Parse JSON that may arrive fenced, prefixed or with trailing prose."""
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from model output: {text[:300]!r}")
