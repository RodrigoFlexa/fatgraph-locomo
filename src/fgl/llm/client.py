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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from fgl.config import LLMConfig

# NOTE: openai / httpx are imported lazily inside AzureLLM on purpose.
# Importing them at module scope makes `import fgl.llm` -- and therefore the
# whole test suite and `--dry-run` -- hard-require the Azure stack on machines
# that only want the offline backends.

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
    #: completions that came back empty -- never a legitimate answer, always a
    #: broken backend (bad deployment, content filter, gateway swallowing the body)
    empty_responses: int = 0
    #: JSON completions that failed to parse and fell back to a default
    json_failures: int = 0
    by_purpose: dict = field(default_factory=dict)

    def add(
        self,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool = False,
        empty: bool = False,
    ) -> None:
        self.calls += 1
        if cached:
            self.cached_calls += 1
        if empty:
            self.empty_responses += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        slot = self.by_purpose.setdefault(
            purpose,
            {
                "calls": 0, "cached_calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "empty_responses": 0, "json_failures": 0,
            },
        )
        slot["calls"] += 1
        slot["cached_calls"] += int(cached)
        slot["empty_responses"] += int(empty)
        slot["prompt_tokens"] += prompt_tokens
        slot["completion_tokens"] += completion_tokens

    def add_json_failure(self, purpose: str) -> None:
        self.json_failures += 1
        slot = self.by_purpose.setdefault(
            purpose,
            {
                "calls": 0, "cached_calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "empty_responses": 0, "json_failures": 0,
            },
        )
        slot["json_failures"] = slot.get("json_failures", 0) + 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def healthy(self) -> bool:
        """False when the backend is systematically returning nothing."""
        live = self.calls - self.cached_calls
        return not live or (self.empty_responses / live) < 0.5

    def warnings(self) -> list[str]:
        out = []
        if self.empty_responses:
            out.append(
                f"{self.empty_responses}/{self.calls} respostas do LLM vieram VAZIAS"
            )
        if self.json_failures:
            out.append(
                f"{self.json_failures} respostas JSON não puderam ser parseadas "
                "(usou-se o valor padrão)"
            )
        live = self.calls - self.cached_calls
        if live and self.completion_tokens == 0:
            out.append(
                "o backend não gerou nenhum token de completion — "
                "os resultados não têm significado"
            )
        return out

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "empty_responses": self.empty_responses,
            "json_failures": self.json_failures,
            "healthy": self.healthy,
            "by_purpose": self.by_purpose,
        }


class LLMError(RuntimeError):
    pass


class LLMUnhealthy(LLMError):
    """The backend answers, but with nothing usable. Fail fast, do not pretend."""


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
        #: last raw response, kept for the error message when things go wrong
        self.last_raw: dict = {}

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
        empty = not (text or "").strip()
        self.usage.add(purpose, ptok, ctok, cached=False, empty=empty)
        self.last_raw = {
            "purpose": purpose,
            "text": text,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "prompt_head": prompt[:300],
        }
        if empty:
            self._on_empty(purpose, prompt, ptok, ctok, max_tokens)
        else:
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

    # -- health ------------------------------------------------------------
    def _on_empty(
        self, purpose: str, prompt: str, ptok: int, ctok: int, max_tokens: int
    ) -> None:
        """An empty completion is never a valid answer -- decide whether to abort.

        Silently treating it as an abstention is what turns a broken backend into
        a full, plausible-looking results table where every condition scores the
        same. We refuse to do that: the very first empty response aborts the run,
        and so does a sustained empty rate.
        """
        if not self.cfg.fail_on_empty:
            return
        live = self.usage.calls - self.usage.cached_calls
        first = live <= 1
        sustained = (
            live >= self.cfg.health_check_calls
            and self.usage.empty_responses / live > self.cfg.max_empty_rate
        )
        if not (first or sustained):
            return
        finish = getattr(self, "last_finish_reason", None)
        reasoning = getattr(self, "last_reasoning_tokens", 0)

        if finish == "length" or (reasoning and reasoning >= ctok > 0):
            cause = (
                f"\n[CAUSA MAIS PROVÁVEL] finish_reason={finish!r}"
                + (f", reasoning_tokens={reasoning}" if reasoning else "")
                + f", orçamento pedido={max_tokens}.\n"
                "Este é o comportamento clássico de um modelo de *reasoning* "
                "(o1/o3/o4-mini, família gpt-5): `max_completion_tokens` é um\n"
                "orçamento COMPARTILHADO entre os tokens de raciocínio internos e a "
                "resposta visível. Se ele acaba durante o raciocínio,\n"
                "a API devolve content=\"\" com finish_reason=\"length\".\n"
                f"\nCorreção: aumente o orçamento. Para responder perguntas o padrão "
                f"é {max_tokens} tokens, curto demais para reasoning:\n"
                "    fgl run G1 --set retrieval.answer_max_tokens=3000 "
                "--set llm.max_tokens=8000\n"
                "Ou aponte para um deployment sem reasoning (gpt-4o-mini) em "
                "FGL_LLM_DEPLOYMENT."
            )
        else:
            cause = (
                "\nCausas comuns:\n"
                "  • orçamento de tokens curto demais para um modelo de reasoning\n"
                "    (finish_reason='length' → --set retrieval.answer_max_tokens=3000)\n"
                "  • nome de deployment errado em llm.deployment / FGL_LLM_DEPLOYMENT\n"
                "  • filtro de conteúdo do Azure devolvendo content=None\n"
                "  • gateway/proxy corporativo engolindo o corpo da resposta"
            )

        raise LLMUnhealthy(
            f"O backend devolveu uma resposta VAZIA "
            f"({self.usage.empty_responses}/{live} chamadas até agora), "
            f"propósito={purpose!r}, deployment={self.cfg.deployment!r}, "
            f"prompt_tokens={ptok}, completion_tokens={ctok}, "
            f"finish_reason={finish!r}.\n"
            "\nUma resposta vazia nunca é uma resposta válida. Continuar produziria "
            "uma tabela de resultados sem significado: toda pergunta viraria\n"
            "'Not mentioned in the conversation', o que dá adversarial=1.000 e "
            "~0.01 no resto, igual em todas as condições.\n"
            + cause
            + "\n\nDiagnostique com:  fgl doctor\n"
            "Para tolerar respostas vazias mesmo assim (não recomendado): "
            "--set llm.fail_on_empty=false"
        )

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        purpose: str = "misc",
        max_tokens: int | None = None,
        default: Any = None,
    ) -> Any:
        """:meth:`complete` in JSON mode, tolerant of fenced/notated output.

        A parse failure that falls back to ``default`` is **counted**, so a
        backend that never produces valid JSON shows up in ``metrics.json``
        instead of quietly yielding an empty knowledge graph.
        """
        raw = self.complete(
            prompt, system=system, purpose=purpose, json_mode=True, max_tokens=max_tokens
        )
        try:
            return parse_json_loose(raw)
        except ValueError:
            self.usage.add_json_failure(purpose)
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
    if marker == "answer" or marker == "answer_open":
        return _fake_answer(prompt)
    if marker == "judge":
        return json.dumps(_fake_judge(prompt))
    return "Not mentioned in the conversation"


def _task_marker(prompt: str) -> str:
    m = re.search(r"^#\s*TASK:\s*([a-z_]+)\s*$", prompt, flags=re.MULTILINE)
    return m.group(1) if m else ""


#: closed-class tokens that are never entities.  Short and blunt on purpose:
#: this is a test double, not a tagger.
_FAKE_STOPWORDS = frozenset(
    """
    about above after again against because before being below between both
    could doing during each further having other should still their theirs
    them then there these thing things this those through under until very
    what when where which while with would your yours yeah okay thanks sure
    really thats gonna wanna just like well been have were also into over
    that they from than some more will only even most such here much many
    going know think want said tell told make made take took come came
    """.split()
)


def _fake_entities(text: str) -> list[str]:
    """Entity-ish tokens of one turn, lower-cased and deduplicated.

    Contractions are dropped wholesale: capitalisation cannot separate a proper
    noun from a sentence-initial ``It's``/``That``/``I've`` without a tagger,
    and those were the tokens that ended up as high-degree hubs.
    """
    out = []
    for raw in text.split():
        w = raw.strip(".,!?;:'\"()[]").lower()
        if len(w) <= 3 or "'" in w or not w.isalpha():
            continue
        if w in _FAKE_STOPWORDS:
            continue
        out.append(w)
    return list(dict.fromkeys(out))


def _fake_extract(prompt: str) -> list[dict]:
    """Turn each ``[D<i>:<j>] <speaker>: <text>`` line into one toy triple.

    The pairing is **entity-to-entity**, and the speaker is only one endpoint
    when the turn offers nothing else.  That matters more than it looks: the
    obvious version -- ``(speaker, first_long_word)`` -- makes every fact hang
    off a speaker, and the graph comes out a double star of stopwords, degree 1
    on 77% of its vertices and two faces total.  On that shape sigma is
    *provably* redundant with phi (in a star, phi = sigma o alpha just walks the
    hub's own orbit) and face coverage cannot discriminate, so G4/G5/G6 measure
    zero by construction and `--dry-run` reports a degeneracy that belongs to
    the test double rather than to LoCoMo.  A vocabulary that recurs across
    turns is what gives both endpoints degree > 1 and the surface more than one
    face -- i.e. the only regime in which the offline smoke test says anything.
    """
    turns: list[tuple[str, str, list[str]]] = []
    seen: Counter[str] = Counter()
    for line in prompt.splitlines():
        m = re.match(r"^\[(D\d+:\d+)\]\s*([^:]+):\s*(.+)$", line.strip())
        if not m:
            continue
        turn, speaker, text = m.groups()
        ents = _fake_entities(text)
        turns.append((turn, speaker.strip(), ents))
        seen.update(ents)

    facts: list[dict] = []
    for turn, speaker, ents in turns:
        ents = [e for e in ents if e != speaker.lower()]
        if not ents:
            continue
        # Prefer the tokens that RECUR across the prompt. Picking the first two
        # of each turn would mostly pick hapaxes, and a vertex named once has
        # degree 1 -- the very shape this function exists to avoid. Frequency
        # is the cheapest stand-in for "this is an entity the conversation
        # keeps coming back to", which is what a real extractor yields.
        ents.sort(key=lambda w: (-seen[w], w))
        e1, e2 = (ents[0], ents[1]) if len(ents) > 1 else (speaker, ents[0])
        facts.append(
            {
                "entity_1": e1,
                "relation": "mentions",
                "entity_2": e2,
                "fact_text": f"{e1} and {e2} came up when {speaker} was talking.",
                "turn_ids": [turn],
            }
        )
    return facts[:8]


def _fake_judge(prompt: str) -> dict:
    """Stand-in verdict, strict enough not to look like a measurement.

    The naive version -- any shared token -- accepted "Not mentioned in the
    conversation" against "The week before 27 June 2023" because both contain
    "the", which is the same failure the first `_fake_extract` had: an offline
    number that looks like a result and is an artefact of the stub. So content
    words only, and an abstention is never right unless the reference is one.
    """
    def field(name: str) -> str:
        m = re.search(rf"^{name}:\s*(.+)$", prompt, flags=re.MULTILINE)
        return m.group(1).strip() if m else ""

    gold, pred = field("REFERENCE ANSWER"), field("CANDIDATE ANSWER")
    abstained = "not mentioned" in pred.lower() or "no information" in pred.lower()
    gold_abstains = "not mentioned" in gold.lower() or "no information" in gold.lower()
    if abstained or gold_abstains:
        return {"correct": abstained and gold_abstains, "reason": "fake: abstenção"}

    content = lambda s: {  # noqa: E731
        w.strip(".,!?;:'\"") for w in s.lower().split()
        if len(w) > 3 and w not in _FAKE_STOPWORDS
    }
    g, p = content(gold), content(pred)
    if not g:
        return {"correct": g == p, "reason": "fake: referência sem conteúdo"}
    return {
        "correct": len(g & p) / len(g) >= 0.5,
        "reason": "fake: sobreposição de palavras de conteúdo",
    }


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
        from fgl.llm.azure import AzureLLM  # lazy: keeps openai optional

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
