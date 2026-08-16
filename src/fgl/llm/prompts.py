"""Versioned prompt loading.

Prompts live in ``prompts/*.txt``, one file per LLM task, and are formatted with
``str.format``.  Literal braces in a template must be doubled (``{{`` / ``}}``)
-- every template in this repository already is.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path


class PromptLibrary:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        if not self.dir.exists():
            raise FileNotFoundError(f"prompts directory not found: {self.dir}")

    @functools.lru_cache(maxsize=None)
    def _raw(self, name: str) -> str:
        path = self.dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"prompt {name!r} not found at {path}")
        return path.read_text(encoding="utf-8")

    def render(self, name: str, **kwargs) -> str:
        try:
            return self._raw(name).format(**kwargs)
        except KeyError as exc:
            raise KeyError(f"prompt {name!r} is missing placeholder {exc}") from exc

    def version(self, name: str) -> str:
        """Content hash -- goes into the results manifest for traceability."""
        return hashlib.sha1(self._raw(name).encode("utf-8")).hexdigest()[:12]

    def manifest(self) -> dict[str, str]:
        return {p.stem: self.version(p.stem) for p in sorted(self.dir.glob("*.txt"))}


SYSTEM_EXTRACTOR = (
    "You are a precise information-extraction engine. You only output valid JSON."
)
SYSTEM_JUDGE = (
    "You are a careful analyst. You answer only with valid JSON, never with prose."
)
SYSTEM_ANSWERER = (
    "You answer questions about a conversation using only the memories you are "
    "given. You reply with a short extractive phrase and nothing else."
)
