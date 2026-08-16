"""Project layout resolution.

Every path in the framework is resolved against the *project root* — the
directory holding ``pyproject.toml`` — so the CLI, the test suite and the
notebooks all agree on where things live regardless of the current directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MARKERS = ("pyproject.toml", ".git")


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Walk up from this file until a project marker is found.

    ``FGL_PROJECT_ROOT`` overrides the search, which is what lets the notebooks
    and scheduled jobs run from anywhere.
    """
    override = os.environ.get("FGL_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if any((parent / m).exists() for m in MARKERS):
            return parent
    return here.parents[2]


@dataclass(frozen=True)
class Paths:
    """Resolved directories for one run."""

    root: Path
    configs: Path
    conditions: Path
    prompts: Path
    data: Path
    external: Path
    locomo_repo: Path
    locomo_file: Path
    artifacts: Path
    graphs: Path
    facts: Path
    logs: Path
    cache: Path
    llm_cache: Path
    embedding_cache: Path
    results: Path
    notebooks: Path

    @classmethod
    def build(cls, root: Path | str | None = None) -> "Paths":
        r = Path(root).resolve() if root else project_root()
        data = r / "data"
        external = data / "external"
        locomo_repo = external / "locomo"
        artifacts = r / "artifacts"
        cache = r / ".cache"
        return cls(
            root=r,
            configs=r / "configs",
            conditions=r / "configs" / "conditions",
            prompts=r / "prompts",
            data=data,
            external=external,
            locomo_repo=locomo_repo,
            locomo_file=locomo_repo / "data" / "locomo10.json",
            artifacts=artifacts,
            graphs=artifacts / "graphs",
            facts=artifacts / "facts",
            logs=artifacts / "logs",
            cache=cache,
            llm_cache=cache / "llm",
            embedding_cache=cache / "embeddings",
            results=r / "results",
            notebooks=r / "notebooks",
        )

    def ensure(self) -> "Paths":
        for p in (
            self.data, self.external, self.artifacts, self.graphs, self.facts,
            self.logs, self.cache, self.llm_cache, self.embedding_cache, self.results,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def resolve(self, value: str | Path) -> Path:
        """Interpret a config path: absolute stays, relative hangs off the root."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def to_dict(self) -> dict:
        return {f: str(getattr(self, f)) for f in self.__dataclass_fields__}
