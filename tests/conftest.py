from __future__ import annotations

import pytest

from fgl.config import Config
from fgl.core import FatGraph
from fgl.llm import FakeLLM, PromptLibrary
from fgl.paths import Paths, project_root
from fgl.retrieval import HashingEmbedder

ROOT = project_root()
PATHS = Paths.build(ROOT)

#: Skips the tests that need the real dataset (fetch it with `fgl setup`).
needs_dataset = pytest.mark.skipif(
    not PATHS.locomo_file.exists(),
    reason="LoCoMo dataset missing — run `fgl setup`",
)

#: Variables `load_dotenv` writes into `os.environ`. Because that write is a
#: process-wide side effect, a developer's real `.env` would otherwise leak into
#: the suite and make results depend on whether `fgl setup` had been run.
MANAGED_ENV = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "FGL_LLM_DEPLOYMENT",
    "FGL_EMBEDDING_PROVIDER",
    "FGL_EMBEDDING_MODEL",
    "FGL_EMBEDDING_AZURE_DEPLOYMENT",
    # corporate-gateway settings; a real .env pointing at a config.ini used to
    # leak straight into the suite and fail the credential tests
    "FGL_AZURE_CONFIG_INI",
    "FGL_AZURE_CONFIG_SECTION",
    "FGL_CA_BUNDLE",
    "FGL_AZURE_USE_BASE_URL",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path_factory):
    """Every test starts from a clean, credential-free environment.

    Two leaks to plug: ``load_dotenv`` writes into ``os.environ`` process-wide,
    and ``Settings.load()`` with no argument reads the *developer's real*
    ``.env`` at the project root. Without this fixture the suite would pass or
    fail depending on whether the machine happens to have credentials.
    """
    for name in MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)
    # make the default `.env` lookup land somewhere empty
    empty = tmp_path_factory.mktemp("no-project-env")
    monkeypatch.setenv("FGL_PROJECT_ROOT", str(ROOT))
    monkeypatch.setattr(
        "fgl.settings.project_root", lambda: empty, raising=False
    )


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Offline condition with every writable path redirected into tmp_path."""
    c = Config.load("test_offline")
    c.paths.facts_cache = str(tmp_path / "facts")
    c.paths.graphs_dir = str(tmp_path / "graphs")
    c.paths.results_dir = str(tmp_path / "results")
    c.paths.logs_dir = str(tmp_path / "logs")
    c.embeddings.cache_dir = str(tmp_path / "emb")
    c.llm.cache_dir = str(tmp_path / "llm")
    return c


@pytest.fixture
def prompts() -> PromptLibrary:
    return PromptLibrary(PATHS.prompts)


@pytest.fixture
def llm(cfg) -> FakeLLM:
    return FakeLLM(cfg.llm)


@pytest.fixture
def embedder(cfg) -> HashingEmbedder:
    return HashingEmbedder(cfg.embeddings.dim)


# --------------------------------------------------------------------------- #
# Reference graphs                                                             #
# --------------------------------------------------------------------------- #


def build_triangle() -> tuple[FatGraph, dict]:
    """The reference synthetic fatgraph: 3 vertices, 3 edges, 2 faces, g=0."""
    g = FatGraph()
    v = {name: g.add_vertex(name) for name in ("Caroline", "Melanie", "support group")}
    e1 = g.add_edge(
        v["Caroline"], v["Melanie"],
        {"text": "Caroline and Melanie are close friends.",
         "turn_ids": ["D1:1"], "timestamp": "2023-05-08T13:56:00", "session_id": "S1"},
    )
    e2 = g.add_edge(
        v["Melanie"], v["support group"],
        {"text": "Melanie recommended the support group to Caroline.",
         "turn_ids": ["D1:2"], "timestamp": "2023-05-08T13:57:00", "session_id": "S1"},
    )
    e3 = g.add_edge(
        v["support group"], v["Caroline"],
        {"text": "Caroline attended the support group on 7 May 2023.",
         "turn_ids": ["D1:3"], "timestamp": "2023-05-08T13:58:00", "session_id": "S1"},
    )
    return g, {"vertices": v, "edges": (e1, e2, e3)}


def build_torus() -> FatGraph:
    """One vertex, two loops arranged as a b a' b' -> genus 1, one face."""
    g = FatGraph()
    v = g.add_vertex("v")
    g.add_edge(v, v, {"text": "loop a"})                  # sigma = [h1, h2]
    g.add_edge(v, v, {"text": "loop b"}, pos1=1, pos2=3)  # sigma = [h1,h3,h2,h4]
    return g
