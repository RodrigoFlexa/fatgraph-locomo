"""fgl — fatgraph memory for LLM agents, benchmarked on LoCoMo.

Public surface::

    from fgl import Config, FatGraph, Runner, load_settings

    settings = load_settings()                  # .env + environment
    cfg = Config.load("G1-fatgraph-min")        # configs/conditions/*.yaml
    metrics = Runner(cfg).run(conversations)

Sub-packages
------------
``fgl.core``        the fatgraph itself: alpha, sigma, phi, faces, Euler
``fgl.memory``      ingestion: extraction, entity resolution, curation
``fgl.retrieval``   embeddings, vector index, face-based retrieval
``fgl.llm``         LLM client (Azure / offline fake) and versioned prompts
``fgl.data``        LoCoMo dataset access
``fgl.evaluation``  the official scorer and the report builders
``fgl.baselines``   B1 full-context, B2 rag-turns, B3 rag-facts
"""

from fgl.config import Config
from fgl.core import FatGraph
from fgl.paths import Paths, project_root
from fgl.settings import Settings, load_settings

__version__ = "2.0.0"

__all__ = [
    "Config",
    "FatGraph",
    "Paths",
    "Settings",
    "load_settings",
    "project_root",
    "__version__",
]


def __getattr__(name: str):
    # Runner pulls in numpy-heavy modules; keep `import fgl` cheap for the CLI.
    if name == "Runner":
        from fgl.pipeline import Runner

        return Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
