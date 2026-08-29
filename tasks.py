"""Cross-platform dev tasks — the Windows-safe equivalent of ``make <target>``.

The Makefile depends on ``make``, ``grep``, ``awk``, ``rm`` and ``find``, none
of which ship with a stock Windows install. This script uses only the stdlib
and calls every subprocess in list form (never ``shell=True``), so it behaves
identically under PowerShell and under bash. On Linux/macOS the Makefile is a
thin wrapper that forwards to this same file, so there is one place that
knows what each target does.

Usage::

    python tasks.py test
    python tasks.py smoke
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(*cmd: str) -> None:
    print(f"+ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _py(*args: str) -> None:
    _run(sys.executable, *args)


def t_install() -> None:
    _py("-m", "pip", "install", "-e", ".[all]")


def t_setup() -> None:
    _run("fgl", "setup")


def t_test() -> None:
    _py("-m", "pytest", "tests/")


def t_lint() -> None:
    _py("-m", "ruff", "check", "src", "tests")
    _py("-m", "ruff", "format", "--check", "src", "tests")


def t_smoke() -> None:
    _run(
        "fgl", "run-all", "--dry-run",
        "--limit-conversations", "1", "--limit-questions", "10",
        "--continue-on-error",
    )


def t_ingest() -> None:
    _run("fgl", "ingest", "G1")


def t_qa() -> None:
    _run("fgl", "qa", "G1")


def t_all() -> None:
    _run("fgl", "run-all")


def t_report() -> None:
    _run("fgl", "report")


def t_notebooks() -> None:
    _py("-m", "jupyter", "lab", "notebooks/")


def _rmtree(*relative_paths: str) -> None:
    for rel in relative_paths:
        p = ROOT / rel
        if p.exists():
            print(f"removing {p}")
            shutil.rmtree(p, ignore_errors=True)


def t_clean_dry() -> None:
    _rmtree(
        "results-dry", "artifacts/graphs-dry", "artifacts/logs-dry",
        "artifacts/facts-dry", ".cache/embeddings-dry",
    )


def t_clean() -> None:
    t_clean_dry()
    _rmtree("artifacts", ".cache", ".pytest_cache", ".ruff_cache")
    for d in ROOT.rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)


TASKS = {
    "install": t_install,
    "setup": t_setup,
    "test": t_test,
    "lint": t_lint,
    "smoke": t_smoke,
    "ingest": t_ingest,
    "qa": t_qa,
    "all": t_all,
    "report": t_report,
    "notebooks": t_notebooks,
    "clean-dry": t_clean_dry,
    "clean": t_clean,
}


def t_help() -> None:
    print("usage: python tasks.py <target>\n\ntargets:")
    for name in TASKS:
        print(f"  {name}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "-h", "--help"):
        t_help()
        return
    name = sys.argv[1]
    task = TASKS.get(name)
    if task is None:
        print(f"unknown target: {name}", file=sys.stderr)
        t_help()
        sys.exit(2)
    task()


if __name__ == "__main__":
    main()
