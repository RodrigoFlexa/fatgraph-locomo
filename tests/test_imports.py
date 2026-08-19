"""Every ``from fgl... import X`` in the repo names something that exists.

Why this file exists, written the day it was needed
---------------------------------------------------
``test_sweep_runs_end_to_end_without_an_llm`` imported ``load_locomo`` from
``fgl.data.locomo``. The real function is ``load_conversations``. The test was
decorated ``@needs_dataset``, so on every machine without the LoCoMo file --
which is every machine the suite was written on -- pytest skipped it *before*
executing the import, and the suite was green for as long as nobody had the
dataset. The bug surfaced on the one machine that did.

That is a general hazard and not a one-off slip:

* imports inside a function body are not checked until the body runs;
* ``skipif`` decorators stop the body from ever running;
* the same is true of the lazy ``import`` statements this codebase uses
  deliberately in CLI commands, to keep ``fgl --help`` fast.

So the guard has to be static. This walks every Python file under ``src/`` and
``tests/``, finds every ``from fgl...`` import at ANY nesting depth, imports the
module and checks the attribute is really there. It costs about a second and it
makes "green suite" mean what people assume it means.

It deliberately does NOT execute the importing module -- only the module being
imported *from*. A test file that needs a missing optional dependency still
gets checked for typos in its ``fgl`` imports.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from conftest import PATHS

ROOTS = ("src", "tests")

#: Names a module is allowed not to have because the import is conditional on
#: an optional dependency that this environment may lack. Empty on purpose:
#: nothing in `fgl` is optional at the level of a symbol, only at the level of
#: a third-party package, and those are not `fgl` imports. Kept as a named
#: escape hatch so a future exception has to be argued for in writing.
ALLOWED_MISSING: dict[str, set[str]] = {}


def _python_files() -> list[Path]:
    out: list[Path] = []
    for root in ROOTS:
        base = PATHS.root / root
        if base.exists():
            out.extend(
                p for p in base.rglob("*.py") if "__pycache__" not in p.parts
            )
    return sorted(out)


def _fgl_imports(path: Path) -> list[tuple[int, str, str]]:
    """``(lineno, module, name)`` for every ``from fgl... import name``.

    ``ast.walk`` rather than iterating the module body, precisely so that
    imports nested inside functions -- the ones a skipped test never reaches --
    are covered.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:  # pragma: no cover - would fail elsewhere first
        pytest.fail(f"{path} does not parse: {exc}")
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level or not node.module or not node.module.startswith("fgl"):
            continue
        for alias in node.names:
            if alias.name != "*":
                found.append((node.lineno, node.module, alias.name))
    return found


def test_the_repo_has_python_files_to_check():
    """A path bug here would make every assertion below vacuously true."""
    files = _python_files()
    assert len(files) > 30, f"only found {len(files)} files under {ROOTS}"


def test_every_fgl_import_names_something_that_exists():
    """Including the ones inside skipped tests and lazy CLI command bodies."""
    problems: list[str] = []
    for path in _python_files():
        rel = path.relative_to(PATHS.root)
        for lineno, module, name in _fgl_imports(path):
            try:
                mod = importlib.import_module(module)
            except Exception as exc:  # pragma: no cover - environment-specific
                problems.append(f"{rel}:{lineno} cannot import {module}: {exc}")
                continue
            if name in ALLOWED_MISSING.get(module, ()):
                continue
            if not hasattr(mod, name):
                problems.append(
                    f"{rel}:{lineno} `from {module} import {name}` -- "
                    f"{module} has no attribute {name!r}"
                )
    assert not problems, "\n" + "\n".join(problems)


def test_the_loader_the_dataset_gated_tests_use_is_the_real_one():
    """The specific bug, pinned by name.

    Regression guard rather than a duplicate of the sweep above: it says out
    loud which symbol was wrong, so a future rename of `load_conversations`
    fails here with an obvious message instead of inside a test that most
    machines skip.
    """
    from fgl.data import locomo

    assert hasattr(locomo, "load_conversations")
    assert not hasattr(locomo, "load_locomo"), (
        "if this name now exists, update the comment in this file -- the guard "
        "was written because it did not"
    )
