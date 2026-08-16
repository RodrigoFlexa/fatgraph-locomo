"""``fgl`` — the command line interface.

    fgl info                                  what the environment looks like
    fgl setup                                 fetch the LoCoMo dataset
    fgl config list | show G1 | keys | diff   inspect the resolved configuration
    fgl ingest G1                             build the memory graphs
    fgl qa G1                                 answer + score
    fgl run G1                                ingest + qa
    fgl run-all                               every condition + the final report
    fgl report                                rebuild the tables from results/

Every command that loads a configuration accepts ``--set dotted.key=value``,
repeatable, so any knob can be swept from the shell without editing YAML::

    fgl run G1 --set retrieval.top_m_anchors=8 --set retrieval.budget_tokens=4000
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table

from fgl import __version__
from fgl.config import Config, ConfigError, list_conditions, resolve_condition
from fgl.paths import Paths, project_root
from fgl.settings import load_settings

app = typer.Typer(
    name="fgl",
    help="Fatgraph long-term memory for LLM agents, benchmarked on LoCoMo.",
    add_completion=True,
    rich_markup_mode="rich",
)
config_app = typer.Typer(help="Inspect and resolve configurations.", no_args_is_help=True)
app.add_typer(config_app, name="config")

console = Console()
err = Console(stderr=True)

LOCOMO_URL = "https://github.com/snap-research/locomo"
LOCOMO_BRANCH = "code"

# --------------------------------------------------------------------------- #
# Shared options                                                               #
# --------------------------------------------------------------------------- #

OptSet = typer.Option(
    None, "--set", "-s",
    help="Override a config value: [cyan]section.key=value[/]. Repeatable.",
)
OptConversations = typer.Option(
    None, "--conversation", "-c",
    help="Restrict to these sample_ids (repeatable). Default: all 10.",
)
OptLimitConv = typer.Option(0, "--limit-conversations", "-n", help="0 = no limit.")
OptLimitQ = typer.Option(
    0, "--limit-questions", "-q",
    help="0 = all questions (the reported protocol). Use >0 only for smoke runs.",
)
OptDry = typer.Option(
    False, "--dry-run", "-d",
    help="Offline deterministic backends: no network, no model download, no spend.",
)


def _load(condition: str, overrides, dry_run: bool) -> Config:
    """Resolve a configuration.

    Precedence, lowest to highest: ``configs/base.yaml`` -> the condition YAML
    -> ``.env`` / environment -> ``--dry-run`` -> ``--set``.  ``--set`` is applied
    last on purpose: an explicit override must never be silently discarded, not
    even by ``--dry-run``.
    """
    settings = load_settings()
    try:
        cfg = Config.load(condition, settings=settings)
        if dry_run:
            _make_offline(cfg)
        if overrides:
            cfg.apply_overrides(overrides)
        cfg.validate()
    except ConfigError as exc:
        err.print(f"[red]config error:[/] {exc}")
        raise typer.Exit(2)
    if cfg.requires_azure() and not settings.azure_ready:
        err.print(
            Panel(
                f"[yellow]{settings.explain_missing()}[/]\n\n"
                f"Edite [cyan]{settings.dotenv_path}[/] "
                "(comece de [cyan].env.example[/]) ou exporte as variáveis.\n"
                "Ou rode offline com [cyan]--dry-run[/].",
                title="[red]Azure não configurado",
                border_style="red",
            )
        )
        raise typer.Exit(3)
    return cfg


def _make_offline(cfg: Config) -> None:
    cfg.llm.provider = "fake"
    cfg.llm.cache_enabled = False
    cfg.embeddings.provider = "hashing"
    cfg.embeddings.cache_dir = ".cache/embeddings-dry"
    cfg.paths.facts_cache = "artifacts/facts-dry"
    cfg.paths.graphs_dir = "artifacts/graphs-dry"
    cfg.paths.logs_dir = "artifacts/logs-dry"
    cfg.paths.results_dir = "results-dry"


def _dataset(cfg: Config, conversations, limit: int):
    from fgl.pipeline import load_dataset, select_conversations

    try:
        convs = load_dataset(cfg)
    except FileNotFoundError as exc:
        err.print(f"[red]{exc}")
        raise typer.Exit(4)
    try:
        return select_conversations(convs, conversations, limit)
    except ValueError as exc:
        err.print(f"[red]{exc}")
        raise typer.Exit(2)


class _Bar:
    """Adapter between :class:`rich.progress.Progress` and ``Runner.progress``."""

    def __init__(self, progress: Progress, label: str) -> None:
        self.p = progress
        self.tasks: dict[str, int] = {}
        self.label = label

    def __call__(self, stage: str, done: int, total: int, detail: str) -> None:
        key = stage
        if key not in self.tasks:
            self.tasks[key] = self.p.add_task(f"{self.label} · {stage}", total=total or 1)
        self.p.update(
            self.tasks[key], completed=done, total=total or 1,
            description=f"{self.label} · {stage} [dim]{detail}[/]",
        )


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


# --------------------------------------------------------------------------- #
# info / setup                                                                 #
# --------------------------------------------------------------------------- #


@app.command()
def info() -> None:
    """Show versions, paths, credentials and dataset status."""
    paths = Paths.build()
    settings = load_settings()

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_row("fgl", __version__)
    t.add_row("python", sys.version.split()[0])
    t.add_row("project root", str(paths.root))
    t.add_row("package", _install_status(paths))
    console.print(Panel(t, title="[bold]environment", border_style="cyan"))

    d = Table(show_header=True, header_style="bold")
    d.add_column("dependency"); d.add_column("status")
    for mod, why in [
        ("numpy", "required"), ("yaml", "required"), ("typer", "required"),
        ("openai", "Azure backend"), ("sentence_transformers", "default embeddings"),
        ("nltk", "official Porter stemmer"), ("faiss", "optional index"),
        ("tiktoken", "exact token counts"), ("pandas", "notebooks"),
        ("matplotlib", "notebooks"),
    ]:
        ok = _has(mod)
        d.add_row(mod, f"[green]✓[/] {why}" if ok else f"[yellow]—[/] {why} (not installed)")
    console.print(Panel(d, title="[bold]dependencies", border_style="cyan"))

    e = Table(show_header=True, header_style="bold")
    e.add_column("setting"); e.add_column("value")
    e.add_row(".env", f"[green]{settings.dotenv_path}[/]" if settings.dotenv_found
              else f"[yellow]not found at {settings.dotenv_path}[/]")
    for k, v in settings.redacted().items():
        if k == "dotenv_found":
            continue
        e.add_row(k, str(v) if v is not None else "[dim]unset[/]")
    e.add_row(
        "azure ready",
        "[green]yes[/]" if settings.azure_ready
        else f"[yellow]no — {settings.explain_missing()}[/]",
    )
    console.print(Panel(e, title="[bold]settings (.env + environment)", border_style="cyan"))

    ds = Table(show_header=True, header_style="bold")
    ds.add_column("artefact"); ds.add_column("path"); ds.add_column("status")
    have = paths.locomo_file.exists()
    ds.add_row("LoCoMo data", str(paths.locomo_file),
               "[green]✓[/]" if have else "[yellow]missing — run `fgl setup`[/]")
    for label, p in [("results", paths.results), ("artifacts", paths.artifacts),
                     ("cache", paths.cache)]:
        n = len(list(p.rglob("*"))) if p.exists() else 0
        ds.add_row(label, str(p), f"{n} files" if n else "[dim]empty[/]")
    console.print(Panel(ds, title="[bold]data & outputs", border_style="cyan"))

    conds = list_conditions()
    c = Table(show_header=True, header_style="bold")
    c.add_column("condition"); c.add_column("file"); c.add_column("results")
    for stem, cond, path in conds:
        done = (paths.results / cond / "metrics.json").exists()
        c.add_row(cond, path.name, "[green]✓[/]" if done else "[dim]—[/]")
    console.print(Panel(c, title="[bold]conditions", border_style="cyan"))


@app.command()
def setup(
    force: bool = typer.Option(False, "--force", help="Re-clone even if present."),
) -> None:
    """Prepare the project: create .env and fetch the official LoCoMo dataset.

    Safe to re-run. Creating ``.env`` happens first and unconditionally, so the
    command is still useful when the dataset is already in place.
    """
    paths = Paths.build().ensure()
    _ensure_dotenv(paths)

    dest = paths.locomo_repo
    if paths.locomo_file.exists() and not force:
        console.print(f"[green]✓[/] dataset already at [cyan]{paths.locomo_file}[/]")
        _next_steps(paths)
        raise typer.Exit(0)
    if dest.exists():
        shutil.rmtree(dest)

    if not shutil.which("git"):
        err.print(
            "[red]git not found on PATH.[/] Install git, or download "
            f"{LOCOMO_URL}/blob/{LOCOMO_BRANCH}/data/locomo10.json manually to "
            f"[cyan]{paths.locomo_file}[/]"
        )
        raise typer.Exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"cloning [cyan]{LOCOMO_URL}[/] (branch [cyan]{LOCOMO_BRANCH}[/]) …")
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "-b", LOCOMO_BRANCH, LOCOMO_URL, str(dest)]
    )
    if r.returncode != 0 or not paths.locomo_file.exists():
        err.print("[red]clone failed or data file missing")
        raise typer.Exit(1)

    console.print(f"[green]✓[/] dataset at [cyan]{paths.locomo_file}[/]")
    _next_steps(paths)


def _ensure_dotenv(paths: Paths) -> None:
    env, example = paths.root / ".env", paths.root / ".env.example"
    if env.exists():
        console.print(f"[green]✓[/] [cyan].env[/] already exists (left untouched)")
    elif example.exists():
        shutil.copy(example, env)
        console.print(f"[green]✓[/] created [cyan]{env}[/] from .env.example")
    else:
        console.print("[yellow]![/] no .env.example to copy from")


def _next_steps(paths: Paths) -> None:
    ready = load_settings().azure_ready
    console.print(
        Panel(
            ("[green]Credentials look set.[/]\n" if ready else
             f"[yellow]1.[/] Fill in [cyan]{paths.root / '.env'}[/] "
             "(AZURE_OPENAI_ENDPOINT / API_KEY / API_VERSION)\n")
            + "[yellow]2.[/] [cyan]fgl info[/] — confirm what got picked up\n"
            + "[yellow]3.[/] [cyan]fgl run-all --dry-run -n 1 -q 10[/] — free smoke test\n"
            + "[yellow]4.[/] [cyan]fgl run G1 -n 1[/] — first real run, one conversation\n"
            + "[yellow]5.[/] [cyan]fgl run-all[/] — the full study",
            title="[bold]next steps",
            border_style="cyan",
        )
    )


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #


@config_app.command("list")
def config_list() -> None:
    """List the available conditions."""
    t = Table(header_style="bold")
    t.add_column("condition", style="cyan"); t.add_column("file"); t.add_column("summary")
    for stem, cond, path in list_conditions():
        try:
            cfg = Config.from_yaml(path)
            summary = (
                f"σ={cfg.ingest.sigma_policy}, curation={cfg.curation.curation}, "
                f"consolidation={cfg.curation.consolidation}"
            )
        except Exception as exc:  # noqa: BLE001
            summary = f"[red]{exc}"
        t.add_row(cond, path.name, summary)
    console.print(t)


@config_app.command("show")
def config_show(
    condition: str = typer.Argument(..., help="Condition name, id or prefix (e.g. G1)."),
    set_: Optional[list[str]] = OptSet,
    dry_run: bool = OptDry,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of YAML."),
    paths_only: bool = typer.Option(False, "--paths", help="Show resolved paths only."),
) -> None:
    """Print the fully resolved configuration (base + condition + env + --set)."""
    cfg = _load(condition, set_, dry_run)
    if paths_only:
        t = Table(header_style="bold")
        t.add_column("key", style="cyan"); t.add_column("resolved path")
        for k, v in cfg.resolved_paths().items():
            t.add_row(k, str(v) + ("" if v.exists() else "  [yellow](missing)[/]"))
        console.print(t)
        return
    if as_json:  # machine-readable: no decoration, safe to pipe into jq
        typer.echo(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return
    console.print(
        Panel(
            Syntax(cfg.to_yaml(), "yaml", theme="ansi_dark"),
            title=f"[bold]{cfg.condition}[/]  [dim]{cfg.source}",
            border_style="cyan",
        )
    )


@config_app.command("keys")
def config_keys(
    grep: str = typer.Argument("", help="Filter keys containing this substring."),
) -> None:
    """List every key accepted by --set, with its current default."""
    cfg = Config()
    t = Table(header_style="bold")
    t.add_column("key", style="cyan"); t.add_column("type"); t.add_column("default")
    for k, v in cfg.flat().items():
        if grep and grep.lower() not in k.lower():
            continue
        t.add_row(k, type(v).__name__, str(v))
    console.print(t)


@config_app.command("diff")
def config_diff(
    left: str = typer.Argument(..., help="Baseline condition."),
    right: str = typer.Argument(..., help="Condition to compare against it."),
) -> None:
    """Show exactly which settings differ between two conditions."""
    a, b = _load(left, None, True), _load(right, None, True)
    d = a.diff(b)
    d.pop("condition", None)
    if not d:
        console.print("[yellow]identical (apart from the condition id)")
        return
    t = Table(header_style="bold")
    t.add_column("key", style="cyan"); t.add_column(a.condition); t.add_column(b.condition)
    for k, (va, vb) in sorted(d.items()):
        t.add_row(k, str(va), f"[green]{vb}[/]")
    console.print(t)


@config_app.command("validate")
def config_validate() -> None:
    """Load and validate every condition; non-zero exit if any fails."""
    bad = 0
    for stem, cond, path in list_conditions():
        try:
            Config.from_yaml(path).validate()
            console.print(f"[green]✓[/] {cond}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            console.print(f"[red]✗[/] {cond}: {exc}")
    raise typer.Exit(1 if bad else 0)


# --------------------------------------------------------------------------- #
# ingest / qa / run                                                            #
# --------------------------------------------------------------------------- #


@app.command()
def ingest(
    condition: str = typer.Argument(..., help="Condition name, id or prefix."),
    set_: Optional[list[str]] = OptSet,
    conversation: Optional[list[str]] = OptConversations,
    limit_conversations: int = OptLimitConv,
    dry_run: bool = OptDry,
    force: bool = typer.Option(False, "--force", help="Rebuild existing graphs."),
) -> None:
    """Build the fatgraph memory (dialogues only — questions are never read)."""
    from fgl.pipeline import Runner

    cfg = _load(condition, set_, dry_run)
    convs = _dataset(cfg, conversation, limit_conversations)
    console.print(f"[bold]{cfg.condition}[/] · ingesting {len(convs)} conversation(s)")

    with _progress() as bar:
        runner = Runner(cfg, progress=_Bar(bar, cfg.condition))
        reports = runner.ingest_only(convs, force=force)

    t = Table(header_style="bold")
    for col in ("conversation", "facts", "V", "E", "F", "C", "genus",
                "incongr.", "collapses", "consolid."):
        t.add_column(col)
    for r in reports:
        g = r["graph_stats"]
        t.add_row(r["sample_id"], str(r["n_facts"]), str(g["V"]), str(g["E"]),
                  str(g["F"]), str(g["C"]), str(g["genus"]), str(r["n_incongruent"]),
                  str(r["n_collapses"]), str(r["n_consolidations"]))
    console.print(t)
    _print_cost(runner)


@app.command()
def qa(
    condition: str = typer.Argument(..., help="Condition name, id or prefix."),
    set_: Optional[list[str]] = OptSet,
    conversation: Optional[list[str]] = OptConversations,
    limit_conversations: int = OptLimitConv,
    limit_questions: int = OptLimitQ,
    dry_run: bool = OptDry,
) -> None:
    """Answer every LoCoMo question and score it (ingests first if needed)."""
    _run_condition(condition, set_, conversation, limit_conversations,
                   limit_questions, dry_run)


@app.command()
def run(
    condition: str = typer.Argument(..., help="Condition name, id or prefix."),
    set_: Optional[list[str]] = OptSet,
    conversation: Optional[list[str]] = OptConversations,
    limit_conversations: int = OptLimitConv,
    limit_questions: int = OptLimitQ,
    dry_run: bool = OptDry,
) -> None:
    """Ingest + QA for one condition. Alias of `qa` (which ingests on demand)."""
    _run_condition(condition, set_, conversation, limit_conversations,
                   limit_questions, dry_run)


def _run_condition(condition, set_, conversation, limit_conversations,
                   limit_questions, dry_run) -> dict:
    from fgl.evaluation import markdown_table
    from fgl.pipeline import Runner

    cfg = _load(condition, set_, dry_run)
    convs = _dataset(cfg, conversation, limit_conversations)
    n_q = sum(len(c.questions) for c in convs) if not limit_questions else \
        limit_questions * len(convs)
    console.print(
        f"[bold]{cfg.condition}[/] · {len(convs)} conversation(s), ~{n_q} question(s)"
        + ("  [yellow](dry-run)[/]" if dry_run else "")
    )

    with _progress() as bar:
        runner = Runner(cfg, progress=_Bar(bar, cfg.condition))
        metrics = runner.run(convs, limit_questions=limit_questions)

    console.print()
    console.print(markdown_table({cfg.condition: metrics}))
    console.print()
    _print_cost(runner)
    console.print(f"results → [cyan]{runner.results_dir() / 'metrics.json'}[/]")
    return metrics


@app.command("run-all")
def run_all(
    condition: Optional[list[str]] = typer.Option(
        None, "--condition", "-C", help="Restrict to these conditions (repeatable)."
    ),
    set_: Optional[list[str]] = OptSet,
    conversation: Optional[list[str]] = OptConversations,
    limit_conversations: int = OptLimitConv,
    limit_questions: int = OptLimitQ,
    dry_run: bool = OptDry,
    continue_on_error: bool = typer.Option(
        False, "--continue-on-error", help="Keep going if one condition fails."
    ),
) -> None:
    """Run every condition, then write the comparison report.

    The shared fact-extraction cache is warmed first so B3 and G1 consume
    byte-identical facts — the whole point of the B3 ablation.
    """
    from fgl.evaluation import build_report, load_results, write_report
    from fgl.memory.ingest import FactExtractor
    from fgl.pipeline import Runner

    order = ["G1", "B3", "B2", "G2", "G3", "B1"]  # B1 last: by far the priciest
    wanted = condition or order
    cfgs = []
    for name in wanted:
        try:
            cfgs.append(_load(name, set_, dry_run))
        except typer.Exit:
            raise
    if not cfgs:
        err.print("[red]no conditions selected"); raise typer.Exit(2)

    convs = _dataset(cfgs[0], conversation, limit_conversations)

    # ---- warm the shared extraction cache ------------------------------
    console.print(f"[bold]warming the shared fact cache[/] over {len(convs)} conversation(s)")
    warm = Runner(cfgs[0])
    extractor = FactExtractor(
        warm.llm, warm.prompts,
        warm.paths.resolve(cfgs[0].paths.facts_cache),
        cfgs[0].ingest.max_facts_per_session,
    )
    with _progress() as bar:
        b = _Bar(bar, "extract")
        for i, conv in enumerate(convs):
            b("facts", i, len(convs), conv.sample_id)
            extractor.extract_all(conv)
        b("facts", len(convs), len(convs), "done")

    # ---- run each condition ---------------------------------------------
    failures = []
    for cfg in cfgs:
        console.rule(f"[bold cyan]{cfg.condition}")
        try:
            with _progress() as bar:
                Runner(cfg, progress=_Bar(bar, cfg.condition)).run(
                    convs, limit_questions=limit_questions
                )
        except Exception as exc:  # noqa: BLE001
            failures.append((cfg.condition, exc))
            err.print(f"[red]{cfg.condition} failed:[/] {exc}")
            if not continue_on_error:
                raise typer.Exit(1)

    # ---- report -----------------------------------------------------------
    paths = Paths.build()
    results_dir = paths.resolve(cfgs[0].paths.results_dir)
    results = load_results(results_dir)
    out = write_report(results, results_dir / "report.md")
    console.rule("[bold]report")
    console.print(build_report(results))
    console.print(f"report → [cyan]{out}[/]")
    if failures:
        err.print(f"[yellow]{len(failures)} condition(s) failed: "
                  + ", ".join(c for c, _ in failures))
        raise typer.Exit(1)


@app.command()
def report(
    results_dir: Optional[Path] = typer.Option(
        None, "--results-dir", "-r", help="Defaults to results/ (or results-dry/ with -d)."
    ),
    dry_run: bool = OptDry,
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Where to write report.md."),
) -> None:
    """Rebuild the comparison tables from whatever is already in results/."""
    from fgl.evaluation import build_report, load_results, write_report

    paths = Paths.build()
    rd = results_dir or (paths.root / ("results-dry" if dry_run else "results"))
    results = load_results(rd)
    if not results:
        err.print(f"[yellow]no metrics.json found under {rd}")
        raise typer.Exit(1)
    console.print(build_report(results))
    written = write_report(results, out or (rd / "report.md"))
    console.print(f"\nreport → [cyan]{written}[/]")


def _print_cost(runner) -> None:
    u = runner.llm.usage.to_dict()
    console.print(
        f"[dim]LLM: {u['calls']} calls ({u['cached_calls']} cached) · "
        f"{u['prompt_tokens']:,} prompt + {u['completion_tokens']:,} completion tokens[/]"
    )


def _has(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _install_status(paths: Paths) -> str:
    """Editable install, or a stale copy?

    On Ubuntu 22.04 ``python3 -m venv`` seeds setuptools 59, which predates
    PEP 660. ``pip install -e .`` then silently *copies* the package into
    site-packages instead of linking it, so edits under ``src/`` have no effect
    and the symptom is baffling. Detect it and say what to do.
    """
    import fgl

    installed = Path(fgl.__file__).resolve().parent
    source = (paths.root / "src" / "fgl").resolve()
    if not source.exists():
        return f"[green]installed[/] {installed}"
    if installed == source:
        return f"[green]editable[/] → {source}"
    return (
        f"[red]COPY, not editable[/] ({installed})\n"
        "edits under src/ will be ignored — reinstall with:\n"
        "  pip install --upgrade pip setuptools wheel && pip install -e ."
    )


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Print the version and exit.", is_eager=True
    ),
) -> None:
    if version:
        typer.echo(f"fgl {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


def main() -> None:  # console_scripts entry point
    app()


if __name__ == "__main__":
    main()
