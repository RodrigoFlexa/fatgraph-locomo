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

import numpy as np
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
    if settings.ini_path:
        e.add_row("config.ini", settings.ini_path)
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
def doctor(
    condition: str = typer.Option("G1", "--condition", "-C", help="Config to test with."),
    set_: Optional[list[str]] = OptSet,
    show_prompt: bool = typer.Option(False, "--show-prompt", help="Print the full prompt."),
) -> None:
    """Make ONE real LLM call and one embedding, and show exactly what came back.

    This is the command to reach for when every condition scores the same, or
    when the answers are all "Not mentioned in the conversation": it separates
    "the backend is broken" from "the retrieval is bad".
    """
    from fgl.llm import LLMError, build_llm
    from fgl.retrieval import build_embedder

    cfg = _load(condition, set_, dry_run=False)
    ok = True

    # ---- 0. what we are about to send --------------------------------------
    from fgl.llm.azure import is_reasoning_deployment

    settings = load_settings()
    reasoning = (
        is_reasoning_deployment(cfg.llm.deployment)
        if cfg.llm.api_style == "auto"
        else cfg.llm.api_style == "reasoning"
    )
    t0 = Table(show_header=False, box=None, padding=(0, 2))
    t0.add_column(style="bold cyan")
    t0.add_row("deployment", cfg.llm.deployment)
    t0.add_row("família", "[magenta]reasoning[/]" if reasoning else "chat")
    t0.add_row(
        "limite de tokens",
        (f"max_completion_tokens={max(cfg.retrieval.answer_max_tokens, cfg.llm.reasoning_min_tokens)}"
         if reasoning and cfg.llm.reasoning_min_tokens > 0
         else "(omitido)" if reasoning
         else f"max_tokens={cfg.retrieval.answer_max_tokens}"),
    )
    t0.add_row("temperature", "(não enviada)" if reasoning else str(cfg.llm.temperature))
    if reasoning and cfg.llm.reasoning_effort:
        t0.add_row("reasoning_effort", cfg.llm.reasoning_effort)
    t0.add_row("endpoint como", "base_url" if settings.use_base_url else "azure_endpoint")
    t0.add_row("CA bundle", settings.ca_bundle or "(padrão do sistema)")
    t0.add_row("config.ini", settings.ini_path or "(usando .env/ambiente)")
    console.print(Panel(t0, title="[bold]0. forma da requisição", border_style="cyan"))

    # ---- 1. chat completion ------------------------------------------------
    cfg.llm.cache_enabled = False  # always hit the real backend
    cfg.llm.fail_on_empty = False  # we want to *see* the empty, not raise on it
    console.print(f"[bold]1. chat completion[/] · deployment=[cyan]{cfg.llm.deployment}[/]")
    prompt = (
        "Answer with a short phrase, nothing else.\n\n"
        "CONTEXT: Caroline attended the LGBTQ support group on 7 May 2023.\n"
        "QUESTION: When did Caroline attend the support group? Short answer:"
    )
    if show_prompt:
        console.print(Panel(prompt, border_style="dim"))
    llm = None
    try:
        llm = build_llm(cfg.llm)
        text = llm.complete(prompt, purpose="doctor", max_tokens=32)
        u = llm.usage
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="bold cyan")
        t.add_row("resposta (repr)", repr(text)[:300])
        t.add_row("comprimento", str(len(text or "")))
        t.add_row("prompt_tokens", str(u.prompt_tokens))
        t.add_row("completion_tokens", str(u.completion_tokens))
        t.add_row("finish_reason", str(getattr(llm, "last_finish_reason", None)))
        rt = getattr(llm, "last_reasoning_tokens", 0)
        if rt:
            t.add_row("reasoning_tokens", str(rt))
        console.print(t)
        if not (text or "").strip():
            ok = False
            console.print(
                Panel(
                    "A resposta veio [red]VAZIA[/].\n\n"
                    "É exatamente isso que faz todas as condições empatarem com\n"
                    "adversarial=1.000: cada pergunta vira uma abstenção.\n\n"
                    "Verifique, nesta ordem:\n"
                    "  1. o nome do deployment existe no seu recurso Azure?\n"
                    "  2. o gateway/proxy corporativo devolve o corpo da resposta?\n"
                    "  3. o filtro de conteúdo está zerando 'content'?\n"
                    "  4. sua adaptação de src/fgl/llm/client.py devolve "
                    "resp.choices[0].message.content?",
                    title="[red]backend não utilizável",
                    border_style="red",
                )
            )
        elif u.completion_tokens == 0:
            console.print(
                "[yellow]![/] texto veio, mas completion_tokens=0 — o gateway "
                "provavelmente não repassa o bloco 'usage' (só afeta o relatório de custo)"
            )
        else:
            console.print("[green]✓[/] o backend responde normalmente")
    except LLMError as exc:
        ok = False
        console.print(Panel(str(exc), title="[red]falha na chamada", border_style="red"))
    except Exception as exc:  # noqa: BLE001
        ok = False
        console.print(
            Panel(f"{type(exc).__name__}: {exc}", title="[red]erro", border_style="red")
        )

    # ---- 2. JSON mode ------------------------------------------------------
    console.print("\n[bold]2. JSON mode[/] (a extração de fatos depende disso)")
    if llm is None:
        console.print("[dim]pulado — a chamada de chat não chegou a funcionar[/]")
    else:
        try:
            raw = llm.complete(
                '# TASK: doctor\nReturn STRICT JSON: {"ok": true}',
                purpose="doctor", json_mode=True, max_tokens=32,
            )
            console.print(f"  bruto: {raw[:160]!r}")
            from fgl.llm import parse_json_loose

            parse_json_loose(raw)
            console.print("[green]✓[/] JSON parseável")
        except Exception as exc:  # noqa: BLE001
            ok = False
            console.print(f"[red]✗[/] {type(exc).__name__}: {exc}")
            console.print(
                "[yellow]  sem JSON válido a extração devolve zero fatos, "
                "o grafo fica vazio e tudo vira abstenção[/]"
            )

    # ---- 3. embeddings -----------------------------------------------------
    console.print(f"\n[bold]3. embeddings[/] · provider=[cyan]{cfg.embeddings.provider}[/]")
    try:
        emb = build_embedder(cfg.embeddings)
        v = emb.encode_one("Caroline attended the support group.")
        console.print(f"[green]✓[/] dim={len(v)}  norma={float((v @ v) ** 0.5):.3f}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        console.print(f"[red]✗[/] {type(exc).__name__}: {exc}")

    console.print()
    if ok:
        console.print("[green]tudo certo — pode rodar `fgl run G1 -n 1`[/]")
    else:
        console.print("[red]corrija os itens acima antes de rodar o estudo[/]")
    raise typer.Exit(0 if ok else 1)


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


@app.command("verify-topical")
def verify_topical(
    conversation: Optional[list[str]] = OptConversations,
    limit_conversations: int = typer.Option(
        1, "--limit-conversations", "-n", help="How many conversations to test."
    ),
    set_: Optional[list[str]] = OptSet,
    dry_run: bool = OptDry,
    baseline: str = typer.Option("G1", "--baseline", help="Condition to compare against."),
) -> None:
    """Is the topical extraction actually removing the speaker hub? (1 conversa)

    The cheap gate before re-ingesting everything. It ingests ONE conversation
    with each prompt and compares the three numbers that decide whether the
    change did what it was meant to:

      * share of edges touching a speaker -- the disease (86.6% in v1);
      * degree of the bridge vertex between evidence facts -- the symptom
        that kills sigma (median 164 in v1, so k=4 covers 7.3%);
      * fraction of degree-1 vertices -- the risk, since entity-to-entity
        linking can trade one hub for a field of leaves.

    If the speakers stay at ~200 the prompt did not take, and re-ingesting ten
    conversations would only buy the same graph at ten times the price.
    """
    import statistics as st

    from fgl.pipeline import Runner

    rows = []
    for name in (baseline, "T1"):
        cfg = _load(name, set_, dry_run)
        runner = Runner(cfg)
        convs = _dataset(cfg, conversation, limit_conversations)
        touch = tot = 0
        bridge_degrees: list[int] = []
        deg1: list[float] = []
        third: list[int] = []
        for conv in convs:
            graph, _ = runner._ingest(conv)  # noqa: SLF001
            spk = {conv.speaker_a.lower(), conv.speaker_b.lower()}
            sv = {v for v, vx in graph.vertices.items() if vx.name.lower() in spk}
            for e in graph.edges():
                a, b = graph.edge_endpoints(e)
                tot += 1
                touch += int(a in sv or b in sv)
            deg1.append(graph.star_stats()["degree_1_frac"])
            ranked = sorted(graph.vertices, key=lambda v: -graph.degree(v))
            third.append(graph.degree(ranked[2]) if len(ranked) > 2 else 0)

            t2e: dict[str, list[str]] = {}
            for e in graph.edges():
                for t in graph.get_edge_attr(e, "turn_ids") or ():
                    t2e.setdefault(t, []).append(e)
            for q in conv.questions:
                ev = [x for x in (q.evidence or []) if x in t2e]
                if len(ev) < 2 or len(ev) != len(q.evidence or []):
                    continue
                vs = [set(graph.edge_endpoints(t2e[x][0])) for x in ev]
                for v in set.intersection(*vs):
                    bridge_degrees.append(graph.degree(v))
        rows.append({
            "cond": cfg.condition,
            "speaker_edges": touch / tot if tot else 0.0,
            "bridge_median": st.median(bridge_degrees) if bridge_degrees else 0.0,
            "bridge_n": len(bridge_degrees),
            "orbit_k4": (
                sum(1 for d in bridge_degrees if d - 1 <= 4) / len(bridge_degrees)
                if bridge_degrees else 0.0
            ),
            "degree_1_frac": st.mean(deg1) if deg1 else 0.0,
            "third_degree": st.mean(third) if third else 0.0,
        })

    t = Table(title="A extração tópica removeu o hub do falante?")
    t.add_column("métrica")
    for r in rows:
        t.add_column(r["cond"], justify="right")

    # Relative movement, not a pass/fail against a guessed threshold. An
    # earlier version of this command called a 29% reduction in the speaker
    # hub a failure, because the bar had been set by intuition rather than by
    # the data. Direction and magnitude are what a one-conversation probe can
    # honestly report; the verdict belongs to the full run.
    def row(label, key, fmt, better: str):
        a, b = rows[0][key], rows[-1][key]
        if a:
            delta = (b - a) / abs(a)
            arrow = f"{delta:+.0%}"
        else:
            arrow = "-"
        moved_right = (b < a) if better == "down" else (b > a)
        style = "green" if moved_right else "dim"
        t.add_row(label, fmt.format(a), fmt.format(b), f"[{style}]{arrow}[/]")

    t.add_column("Δ", justify="right")
    row("arestas tocando um falante", "speaker_edges", "{:.1%}", "down")
    row("grau mediano do vértice-ponte", "bridge_median", "{:.0f}", "down")
    row("órbita coberta com k=4", "orbit_k4", "{:.1%}", "up")
    row("grau do 3º maior vértice", "third_degree", "{:.0f}", "up")
    row("vértices de grau 1 (risco)", "degree_1_frac", "{:.1%}", "down")
    console.print(t)
    console.print(
        f"[dim]pontes medidas: {rows[0]['bridge_n']} vs {rows[-1]['bridge_n']} "
        "pares de evidência. Se esse número CAIR, veja `fgl diagnose T1`: pode "
        "ser ponte espúria via falante sumindo (bom) ou par de evidência "
        "desconectando (ruim), e só o histograma de distância separa os dois.[/]"
    )

    moved = rows[-1]["speaker_edges"] < rows[0]["speaker_edges"] * 0.9
    safe = rows[-1]["degree_1_frac"] <= rows[0]["degree_1_frac"] + 0.05
    if moved and safe:
        console.print(
            "\n[green]O prompt mexeu no substrato e não fragmentou.[/] Vale "
            "reingerir: [cyan]fgl run-all -C T1 -C G1 -C B3[/]\n"
            "[dim]Se o hub do falante persistir, ele pode ser propriedade do "
            "dataset e não do prompt — nesse caso G11 (hub como stopword) "
            "ataca o mesmo problema sem reingest nenhum.[/]"
        )
    elif not safe:
        console.print(
            "\n[red]O grau 1 subiu:[/] trocamos um hub por um campo de folhas. "
            "A resolução de entidades precisa unificar variantes antes disto "
            "valer a pena."
        )
    else:
        console.print(
            "\n[yellow]O substrato mal se moveu nesta conversa.[/] Antes de "
            "reingerir as dez, rode [cyan]fgl run-all -C G11[/]: ela ataca o "
            "mesmo hub em tempo de recuperação, sobre os grafos que já existem."
        )


@app.command()
def diagnose(
    condition: str = typer.Argument("G1", help="Condition whose graphs to inspect."),
    set_: Optional[list[str]] = OptSet,
    conversation: Optional[list[str]] = OptConversations,
    limit_conversations: int = OptLimitConv,
    dry_run: bool = OptDry,
    show: int = typer.Option(6, "--show", help="Concrete failing cases to print."),
    out: str = typer.Option("", "--out", help="Also write the numbers to this JSON."),
    allow_ingest: bool = typer.Option(
        False, "--allow-ingest",
        help="Build the graphs if missing (costs a full extraction).",
    ),
) -> None:
    """Where does the answer stop being reachable in the memory graph?

    Every condition so far varied the retrieval policy while assuming the graph
    encodes the answer and that the answer is reachable. This tests that. It
    walks the chain of ceilings -- extraction, connectivity, distance, shared
    vertex, common face, cosine rank -- each conditional on the previous, so the
    first big drop is the layer that actually costs the points.

    Only the last rung is something a retrieval policy can fix. A drop before it
    is an ingest problem wearing a retrieval costume.

    Costs no LLM calls -- it reads graphs that already exist, and refuses to
    run when they do not rather than quietly building them.
    """
    import json as _json

    from fgl.evaluation.diagnose import (
        Diagnostician, by_category, failing_cases, waterfall,
    )
    from fgl.pipeline import Runner
    from fgl.retrieval.embeddings import build_index

    cfg = _load(condition, set_, dry_run)
    runner = Runner(cfg)
    convs = _dataset(cfg, conversation, limit_conversations)

    # A diagnostic must not silently become an ingest. `_ingest` builds the
    # graph when it is missing, so pointing this at a condition that was never
    # ingested turns a free read into a paid extraction over every
    # conversation -- which is exactly what happened the first time, half an
    # hour of Azure calls behind a command documented as costing nothing.
    missing = [
        conv.sample_id
        for conv in convs
        if not runner._graph_path(conv).with_suffix(".json").exists()  # noqa: SLF001
    ]
    if missing and not allow_ingest:
        err.print(
            f"[red]{cfg.condition} não tem grafo para "
            f"{len(missing)}/{len(convs)} conversas[/] "
            f"({', '.join(missing[:4])}{'...' if len(missing) > 4 else ''})\n"
            "Este comando LÊ grafos, não os constrói — deixá-lo construir "
            "custaria uma extração completa.\n\n"
            f"  Construa antes:  [cyan]fgl ingest {cfg.condition}[/]\n"
            f"  Ou aceite o custo: [cyan]fgl diagnose {condition} --allow-ingest[/]"
        )
        raise typer.Exit(2)

    traces = []
    for conv in convs:
        graph, _ = runner._ingest(conv)  # noqa: SLF001
        index = build_index(cfg.index, runner.embedder.dim)
        ids, vecs = [], []
        for hid, he in graph.H.items():
            if he.embedding is not None:
                ids.append(hid)
                vecs.append(he.embedding)
        if ids:
            index.add(ids, np.vstack(vecs))
        doc = Diagnostician(graph, runner.embedder, index)
        traces += [doc.trace(q) for q in conv.questions if q.evidence]

    overall = waterfall(traces)
    per_cat = by_category(traces)
    _print_waterfall(cfg.condition, overall, per_cat)

    if show:
        _print_failing(failing_cases(traces, show))
    if out:
        Path(out).write_text(
            _json.dumps({"overall": overall, "per_category": per_cat}, indent=2),
            encoding="utf-8",
        )
        console.print(f"[dim]números → {out}")


def _print_waterfall(condition: str, overall: dict, per_cat: dict) -> None:
    console.print(f"\n[bold]Cascata de tetos — {condition}[/]")
    console.print(
        "[dim]cada degrau é condicional ao anterior; a primeira queda grande é "
        "a camada que custa os pontos.[/]\n"
    )
    rows = [
        ("1. turnos de evidência extraídos", "evidence_turns_extracted",
         "abaixo disto nenhuma política recupera"),
        ("   perguntas com TODA a evidência", "questions_fully_extracted", ""),
        ("2. fatos no mesmo componente", "same_component", "senão não há caminho"),
        ("3. compartilham um vértice", "shares_a_vertex", "σ alcançaria em 1 salto"),
        ("4. numa face comum", "on_a_common_face", "a face alcançaria"),
        ("5. evidência no top-10 por cosseno", "evidence_within_top_10",
         "o único degrau que a recuperação conserta"),
    ]
    t = Table(show_lines=False)
    t.add_column("degrau")
    for cat in per_cat:
        t.add_column(cat[:9], justify="right")
    t.add_column("geral", justify="right", style="bold")
    t.add_column("", style="dim")
    for label, key, note in rows:
        cells = [f"{per_cat[c].get(key, 0):.1%}" if key in per_cat[c] else "-"
                 for c in per_cat]
        t.add_row(label, *cells, f"{overall.get(key, 0):.1%}", note)
    console.print(t)

    if "distance_median" in overall:
        console.print(
            f"\n  distância mediana entre fatos de evidência: "
            f"[bold]{overall['distance_median']:.0f}[/] saltos   "
            f"histograma {overall.get('distance_hist', {})}"
        )
    if "worst_evidence_rank_median" in overall:
        console.print(
            f"  rank mediano do fato de evidência mais difícil: "
            f"[bold]{overall['worst_evidence_rank_median']:.0f}[/]   "
            + "  ".join(
                f"top-{k}: {overall.get(f'evidence_within_top_{k}', 0):.0%}"
                for k in (5, 10, 20, 50, 100)
            )
        )
        console.print(
            "[dim]  rank ~12 diz 'aumente k'. rank ~900 diz que a pergunta e a "
            "evidência não se parecem, e nenhum k resolve.[/]"
        )


def _print_failing(cases) -> None:
    console.print("\n[bold]Casos concretos[/] [dim](o que as porcentagens escondem)[/]")
    for c in cases:
        status = (
            "[red]evidência NUNCA extraída[/]"
            if not c.fully_extracted
            else f"[yellow]extraída, mas rank {c.worst_rank}[/]"
        )
        console.print(f"\n  {status}")
        console.print(f"  P    : {c.question[:88]}")
        console.print(f"  gold : {c.gold[:70]}")
        console.print(
            f"  evid : {c.evidence}   extraídos: {c.covered or '[]'}"
        )
        if c.fully_extracted and c.distance is not None:
            console.print(
                f"  grafo: distância {c.distance}, mesmo componente="
                f"{c.same_component}, mesma face={c.same_face}, "
                f"compartilha vértice={c.shares_vertex}"
            )


@app.command()
def judge(
    condition: Optional[list[str]] = typer.Option(
        None, "--condition", "-C", help="Restrict to these conditions (repeatable)."
    ),
    set_: Optional[list[str]] = OptSet,
    dry_run: bool = OptDry,
    limit: int = typer.Option(0, "--limit", "-n", help="Judge only the first N rows."),
    show: int = typer.Option(
        12, "--show", help="How many judge/F1 disagreements to print for inspection."
    ),
) -> None:
    """Re-score saved predictions with an LLM judge (does not re-answer).

    Token-overlap F1 punishes paraphrase, and the ceiling analysis showed why
    that matters: on questions whose retrieval already put every evidence turn
    in the prompt, F1 is only 0.515. This separates "the answerer is wrong"
    from "the metric says wrong".

    Runs over `results/<condition>/predictions.jsonl`, so no question is
    answered again and every condition already on disk can be re-scored.
    """
    from fgl.evaluation import load_results
    from fgl.evaluation.judge import (
        Judge, disagreements, judge_metrics, load_predictions, write_judged,
    )
    from fgl.pipeline import Runner

    cfg = _load(condition[0] if condition else "G1", set_, dry_run)
    runner = Runner(cfg)
    results_dir = runner.paths.resolve(cfg.paths.results_dir)
    wanted = condition or sorted(
        p.name for p in results_dir.iterdir() if (p / "predictions.jsonl").exists()
    )

    judge_obj = Judge(runner.llm, runner.prompts)
    for name in wanted:
        try:
            cond_cfg = _load(name, set_, dry_run)
        except typer.Exit:
            cond_cfg = cfg
        path = results_dir / cond_cfg.condition / "predictions.jsonl"
        if not path.exists():
            err.print(f"[yellow]sem predições para {cond_cfg.condition}, pulando")
            continue

        rows = load_predictions(path)
        if limit:
            rows = rows[:limit]
        console.print(f"[bold]julgando[/] {cond_cfg.condition} — {len(rows)} respostas")
        with _progress() as bar:
            task = bar.add_task("judge", total=len(rows))
            judged = judge_obj.judge_all(
                rows, progress=lambda i, n: bar.update(task, completed=i)
            )
        write_judged(path, judged)

        block = judge_metrics(judged)
        mpath = path.with_name("metrics.json")
        if mpath.exists():
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
            metrics.setdefault("overall", {}).update(
                {k: v for k, v in block.items() if k != "judge_per_category"}
            )
            for cat, entry in block.get("judge_per_category", {}).items():
                metrics.setdefault("per_category", {}).setdefault(cat, {})["judge"] = (
                    entry["judge"]
                )
            mpath.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        console.print(
            f"  F1 {block.get('judge_micro', 0):.4f} (juiz) vs "
            f"{np_mean_f1(judged):.4f} (tokens)   "
            f"concordância {block.get('judge_f1_agreement', 0):.1%}   "
            f"juiz aceita/F1 rejeita: {block.get('judge_yes_f1_low', 0)}   "
            f"juiz rejeita/F1 aceita: {block.get('judge_no_f1_high', 0)}"
        )
        if show:
            _print_disagreements(disagreements(judged, limit=show))

    console.print(
        "\n[dim]Leia as discordâncias acima antes de citar o número do juiz. "
        "Se as da coluna 'juiz rejeita' parecerem corretas, ele está severo "
        "demais e o prompt precisa de ajuste.[/]"
    )


def np_mean_f1(judged) -> float:
    import numpy as np

    return float(np.mean([j.f1 for j in judged])) if judged else 0.0


def _print_disagreements(rows: list[dict]) -> None:
    if not rows:
        console.print("  [dim](sem discordâncias)")
        return
    t = Table(show_lines=False)
    for col in ("cat", "F1", "juiz", "gold", "predição"):
        t.add_column(col, overflow="fold")
    for r in rows:
        verdict = "[green]aceita[/]" if r["judge"] else "[red]rejeita[/]"
        t.add_row(
            str(r["category"])[:10], f"{r['f1']:.2f}", verdict,
            r["gold"][:60], r["prediction"][:60],
        )
    console.print(t)


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
    _print_sanity(metrics)
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

    # G4/G5/G6 right after G1: they reuse G1's graphs, so G1 must build them
    # first. B1 last: by far the priciest.
    # G7/G8 junto de G4: reusam os grafos da G1 e são as duas condições de
    # decisão (sigma sem passeio; e o teste de ordem). G9 constrói os seus, por
    # reescrever sigma. B1 por último: de longe a mais cara.
    # G10 (a proposta) e B3 (o alvo) primeiro: é a comparação que decide.
    # B1 por último, de longe a mais cara.
    order = ["G10", "B3", "G9", "G1", "G4", "G7", "G8", "G5", "G6", "B2", "G2", "G3", "B1"]
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
        prompt_name=cfgs[0].ingest.extract_prompt,
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


def _print_sanity(metrics: dict) -> None:
    """Loudly refuse to let a degenerate run pass for a result."""
    sanity = metrics.get("sanity") or {}
    if sanity.get("ok", True):
        return
    console.print(
        Panel(
            "\n".join(f"• {w}" for w in sanity.get("warnings", []))
            + "\n\n[bold]Estes números não devem ser interpretados.[/]\n"
            "Diagnostique com [cyan]fgl doctor[/].",
            title="[red]corrida suspeita",
            border_style="red",
        )
    )


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
