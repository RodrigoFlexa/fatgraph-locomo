"""CLI entry point for CLIO, wired into ``fgl clio ...`` as a sub-app (see
``fgl/cli.py``'s one-line ``app.add_typer(clio_app, name="clio")``). This
is the thing to actually run to check M5 (extraction) and M8 (the agent
loop) against a real deployment -- the two places this package calls an
LLM, and the two milestones no offline pytest run can validate for
QUALITY (only for plumbing).

Mirrors ``fgl doctor``'s pattern for real-credential runs: reads ``.env``
via :func:`fgl.settings.load_settings`, the same way every other real call
in this repository does, so there is nothing new to configure.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from fgl.clio.catalog import load_catalog

clio_app = typer.Typer(help="CLIO: bitemporal-graph long-term memory (M1-M8).")
console = Console()

_DEMO_CONVERSATION = [
    ("2023-01-14", "I started at Vertex this week, I'm living in Recife"),
    ("2023-03-02", "My manager here is Bia, she also likes climbing"),
    ("2023-06-20", "I moved to Salvador last month, I'm still at Vertex remotely"),
    ("2023-09-05", "I left Vertex, joined Kaia. My boss now is Rui"),
    ("2023-11-11", "Went climbing again over the weekend"),
    ("2023-12-01", "Actually Bia was never my manager, she was on another team"),
]

_DEMO_QUESTIONS = [
    "Where does Melanie work now?",
    "Where did Melanie live in February 2023?",
    "How many times has Melanie mentioned climbing?",
]


def _build_backends(fake: bool, no_cache: bool = False):
    """Returns ``(llm, embedder)``. Shared by ``demo`` and ``bench`` so
    both hit the exact same credentials/cache behaviour."""
    if fake:
        from fgl.clio.demo_fake import demo_fake_responder
        from fgl.config import LLMConfig
        from fgl.llm.client import FakeLLM
        from fgl.retrieval.embeddings import HashingEmbedder

        return (
            FakeLLM(LLMConfig(provider="fake"), responder=demo_fake_responder()),
            HashingEmbedder(dim=128),
        )

    from fgl.config import Config
    from fgl.llm.client import build_llm
    from fgl.retrieval.embeddings import build_embedder
    from fgl.settings import load_settings

    # A bare `Config()` here is just a dataclass tree (no YAML/condition
    # involved) -- only its `.llm`/`.embeddings` sub-objects are used, as
    # the target `settings.apply_to()` (the same .env overlay `fgl doctor`
    # uses) needs to write real credentials into.
    settings = load_settings()
    shim = Config()
    settings.apply_to(shim)
    # Cached by default, like every other real call in this repository
    # (fgl run/fgl qa) -- re-running the same turns/questions (e.g. after
    # bumping --sessions, or resuming a `bench` run) must not re-pay for
    # prompts already answered. --no-cache is for deliberately forcing a
    # fresh call.
    shim.llm.cache_enabled = not no_cache
    return build_llm(shim.llm), build_embedder(shim.embeddings)


def _build_clio(fake: bool, no_cache: bool = False):
    from fgl.clio.config import ClioConfig
    from fgl.clio.facade import Clio

    llm, embedder = _build_backends(fake, no_cache)
    return Clio.build(config=ClioConfig.default(), llm=llm, embedder=embedder)


def _load_locomo_turns(
    conversation_index: int, max_sessions: int, max_turns_per_session: int
):
    from fgl.data.locomo import load_conversations
    from fgl.paths import Paths, project_root

    data_file = Paths.build(project_root()).locomo_file
    if not data_file.exists():
        raise typer.BadParameter(
            f"LoCoMo dataset not found at {data_file} -- run `fgl setup` first"
        )
    conversations = load_conversations(data_file)
    conv = conversations[conversation_index]
    turns = []
    for session in conv.sessions[:max_sessions]:
        ts = datetime.fromisoformat(session.timestamp)
        for turn in session.turns[:max_turns_per_session]:
            turns.append((ts, turn.speaker, turn.text))
    return conv, turns


@clio_app.command()
def demo(
    fake: bool = typer.Option(
        False, "--fake", help="Use FakeLLM instead of a real deployment."
    ),
    question: list[str] = typer.Option(
        [], "--question", "-q", help="Ask this instead of the built-in demo questions."
    ),
    locomo: int = typer.Option(
        -1,
        "--locomo",
        help="Conversation index (0-9) to ingest from the real LoCoMo dataset "
        "instead of the built-in demo conversation.",
    ),
    sessions: int = typer.Option(
        2, "--sessions", help="Sessions to ingest, with --locomo."
    ),
    turns_per_session: int = typer.Option(
        6, "--turns-per-session", help="Turns per session to ingest, with --locomo."
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Force fresh LLM calls instead of reusing the on-disk cache.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="On a 0-proposition turn, print the raw LLM response that produced it.",
    ),
    debug_chars: int = typer.Option(
        0,
        "--debug-chars",
        help="Truncate the --debug raw response to this many characters. "
        "0 (the default) prints it whole -- the old 500-char cut made a "
        "complete response look truncated, which sends you hunting for a "
        "JSON error that is not there.",
    ),
) -> None:
    """Ingests a conversation, consolidates (with folding), and answers a
    few questions -- end to end, M5 through M8.

    Offline sanity check (no network, no cost):

        fgl clio demo --fake

    Against the real deployment in .env (the credentials `fgl doctor` uses).
    Cached like every other real command in this repo, so re-running with a
    bigger --sessions only pays for the NEW calls:

        fgl clio demo

    Against a real LoCoMo conversation instead of the built-in toy one:

        fgl clio demo --locomo 0 --sessions 3

    Every turn extracting 0 propositions is NOT normal on real dialogue --
    diagnose it with --debug --no-cache (the second flag forces a fresh
    call so you see what the model says now, not a replayed empty result):

        fgl clio demo --locomo 0 --sessions 2 --debug --no-cache
    """
    clio = _build_clio(fake, no_cache=no_cache)
    backend = "FakeLLM (offline)" if fake else clio.llm.cfg.deployment
    console.print(f"[bold]Backend:[/] {backend}")

    # Turn-level tallies, so the run ends with the one number that decides
    # whether a benchmark is worth paying for: of everything the extractor
    # produced, how much reached the graph, and where the rest went.
    tally = {
        "turns": 0,
        "silent_turns": 0,
        "raw": 0,
        "kept": 0,
        "unmapped": 0,
        "span_downgraded": 0,
    }
    suggestions: Counter[str] = Counter()
    rejections: Counter[str] = Counter()

    def _ingest_and_report(text: str, speaker: str, session_id: str, ts: datetime):
        result = clio.ingest(text, speaker=speaker, session_id=session_id, ts=ts)
        tally["turns"] += 1
        tally["raw"] += result.raw_count
        tally["kept"] += len(result.propositions)
        tally["unmapped"] += len(result.unmapped)
        tally["span_downgraded"] += result.span_downgrades
        if result.raw_count == 0:
            tally["silent_turns"] += 1
        for entry in result.unmapped:
            suggestions[entry.suggested_relation or "(unnamed)"] += 1
        for rejection in result.rejected:
            rejections[rejection.reason] += 1
        # A turn that produced nothing AT ALL is the only one whose raw
        # response is worth printing. A turn that produced items which
        # were then unmapped or rejected is explained by the counters
        # instead -- printing its JSON just buries them.
        if debug and not fake and result.raw_count == 0:
            raw = clio.llm.last_raw.get("text", "")
            shown = raw[:debug_chars] if debug_chars else raw
            console.print(f"    [red]raw response:[/] {shown!r}")
        return result

    def _report_extraction() -> None:
        console.print(
            f"\n[bold]Extraction:[/] {tally['turns']} turns, "
            f"{tally['raw']} item(s) proposed, {tally['kept']} kept, "
            f"{tally['unmapped']} unmapped, {sum(rejections.values())} rejected"
        )
        if tally["turns"]:
            silent = tally["silent_turns"] / tally["turns"]
            console.print(
                f"  [dim]{tally['silent_turns']}/{tally['turns']} turns "
                f"({silent:.0%}) produced nothing at all[/]"
            )
        if tally["span_downgraded"]:
            console.print(
                f"  [yellow]{tally['span_downgraded']} span(s) were not verbatim[/] "
                "-> downgraded to `contextual` = confidence 0.40, below "
                "tau_promote: each needs THREE independent episodes to "
                "reach the graph"
            )
        if rejections:
            t = Table(title="Rejected by reason (spec 6.5)")
            t.add_column("reason")
            t.add_column("n", justify="right")
            for reason, n in rejections.most_common():
                t.add_row(reason, str(n))
            console.print(t)
        if suggestions:
            t = Table(title="UNMAPPED: what the catalog could not express (spec 4.4)")
            t.add_column("suggested_relation")
            t.add_column("n", justify="right")
            for name, n in suggestions.most_common(15):
                t.add_row(name, str(n))
            console.print(t)
            if len(suggestions) > 15:
                console.print(f"  [dim]... and {len(suggestions) - 15} more[/]")

    if locomo >= 0:
        conv, turns = _load_locomo_turns(locomo, sessions, turns_per_session)
        console.print(
            f"[bold]Conversation:[/] {conv.sample_id} "
            f"({conv.speaker_a} & {conv.speaker_b}), {len(turns)} turns"
        )
        for ts, speaker, text in turns:
            result = _ingest_and_report(text, speaker, conv.sample_id, ts)
            console.print(
                f"[dim]{ts.date()}[/] {speaker}: {text[:70]}  ->  {result.summary()}"
            )
            if result.unmapped:
                console.print(
                    f"    [yellow]unmapped:[/] "
                    f"{[u.suggested_relation for u in result.unmapped]}"
                )
    else:
        console.print("[bold]Conversation:[/] built-in demo (Melanie)")
        for date, text in _DEMO_CONVERSATION:
            result = _ingest_and_report(
                text, "Melanie", "demo", datetime.strptime(date, "%Y-%m-%d")
            )
            console.print(f"[dim]{date}[/] {text}  ->  {result.summary()}")
            if result.unmapped:
                console.print(
                    f"  [yellow]unmapped:[/] "
                    f"{[u.suggested_relation for u in result.unmapped]}"
                )

    _report_extraction()

    report = clio.consolidate()
    console.print(
        f"\n[bold]Consolidated:[/] {len(report.applied)} edges touched, "
        f"{len(report.promoted)} promoted, {len(report.folded)} folded"
    )
    for rec in report.folded:
        kept = clio.graph.get_entity(rec.kept)
        console.print(
            f"  [green]folded[/] {rec.snapshot['canonical_name']!r} -> "
            f"{kept.canonical_name!r} (score={rec.score:.2f})"
        )

    t = Table(title="Graph state")
    for col in ("subject", "relation", "object", "valid", "r", "flags"):
        t.add_column(col)
    for e in clio.graph.all_edges():
        src = clio.graph.get_entity(e.src_id).canonical_name
        dst = clio.graph.get_entity(e.dst_id).canonical_name
        window = (
            f"{e.t_valid.start.date() if e.t_valid.start else '-inf'}.."
            f"{e.t_valid.end.date() if e.t_valid.end else 'now'}"
        )
        # NEG and UNANCHORED are the two states a plain valid-interval
        # column cannot show, and both change what the edge MEANS: a
        # denial is not a claim, and an undated fact is a volatility
        # default rather than something the conversation actually said.
        flags = " ".join(
            f
            for f, on in (
                ("NEG", not e.polarity),
                ("UNANCHORED", e.unanchored),
                ("CONFLICT", e.conflict_flag),
                ("RETRACTED", e.t_tx.end is not None),
            )
            if on
        )
        t.add_row(src, e.label, dst, window, str(e.reinforcement), flags)
    console.print(t)

    questions = question or (_DEMO_QUESTIONS if locomo < 0 else [])
    for q in questions:
        console.print(f"\n[bold cyan]Q:[/] {q}")
        trace = clio.ask(q)
        for step in trace.steps:
            console.print(f"  [dim]{step.action}({step.args}) -- {step.reason}[/]")
        console.print(f"[bold green]A:[/] {trace.answer}")

    if not fake:
        u = clio.llm.usage
        console.print(
            f"\n[dim]LLM usage: {u.calls} calls, {u.total_tokens} tokens, "
            f"{u.empty_responses} empty, {u.json_failures} JSON parse failures[/]"
        )
        if u.json_failures:
            console.print(
                f"[yellow]{u.json_failures} extraction responses did not parse as "
                "JSON and silently became 0 propositions each -- rerun with "
                "--debug --no-cache to see one raw response and find out why "
                "(markdown fences, a truncated/reasoning-truncated response, "
                "or a different schema than the prompt asked for).[/]"
            )


@clio_app.command()
def gate1(
    locomo: int = typer.Option(0, "--locomo", help="Conversation index (0-9)."),
    fake: bool = typer.Option(False, "--fake", help="Offline structural check only."),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Force fresh LLM calls instead of reusing the cache."
    ),
    out: str = typer.Option(
        "", "--out", help="Also write the report as JSON to this path."
    ),
) -> None:
    """Gate 1 -- extraction fidelity. Ingests one whole conversation and
    reports what fraction of the turns the OFFICIAL QUESTIONS cite as
    evidence actually produced a proposition.

    This is the number that decides whether benchmarking is worth paying
    for, and it costs about a third of what benchmarking the same
    conversation costs, because it makes zero question-answering calls.
    A question whose evidence turns produced nothing cannot be answered
    from the graph however good the access algebra is -- so measuring F1
    before this tells you a number without telling you which half of the
    system to fix.

        fgl clio gate1 --locomo 0
    """
    import json as _json

    from fgl.clio.config import ClioConfig
    from fgl.clio.gate import run_gate1
    from fgl.data.locomo import load_conversations
    from fgl.llm.prompts import PromptLibrary
    from fgl.paths import Paths, project_root

    paths = Paths.build(project_root())
    if not paths.locomo_file.exists():
        raise typer.BadParameter(
            f"LoCoMo dataset not found at {paths.locomo_file} -- run `fgl setup` first"
        )
    conv = load_conversations(paths.locomo_file)[locomo]
    llm, embedder = _build_backends(fake, no_cache)
    cfg = ClioConfig.default()
    console.print(
        f"[bold]Backend:[/] {'FakeLLM (offline)' if fake else llm.cfg.deployment}\n"
        f"[bold]Conversation:[/] {conv.sample_id} "
        f"({conv.speaker_a} & {conv.speaker_b}), {conv.n_turns} turns, "
        f"{len(conv.questions)} questions"
    )

    done = {"n": 0}

    def _on_turn(turn, result) -> None:
        done["n"] += 1
        if done["n"] % 25 == 0:
            console.print(f"  [dim]ingested {done['n']}/{conv.n_turns} turns[/]")

    report, _ = run_gate1(
        conv,
        load_catalog(cfg.catalog_path),
        llm,
        embedder,
        PromptLibrary(paths.prompts),
        cfg,
        on_turn=_on_turn,
    )

    console.print(
        f"\n[bold]Extraction:[/] {report.raw_items} proposed, "
        f"{report.kept_items} kept, {report.unmapped_items} unmapped, "
        f"{report.rejected_items} rejected  "
        f"({report.turns_with_propositions}/{report.n_turns} turns produced something, "
        f"{report.turns_fully_suppressed} proposed something that was wholly refused)"
    )
    if report.rejections_by_reason:
        t = Table(title="Rejected by reason")
        t.add_column("reason")
        t.add_column("n", justify="right")
        for reason, n in sorted(report.rejections_by_reason.items(), key=lambda kv: -kv[1]):
            t.add_row(reason, str(n))
        console.print(t)
    console.print(
        f"[bold]GATE 1 -- evidence turn coverage:[/] "
        f"[bold cyan]{report.turn_coverage:.1%}[/] "
        f"({report.evidence_turns_covered}/{report.evidence_turns} turns)  ·  "
        f"questions fully covered: {report.fully_covered_questions}/"
        f"{report.n_questions}"
    )
    silent = len(report.evidence_turns_silent)
    suppressed = len(report.evidence_turns_fully_suppressed)
    if silent or suppressed:
        console.print(
            f"  [dim]uncovered evidence turns: {silent} the model said nothing "
            f"about, {suppressed} it spoke about and the pipeline refused[/]"
        )
    if report.dangling_evidence:
        console.print(
            f"  [dim]{len(report.dangling_evidence)} evidence id(s) cited by a "
            "question but absent from the conversation -- excluded[/]"
        )

    t = Table(title="Coverage by question category")
    for col in ("category", "n", "full", "partial", "none", "turn coverage"):
        t.add_column(col)
    for cat in sorted(report.per_category.values(), key=lambda c: c.category):
        t.add_row(
            cat.name,
            str(cat.n_questions),
            str(cat.fully_covered),
            str(cat.partially_covered),
            str(cat.uncovered),
            f"{cat.turn_coverage:.1%}",
        )
    console.print(t)

    if report.uncovered_examples:
        console.print("\n[bold]Most-cited evidence turns that produced nothing:[/]")
        for dia_id, text in report.uncovered_examples:
            console.print(f"  [yellow]{dia_id}[/] {text[:110]}")

    if out:
        Path(out).write_text(
            _json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"\n[bold]Wrote:[/] {out}")

    if not fake:
        u = llm.usage
        console.print(
            f"[dim]LLM usage: {u.calls} calls ({u.cached_calls} served from "
            f"cache), {u.total_tokens} tokens, {u.json_failures} JSON parse "
            "failures[/]"
        )


@clio_app.command()
def bench(
    fake: bool = typer.Option(
        False,
        "--fake",
        help="Use FakeLLM instead of a real deployment (structural check only).",
    ),
    limit_conversations: int = typer.Option(
        0, "-n", "--limit-conversations", help="0 = all 10 LoCoMo conversations."
    ),
    limit_questions: int = typer.Option(
        0, "-q", "--limit-questions", help="Per conversation. 0 = all official questions."
    ),
    results_dir: str = typer.Option(
        "results", "--results-dir", help="Where <name>/metrics.json gets written."
    ),
    name: str = typer.Option(
        "CLIO", "--name", help="Condition/directory name -- what shows up in `fgl report`."
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Force fresh LLM calls instead of reusing the on-disk cache.",
    ),
) -> None:
    """Runs the full LoCoMo benchmark through CLIO (ingest -> consolidate
    -> fold -> answer every official question -> score with the same
    scorer every other condition uses) and writes
    results/<name>/{metrics.json,predictions.jsonl}. Pick it up with
    `fgl report` afterwards -- no separate registration needed, it is a
    plain directory scan.

    Free structural check (extraction script only knows the built-in demo
    conversation, so F1 will be near zero -- this proves the PLUMBING,
    not answer quality):

        fgl clio bench --fake -n 1 -q 5

    Small real run, one conversation, first 20 questions:

        fgl clio bench -n 1 -q 20

    The full benchmark (expensive: 10 conversations, 1986 questions, each
    needing several agent-loop calls plus one extraction call per turn):

        fgl clio bench

    Then:

        fgl report
    """
    from fgl.clio.evaluation import run_benchmark
    from fgl.paths import Paths, project_root

    # Loaded for real even under --fake: the point of --fake here is to
    # skip the LLM, not to fabricate conversation/session/turn counts too.
    data_file = Paths.build(project_root()).locomo_file
    if not data_file.exists():
        raise typer.BadParameter(
            f"LoCoMo dataset not found at {data_file} -- run `fgl setup` first"
        )

    llm, embedder = _build_backends(fake, no_cache)
    console.print(
        f"[bold]Backend:[/] {'FakeLLM (offline)' if fake else llm.cfg.deployment}"
    )

    running_f1: list[float] = []

    def _on_done(conv, result) -> None:
        conv_f1 = (
            sum(o.f1 for o in result.outcomes) / len(result.outcomes)
            if result.outcomes
            else 0.0
        )
        running_f1.append(conv_f1)
        console.print(
            f"[green]done[/] {conv.sample_id}: {result.n_turns} turns, "
            f"{result.n_questions} questions, f1={conv_f1:.4f} "
            f"({result.n_entities} entities, {result.n_folds} folds) "
            f"-- running mean f1={sum(running_f1) / len(running_f1):.4f}"
        )

    metrics = run_benchmark(
        data_file=data_file,
        llm=llm,
        embedder=embedder,
        limit_conversations=limit_conversations or None,
        limit_questions=limit_questions or None,
        results_dir=results_dir,
        condition_name=name,
        on_conversation_done=_on_done,
    )

    console.print(f"\n[bold]Wrote:[/] {Path(results_dir) / name}/metrics.json")
    t = Table(title=f"{name}: per-category F1")
    for col in ("category", "n", "f1", "abstention_rate"):
        t.add_column(col)
    for cat, row in metrics["per_category"].items():
        t.add_row(cat, str(row["n"]), f"{row['f1']:.4f}", f"{row['abstention_rate']:.4f}")
    console.print(t)
    o = metrics["overall"]
    console.print(
        f"[bold]overall:[/] n={o['n']}  f1_micro={o['f1_micro']:.4f}  "
        f"f1_macro={o['f1_macro']:.4f}  f1_substantive={o['f1_substantive']:.4f}  "
        f"abstention_rate={o['abstention_rate']:.4f}"
    )
    if not fake:
        u = metrics["cost"]
        console.print(
            f"[dim]LLM usage: {u['calls']} calls ({u['cached_calls']} cached), "
            f"{u['total_tokens']} tokens, {u['empty_responses']} empty[/]"
        )
    console.print("\nRun [bold]fgl report[/] to see it alongside the other conditions.")
