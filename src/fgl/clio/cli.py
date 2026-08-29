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

from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

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


def _build_clio(fake: bool):
    from fgl.clio.config import ClioConfig
    from fgl.clio.facade import Clio

    if fake:
        from fgl.clio.demo_fake import demo_fake_responder
        from fgl.config import LLMConfig
        from fgl.llm.client import FakeLLM
        from fgl.retrieval.embeddings import HashingEmbedder

        return Clio.build(
            config=ClioConfig.default(),
            llm=FakeLLM(LLMConfig(provider="fake"), responder=demo_fake_responder()),
            embedder=HashingEmbedder(dim=128),
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
    shim.llm.cache_enabled = False
    llm = build_llm(shim.llm)
    embedder = build_embedder(shim.embeddings)
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
) -> None:
    """Ingests a conversation, consolidates (with folding), and answers a
    few questions -- end to end, M5 through M8.

    Offline sanity check (no network, no cost):

        fgl clio demo --fake

    Against the real deployment in .env (the credentials `fgl doctor` uses):

        fgl clio demo

    Against a real LoCoMo conversation instead of the built-in toy one:

        fgl clio demo --locomo 0 --sessions 3
    """
    clio = _build_clio(fake)
    backend = "FakeLLM (offline)" if fake else clio.llm.cfg.deployment
    console.print(f"[bold]Backend:[/] {backend}")

    if locomo >= 0:
        conv, turns = _load_locomo_turns(locomo, sessions, turns_per_session)
        console.print(
            f"[bold]Conversation:[/] {conv.sample_id} "
            f"({conv.speaker_a} & {conv.speaker_b}), {len(turns)} turns"
        )
        for ts, speaker, text in turns:
            result = clio.ingest(text, speaker=speaker, session_id=conv.sample_id, ts=ts)
            console.print(
                f"[dim]{ts.date()}[/] {speaker}: {text[:70]}  "
                f"->  {len(result.propositions)} proposition(s)"
            )
    else:
        console.print("[bold]Conversation:[/] built-in demo (Melanie)")
        for date, text in _DEMO_CONVERSATION:
            result = clio.ingest(
                text,
                speaker="Melanie",
                session_id="demo",
                ts=datetime.strptime(date, "%Y-%m-%d"),
            )
            console.print(
                f"[dim]{date}[/] {text}  ->  {len(result.propositions)} proposition(s)"
            )
            if result.unmapped:
                console.print(
                    f"  [yellow]unmapped:[/] "
                    f"{[u.suggested_relation for u in result.unmapped]}"
                )

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
    for col in ("subject", "relation", "object", "valid", "conflict"):
        t.add_column(col)
    for e in clio.graph.all_edges():
        src = clio.graph.get_entity(e.src_id).canonical_name
        dst = clio.graph.get_entity(e.dst_id).canonical_name
        window = (
            f"{e.t_valid.start.date() if e.t_valid.start else '-inf'}.."
            f"{e.t_valid.end.date() if e.t_valid.end else 'now'}"
        )
        t.add_row(src, e.label, dst, window, str(e.conflict_flag))
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
            f"{u.empty_responses} empty[/]"
        )
