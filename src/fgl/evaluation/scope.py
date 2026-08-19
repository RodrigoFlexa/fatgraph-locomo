"""The method's scope conditions, measured against whatever corpus is loaded.

Why a module and not a paragraph
--------------------------------
A method with declared scope conditions is a method. A method with hidden ones
is a method fitted to a benchmark -- and the two can be technically identical.
The difference is entirely whether the assumptions are written down and whether
anyone can check them.

``docs/ASSUMPTIONS.md`` writes them down. This module makes them *runnable*: it
takes the loaded corpus and reports, per condition, the measured value, whether
the condition holds, and what the design falls back to when it does not. Point
it at LoCoMo and it reproduces the statistics the design was derived from;
point it at a new corpus and it tells you, before a single retrieval, which
parts of L2 are still standing.

Two kinds of check, and the distinction is load-bearing
-------------------------------------------------------
``runtime``  computable from the corpus a deployment would actually have --
             the transcripts, and at most the question texts. A runtime check
             can be run on unlabelled data, which means it can be run in
             production and not only in a paper.
``audit``    needs the gold evidence or the gold answers. These are the checks
             that produced the design in the first place, and they are exactly
             the ones a new corpus will not be able to run. They are kept, and
             kept labelled, because pretending the design did not come from
             them would be the dishonest version of this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from fgl.data.locomo import Conversation
from fgl.memory.slots import (
    KIND_EPISODE,
    LEGACY_QUESTION_NOUN_STOP,
    actor_key,
    granularity_of,
    match_actor,
    question_time_slots,
)

RUNTIME = "runtime"
AUDIT = "audit"


@dataclass
class Check:
    """One scope condition and its measurement on this corpus."""

    id: str
    statement: str
    kind: str  # RUNTIME | AUDIT
    measure: str
    criterion: str
    value: Any = None
    holds: Optional[bool] = None
    degrades_to: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "kind": self.kind,
            "measure": self.measure,
            "criterion": self.criterion,
            "value": self.value,
            "holds": self.holds,
            "degrades_to": self.degrades_to,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------- #
# The checks                                                                   #
# --------------------------------------------------------------------------- #


def check_participants(convs: Sequence[Conversation]) -> Check:
    """S1 -- how many people are in a conversation."""
    counts = []
    for conv in convs:
        speakers = {t.speaker for t in conv.turns()}
        counts.append(len(speakers))
    mean = sum(counts) / len(counts) if counts else 0.0
    return Check(
        id="S1",
        statement=(
            "The corpus is dialogue between a small, fixed set of named "
            "participants, so 'who said this' is a partition of the memory "
            "rather than a free-text attribute."
        ),
        kind=RUNTIME,
        measure="mean distinct speakers per conversation",
        criterion="<= 4 for the actor prior to be a strong partition",
        value=round(mean, 2),
        holds=bool(counts) and mean <= 4.0,
        degrades_to=(
            "With `slots.calibration=derived` the prior re-derives itself "
            "(floor = 1/n_speakers, full = median dominant-contributor share) "
            "instead of needing a re-sweep; above ~8 participants an episode "
            "stops being an adjacency pair and the episode segmenter, not the "
            "prior, is what needs rethinking."
        ),
        detail={"per_conversation": counts},
    )


def check_question_names_one_actor(convs: Sequence[Conversation]) -> Check:
    """S2 -- does a question identify exactly one participant?"""
    named_one = 0
    named_none = 0
    named_many = 0
    total = 0
    for conv in convs:
        keys = [k for k in (actor_key(conv.speaker_a), actor_key(conv.speaker_b)) if k]
        for q in conv.questions:
            total += 1
            found = set(match_actor(q.prompt_question(), keys))
            if len(found) == 1:
                named_one += 1
            elif not found:
                named_none += 1
            else:
                named_many += 1
    frac = named_one / total if total else 0.0
    return Check(
        id="S2",
        statement=(
            "A question identifies exactly one participant, so the actor slot "
            "of the query tuple is filled."
        ),
        kind=RUNTIME,
        measure="share of questions naming exactly one participant",
        criterion=">= 0.80 for the actor prior to be worth applying at all",
        value=round(frac, 4),
        holds=total > 0 and frac >= 0.80,
        degrades_to=(
            "The prior is already silent when no actor is linked (it is a "
            "multiplication that never happens), so a corpus failing this "
            "loses the channel rather than being harmed by it -- the same is "
            "true of the corner test, which abstains from abstaining when the "
            "question names nobody."
        ),
        detail={
            "n_questions": total,
            "named_one": named_one,
            "named_none": named_none,
            "named_multiple": named_many,
        },
    )


def check_evidence_belongs_to_named_actor(convs: Sequence[Conversation]) -> Check:
    """S3 -- AUDIT. Is the evidence the named participant's own turn?

    This is the measurement the actor prior was designed from (96-100% on
    LoCoMo) and it is exactly the one a new corpus cannot run, because it needs
    the gold evidence annotation. Reported as an audit, never as a runtime
    gate, so the dependence is visible rather than inherited silently.
    """
    hit = 0
    total = 0
    for conv in convs:
        keys = [k for k in (actor_key(conv.speaker_a), actor_key(conv.speaker_b)) if k]
        speaker_of = {t.dia_id: actor_key(t.speaker) for t in conv.turns()}
        for q in conv.questions:
            found = set(match_actor(q.prompt_question(), keys))
            if len(found) != 1 or not q.evidence:
                continue
            named = next(iter(found))
            total += 1
            if any(speaker_of.get(e) == named for e in q.evidence):
                hit += 1
    frac = hit / total if total else 0.0
    return Check(
        id="S3",
        statement=(
            "When a question names one participant, the evidence is that "
            "participant's own turn."
        ),
        kind=AUDIT,
        measure="share of single-actor questions whose evidence includes a turn by that actor",
        criterion=">= 0.90 (this is the statistic the multiplicative prior encodes)",
        value=round(frac, 4),
        holds=total > 0 and frac >= 0.90,
        degrades_to=(
            "`actor_prior_floor` is what keeps the residual: an episode the "
            "named person did not contribute to is demoted, never deleted. "
            "The derived floor (1/n_speakers) is the corpus-free way to set "
            "how much residual to keep."
        ),
        detail={"n_single_actor_questions": total, "n_with_actor_evidence": hit},
    )


def check_temporal_granularity(convs: Sequence[Conversation]) -> Check:
    """S4 -- at what grain do questions ask about time?"""
    counts = {"day": 0, "month": 0, "year": 0}
    with_time = 0
    total = 0
    for conv in convs:
        for q in conv.questions:
            total += 1
            slots = question_time_slots(q.prompt_question())
            if not slots:
                continue
            with_time += 1
            finest = granularity_of(slots[0])
            if finest in counts:
                counts[finest] += 1
    return Check(
        id="S4",
        statement=(
            "Time references in questions land at a grain the memory also "
            "indexes."
        ),
        kind=RUNTIME,
        measure="finest granularity named, over questions that name a date",
        criterion=(
            "no criterion -- this used to BE the parameter "
            "(`month`, chosen because LoCoMo asks by month) and is now "
            "measured for the record only"
        ),
        value=dict(counts),
        holds=True,
        degrades_to=(
            "Nothing to degrade: the granularity parameter was deleted. With "
            "`slots.time_granularities=year,month,day` every level is indexed, "
            "a question emits every level it names, and the degree damping "
            "decides which one carries the match. A corpus that asks by day or "
            "by year needs no change and no re-measurement."
        ),
        detail={"n_questions": total, "n_with_a_date": with_time},
    )


def check_question_template(
    convs: Sequence[Conversation], extractor=None, top_k: int = 12
) -> Check:
    """S5 -- how templated is the question generator?

    The check that explains the ugliest line of the original design. A hand
    stoplist containing "conversation", "date" and "type" is not a fact about
    English, it is a fingerprint of one generator. This measures the
    fingerprint: the most frequent nouns in the question corpus, and how many
    of them the legacy hand-written list already contains.
    """
    from fgl.memory.calibration import question_noun_frequencies

    questions = [q.prompt_question() for conv in convs for q in conv.questions]
    if extractor is None or not questions:
        return Check(
            id="S5", statement="", kind=RUNTIME, measure="", criterion="",
            value=None, holds=None,
            detail={"skipped": "needs a spaCy extractor and a question corpus"},
        )
    df = question_noun_frequencies(questions, extractor)
    top = sorted(df.items(), key=lambda kv: -kv[1])[:top_k]
    covered = sum(1 for w, _ in top if w in LEGACY_QUESTION_NOUN_STOP)
    head = top[0][1] if top else 0.0
    return Check(
        id="S5",
        statement=(
            "The question set is generated from a template, so some of its "
            "nouns are framing rather than content -- and the framing words "
            "are a property of the generator, not of the language."
        ),
        kind=RUNTIME,
        measure="document frequency of the commonest question noun",
        criterion=(
            ">= 0.05 means the set is templated enough that question-side "
            "filtering is doing real work"
        ),
        value=round(head, 4),
        holds=head >= 0.05,
        degrades_to=(
            "`slots.question_stop=derived` estimates the framing set from the "
            "question/memory frequency ratio instead of naming the words, so "
            "a different template is handled without editing a list. It is "
            "TRANSDUCTIVE -- it reads the text of the questions in advance -- "
            "which is fine for a benchmark and wrong for a one-question-at-a-"
            "time deployment; that deployment should use "
            "`question_stop=literal` or `none` and accept the loss, which "
            "`fgl slots-sweep` will price."
        ),
        detail={
            "n_questions": len(questions),
            "top_nouns": [
                {"word": w, "df": round(d, 4),
                 "in_legacy_stoplist": w in LEGACY_QUESTION_NOUN_STOP}
                for w, d in top
            ],
            "top_k_covered_by_legacy_list": covered,
        },
    )


def check_episode_structure(graphs: Sequence[Any]) -> Check:
    """S6 -- is the adjacency pair a real unit here, or an artefact?"""
    turns_per_ep: list[int] = []
    multi_speaker = 0
    total = 0
    for graph in graphs:
        for vx in graph.vertices.values():
            if vx.meta.get("kind") != KIND_EPISODE:
                continue
            total += 1
            turns_per_ep.append(len(vx.meta.get("turn_ids", [])))
            content = vx.meta.get("speaker_content", {}) or {}
            if sum(1 for n in content.values() if n > 0) >= 2:
                multi_speaker += 1
    mean = sum(turns_per_ep) / len(turns_per_ep) if turns_per_ep else 0.0
    frac = multi_speaker / total if total else 0.0
    return Check(
        id="S6",
        statement=(
            "The atomic memory is the adjacency pair: a reply carries the "
            "value and the turn above it carries the topic, so the two must "
            "share an index unit."
        ),
        kind=RUNTIME,
        measure="share of episodes containing content from at least two speakers",
        criterion=">= 0.60, otherwise the episode is just a turn with padding",
        value=round(frac, 4),
        holds=total > 0 and frac >= 0.60,
        degrades_to=(
            "On monologue-shaped corpora (documents, single-author notes) the "
            "episode collapses toward the turn and `sibling_frac` stops "
            "buying anything -- the model degrades to L1-with-typed-slots "
            "rather than breaking."
        ),
        detail={"n_episodes": total, "mean_turns_per_episode": round(mean, 2)},
    )


def check_degree_scale(graphs: Sequence[Any], cfg=None) -> Check:
    """S7 -- does an absolute hub cut-off still mean anything at this size?"""
    from fgl.memory.calibration import degrees_by_kind, hub_degree_by_quantile

    absolute = getattr(getattr(cfg, "slots", None), "hub_degree", 60)
    quantile = getattr(getattr(cfg, "slots", None), "hub_degree_quantile", 0.99)
    minimum = getattr(getattr(cfg, "slots", None), "hub_degree_min", 8)

    derived_all: dict[str, list[int]] = {}
    above: dict[str, list[float]] = {}
    for graph in graphs:
        thresholds, _ = hub_degree_by_quantile(graph, quantile, minimum, absolute)
        for kind, cut in thresholds.items():
            derived_all.setdefault(kind, []).append(cut)
        for kind, degs in degrees_by_kind(graph).items():
            if degs:
                above.setdefault(kind, []).append(
                    sum(1 for d in degs if d >= absolute) / len(degs)
                )

    mean_derived = {
        k: round(sum(v) / len(v), 1) for k, v in sorted(derived_all.items()) if v
    }
    mean_above = {
        k: round(sum(v) / len(v), 4) for k, v in sorted(above.items()) if v
    }
    # the absolute cut-off is "still meaningful" when it selects a small tail
    # rather than everything or nothing
    worst = max(mean_above.values(), default=0.0)
    return Check(
        id="S7",
        statement=(
            "A high-degree slot cannot discriminate, so it is treated as a "
            "filter rather than enumerated -- and 'high' has to be relative "
            "to this corpus."
        ),
        kind=RUNTIME,
        measure=(
            f"share of slots at or above the absolute cut-off "
            f"(slots.hub_degree = {absolute}), per kind"
        ),
        criterion="<= 0.10 for any kind, otherwise the absolute number is eating the graph",
        value=mean_above,
        holds=worst <= 0.10,
        degrades_to=(
            "`slots.calibration=derived` replaces the absolute count with a "
            "per-kind quantile of the degree distribution, which is scale-free "
            "by construction. Derived cut-offs on this corpus: "
            f"{mean_derived}."
        ),
        detail={"derived_hub_degree_by_kind": mean_derived,
                "absolute_hub_degree": absolute},
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def run_scope(
    convs: Sequence[Conversation],
    cfg=None,
    graphs: Optional[Sequence[Any]] = None,
    extractor=None,
) -> dict:
    """Every scope condition, measured on this corpus.

    ``graphs`` is optional: S6 and S7 need a built memory, and the corpus-only
    checks are the ones worth running first, before paying for an ingest.
    """
    checks = [
        check_participants(convs),
        check_question_names_one_actor(convs),
        check_evidence_belongs_to_named_actor(convs),
        check_temporal_granularity(convs),
        check_question_template(convs, extractor),
    ]
    if graphs:
        checks.append(check_episode_structure(graphs))
        checks.append(check_degree_scale(graphs, cfg))

    checks = [c for c in checks if c.statement or c.detail.get("skipped")]
    runtime = [c for c in checks if c.kind == RUNTIME and c.holds is not None]
    return {
        "n_conversations": len(convs),
        "n_questions": sum(len(c.questions) for c in convs),
        "condition": getattr(cfg, "condition", ""),
        "checks": [c.as_dict() for c in checks],
        "runtime_conditions_held": sum(1 for c in runtime if c.holds),
        "runtime_conditions_total": len(runtime),
    }


def format_scope(report: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"scope check · {report['n_conversations']} conversation(s) · "
        f"{report['n_questions']} questions"
        + (f" · {report['condition']}" if report["condition"] else "")
    )
    lines.append("")
    for c in report["checks"]:
        if c.get("detail", {}).get("skipped"):
            lines.append(f"{c['id']}  SKIPPED  {c['detail']['skipped']}")
            continue
        mark = {True: "HOLDS", False: "FAILS", None: "  --  "}[c["holds"]]
        tag = "audit (needs gold labels)" if c["kind"] == AUDIT else "runtime"
        lines.append(f"{c['id']}  [{mark}]  ({tag})")
        lines.append(f"    {c['statement']}")
        lines.append(f"    measured: {c['measure']} = {c['value']}")
        lines.append(f"    criterion: {c['criterion']}")
        lines.append(f"    if it fails: {c['degrades_to']}")
        lines.append("")
    lines.append(
        f"{report['runtime_conditions_held']}/{report['runtime_conditions_total']} "
        "runtime scope conditions hold on this corpus."
    )
    lines.append(
        "Audit conditions are listed for provenance: they are how the design "
        "was derived and they need annotations a new corpus will not have."
    )
    return "\n".join(lines)
