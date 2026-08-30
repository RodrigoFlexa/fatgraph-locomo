"""The access ceiling: three nested bounds that say WHERE a question is
lost, measured with zero LLM calls against a memory loaded from disk.

The first benchmark put extraction coverage at 82% and answer quality at
f1 0.26, which says the loss is downstream of extraction but not where.
"Downstream" is three different places, and each one bounds the next:

1. **extraction** -- the evidence turn produced at least one proposition
   that survived validation. This is what ``fgl clio gate1`` measures, and
   it deliberately counts STAGED propositions.
2. **promotion** -- that proposition actually reached the graph, i.e. it
   appears in some edge's provenance. A proposition below ``tau_promote``
   (an implicature at 0.55, anything downgraded to contextual at 0.40)
   sits in staging and is invisible to the reader: ``follow`` walks edges,
   and ``evidence`` resolves an edge's provenance. Gate 1 explicitly did
   not measure this and called it a separate gate. This is that gate.
3. **reachability** -- the edge is reachable from where ``anchor`` puts
   the agent for THIS question, within the movement budget. A fact in the
   graph that no path from the entry point reaches is unanswerable no
   matter how well the agent chooses.

Read together they localise the loss without a single model call:
promotion far below extraction means the confidence thresholds are eating
the memory; reachability far below promotion means the graph is
disconnected or ``anchor`` lands in the wrong place; both healthy while
F1 stays low means the agent's CHOICES or the answer prompt are the
problem, not the representation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from fgl.clio.access.movements import anchor
from fgl.data.locomo import CATEGORY_NAMES, Conversation


@dataclass
class CeilingRow:
    category: int
    n_questions: int = 0
    evidence_turns: int = 0
    extracted: int = 0
    promoted: int = 0
    reachable: int = 0
    episodic: int = 0
    combined: int = 0
    #: questions all of whose evidence turns clear each bar
    full_extracted: int = 0
    full_promoted: int = 0
    full_reachable: int = 0
    full_episodic: int = 0
    full_combined: int = 0

    @property
    def name(self) -> str:
        return CATEGORY_NAMES.get(self.category, str(self.category))


@dataclass
class CeilingReport:
    sample_id: str
    per_category: dict[int, CeilingRow] = field(default_factory=dict)
    n_edges: int = 0
    n_entities: int = 0
    n_propositions: int = 0
    n_promoted_propositions: int = 0
    #: evidence turns that were extracted but never promoted, with text --
    #: the concrete cost of the confidence thresholds
    lost_to_promotion: list[tuple[str, str]] = field(default_factory=list)
    #: promoted but unreachable from their own question's anchor
    lost_to_reach: list[tuple[str, str]] = field(default_factory=list)

    def _sum(self, attr: str) -> int:
        return sum(getattr(r, attr) for r in self.per_category.values())

    def to_dict(self) -> dict:
        turns = self._sum("evidence_turns") or 1
        questions = self._sum("n_questions") or 1
        return {
            "sample_id": self.sample_id,
            "graph": {
                "entities": self.n_entities,
                "edges": self.n_edges,
                "propositions": self.n_propositions,
                "promoted_propositions": self.n_promoted_propositions,
            },
            "evidence_turns": self._sum("evidence_turns"),
            "ceilings": {
                "extraction": round(self._sum("extracted") / turns, 4),
                "promotion": round(self._sum("promoted") / turns, 4),
                "reachability": round(self._sum("reachable") / turns, 4),
            },
            "question_ceilings": {
                "extraction": round(self._sum("full_extracted") / questions, 4),
                "promotion": round(self._sum("full_promoted") / questions, 4),
                "reachability": round(self._sum("full_reachable") / questions, 4),
            },
            "episodic_anchor": {
                "evidence_turns": round(self._sum("episodic") / turns, 4),
                "questions_fully": round(self._sum("full_episodic") / questions, 4),
            },
            "combined_access": {
                "evidence_turns": round(self._sum("combined") / turns, 4),
                "questions_fully": round(self._sum("full_combined") / questions, 4),
            },
            "per_category": {
                r.name: {
                    "n": r.n_questions,
                    "evidence_turns": r.evidence_turns,
                    "extraction": round(r.extracted / (r.evidence_turns or 1), 4),
                    "promotion": round(r.promoted / (r.evidence_turns or 1), 4),
                    "reachability": round(r.reachable / (r.evidence_turns or 1), 4),
                    "episodic_anchor": round(r.episodic / (r.evidence_turns or 1), 4),
                    "combined_access": round(r.combined / (r.evidence_turns or 1), 4),
                    "questions_fully_reachable": r.full_reachable,
                }
                for r in sorted(self.per_category.values(), key=lambda r: r.category)
            },
        }


def _episodes_of(clio, proposition_ids) -> set[str]:
    out: set[str] = set()
    for pid in proposition_ids:
        try:
            out.add(clio.staging.get(pid).episode_id)
        except KeyError:
            continue
    return out


def reachable_episodes(clio, seed_vertex_ids, max_hops: int) -> set[str]:
    """Every episode the reader could materialise from these entry points
    within ``max_hops`` traversals.

    An optimistic bound on purpose: it walks edges in BOTH directions and
    ignores which label the agent would have had to name, because the
    question being answered here is whether a perfect agent COULD get
    there -- not whether this one did. Temporal windows are likewise not
    narrowed: a tighter walk would measure the agent's competence again
    instead of the graph's capacity.
    """
    seen_vertices = set(seed_vertex_ids)
    frontier = deque((v, 0) for v in seed_vertex_ids)
    episodes: set[str] = set()
    while frontier:
        vertex, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for edge in clio.graph.edges_incident(vertex, live_only=True):
            episodes |= _episodes_of(clio, edge.provenance)
            other = edge.dst_id if edge.src_id == vertex else edge.src_id
            if other not in seen_vertices:
                seen_vertices.add(other)
                frontier.append((other, depth + 1))
    return episodes


def measure_ceilings(
    conv: Conversation, clio, max_hops: int = 6, max_examples: int = 10
) -> CeilingReport:
    # `anchor` is only as good as the index behind it, and a memory that
    # was just built (rather than loaded) has never had one rebuilt --
    # Clio.ask() does it, nothing else does. Without this the reachability
    # ceiling reads 0% for a perfectly reachable graph, which is a lie in
    # the direction that would have sent the next fix to the wrong place.
    clio.entity_index.rebuild(clio.graph)
    clio.episode_index.rebuild(clio.log)

    report = CeilingReport(sample_id=conv.sample_id)
    report.n_edges = len(clio.graph.all_edges())
    report.n_entities = len([e for e in clio.graph.all_entities() if e.merged_into is None])

    propositions = clio.staging.all()
    report.n_propositions = len(propositions)

    # which episodes produced a proposition at all, and which produced one
    # that actually reached an edge
    extracted_episodes = {p.episode_id for p in propositions}
    promoted_ids: set[str] = set()
    for edge in clio.graph.all_edges():
        promoted_ids.update(edge.provenance)
    report.n_promoted_propositions = len(promoted_ids)
    promoted_episodes = _episodes_of(clio, promoted_ids)

    by_dia = {ep.id for ep in clio.log.all()}

    for q in conv.questions:
        if q.is_adversarial or not q.evidence:
            continue
        row = report.per_category.setdefault(q.category, CeilingRow(q.category))
        row.n_questions += 1

        state = anchor(
            q.prompt_question(),
            clio.graph,
            index=clio.entity_index,
            episode_index=clio.episode_index,
            k=5,
            episode_k=clio.config.access.anchor_episode_k,
        )
        episodic = set(state.candidate_episode_ids)
        reach = reachable_episodes(
            clio, [t.vertex_id for t in state.trails], max_hops=max_hops
        )

        hits = {
            "extracted": 0,
            "promoted": 0,
            "reachable": 0,
            "episodic": 0,
            "combined": 0,
        }
        total = 0
        for dia_id in q.evidence:
            if dia_id not in by_dia:
                continue
            total += 1
            row.evidence_turns += 1
            if dia_id in extracted_episodes:
                hits["extracted"] += 1
                row.extracted += 1
            if dia_id in promoted_episodes:
                hits["promoted"] += 1
                row.promoted += 1
            elif (
                dia_id in extracted_episodes
                and len(report.lost_to_promotion) < max_examples
            ):
                turn = conv.turn_by_id(dia_id)
                if turn is not None:
                    report.lost_to_promotion.append(
                        (dia_id, f"{turn.speaker}: {turn.text}")
                    )
            if dia_id in reach:
                hits["reachable"] += 1
                row.reachable += 1
            elif dia_id in promoted_episodes and len(report.lost_to_reach) < max_examples:
                turn = conv.turn_by_id(dia_id)
                if turn is not None:
                    report.lost_to_reach.append((dia_id, f"{turn.speaker}: {turn.text}"))
            if dia_id in episodic:
                hits["episodic"] += 1
                row.episodic += 1
            if dia_id in reach or dia_id in episodic:
                hits["combined"] += 1
                row.combined += 1

        if total and hits["extracted"] == total:
            row.full_extracted += 1
        if total and hits["promoted"] == total:
            row.full_promoted += 1
        if total and hits["reachable"] == total:
            row.full_reachable += 1
        if total and hits["episodic"] == total:
            row.full_episodic += 1
        if total and hits["combined"] == total:
            row.full_combined += 1

    return report


__all__ = ["CeilingReport", "CeilingRow", "measure_ceilings", "reachable_episodes"]
