"""How far the evidence actually is -- the measurement that decides L3.

Why this exists before L3 and not after
----------------------------------------
L1's error analysis produced one number that reframed the whole project: of the
evidence turns L1 failed to retrieve, only 5-13% shared even one entity vertex
with the question, so 87-95% of the misses were turns the graph offered **no
path to at all**. That number is what justified typed slots -- it said the
problem was reachability, not ranking.

This module asks the same question one hop further out, and it is the honest
gate on the propagation condition. A longer walk can only find evidence that is
*reachable in that many hops*. So before spending a run on L3::

    hop 1        a query slot is incident to the evidence's episode
    hop 2        the evidence's episode shares a slot with a hop-1 episode
    hop 3        one more
    unreachable  no path within the bound, hubs excluded as bridges

If the misses are mostly at hop 2, L3 has a target and the size of that bucket
is its ceiling. If they are unreachable, no walk on this graph will find them
and the answer is a different ingest, not a different read. Either way the
number costs nothing: no LLM, no answering, one BFS per question.

The quadrangulation numbers
---------------------------
Reported alongside, because they are the quantitative version of the
ribbon-graph argument and they take one pass over the graph.

For a bipartite graph every face of an embedding has length >= 4, so
``F <= E/2``, and Euler (``F = 2C - 2g + E - V``) turns that into a floor on
the genus. Comparing the *achieved* F against that ceiling says how much of the
ribbon structure the current rotation is using. On the measured L2 graphs it is
about 0.3% -- 108 faces against a ceiling near 34,000, average face length over
a thousand half-edges -- which is the precise sense in which "the faces carry no
information" is a fact rather than an impression.

And the object a minimum-genus (quadrangular) rotation would make into a face
is exactly ``e1 - s_a - e2 - s_b - e1``: a pair of episodes agreeing on two
different slots. That pair is counted here directly. If the count is large and
the hop-2 bucket is empty, the theory is pointing at something the data does not
need; if both are large, the walk in :mod:`fgl.retrieval.propagation` is the
cheap way to use it and the rotation is the expensive one.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

from fgl.data.locomo import CATEGORY_NAMES, Conversation
from fgl.memory.slots import KIND_EPISODE

#: Hops we distinguish. Beyond this everything is "unreachable" -- and on a
#: graph this cycle-rich, a fourth hop reaches most of the corpus anyway, which
#: is a statement about the graph rather than about the question.
MAX_HOPS = 3


@dataclass
class _Bucket:
    n: int = 0
    by_hop: dict[int, int] = field(default_factory=dict)
    unreachable: int = 0

    def add(self, hop: Optional[int]) -> None:
        self.n += 1
        if hop is None:
            self.unreachable += 1
        else:
            self.by_hop[hop] = self.by_hop.get(hop, 0) + 1

    def as_dict(self) -> dict:
        n = max(self.n, 1)
        return {
            "n": self.n,
            "share_by_hop": {
                h: round(self.by_hop.get(h, 0) / n, 4)
                for h in range(1, MAX_HOPS + 1)
            },
            "share_unreachable": round(self.unreachable / n, 4),
        }


def episode_hops(
    graph, seeds: Sequence[str], is_hub, max_hops: int = MAX_HOPS
) -> dict[str, int]:
    """Hop at which each episode is first reached from ``seeds`` (slot vertices).

    A hop is an *arrival at an episode*, matching how
    :func:`fgl.retrieval.propagation.propagate` counts them, so this diagnostic
    and that operator agree on what "hop 2" means. Hubs may be seeds and may be
    arrived at, but never relayed through -- the same rule, so the profile
    measures the graph the walk will actually see rather than an idealised one.
    """
    eps_of_slot: dict[str, list[str]] = {}
    slots_of_ep: dict[str, list[str]] = {}
    for hid, he in graph.H.items():
        vid = he.vertex_id
        other = graph.H[graph.alpha[hid]].vertex_id
        if graph.vertices[vid].meta.get("kind") != KIND_EPISODE:
            continue
        eps_of_slot.setdefault(other, []).append(vid)
        if not is_hub(other):
            slots_of_ep.setdefault(vid, []).append(other)

    reached: dict[str, int] = {}
    frontier_slots = [s for s in dict.fromkeys(seeds) if s in eps_of_slot]
    seen_slots = set(frontier_slots)
    for hop in range(1, max_hops + 1):
        next_slots: list[str] = []
        new_eps: list[str] = []
        for slot in frontier_slots:
            for ep in eps_of_slot.get(slot, ()):
                if ep not in reached:
                    reached[ep] = hop
                    new_eps.append(ep)
        if hop == max_hops:
            break
        for ep in new_eps:
            for s in slots_of_ep.get(ep, ()):
                if s not in seen_slots:
                    seen_slots.add(s)
                    next_slots.append(s)
        if not next_slots:
            break
        frontier_slots = next_slots
    return reached


def quadrangulation_stats(graph, is_hub) -> dict:
    """How much of the ribbon structure the current rotation is using.

    Euler with ``C`` components: ``V - E + F = 2C - 2g``. For a bipartite graph
    every face has length >= 4, hence ``F <= E/2``, hence a floor on ``g``. The
    gap between the achieved ``F`` and that ceiling is the measurement.
    """
    stats = graph.stats()
    V, E, F, C = stats["V"], stats["E"], stats["F"], stats["C"]
    f_ceiling = E // 2  # bipartite: every face has length >= 4
    # F = 2C - 2g + E - V  =>  g = (2C + E - V - F) / 2
    g_now = (2 * C + E - V - F) / 2.0
    g_floor = max(0.0, (2 * C + E - V - f_ceiling) / 2.0)

    # the object a quadrangular rotation would turn into a face: two episodes
    # agreeing on two different non-hub slots
    eps_of_slot: dict[str, list[str]] = {}
    for hid, he in graph.H.items():
        vid = he.vertex_id
        other = graph.H[graph.alpha[hid]].vertex_id
        if graph.vertices[vid].meta.get("kind") != KIND_EPISODE:
            continue
        if not is_hub(other):
            eps_of_slot.setdefault(other, []).append(vid)
    shared: dict[tuple[str, str], int] = {}
    for eps in eps_of_slot.values():
        uniq = sorted(set(eps))
        # a slot on many episodes contributes quadratically and is exactly the
        # kind of slot that means nothing; the hub filter above already removed
        # the worst, and this bound keeps the count from being dominated by
        # what is left
        if len(uniq) > 64:
            continue
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                key = (uniq[i], uniq[j])
                shared[key] = shared.get(key, 0) + 1
    n_pairs_2plus = sum(1 for v in shared.values() if v >= 2)

    return {
        "V": V, "E": E, "F": F, "C": C,
        "genus": int(g_now),
        "mean_face_length": round(2 * E / max(F, 1), 1),
        "max_face_length": stats.get("face_length_hist") and max(
            int(k) for k in stats["face_length_hist"]
        ) or None,
        "faces_ceiling_bipartite": f_ceiling,
        "faces_used_frac": round(F / max(f_ceiling, 1), 5),
        "genus_floor_bipartite": int(g_floor),
        "genus_over_floor": round(g_now / max(g_floor, 1.0), 2),
        "episode_pairs_sharing_2plus_slots": n_pairs_2plus,
    }


def run_hop_profile(
    condition: str,
    conversations: Sequence[Conversation],
    root=None,
    force_ingest: bool = False,
    progress=None,
) -> dict:
    """Where the annotated evidence sits, in hops, and what L2 already got.

    Two profiles, and the second is the one that matters:

    ``all_evidence``   every annotated evidence turn;
    ``missed``         only the evidence turns the condition's retriever did
                       NOT put in the prompt. This is the headroom -- the share
                       of it at hop 2 is what a longer walk can address, and
                       the share unreachable is what it cannot.
    """
    from fgl.evaluation.slots_oracle import run_oracle  # noqa: F401  (doc link)
    from fgl.config import Config
    from fgl.llm import build_llm
    from fgl.pipeline import Runner, _RETRIEVERS, _build_retriever

    say = progress or (lambda *a: None)
    cfg = Config.load(condition, root=root)
    cfg.llm.provider = "fake"
    cfg.llm.cache_enabled = False
    runner = Runner(cfg, root=root, llm=build_llm(cfg.llm))
    retriever_cls = _RETRIEVERS[cfg.retrieval.mode]

    all_ev: dict[str, _Bucket] = {c: _Bucket() for c in CATEGORY_NAMES.values()}
    missed: dict[str, _Bucket] = {c: _Bucket() for c in CATEGORY_NAMES.values()}
    graph_stats: list[dict] = []

    for i, conv in enumerate(conversations):
        say(condition, i, len(conversations), conv.sample_id)
        graph, _ = runner._ingest(conv, force=force_ingest)  # noqa: SLF001
        retriever = _build_retriever(
            retriever_cls, graph, runner.embedder, cfg,
            {s.id: s.date_time_raw for s in conv.sessions}, conv,
        )
        graph_stats.append(quadrangulation_stats(graph, retriever.is_hub))

        episode_of_turn = {}
        for vid, vx in graph.vertices.items():
            if vx.meta.get("kind") != KIND_EPISODE:
                continue
            for t in vx.meta.get("turn_ids", ()):
                episode_of_turn[t] = vid

        for q in conv.questions:
            if not q.evidence:
                continue
            result = retriever.retrieve(q.prompt_question())
            got = set(result.turn_ids)
            slots = retriever.parse_question(q.prompt_question())
            seeds = []
            for kind, key in slots.as_pairs():
                vid = retriever._resolve_slot(kind, key)  # noqa: SLF001
                if vid is not None:
                    seeds.append(vid)
            hops = episode_hops(graph, seeds, retriever.is_hub)
            cat = q.category_name
            for turn in q.evidence:
                ep = episode_of_turn.get(turn)
                hop = hops.get(ep) if ep else None
                all_ev[cat].add(hop)
                if turn not in got:
                    missed[cat].add(hop)

    def merge(stats: list[dict]) -> dict:
        if not stats:
            return {}
        out: dict = {}
        for k in stats[0]:
            vals = [s[k] for s in stats if isinstance(s.get(k), (int, float))]
            if vals:
                out[k] = round(sum(vals) / len(vals), 4) if isinstance(
                    vals[0], float
                ) else int(sum(vals))
        # shares are averages, not sums
        for k in ("faces_used_frac", "genus_over_floor", "mean_face_length"):
            vals = [s[k] for s in stats if isinstance(s.get(k), (int, float))]
            if vals:
                out[k] = round(sum(vals) / len(vals), 5)
        return out

    return {
        "condition": cfg.condition,
        "n_conversations": len(conversations),
        "max_hops": MAX_HOPS,
        "all_evidence": {k: v.as_dict() for k, v in all_ev.items() if v.n},
        "missed": {k: v.as_dict() for k, v in missed.items() if v.n},
        "graph": merge(graph_stats),
        "note": (
            "`missed` is the headroom: the share at hop 2 is what a longer "
            "walk can reach, the share unreachable is what it cannot and what "
            "would need a different ingest instead."
        ),
    }


def format_hop_profile(report: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"hop profile · {report['condition']} · "
        f"{report['n_conversations']} conversation(s) · zero LLM calls"
    )
    for title, key in (("all annotated evidence", "all_evidence"),
                       ("evidence this condition MISSED (the headroom)", "missed")):
        block = report.get(key) or {}
        if not block:
            continue
        lines.append("")
        lines.append(title)
        lines.append(
            f"{'category':<14}{'n':>7}" +
            "".join(f"{'hop ' + str(h):>9}" for h in range(1, report["max_hops"] + 1))
            + f"{'unreach':>10}"
        )
        for cat, b in block.items():
            row = f"{cat:<14}{b['n']:>7}"
            for h in range(1, report["max_hops"] + 1):
                row += f"{b['share_by_hop'].get(h, b['share_by_hop'].get(str(h), 0.0)):>9.3f}"
            row += f"{b['share_unreachable']:>10.3f}"
            lines.append(row)

    g = report.get("graph") or {}
    if g:
        lines.append("")
        lines.append("ribbon structure (how much of it the rotation is using)")
        lines.append(
            f"  V={g.get('V')}  E={g.get('E')}  F={g.get('F')}  "
            f"genus={g.get('genus')}"
        )
        lines.append(
            f"  faces used: {g.get('F')} of a bipartite ceiling of "
            f"{g.get('faces_ceiling_bipartite')} "
            f"({g.get('faces_used_frac', 0) * 100:.2f}%)"
        )
        lines.append(
            f"  genus {g.get('genus')} against a floor of "
            f"{g.get('genus_floor_bipartite')} "
            f"({g.get('genus_over_floor')}x)"
        )
        lines.append(
            f"  mean face length {g.get('mean_face_length')} half-edges"
        )
        lines.append(
            f"  episode pairs sharing 2+ non-hub slots (the 4-cycles a "
            f"quadrangular rotation would make faces): "
            f"{g.get('episode_pairs_sharing_2plus_slots')}"
        )
    lines.append("")
    lines.append("  " + report.get("note", ""))
    return "\n".join(lines)
