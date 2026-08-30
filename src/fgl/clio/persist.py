"""Saving and loading a whole CLIO memory (spec 12.1's persistence, in the
one shape this project actually needs today).

Not the SQL schema the spec sketches: a single JSON document holding every
store. The reason is what persistence is FOR here, and it is not
production durability -- it is making the reading side debuggable. Every
access experiment currently costs re-ingesting a whole conversation (419
turns, ~6 minutes even with the extraction cache warm), which means any
change to the access algebra or the answer prompt is a guess measured at
3.3M tokens a try. With this, a memory is built once and every subsequent
experiment starts from disk with zero LLM calls.

Round-trips exactly what the graph needs to answer: episodes,
propositions (promoted or still staged), entities including folded-away
aliases, bitemporal edges with polarity and the unanchored flag, mentions
with their links, the fold journal, and the unmapped queue. The id
counters ride along too, so a memory loaded from disk keeps minting ids
where it left off instead of colliding with what is already there.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fgl.clio.catalog import Catalog
from fgl.clio.config import ClioConfig
from fgl.clio.consolidate.journal import FoldRecord
from fgl.clio.types import (
    Edge,
    Entity,
    EvidenceKind,
    Interval,
    Mention,
    Operation,
    Proposition,
)

FORMAT_VERSION = 1


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _undt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _interval(iv: Interval | None) -> dict | None:
    if iv is None:
        return None
    return {"start": _dt(iv.start), "end": _dt(iv.end), "granularity": iv.granularity}


def _uninterval(raw: dict | None) -> Interval | None:
    if raw is None:
        return None
    return Interval(_undt(raw["start"]), _undt(raw["end"]), raw.get("granularity"))


def dump_memory(clio) -> dict[str, Any]:
    """The whole memory as a plain dict. Separate from :func:`save_memory`
    so a caller can embed it in a bigger document or diff two memories."""
    return {
        "format_version": FORMAT_VERSION,
        "episodes": [
            {
                "id": e.id,
                "session_id": e.session_id,
                "speaker": e.speaker,
                "text": e.text,
                "ts_ingest": _dt(e.ts_ingest),
                "seq": e.seq,
                "meta": e.meta,
            }
            for e in clio.log.all()
        ],
        "entities": [
            {
                "id": e.id,
                "canonical_name": e.canonical_name,
                "type": e.type,
                "aliases": e.aliases,
                "created_from": e.created_from,
                "merged_into": e.merged_into,
                "provisional": e.provisional,
            }
            for e in clio.graph.all_entities()
        ],
        "edges": [
            {
                "id": e.id,
                "src_id": e.src_id,
                "label": e.label,
                "dst_id": e.dst_id,
                "t_valid": _interval(e.t_valid),
                "t_tx": _interval(e.t_tx),
                "provenance": e.provenance,
                "reinforcement": e.reinforcement,
                "last_confirmed": _dt(e.last_confirmed),
                "confidence": e.confidence,
                "conflict_flag": e.conflict_flag,
                "polarity": e.polarity,
                "unanchored": e.unanchored,
            }
            for e in clio.graph.all_edges()
        ],
        "propositions": [
            {
                "id": p.id,
                "subject_id": p.subject_id,
                "relation": p.relation,
                "object_id": p.object_id,
                "operation": p.operation.value,
                "polarity": p.polarity,
                "time_expression": p.time_expression,
                "t_valid": _interval(p.t_valid),
                "t_tx": _interval(p.t_tx),
                "evidence_kind": p.evidence_kind.value,
                "confidence": p.confidence,
                "span": p.span,
                "episode_id": p.episode_id,
                "status": p.status,
                "unanchored": p.unanchored,
                "subject_ref": p.subject_ref,
                "object_ref": p.object_ref,
            }
            for p in clio.staging.all()
        ],
        "mentions": [
            {
                "id": m.id,
                "episode_id": m.episode_id,
                "entity_id": m.entity_id,
                "surface": m.surface,
                "ts": _dt(m.ts),
                "proposition_id": m.proposition_id,
            }
            for m in clio.mentions.all()
        ],
        "fold_journal": [
            {
                "id": r.id,
                "kept": r.kept,
                "absorbed": r.absorbed,
                "score": r.score,
                "trigger_episode": r.trigger_episode,
                "migrated_edge_ids": r.migrated_edge_ids,
                "snapshot": r.snapshot,
                "created_at": _dt(r.created_at),
                "reverted": r.reverted,
            }
            for r in clio.journal.all()
        ],
        "unmapped": [
            {
                "id": u.id,
                "suggested_relation": u.suggested_relation,
                "subject_ref": u.subject_ref,
                "object_ref": u.object_ref,
                "span": u.span,
                "episode_id": u.episode_id,
            }
            for u in clio.unmapped.all()
        ],
    }


def save_memory(clio, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dump_memory(clio), ensure_ascii=False), encoding="utf-8")
    return p


def _next_seq(ids: list[str], prefix: str) -> int:
    """One past the highest numeric suffix seen, so a loaded store keeps
    minting fresh ids instead of colliding with what it just read."""
    highest = -1
    for i in ids:
        if not i.startswith(prefix):
            continue
        try:
            highest = max(highest, int(i[len(prefix) :]))
        except ValueError:
            continue
    return highest + 1


def load_memory(
    path: str | Path,
    catalog: Catalog,
    llm=None,
    embedder=None,
    prompts=None,
    config: ClioConfig | None = None,
):
    """Rebuilds a :class:`~fgl.clio.facade.Clio` from disk.

    ``llm`` and ``prompts`` may be ``None`` for a read-only session -- the
    access algebra never calls a model, and that is the whole point of
    being able to load one of these.
    """
    from fgl.clio.facade import Clio
    from fgl.retrieval.embeddings import HashingEmbedder

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"memory format version {raw.get('format_version')!r} != {FORMAT_VERSION}"
        )

    clio = Clio(
        catalog,
        llm,
        embedder or HashingEmbedder(dim=256),
        prompts,
        config or ClioConfig.default(),
    )

    for e in raw["episodes"]:
        clio.log.append(
            session_id=e["session_id"],
            speaker=e["speaker"],
            text=e["text"],
            ts_ingest=_undt(e["ts_ingest"]),
            meta=e.get("meta"),
            episode_id=e["id"],
        )

    graph = clio.graph
    for e in raw["entities"]:
        graph._entities[e["id"]] = Entity(
            id=e["id"],
            canonical_name=e["canonical_name"],
            type=e["type"],
            aliases=list(e["aliases"]),
            created_from=e.get("created_from", ""),
            merged_into=e.get("merged_into"),
            provisional=e.get("provisional", False),
        )
    graph._next_entity_seq = _next_seq(list(graph._entities), "ent_")

    for e in raw["edges"]:
        edge = Edge(
            id=e["id"],
            src_id=e["src_id"],
            label=e["label"],
            dst_id=e["dst_id"],
            t_valid=_uninterval(e["t_valid"]),
            t_tx=_uninterval(e["t_tx"]),
            provenance=list(e["provenance"]),
            reinforcement=e["reinforcement"],
            last_confirmed=_undt(e["last_confirmed"]),
            confidence=e["confidence"],
            conflict_flag=e["conflict_flag"],
            polarity=e.get("polarity", True),
            unanchored=e.get("unanchored", False),
        )
        graph._edges[edge.id] = edge
        graph._incident[edge.src_id].add(edge.id)
        graph._incident[edge.dst_id].add(edge.id)
    graph._next_edge_seq = _next_seq(list(graph._edges), "edg_")

    clio.staging.insert(
        Proposition(
            id=p["id"],
            subject_id=p["subject_id"],
            relation=p["relation"],
            object_id=p["object_id"],
            operation=Operation(p["operation"]),
            polarity=p["polarity"],
            time_expression=p["time_expression"],
            t_valid=_uninterval(p["t_valid"]),
            t_tx=_uninterval(p["t_tx"]),
            evidence_kind=EvidenceKind(p["evidence_kind"]),
            confidence=p["confidence"],
            span=p["span"],
            episode_id=p["episode_id"],
            status=p["status"],
            unanchored=p.get("unanchored", False),
            subject_ref=p.get("subject_ref", ""),
            object_ref=p.get("object_ref", ""),
        )
        for p in raw["propositions"]
    )

    for m in raw["mentions"]:
        clio.mentions._mentions.append(
            Mention(
                id=m["id"],
                episode_id=m["episode_id"],
                entity_id=m["entity_id"],
                surface=m["surface"],
                ts=_undt(m["ts"]),
                proposition_id=m.get("proposition_id"),
            )
        )
    clio.mentions._next_seq = _next_seq([m.id for m in clio.mentions.all()], "mnt_")

    for r in raw["fold_journal"]:
        clio.journal._records.append(
            FoldRecord(
                id=r["id"],
                kept=r["kept"],
                absorbed=r["absorbed"],
                score=r["score"],
                trigger_episode=r["trigger_episode"],
                migrated_edge_ids=list(r["migrated_edge_ids"]),
                snapshot=r["snapshot"],
                created_at=_undt(r["created_at"]),
                reverted=r["reverted"],
            )
        )
    clio.journal._next_seq = _next_seq([r.id for r in clio.journal.all()], "fold_")

    for u in raw["unmapped"]:
        clio.unmapped.append(
            suggested_relation=u["suggested_relation"],
            subject_ref=u["subject_ref"],
            object_ref=u["object_ref"],
            span=u["span"],
            episode_id=u["episode_id"],
        )

    clio.entity_index.rebuild(clio.graph)
    clio.episode_index.rebuild(clio.log)
    return clio


__all__ = ["dump_memory", "save_memory", "load_memory", "FORMAT_VERSION"]
