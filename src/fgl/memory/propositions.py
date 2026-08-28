"""The atomic unit of MECA: the **attested proposition**, and its store.

Why the unit changed
--------------------
Every memory model in this repo up to L6 stores *pointers to text*. The slots
of MEST are surface forms -- noun chunks, verb lemmas, WordNet hypernyms --
incident to the turns that contain them. The memory never asserts anything: it
records that the concept ``painting`` touches turns D5:3 and D9:1. Every
question therefore has to re-derive the fact from raw dialogue at answer time,
under noise and a token budget, which is why multi-hop stalls at f1 0.476 even
when `recall_context` is 1.0 -- the evidence is all there and the proposition
is not.

MECA stores the result of comprehension instead. Reading happens once per
document and may be expensive; answering happens N times and must be cheap.

Every field defends itself
--------------------------
``subject``/``object`` resolved
    Turns similarity search into *lookup*. Without resolution there is no such
    question as "the proposition whose subject is X".
``qualifiers`` as an open dict
    Fixed schema, open content. Place, instrument, quantity, company all enter
    as labelled roles; none becomes a new vertex kind and no per-domain
    ontology is needed.
``valid_from``/``valid_to`` vs ``asserted_at`` -- two clocks
    "Last month I quit", said in 2023-06, is asserted_at=2023-06 and
    valid_from≈2023-05. Most triple stores fuse the two, which makes both "when
    did she leave?" and supersession unanswerable.
``modality``/``polarity``
    A plan is not a fact and a negation is not an absence. In conversational
    text this is the single largest source of hallucination: retrieving "I'm
    moving to Lisbon" and answering "lives in Lisbon".
``evidence`` (required, >= 1)
    Nothing exists in the memory without a span that supports it. This is what
    makes an LLM-driven ingest auditable rather than reckless.
``derived_from``
    An inferred proposition is marked, never confused with a stated one, and
    the whole inference pass is one ablation flag.
``superseded_by``
    The store holds *state*, not a bag of facts.

Nothing here mentions speakers, sessions, or two-party dialogue. A source is a
sequence of utterances with an optional author and an optional timestamp.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Vocabulary                                                                   #
# --------------------------------------------------------------------------- #

MOD_ASSERTED = "asserted"
MOD_INTENDED = "intended"
MOD_DESIRED = "desired"
MOD_HYPOTHETICAL = "hypothetical"
MOD_ASKED = "asked"
MOD_REPORTED = "reported"

#: Ordered by how much weight an answer may put on them. ``asserted`` and
#: ``reported`` are claims about the world; the rest are claims about someone's
#: mind, and answering a factual question from one of them is the failure this
#: field exists to prevent.
MODALITIES: tuple[str, ...] = (
    MOD_ASSERTED, MOD_REPORTED, MOD_INTENDED, MOD_DESIRED,
    MOD_HYPOTHETICAL, MOD_ASKED,
)
FACTUAL_MODALITIES: frozenset[str] = frozenset({MOD_ASSERTED, MOD_REPORTED})

#: Argument roles that every proposition has, in the canonical order the ribbon
#: rotation uses. Qualifiers are open and slot in between ``object`` and
#: ``time``, sorted by role name so the rotation is deterministic.
ROLE_SUBJECT = "subject"
ROLE_PREDICATE = "predicate"
ROLE_OBJECT = "object"
ROLE_TIME = "time"
CORE_ROLES: tuple[str, ...] = (ROLE_SUBJECT, ROLE_PREDICATE, ROLE_OBJECT)

KIND_PROPOSITION = "proposition"
KIND_ENTITY = "entity"
KIND_PREDICATE = "predicate"
KIND_TIME = "time"
KIND_LITERAL = "literal"

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s'-]", re.UNICODE)


def normalise(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace. The key for lookup."""
    if not text:
        return ""
    return _WS.sub(" ", _PUNCT.sub(" ", text.strip().lower())).strip()


# --------------------------------------------------------------------------- #
# Time                                                                         #
# --------------------------------------------------------------------------- #

_PRECISIONS = {4: "year", 7: "month", 10: "day"}


@dataclass(frozen=True, order=False)
class TimePoint:
    """An ISO date prefix plus how much of it is real.

    Stored as a prefix (``2023``, ``2023-05``, ``2023-05-08``) rather than a
    datetime on purpose: "sometime in 2023" is a fact the corpus states, and
    widening it to midnight on 1 January is a fabrication. Comparison and
    containment are prefix operations, which is exactly the semantics wanted.
    """

    value: str
    precision: str = "day"

    @classmethod
    def parse(cls, raw: object) -> Optional[TimePoint]:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        m = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", text)
        if not m:
            return None
        parts = [p for p in m.groups() if p]
        value = "-".join(parts)
        return cls(value=value, precision=_PRECISIONS.get(len(value), "day"))

    def contains(self, other: TimePoint) -> bool:
        """``2023`` contains ``2023-05``; the reverse is false."""
        return other.value.startswith(self.value)

    def overlaps(self, other: TimePoint) -> bool:
        return self.contains(other) or other.contains(self)

    def __lt__(self, other: TimePoint) -> bool:
        return self.value < other.value

    def __le__(self, other: TimePoint) -> bool:
        return self.value <= other.value

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def earliest(points: Iterable[Optional[TimePoint]]) -> Optional[TimePoint]:
    real = [p for p in points if p is not None]
    return min(real, key=lambda p: p.value) if real else None


# --------------------------------------------------------------------------- #
# Evidence                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class Evidence:
    """A verbatim span and where it came from. Required, never summarised."""

    doc_id: str
    span: str
    timestamp: str = ""
    date_raw: str = ""
    author: str = ""

    def key(self) -> tuple[str, str]:
        return (self.doc_id, normalise(self.span))

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "span": self.span,
            "timestamp": self.timestamp, "date_raw": self.date_raw,
            "author": self.author,
        }


# --------------------------------------------------------------------------- #
# The proposition                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class Proposition:
    """One resolved, time-scoped, modality-tagged claim with its proof."""

    subject: str
    predicate: str
    object: str = ""
    #: whether ``object`` names an entity (joinable) or is a literal value
    object_is_entity: bool = False
    qualifiers: dict[str, str] = field(default_factory=dict)

    valid_from: Optional[TimePoint] = None
    valid_to: Optional[TimePoint] = None
    asserted_at: Optional[TimePoint] = None

    modality: str = MOD_ASSERTED
    polarity: bool = True

    evidence: list[Evidence] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    #: Analytical relationships between propositions.  They are deliberately
    #: not qualifiers: a link is neither a fact argument nor answer context.
    links: dict[str, list[str]] = field(default_factory=dict)

    confidence: float = 1.0
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    pid: str = ""

    # ------------------------------------------------------------- identity --
    def signature(self) -> tuple:
        """What makes two propositions *the same claim*.

        Qualifiers are deliberately OUT: "cooked chicken" and "cooked chicken
        on Sunday" are the same claim at different resolutions, and merging
        them (keeping the union of qualifiers) is right. Time is out for the
        same reason -- a repeated claim is evidence, not a second fact. What is
        IN is polarity and modality, because "did not go" and "went", or
        "plans to go" and "went", are different claims that a bag of triples
        silently conflates.
        """
        return (
            normalise(self.subject), normalise(self.predicate),
            normalise(self.object), self.polarity, self.modality,
        )

    def make_pid(self) -> str:
        raw = "|".join(str(x) for x in self.signature())
        ev = ",".join(sorted(e.doc_id for e in self.evidence))
        return "p" + hashlib.blake2b(
            f"{raw}#{ev}".encode(), digest_size=8
        ).hexdigest()

    # -------------------------------------------------------------- queries --
    @property
    def is_derived(self) -> bool:
        return bool(self.derived_from)

    @property
    def is_current(self) -> bool:
        return self.superseded_by is None

    @property
    def is_factual(self) -> bool:
        """A claim about the world rather than about somebody's mind."""
        return self.modality in FACTUAL_MODALITIES

    def when(self) -> Optional[TimePoint]:
        """Best available time: when it was true, else when it was said."""
        return self.valid_from or self.asserted_at

    def entities(self) -> list[str]:
        """Every entity this proposition touches -- the join surface."""
        out = [self.subject]
        if self.object and self.object_is_entity:
            out.append(self.object)
        return [e for e in out if e]

    def arguments(self) -> list[str]:
        """Every argument VALUE this proposition carries, normalised.

        Wider than :meth:`entities`: the subject, the object whether or not the
        extractor marked it as an entity, and every qualifier value. This is
        the join surface, and it has to match what a phi-walk can pivot
        through -- the walk goes through argument vertices, all of them.
        """
        out = [normalise(self.subject)]
        if self.object:
            out.append(normalise(self.object))
        for role, value in self.qualifiers.items():
            if value and not role.startswith("_"):
                out.append(normalise(value))
        return [v for v in out if v]

    def roles(self) -> list[tuple[str, str, str]]:
        """``(role, value, kind)`` in the canonical rotation order.

        This ordering IS the ribbon rotation at a proposition vertex, which is
        what makes a corner a pair of adjacent arguments and a question "a
        corner with one side blank". Qualifiers are sorted by role name so the
        order is a function of the proposition and not of dict insertion.
        """
        out: list[tuple[str, str, str]] = [
            (ROLE_SUBJECT, self.subject, KIND_ENTITY),
            (ROLE_PREDICATE, self.predicate, KIND_PREDICATE),
        ]
        if self.object:
            out.append((
                ROLE_OBJECT, self.object,
                KIND_ENTITY if self.object_is_entity else KIND_LITERAL,
            ))
        for role in sorted(self.qualifiers):
            value = self.qualifiers[role]
            if value and not role.startswith("_"):
                out.append((role, value, KIND_LITERAL))
        point = self.when()
        if point is not None:
            out.append((ROLE_TIME, point.value, KIND_TIME))
        return out

    # --------------------------------------------------------------- render --
    def statement(self) -> str:
        """The claim as one short sentence -- what the generator reads."""
        neg = "" if self.polarity else "not "
        head = f"{self.subject} {neg}{self.predicate}"
        if self.object:
            head = f"{head} {self.object}"
        extra = [
            f"{role}: {val}" for role, val in sorted(self.qualifiers.items())
            if val and not role.startswith("_")
        ]
        if extra:
            head = f"{head} ({'; '.join(extra)})"
        return head.strip()

    def label(self) -> str:
        """``[2023-03, asserted]`` -- the prefix shown before the statement."""
        point = self.when()
        when = point.value if point else "undated"
        tag = self.modality if self.modality != MOD_ASSERTED else "stated"
        if not self.is_current:
            tag += ", superseded"
        if self.is_derived:
            tag += ", inferred"
        return f"[{when}, {tag}]"

    def as_dict(self) -> dict:
        return {
            "pid": self.pid, "subject": self.subject, "predicate": self.predicate,
            "object": self.object, "object_is_entity": self.object_is_entity,
            "qualifiers": dict(self.qualifiers),
            "valid_from": self.valid_from.value if self.valid_from else None,
            "valid_to": self.valid_to.value if self.valid_to else None,
            "asserted_at": self.asserted_at.value if self.asserted_at else None,
            "modality": self.modality, "polarity": self.polarity,
            "evidence": [e.as_dict() for e in self.evidence],
            "derived_from": list(self.derived_from),
            "superseded_by": self.superseded_by,
            "links": {kind: list(pids) for kind, pids in self.links.items()},
            "confidence": round(self.confidence, 4),
        }


def proposition_from_dict(d: dict) -> Proposition:
    """Inverse of :meth:`Proposition.as_dict`.

    The graph is the ONLY artifact: the whole proposition rides in the
    proposition vertex's ``meta``, so a cached graph reloaded from disk
    reconstructs the store exactly and there is no second file to drift out of
    sync with it.
    """
    qualifiers = dict(d.get("qualifiers") or {})
    # Graphs written before links became a first-class field put them in a
    # private qualifier.  Read them safely, but never let them re-enter the
    # semantic proposition or its rendered text.
    links = {kind: list(pids) for kind, pids in (d.get("links") or {}).items()}
    legacy = qualifiers.pop("_links", "")
    for tag in str(legacy).split():
        if ":" not in tag:
            continue
        kind, pid = tag.split(":", 1)
        if kind and pid:
            links.setdefault(kind, []).append(pid)
    return Proposition(
        subject=d.get("subject", ""),
        predicate=d.get("predicate", ""),
        object=d.get("object", ""),
        object_is_entity=bool(d.get("object_is_entity", False)),
        qualifiers=qualifiers,
        valid_from=TimePoint.parse(d.get("valid_from")),
        valid_to=TimePoint.parse(d.get("valid_to")),
        asserted_at=TimePoint.parse(d.get("asserted_at")),
        modality=d.get("modality", MOD_ASSERTED),
        polarity=bool(d.get("polarity", True)),
        evidence=[
            Evidence(
                doc_id=e.get("doc_id", ""), span=e.get("span", ""),
                timestamp=e.get("timestamp", ""), date_raw=e.get("date_raw", ""),
                author=e.get("author", ""),
            )
            for e in (d.get("evidence") or [])
        ],
        derived_from=list(d.get("derived_from") or []),
        superseded_by=d.get("superseded_by"),
        links=links,
        confidence=float(d.get("confidence", 1.0)),
        pid=d.get("pid", ""),
    )


# --------------------------------------------------------------------------- #
# The store                                                                    #
# --------------------------------------------------------------------------- #


class PropositionStore:
    """Propositions plus the inverted indexes a query plan needs.

    Deliberately a plain object and not the fatgraph: the *flat* reader must be
    able to answer without any rotation at all, so that "does the ribbon
    structure pay?" compares two readers over one store instead of comparing
    two different memories.
    """

    def __init__(self) -> None:
        self.propositions: dict[str, Proposition] = {}
        self.by_subject: dict[str, list[str]] = {}
        self.by_entity: dict[str, list[str]] = {}
        self.by_predicate: dict[str, list[str]] = {}
        #: normalised argument VALUE -> pids. The join surface, and wider than
        #: ``by_entity`` on purpose: the face walk in the ribbon reader pivots
        #: through any argument vertex, so the flat reader has to be able to
        #: pivot through any argument too, or the two would differ for a reason
        #: that has nothing to do with the rotation. It also survives an
        #: extractor that forgets to mark ``object_is_entity``, which is the
        #: common case rather than the exception.
        self.by_argument: dict[str, list[str]] = {}
        #: canonical entity name -> aliases seen for it
        self.entity_aliases: dict[str, set[str]] = {}
        #: any normalised surface form -> the canonical entity lookup key.
        #: This is consulted by readers; aliases are not display-only data.
        self.alias_to_canonical: dict[str, str] = {}
        #: Conversation participants are identity anchors, never candidates
        #: for semantic consolidation with a description or another speaker.
        self.entity_anchors: set[str] = set()
        #: collective/generic phrase (its own entity, never fused with
        #: anything) -> the individually-named entities it plausibly groups.
        #: Built by ``consolidate.link_collective_references``; consulted at
        #: retrieval time only, never rendered into ``statement()``/``roles()``.
        self.entity_links: dict[str, list[str]] = {}

    def register_entity_anchor(self, name: str) -> None:
        """Register a source-supplied participant as an immutable identity."""
        key = normalise(name)
        if not key:
            return
        self.entity_anchors.add(key)
        self.entity_aliases.setdefault(key, set()).add(name)
        self.alias_to_canonical[key] = key

    def entity_key(self, entity: str) -> str:
        """Resolve a surface form through the alias table, if it has one."""
        key = normalise(entity)
        return self.alias_to_canonical.get(key, key)

    # ------------------------------------------------------------------ io --
    def add(self, prop: Proposition) -> str:
        if not prop.pid:
            prop.pid = prop.make_pid()
        # A pid collision is the same claim from the same evidence: merge
        # rather than overwrite, so ingestion is idempotent.
        existing = self.propositions.get(prop.pid)
        if existing is not None:
            merge_into(existing, prop)
            return existing.pid
        self.propositions[prop.pid] = prop
        self.by_subject.setdefault(normalise(prop.subject), []).append(prop.pid)
        self.by_predicate.setdefault(normalise(prop.predicate), []).append(prop.pid)
        for ent in prop.entities():
            self.by_entity.setdefault(normalise(ent), []).append(prop.pid)
        for value in prop.arguments():
            self.by_argument.setdefault(value, []).append(prop.pid)
        return prop.pid

    def remove(self, pid: str) -> None:
        prop = self.propositions.pop(pid, None)
        if prop is None:
            return
        for table, key in (
            (self.by_subject, normalise(prop.subject)),
            (self.by_predicate, normalise(prop.predicate)),
        ):
            if key in table and pid in table[key]:
                table[key].remove(pid)
        for ent in prop.entities():
            key = normalise(ent)
            if key in self.by_entity and pid in self.by_entity[key]:
                self.by_entity[key].remove(pid)
        for value in prop.arguments():
            if value in self.by_argument and pid in self.by_argument[value]:
                self.by_argument[value].remove(pid)

    # -------------------------------------------------------------- queries --
    def __len__(self) -> int:
        return len(self.propositions)

    def __iter__(self) -> Iterator[Proposition]:
        return iter(self.propositions.values())

    def get(self, pid: str) -> Optional[Proposition]:
        return self.propositions.get(pid)

    def about(self, entity: str) -> list[Proposition]:
        """Every proposition an entity participates in, chronologically.

        This is the flat reader's timeline: fetch, then sort. The ribbon
        reader gets the same list from a sigma-orbit with no sort at all, and
        the difference between those two is one of the four things M1 and M2
        are allowed to disagree about.
        """
        pids = self.by_entity.get(self.entity_key(entity), [])
        return sorted(
            (self.propositions[p] for p in pids if p in self.propositions),
            key=_chrono_key,
        )

    def with_subject(self, entity: str) -> list[Proposition]:
        pids = self.by_subject.get(normalise(entity), [])
        return sorted(
            (self.propositions[p] for p in pids if p in self.propositions),
            key=_chrono_key,
        )

    def with_predicate(self, predicate: str) -> list[Proposition]:
        pids = self.by_predicate.get(normalise(predicate), [])
        return sorted(
            (self.propositions[p] for p in pids if p in self.propositions),
            key=_chrono_key,
        )

    def knows_entity(self, entity: str) -> bool:
        """The distinction abstention actually needs.

        "I have never heard of this person" and "I know them but nobody ever
        said that about them" are different answers, and only a store of
        resolved propositions can tell them apart.
        """
        return self.entity_key(entity) in self.by_entity

    def predicates(self) -> list[str]:
        return sorted(self.by_predicate)

    def chronological(self) -> list[Proposition]:
        return sorted(self.propositions.values(), key=_chrono_key)

    def stats(self) -> dict:
        mods: dict[str, int] = {}
        for p in self.propositions.values():
            mods[p.modality] = mods.get(p.modality, 0) + 1
        derived = sum(1 for p in self.propositions.values() if p.is_derived)
        superseded = sum(1 for p in self.propositions.values() if not p.is_current)
        n_ev = sum(len(p.evidence) for p in self.propositions.values())
        # How many propositions touch the single busiest entity, against the
        # corpus's own median -- a ratio, so the health gate that reads it
        # does not need re-tuning when the corpus size changes. A ratio in
        # the tens or hundreds is the signature of an identity collapse: one
        # node absorbing what should have been many distinct people, the
        # failure a corrupted run of MECA once produced (D37).
        #
        # Conversation participants (`entity_anchors`) are the one legitimate
        # exception: a two-person dialogue is ABOUT those two people, so they
        # are named in most propositions by construction -- on a real
        # ingestion, Caroline alone sits at 500+ incidences while the median
        # entity sits at 1, and that is correct, not corruption. The signal
        # that actually distinguishes "identity collapse" from "this is who
        # the conversation is about" has to look past the anchors: it is a
        # NON-anchor entity absorbing an anchor-sized share that is the
        # "Caroline's friends, family and mentors' support" bug.
        incidences = sorted(len(pids) for pids in self.by_entity.values())
        max_incidence = incidences[-1] if incidences else 0
        median_incidence = incidences[len(incidences) // 2] if incidences else 0
        non_anchor = sorted(
            len(pids) for key, pids in self.by_entity.items()
            if key not in self.entity_anchors
        )
        max_non_anchor = non_anchor[-1] if non_anchor else 0
        median_non_anchor = non_anchor[len(non_anchor) // 2] if non_anchor else 0
        return {
            "n_propositions": len(self.propositions),
            "n_entities": len(self.by_entity),
            "n_predicates": len(self.by_predicate),
            "n_derived": derived,
            "n_superseded": superseded,
            "mean_evidence": round(n_ev / max(len(self.propositions), 1), 3),
            "by_modality": dict(sorted(mods.items())),
            #: informational only -- includes anchors, so it is expected to be
            #: large. The health gate reads the non-anchor pair below instead.
            "max_entity_incidence": max_incidence,
            "max_entity_incidence_ratio": (
                round(max_incidence / median_incidence, 2) if median_incidence else 0.0
            ),
            "max_non_anchor_entity_incidence": max_non_anchor,
            "max_non_anchor_entity_incidence_ratio": (
                round(max_non_anchor / median_non_anchor, 2)
                if median_non_anchor else 0.0
            ),
        }


def _chrono_key(p: Proposition) -> tuple[str, str]:
    point = p.when()
    return (point.value if point else "9999", p.pid)


def merge_into(target: Proposition, other: Proposition) -> None:
    """Fold ``other`` into ``target``: same claim, more support.

    Repetition across a corpus becomes **support** here -- one proposition
    carrying two spans. In every earlier model in this repo it became a second
    unit competing for the same token budget.
    """
    seen = {e.key() for e in target.evidence}
    for ev in other.evidence:
        if ev.key() not in seen:
            target.evidence.append(ev)
            seen.add(ev.key())
    # widen qualifiers rather than replace: "cooked chicken" plus "cooked
    # chicken on Sunday" is one claim known at two resolutions
    for role, value in other.qualifiers.items():
        target.qualifiers.setdefault(role, value)
    target.valid_from = earliest([target.valid_from, other.valid_from])
    target.asserted_at = earliest([target.asserted_at, other.asserted_at])
    if other.valid_to is not None and target.valid_to is None:
        target.valid_to = other.valid_to
    for pid in other.derived_from:
        if pid not in target.derived_from:
            target.derived_from.append(pid)
    for kind, pids in other.links.items():
        bucket = target.links.setdefault(kind, [])
        for pid in pids:
            if pid not in bucket:
                bucket.append(pid)
    # Confidence is whatever the extractor reported, taken at its best. It is
    # NOT raised on merge: the honest measure of "the corpus said this more
    # than once" is len(evidence), which is auditable and cannot be inflated by
    # arithmetic, and a number that creeps toward 1.0 every time two records
    # touch would read as certainty the memory has not earned.
    target.confidence = max(target.confidence, other.confidence)


# --------------------------------------------------------------------------- #
# The fatgraph view                                                            #
# --------------------------------------------------------------------------- #


def build_graph(store: PropositionStore, embedder=None) -> object:
    """Lay the store out as a fatgraph. **Both** conditions build this.

    The rotation is *additional* structure over the same incidences, so the
    flat reader can ignore it entirely and the comparison stays honest: one
    store, two readers, and any measured delta is the reader.

    Two rotations carry the claim:

    ``sigma`` at a proposition vertex
        follows :meth:`Proposition.roles` -- subject, predicate, object,
        qualifiers (sorted), time. Its CORNERS are therefore adjacent argument
        pairs, and a question is a corner with one side blank.
    ``sigma`` at an entity vertex
        is chronological **by construction**, because propositions are added in
        chronological order and each incidence appends. "Everything about X, in
        order" is then an orbit read of cost O(degree), not a fetch-and-sort.
    """
    from fgl.core import FatGraph

    graph = FatGraph()
    vids: dict[tuple[str, str], str] = {}

    def vertex(kind: str, value: str) -> str:
        key = (kind, normalise(value) or value)
        vid = vids.get(key)
        if vid is None:
            vid = graph.add_vertex(
                name=value,
                aliases=sorted(store.entity_aliases.get(normalise(value), ()))
                if kind == KIND_ENTITY else (),
                meta={
                    "kind": kind,
                    "key": key[1],
                    "identity_anchor": kind == KIND_ENTITY
                    and key[1] in store.entity_anchors,
                    # a collective/generic phrase's members, if any -- never
                    # identity, just a one-hop pointer forward at read time
                    "collective_members": (
                        sorted(store.entity_links.get(key[1], []))
                        if kind == KIND_ENTITY and key[1] in store.entity_links
                        else []
                    ),
                },
            )
            vids[key] = vid
        return vid

    # Chronological order is load-bearing: it is what makes every entity's
    # orbit a timeline without a single sort at read time.
    for prop in store.chronological():
        pvid = vertex(KIND_PROPOSITION, prop.statement())
        graph.vertices[pvid].meta.update({
            "kind": KIND_PROPOSITION,
            "pid": prop.pid,
            "modality": prop.modality,
            "polarity": prop.polarity,
            "derived": prop.is_derived,
            "current": prop.is_current,
            "when": prop.when().value if prop.when() else "",
            "statement": prop.statement(),
            "label": prop.label(),
            "turn_ids": [e.doc_id for e in prop.evidence],
            # the whole claim rides here so a cached graph IS the memory
            "prop": prop.as_dict(),
        })
        text = prop.statement()
        for role, value, kind in prop.roles():
            avid = vertex(kind, value)
            graph.add_edge(
                pvid, avid,
                {
                    "text": text,
                    "turn_ids": [e.doc_id for e in prop.evidence],
                    "timestamp": prop.evidence[0].timestamp if prop.evidence else "",
                    "session_id": "",
                    "meta": {"role": role, "pid": prop.pid, "kind": kind},
                },
            )
    return graph


def propositions_from_graph(graph) -> list[str]:
    """The pids a graph holds, in rotation order. Used by the ribbon reader."""
    return [
        vx.meta.get("pid", "")
        for vx in graph.vertices.values()
        if vx.meta.get("kind") == KIND_PROPOSITION
    ]


def orbit_pids(graph, vertex_id: str) -> list[str]:
    """Proposition pids around a vertex, in sigma order (no sorting).

    The ribbon read. At an entity vertex this is the entity's timeline; the
    flat reader gets the same set from an inverted index and has to sort it.
    """
    out: list[str] = []
    for h in graph.sigma.get(vertex_id, ()):
        other = graph.H[graph.alpha[h]].vertex_id
        pid = graph.vertices[other].meta.get("pid", "")
        if pid and (not out or out[-1] != pid):
            out.append(pid)
    return out


def corners(graph, vertex_id: str) -> list[tuple[str, str]]:
    """Adjacent role pairs around a proposition vertex.

    ``(subject, predicate)``, ``(predicate, object)``, ``(object, time)`` ...
    A question with a hole is one of these with one side blank, which is the
    ribbon reader's hole test -- the flat reader does a field comparison
    instead.
    """
    ring = list(graph.sigma.get(vertex_id, ()))
    if len(ring) < 2:
        return []
    roles = [graph.H[h].meta.get("role", "") for h in ring]
    return [(roles[i], roles[(i + 1) % len(roles)]) for i in range(len(roles))]


def entity_vertex(graph, value: str) -> Optional[str]:
    key = normalise(value)
    for vid, vx in graph.vertices.items():
        if vx.meta.get("kind") == KIND_ENTITY and vx.meta.get("key") == key:
            return vid
    return None


def vertices_of_kind(graph, kind: str) -> dict[str, str]:
    """``normalised key -> vertex id`` for one vertex kind."""
    return {
        vx.meta.get("key", ""): vid
        for vid, vx in graph.vertices.items()
        if vx.meta.get("kind") == kind
    }


def parse_modality(raw: object) -> str:
    text = normalise(str(raw or ""))
    for mod in MODALITIES:
        if text == mod:
            return mod
    # a model that answers "plan" or "wish" instead of the enum should not
    # silently become an assertion -- that is the exact failure this field
    # exists to prevent
    aliases = {
        "plan": MOD_INTENDED, "planned": MOD_INTENDED, "intent": MOD_INTENDED,
        "future": MOD_INTENDED, "wish": MOD_DESIRED, "want": MOD_DESIRED,
        "hope": MOD_DESIRED, "hypothesis": MOD_HYPOTHETICAL,
        "conditional": MOD_HYPOTHETICAL, "question": MOD_ASKED,
        "fact": MOD_ASSERTED, "stated": MOD_ASSERTED, "past": MOD_ASSERTED,
        "hearsay": MOD_REPORTED,
    }
    return aliases.get(text, MOD_HYPOTHETICAL)


def coerce_proposition(raw: dict, evidence: Sequence[Evidence],
                       asserted_at: Optional[TimePoint]) -> Optional[Proposition]:
    """Build a Proposition from one LLM JSON object, or ``None`` if unusable.

    Conservative by construction: a record without a subject or a predicate is
    dropped rather than repaired, and an unrecognised modality becomes
    ``hypothetical`` rather than ``asserted`` -- the direction of the error
    matters, because a wrong ``asserted`` is a hallucination and a wrong
    ``hypothetical`` is only a missed answer.
    """
    subject = str(raw.get("subject") or "").strip()
    predicate = str(raw.get("predicate") or "").strip()
    if not subject or not predicate:
        return None
    obj = str(raw.get("object") or "").strip()
    quals_raw = raw.get("qualifiers") or {}
    qualifiers = {
        normalise(str(k)).replace(" ", "_"): str(v).strip()
        for k, v in quals_raw.items()
        if str(v or "").strip() and normalise(str(k)) not in CORE_ROLES
    } if isinstance(quals_raw, dict) else {}
    polarity = raw.get("polarity", True)
    if isinstance(polarity, str):
        polarity = normalise(polarity) not in ("negative", "-", "false", "no")
    return Proposition(
        subject=subject,
        predicate=predicate,
        object=obj,
        object_is_entity=bool(raw.get("object_is_entity", False)) and bool(obj),
        qualifiers=qualifiers,
        valid_from=TimePoint.parse(raw.get("valid_from")),
        valid_to=TimePoint.parse(raw.get("valid_to")),
        asserted_at=asserted_at,
        modality=parse_modality(raw.get("modality")),
        polarity=bool(polarity),
        evidence=list(evidence),
        confidence=float(raw.get("confidence", 1.0) or 1.0),
    )


def store_from_graph(graph) -> PropositionStore:
    """Rebuild the store from a (possibly cached) graph.

    Both readers call this, so a run that reuses ``artifacts/graphs/`` gets
    exactly the memory the ingest built -- aliases, supersession links and all.
    """
    store = PropositionStore()
    for vx in graph.vertices.values():
        if vx.meta.get("kind") != KIND_PROPOSITION:
            continue
        raw = vx.meta.get("prop")
        if not isinstance(raw, dict):
            continue
        prop = proposition_from_dict(raw)
        if not prop.pid:
            prop.pid = prop.make_pid()
        store.propositions[prop.pid] = prop
        store.by_subject.setdefault(normalise(prop.subject), []).append(prop.pid)
        store.by_predicate.setdefault(normalise(prop.predicate), []).append(prop.pid)
        for ent in prop.entities():
            store.by_entity.setdefault(normalise(ent), []).append(prop.pid)
        for value in prop.arguments():
            store.by_argument.setdefault(value, []).append(prop.pid)
    for vx in graph.vertices.values():
        if vx.meta.get("kind") != KIND_ENTITY:
            continue
        canonical = vx.meta.get("key", normalise(vx.name))
        # aliases/alias_to_canonical only mean something when there ARE
        # aliases; identity_anchor and collective_members are per-vertex
        # facts that must survive the round trip regardless -- most
        # collective-phrase entities ("audrey's dogs") never pick up an
        # alias at all, and gating on `vx.aliases` here would silently drop
        # them on every cached-graph run.
        if vx.aliases:
            aliases = store.entity_aliases.setdefault(canonical, set())
            aliases.update(vx.aliases)
            aliases.add(vx.name)
            for alias in aliases:
                store.alias_to_canonical[normalise(alias)] = canonical
        if vx.meta.get("identity_anchor"):
            store.entity_anchors.add(canonical)
        members = vx.meta.get("collective_members")
        if members:
            store.entity_links[canonical] = list(members)
    for key in store.by_entity:
        store.alias_to_canonical.setdefault(key, key)
    return store


def proposition_vertex_index(graph) -> dict[str, str]:
    """``pid -> vertex id``. The ribbon reader's way back into the rotation."""
    return {
        vx.meta.get("pid", ""): vid
        for vid, vx in graph.vertices.items()
        if vx.meta.get("kind") == KIND_PROPOSITION and vx.meta.get("pid")
    }


def sharing_argument(store: PropositionStore, value: str) -> list[str]:
    """pids that carry ``value`` in any argument role."""
    return list(store.by_argument.get(normalise(value), ()))
