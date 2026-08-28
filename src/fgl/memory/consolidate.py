"""Consolidation: turn a pile of extracted propositions into a *state*.

Extraction produces claims one passage at a time and cannot see across the
corpus. Consolidation is where the corpus becomes a memory:

1. **entity resolution** -- "she", "Mel" and "Melanie" become one entity, so
   the store can be *queried* instead of searched;
2. **deduplication** -- the same claim stated twice becomes one proposition
   with two pieces of evidence. Repetition becomes confidence; in every earlier
   model here it became a duplicate competing for the same token budget;
3. **predicate functionality, estimated from the corpus** -- does this
   predicate admit one value at a time ("lives in") or many ("read")? Needed
   for supersession and answered *without an ontology and without gold
   labels*, which is the D30 criterion applied to a new parameter;
4. **the timeline** -- a later claim on a functional predicate closes the
   earlier one, so a query with no time returns the state that is *current*
   rather than every value ever mentioned;
5. **typed links** -- ``elaborates``, ``updates``, ``contradicts``. A
   contradiction is kept, both sides, and surfaced at answer time rather than
   silently resolved.

Every threshold here is a corpus-derived quantile with an absolute fallback,
never a swept literal: the question "does this parameter need the gold labels
to be set?" has to answer *no* for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgl.memory.propositions import (
    Proposition,
    PropositionStore,
    merge_into,
    normalise,
)

LINK_ELABORATES = "elaborates"
LINK_UPDATES = "updates"
LINK_CONTRADICTS = "contradicts"


@dataclass
class ConsolidationReport:
    entities_before: int = 0
    entities_after: int = 0
    propositions_before: int = 0
    propositions_after: int = 0
    merged: int = 0
    superseded: int = 0
    contradictions: int = 0
    elaborations: int = 0
    entity_threshold: float = 0.0
    predicate_threshold: float = 0.0
    functional_threshold: float = 0.0
    functional_predicates: int = 0
    source: str = "derived"
    collective_links: int = 0

    def as_dict(self) -> dict:
        return {
            "entities": {"before": self.entities_before, "after": self.entities_after},
            "propositions": {
                "before": self.propositions_before, "after": self.propositions_after,
                "merged": self.merged,
            },
            "timeline": {
                "superseded": self.superseded,
                "contradictions": self.contradictions,
                "elaborations": self.elaborations,
                "functional_predicates": self.functional_predicates,
            },
            "thresholds": {
                "entity": round(self.entity_threshold, 4),
                "predicate": round(self.predicate_threshold, 4),
                "functional": round(self.functional_threshold, 4),
                "source": self.source,
            },
            "collective_links": self.collective_links,
        }


# --------------------------------------------------------------------------- #
# Derived thresholds                                                           #
# --------------------------------------------------------------------------- #


def pairwise_quantile(
    vectors: np.ndarray, quantile: float, floor: float, max_rows: int = 400
) -> tuple[float, str]:
    """Quantile of the observed pairwise cosine, with an honest fallback.

    A property of the *encoder on this corpus*, not of the task: swapping the
    embedding model moves the whole distribution and this estimator follows it
    instead of needing a re-sweep. Same recipe as
    ``slots.concept_link_quantile`` -- reused rather than reinvented, because
    it is the part of the previous line that survived scrutiny.
    """
    n = len(vectors)
    if n < 8:
        return floor, "fallback"
    if n > max_rows:
        # deterministic stride, never a random sample: the threshold must be a
        # function of the corpus and not of a seed
        step = max(1, n // max_rows)
        vectors = vectors[::step]
        n = len(vectors)
    sims = vectors @ vectors.T
    iu = np.triu_indices(n, k=1)
    values = sims[iu]
    if values.size == 0:
        return floor, "fallback"
    return max(float(np.quantile(values, quantile)), floor), "derived"


def predicate_functionality(store: PropositionStore) -> dict[str, float]:
    """How single-valued each predicate is, measured on this corpus.

    ``functionality(p)`` = the mean over subjects of ``1 / |distinct objects|``.
    "lives in" comes out near 1.0 and "read" near 0.2, with no ontology, no
    hand-written list and no gold label -- the corpus says it.

    The smooth form, not "the fraction of subjects with exactly one object",
    and the reason is a circularity that the sharp form walks straight into: a
    subject who MOVED has two cities, so the very supersession this estimator
    exists to license is also the thing that makes the predicate look
    non-functional. One mover among four people drags the sharp score to 0.75
    and can push the predicate under any reasonable cut; the same mover costs
    the smooth score 0.125. Heavy accumulation ("read", with a dozen books) is
    still punished hard, which is the separation that matters.

    Only factual, current, positive propositions vote: a plan and a negation
    are not evidence about how many values a relation admits.
    """
    by_pred: dict[str, dict[str, set[str]]] = {}
    for prop in store:
        if not prop.is_factual or not prop.polarity or not prop.object:
            continue
        pred = normalise(prop.predicate)
        by_pred.setdefault(pred, {}).setdefault(
            normalise(prop.subject), set()
        ).add(normalise(prop.object))
    out: dict[str, float] = {}
    for pred, subjects in by_pred.items():
        if not subjects:
            continue
        out[pred] = sum(1.0 / len(objs) for objs in subjects.values()) / len(subjects)
    return out


# --------------------------------------------------------------------------- #
# 1. Entity resolution                                                         #
# --------------------------------------------------------------------------- #


def resolve_entities(
    store: PropositionStore, embedder, quantile: float = 0.995, floor: float = 0.80
) -> tuple[int, int, float, str]:
    """Resolve only lexical aliases, never semantic neighbours.

    Every proposition subject is an entity in the current schema, including
    descriptions such as "Caroline's support network".  Embedding-based
    transitive clustering therefore turns related descriptions into a single
    person and destroys the query key.  A memory can safely collapse only
    surface forms with a direct lexical relationship (``Mel``/``Melanie``),
    while embeddings remain useful for predicates and retrieval.

    ``quantile`` and ``floor`` remain in the public signature for compatible
    configs/reports, but no longer authorise identity merges.
    """
    names: dict[str, str] = {}
    counts: dict[str, int] = {}
    for prop in store:
        for ent in prop.entities():
            key = normalise(ent)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            if key not in names or (len(ent), ent) < (len(names[key]), names[key]):
                names[key] = ent
    before = len(names)
    if before == 0:
        return 0, 0, 0.0, "lexical_safe"

    # A candidate can join only a directly related canonical form.  There is
    # intentionally no union-find: A~B and B~C must not make A and C one
    # identity by accident.  Conversation participants win every tie; other
    # aliases use frequency, then the shorter display form.
    canonical: dict[str, str] = {key: names[key] for key in names}
    ordered = sorted(
        names,
        key=lambda key: (
            key not in store.entity_anchors,
            -counts[key],
            len(names[key]),
            names[key],
        ),
    )
    accepted: list[str] = []
    for key in ordered:
        matches = [
            candidate for candidate in accepted
            if _is_short_form(key, candidate) or _is_short_form(candidate, key)
        ]
        if matches:
            best = min(
                matches,
                key=lambda candidate: (
                    candidate not in store.entity_anchors,
                    -counts[candidate],
                    len(names[candidate]),
                    names[candidate],
                ),
            )
            canonical[key] = names[best]
        else:
            accepted.append(key)

    groups: dict[str, list[str]] = {}
    for key, value in canonical.items():
        groups.setdefault(normalise(value), []).append(key)
    for target, members in groups.items():
        aliases = store.entity_aliases.setdefault(target, set())
        aliases.add(names[target])
        for member in members:
            aliases.add(names[member])

    _rewrite_entities(store, canonical)
    return before, len(groups), 0.0, "lexical_safe"


def _is_short_form(a: str, b: str) -> bool:
    """``mel`` / ``melanie carter``: whole-word prefix, both non-trivial."""
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    if len(short) < 3 or short == long_:
        return False
    return long_.startswith(short + " ")


def _rewrite_entities(store: PropositionStore, canonical: dict[str, str]) -> None:
    props = list(store.propositions.values())
    store.propositions.clear()
    store.by_subject.clear()
    store.by_entity.clear()
    store.by_predicate.clear()
    store.by_argument.clear()
    store.alias_to_canonical.clear()
    for prop in props:
        prop.subject = canonical.get(normalise(prop.subject), prop.subject)
        if prop.object_is_entity:
            prop.object = canonical.get(normalise(prop.object), prop.object)
        prop.pid = ""  # identity depends on the resolved names
        store.add(prop)
    for source, target in canonical.items():
        store.alias_to_canonical[source] = normalise(target)
    for canonical_key, aliases in store.entity_aliases.items():
        for alias in aliases:
            store.alias_to_canonical[normalise(alias)] = canonical_key
    for key in store.by_entity:
        store.alias_to_canonical.setdefault(key, key)


# --------------------------------------------------------------------------- #
# 2. Deduplication                                                             #
# --------------------------------------------------------------------------- #


def deduplicate(
    store: PropositionStore, embedder, quantile: float = 0.99, floor: float = 0.82
) -> tuple[int, float, str]:
    """Merge propositions that are the same claim in different words.

    Grouped by subject first, so the comparison is O(claims per person) and not
    O(corpus squared). Two propositions merge when subject, polarity, modality
    and object all match and the predicates are closer than the derived cut --
    ``bought a bike`` and ``purchased a bike`` are one memory, not two.
    """
    preds = store.predicates()
    if len(preds) < 2:
        return 0, floor, "fallback"
    vectors = _unit(np.asarray(embedder.encode(preds), dtype=np.float32))
    threshold, source = pairwise_quantile(vectors, quantile, floor)
    index = {p: i for i, p in enumerate(preds)}

    merged = 0
    by_subject: dict[str, list[Proposition]] = {}
    for prop in store.chronological():
        by_subject.setdefault(normalise(prop.subject), []).append(prop)

    for group in by_subject.values():
        keep: list[Proposition] = []
        for prop in group:
            target = None
            for other in keep:
                if (
                    other.polarity == prop.polarity
                    and other.modality == prop.modality
                    and normalise(other.object) == normalise(prop.object)
                ):
                    i, j = index[normalise(other.predicate)], index[normalise(prop.predicate)]
                    if i == j or float(vectors[i] @ vectors[j]) >= threshold:
                        target = other
                        break
            if target is None:
                keep.append(prop)
            else:
                merge_into(target, prop)
                store.remove(prop.pid)
                merged += 1
    return merged, threshold, source


# --------------------------------------------------------------------------- #
# 3-4. Timeline                                                                #
# --------------------------------------------------------------------------- #


def apply_supersession(
    store: PropositionStore, quantile: float = 0.75, floor: float = 0.6
) -> tuple[int, float, int]:
    """A later claim on a single-valued predicate closes the earlier one.

    The cut on functionality is the ``quantile`` of the functionality
    distribution *of this corpus*, floored so a tiny corpus cannot make
    everything functional by accident. When in doubt the memory does NOT
    supersede: keeping two facts costs budget, deleting the true one costs the
    answer.
    """
    functionality = predicate_functionality(store)
    if not functionality:
        return 0, floor, 0
    values = sorted(functionality.values())
    cut = max(float(np.quantile(values, quantile)), floor)
    functional = {p for p, f in functionality.items() if f >= cut}

    superseded = 0
    groups: dict[tuple[str, str], list[Proposition]] = {}
    for prop in store:
        if not prop.is_factual or not prop.polarity or not prop.object:
            continue
        pred = normalise(prop.predicate)
        if pred in functional:
            groups.setdefault((normalise(prop.subject), pred), []).append(prop)

    for chain in groups.values():
        chain.sort(key=lambda p: (p.when().value if p.when() else "9999", p.pid))
        for earlier, later in zip(chain, chain[1:], strict=False):
            if normalise(earlier.object) == normalise(later.object):
                continue
            if earlier.superseded_by is not None:
                continue
            earlier.superseded_by = later.pid
            if earlier.valid_to is None:
                earlier.valid_to = later.when()
            superseded += 1
    return superseded, cut, len(functional)


def link_propositions(
    store: PropositionStore, functional_predicates: set[str] | None = None,
) -> tuple[int, int]:
    """``elaborates`` and ``contradicts``. Both are read at answer time.

    A contradiction is NOT resolved here. Two current claims that disagree,
    with overlapping validity and neither superseding the other, are a fact
    about the corpus, and the honest thing is to show both with their dates and
    let the answer say so.
    """
    elaborations = 0
    contradictions = 0
    groups: dict[tuple[str, str], list[Proposition]] = {}
    for prop in store:
        groups.setdefault(
            (normalise(prop.subject), normalise(prop.predicate)), []
        ).append(prop)

    functional_predicates = functional_predicates or set()
    for (_subject, predicate), chain in groups.items():
        if len(chain) < 2:
            continue
        for i, a in enumerate(chain):
            for b in chain[i + 1:]:
                same_object = normalise(a.object) == normalise(b.object)
                if same_object and a.qualifiers != b.qualifiers:
                    _note(a, LINK_ELABORATES, b.pid)
                    _note(b, LINK_ELABORATES, a.pid)
                    elaborations += 1
                elif (
                    predicate in functional_predicates
                    and
                    not same_object
                    and a.is_current and b.is_current
                    and a.is_factual and b.is_factual
                    and a.polarity == b.polarity
                    and a.object and b.object
                    and a.when() is not None and b.when() is not None
                    and a.when().overlaps(b.when())
                ):
                    _note(a, LINK_CONTRADICTS, b.pid)
                    _note(b, LINK_CONTRADICTS, a.pid)
                    contradictions += 1
    return elaborations, contradictions


def _collective_anchor_by_form(key: str, anchors: set[str]) -> str | None:
    """Lexical only: ``"audrey's dogs"`` -> ``"audrey"``. No embeddings, no
    transitivity -- extraction's own fallback when one passage makes a claim
    about several things at once and there is no single name to give it."""
    for a in anchors:
        for suffix in ("'s ", "\u2019s "):
            prefix = a + suffix
            if key.startswith(prefix) and len(key) > len(prefix):
                return a
    return None


def link_collective_references(store: PropositionStore) -> dict[str, list[str]]:
    """Entity-level, non-identity links from a collective/generic phrase to
    the individually-named things it plausibly groups.

    Found empirically (D37 follow-up): a participant's dogs, adopted one at a
    time across a conversation, end up under SIX different entities --
    ``"audrey's dogs"``, ``"dogs"``, ``"audrey's pups"``, ``"audrey's pets"``,
    ``"fur kids"``, ``"audrey's furry friends"`` -- none pointing at Toby,
    Scout, Pixie... A question about the group binds one of the six and reads
    only generic chatter, never a name. Merging them would repeat the D37
    corruption (synonymy is exactly the judgment embedding clustering got
    wrong); this instead adds a ONE-HOP POINTER FORWARD, still fully
    auditable, from the collective phrase to the specific entities it stands
    in for.

    Detection is lexical only, not transitive, not embedding-based -- the
    same discipline D37 already applies to identity: the key IS a possessive
    of a registered participant ("audrey's dogs", "audrey's furry friends").
    A co-occurrence rule (a bare generic noun like "dogs" or "fur kids"
    naming one anchor as the other argument in most of its own propositions)
    was tried and dropped: verified on real data, it flagged "acoustic
    guitar" and a book title as collective phrases just as readily as
    "dogs", because in a two-participant conversation almost everything
    co-occurs with one of the two anchors most of the time -- the count
    cannot discriminate here, so a bare generic noun is left unlinked rather
    than linked on a signal this domain does not support.

    Returns ``{collective_key: [member_key, ...]}``. Never mutates the store;
    the caller decides where this lives (``store.entity_links``).
    """
    anchors = store.entity_anchors
    if not anchors:
        return {}

    # anchor -> the entities it directly, individually names -- the
    # candidate "members". object_is_entity=True alone is far too loose in
    # practice (extraction marks "sushi", "a farm", "landlords" the same
    # way it marks "Pixie"), so a member also has to look like an actual
    # name: capitalised, one or two tokens, not led by a determiner. That is
    # a property of the TEXT, not a threshold, and it is what separates
    # Toby/Scout/Pixie/Pepper from the generic nouns sharing the same
    # object_is_entity flag.
    _DETERMINERS = (
        "a ", "an ", "the ", "another ", "some ", "any ", "this ", "that ",
        "these ", "those ", "her ", "his ", "my ", "our ", "their ", "its ",
    )
    anchor_named: dict[str, set[str]] = {a: set() for a in anchors}
    for prop in store:
        subj = normalise(prop.subject)
        if subj not in anchors or not prop.object or not prop.object_is_entity:
            continue
        raw = prop.object.strip()
        if not raw or not raw[0].isupper():
            continue
        if raw.lower().startswith(_DETERMINERS):
            continue
        if len(raw.split()) > 2:
            continue
        obj_key = normalise(raw)
        if obj_key and obj_key not in anchors:
            anchor_named[subj].add(obj_key)

    # Rule 2 (bare generic noun -> majority co-occurrence with one anchor)
    # was tried and dropped: in a two-participant conversation almost every
    # entity co-occurs with one of the two anchors most of the time, simply
    # because that anchor authors most of the turns -- it flagged "acoustic
    # guitar" and a book title as collective phrases for Caroline as readily
    # as it flagged "dogs" for Audrey. Verified on real data before cutting
    # it; kept here as rule 1 only, which stays precise because it is a
    # property of the TEXT (a possessive of a registered participant), not
    # a count that this domain cannot discriminate on.
    collective_of: dict[str, str] = {}
    for key in store.by_entity:
        if key in anchors:
            continue
        by_form = _collective_anchor_by_form(key, anchors)
        if by_form:
            collective_of[key] = by_form

    links: dict[str, list[str]] = {}
    for key, anchor in collective_of.items():
        # a member must be a genuinely distinct, individually-named entity --
        # never another collective phrase, never the collective itself.
        members = sorted(
            m for m in anchor_named.get(anchor, ())
            if m not in collective_of and m != key
        )
        if members:
            links[key] = members
    return links


def _note(prop: Proposition, link: str, pid: str) -> None:
    bucket = prop.links.setdefault(link, [])
    if pid not in bucket:
        bucket.append(pid)


def links_of(prop: Proposition, link: str) -> list[str]:
    return list(prop.links.get(link, ()))


#: Substrings that must never appear in a proposition's own text. Every one
#: of these once leaked into ``statement()`` -- the corrupted run that
#: motivated D37 rendered ``contradicts:p...`` straight into the context the
#: generator read. ``links`` is a first-class field precisely so this cannot
#: happen again; ``count_leaked_links`` checks that invariant directly on
#: what the generator will actually see, rather than trusting the field split.
_LEAK_MARKERS = ("_links", "contradicts:", "elaborates:", "updates:")


def count_leaked_links(store: PropositionStore) -> int:
    """How many propositions still show analytical metadata as text.

    Zero on a healthy store, always -- this is not a threshold to tune, it is
    a structural invariant. A non-zero count means a link or a stray
    underscore-prefixed qualifier is bleeding into ``statement()``/``roles()``
    and therefore into the embedding, the render and the answer prompt.
    """
    leaked = 0
    for prop in store:
        text = prop.statement()
        if any(marker in text for marker in _LEAK_MARKERS):
            leaked += 1
            continue
        if any(role.startswith("_") for role in prop.qualifiers):
            leaked += 1
    return leaked


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def consolidate(
    store: PropositionStore,
    embedder,
    entity_quantile: float = 0.995,
    entity_floor: float = 0.80,
    predicate_quantile: float = 0.99,
    predicate_floor: float = 0.82,
    functional_quantile: float = 0.75,
    functional_floor: float = 0.6,
    resolve: bool = True,
    dedupe: bool = True,
    timeline: bool = True,
) -> ConsolidationReport:
    """Run the four stages. Each is a flag so each is one ablation."""
    report = ConsolidationReport(
        propositions_before=len(store),
        entities_before=len(store.by_entity),
    )
    if resolve:
        before, after, threshold, source = resolve_entities(
            store, embedder, entity_quantile, entity_floor
        )
        report.entities_before, report.entities_after = before, after
        report.entity_threshold, report.source = threshold, source
    else:
        report.entities_after = report.entities_before

    if dedupe:
        merged, threshold, _ = deduplicate(
            store, embedder, predicate_quantile, predicate_floor
        )
        report.merged, report.predicate_threshold = merged, threshold

    if timeline:
        superseded, cut, functional = apply_supersession(
            store, functional_quantile, functional_floor
        )
        report.superseded = superseded
        report.functional_threshold = cut
        report.functional_predicates = functional
        functionality = predicate_functionality(store)
        functional_predicates = {
            predicate for predicate, value in functionality.items() if value >= cut
        }
        report.elaborations, report.contradictions = link_propositions(
            store, functional_predicates
        )

    store.entity_links = link_collective_references(store)
    report.collective_links = len(store.entity_links)

    report.propositions_after = len(store)
    return report


def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)
