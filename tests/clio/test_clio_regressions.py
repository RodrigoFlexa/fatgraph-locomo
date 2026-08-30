"""One test per bug the 2026-08-30 audit found in a real run, each pinning
the behaviour that was WRONG before, not just the behaviour that is right
now. Every one of these passed silently as a plausible-looking graph.
"""

from __future__ import annotations

import random
from datetime import datetime

import pytest

from fgl.clio.access.movements import anchor, count, follow, restrict
from fgl.clio.canonical import canonical_entities, canonical_form
from fgl.clio.catalog import load_catalog
from fgl.clio.config import ClioConfig
from fgl.clio.consolidate.pipeline import consolidate
from fgl.clio.graph.store import GraphStore
from fgl.clio.log.mentions import MentionStore
from fgl.clio.log.store import LogStore
from fgl.clio.staging import StagingStore
from fgl.clio.temporal.resolver import resolve_time
from fgl.clio.types import EvidenceKind, Interval, Operation, Proposition

MAY = datetime(2023, 5, 25)
JUNE = datetime(2023, 6, 25)


@pytest.fixture
def catalog():
    return load_catalog(ClioConfig.default().catalog_path)


def _prop(
    pid,
    subject,
    relation,
    obj,
    ts,
    *,
    operation=Operation.ASSERT,
    polarity=True,
    t_valid=None,
    unanchored=False,
    kind=EvidenceKind.LITERAL,
    confidence=0.9,
    episode=None,
):
    return Proposition(
        id=pid,
        subject_id=subject,
        relation=relation,
        object_id=obj,
        operation=operation,
        polarity=polarity,
        t_valid=t_valid if t_valid is not None else Interval(ts, None),
        t_tx=Interval(ts, None),
        evidence_kind=kind,
        confidence=confidence,
        span="x",
        episode_id=episode or pid,
        unanchored=unanchored,
        subject_ref=subject,
        object_ref=obj,
    )


SPEAKER = "new:Melanie"


def _memory(catalog, props):
    graph, staging = GraphStore(), StagingStore()
    staging.insert(props)
    consolidate(catalog, graph, staging, ClioConfig.default())
    return graph, staging


# --------------------------------------------------------------------- #
# 1. polarity was dropped at the graph boundary                           #
# --------------------------------------------------------------------- #


def test_negation_is_not_written_as_affirmation(catalog):
    graph, _ = _memory(
        catalog, [_prop("p0", SPEAKER, "likes", "new:running", MAY, polarity=False)]
    )
    (edge,) = graph.all_edges()
    assert edge.polarity is False, "an explicit denial must not be stored as a claim"


def test_follow_does_not_walk_a_negative_edge(catalog):
    graph, _ = _memory(
        catalog, [_prop("p0", SPEAKER, "likes", "new:running", MAY, polarity=False)]
    )
    state = follow(anchor("Melanie", graph), "likes", graph, catalog)
    assert state.trails == []
    assert state.death_cause == "all_edges_negative"


def test_denial_and_claim_do_not_collapse_into_one_edge(catalog):
    graph, _ = _memory(
        catalog,
        [
            _prop("p0", SPEAKER, "likes", "new:running", MAY),
            _prop("p1", SPEAKER, "likes", "new:running", JUNE, polarity=False),
        ],
    )
    assert len(graph.all_edges()) == 2
    assert {e.polarity for e in graph.all_edges()} == {True, False}
    assert all(e.conflict_flag for e in graph.all_edges()), (
        "opposite polarity at one address over overlapping validity is the "
        "contradiction phase 8 exists to flag"
    )


# --------------------------------------------------------------------- #
# 2. an unresolvable date became "true for all time"                      #
# --------------------------------------------------------------------- #


def test_vague_time_expression_does_not_resolve(catalog):
    """The trigger condition, straight from the failing run: these are
    ordinary English, not exotic phrasings."""
    for expression in (
        "each day",
        "over the weekend",
        "recently",
        "a couple of months ago",
    ):
        interval, tconf = resolve_time(expression, MAY, catalog["practices"])
        assert interval is None and tconf == 0.0, expression


def test_unanchored_fact_is_not_true_for_all_time(catalog):
    graph, _ = _memory(
        catalog,
        [_prop("p0", SPEAKER, "practices", "new:running", MAY, unanchored=True)],
    )
    (edge,) = graph.all_edges()
    assert edge.unanchored is True
    assert edge.t_valid.start == MAY, (
        "an unresolved date must fall back to the relation's volatility "
        "default, never to an interval that intersects every query"
    )
    assert edge.t_valid.end is None


def test_restrict_valid_does_not_reach_an_unanchored_fact(catalog):
    graph, _ = _memory(
        catalog,
        [_prop("p0", SPEAKER, "practices", "new:running", MAY, unanchored=True)],
    )
    state = restrict(
        anchor("Melanie", graph), "valid", Interval(datetime(2024, 1, 1), None)
    )
    state = follow(state, "practices", graph, catalog)
    assert state.trails == []
    assert state.death_cause == "unanchored_evidence"


def test_an_unrestricted_query_still_reaches_it(catalog):
    """The flag narrows validity queries only -- spec 5.4 keeps the
    proposition consultable, it does not hide it."""
    graph, _ = _memory(
        catalog,
        [_prop("p0", SPEAKER, "practices", "new:running", MAY, unanchored=True)],
    )
    state = follow(anchor("Melanie", graph), "practices", graph, catalog)
    assert len(state.trails) == 1


# --------------------------------------------------------------------- #
# 3. a repeated ASSERT wrote a second identical edge                      #
# --------------------------------------------------------------------- #


def test_restating_a_fact_reinforces_instead_of_duplicating(catalog):
    graph, _ = _memory(
        catalog,
        [
            _prop("p0", SPEAKER, "practices", "new:self-care", MAY, episode="e1"),
            _prop("p1", SPEAKER, "practices", "new:self-care", JUNE, episode="e2"),
        ],
    )
    edges = graph.all_edges()
    assert len(edges) == 1, "the same fact stated twice is one edge, confirmed twice"
    assert edges[0].reinforcement == 2
    assert edges[0].provenance == ["p0", "p1"]


def test_earlier_evidence_widens_the_edge_backwards(catalog):
    graph, _ = _memory(
        catalog,
        [
            _prop("p0", SPEAKER, "practices", "new:self-care", JUNE, episode="e1"),
            _prop(
                "p1",
                SPEAKER,
                "practices",
                "new:self-care",
                JUNE,
                t_valid=Interval(MAY, None),
                episode="e2",
            ),
        ],
    )
    (edge,) = graph.all_edges()
    assert edge.t_valid.start == MAY


def test_a_closed_interval_does_not_absorb_a_later_restatement(catalog):
    """ "Recife, then Salvador, then Recife again" is two intervals, and
    reinforcement must not silently erase the gap."""
    graph, _ = _memory(
        catalog,
        [
            _prop("p0", SPEAKER, "lives_in", "new:Recife", datetime(2023, 1, 1)),
            _prop("p1", SPEAKER, "lives_in", "new:Salvador", datetime(2023, 3, 1)),
            _prop("p2", SPEAKER, "lives_in", "new:Recife", datetime(2023, 9, 1)),
        ],
    )
    recife_id = next(e.id for e in graph.all_entities() if e.canonical_name == "Recife")
    recife_edges = [e for e in graph.all_edges() if e.dst_id == recife_id]
    assert len(recife_edges) == 2
    assert [e.t_valid.end is not None for e in recife_edges] == [True, False]


# --------------------------------------------------------------------- #
# 4. one name became two vertices when two relations disagreed on type    #
# --------------------------------------------------------------------- #


def test_one_name_is_one_vertex_across_a_type_class(catalog):
    graph, _ = _memory(
        catalog,
        [
            _prop("p0", SPEAKER, "practices", "new:charity race", MAY),
            _prop("p1", SPEAKER, "attended", "new:charity race", MAY),
        ],
    )
    named = [e for e in graph.all_entities() if e.canonical_name == "charity race"]
    assert len(named) == 1, (
        "'the charity race' is one thing whether it is reached as an "
        "Activity or as an Event"
    )


def test_both_relations_land_on_that_same_vertex(catalog):
    graph, _ = _memory(
        catalog,
        [
            _prop("p0", SPEAKER, "practices", "new:charity race", MAY),
            _prop("p1", SPEAKER, "attended", "new:charity race", MAY),
        ],
    )
    start = anchor("Melanie", graph)
    via_practices = follow(start, "practices", graph, catalog).trails[0].vertex_id
    via_attended = follow(start, "attended", graph, catalog).trails[0].vertex_id
    assert via_practices == via_attended


def test_place_and_organization_are_not_merged(catalog):
    """The type classes are narrow on purpose: "Vertex the company" and
    "Vertex the city" being one vertex would be a bug, not a repair."""
    assert not catalog.types_compatible("Place", "Organization")
    assert catalog.types_compatible("Activity", "Event")


# --------------------------------------------------------------------- #
# 5. mentions were never linked to an entity                              #
# --------------------------------------------------------------------- #


def test_mentions_are_linked_to_their_entity_by_consolidation(catalog):
    graph, staging = GraphStore(), StagingStore()
    mentions = MentionStore()
    mentions.append(episode_id="e1", surface="climbing", ts=MAY, proposition_id="p0")
    staging.insert([_prop("p0", SPEAKER, "practices", "new:climbing", MAY, episode="e1")])
    consolidate(catalog, graph, staging, ClioConfig.default(), mentions=mentions)

    (mention,) = mentions.all()
    assert mention.entity_id is not None
    assert graph.get_entity(mention.entity_id).canonical_name == "climbing"
    assert count(mentions, graph, entity="climbing") == 1


def test_relink_is_idempotent(catalog):
    graph, staging = GraphStore(), StagingStore()
    mentions = MentionStore()
    staging.insert([_prop("p0", SPEAKER, "practices", "new:climbing", MAY, episode="e1")])
    mentions.append(episode_id="e1", surface="climbing", ts=MAY, proposition_id="p0")
    cfg = ClioConfig.default()
    consolidate(catalog, graph, staging, cfg, mentions=mentions)
    first = mentions.all()[0].entity_id
    consolidate(catalog, graph, staging, cfg, mentions=mentions)
    assert mentions.all()[0].entity_id == first
    assert count(mentions, graph, entity="climbing") == 1


# --------------------------------------------------------------------- #
# 6. rebuild and order invariance (spec 12.3 and 17.4)                    #
# --------------------------------------------------------------------- #

SESSIONS = [
    [
        ("s1_p0", "works_at", "new:Vertex", datetime(2023, 1, 14)),
        ("s1_p1", "lives_in", "new:Recife", datetime(2023, 1, 14)),
    ],
    [
        ("s2_p0", "managed_by", "new:Bia", datetime(2023, 3, 2)),
        ("s2_p1", "practices", "new:climbing", datetime(2023, 3, 2)),
    ],
    [
        ("s3_p0", "lives_in", "new:Salvador", datetime(2023, 6, 20)),
        ("s3_p1", "practices", "new:climbing", datetime(2023, 6, 20)),
    ],
    [
        ("s4_p0", "works_at", "new:Kaia", datetime(2023, 9, 5)),
        ("s4_p1", "managed_by", "new:Rui", datetime(2023, 9, 5)),
    ],
]


def _build(catalog, session_order):
    graph, staging = GraphStore(), StagingStore()
    log, mentions = LogStore(), MentionStore()
    cfg = ClioConfig.default()
    for index in session_order:
        props = []
        for pid, relation, obj, ts in SESSIONS[index]:
            log.append(
                session_id=f"s{index}",
                speaker="Melanie",
                text="...",
                ts_ingest=ts,
                episode_id=pid,
            )
            props.append(_prop(pid, SPEAKER, relation, obj, ts))
        staging.insert(props)
        consolidate(catalog, graph, staging, cfg, mentions=mentions)
    return graph, staging


def test_order_invariance_over_shuffled_sessions(catalog):
    """Spec 17.4, the property the spec calls the most important one for
    publication: the CONSOLIDATED graph is a function of the episodes, not
    of the order the sessions arrived in. Turn order inside a session is
    preserved; only whole sessions move.
    """
    reference_graph, _ = _build(catalog, list(range(len(SESSIONS))))
    reference = canonical_form(reference_graph)
    assert reference, "the fixture must actually build a graph"

    rng = random.Random(20)
    for _ in range(20):
        order = list(range(len(SESSIONS)))
        rng.shuffle(order)
        graph, _ = _build(catalog, order)
        assert canonical_form(graph) == reference, f"order {order} diverged"
        assert canonical_entities(graph) == canonical_entities(reference_graph)


def test_rebuild_reproduces_the_same_graph(catalog):
    """Spec 12.3: everything derived is rebuildable from the log. The
    propositions are rewound to the references the extractor produced and
    replayed against an empty graph."""
    from fgl.clio.facade import Clio
    from fgl.llm.prompts import PromptLibrary
    from fgl.retrieval.embeddings import HashingEmbedder

    clio = Clio(
        catalog,
        llm=None,
        embedder=HashingEmbedder(dim=64),
        prompts=PromptLibrary("prompts"),
        config=ClioConfig.default(),
    )
    for session in SESSIONS:
        props = []
        for pid, relation, obj, ts in session:
            clio.log.append(
                session_id="s", speaker="Melanie", text="...", ts_ingest=ts, episode_id=pid
            )
            props.append(_prop(pid, SPEAKER, relation, obj, ts))
        clio.staging.insert(props)
        clio.consolidate()

    before = canonical_form(clio.graph)
    assert before

    clio.rebuild()
    # the rebuilt graph has its own fresh ids; only the canonical form
    # is comparable, which is exactly the point of having one
    assert canonical_form(clio.graph) == before


def test_consolidation_schedule_does_not_change_the_graph(catalog):
    """Canonicity's other axis: WHEN consolidation runs must not change
    what it produces. Session by session and all-at-once have to agree,
    including across a retraction -- phase 5 used to skip a dependent
    whose belief had already been closed, so batching the retraction into
    the same call left an interval open that the incremental schedule
    closed.
    """
    episodes = [
        [("e1_p0", "works_at", "new:Vertex", datetime(2023, 1, 14), Operation.ASSERT)],
        [("e2_p0", "managed_by", "new:Bia", datetime(2023, 3, 2), Operation.ASSERT)],
        [("e3_p0", "works_at", "new:Kaia", datetime(2023, 9, 5), Operation.ASSERT)],
        [("e4_p0", "managed_by", "new:Bia", datetime(2023, 12, 1), Operation.RETRACT)],
    ]

    def build(per_episode: bool):
        graph, staging = GraphStore(), StagingStore()
        cfg = ClioConfig.default()
        for episode in episodes:
            staging.insert(
                [
                    _prop(pid, SPEAKER, relation, obj, ts, operation=operation)
                    for pid, relation, obj, ts, operation in episode
                ]
            )
            if per_episode:
                consolidate(catalog, graph, staging, cfg)
        if not per_episode:
            consolidate(catalog, graph, staging, cfg)
        return graph

    incremental, batched = build(True), build(False)
    assert canonical_form(incremental) == canonical_form(batched)

    bia = next(
        e
        for e in incremental.all_edges()
        if incremental.get_entity(e.dst_id).canonical_name == "Bia"
    )
    assert bia.t_valid.end == datetime(2023, 9, 5), (
        "leaving the job ends who managed you there (phase 5), on the JOB's closing date"
    )
    assert bia.t_tx.end == datetime(2023, 12, 1), (
        "and the later retraction closes the belief -- a different axis, "
        "untouched by the above"
    )


# --------------------------------------------------------------------- #
# 7. an UNMAPPED item with no object id crashed ingestion                 #
# --------------------------------------------------------------------- #


def _unmapped_responder(shape: dict):
    import json

    def responder(prompt: str, system):
        return json.dumps({"propositions": [shape]})

    return responder


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            {
                "operation": "assert",
                "subject_id": "new:Melanie",
                "relation": "UNMAPPED",
                "suggested_relation": "event_was_thought_provoking",
                "object_id": None,
                "polarity": True,
                "time_expression": None,
                "evidence_kind": "literal",
                "span": "The event was really thought-provoking",
            },
            id="object_id present but null",
        ),
        pytest.param(
            {
                "operation": "assert",
                "subject_id": "new:Melanie",
                "relation": "UNMAPPED",
                "suggested_relation": "adopted_pet",
                "object_id": None,
                "object_surface": "a cat",
                "polarity": True,
                "time_expression": None,
                "evidence_kind": "literal",
                "span": "I adopted a cat last week",
            },
            id="object named in object_surface, spec 4.4's own shape",
        ),
    ],
)
def test_unmapped_without_an_object_id_does_not_crash_ingestion(catalog, shape):
    """A relation outside Sigma has no object id to give -- the model
    sends an explicit null, or names the thing in ``object_surface`` the
    way spec 4.4's own example does. ``.get(key, "")`` returns None for a
    key that is PRESENT and null, so the default never applied and the
    mention loop dereferenced it.
    """
    from fgl.clio.index import EntityIndex
    from fgl.clio.ingest.pipeline import ingest_turn
    from fgl.clio.unmapped import UnmappedQueue
    from fgl.config import LLMConfig
    from fgl.llm.client import FakeLLM
    from fgl.llm.prompts import PromptLibrary
    from fgl.retrieval.embeddings import HashingEmbedder

    graph, staging = GraphStore(), StagingStore()
    log, mentions = LogStore(), MentionStore()
    queue = UnmappedQueue()

    result = ingest_turn(
        text="I adopted a cat last week",
        speaker="Melanie",
        session_id="s1",
        ts=MAY,
        log=log,
        graph=graph,
        staging=staging,
        mentions=mentions,
        entity_index=EntityIndex(HashingEmbedder(dim=64)),
        catalog=catalog,
        llm=FakeLLM(
            LLMConfig(provider="fake", cache_enabled=False),
            responder=_unmapped_responder(shape),
        ),
        prompts=PromptLibrary("prompts"),
        unmapped_queue=queue,
    )

    assert result.propositions == []
    assert len(result.unmapped) == 1
    entry = result.unmapped[0]
    assert entry.suggested_relation == shape["suggested_relation"]
    assert isinstance(entry.object_ref, str), "never None -- the mention loop indexes it"
    # the surface, when the model gave one, still reaches the mention
    # table: a turn the catalog cannot express still MENTIONED the thing
    expected = shape.get("object_surface")
    surfaces = [m.surface for m in mentions.all()]
    assert surfaces == ([expected] if expected else [])


# --------------------------------------------------------------------- #
# 8. the fold scorer was inverted for dialogue                            #
# --------------------------------------------------------------------- #


def _fold_memory(catalog, props):
    """A memory with folding enabled, fed hand-classified propositions."""
    from fgl.clio.consolidate.journal import FoldJournal

    graph, staging = GraphStore(), StagingStore()
    log, journal = LogStore(), FoldJournal()
    episodes = {p.episode_id for p in props}
    for eid in sorted(episodes):
        log.append(
            session_id="s", speaker="Melanie", text="...", ts_ingest=MAY, episode_id=eid
        )
    staging.insert(props)
    report = consolidate(
        catalog, graph, staging, ClioConfig.default(), log=log, journal=journal
    )
    return graph, report


def test_a_nickname_folds_into_the_full_name(catalog):
    """ "Mel" and "Melanie" are one person. The old scorer topped out at
    0.765 for this pair even with both structural and role evidence,
    permanently under tau_fold's 0.80, because `contains_as_token`
    demanded whole-token containment and a shortened first name scores 0
    there. Measured on conv-26: she was two people, each holding half her
    facts."""
    graph, report = _fold_memory(
        catalog,
        [
            _prop("p0", "new:Melanie", "attended", "new:charity race", MAY, episode="e1"),
            _prop("p1", "new:Mel", "attended", "new:charity race", MAY, episode="e2"),
        ],
    )
    people = [
        e for e in graph.all_entities() if e.type == "Person" and e.merged_into is None
    ]
    assert len(people) == 1, [e.canonical_name for e in people]
    assert report.folded, "the merge should be journalled, not silent"
    assert "mel" in {a.lower() for a in people[0].aliases} | {
        people[0].canonical_name.lower()
    }


def test_a_common_noun_is_not_absorbed_by_a_phrase_that_mentions_it(catalog):
    """ "family" is not "Our family and moments". The old scorer gave that
    pair 0.9 for substring similarity AND a further 1.0 for token
    containment -- one weak piece of evidence counted twice -- and merged
    them at 0.86 on conv-26."""
    graph, report = _fold_memory(
        catalog,
        [
            _prop("p0", SPEAKER, "likes", "new:family", MAY, episode="e1"),
            _prop("p1", SPEAKER, "likes", "new:Our family and moments", MAY, episode="e2"),
        ],
    )
    names = {e.canonical_name for e in graph.all_entities() if e.merged_into is None}
    assert {"family", "Our family and moments"} <= names
    assert report.folded == []


def test_the_specs_own_rui_example_still_folds(catalog):
    """Spec 8.3's worked example must not regress: a surname appended to a
    known first name is the same person."""
    from fgl.clio.consolidate.fold import contains_as_token, name_similarity

    assert name_similarity("Rui", "Rui Sampaio") == 0.9
    assert contains_as_token("Rui", "Rui Sampaio") == 1.0


# --------------------------------------------------------------------- #
# 9. an inverted subject/object reached the graph as a Person             #
# --------------------------------------------------------------------- #


def test_a_description_cannot_be_created_as_a_person(catalog):
    """`likes` is [Person, Topic]; "transgender stories | likes |
    Caroline" put the topic in the subject slot, and phase 1 created a
    vertex of type Person called "transgender stories" (observed on
    conv-26). A `new:` ref has no type yet, so the signature check could
    not see it -- the one property a name has and a description does not
    is a capital letter."""
    from fgl.clio.ingest.validate import validate_and_bind

    graph = GraphStore()
    log = LogStore()
    episode = log.append(
        session_id="s",
        speaker="Caroline",
        text="The transgender stories were so inspiring!",
        ts_ingest=MAY,
    )
    inverted = {
        "operation": "assert",
        "subject_id": "new:transgender stories",
        "relation": "likes",
        "object_id": "new:Caroline",
        "polarity": True,
        "evidence_kind": "literal",
        "span": "The transgender stories were so inspiring!",
    }
    result = validate_and_bind([inverted], episode, graph, catalog)
    assert result.valid == []
    assert [r.reason for r in result.rejected] == ["person_ref_lacks_a_name"]


def test_a_real_name_in_a_person_slot_still_passes(catalog):
    from fgl.clio.ingest.validate import validate_and_bind

    graph = GraphStore()
    log = LogStore()
    episode = log.append(
        session_id="s", speaker="Caroline", text="I like the stories", ts_ingest=MAY
    )
    ok = {
        "operation": "assert",
        "subject_id": "new:Caroline",
        "relation": "likes",
        "object_id": "new:the stories",
        "polarity": True,
        "evidence_kind": "literal",
        "span": "I like the stories",
    }
    result = validate_and_bind([ok], episode, graph, catalog)
    assert len(result.valid) == 1
    assert result.rejected == []


# --------------------------------------------------------------------- #
# 10. folding created duplicate edges and nothing reconciled them         #
# --------------------------------------------------------------------- #


def test_folding_two_destinations_leaves_one_edge_not_two(catalog):
    """Two edges distinct only because their destinations were distinct
    become one fact once those destinations merge. Spec 8.3 migrates the
    edges and stops; phase 4 skips pairs sharing a destination, so nothing
    reconciled them and conv-26 ended with "Melanie likes Our family and
    moments" twice, each claiming reinforcement 1."""
    graph, report = _fold_memory(
        catalog,
        [
            _prop("p0", SPEAKER, "practices", "new:Rui", MAY, episode="e1"),
            _prop("p1", SPEAKER, "practices", "new:Rui Sampaio", MAY, episode="e2"),
        ],
    )
    assert report.folded, "the two destinations should merge"
    practices = [e for e in graph.all_edges() if e.label == "practices"]
    assert len(practices) == 1, "one fact, one edge"
    assert practices[0].reinforcement == 2
    assert sorted(practices[0].provenance) == ["p0", "p1"]


# --------------------------------------------------------------------- #
# 11. extraction ran on the repository-wide 512-token default             #
# --------------------------------------------------------------------- #


def test_extraction_asks_for_a_budget_that_fits_several_propositions():
    """One proposition serialises to roughly 90 tokens, so 512 truncates a
    turn yielding six of them mid-object and the whole response fails to
    parse -- silently becoming "0 propositions", and it is the RICH turns
    that get lost. 3 of 58 turns on conv-26 were lost exactly this way."""
    from fgl.clio.config import ClioConfig
    from fgl.clio.ingest.context import ExtractionContext
    from fgl.clio.ingest.extractor import extract_propositions
    from fgl.llm.prompts import PromptLibrary

    assert ClioConfig.default().extraction.max_tokens >= 2000

    seen = {}

    class _Recorder:
        def complete_json(self, prompt, **kw):
            seen.update(kw)
            return {"propositions": []}

    log = LogStore()
    episode = log.append(session_id="s", speaker="M", text="hi", ts_ingest=MAY)
    extract_propositions(
        _Recorder(),
        PromptLibrary("prompts"),
        ExtractionContext(episode, [], [], []),
        max_tokens=2000,
    )
    assert seen["max_tokens"] == 2000


# --------------------------------------------------------------------- #
# 12. the person guard was too wide; type mismatches were dropped whole   #
# --------------------------------------------------------------------- #


def _validate_one(catalog, item, text="I said something", graph=None):
    from fgl.clio.ingest.validate import validate_and_bind

    graph = graph if graph is not None else GraphStore()
    episode = LogStore().append(
        session_id="s", speaker="Caroline", text=text, ts_ingest=MAY
    )
    return validate_and_bind([item], episode, graph, catalog), graph


def _item(**kw):
    base = {
        "operation": "assert",
        "subject_id": "new:Caroline",
        "relation": "likes",
        "object_id": "new:pottery",
        "polarity": True,
        "evidence_kind": "literal",
        "span": "I said something",
    }
    base.update(kw)
    return base


def test_a_vague_common_noun_is_allowed_as_an_object(catalog):
    """P5: the answer is written from the EPISODE; the proposition only
    has to LOCATE it. A vertex named "friends" is imprecise but it anchors
    the right turn, and that is its whole job. Applying the person-name
    guard to objects made it the largest source of rejections on conv-26
    (24 of 47) and cost real evidence coverage."""
    result, _ = _validate_one(
        catalog,
        _item(relation="friend_of", object_id="new:friends"),
        text="My friends have been there through everything",
    )
    assert len(result.valid) == 1
    assert result.rejected == []


def test_a_description_in_the_subject_slot_is_still_refused(catalog):
    """The subject is different in kind: it is the write ADDRESS, and a
    mistyped one corrupts the type system rather than merely naming
    something loosely."""
    result, _ = _validate_one(
        catalog,
        _item(subject_id="new:transgender stories", object_id="new:Caroline"),
        text="The transgender stories were inspiring",
    )
    assert result.valid == []
    assert [r.reason for r in result.rejected] == ["person_ref_lacks_a_name"]


def test_a_mistyped_id_is_rebound_when_folding_can_reconcile_it(catalog):
    """The extractor naming the right THING under an id carrying the wrong
    type used to lose the whole proposition. Activity/Event/Topic form a
    declared type class, so a fresh vertex folds back into the mistyped
    one -- the fact survives and the entity does not stay split."""
    graph = GraphStore()
    # an Organization is NOT interchangeable with an Activity, so this is
    # a real signature violation -- unlike Event, which shares Activity's
    # type class and needs no rebinding at all
    program = graph.create_entity("mentorship program", "Organization")
    result, _ = _validate_one(
        catalog,
        _item(relation="practices", object_id=program.id),
        text="I joined a mentorship program",
        graph=graph,
    )
    assert len(result.valid) == 1
    assert result.valid[0]["object_id"] == "new:mentorship program"
    assert result.rebindings == 1


def test_a_mistyped_id_is_still_refused_where_nothing_can_reconcile_it(catalog):
    """Place and Organization form no type class and never fold, so
    rebinding "I live in Vertex" would mint a place called Vertex beside
    the company -- the identity error the narrow classes exist to
    prevent. Rejection is the right answer there."""
    graph = GraphStore()
    vertex = graph.create_entity("Vertex", "Organization")
    result, _ = _validate_one(
        catalog,
        _item(relation="lives_in", object_id=vertex.id),
        text="I live in Vertex",
        graph=graph,
    )
    assert result.valid == []
    assert result.rebindings == 0
    assert [r.reason for r in result.rejected] == ["object_type_violates_signature"]


# --------------------------------------------------------------------- #
# 13. persistence and the access ceilings (the reading-side instruments)  #
# --------------------------------------------------------------------- #


def test_a_memory_round_trips_through_disk(catalog, tmp_path):
    """Every access experiment used to cost re-ingesting a whole
    conversation. This is what makes the reading side debuggable offline,
    so it has to round-trip EXACTLY -- the canonical form is the test,
    because it is id-free and covers polarity, both clocks and the
    unanchored flag."""
    from fgl.clio.facade import Clio
    from fgl.clio.persist import load_memory, save_memory
    from fgl.llm.prompts import PromptLibrary
    from fgl.retrieval.embeddings import HashingEmbedder

    clio = Clio(
        catalog,
        llm=None,
        embedder=HashingEmbedder(dim=64),
        prompts=PromptLibrary("prompts"),
        config=ClioConfig.default(),
    )
    for pid, relation, obj, ts in [
        ("e1_p0", "works_at", "new:Vertex", datetime(2023, 1, 14)),
        ("e2_p0", "managed_by", "new:Bia", datetime(2023, 3, 2)),
        ("e3_p0", "works_at", "new:Kaia", datetime(2023, 9, 5)),
        ("e4_p0", "likes", "new:running", datetime(2023, 9, 5)),
    ]:
        clio.log.append(
            session_id="s", speaker="Melanie", text="...", ts_ingest=ts, episode_id=pid
        )
        clio.staging.insert([_prop(pid, SPEAKER, relation, obj, ts)])
        clio.mentions.append(episode_id=pid, surface=obj[4:], ts=ts, proposition_id=pid)
        clio.consolidate()

    before = canonical_form(clio.graph)
    assert before

    path = save_memory(clio, tmp_path / "mem.json")
    restored = load_memory(path, catalog)

    assert canonical_form(restored.graph) == before
    assert canonical_entities(restored.graph) == canonical_entities(clio.graph)
    assert len(restored.staging.all()) == len(clio.staging.all())
    assert [m.entity_id for m in restored.mentions.all()] == [
        m.entity_id for m in clio.mentions.all()
    ]


def test_a_restored_memory_keeps_minting_fresh_ids(catalog, tmp_path):
    """A loaded store that restarted its counters would hand out ids that
    collide with the ones it just read."""
    from fgl.clio.facade import Clio
    from fgl.clio.persist import load_memory, save_memory
    from fgl.llm.prompts import PromptLibrary
    from fgl.retrieval.embeddings import HashingEmbedder

    clio = Clio(catalog, None, HashingEmbedder(dim=64), PromptLibrary("prompts"))
    clio.log.append(session_id="s", speaker="M", text="...", ts_ingest=MAY, episode_id="e1")
    clio.staging.insert([_prop("e1_p0", SPEAKER, "likes", "new:running", MAY)])
    clio.consolidate()

    restored = load_memory(save_memory(clio, tmp_path / "m.json"), catalog)
    existing = {e.id for e in restored.graph.all_entities()}
    fresh = restored.graph.create_entity("Someone", "Person")
    assert fresh.id not in existing


def test_the_ceilings_are_nested(catalog, tmp_path):
    """extraction >= promotion >= reachability, by construction: a
    proposition cannot reach the graph without existing, and an edge
    cannot be walked to without being in the graph. If this ever inverts,
    the measurement is lying."""
    from fgl.clio.access_audit import reachable_episodes
    from fgl.clio.facade import Clio
    from fgl.llm.prompts import PromptLibrary
    from fgl.retrieval.embeddings import HashingEmbedder

    clio = Clio(catalog, None, HashingEmbedder(dim=64), PromptLibrary("prompts"))
    for pid, relation, obj, ts in [
        ("e1", "works_at", "new:Vertex", datetime(2023, 1, 14)),
        ("e2", "managed_by", "new:Bia", datetime(2023, 3, 2)),
    ]:
        clio.log.append(
            session_id="s", speaker="Melanie", text="...", ts_ingest=ts, episode_id=pid
        )
        clio.staging.insert([_prop(pid, SPEAKER, relation, obj, ts)])
        clio.consolidate()

    melanie = next(e for e in clio.graph.all_entities() if e.canonical_name == "Melanie")
    reached = reachable_episodes(clio, [melanie.id], max_hops=6)
    assert reached == {"e1", "e2"}

    promoted_ids = {p for e in clio.graph.all_edges() for p in e.provenance}
    assert reached <= {clio.staging.get(p).episode_id for p in promoted_ids}


def test_an_unpromoted_proposition_is_invisible_to_the_reader(catalog):
    """The gate gate 1 deliberately did not measure. A single implicature
    scores 0.55, below tau_promote, so it stays in staging -- and the
    reader walks EDGES, so its episode can never be materialised. Gate 1
    counts that turn as covered; the reader cannot use it."""
    from fgl.clio.access_audit import reachable_episodes
    from fgl.clio.facade import Clio
    from fgl.llm.prompts import PromptLibrary
    from fgl.retrieval.embeddings import HashingEmbedder

    clio = Clio(catalog, None, HashingEmbedder(dim=64), PromptLibrary("prompts"))
    clio.log.append(
        session_id="s", speaker="Melanie", text="...", ts_ingest=MAY, episode_id="e1"
    )
    clio.staging.insert(
        [
            _prop(
                "e1_p0",
                SPEAKER,
                "practices",
                "new:climbing",
                MAY,
                kind=EvidenceKind.IMPLICATURE,
                confidence=0.55,
            )
        ]
    )
    clio.consolidate()

    assert clio.graph.all_edges() == [], "0.55 is below tau_promote 0.70"
    assert clio.staging.all()[0].status == "staged"
    people = [e for e in clio.graph.all_entities() if e.type == "Person"]
    assert reachable_episodes(clio, [e.id for e in people], max_hops=6) == set()


def test_the_ceilings_separate_promotion_from_reachability(catalog, tmp_path):
    """The instrument itself needs a case with a known answer, or it can
    report a plausible-looking lie. Two evidence turns: one whose
    proposition reaches the graph, one that stays in staging at 0.55.
    Extraction must see both, promotion only one, and reachability must
    follow promotion -- and it must survive a round-trip through disk,
    because that is how every future experiment will read it."""
    from fgl.clio.access_audit import measure_ceilings
    from fgl.clio.facade import Clio
    from fgl.clio.persist import load_memory, save_memory
    from fgl.data.locomo import Conversation, Question, Session, Turn
    from fgl.llm.prompts import PromptLibrary
    from fgl.retrieval.embeddings import HashingEmbedder

    clio = Clio(catalog, None, HashingEmbedder(dim=64), PromptLibrary("prompts"))
    for pid, relation, obj, kind, conf in [
        ("D1:1", "works_at", "new:Vertex", EvidenceKind.LITERAL, 0.9),
        ("D1:2", "practices", "new:climbing", EvidenceKind.IMPLICATURE, 0.55),
    ]:
        clio.log.append(
            session_id="s", speaker="Melanie", text="...", ts_ingest=MAY, episode_id=pid
        )
        clio.staging.insert(
            [
                _prop(
                    pid + "_p",
                    SPEAKER,
                    relation,
                    obj,
                    MAY,
                    kind=kind,
                    confidence=conf,
                    episode=pid,
                )
            ]
        )
    clio.consolidate()

    conv = Conversation(
        sample_id="synth",
        speaker_a="Melanie",
        speaker_b="Bia",
        sessions=[
            Session(
                num=1,
                date_time_raw="",
                timestamp=MAY.isoformat(),
                turns=[
                    Turn(
                        dia_id="D1:1",
                        session_num=1,
                        speaker="Melanie",
                        text="I work at Vertex",
                    ),
                    Turn(
                        dia_id="D1:2",
                        session_num=1,
                        speaker="Melanie",
                        text="climbing again",
                    ),
                ],
            )
        ],
        questions=[
            Question(
                question="Where does Melanie work?",
                answer="Vertex",
                category=1,
                evidence=["D1:1"],
            ),
            Question(
                question="What does Melanie do?",
                answer="climbing",
                category=1,
                evidence=["D1:2"],
            ),
        ],
    )

    report = measure_ceilings(conv, clio, max_hops=6).to_dict()
    assert report["ceilings"] == {
        "extraction": 1.0,
        "promotion": 0.5,
        "reachability": 0.5,
    }

    restored = load_memory(save_memory(clio, tmp_path / "m.json"), catalog)
    assert measure_ceilings(conv, restored, max_hops=6).to_dict() == report
