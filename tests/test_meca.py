"""MECA: attested propositions, the two readers, and the identity that makes
the ribbon comparison mean anything.

What is pinned here, in the order it matters:

* **the unit carries what it claims to carry** -- modality, polarity, two
  clocks, required evidence -- and a claim without a locatable span never
  enters the memory;
* **verification is load-bearing**: a verifier that rejects a claim removes it,
  and a malformed verdict rejects rather than accepts;
* **consolidation turns claims into a state**: repetition becomes confidence,
  a later claim on a single-valued predicate closes the earlier one, and
  "single-valued" is measured on the corpus rather than declared;
* **one store, two readers**: with its two degenerate settings the ribbon
  reader reproduces the flat reader FACT FOR FACT. Without that identity the
  M1-vs-M2 comparison would be worth nothing -- the same discipline that proved
  L5 reduces to L2.

Design: ``docs/MECA_DESIGN.md``.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from fgl.config import Config, ConfigError
from fgl.memory.consolidate import (
    consolidate,
    link_propositions,
    predicate_functionality,
)
from fgl.memory.propositions import (
    MOD_ASSERTED,
    MOD_INTENDED,
    Evidence,
    Proposition,
    PropositionStore,
    TimePoint,
    build_graph,
    coerce_proposition,
    corners,
    normalise,
    orbit_pids,
    parse_modality,
    proposition_from_dict,
    store_from_graph,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _ev(doc_id="D1:1", span="said it", ts="2023-05-08T00:00:00", raw="8 May, 2023"):
    return Evidence(doc_id=doc_id, span=span, timestamp=ts, date_raw=raw)


def _prop(subject, predicate, obj="", when="2023-05", modality=MOD_ASSERTED,
          polarity=True, doc_id="D1:1", span="said it", **kw):
    p = Proposition(
        subject=subject, predicate=predicate, object=obj,
        valid_from=TimePoint.parse(when),
        asserted_at=TimePoint.parse(when),
        modality=modality, polarity=polarity,
        evidence=[_ev(doc_id, span)],
        **kw,
    )
    p.pid = p.make_pid()
    return p


class _Embedder:
    """Deterministic bag-of-words embedder: no model download, real cosines.

    ``zlib.crc32``, never ``hash()``: string hashing is salted per process, so
    a helper built on it makes every ordering assertion below pass or fail
    depending on the interpreter's start-up seed. That is the same trap the
    shuffle ablation in ``faces.py`` documents, and it cost one flaky run here
    before the switch.
    """

    dim = 64

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in normalise(str(t)).split():
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def encode_one(self, text):
        return self.encode([text])[0]


# --------------------------------------------------------------------------- #
# The unit                                                                     #
# --------------------------------------------------------------------------- #


def test_time_is_a_prefix_not_a_datetime():
    """"Sometime in 2023" is a fact; widening it to 1 January is a fabrication."""
    year = TimePoint.parse("2023")
    month = TimePoint.parse("2023-05")
    assert year.precision == "year" and month.precision == "month"
    assert year.contains(month) and not month.contains(year)
    assert year.overlaps(month)
    assert TimePoint.parse("") is None and TimePoint.parse(None) is None


def test_the_two_clocks_are_separate():
    """"Last month I quit", said in June, is asserted in June and true in May."""
    p = Proposition(
        subject="Ana", predicate="quit", object="the job",
        valid_from=TimePoint.parse("2023-05"),
        asserted_at=TimePoint.parse("2023-06"),
        evidence=[_ev()],
    )
    assert p.when().value == "2023-05", "when it was TRUE wins for retrieval"
    assert p.asserted_at.value == "2023-06"


def test_modality_and_polarity_are_part_of_the_claim():
    """A plan and a fact are different claims, not one claim seen twice."""
    fact = _prop("Ana", "move to", "Lisbon")
    plan = _prop("Ana", "move to", "Lisbon", modality=MOD_INTENDED)
    negated = _prop("Ana", "move to", "Lisbon", polarity=False)
    assert fact.signature() != plan.signature() != negated.signature()
    assert fact.is_factual and not plan.is_factual


def test_an_unknown_modality_becomes_hypothetical_never_asserted():
    """The direction of the error is the point.

    A wrong `asserted` is a hallucination; a wrong `hypothetical` is only a
    missed answer. A model that answers "plan" instead of the enum must not
    silently produce a fact.
    """
    assert parse_modality("plan") == MOD_INTENDED
    assert parse_modality("something the model made up") == "hypothetical"
    assert parse_modality("asserted") == MOD_ASSERTED


def test_qualifiers_are_open_content_under_a_fixed_schema():
    """The answer to "too many slot types": roles, not vertex kinds."""
    p = _prop("Ana", "cooked", "chicken")
    p.qualifiers = {"location": "at home", "with": "Bruno"}
    roles = [r for r, _, _ in p.roles()]
    assert roles[:3] == ["subject", "predicate", "object"]
    assert "location" in roles and "with" in roles
    assert roles[-1] == "time", "time closes the rotation"


def test_a_claim_without_a_locatable_span_is_dropped():
    assert coerce_proposition({"predicate": "moved"}, [_ev()], None) is None
    assert coerce_proposition({"subject": "Ana"}, [_ev()], None) is None
    ok = coerce_proposition(
        {"subject": "Ana", "predicate": "moved", "object": "Lisbon"}, [_ev()], None
    )
    assert ok is not None and ok.evidence


def test_label_says_what_kind_of_claim_this_is():
    p = _prop("Ana", "move to", "Lisbon", modality=MOD_INTENDED)
    assert "intended" in p.label()
    p.superseded_by = "pX"
    assert "superseded" in p.label()


def test_a_proposition_round_trips_through_its_dict():
    """The graph is the only artifact, so this is what makes caching safe."""
    p = _prop("Ana", "moved to", "Lisbon")
    p.qualifiers = {"with": "Bruno"}
    back = proposition_from_dict(p.as_dict())
    assert back.signature() == p.signature()
    assert back.when().value == p.when().value
    assert [e.doc_id for e in back.evidence] == [e.doc_id for e in p.evidence]


# --------------------------------------------------------------------------- #
# The store                                                                    #
# --------------------------------------------------------------------------- #


def test_repetition_becomes_confidence_not_a_duplicate():
    """In every earlier model here a repeated claim became a second unit
    competing for the same token budget."""
    store = PropositionStore()
    a = _prop("Ana", "moved to", "Lisbon", doc_id="D1:1")
    b = _prop("Ana", "moved to", "Lisbon", doc_id="D9:2")
    store.add(a)
    store.add(b)
    assert len(store) == 2, "different evidence -> different pid before dedup"
    from fgl.memory.propositions import merge_into

    merge_into(a, b)
    assert len(a.evidence) == 2, "one claim, two spans"
    assert {e.doc_id for e in a.evidence} == {"D1:1", "D9:2"}
    # support is counted, never inflated: a number that crept toward 1.0 every
    # time two records touched would read as certainty nothing earned
    assert a.confidence == b.confidence == 1.0


def test_knowing_an_entity_and_knowing_a_claim_are_different_questions():
    """The distinction abstention actually needs, and the reason it is a
    lookup here rather than a score with a threshold."""
    store = PropositionStore()
    store.add(_prop("Ana", "moved to", "Lisbon"))
    assert store.knows_entity("Ana") is True
    assert store.knows_entity("Melanie") is False
    assert store.with_predicate("painted") == []


def test_about_is_chronological():
    store = PropositionStore()
    store.add(_prop("Ana", "moved to", "Lisbon", when="2023-09"))
    store.add(_prop("Ana", "left", "Porto", when="2023-02"))
    assert [p.when().value for p in store.about("Ana")] == ["2023-02", "2023-09"]


# --------------------------------------------------------------------------- #
# Consolidation                                                                #
# --------------------------------------------------------------------------- #


def test_functionality_is_measured_on_the_corpus_not_declared():
    """"lives in" admits one value at a time and "read" does not -- and the
    corpus says so, with no ontology and no gold label."""
    store = PropositionStore()
    for subj, city in (("Ana", "Lisbon"), ("Bruno", "Porto"), ("Cara", "Faro")):
        store.add(_prop(subj, "lives in", city))
    for book in ("Dune", "Emma", "Ulysses"):
        store.add(_prop("Ana", "read", book))

    f = predicate_functionality(store)
    assert f["lives in"] == 1.0
    assert f["read"] < 0.5


def test_a_later_claim_closes_an_earlier_one_on_a_functional_predicate():
    store = PropositionStore()
    for subj, city in (("Bruno", "Porto"), ("Cara", "Faro"), ("Dinis", "Braga")):
        store.add(_prop(subj, "lives in", city))
    old = _prop("Ana", "lives in", "Lisbon", when="2022-01")
    new = _prop("Ana", "lives in", "Berlin", when="2023-08")
    store.add(old)
    store.add(new)

    consolidate(store, _Embedder(), resolve=False, dedupe=False, timeline=True)
    assert old.superseded_by == new.pid
    assert old.valid_to is not None and old.valid_to.value == "2023-08"
    assert new.is_current, "the latest claim stays current"


def test_supersession_never_fires_on_a_multi_valued_predicate():
    """Keeping two facts costs budget; deleting the true one costs the answer."""
    store = PropositionStore()
    for book in ("Dune", "Emma", "Ulysses", "Persuasion"):
        store.add(_prop("Ana", "read", book, when=f"2023-0{len(book) % 8 + 1}"))
    consolidate(store, _Embedder(), resolve=False, dedupe=False, timeline=True)
    assert all(p.is_current for p in store), "a reading list is not a state"


def test_a_contradiction_is_kept_and_flagged_not_resolved():
    store = PropositionStore()
    a = _prop("Ana", "works at", "Acme", when="2023-03")
    b = _prop("Ana", "works at", "Globex", when="2023-03")
    store.add(a)
    store.add(b)
    elaborations, contradictions = link_propositions(store)
    assert contradictions == 1
    assert a.is_current and b.is_current, "both sides survive"


def test_entity_resolution_collapses_short_forms():
    store = PropositionStore()
    store.add(_prop("Melanie Carter", "painted", "a mural"))
    store.add(_prop("Melanie", "joined", "a class"))
    consolidate(store, _Embedder(), resolve=True, dedupe=False, timeline=False)
    subjects = {normalise(p.subject) for p in store}
    assert subjects == {"melanie carter"}, "the longest surface wins"


# --------------------------------------------------------------------------- #
# The fatgraph view                                                            #
# --------------------------------------------------------------------------- #


def _store_with_two_claims():
    store = PropositionStore()
    store.add(_prop("Ana", "adopted", "a pup", when="2023-02", doc_id="D1:3"))
    store.add(_prop("Ana", "moved to", "Lisbon", when="2023-09", doc_id="D7:1"))
    return store


def test_the_rotation_at_a_proposition_is_the_argument_order():
    graph = build_graph(_store_with_two_claims(), _Embedder())
    pvid = next(
        vid for vid, vx in graph.vertices.items()
        if vx.meta.get("kind") == "proposition"
    )
    pairs = corners(graph, pvid)
    roles = [a for a, _ in pairs]
    assert roles[0] == "subject" and roles[1] == "predicate"
    assert "time" in roles, "the time slot closes the ring"


def test_an_entity_orbit_is_chronological_with_no_sorting():
    """The ribbon claim, stated as a property of the build."""
    store = _store_with_two_claims()
    graph = build_graph(store, _Embedder())
    evid = next(
        vid for vid, vx in graph.vertices.items()
        if vx.meta.get("kind") == "entity" and vx.meta.get("key") == "ana"
    )
    pids = orbit_pids(graph, evid)
    whens = [store.get(p).when().value for p in pids]
    assert whens == sorted(whens)


def test_the_graph_reconstructs_the_store_exactly():
    store = _store_with_two_claims()
    back = store_from_graph(build_graph(store, _Embedder()))
    assert {p.signature() for p in back} == {p.signature() for p in store}


def test_the_store_survives_a_save_and_load(tmp_path):
    """The caching path, which is where a silent failure would live.

    The graph is the ONLY artifact: every proposition rides in its own vertex's
    meta. A run that reuses `artifacts/graphs/` has to get back the memory the
    ingest built -- aliases, supersession and all -- or the second condition is
    reading something the first never wrote.
    """
    from fgl.core import FatGraph

    store = _store_with_two_claims()
    store.add(_prop("Ana", "lives in", "Porto", when="2021-01"))
    graph = build_graph(store, _Embedder())
    graph.save(tmp_path / "g")
    back = store_from_graph(FatGraph.load(tmp_path / "g"))

    assert {p.signature() for p in back} == {p.signature() for p in store}
    assert all(p.evidence for p in back), "provenance survives the round trip"
    assert {p.when().value for p in back} == {p.when().value for p in store}


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


def test_the_two_conditions_differ_only_in_the_reader():
    m1 = Config.load("M1_meca_flat")
    m2 = Config.load("M2_meca_ribbon")
    a, b = dict(m1.meca.__dict__), dict(m2.meca.__dict__)
    differing = {k for k in a if a[k] != b[k]}
    assert differing <= {"reader", "ribbon_order", "ribbon_join", "ribbon_walk_max"}
    assert m1.meca.reader == "flat" and m2.meca.reader == "ribbon"
    assert m1.ingest.mode == m2.ingest.mode == "meca"


def test_m2_borrows_m1_graphs():
    """One store, two readers -- enforced by the config, not by good intentions."""
    assert Config.load("M2_meca_ribbon").paths.graphs_condition == "M1-meca-flat"


def test_meca_refuses_to_be_half_configured():
    with pytest.raises(ConfigError, match="switched together"):
        Config.load("M1_meca_flat", overrides=["retrieval.mode=slots"])


@pytest.mark.parametrize(
    "override,match",
    [
        ("meca.reader=magic", "meca.reader"),
        ("meca.ribbon_order=magic", "ribbon_order"),
        ("meca.ribbon_join=magic", "ribbon_join"),
        ("meca.passage_quantile=1.5", "passage_quantile"),
        ("meca.entity_quantile=2.0", "entity_quantile"),
    ],
)
def test_meca_config_is_validated(override, match):
    with pytest.raises(ConfigError, match=match):
        Config.load("M1_meca_flat", overrides=[override])


def test_meca_brings_its_own_answer_prompt():
    """`answer.txt` takes {speaker_a}/{speaker_b} and routes by question shape.
    Both are properties of this benchmark, and a general method carries neither.
    """
    cfg = Config.load("M1_meca_flat")
    assert cfg.retrieval.answer_prompt == "meca_answer"
    assert Config.load("L2d").retrieval.answer_prompt == "", "L-line unchanged"


# --------------------------------------------------------------------------- #
# The two readers                                                              #
# --------------------------------------------------------------------------- #


def _memory():
    """A store whose only multi-hop answer needs a join through a connector.

    Ana is linked to the mural; Bruno is linked to the mural too. Nothing links
    Ana to Bruno directly, so "who saw Ana's work" is answerable only by going
    Ana -> mural -> Bruno. That is the shape the whole method exists for.
    """
    store = PropositionStore()
    store.add(_prop("Ana", "painted", "the mural", when="2023-02", doc_id="D1:3",
                    span="I finally painted the mural"))
    store.add(_prop("Ana", "moved to", "Lisbon", when="2023-09", doc_id="D7:1",
                    span="we moved to Lisbon in September"))
    store.add(_prop("Bruno", "photographed", "the mural", when="2023-06",
                    doc_id="D5:2", span="I photographed the mural downtown"))
    store.add(_prop("Cara", "cooked", "risotto", when="2023-04", doc_id="D3:1",
                    span="I cooked risotto again"))
    return store


def _retriever(reader, **overrides):
    from fgl.retrieval.meca import MecaRetriever

    cfg = Config.load("M1_meca_flat" if reader == "flat" else "M2_meca_ribbon")
    for key, value in overrides.items():
        setattr(cfg.meca, key, value)
    store = _memory()
    graph = build_graph(store, _Embedder())
    return MecaRetriever(graph, _Embedder(), cfg, {})


def test_the_reader_finds_what_the_memory_directly_holds():
    r = _retriever("flat")
    result = r.retrieve("What did Ana paint?")
    assert result.facts
    assert any("mural" in f.text for f in result.facts)
    assert "D1:3" in result.turn_ids


def test_the_emitted_fact_carries_the_claim_and_its_source():
    """Statement to compose over, span to quote and to check it against."""
    r = _retriever("flat")
    texts = [f.text for f in r.retrieve("What did Ana paint?").facts]
    mural = next(t for t in texts if "mural" in t)
    assert "[2023-02, stated]" in mural, "the claim, labelled"
    assert "I finally painted the mural" in mural, "and the exact source text"


def test_the_join_reaches_the_second_hop_through_a_connector():
    r = _retriever("flat")
    result = r.retrieve("Who photographed what Ana painted?")
    joined = [f for f in result.facts if f.source == "meca_join"]
    assert joined, "the second step of the plan must fire"
    assert any("Bruno" in f.text for f in joined)
    assert any(f.via_entity for f in joined), "the connector is named"


def test_the_join_can_be_switched_off():
    r = _retriever("flat", join_steps=0)
    result = r.retrieve("Who photographed what Ana painted?")
    assert not [f for f in result.facts if f.source == "meca_join"]


def test_an_unknown_entity_is_reported_as_unknown():
    """A lookup with an answer, not a score with a threshold -- which is what
    the D36 attestation could not be, because MEST stores surface forms."""
    r = _retriever("flat")
    assert r.retrieve("What did Ana paint?").abstain_reason == ""
    assert r.store.knows_entity("Zoltan") is False


def test_a_plan_is_labelled_and_ranked_below_a_fact():
    store = _memory()
    store.add(_prop("Ana", "painted", "a portrait", when="2023-03",
                    modality=MOD_INTENDED, doc_id="D2:1",
                    span="I might paint a portrait next"))
    from fgl.retrieval.meca import MecaRetriever

    cfg = Config.load("M1_meca_flat")
    r = MecaRetriever(build_graph(store, _Embedder()), _Embedder(), cfg, {})
    texts = [f.text for f in r.retrieve("What did Ana paint?").facts]
    assert any("intended" in t for t in texts), "a plan is never silently a fact"
    plan = next(i for i, t in enumerate(texts) if "portrait" in t)
    fact = next(i for i, t in enumerate(texts) if "mural" in t)
    assert fact < plan, "the fact outranks the plan"


# --- the identity that makes M1 vs M2 mean anything ------------------------


def test_the_ribbon_reader_reproduces_the_flat_reader_exactly():
    """With its two degenerate settings the rotation is not consulted at all.

    Same discipline as ``reduces_to_l2()``: without this, a measured M1-M2 delta
    could be any incidental difference between two implementations rather than
    the contribution of the rotation.
    """
    flat = _retriever("flat")
    ribbon = _retriever("ribbon", ribbon_order="score", ribbon_join="index")
    for question in (
        "What did Ana paint?",
        "Who photographed what Ana painted?",
        "Where did Ana move?",
        "What did Cara cook in 2023?",
    ):
        a = flat.retrieve(question)
        b = ribbon.retrieve(question)
        assert [f.text for f in a.facts] == [f.text for f in b.facts], question
        assert [f.source for f in a.facts] == [f.source for f in b.facts], question
        assert a.tokens_used == b.tokens_used, question


def test_the_ribbon_reader_is_allowed_to_differ_only_where_it_should():
    """Turned on, it may reorder and may reach a different join -- and those
    two are exactly the places the design says the rotation could pay."""
    flat = _retriever("flat")
    ribbon = _retriever("ribbon")
    q = "Who photographed what Ana painted?"
    a, b = flat.retrieve(q), ribbon.retrieve(q)
    # the same memory is behind both, so neither may invent a turn id
    assert set(b.turn_ids) <= {"D1:3", "D7:1", "D5:2", "D3:1"}
    assert set(a.turn_ids) <= {"D1:3", "D7:1", "D5:2", "D3:1"}
    assert b.facts, "the ribbon reader must still answer"


def test_the_ribbon_timeline_matches_the_flat_timeline():
    """Orbit read and index-plus-sort must agree on the SET, whatever the order.

    A disagreement here means the rotation and the indexes have drifted apart,
    which would make every other ribbon result uninterpretable.
    """
    from fgl.retrieval.meca import FlatReader, RibbonReader

    r = _retriever("ribbon")
    flat_ids = {p.pid for p in FlatReader(r).timeline("Ana")}
    ribbon_ids = {p.pid for p in RibbonReader(r).timeline("Ana")}
    assert flat_ids == ribbon_ids


def test_budget_truncation_is_respected():
    r = _retriever("flat")
    r.cfg.retrieval.budget_tokens = 12
    result = r.retrieve("What did Ana paint?")
    assert len(result.facts) <= 2


# --------------------------------------------------------------------------- #
# Ingestion, end to end and offline                                            #
# --------------------------------------------------------------------------- #


def _conversation():
    from fgl.data.locomo import Conversation, Session, Turn

    session = Session(num=1, date_time_raw="1:56 pm on 8 May, 2023",
                      timestamp="2023-05-08T13:56:00")
    session.turns = [
        Turn("D1:1", "Jon", "How did the dance competition go last weekend?", 1),
        Turn("D1:2", "Gina", "We just did a contemporary piece called Finding Freedom.", 1),
        Turn("D1:3", "Jon", "I adopted a pup from a shelter in Stamford.", 1),
        Turn("D1:4", "Gina", "Roasted chicken is one of my favorites, I cooked it again.", 1),
    ]
    return Conversation(
        sample_id="conv-meca", speaker_a="Jon", speaker_b="Gina",
        sessions=[session], questions=[],
    )


@pytest.fixture
def meca_built(cfg, embedder, prompts):
    from fgl.llm import FakeLLM
    from fgl.memory.comprehend import MecaIngestor

    c = Config.load("M1_meca_flat")
    for attr in ("facts_cache", "graphs_dir", "results_dir", "logs_dir"):
        setattr(c.paths, attr, getattr(cfg.paths, attr))
    c.embeddings = cfg.embeddings
    c.llm = cfg.llm
    llm = FakeLLM(c.llm)
    graph, report = MecaIngestor(c, llm, embedder, prompts).ingest(_conversation())
    return graph, report, c, llm


def test_ingest_builds_a_proposition_memory_offline(meca_built):
    graph, report, _, _ = meca_built
    store = store_from_graph(graph)
    assert len(store) > 0, "the offline backend must exercise the whole path"
    assert all(p.evidence for p in store), "no proposition without provenance"
    assert report.graph_stats["n_propositions"] == len(store)
    # Every ingestor feeds the common CLI summary, including M1/M2's
    # proposition memory.
    for key in ("V", "E", "F", "C", "genus"):
        assert report.graph_stats[key] == graph.stats()[key]


def test_every_proposition_points_at_a_real_turn(meca_built):
    graph, _, _, _ = meca_built
    real = {"D1:1", "D1:2", "D1:3", "D1:4"}
    for prop in store_from_graph(graph):
        for ev in prop.evidence:
            assert ev.doc_id in real, "a fabricated turn id is the one thing forbidden"


def test_the_three_stages_are_three_separate_calls(meca_built):
    _, _, _, llm = meca_built
    purposes = {p for p, _ in llm.prompts}
    assert "ingest/meca_extract" in purposes
    assert "ingest/meca_verify" in purposes


def test_a_rejecting_verifier_empties_the_memory(cfg, embedder, prompts):
    """Verification is load-bearing, not decorative.

    This is the arm the default fake backend deliberately does NOT take: a
    default that rejected everything would make every offline run pass with an
    empty memory, which is the failure hardest to notice.
    """
    from fgl.llm import FakeLLM
    from fgl.llm.client import default_fake_responder
    from fgl.memory.comprehend import MecaIngestor

    def rejecting(prompt, system=None):
        if "# TASK: meca_verify" in prompt:
            import re as _re

            n = len(_re.findall(r"^\s*(\d+)\.\s+CLAIM:", prompt, flags=_re.MULTILINE))
            return json.dumps(
                {"verdicts": [{"id": i + 1, "supported": False} for i in range(n)]}
            )
        return default_fake_responder(prompt, system)

    c = Config.load("M1_meca_flat")
    for attr in ("facts_cache", "graphs_dir", "results_dir", "logs_dir"):
        setattr(c.paths, attr, getattr(cfg.paths, attr))
    c.embeddings, c.llm = cfg.embeddings, cfg.llm
    graph, report = MecaIngestor(
        c, FakeLLM(c.llm, responder=rejecting), embedder, prompts
    ).ingest(_conversation())

    assert len(store_from_graph(graph)) == 0
    assert report.graph_stats["comprehension"]["rejected_unverified"] > 0


def test_a_malformed_verdict_rejects_rather_than_accepts(cfg, embedder, prompts):
    """The direction of the default is the whole safety argument."""
    from fgl.llm import FakeLLM
    from fgl.llm.client import default_fake_responder
    from fgl.memory.comprehend import MecaIngestor

    def garbage(prompt, system=None):
        if "# TASK: meca_verify" in prompt:
            return json.dumps({"verdicts": []})
        return default_fake_responder(prompt, system)

    c = Config.load("M1_meca_flat")
    for attr in ("facts_cache", "graphs_dir", "results_dir", "logs_dir"):
        setattr(c.paths, attr, getattr(cfg.paths, attr))
    c.embeddings, c.llm = cfg.embeddings, cfg.llm
    graph, _ = MecaIngestor(
        c, FakeLLM(c.llm, responder=garbage), embedder, prompts
    ).ingest(_conversation())
    assert len(store_from_graph(graph)) == 0


def test_verification_off_is_the_ablation_not_the_default(meca_built, cfg, embedder,
                                                          prompts):
    from fgl.llm import FakeLLM
    from fgl.memory.comprehend import MecaIngestor, NullVerifier

    c = Config.load("M1_meca_flat", overrides=["meca.verify=false"])
    for attr in ("facts_cache", "graphs_dir", "results_dir", "logs_dir"):
        setattr(c.paths, attr, getattr(cfg.paths, attr))
    c.embeddings, c.llm = cfg.embeddings, cfg.llm
    llm = FakeLLM(c.llm)
    ing = MecaIngestor(c, llm, embedder, prompts)
    assert isinstance(ing.verifier, NullVerifier)
    ing.ingest(_conversation())
    assert not any(p.startswith("ingest/meca_verify") for p, _ in llm.prompts)


def test_segmentation_keeps_every_utterance_exactly_once():
    from fgl.memory.comprehend import Utterance, segment

    utterances = [Utterance(doc_id=f"D1:{i}", text=f"line number {i}")
                  for i in range(1, 12)]
    passages = segment(utterances, _Embedder(), 2, 4, 0.65)
    seen = [u.doc_id for p in passages for u in p.utterances]
    assert seen == [u.doc_id for u in utterances]
    assert all(len(p.utterances) <= 4 for p in passages)


def test_a_paraphrased_span_still_lands_on_a_real_utterance():
    """The extractor sometimes rewrites instead of copying. It must still be
    anchored to a real source, or dropped -- never anchored to a made-up id."""
    from fgl.memory.comprehend import Passage, Utterance

    p = Passage([
        Utterance(doc_id="D1:1", text="I adopted a pup from a shelter in Stamford."),
        Utterance(doc_id="D1:2", text="Roasted chicken is one of my favorites."),
    ])
    assert p.locate("adopted a pup from a shelter").doc_id == "D1:1"
    assert p.locate("adopted a pup shelter Stamford").doc_id == "D1:1"
    assert p.locate("something about quantum chromodynamics entirely") is None
