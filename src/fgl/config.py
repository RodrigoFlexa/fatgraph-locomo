"""Experiment configuration: YAML files -> dataclasses, with CLI overrides.

Resolution order for any single value, highest priority first:

1. ``--set dotted.key=value`` on the command line;
2. environment / ``.env`` (only for what :class:`fgl.settings.Settings` covers);
3. the condition YAML in ``configs/conditions/``;
4. ``configs/base.yaml``;
5. the dataclass defaults below.

No secret is ever read from here. Credentials live in ``.env`` / the
environment; only the Azure *deployment name* appears in YAML.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from fgl.paths import Paths, project_root


# --------------------------------------------------------------------------- #
# Sections                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class LLMConfig:
    provider: str = "azure"  # azure | fake
    deployment: str = "gpt-4o-mini"  # Azure deployment name, never a secret
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 1234
    request_timeout: float = 60.0
    max_retries: int = 6
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    cache_dir: str = ".cache/llm"
    cache_enabled: bool = True

    # --- health guards ----------------------------------------------------
    #: abort instead of silently turning empty completions into abstentions
    fail_on_empty: bool = True
    #: how many live calls before the sustained-empty check kicks in
    health_check_calls: int = 5
    #: tolerated fraction of empty completions past that point
    max_empty_rate: float = 0.2

    # --- model family ------------------------------------------------------
    #: auto | chat | reasoning. Reasoning deployments (o1/o3/o4-mini, gpt-5)
    #: take max_completion_tokens, reject a custom temperature, and burn the
    #: token budget on internal reasoning before emitting anything.
    api_style: str = "auto"
    #: floor applied to the token budget for reasoning models; 0 omits the cap
    #: entirely, which is what a working notebook against gpt-5 usually does
    reasoning_min_tokens: int = 4000
    #: some gateways reject an explicit temperature even on chat models
    send_temperature: bool = True
    #: reasoning budget hint: minimal | low | medium | high | "" to omit.
    #: LoCoMo answers are short and extractive, so "low" keeps the reasoning
    #: tokens (which are billed) from dominating the cost.
    reasoning_effort: str = "low"


@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers"  # sentence-transformers | azure | hashing
    model: str = "all-MiniLM-L6-v2"
    azure_deployment: str = "embedding-3-large-global"
    dim: int = 384  # hashing provider only
    batch_size: int = 64
    cache_dir: str = ".cache/embeddings"
    normalize: bool = True

    # --- azure provider only ----------------------------------------------
    #: Matryoshka truncation (``text-embedding-3-*``). None = the deployment's
    #: full width (3072 for -3-large, 1536 for -3-small). This is the one knob
    #: that trades index memory for quality; it is part of the cache key, so
    #: changing it invalidates rather than corrupts.
    azure_dimensions: Optional[int] = None
    #: per-input cap. The -3- family accepts 8192 tokens; a longer input is
    #: truncated rather than sent to fail.
    azure_max_input_tokens: int = 8192
    #: per-request cap, summed over the batch. Gateways enforce a total that
    #: ``batch_size`` alone does not respect.
    azure_max_batch_tokens: int = 60_000


@dataclass
class IndexConfig:
    backend: str = "numpy"  # numpy | faiss
    metric: str = "cosine"


@dataclass
class EntityConfig:
    match_threshold: float = 0.85
    llm_threshold: float = 0.70
    max_candidates: int = 5


@dataclass
class IngestConfig:
    #: triples | bipartite. "triples" is every existing condition (B3, G1-G11,
    #: T1): an LLM extracts (entity_1, relation, entity_2) and each becomes an
    #: edge between two content vertices. "bipartite" is L1: no LLM, a vertex
    #: per turn and per canonical entity, an edge per observed mention -- see
    #: fgl.memory.ingest_bipartite and docs/DECISIONS.md item L1.
    mode: str = "triples"
    sigma_policy: str = "sigma-time"  # sigma-time | sigma-agent
    #: Which extraction prompt to use.
    #:
    #: `extract_facts` (v1) names the speaker as one endpoint of most facts, and
    #: the measured consequence is that 86.6% of edges touch a speaker, the two
    #: speakers are the two highest-degree vertices in every conversation (~200
    #: against 17 for the third), and the median bridge vertex between a pair of
    #: evidence facts has degree 164. Entity sharing then means "both are about
    #: Caroline", so sigma's orbit covers the real bridge in 7.3% of cases and
    #: raising k plateaus at 11%. Every retrieval mechanism G4-G10 was tested on
    #: that substrate.
    #:
    #: `extract_facts_topical` (v2) connects the two *topics* a fact relates and
    #: records the speaker as metadata. `fact_text` is unchanged either way, so
    #: B3 embeds the same sentences and the only variable left between it and
    #: the graph conditions is the vertex assignment.
    extract_prompt: str = "extract_facts"
    detect_incongruence: bool = True
    max_facts_per_session: int = 0
    allow_self_loops: bool = False
    sigma_agent_max_trails: int = 8
    sigma_agent_trail_chars: int = 160


@dataclass
class CurationConfig:
    curation: bool = False
    consolidation: bool = False
    min_face_len: int = 4  # L
    min_stable_sessions: int = 2  # k
    #: Whitehead moves preserve the surface -- `FatGraph.whitehead_flip` asserts
    #: that genus and F are unchanged, because contracting and re-expanding an
    #: edge is a spine move. They therefore cannot reshape faces; the move that
    #: does is a transposition in sigma, below.
    whitehead_flip: bool = False  # phase 2
    #: After ingest, hill-climb on sigma to maximise the face count. By Euler
    #: (F = 2C - 2g + E - V) that is exactly minimising genus, and over fixed V
    #: and E more faces means shorter ones -- which attacks the measured failure
    #: of face retrieval, namely boundary walks of 300+ half-edges produced by
    #: ordering sigma by the clock. Needs its own graphs: a different sigma is a
    #: different ribbon graph, so such a condition cannot borrow G1's.
    maximize_faces: bool = False
    maximize_faces_passes: int = 6
    #: bound on the pair enumeration inside one vertex; the speaker hubs reach
    #: degree in the hundreds and would dominate the cost quadratically
    maximize_faces_degree_scan: int = 48


@dataclass
class RetrievalConfig:
    #: walk | bipartite. "walk" is every existing condition: anchor by
    #: cosine, walk_face from each anchor. "bipartite" is L1's degree-aware
    #: retriever (fgl.retrieval.bipartite.BipartiteRetriever) and only makes
    #: sense paired with ingest.mode=bipartite.
    mode: str = "walk"
    top_m_anchors: int = 5
    budget_tokens: int = 2000
    level2_boost: float = 0.05
    shadowed_penalty: float = 0.10
    incongruent_abstain: bool = True
    recall_ks: tuple[int, ...] = (5, 10)
    max_facts_in_prompt: int = 40
    #: ceiling on the share of ``max_facts_in_prompt`` the multi-hop mechanisms
    #: (sigma / coverage / geodesic) may occupy when the two compete. Absolute
    #: priority for the joins would let coverage evict *every* anchor fact, and
    #: G5/G6 would then stop being supersets of G1 -- the very comparison the
    #: conditions exist to make. Ignored when both flags are off.
    max_facts_join_frac: float = 0.5
    #: token budget for the answer itself. 64 is plenty for an extractive
    #: phrase, but a *reasoning* deployment spends this budget on internal
    #: reasoning first and then returns an empty string -- give it thousands.
    answer_max_tokens: int = 64

    #: Route LoCoMo category 3 (open-domain) to a prompt that permits
    #: inference. Those questions ask what is *likely*, so under the extractive
    #: instruction the model abstains on a prompt that already holds the
    #: evidence -- measured at F1 0.069 for full-context, whose evidence recall
    #: in that category was 0.97. Applies to the baselines too, so the arms
    #: stay comparable.
    open_domain_inference: bool = True

    #: Shuffle the retrieved facts before rendering the prompt.
    #: The entire distinctive content of a ribbon graph over a plain graph is
    #: *order* -- sigma is a cyclic ordering, phi is the walk it induces. So
    #: this is the load-bearing ablation for the whole thesis: if F1 does not
    #: move when the order is destroyed, then no amount of sigma optimisation
    #: can help, and the honest conclusion is that ordering carries no signal
    #: for the reader. Deterministic, seeded from `seed`.
    shuffle_context: bool = False

    #: Override the answer prompt family. Empty keeps the open-domain/set
    #: routing above; a name here uses that prompt for every question. A memory
    #: model whose context is not raw dialogue lines needs its own instructions,
    #: and the routing it replaces keys off THIS benchmark's question shapes.
    answer_prompt: str = ""

    # --- sigma expansion (multi-hop) --------------------------------------
    # A multi-hop question needs two memories that *share an entity*, i.e. two
    # half-edges in the same sigma-orbit. phi = sigma o alpha leaves the vertex
    # at every step, so the face only comes back to that entity after a full
    # lap -- usually past the token budget. Walking sigma directly is the join.
    # Off by default: G1/G2/G3 must keep producing byte-identical numbers.
    #: master switch. False => retrieve() behaves exactly as before.
    sigma_expand: bool = False
    #: how many neighbours to keep per orbit (after reranking)
    sigma_expand_k: int = 4
    #: expand from alpha(h) too, i.e. from *both* entities of the anchor memory
    sigma_expand_both_ends: bool = True
    #: only the top-N anchors get expanded (0 = all of them)
    sigma_expand_max_anchors: int = 2
    #: fraction of budget_tokens reserved for sigma neighbours
    sigma_budget_frac: float = 0.4
    #: rank the whole orbit by similarity to the question instead of taking the
    #: cyclic successors. Under sigma-time the successor is merely the
    #: chronologically adjacent fact, which is rarely the bridge.
    sigma_rerank: bool = True
    #: cap on how many half-edges of one orbit get scored (high-degree hubs).
    #: 0 = no cap, matching sigma_expand_max_anchors and max_facts_per_session.
    sigma_max_orbit_scan: int = 64

    #: Skip vertices of degree >= this when expanding sigma. 0 disables.
    #:
    #: The speaker is the `"the"` of this graph. Measured: 86% of edges touch
    #: one of the two speakers, they are the two highest-degree vertices in
    #: every conversation (~200 against 9-18 for the third), and the median
    #: vertex shared by a pair of evidence facts has degree 115. So "these two
    #: memories share an entity" degenerates into "both are about Caroline",
    #: true of half the graph, and sigma's orbit contains the real bridge in
    #: 7.7% of cases -- raising k plateaus at 11% because degrees reach 229.
    #:
    #: Information retrieval settled this in the 1960s: you do not index a
    #: stopword. A hub vertex carries no discriminative information, so the
    #: orbit through it is noise, and skipping it forces the bridge to be a
    #: real entity or to not exist. This is the same idea applied to vertices
    #: instead of terms -- and it is the test that decides whether semantic
    #: bridges exist in this data at all, because it needs no re-ingest.
    #:
    #: The threshold is not a guess: degrees are bimodal with an empty band
    #: between them. Across the ten conversations the speakers never fall below
    #: 95 and the third-ranked vertex never rises above 50, so a cut at 60
    #: removes exactly the 20 speaker vertices and nothing else -- 100%
    #: precision, measured, not tuned.
    sigma_skip_hub_degree: int = 0

    # --- face as the unit of retrieval (G10) -------------------------------
    # One line of difference from the k-NN baseline: B3 returns the top-k
    # facts, this returns the *faces containing* them, whole. What a face adds
    # is the memory that does not match the question but belongs to the same
    # narrative unit as one that does -- corroboration, which k independent
    # matches cannot produce.
    #
    # A face is a SET here, not a path. Three measurements forced that: walking
    # phi lost 0.21 of multi-hop recall; picking faces by entity coverage was
    # null (coverage saturated at 0.955); and permuting the prompt was null in
    # multi-hop, so sequence never carried the signal. Membership did.
    #
    # PRECONDITION: curation.maximize_faces. Under the clock-ordered rotation
    # 19 faces hold 75% of the memory and the median half-edge sits in a face
    # of 263 -- there is no "unit" to retrieve. After the genus search the
    # distribution is unimodal with median 36. The unit only exists there.
    #
    # Ranking is max member similarity, on purpose: a face containing a
    # relevant fact IS the coverage signal, with no linker, threshold,
    # aggregation mode or weight to tune.
    face_units: bool = False

    # --- face-first retrieval by entity coverage (multi-hop) ---------------
    # The anchor ranking asks "which fact looks like the question?", which is
    # what every RAG does and what makes single-hop easy. A multi-hop question
    # names two entities and the answer lies on a trail *between* them, so the
    # useful question is "which face covers the entities the question names?".
    # Coverage is a structural signal: a face through both vertices is a
    # candidate bridge even when none of its facts resembles the question --
    # precisely what cosine cannot express.
    # Off by default, for the same reason as sigma_expand.
    face_coverage: bool = False
    #: weight of the coverage term against the similarity term
    coverage_weight: float = 1.0
    #: how to aggregate member similarity into a face score: max | top2 | mean.
    #: mean penalises long faces, which are exactly the cross-session ones.
    coverage_sim_aggregate: str = "top2"
    #: fraction of budget_tokens reserved for the covering faces
    coverage_budget_frac: float = 0.4
    #: at most this many entities are linked from the question
    coverage_max_entities: int = 4
    #: at most this many candidate faces get scored, *in total*. The candidates
    #: are drawn round-robin across the question's entities: taking them entity
    #: by entity would let the first one exhaust the cap, and a bridge face is
    #: by definition one that shows up under more than one of them.
    coverage_max_faces: int = 24
    #: at most this many memories taken from *each* covering face. Faces are
    #: long (200+ is normal, see COERENCIA C9), so without this cap a single
    #: trail fills max_facts_in_prompt on its own and starves both the anchor
    #: walk and the sigma expansion -- measured, not hypothesised.
    coverage_max_facts_per_face: int = 20
    #: minimum cosine to accept a vertex as "named by the question" when the
    #: match is not a literal surface match
    coverage_entity_threshold: float = 0.75
    #: when no face covers 2+ of the question's entities, retrieve the shortest
    #: path between them instead: a length-2 path *is* the bridging memory pair
    coverage_geodesic_fallback: bool = True
    coverage_geodesic_max_depth: int = 3


@dataclass
class BipartiteConfig:
    """Knobs for condition L1 (``ingest.mode=retrieval.mode=bipartite``).

    Inert for every other condition, the same way ``retrieval.sigma_*`` is
    inert unless ``sigma_expand`` is on.
    """

    # --- extraction (fgl.memory.ner) ---------------------------------------
    #: spaCy pipeline. Needs a parser (for noun_chunks), not just an NER
    #: component -- "en_core_web_sm" has one; a bare NER-only model would not.
    ner_model: str = "en_core_web_sm"
    max_chunk_words: int = 6
    min_entity_chars: int = 3
    #: resolve relative dates ("last Saturday") against the session timestamp
    #: at ingest time (fgl.memory.temporal). Measured: 69.5% of LoCoMo's
    #: temporal-category evidence turns contain one.
    resolve_temporal: bool = True

    # --- retrieval (fgl.retrieval.bipartite) --------------------------------
    #: degree at or above which an entity vertex is a hub: never enumerated
    #: into the walk (it would join everything to everything), only used as a
    #: filter/intersection signal. Same "do not index a stopword" reasoning
    #: as retrieval.sigma_skip_hub_degree, but measured fresh on THIS graph's
    #: degree distribution -- a triples-graph threshold does not transfer,
    #: because the two graphs are not the same shape. Set from
    #: `fgl diagnose --bipartite` before the first real run, not guessed.
    bridge_max_degree: int = 20
    #: weight of the dense/cosine backstop against the question. Kept as a
    #: full participant, not a last resort: NER misses adjectives, feelings,
    #: and anything category 3 (open-domain) asks about, and B3/G1 already
    #: show dense retrieval alone gets single-hop to a reasonable place.
    dense_weight: float = 1.0
    #: weight of "this turn is incident to a linked non-hub entity"
    entity_weight: float = 1.0
    #: weight of "this turn is incident to a bridge entity found by
    #: intersecting two linked entities' neighbourhoods" -- the multi-hop
    #: mechanism proper, weighted above a plain entity hit because a bridge
    #: is evidence of a JOIN, which a single incident entity is not.
    bridge_weight: float = 1.5
    #: weight of "this turn is also incident to a linked HUB entity" -- a
    #: filter bonus, deliberately smaller than entity_weight: a hub match
    #: alone (e.g. both turns are just about the same speaker) is the
    #: pattern this design exists to stop treating as a bridge.
    hub_weight: float = 0.5
    #: cap on how many of a candidate turn's OTHER entities get scanned when
    #: looking for a bridge -- bounds the cost on turns incident to a hub.
    max_bridge_scan: int = 12
    #: when the question names both speakers explicitly, boost turns so the
    #: final candidate set includes at least one from each -- the
    #: "Caroline and Melanie" pattern (measured: 50/282 multi-hop questions,
    #: i.e. the minority; most multi-hop in LoCoMo is one person's own facts
    #: spread across sessions, which needs no speaker logic at all).
    speaker_filter: bool = True
    #: when the question names exactly ONE speaker, drop candidate turns
    #: spoken by the other one. This is a different mechanism from
    #: ``speaker_filter`` above, which only fires on the rare two-speaker
    #: question and only boosts; measured on L1's own predictions, 98.5-99.7%
    #: of questions name exactly one speaker, the evidence turn belongs to
    #: that speaker in 96-100% of cases (multi-hop 244/244, open-domain
    #: 72/72), and 24% of every context was being spent on the other one.
    #: The speaker is still not a vertex -- ``meta["speaker"]`` is a turn
    #: attribute, so this changes no topology and cannot recreate the hub.
    speaker_partition: bool = False
    #: never let the partition leave fewer than this many candidates: on the
    #: handful of questions where the named speaker is not the one who said
    #: it, an over-eager filter would turn a ranked miss into an empty context
    #: (and a forced abstention), which is strictly worse.
    speaker_partition_min: int = 8


@dataclass
class SlotsConfig:
    """Knobs for condition L2 (``ingest.mode=retrieval.mode=slots``).

    Inert for every other condition. The design these numbers parametrise is
    documented in :mod:`fgl.memory.slots`; what follows is only what each knob
    trades off, so a sweep can be read without opening the module.
    """

    # --- extraction --------------------------------------------------------
    #: spaCy pipeline. Needs a parser (noun chunks) and a tagger (verb lemmas).
    ner_model: str = "en_core_web_sm"
    max_chunk_words: int = 6
    min_concept_chars: int = 3
    #: resolve relative dates against the session timestamp, exactly as L1
    resolve_temporal: bool = True
    #: lift concepts to WordNet hypernyms. Turns itself off with a recorded
    #: flag in ``graph_stats["wordnet_types"]`` when the corpus is missing,
    #: rather than failing the run -- the channel is additive.
    lift_types: bool = True
    max_types_per_concept: int = 6

    # --- episodes ----------------------------------------------------------
    #: turns glued unconditionally before cohesion is even consulted. 2 = the
    #: adjacency pair, which is the entire point: a reply carries the value
    #: and none of the topic, so a question turn and its answer turn must not
    #: be able to land in different episodes.
    episode_min_turns: int = 2
    #: hard ceiling, so one long on-topic stretch cannot become a 40-turn
    #: vertex that costs the whole prompt budget in a single retrieval.
    episode_max_turns: int = 6
    #: minimum concept overlap for a turn to extend the current episode,
    #: as a fraction of the smaller of the two concept sets.
    episode_cohesion: float = 0.10

    # --- per-episode slot caps (bound the edge count, not the quality) -----
    max_concepts: int = 24
    max_predicates: int = 12
    max_types: int = 24

    # --- question linking --------------------------------------------------
    max_question_concepts: int = 6
    max_question_predicates: int = 4
    max_question_types: int = 8
    #: cosine floor for linking a question noun to a concept vertex it does
    #: not match literally. Only concepts get this fallback: every other kind
    #: is a lemma, a bucket or a name, where "nearly" is not a thing.
    concept_link_threshold: float = 0.75

    # --- scoring -----------------------------------------------------------
    #: All five structural channels are damped by 1/(1+log(degree)) before
    #: these weights apply, so a weight is "how much this KIND of evidence is
    #: worth", not "how common this slot is" -- the two were confounded in L1,
    #: where a hit on "dog" (degree 128) scored the same as one on "sunset".
    dense_weight: float = 1.0
    actor_weight: float = 1.0
    predicate_weight: float = 1.2
    concept_weight: float = 1.5
    type_weight: float = 0.6
    time_weight: float = 0.8
    #: weight for an actor merely NAMED in an episode rather than speaking in
    #: it -- enough to be findable, not enough to outrank a speaker.
    mention_weight: float = 0.25
    #: the actor is applied as a multiplicative prior, not as a summand:
    #: score *= floor + (1 - floor) * min(contribution / full, 1). ``floor`` is
    #: what an episode the named person did not contribute to keeps -- 0 would
    #: be a hard filter (and would delete the ~2-4% of questions whose evidence
    #: is the *other* speaker's turn), 1 would disable the prior entirely.
    actor_prior_floor: float = 0.35
    #: contribution share at which the prior is already fully satisfied. Half
    #: the episode's content is "this is their exchange"; demanding all of it
    #: would penalise every normal two-sided turn pair.
    actor_prior_full: float = 0.5
    #: share of a sibling turn's similarity that a turn inherits from its own
    #: exchange. This is the reply rule as arithmetic: the turn that *answers*
    #: a question rarely resembles it ("We just did a contemporary piece
    #: called 'Finding Freedom'"), so it has to be retrievable through the
    #: turn next to it. The episode-level dense term already carries most of
    #: this (see fgl.retrieval.slots), so this is the fine adjustment on top:
    #: measured optimum is shallow between 0.0 and 0.4, and it degrades above
    #: 0.7, where a whole exchange becomes as retrievable as its best line and
    #: breadth collapses (25 episodes for 56 turns instead of 35).
    sibling_frac: float = 0.2
    #: degree at or above which a slot stops being enumerated and becomes a
    #: flat filter bonus. Same rule as ``bipartite.bridge_max_degree``, but it
    #: now applies per KIND, which is what lets an actor vertex be a partition
    #: (high degree, still useful) instead of a hub (high degree, useless).
    hub_degree: int = 60
    hub_weight: float = 0.2
    #: answer a list question by enumerating an orbit instead of ranking
    #: episodes (fgl.retrieval.slots, step 2b). Category 1 is scored item by
    #: item, so an incomplete list is penalised in proportion -- and the
    #: sigma-orbit of a slot IS the list, already in chronological order.
    enumerate_sets: bool = True
    #: score added to every episode of the enumerated orbit. Deliberately above
    #: any single channel weight: on a set question the orbit is not evidence
    #: *for* an answer, it is the answer, and it has to survive truncation.
    set_orbit_boost: float = 2.0
    #: exponent on the degree damping ``1/(1+log(deg)) ** slot_damping``.
    #: 0 disables it (every orbit member scores the same, which is what L1
    #: does); 1 is full damping. Between them is the precision/enumeration
    #: trade-off: a multi-hop question is answered by a whole orbit, a
    #: single-hop one by the rarest member of it.
    slot_damping: float = 1.0

    # --- deterministic abstention -----------------------------------------
    #: abstain when the question's (actor, specific-slot) corner exists
    #: nowhere in the graph. OFF by default on purpose: it is the only
    #: mechanism here that can *remove* a correct answer, so it should be
    #: turned on from a measured false-positive rate (``fgl slots-oracle``),
    #: not from the fact that it is the elegant thing to do.
    abstain_on_empty_corner: bool = False
    #: share of an episode's content a person must contribute for the episode
    #: to count as *theirs* in the corner test. An episode is an adjacency
    #: pair, so both speakers appear in nearly all of them; at 0.0 the corner
    #: exists everywhere and the test never fires. 0.5 = the majority.
    corner_actor_min: float = 0.5

    # --- calibration: where the numbers above come from --------------------
    # Every threshold in this section began as a value swept against one
    # benchmark's annotations. That is normal practice and it is also the
    # method's main liability: a number chosen by looking at the answers
    # cannot be inherited by a second corpus, and a reviewer cannot check it.
    # The knobs below switch each one from a literal to an estimator measured
    # on the unlabelled corpus at build time (fgl.memory.calibration), so the
    # question "does this parameter need the gold labels?" has the answer
    # "no" for every one of them. Provenance is recorded per knob in the
    # ingest report -- `derived` vs `absolute` vs `fallback` -- rather than
    # asserted in a comment.
    #
    # DEFAULT IS `derived`: the honest default of the method is the estimator.
    # `configs/conditions/L2_slots.yaml` pins `absolute` explicitly so the
    # already-measured L2 numbers reproduce byte for byte, and
    # `L2d_derived.yaml` is the same condition with every estimator on -- the
    # delta between the two IS the calibration debt, measured rather than
    # argued about.
    calibration: str = "derived"

    #: hub cut-off as a quantile of the per-KIND degree distribution instead of
    #: an absolute incidence count. The absolute form is not merely inelegant,
    #: it is wrong under rescaling: on a corpus ten times longer every slot
    #: crosses a fixed 60 and the whole graph becomes a hub. Per kind because
    #: the kinds have incomparable degree scales by construction -- an actor is
    #: incident to about half the episodes and a concept to three.
    hub_degree_quantile: float = 0.99
    #: floor under the derived cut-off. Not a tuning knob: it comes from the
    #: arithmetic of the damping term (below ~e^2 incidences the damping factor
    #: is still above 1/3, so the slot is still contributing and flattening it
    #: to `hub_weight` would lose more than it saves).
    hub_degree_min: int = 8

    #: concept_link_threshold as a quantile of the observed concept-to-concept
    #: cosine distribution rather than an absolute cosine. An absolute cosine
    #: is a property of the encoder at least as much as of the task: swap the
    #: embedding model and 0.75 means something else. The quantile asks the
    #: question the literal was standing in for -- "closer than chance in this
    #: corpus under this encoder".
    concept_link_quantile: float = 0.995
    #: never link below this cosine however loose the corpus' own distribution
    concept_link_min: float = 0.55

    #: source of the question-side noun stoplist.
    #: ``literal``  the frozen LEGACY_QUESTION_NOUN_STOP -- reproduces the
    #:              measured numbers, and is the only honest setting when the
    #:              question distribution is not available in advance;
    #: ``derived``  estimated by contrasting how often a noun appears in the
    #:              question corpus against how often it appears in the memory
    #:              (fgl.memory.calibration.derive_question_stop). Uses no gold
    #:              answer, evidence or category -- but it does read the text of
    #:              the questions, which is transductive; see ASSUMPTIONS.md S5;
    #: ``none``     no question-side filtering at all, i.e. the ablation that
    #:              says how much the stoplist was worth in the first place.
    question_stop: str = "derived"
    #: minimum share of questions a noun must appear in before it can be called
    #: framing at all -- keeps a rare accident out of the list.
    question_stop_df: float = 0.01
    #: how over-represented in questions relative to the memory a noun has to
    #: be. A topic word is common in questions BECAUSE it is common in the
    #: conversations, so its ratio sits near 1; a template word is common in
    #: questions and absent from what anyone said.
    question_stop_ratio: float = 3.0

    #: granularities the time channel indexes, comma-separated, coarse to fine.
    #: ``month`` alone is the original single-resolution index, chosen because
    #: it is the grain LoCoMo questions happen to use -- a true observation
    #: about one question generator and a bad reason for a parameter. Indexing
    #: every level instead REMOVES the parameter: a question emits every level
    #: it names and the degree damping already in the scorer decides which one
    #: carries the match (a year vertex sits on most of the corpus and is
    #: damped to nothing; a day vertex sits on a handful and scores full).
    time_granularities: str = "year,month,day"


@dataclass
class PropagationConfig:
    """Knobs for condition L3 (``retrieval.mode=propagation``).

    Inert for every other condition. The operator is documented in
    :mod:`fgl.retrieval.propagation`; what follows is only what each knob
    trades off.

    **The reduction is the design constraint.** ``hops=1`` with
    ``normalization="none"`` and ``dense_seed=0`` reproduces condition L2's
    structural read exactly -- not approximately, not "in spirit". That is
    asserted in ``tests/test_propagation.py`` and reported by
    :func:`fgl.retrieval.propagation.reduces_to_l2`, and it is what makes the
    sweep over ``hops`` a curve whose leftmost point is the published L2
    number instead of a comparison between two unrelated systems.
    """

    #: length of the walk, counted in ARRIVALS AT EPISODES. 1 = L2's single
    #: hop. 2 = the join: episodes sharing a slot with an episode the question
    #: named, which is what a multi-hop question asks for and the only place
    #: the extra structure can pay. 3 rarely adds reach that damping has not
    #: already erased, but it is in the sweep grid so that claim is measured.
    hops: int = 2
    #: mass kept per further hop (the complement of PageRank's restart
    #: probability). Below 1 it guarantees a direct mention always outranks a
    #: shared neighbour at equal support, which is the right prior.
    decay: float = 0.5
    #: ``none`` | ``rw`` | ``sym`` -- see fgl.retrieval.propagation.
    #: ``sym`` is the spectral normalisation and the principled default: it
    #: damps the EPISODE side too, which ``1/(1+log deg)`` never did.
    #: ``none`` exists so the reduction to L2 is exact.
    normalization: str = "sym"
    #: subtract each incidence's own incoming flow before relaying, i.e. run
    #: the Hashimoto non-backtracking operator. Without it, hop 2 is mostly the
    #: seed reflected off its own episodes -- it looks like a join and is not
    #: one. Costs one extra bincount per hop.
    non_backtracking: bool = True
    #: how much of the walk's personalisation comes from the dense channel, so
    #: a turn that resembles the question can lend that resemblance to the
    #: episode next to it in the graph. 0 keeps the dense channel purely
    #: additive at emission, exactly as L2 has it.
    dense_seed: float = 0.0
    #: let hub slots relay mass. OFF, and it is the same rule the scorer and
    #: the Steiner metric obey: **a hub is a filter, never a bridge.** Turning
    #: it on is the ablation that shows why -- mass enters the actor vertex and
    #: leaves smeared over half the corpus.
    bridge_hubs: bool = False


@dataclass
class SteinerConfig:
    """Knobs for the connection read of condition L4 (``retrieval.mode=unified``).

    The read itself is documented in :mod:`fgl.retrieval.steiner`. Note what is
    NOT here: an abstention threshold. It is calibrated per conversation against
    the cost of random slot tuples of the same size, so the only thing to
    configure is which tail counts as far.
    """

    enabled: bool = True
    #: weight of the join channel, relative to the typed channels. The score is
    #: scale-free by construction (best root gets ``weight``, twice as far gets
    #: half), so there is no sharpness exponent to sweep alongside it.
    weight: float = 1.5
    #: how many of the question's slots the connection must hold together,
    #: rarest first. More terminals means a stricter conjunction and one more
    #: shortest-path computation each; beyond four the rarest slots are already
    #: doing the discriminating.
    max_terminals: int = 4
    #: path cost beyond which two things are not connected in any useful sense.
    #: In units of ``1 + log(degree)`` per slot hop plus 1 per episode, so ~12
    #: is three or four hops through moderately common slots.
    max_cost: float = 12.0
    #: per-source distance cache. Questions in one conversation reuse each
    #: other's sources heavily, and the null calibration reuses them again.
    cache_size: int = 4096

    #: use the connection cost as the abstention signal, superseding the binary
    #: corner test where it has more resolution. Unlike the corner test this
    #: ships ON: the corner test was measured as a losing trade (20/446 caught
    #: for 38/1540 false positives), and the last run's adversarial regression
    #: (0.666 -> 0.608, as retrieval improved) is what it was supposed to
    #: prevent.
    abstain: bool = True
    #: tail of the NULL distribution above which a question counts as
    #: unsupported: "these slots sit further apart than 95% of arbitrary
    #: combinations of the same size in this memory". Derived per conversation,
    #: never fitted to an answer key.
    abstain_quantile: float = 0.95
    #: random tuples drawn per terminal count when building that null
    null_samples: int = 120
    #: size of the slot pool the tuples are drawn from. Small on purpose: every
    #: tuple reuses the same few dozen sources, so the calibration costs about
    #: ``null_pool`` shortest-path computations rather than
    #: ``null_samples * k``.
    null_pool: int = 64


@dataclass
class MecaConfig:
    """MECA: read once, deeply; answer many times, cheaply.

    The cost structure of ordinary RAG is inverted -- ingestion is cheap and
    dumb, answering is expensive and pays that price once per question, redoing
    the same comprehension every time. MECA moves resolution (coreference,
    temporal grounding, modality, implicature, entity identity) to ingest,
    where it is paid once, and stores the *result*: attested propositions
    rather than pointers into text.

    Every knob here is either a structural bound declared as a premise or a
    corpus-derived quantile. None of them needs the gold labels, which is the
    criterion ``docs/ASSUMPTIONS.md`` holds every threshold in this repo to.
    """

    # --- segmentation (structural bounds, not swept knobs) -----------------
    passage_min: int = 2
    passage_max: int = 6
    #: cut at this quantile of THIS source's own coherence-drop distribution
    passage_quantile: float = 0.65

    # --- comprehension -----------------------------------------------------
    #: second pass: what follows from the passage that was not said outright
    infer: bool = True
    #: entailment check against the cited span. Off is the ablation that
    #: measures what verification buys -- it is not a performance option.
    verify: bool = True
    extract_max_tokens: int = 2000
    infer_max_tokens: int = 1200
    verify_max_tokens: int = 1200

    # --- consolidation (each stage is one ablation) ------------------------
    resolve_entities: bool = True
    deduplicate: bool = True
    timeline: bool = True
    entity_quantile: float = 0.995
    entity_floor: float = 0.80
    predicate_quantile: float = 0.99
    predicate_floor: float = 0.82
    #: a predicate is single-valued (and so can be superseded) above this
    #: quantile of the corpus's own functionality distribution
    functional_quantile: float = 0.75
    #: floor on functionality. Low on purpose: the estimator is a mean of
    #: 1/|distinct objects|, and one subject who moved house costs a predicate
    #: 0.125 rather than a whole vote, so a high floor would refuse to
    #: supersede exactly where supersession is called for.
    functional_floor: float = 0.60

    # --- reading -----------------------------------------------------------
    #: ``flat`` = inverted indexes; ``ribbon`` = sigma orbits, corners and face
    #: walks over the SAME store. One memory, two readers: any measured delta
    #: is the reader and cannot be anything else.
    reader: str = "flat"
    #: second step of the query plan, over propositions rather than over a
    #: similarity graph. 0 disables the join entirely.
    join_steps: int = 1
    #: how many propositions the second step may bring back
    join_budget: int = 8
    #: emit statements whose modality is not factual (plans, wishes). They are
    #: labelled either way; this decides whether they compete for budget.
    emit_non_factual: bool = True
    #: emit propositions the timeline marked superseded, labelled as such. A
    #: question that names a past time needs them.
    emit_superseded: bool = True
    #: weight of the dense channel when structure alone under-fills the budget
    dense_weight: float = 1.0

    # --- ribbon reader only ------------------------------------------------
    #: ``orbit`` = emit in sigma order (chronological, local); ``score`` = emit
    #: by score, which reproduces the flat reader exactly. The identity is
    #: pinned by a test, the same discipline that proved L5 reduces to L2.
    ribbon_order: str = "orbit"
    #: ``face`` = find the join by a phi-walk; ``index`` = by re-query, which is
    #: what the flat reader does.
    ribbon_join: str = "face"
    #: bound on the face walk, in half-edges
    ribbon_walk_max: int = 64


@dataclass
class SupportConfig:
    """The abstention decision, moved out of the prompt and into the graph.

    Why this exists, in one measurement: in every results file this project has
    produced, ``adversarial/f1`` equals ``adversarial/abstention_rate`` exactly.
    Adversarial is not a category of question -- it is the direct measurement of
    the abstention policy, over 446 of 1986 questions. Between two runs of the
    SAME condition, substantive F1 rose (0.5263 -> 0.5347) while micro fell
    0.069, and every point of the loss was that rate collapsing from 0.5762 to
    0.2420. In the same unit, solving multi-hop completely is worth +0.081
    micro; solving the decision to answer is worth +0.170.

    Off by default, like every other mechanism here that can DELETE a correct
    answer: L1 through L6 must stay byte-identical when this section is
    untouched. See :mod:`fgl.retrieval.support` and
    ``docs/PROPOSTA_ATESTADO.md``.
    """

    enabled: bool = False
    #: act on an ``absent`` verdict. False scores and reports the attestation
    #: without letting it remove anything -- which is how you measure the
    #: operating curve before paying for it.
    abstain: bool = True

    # --- where the cut comes from -----------------------------------------
    #: ``otsu`` (default) is the label-free bimodal cut and has no free
    #: parameter at all; ``quantile`` asserts what fraction of the question set
    #: is unanswerable, which is a fact about the benchmark and therefore the
    #: honest fallback rather than the default; ``absolute`` pins ``floor`` to
    #: reproduce an old number.
    method: str = "otsu"
    quantile: float = 0.2
    floor: float = 0.0
    #: histogram resolution for Otsu. Not a tuning knob: the cut moves by less
    #: than one bin width, which is what a sweep over it should show.
    bins: int = 64

    #: candidates read for the concentration and margin features, and for the
    #: shape classification
    top_k: int = 8


@dataclass
class BridgeConfig:
    """Condition L6: LLM-synthesised connections between episodes that share
    no slot at all.

    Everything up to L5 can only score an edge the deterministic ingest
    already put in the graph -- Steiner, the walk, sigma all operate on
    incidences ``extract`` and ``ingest_slots`` already wrote. Two episodes
    whose connection is thematic or causal but names no entity in common
    (different surface forms, or genuinely no shared noun) are invisible to
    every one of those channels by construction: none of them reads two
    episodes' text at the same time. This is the one place in the L line an
    LLM looks at raw content during ingestion rather than during the answer
    call, and it is off by default -- L1 through L5 must stay byte-identical
    when this section is untouched.

    See ``docs/L6_DESIGN_bridge_synthesis.md`` for the two-stage design and
    why the prompt takes no speaker/session parameters (an ingest mechanism
    that only works on two-party dated dialogue is not a general mechanism).
    """

    enabled: bool = False

    # --- stage 1: candidate pairs, zero LLM -------------------------------
    #: nearest neighbours considered per episode, by embedding cosine
    top_k: int = 6
    #: quantile of the observed episode-episode cosine distribution that
    #: counts as "close enough to investigate" -- same estimator as
    #: `slots.concept_link_quantile`, just over episode vectors instead of
    #: concept vectors (see `fgl.memory.calibration.concept_link_threshold_by_quantile`,
    #: which this reuses directly). Derived PER CONVERSATION: a corpus of
    #: another size or domain gets its own threshold, never this number.
    quantile: float = 0.98
    #: floor under the derived quantile, and the value used outright when the
    #: conversation has too few episodes for a quantile to mean anything.
    floor: float = 0.35
    #: a candidate pair already reachable within this many hops through the
    #: EXISTING slot vocabulary (see `fgl.evaluation.hops.episode_hops`) is
    #: skipped -- sigma/Steiner/the walk already have a shot at it for zero
    #: LLM cost, so spending a call there would be redundant with achado 2,
    #: not a test of achado 3.
    skip_within_hops: int = 1
    #: hard ceiling on candidate pairs sent to stage 2, independent of how
    #: loose the derived quantile turns out to be on an unfamiliar corpus.
    #: Exists so a bad derivation costs a warning, never an unbounded bill.
    max_candidates: int = 400

    # --- stage 2: judgment + synthesis, one LLM call per surviving pair ----
    #: minimum characters of `bridge_text` accepted -- guards against a
    #: parseable but empty/near-empty response being treated as a real link.
    min_bridge_chars: int = 8


@dataclass
class BaselineConfig:
    rag_top_k: int = 10
    full_context_max_tokens: int = 110_000
    full_context_truncate: str = "head"


@dataclass
class PathsConfig:
    locomo_repo: str = "data/external/locomo"
    data_file: str = "data/external/locomo/data/locomo10.json"
    prompts_dir: str = "prompts"
    results_dir: str = "results"
    graphs_dir: str = "artifacts/graphs"
    facts_cache: str = "artifacts/facts"
    logs_dir: str = "artifacts/logs"
    #: read the memory graphs from *another* condition's directory instead of
    #: ``graphs_dir/<condition>``. Empty = use this condition's own name.
    #: Lets a retrieval-only ablation (G4) run on G1's byte-identical graphs,
    #: so the delta isolates retrieval and costs no LLM calls.
    graphs_condition: str = ""


SECTIONS: dict[str, type] = {
    "llm": LLMConfig,
    "embeddings": EmbeddingConfig,
    "index": IndexConfig,
    "entities": EntityConfig,
    "ingest": IngestConfig,
    "curation": CurationConfig,
    "retrieval": RetrievalConfig,
    "bipartite": BipartiteConfig,
    "slots": SlotsConfig,
    "propagation": PropagationConfig,
    "steiner": SteinerConfig,
    "bridges": BridgeConfig,
    "meca": MecaConfig,
    "support": SupportConfig,
    "baselines": BaselineConfig,
    "paths": PathsConfig,
}


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    condition: str = "G1-fatgraph-min"
    seed: int = 1234
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    entities: EntityConfig = field(default_factory=EntityConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    bipartite: BipartiteConfig = field(default_factory=BipartiteConfig)
    slots: SlotsConfig = field(default_factory=SlotsConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    steiner: SteinerConfig = field(default_factory=SteinerConfig)
    bridges: BridgeConfig = field(default_factory=BridgeConfig)
    meca: MecaConfig = field(default_factory=MecaConfig)
    support: SupportConfig = field(default_factory=SupportConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    #: filled in by :meth:`load`, for provenance in the results manifest
    source: Optional[str] = None

    # ------------------------------------------------------------- loading --
    @classmethod
    def load(
        cls,
        condition: str | Path,
        overrides: Iterable[str] | None = None,
        root: Path | None = None,
        settings=None,
    ) -> "Config":
        """Load by condition name (``G1``), file stem or explicit path.

        ``overrides`` are ``dotted.key=value`` strings from ``--set``.
        ``settings`` is an optional :class:`fgl.settings.Settings` whose model
        choices are overlaid *before* the CLI overrides, so ``--set`` always wins.
        """
        path = resolve_condition(condition, root)
        cfg = cls.from_yaml(path)
        cfg.source = str(path)
        if settings is not None:
            settings.apply_to(cfg)
        if overrides:
            cfg.apply_overrides(overrides)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "Config":
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        base_name = raw.pop("extends", None)
        if base_name:
            base_path = (p.parent / base_name).resolve()
            if not base_path.exists():  # allow `extends: base.yaml` from conditions/
                base_path = (p.parent.parent / base_name).resolve()
            merged = deep_merge(cls.from_yaml(base_path).to_dict(), raw)
            merged.pop("source", None)
        else:
            merged = raw
        if overrides:
            merged = deep_merge(merged, overrides)
        return cls.from_dict(merged)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = copy.deepcopy(d)
        d.pop("source", None)
        kwargs: dict[str, Any] = {}
        for key, klass in SECTIONS.items():
            payload = d.pop(key, {}) or {}
            unknown = set(payload) - set(klass.__dataclass_fields__)
            if unknown:
                raise ConfigError(
                    f"unknown keys in config.{key}: {sorted(unknown)} "
                    f"(valid: {sorted(klass.__dataclass_fields__)})"
                )
            if key == "retrieval" and "recall_ks" in payload:
                payload["recall_ks"] = tuple(payload["recall_ks"])
            kwargs[key] = klass(**payload)
        unknown_top = set(d) - {"condition", "seed"}
        if unknown_top:
            raise ConfigError(f"unknown top-level config keys: {sorted(unknown_top)}")
        kwargs.update(d)
        return cls(**kwargs)

    # ----------------------------------------------------------- overrides --
    def apply_overrides(self, overrides: Iterable[str]) -> "Config":
        """Apply ``--set dotted.key=value`` strings, typed from the dataclass."""
        for item in overrides:
            if "=" not in item:
                raise ConfigError(
                    f"malformed override {item!r}; expected section.key=value"
                )
            dotted, _, raw = item.partition("=")
            self.set(dotted.strip(), raw.strip())
        return self

    def set(self, dotted: str, raw: str) -> None:
        parts = dotted.split(".")
        target: Any = self
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ConfigError(f"unknown config section {part!r} in {dotted!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not is_dataclass(target) or leaf not in target.__dataclass_fields__:
            raise ConfigError(
                f"unknown config key {dotted!r}. Try: fgl config keys"
            )
        current = getattr(target, leaf)
        setattr(target, leaf, coerce(raw, current))

    def get(self, dotted: str) -> Any:
        target: Any = self
        for part in dotted.split("."):
            target = getattr(target, part)
        return target

    # ---------------------------------------------------------- validation --
    def validate(self) -> "Config":
        if self.llm.provider not in ("azure", "fake"):
            raise ConfigError(f"llm.provider must be azure|fake, got {self.llm.provider!r}")
        if self.embeddings.provider not in ("sentence-transformers", "azure", "hashing"):
            raise ConfigError(f"unknown embeddings.provider {self.embeddings.provider!r}")
        if self.llm.api_style not in ("auto", "chat", "reasoning"):
            raise ConfigError(
                f"llm.api_style must be auto|chat|reasoning, got {self.llm.api_style!r}"
            )
        if self.index.backend not in ("numpy", "faiss"):
            raise ConfigError(f"index.backend must be numpy|faiss")
        if self.ingest.mode not in ("triples", "bipartite", "slots", "meca"):
            raise ConfigError(
                "ingest.mode must be triples|bipartite|slots|meca, got "
                f"{self.ingest.mode!r}"
            )
        if self.retrieval.mode not in (
            "walk", "bipartite", "slots", "propagation", "unified", "meca"
        ):
            raise ConfigError(
                "retrieval.mode must be walk|bipartite|slots|propagation|"
                f"unified|meca, got {self.retrieval.mode!r}"
            )
        # Each non-default memory model builds its own kind of vertex, and a
        # retriever that does not know that kind cannot interpret the graph at
        # all (it would silently score turn vertices as if they were entities).
        # So the pair is checked as a pair, not as two independent settings.
        # One ingest can serve several reads: L2, L3 and L4 are three
        # questions asked of the SAME typed episode-slot graph, which is why
        # they can borrow each other's graphs byte for byte and why a delta
        # between them isolates the read. What must never pair is an ingest
        # with a retriever that cannot interpret its vertex kinds.
        _MODEL_PAIRS = {
            "bipartite": ("bipartite",),
            "slots": ("slots", "propagation", "unified"),
            "triples": ("walk",),
            # MECA is one memory read two ways: `meca.reader` picks flat or
            # ribbon, and both interpret the same proposition vertices. The
            # reader is NOT a retrieval.mode, precisely so the two arms cannot
            # accidentally end up reading two different memories.
            "meca": ("meca",),
        }
        allowed = _MODEL_PAIRS[self.ingest.mode]
        if self.retrieval.mode not in allowed:
            raise ConfigError(
                f"ingest.mode={self.ingest.mode!r} builds vertices that "
                f"retrieval.mode={self.retrieval.mode!r} cannot interpret: "
                f"expected retrieval.mode in {list(allowed)!r} -- the two must "
                "be switched together"
            )
        if self.ingest.mode == "bipartite":
            if self.bipartite.bridge_max_degree < 2:
                raise ConfigError("bipartite.bridge_max_degree must be >= 2")
            if self.bipartite.max_chunk_words < 1:
                raise ConfigError("bipartite.max_chunk_words must be >= 1")
            if self.bipartite.min_entity_chars < 1:
                raise ConfigError("bipartite.min_entity_chars must be >= 1")
            if self.bipartite.max_bridge_scan < 1:
                raise ConfigError("bipartite.max_bridge_scan must be >= 1")
            if self.bipartite.speaker_partition_min < 0:
                raise ConfigError("bipartite.speaker_partition_min must be >= 0")
        if self.ingest.mode == "slots":
            sl = self.slots
            if sl.episode_min_turns < 1:
                raise ConfigError("slots.episode_min_turns must be >= 1")
            if sl.episode_max_turns < sl.episode_min_turns:
                raise ConfigError(
                    "slots.episode_max_turns must be >= slots.episode_min_turns"
                )
            if not 0.0 <= sl.episode_cohesion <= 1.0:
                raise ConfigError("slots.episode_cohesion must be in [0, 1]")
            if sl.hub_degree < 2:
                raise ConfigError("slots.hub_degree must be >= 2")
            if not 0.0 <= sl.concept_link_threshold <= 1.0:
                raise ConfigError("slots.concept_link_threshold must be in [0, 1]")
            if not 0.0 <= sl.corner_actor_min <= 1.0:
                raise ConfigError("slots.corner_actor_min must be in [0, 1]")
            if sl.set_orbit_boost < 0.0:
                raise ConfigError("slots.set_orbit_boost must be >= 0")
            if sl.slot_damping < 0.0:
                raise ConfigError("slots.slot_damping must be >= 0")
            if not 0.0 <= sl.actor_prior_floor <= 1.0:
                raise ConfigError("slots.actor_prior_floor must be in [0, 1]")
            if not 0.0 < sl.actor_prior_full <= 1.0:
                raise ConfigError("slots.actor_prior_full must be in (0, 1]")
            if not 0.0 <= sl.sibling_frac <= 2.0:
                raise ConfigError("slots.sibling_frac must be in [0, 2]")
            for name in ("max_concepts", "max_predicates", "max_types",
                         "max_question_concepts", "max_question_predicates",
                         "max_question_types", "max_types_per_concept"):
                if getattr(sl, name) < 1:
                    raise ConfigError(f"slots.{name} must be >= 1")
            if sl.calibration not in ("absolute", "derived"):
                raise ConfigError(
                    "slots.calibration must be absolute|derived, got "
                    f"{sl.calibration!r}"
                )
            if sl.question_stop not in ("literal", "derived", "none"):
                raise ConfigError(
                    "slots.question_stop must be literal|derived|none, got "
                    f"{sl.question_stop!r}"
                )
            if not 0.5 <= sl.hub_degree_quantile < 1.0:
                raise ConfigError("slots.hub_degree_quantile must be in [0.5, 1)")
            if sl.hub_degree_min < 2:
                raise ConfigError("slots.hub_degree_min must be >= 2")
            if not 0.5 <= sl.concept_link_quantile < 1.0:
                raise ConfigError("slots.concept_link_quantile must be in [0.5, 1)")
            if not 0.0 <= sl.concept_link_min <= 1.0:
                raise ConfigError("slots.concept_link_min must be in [0, 1]")
            if not 0.0 <= sl.question_stop_df <= 1.0:
                raise ConfigError("slots.question_stop_df must be in [0, 1]")
            if sl.question_stop_ratio < 1.0:
                raise ConfigError(
                    "slots.question_stop_ratio must be >= 1 (below 1 a word "
                    "commoner in the memory than in the questions would be "
                    "called framing, which inverts the estimator)"
                )
            # imported here, not at module scope: fgl.memory.slots reaches
            # fgl.memory.entities, which imports this module back.
            from fgl.memory.slots import parse_granularities

            try:
                grans = parse_granularities(sl.time_granularities)
            except ValueError as exc:
                raise ConfigError(f"slots.time_granularities: {exc}") from exc
            if not grans:
                raise ConfigError(
                    "slots.time_granularities must name at least one of "
                    "year,month,day -- an empty time channel is `time_weight=0`, "
                    "not an empty granularity list"
                )
        if self.retrieval.mode in ("propagation", "unified"):
            pg = self.propagation
            if pg.hops < 1:
                raise ConfigError("propagation.hops must be >= 1")
            if not 0.0 < pg.decay <= 1.0:
                raise ConfigError("propagation.decay must be in (0, 1]")
            if pg.normalization not in ("none", "rw", "sym"):
                raise ConfigError(
                    "propagation.normalization must be none|rw|sym, got "
                    f"{pg.normalization!r}"
                )
            if pg.dense_seed < 0.0:
                raise ConfigError("propagation.dense_seed must be >= 0")
        if self.retrieval.mode == "unified":
            st = self.steiner
            if st.max_terminals < 2:
                raise ConfigError(
                    "steiner.max_terminals must be >= 2 -- a connection needs "
                    "two things to connect"
                )
            if st.max_cost <= 0.0:
                raise ConfigError("steiner.max_cost must be > 0")
            if st.weight < 0.0:
                raise ConfigError("steiner.weight must be >= 0")
            if not 0.5 <= st.abstain_quantile < 1.0:
                raise ConfigError(
                    "steiner.abstain_quantile must be in [0.5, 1) -- it is the "
                    "upper tail of a null distribution, not a score"
                )
            if st.null_samples < 10:
                raise ConfigError("steiner.null_samples must be >= 10")
            if st.null_pool <= st.max_terminals:
                raise ConfigError(
                    "steiner.null_pool must exceed steiner.max_terminals: a "
                    "tuple cannot be drawn without replacement otherwise"
                )
        if self.bridges.enabled:
            if self.ingest.mode != "slots":
                raise ConfigError(
                    "bridges.enabled requires ingest.mode=slots -- a bridge is "
                    "a synthetic episode incident to existing slot vertices, "
                    f"which only the typed-slot graph has, got ingest.mode="
                    f"{self.ingest.mode!r}"
                )
            br = self.bridges
            if br.top_k < 1:
                raise ConfigError("bridges.top_k must be >= 1")
            if not 0.5 <= br.quantile < 1.0:
                raise ConfigError("bridges.quantile must be in [0.5, 1)")
            if not 0.0 <= br.floor <= 1.0:
                raise ConfigError("bridges.floor must be in [0, 1]")
            if br.skip_within_hops < 1:
                raise ConfigError("bridges.skip_within_hops must be >= 1")
            if br.max_candidates < 1:
                raise ConfigError("bridges.max_candidates must be >= 1")
            if br.min_bridge_chars < 1:
                raise ConfigError("bridges.min_bridge_chars must be >= 1")

        if self.ingest.mode == "meca" or self.retrieval.mode == "meca":
            # The ingest/retrieval pairing itself is already enforced by
            # _MODEL_PAIRS above -- MECA is a memory model, not a read, and
            # reading a proposition store with a slot retriever would compare
            # two different memories while claiming to compare two reads.
            mc = self.meca
            if mc.reader not in ("flat", "ribbon"):
                raise ConfigError(
                    f"meca.reader must be flat | ribbon (got {mc.reader!r})"
                )
            if mc.ribbon_order not in ("orbit", "score"):
                raise ConfigError("meca.ribbon_order must be orbit | score")
            if mc.ribbon_join not in ("face", "index"):
                raise ConfigError("meca.ribbon_join must be face | index")
            if mc.passage_min < 1 or mc.passage_max < mc.passage_min:
                raise ConfigError(
                    "meca.passage_min >= 1 and meca.passage_max >= passage_min"
                )
            if not 0.0 <= mc.passage_quantile < 1.0:
                raise ConfigError("meca.passage_quantile must be in [0, 1)")
            for name in ("entity_quantile", "predicate_quantile", "functional_quantile"):
                if not 0.0 <= getattr(mc, name) < 1.0:
                    raise ConfigError(f"meca.{name} must be in [0, 1)")
            if mc.join_steps < 0:
                raise ConfigError("meca.join_steps must be >= 0")

        if self.support.enabled:
            if self.ingest.mode != "slots":
                raise ConfigError(
                    "support.enabled requires ingest.mode=slots -- the attestation "
                    "is a statement about the typed-slot projection of a question, "
                    "and there are no slots to project onto otherwise."
                )
            sp = self.support
            if sp.method not in ("otsu", "quantile", "absolute"):
                raise ConfigError(
                    "support.method must be otsu | quantile | absolute "
                    f"(got {sp.method!r})"
                )
            if not 0.0 <= sp.quantile < 1.0:
                raise ConfigError("support.quantile must be in [0, 1)")
            if not 0.0 <= sp.floor <= 1.0:
                raise ConfigError("support.floor must be in [0, 1]")
            if sp.bins < 8:
                raise ConfigError("support.bins must be >= 8")
            if sp.top_k < 2:
                raise ConfigError(
                    "support.top_k must be >= 2 -- concentration and margin are "
                    "undefined on a single candidate"
                )
        if self.ingest.extract_prompt not in ("extract_facts", "extract_facts_topical"):
            raise ConfigError(
                "ingest.extract_prompt must be extract_facts|extract_facts_topical, "
                f"got {self.ingest.extract_prompt!r}"
            )
        if self.ingest.sigma_policy not in ("sigma-time", "sigma-agent"):
            raise ConfigError(
                f"ingest.sigma_policy must be sigma-time|sigma-agent, "
                f"got {self.ingest.sigma_policy!r}"
            )
        if not 0 <= self.entities.llm_threshold <= self.entities.match_threshold <= 1:
            raise ConfigError(
                "require 0 <= entities.llm_threshold <= entities.match_threshold <= 1"
            )
        if self.retrieval.top_m_anchors < 1:
            raise ConfigError("retrieval.top_m_anchors must be >= 1")
        if self.retrieval.budget_tokens < 1:
            raise ConfigError("retrieval.budget_tokens must be >= 1")
        if self.curation.min_face_len < 2:
            raise ConfigError("curation.min_face_len must be >= 2")
        if self.curation.maximize_faces:
            if self.curation.maximize_faces_passes < 1:
                raise ConfigError("curation.maximize_faces_passes must be >= 1")
            if self.curation.maximize_faces_degree_scan < 0:
                raise ConfigError(
                    "curation.maximize_faces_degree_scan must be >= 0 (0 = no bound)"
                )
            if self.paths.graphs_condition:
                raise ConfigError(
                    "curation.maximize_faces rewrites sigma, so the graph is a "
                    "different ribbon graph and cannot be borrowed: leave "
                    f"paths.graphs_condition empty (got "
                    f"{self.paths.graphs_condition!r})"
                )
        if self.retrieval.sigma_expand:
            if self.retrieval.sigma_expand_k < 1:
                raise ConfigError("retrieval.sigma_expand_k must be >= 1")
            if not 0.0 <= self.retrieval.sigma_budget_frac < 1.0:
                raise ConfigError(
                    "retrieval.sigma_budget_frac must be in [0, 1) -- the face "
                    "walk needs what is left of the budget"
                )
            if self.retrieval.sigma_expand_max_anchors < 0:
                raise ConfigError(
                    "retrieval.sigma_expand_max_anchors must be >= 0 (0 = all anchors)"
                )
            if self.retrieval.sigma_max_orbit_scan < 0:
                raise ConfigError(
                    "retrieval.sigma_max_orbit_scan must be >= 0 (0 = no cap)"
                )
            if self.retrieval.sigma_skip_hub_degree < 0:
                raise ConfigError(
                    "retrieval.sigma_skip_hub_degree must be >= 0 (0 = disabled)"
                )
            if 0 < self.retrieval.sigma_skip_hub_degree <= 2:
                raise ConfigError(
                    "retrieval.sigma_skip_hub_degree <= 2 would skip almost every "
                    "vertex and disable the expansion instead of focusing it"
                )
        if self.retrieval.face_units:
            for flag in ("sigma_expand", "face_coverage"):
                if getattr(self.retrieval, flag):
                    raise ConfigError(
                        f"retrieval.face_units retrieves whole faces; "
                        f"retrieval.{flag} is a different retrieval policy and "
                        "combining them would measure neither"
                    )
            if not self.curation.maximize_faces:
                raise ConfigError(
                    "retrieval.face_units requires curation.maximize_faces: "
                    "under the clock-ordered rotation a handful of faces hold "
                    "most of the memory (median half-edge in a face of 263), "
                    "so there is no unit to retrieve"
                )
        if self.retrieval.face_coverage:
            if self.retrieval.coverage_sim_aggregate not in ("max", "top2", "mean"):
                raise ConfigError(
                    "retrieval.coverage_sim_aggregate must be max|top2|mean, got "
                    f"{self.retrieval.coverage_sim_aggregate!r}"
                )
            if not 0.0 <= self.retrieval.coverage_budget_frac < 1.0:
                raise ConfigError(
                    "retrieval.coverage_budget_frac must be in [0, 1)"
                )
            if self.retrieval.coverage_max_entities < 1:
                raise ConfigError("retrieval.coverage_max_entities must be >= 1")
            if self.retrieval.coverage_max_faces < 1:
                raise ConfigError("retrieval.coverage_max_faces must be >= 1")
            if self.retrieval.coverage_geodesic_max_depth < 1:
                raise ConfigError("retrieval.coverage_geodesic_max_depth must be >= 1")
        if self.retrieval.sigma_expand or self.retrieval.face_coverage:
            if not 0.0 < self.retrieval.max_facts_join_frac <= 1.0:
                raise ConfigError(
                    "retrieval.max_facts_join_frac must be in (0, 1]"
                )
            # The anchor index L2-normalises internally, so anchors are cosine
            # either way; the coverage and sigma scorers dot the raw vectors.
            # With normalisation off, `sim` leaves [-1, 1] and the
            # `sim + coverage_weight * coverage` sum stops meaning anything.
            if not self.embeddings.normalize:
                raise ConfigError(
                    "retrieval.sigma_expand / face_coverage require "
                    "embeddings.normalize=true: they score half-edges by raw "
                    "dot product, which is only cosine on unit vectors"
                )
        reserved = 0.0
        if self.retrieval.sigma_expand:
            reserved += self.retrieval.sigma_budget_frac
        if self.retrieval.face_coverage:
            reserved += self.retrieval.coverage_budget_frac
        if reserved >= 1.0:
            raise ConfigError(
                f"sigma_budget_frac + coverage_budget_frac = {reserved:.2f} >= 1: "
                "nothing would be left for the anchor face walk"
            )
        return self

    def requires_azure(self) -> bool:
        return self.llm.provider == "azure" or self.embeddings.provider == "azure"

    # -------------------------------------------------------------- output --
    def to_dict(self) -> dict:
        d = asdict(self)
        d["retrieval"]["recall_ks"] = list(self.retrieval.recall_ks)
        return d

    def to_yaml(self) -> str:
        d = self.to_dict()
        d.pop("source", None)
        return yaml.safe_dump(d, sort_keys=False, allow_unicode=True)

    def flat(self) -> dict[str, Any]:
        """``{"llm.deployment": "gpt-4o-mini", ...}`` — what ``--set`` accepts."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value):
                for sub in fields(value):
                    out[f"{f.name}.{sub.name}"] = getattr(value, sub.name)
            elif f.name != "source":
                out[f.name] = value
        return out

    def diff(self, other: "Config") -> dict[str, tuple[Any, Any]]:
        a, b = self.flat(), other.flat()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    def resolved_paths(self, root: Path | None = None) -> dict[str, Path]:
        p = Paths.build(root or project_root())
        return {f.name: p.resolve(getattr(self.paths, f.name)) for f in fields(self.paths)}


# --------------------------------------------------------------------------- #
# Condition discovery                                                          #
# --------------------------------------------------------------------------- #


def conditions_dir(root: Path | None = None) -> Path:
    return Paths.build(root or project_root()).conditions


def list_conditions(root: Path | None = None) -> list[tuple[str, str, Path]]:
    """``[(name, condition_id, path), ...]`` for every YAML in conditions/."""
    out = []
    for p in sorted(conditions_dir(root).glob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cond = raw.get("condition", p.stem)
        except Exception:
            cond = p.stem
        out.append((p.stem, cond, p))
    return out


def resolve_condition(condition: str | Path, root: Path | None = None) -> Path:
    """Accept a path, a file stem (``G1_fatgraph_min``), an id (``G1-fatgraph-min``)
    or a short prefix (``G1``)."""
    p = Path(condition)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()

    key = str(condition).strip().lower().replace("-", "_")
    entries = list_conditions(root)
    if not entries:
        raise ConfigError(f"no conditions found in {conditions_dir(root)}")

    exact = [e for e in entries if e[0].lower() == key or e[1].lower().replace("-", "_") == key]
    if len(exact) == 1:
        return exact[0][2]

    # The leading id segment, matched whole: `G1` names G1_fatgraph_min and not
    # G10_face_units. Without this a plain prefix search makes every short id
    # ambiguous the moment a tenth condition exists -- which is a property of
    # the numbering, not of the user's request.
    ident = [e for e in entries if e[0].split("_")[0].lower() == key]
    if len(ident) == 1:
        return ident[0][2]

    prefix = [e for e in entries if e[0].lower().startswith(key) or
              e[1].lower().replace("-", "_").startswith(key)]
    if len(prefix) == 1:
        return prefix[0][2]
    if len(prefix) > 1:
        raise ConfigError(
            f"{condition!r} is ambiguous: {sorted(e[1] for e in prefix)}"
        )
    raise ConfigError(
        f"unknown condition {condition!r}. Available: "
        + ", ".join(sorted(e[1] for e in entries))
    )


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def coerce(raw: str, current: Any) -> Any:
    """Cast a CLI string to the type of the value it replaces.

    Every failure surfaces as :class:`ConfigError` so the CLI can report it as a
    configuration problem (exit 2) instead of crashing with a traceback.
    """
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ConfigError(f"expected a boolean (true/false), got {raw!r}")
    if isinstance(current, tuple):
        try:
            return tuple(int(x) for x in raw.strip("[]() ").split(",") if x.strip())
        except ValueError as exc:
            raise ConfigError(
                f"expected a comma-separated list of integers, got {raw!r}"
            ) from exc
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"expected an integer, got {raw!r}") from exc
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"expected a number, got {raw!r}") from exc
    if current is None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw
