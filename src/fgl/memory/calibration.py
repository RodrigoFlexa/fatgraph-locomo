"""Corpus-derived replacements for the literals L2 used to hard-code.

Why this module exists
----------------------
Every threshold in :class:`fgl.config.SlotsConfig` started life as a number
swept against one benchmark. That is ordinary practice, but it leaves the
method with a property no reviewer can check and no second corpus can inherit:
the numbers were chosen *by looking at the answers*. The test that separates
"a method" from "a method fitted to this dataset" is narrow and mechanical:

    does the parameter need the gold labels to be set?

If yes, it is calibration debt. If it can be estimated from the unlabelled
corpus at build time, it is just an algorithm with an estimator in it.

This module implements the estimator side. Nothing here reads a gold answer,
an evidence annotation or a question category -- the inputs are the graph the
memory just built and (for the framing stoplist) the *text* of the questions
that will be asked. Each estimated quantity carries its provenance and the
measurement behind it in :class:`Calibration`, so a results directory records
"hub_degree=73, derived, 99th percentile of 412 concept degrees" instead of
"hub_degree=60" with a comment pointing at a sweep nobody can rerun.

What is derived, and from what
------------------------------
``hub_degree``
    Was an absolute count (60). An absolute count is not just inelegant, it is
    wrong under rescaling: on a corpus ten times longer every slot crosses it
    and the whole graph becomes a hub. Derived as a **quantile of the degree
    distribution of that kind**, so "hub" means "in the top 1% of concepts by
    degree in *this* memory" and the number moves with the corpus. Per kind,
    because the kinds have incomparable degree scales by construction: an
    actor is incident to roughly half the episodes and a concept to three.

``concept_link_threshold``
    Was an absolute cosine (0.75), which is a property of the embedding model
    at least as much as of the task -- swap the encoder and the number is
    meaningless. Derived as a **high quantile of the observed concept-to-
    concept cosine distribution**, i.e. "closer than 99.5% of unrelated pairs
    in this corpus under this encoder", which is the thing 0.75 was standing
    in for.

``actor_prior_floor`` / ``actor_prior_full``
    Were 0.35 / 0.5, fitted to a corpus of two-speaker dialogues. Derived from
    the two quantities that actually govern them: the number of distinct
    speakers (the prior should be *stronger* when naming one person excludes
    more of the corpus) and the **median share of the dominant contributor in
    an episode** (the point at which "this exchange is theirs" is already
    true). On a two-speaker corpus the derived pair lands near the swept one;
    on a six-party meeting corpus it moves on its own instead of having to be
    re-swept.

``QUESTION_NOUN_STOP``
    Was a hand-written list, and the honest reading of it is that it fits the
    *question generator* of one benchmark rather than any property of English:
    "conversation", "date" and "type" are template words. Derived by
    contrasting two distributions the system already has -- how often a noun
    appears in the question corpus against how often it appears in the memory.
    A topic word is common in both; a framing word is common in questions and
    absent from what people actually said. That ratio is the question-side
    analogue of IDF, and :func:`derive_question_stop` recovers the hand list
    on LoCoMo (see ``tests/test_calibration.py``), which is the evidence that
    the mechanism subsumes the hack rather than merely replacing it.

One honesty note, stated here rather than buried
------------------------------------------------
The framing stoplist is fitted on the *text* of the questions that will be
asked. That is transductive. It uses no labels -- not the answer, not the
evidence, not the category -- so it is not leakage in the sense that matters
for a recall or F1 number, but it is a dependence on seeing the query
distribution, and a deployment that answers one question at a time cannot do
it. ``slots.question_stop="literal"`` and ``"none"`` are the two honest
fallbacks for that setting, and :mod:`fgl.evaluation.scope` reports which was
used. See ``docs/ASSUMPTIONS.md``, scope condition S5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from fgl.memory.entities import normalize_name
from fgl.memory.slots import (
    KIND_ACTOR,
    KIND_CONCEPT,
    KIND_EPISODE,
    KIND_PREDICATE,
    KIND_TIME,
    KIND_TYPE,
    LEGACY_QUESTION_NOUN_STOP,
)

#: Kinds that get their own degree distribution. ``episode`` is excluded: it is
#: the other side of the bipartition, and its degree measures how much was said
#: in an exchange rather than how discriminating a slot is.
CALIBRATED_KINDS: tuple[str, ...] = (
    KIND_ACTOR,
    KIND_PREDICATE,
    KIND_CONCEPT,
    KIND_TYPE,
    KIND_TIME,
)

#: Below this many vertices of a kind, a quantile is an artefact of the sample
#: rather than a measurement, so the absolute fallback is used and said so.
MIN_VERTICES_FOR_QUANTILE = 12

#: Below this many questions, the framing-word contrast is noise.
MIN_QUESTIONS_FOR_STOP = 50

ABSOLUTE = "absolute"
DERIVED = "derived"
FALLBACK = "fallback"  # derivation was asked for but the corpus was too small


# --------------------------------------------------------------------------- #
# The result                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class Calibration:
    """Every number the retriever needs that used to be a literal.

    ``source`` maps each knob to ``absolute`` / ``derived`` / ``fallback`` and
    ``evidence`` carries the measurement behind each derived value. Both are
    written into the ingest report, which is what makes the claim "this number
    was not chosen by looking at the answers" auditable after the run instead
    of asserted in a comment.
    """

    hub_degree_by_kind: dict[str, int] = field(default_factory=dict)
    hub_degree_default: int = 60
    concept_link_threshold: float = 0.75
    actor_prior_floor: float = 0.35
    actor_prior_full: float = 0.5
    question_noun_stop: frozenset[str] = frozenset()
    source: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, object] = field(default_factory=dict)

    def hub_degree(self, kind: str) -> int:
        """Hub cut-off for one slot kind, in incidences."""
        return self.hub_degree_by_kind.get(kind, self.hub_degree_default)

    def as_dict(self) -> dict:
        return {
            "hub_degree_by_kind": dict(sorted(self.hub_degree_by_kind.items())),
            "hub_degree_default": self.hub_degree_default,
            "concept_link_threshold": round(self.concept_link_threshold, 4),
            "actor_prior_floor": round(self.actor_prior_floor, 4),
            "actor_prior_full": round(self.actor_prior_full, 4),
            "n_question_noun_stop": len(self.question_noun_stop),
            "question_noun_stop": sorted(self.question_noun_stop),
            "source": dict(sorted(self.source.items())),
            "evidence": self.evidence,
        }

    def summary(self) -> str:
        """One line per knob: value, provenance, and what it was measured on."""
        lines = []
        for knob in sorted(self.source):
            src = self.source[knob]
            ev = self.evidence.get(knob, "")
            if knob == "hub_degree":
                val = ", ".join(
                    f"{k}={v}" for k, v in sorted(self.hub_degree_by_kind.items())
                ) or str(self.hub_degree_default)
            elif knob == "question_noun_stop":
                val = f"{len(self.question_noun_stop)} words"
            else:
                val = f"{getattr(self, knob, ''):.4g}"
            lines.append(f"  {knob:<24} {val:<40} [{src}] {ev}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Estimators                                                                   #
# --------------------------------------------------------------------------- #


def degrees_by_kind(graph) -> dict[str, list[int]]:
    """Degree of every slot vertex, bucketed by kind.

    Concept vertices created by :class:`fgl.memory.entities.EntityResolver`
    carry ``kind`` only because the slot ingest stamps it, and a vertex with no
    ``kind`` at all is a resolver-made concept, so it is counted as one -- the
    same defaulting :func:`fgl.memory.ingest_slots._kind_histogram` uses.
    """
    out: dict[str, list[int]] = {k: [] for k in CALIBRATED_KINDS}
    for vid, vx in graph.vertices.items():
        kind = vx.meta.get("kind", KIND_CONCEPT)
        if kind == KIND_EPISODE or kind not in out:
            continue
        out[kind].append(graph.degree(vid))
    return out


def hub_degree_by_quantile(
    graph,
    quantile: float,
    minimum: int,
    absolute_fallback: int,
) -> tuple[dict[str, int], dict[str, object]]:
    """Per-kind hub cut-off as a quantile of that kind's degree distribution.

    Returns ``(threshold_by_kind, evidence)``. The threshold is *inclusive*,
    matching the retriever's ``degree >= hub_degree`` test, so a quantile of
    0.99 makes roughly the top 1% of that kind's vertices hubs regardless of
    how large the corpus is -- which is the property the absolute 60 lacked.

    ``minimum`` is a floor, not a tuning knob: on a graph where every concept
    has degree 2 the 99th percentile is 3, and calling a degree-3 concept a
    hub would delete the only signal the graph has. It is set from the
    arithmetic of the damping term rather than from recall -- below roughly
    ``e**2`` incidences the damping factor is still above 1/3, so the slot is
    contributing, and only above that is treating it as a flat filter bonus a
    smaller loss than enumerating it.
    """
    thresholds: dict[str, int] = {}
    evidence: dict[str, object] = {"quantile": quantile, "per_kind": {}}
    for kind, degs in degrees_by_kind(graph).items():
        if len(degs) < MIN_VERTICES_FOR_QUANTILE:
            thresholds[kind] = absolute_fallback
            evidence["per_kind"][kind] = {
                "n_vertices": len(degs),
                "threshold": absolute_fallback,
                "source": FALLBACK,
            }
            continue
        arr = np.asarray(degs, dtype=float)
        cut = int(max(minimum, math.ceil(float(np.quantile(arr, quantile)))))
        thresholds[kind] = cut
        evidence["per_kind"][kind] = {
            "n_vertices": len(degs),
            "median_degree": float(np.median(arr)),
            "max_degree": int(arr.max()),
            "threshold": cut,
            "frac_above": round(float((arr >= cut).mean()), 4),
            "source": DERIVED,
        }
    return thresholds, evidence


def concept_link_threshold_by_quantile(
    matrix: Optional[np.ndarray],
    quantile: float,
    floor: float,
    ceiling: float = 0.99,
    max_rows: int = 600,
    rng_seed: int = 1234,
) -> tuple[float, dict[str, object]]:
    """Cosine floor for the paraphrase fallback, from this corpus' own geometry.

    ``matrix`` is the unit-normalised concept embedding matrix the retriever
    already builds. The threshold answers "how close is closer than chance
    *here*", which is what an absolute 0.75 was standing in for and which
    changes the moment the encoder changes.

    Sampled rather than exhaustive above ``max_rows``: the quantile of a
    600x600 pair sample is stable to well under the precision anyone reads
    this number at, and the full matrix is quadratic in a quantity that grows
    with the corpus.
    """
    if matrix is None or len(matrix) < MIN_VERTICES_FOR_QUANTILE:
        return floor if matrix is None else floor, {
            "n_concepts": 0 if matrix is None else int(len(matrix)),
            "source": FALLBACK,
        }
    rows = matrix
    if len(rows) > max_rows:
        rng = np.random.default_rng(rng_seed)
        rows = rows[rng.choice(len(rows), size=max_rows, replace=False)]
    sims = rows @ rows.T
    iu = np.triu_indices(len(rows), k=1)
    pairs = sims[iu]
    if pairs.size == 0:
        return floor, {"n_concepts": int(len(rows)), "source": FALLBACK}
    cut = float(np.quantile(pairs, quantile))
    cut = float(min(max(cut, floor), ceiling))
    return cut, {
        "n_concepts": int(len(matrix)),
        "n_pairs_sampled": int(pairs.size),
        "quantile": quantile,
        "median_pair_cosine": round(float(np.median(pairs)), 4),
        "threshold": round(cut, 4),
        "clipped_to_floor": bool(cut <= floor + 1e-9),
        "source": DERIVED,
    }


def actor_prior_from_graph(
    graph,
    floor_min: float = 0.05,
    floor_max: float = 0.90,
) -> tuple[float, float, dict[str, object]]:
    """``(floor, full)`` for the multiplicative actor prior, measured here.

    Two quantities, each with a reason rather than a sweep behind it:

    ``floor = 1 / n_speakers``
        what an episode keeps when the named person contributed nothing to it.
        Naming one of ``S`` speakers leaves ``1/S`` of the corpus as the prior
        expectation that the evidence is theirs anyway, so the residual an
        un-owned episode keeps is exactly that. It also has the behaviour the
        absolute 0.35 lacked: with two speakers the prior is mild (0.5), and
        with eight it sharpens to 0.125 on its own, because naming one of
        eight excludes far more than naming one of two.

    ``full = median dominant-contributor share``
        the point at which "this exchange is theirs" is already true. An
        episode is an adjacency pair, so the dominant speaker's share has a
        corpus-specific typical value -- near 0.5-0.6 on two-party dialogue,
        much lower on a six-party meeting. Demanding a fixed 0.5 in the second
        case would leave the prior permanently unsatisfied and demote every
        episode equally, which is the failure mode this replaces.
    """
    speakers: set[str] = set()
    dominant: list[float] = []
    for vx in graph.vertices.values():
        if vx.meta.get("kind") != KIND_EPISODE:
            continue
        content: Mapping[str, int] = vx.meta.get("speaker_content", {}) or {}
        total = sum(content.values())
        if not total:
            continue
        speakers.update(k for k, n in content.items() if n > 0)
        dominant.append(max(content.values()) / total)

    n_speakers = max(len(speakers), 1)
    floor = float(min(max(1.0 / n_speakers, floor_min), floor_max))
    full = float(np.median(dominant)) if dominant else 0.5
    # A `full` of 1.0 would mean the prior is only ever satisfied by a
    # monologue; clamp into the open interval the config validator requires.
    full = float(min(max(full, 0.05), 1.0))
    return floor, full, {
        "n_speakers": n_speakers,
        "n_episodes_with_content": len(dominant),
        "median_dominant_share": round(full, 4),
        "floor": round(floor, 4),
        "source": DERIVED if dominant else FALLBACK,
    }


def question_noun_frequencies(
    questions: Sequence[str], extractor
) -> dict[str, float]:
    """Document frequency of each noun key over the question corpus, in [0, 1].

    Parsed with the *same* extractor the memory was built with, for the same
    reason retrieval parses questions with it: two sides lemmatised differently
    never meet.
    """
    if not questions:
        return {}
    counts: dict[str, int] = {}
    for ex in extractor.extract_many(list(questions)):
        seen: set[str] = set()
        for cand in ex.candidates:
            key = normalize_name(cand.text)
            if key and key not in seen:
                seen.add(key)
                counts[key] = counts.get(key, 0) + 1
    n = float(len(questions))
    return {k: v / n for k, v in counts.items()}


def memory_noun_frequencies(graph) -> dict[str, float]:
    """Share of episodes each concept is incident to, in [0, 1].

    Keyed by every surface the concept vertex answers to (its key, its name and
    its aliases), so a question noun is looked up the same way the retriever
    looks it up.
    """
    n_ep = sum(
        1 for vx in graph.vertices.values() if vx.meta.get("kind") == KIND_EPISODE
    )
    if not n_ep:
        return {}
    out: dict[str, float] = {}
    for vid, vx in graph.vertices.items():
        if vx.meta.get("kind", KIND_CONCEPT) != KIND_CONCEPT:
            continue
        share = graph.degree(vid) / float(n_ep)
        for surface in (vx.meta.get("key", ""), vx.name, *vx.aliases):
            key = normalize_name(surface)
            if key:
                out[key] = max(out.get(key, 0.0), share)
    return out


def derive_question_stop(
    question_df: Mapping[str, float],
    memory_df: Mapping[str, float],
    min_df: float,
    min_ratio: float,
    memory_floor: float = 0.005,
) -> tuple[frozenset[str], dict[str, object]]:
    """Nouns that frame questions rather than name their content.

    A noun is framing when it is *both* common in the question corpus and
    over-represented there relative to the memory::

        df_q(w) >= min_df   and   df_q(w) / max(df_mem(w), memory_floor) >= min_ratio

    The ratio is what carries the argument. A topic word is common in the
    questions because it is common in the conversations -- "dog" is asked about
    because someone talked about a dog, so both frequencies move together and
    the ratio stays near 1. A template word is common in the questions and
    nearly absent from what anyone said, because it comes from the generator
    and not from the corpus: that is a large ratio, and it is a property of
    *any* templated question set, not of this one.

    ``memory_floor`` keeps the ratio finite for a word the memory never
    mentions at all, and sets the scale at which "absent from the memory"
    stops being distinguishable from "rare in the memory".
    """
    stop: set[str] = set()
    ranked: list[tuple[float, str, float, float]] = []
    for word, dfq in question_df.items():
        dfm = memory_df.get(word, 0.0)
        ratio = dfq / max(dfm, memory_floor)
        ranked.append((ratio, word, dfq, dfm))
        if dfq >= min_df and ratio >= min_ratio:
            stop.add(word)
    ranked.sort(reverse=True)
    return frozenset(stop), {
        "n_questions_vocab": len(question_df),
        "min_df": min_df,
        "min_ratio": min_ratio,
        "n_selected": len(stop),
        # the ten strongest framing candidates, so the derivation can be
        # eyeballed against the hand-written list it replaces
        "top": [
            {"word": w, "df_question": round(dq, 4), "df_memory": round(dm, 4),
             "ratio": round(r, 2)}
            for r, w, dq, dm in ranked[:10]
        ],
        "source": DERIVED,
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def calibrate(
    cfg,
    graph,
    concept_matrix: Optional[np.ndarray] = None,
    question_corpus: Optional[Sequence[str]] = None,
    extractor=None,
) -> Calibration:
    """Resolve every calibrated knob for one memory, honouring ``slots.calibration``.

    ``slots.calibration="absolute"`` reproduces the swept L2 numbers verbatim
    and records that it did; ``"derived"`` estimates each one from ``graph``
    (and, for the framing stoplist, ``question_corpus``) and falls back to the
    absolute value -- recorded as ``fallback``, never silently -- wherever the
    corpus is too small for the estimate to mean anything.
    """
    sl = cfg.slots
    derived = sl.calibration == DERIVED
    cal = Calibration(
        hub_degree_default=sl.hub_degree,
        concept_link_threshold=sl.concept_link_threshold,
        actor_prior_floor=sl.actor_prior_floor,
        actor_prior_full=sl.actor_prior_full,
    )

    # --- hub degree -------------------------------------------------------
    if derived:
        thresholds, ev = hub_degree_by_quantile(
            graph, sl.hub_degree_quantile, sl.hub_degree_min, sl.hub_degree
        )
        cal.hub_degree_by_kind = thresholds
        cal.source["hub_degree"] = DERIVED
        cal.evidence["hub_degree"] = ev
    else:
        cal.hub_degree_by_kind = {k: sl.hub_degree for k in CALIBRATED_KINDS}
        cal.source["hub_degree"] = ABSOLUTE
        cal.evidence["hub_degree"] = {"value": sl.hub_degree}

    # --- concept link threshold ------------------------------------------
    if derived:
        cut, ev = concept_link_threshold_by_quantile(
            concept_matrix, sl.concept_link_quantile, sl.concept_link_min
        )
        cal.concept_link_threshold = cut
        cal.source["concept_link_threshold"] = ev.get("source", DERIVED)
        cal.evidence["concept_link_threshold"] = ev
    else:
        cal.source["concept_link_threshold"] = ABSOLUTE
        cal.evidence["concept_link_threshold"] = {"value": sl.concept_link_threshold}

    # --- actor prior ------------------------------------------------------
    if derived:
        floor, full, ev = actor_prior_from_graph(graph)
        cal.actor_prior_floor = floor
        cal.actor_prior_full = full
        cal.source["actor_prior_floor"] = ev["source"]
        cal.source["actor_prior_full"] = ev["source"]
        cal.evidence["actor_prior_floor"] = ev
        cal.evidence["actor_prior_full"] = ev
    else:
        cal.source["actor_prior_floor"] = ABSOLUTE
        cal.source["actor_prior_full"] = ABSOLUTE
        cal.evidence["actor_prior_floor"] = {"value": sl.actor_prior_floor}
        cal.evidence["actor_prior_full"] = {"value": sl.actor_prior_full}

    # --- question framing stoplist ---------------------------------------
    mode = sl.question_stop
    if mode == "none":
        cal.question_noun_stop = frozenset()
        cal.source["question_noun_stop"] = ABSOLUTE
        cal.evidence["question_noun_stop"] = {"mode": "none"}
    elif mode == "derived" and question_corpus and extractor is not None and (
        len(question_corpus) >= MIN_QUESTIONS_FOR_STOP
    ):
        qdf = question_noun_frequencies(question_corpus, extractor)
        mdf = memory_noun_frequencies(graph)
        stop, ev = derive_question_stop(
            qdf, mdf, sl.question_stop_df, sl.question_stop_ratio
        )
        ev["n_questions"] = len(question_corpus)
        cal.question_noun_stop = stop
        cal.source["question_noun_stop"] = DERIVED
        cal.evidence["question_noun_stop"] = ev
    else:
        cal.question_noun_stop = LEGACY_QUESTION_NOUN_STOP
        cal.source["question_noun_stop"] = (
            ABSOLUTE if mode == "literal" else FALLBACK
        )
        cal.evidence["question_noun_stop"] = {
            "mode": mode,
            "n_questions": len(question_corpus or ()),
            "why": (
                "literal list requested"
                if mode == "literal"
                else "derivation needs a question corpus of at least "
                f"{MIN_QUESTIONS_FOR_STOP} and an extractor"
            ),
        }

    return cal


def merge_question_corpora(convs: Iterable) -> list[str]:
    """Every question text across conversations, as the framing estimator wants it.

    Text only: no answer, no evidence, no category. Kept as a named helper so
    that constraint is enforced in one place rather than at each call site.
    """
    out: list[str] = []
    for conv in convs:
        for q in getattr(conv, "questions", ()):
            out.append(q.prompt_question())
    return out
