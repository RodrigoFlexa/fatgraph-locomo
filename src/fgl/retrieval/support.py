"""Support attestation: what SHAPE of support a question has in this memory.

The measurement this exists for
-------------------------------
In every results file the project has produced, ``adversarial/f1`` is *equal*
to ``adversarial/abstention_rate``, digit for digit (L2d: 0.5762 == 0.5762).
Adversarial is not a category of question; it is the direct measurement of the
abstention policy, and it is 446 of 1986 questions -- more than multi-hop (282)
and open-domain (96) together. Between the run of 2026-08-20 and the one after
it, L2d's substantive F1 *rose* (0.5263 -> 0.5347) while micro fell 0.069:
every point of the loss was the adversarial abstention rate collapsing from
0.5762 to 0.2420. In the same unit, solving multi-hop completely is worth
+0.081 micro; solving the decision to answer is worth +0.170.

Today that decision is delegated in full to the generator, by one line buried
in a six-rule prompt, reading two thousand tokens of retrieved memory. This
module makes it a property of the graph instead, computed before any LLM call.

The four shapes
---------------
A question arrives already parsed into a slot tuple with a hole -- (who,
did-what, with-what, when). How that tuple projects into the graph is a
structural fact with four cases:

``direct``    one episode covers the filled slots and the hole;
``composed``  no single episode does, but two do jointly through a connector
              -- the multi-hop case, which should be presented AS a join;
``conflict``  two episodes both cover it, in different sessions -- the update
              case, which should be presented in time order;
``absent``    the filled slots never co-occur above the derived floor -- the
              abstention case, which should cost almost no context at all.

Where the number comes from
---------------------------
Same criterion as every other threshold here (``docs/ASSUMPTIONS.md``): *does
the parameter need the gold labels to be set?* No. The cut is Otsu's method on
the corpus's own support-score histogram -- the classical label-free bimodal
threshold, which maximises between-class variance and has no free parameter at
all. It never sees a category, an answer, or an annotation. Deriving the cut
from the known adversarial rate would be exactly the reverse-engineering the
calibration work (D30) exists to avoid, and is not done here.

The score has no weights either, and its two halves are combined differently
on purpose:

* the **gate** (vocabulary present, corner owned) is *conjunctive* -- these are
  refutations, and one refutation is enough. Geometric mean.
* the **evidence** (co-occurrence, concentration, margin, dense peak) is
  *averaged* with equal weight -- no single one of these is decisive and
  weighting them would be a swept parameter wearing a disguise.

Equal weights and the choice of combinator are declared model assumptions, the
same status as ``w_kappa``, ``lambda`` and ``gamma`` in the method section --
asserted, ablatable, and not dressed up as measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

SHAPE_DIRECT = "direct"
SHAPE_COMPOSED = "composed"
SHAPE_CONFLICT = "conflict"
SHAPE_ABSENT = "absent"

SHAPES: tuple[str, ...] = (SHAPE_DIRECT, SHAPE_COMPOSED, SHAPE_CONFLICT, SHAPE_ABSENT)

#: reasons an attestation can be ``absent``. The first two are structural
#: refutations inherited from the corner test; the third is the score falling
#: below the derived cut.
REASON_MISSING_SLOT = "missing_slot"
REASON_EMPTY_CORNER = "empty_corner"
REASON_LOW_SUPPORT = "low_support"


# --------------------------------------------------------------------------- #
# Inputs and output                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class SupportInputs:
    """Everything the attestation needs, as plain data.

    Deliberately not a handle on the retriever: the scoring rule is the claim
    this module makes, and a claim you can only exercise by building a graph is
    a claim nobody will check.
    """

    #: specific slots (concept/predicate/type) the question named
    asked_specific: int = 0
    #: how many of those resolved to a vertex in this memory
    linked_specific: int = 0
    #: the corner test's value, and its reason when it refutes. 1.0/"" when the
    #: question names no actor or no specific slot (nothing to refute).
    corner: float = 1.0
    corner_reason: str = ""
    #: resolved specific slot vertex -> the episodes in its sigma-orbit
    slot_orbits: Mapping[str, frozenset[str]] = field(default_factory=dict)
    #: candidate episode scores, any order
    candidate_scores: Sequence[float] = ()
    #: for the highest-scoring episodes: which of the question's specific slot
    #: vertices each one carries, and which session it belongs to
    episode_slots: Mapping[str, frozenset[str]] = field(default_factory=dict)
    episode_sessions: Mapping[str, str] = field(default_factory=dict)
    #: best cosine any episode reached on the dense channel
    dense_top: Optional[float] = None
    #: set questions are answered by a whole orbit, so a flat score
    #: distribution is correct for them and concentration must not be read as
    #: weak support
    is_set: bool = False


@dataclass
class Attestation:
    """The verdict, its witness, and every number that produced it."""

    shape: str = SHAPE_DIRECT
    score: float = 1.0
    reason: str = ""
    threshold: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    #: episode vertex ids that justify the shape -- one for ``direct``, the two
    #: sides for ``composed`` and ``conflict``, empty for ``absent``
    witness: list[str] = field(default_factory=list)

    @property
    def abstains(self) -> bool:
        return self.shape == SHAPE_ABSENT

    def as_dict(self) -> dict:
        return {
            "shape": self.shape,
            "score": round(self.score, 4),
            "reason": self.reason,
            "threshold": round(self.threshold, 4),
            "features": {k: round(v, 4) for k, v in sorted(self.features.items())},
            "witness": list(self.witness),
        }


# --------------------------------------------------------------------------- #
# Features                                                                     #
# --------------------------------------------------------------------------- #


def vocabulary_presence(asked: int, linked: int) -> Optional[float]:
    """Fraction of the question's content words this memory has ever heard.

    The strongest negative signal available and, until now, thrown away: the
    retriever computed the unlinked slots and used them only inside the corner
    test's fallback. A question naming three things none of which exist in the
    memory is not a hard question, it is an unanswerable one.
    """
    if asked <= 0:
        return None  # nothing was asked of the vocabulary; not a vote
    return linked / asked


def cooccurrence(slot_orbits: Mapping[str, frozenset[str]]) -> Optional[float]:
    """Do the question's content slots ever land in the same episode?

    ``|A ∩ B| / min(|A|, |B|)`` over the best pair, so a rare slot fully
    contained in a common one still scores 1.0 -- the asymmetry is the point:
    "the painting Melanie mentioned" is rare and sits inside "Melanie", not the
    other way round.
    """
    vids = [v for v, eps in slot_orbits.items() if eps]
    if len(vids) < 2:
        return None  # undefined with one slot, not zero
    best = 0.0
    for i, a in enumerate(vids):
        for b in vids[i + 1 :]:
            ea, eb = slot_orbits[a], slot_orbits[b]
            denom = min(len(ea), len(eb))
            if denom:
                best = max(best, len(ea & eb) / denom)
    return best


def concentration(scores: Sequence[float], top_k: int) -> Optional[float]:
    """1 − normalised entropy of the top scores.

    Real support is peaked; noise is flat. An adversarial question still
    retrieves *something* -- every question does, the budget is always spent --
    but what it retrieves is a plateau of weakly similar episodes rather than a
    spike, and that difference is visible without knowing the answer.
    """
    vals = sorted((max(0.0, float(s)) for s in scores), reverse=True)[:top_k]
    if len(vals) < 2:
        return None
    total = sum(vals)
    if total <= 0:
        return 0.0
    ps = [v / total for v in vals if v > 0]
    if len(ps) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in ps)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(vals))))


def margin(scores: Sequence[float]) -> Optional[float]:
    """Relative gap between the best and second-best episode."""
    vals = sorted((max(0.0, float(s)) for s in scores), reverse=True)[:2]
    if len(vals) < 2:
        return None
    if vals[0] <= 0:
        return 0.0
    return max(0.0, min(1.0, (vals[0] - vals[1]) / vals[0]))


# --------------------------------------------------------------------------- #
# The attestation                                                              #
# --------------------------------------------------------------------------- #


def support_score(inputs: SupportInputs, top_k: int = 8) -> tuple[float, dict[str, float]]:
    """The continuous score, and every feature that went into it.

    Replaces the binary corner test, which measured 20/446 true positives
    against 38/1540 false positives -- near break-even, which is why it ships
    disabled. A single predicate cannot separate two distributions; the point
    of a score is that the operating point becomes a choice you can see.
    """
    feats: dict[str, float] = {}

    voc = vocabulary_presence(inputs.asked_specific, inputs.linked_specific)
    if voc is not None:
        feats["vocabulary"] = voc
    feats["corner"] = max(0.0, min(1.0, inputs.corner))

    # --- the gate: conjunctive, because these are refutations ---------------
    gate_terms = [feats["corner"]] + ([voc] if voc is not None else [])
    gate = math.prod(gate_terms) ** (1.0 / len(gate_terms))

    # --- the evidence: averaged, because none of these is decisive ----------
    soft: list[float] = []
    co = cooccurrence(inputs.slot_orbits)
    if co is not None:
        feats["cooccurrence"] = co
        soft.append(co)
    # A set question is answered by a whole orbit, so a flat distribution is
    # the correct shape for it -- reading that as weak support would abstain on
    # exactly the questions the orbit enumeration exists to answer.
    if not inputs.is_set:
        con = concentration(inputs.candidate_scores, top_k)
        if con is not None:
            feats["concentration"] = con
            soft.append(con)
        mar = margin(inputs.candidate_scores)
        if mar is not None:
            feats["margin"] = mar
            soft.append(mar)
    if inputs.dense_top is not None:
        d = max(0.0, min(1.0, float(inputs.dense_top)))
        feats["dense_top"] = d
        soft.append(d)

    strength = sum(soft) / len(soft) if soft else 1.0
    feats["gate"] = gate
    feats["strength"] = strength
    return gate * strength, feats


def classify_shape(
    inputs: SupportInputs, top_k: int = 8
) -> tuple[str, list[str]]:
    """``direct`` / ``composed`` / ``conflict``, with the episodes that witness it.

    Called only for questions that survived the abstention cut, so ``absent``
    is not produced here.
    """
    need = {v for v, eps in inputs.slot_orbits.items() if eps}
    ranked = sorted(
        inputs.episode_slots,
        key=lambda vid: -_score_of(vid, inputs),
    )[:top_k]
    if not ranked or not need:
        return SHAPE_DIRECT, ranked[:1]

    full = [vid for vid in ranked if need <= inputs.episode_slots.get(vid, frozenset())]

    # Two complete witnesses in different sessions: the memory holds two
    # answers at two times. That is an update, not a contradiction to hide.
    if len(full) >= 2:
        sessions = {inputs.episode_sessions.get(v, "") for v in full[:4]}
        if len(sessions) >= 2:
            return SHAPE_CONFLICT, full[:2]
    if full:
        return SHAPE_DIRECT, full[:1]

    # No single episode carries the whole tuple. Does a pair?
    if len(need) >= 2:
        for i, a in enumerate(ranked):
            sa = inputs.episode_slots.get(a, frozenset())
            for b in ranked[i + 1 :]:
                if need <= (sa | inputs.episode_slots.get(b, frozenset())):
                    return SHAPE_COMPOSED, [a, b]
    return SHAPE_DIRECT, ranked[:1]


def _score_of(vid: str, inputs: SupportInputs) -> float:
    # episode_slots is built from the ranked candidates, so insertion order is
    # already the ranking; this keeps the tie-break stable when it is not.
    try:
        return list(inputs.episode_slots).index(vid) * -1.0
    except ValueError:  # pragma: no cover
        return 0.0


def attest(
    inputs: SupportInputs, threshold: float = 0.0, top_k: int = 8
) -> Attestation:
    """Score, cut, and classify -- the whole verdict in one call."""
    score, feats = support_score(inputs, top_k)

    if inputs.corner_reason:
        return Attestation(
            shape=SHAPE_ABSENT, score=score, reason=inputs.corner_reason,
            threshold=threshold, features=feats,
        )
    if score < threshold:
        return Attestation(
            shape=SHAPE_ABSENT, score=score, reason=REASON_LOW_SUPPORT,
            threshold=threshold, features=feats,
        )
    shape, witness = classify_shape(inputs, top_k)
    return Attestation(
        shape=shape, score=score, reason="", threshold=threshold,
        features=feats, witness=witness,
    )


# --------------------------------------------------------------------------- #
# Where the cut comes from                                                     #
# --------------------------------------------------------------------------- #


def otsu_threshold(scores: Sequence[float], bins: int = 64) -> float:
    """Otsu's method: the cut that maximises between-class variance.

    Chosen because it has *no free parameter* and needs no label. The support
    score is expected to be bimodal -- questions this memory can answer, and
    questions it cannot -- and Otsu finds the valley between two modes from the
    histogram alone. Picking the cut from the corpus's known unanswerable rate
    would need the categories, which is the reverse-engineering the whole
    calibration line exists to avoid.

    Returns 0.0 (abstain on nothing) when the distribution is degenerate: a
    single mode has no valley, and inventing one would delete correct answers
    to satisfy a formula.
    """
    vals = [float(s) for s in scores if s == s]  # drop NaN
    if len(vals) < 2:
        return 0.0
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return 0.0

    width = (hi - lo) / bins
    hist = [0] * bins
    for v in vals:
        idx = min(bins - 1, int((v - lo) / width))
        hist[idx] += 1

    total = len(vals)
    sum_all = sum(h * (lo + (i + 0.5) * width) for i, h in enumerate(hist))
    w_bg = 0.0
    sum_bg = 0.0
    best_var, best_t = -1.0, 0.0
    for i, h in enumerate(hist):
        w_bg += h
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += h * (lo + (i + 0.5) * width)
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var = var
            best_t = lo + (i + 1) * width
    return best_t


def calibrate_threshold(
    scores: Sequence[float], method: str = "otsu", quantile: float = 0.2,
    floor: float = 0.0, bins: int = 64,
) -> tuple[float, str]:
    """Returns ``(threshold, provenance)``; provenance goes into the report.

    ``otsu``      label-free bimodal cut, the default and the defensible one;
    ``quantile``  declared fraction of the question set -- honest fallback, but
                  it asserts how many questions are unanswerable, which is a
                  fact about the benchmark;
    ``absolute``  the literal ``floor``, for reproducing an old number.
    """
    if method == "absolute":
        return floor, "absolute"
    if method == "quantile":
        vals = sorted(float(s) for s in scores)
        if not vals:
            return floor, "fallback"
        i = min(len(vals) - 1, max(0, int(round(quantile * (len(vals) - 1)))))
        return vals[i], "quantile"
    if method != "otsu":
        raise ValueError(f"unknown support threshold method {method!r}")
    t = otsu_threshold(scores, bins)
    return (t, "otsu") if t > 0.0 else (floor, "fallback")


# --------------------------------------------------------------------------- #
# The two-sided objective                                                      #
# --------------------------------------------------------------------------- #


def auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(a random unanswerable question scores below a random answerable one).

    Rank-based (Mann-Whitney), so ties count half and no binning is involved.
    0.5 is a coin flip and means the mechanism has no chain.
    """
    if not positive or not negative:
        return 0.5
    merged = sorted([(v, 0) for v in negative] + [(v, 1) for v in positive])
    ranks: list[float] = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_pos = sum(r for r, (_, lab) in zip(ranks, merged, strict=True) if lab == 1)
    n_pos, n_neg = len(positive), len(negative)
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


#: F1 of the two halves of the benchmark under the condition the projection is
#: read against. From ``results/L2d-derived/metrics.json`` (2026-08-20, 1986
#: questions, gpt-5-mini): ``f1_substantive`` over n=1540, and adversarial F1,
#: which *is* the abstention rate. Constants here only scale a projection --
#: nothing in the scoring or the threshold reads them.
REFERENCE_F1_SUBSTANTIVE = 0.5263
REFERENCE_F1_ADVERSARIAL = 0.5762


def operating_curve(
    substantive: Sequence[float],
    adversarial: Sequence[float],
    thresholds: Sequence[float] | None = None,
    f1_substantive: float = REFERENCE_F1_SUBSTANTIVE,
    f1_adversarial: float = REFERENCE_F1_ADVERSARIAL,
) -> list[dict]:
    """Abstention caught vs. correct answers deleted, at every cut.

    ``net_questions`` is the point of the table. A mechanism is not judged by
    what it gains on the questions it targets; multi-hop is 282 questions and
    adversarial is 446, so a gain of +0.05 on one against a loss of 0.03 on the
    other is net zero. Here the same arithmetic is explicit::

        caught * n_adv * (1 - f1_adversarial)   questions won
      - deleted * n_sub * f1_substantive        questions destroyed

    A curve whose maximum ``net_questions`` is near zero is the proposal dying
    cheaply, which is what the number is for.
    """
    all_scores = sorted(set(list(substantive) + list(adversarial)))
    if thresholds is None:
        if len(all_scores) <= 40:
            thresholds = all_scores
        else:
            step = len(all_scores) / 40.0
            thresholds = [all_scores[min(len(all_scores) - 1, int(i * step))]
                          for i in range(40)]

    n_sub, n_adv = len(substantive), len(adversarial)
    rows: list[dict] = []
    for t in thresholds:
        caught = sum(1 for s in adversarial if s < t) / n_adv if n_adv else 0.0
        deleted = sum(1 for s in substantive if s < t) / n_sub if n_sub else 0.0
        won = caught * n_adv * (1.0 - f1_adversarial)
        lost = deleted * n_sub * f1_substantive
        rows.append({
            "threshold": round(float(t), 4),
            "adversarial_caught": round(caught, 4),
            "substantive_deleted": round(deleted, 4),
            "net_questions": round(won - lost, 1),
            "net_micro": round((won - lost) / max(n_sub + n_adv, 1), 4),
        })
    return rows
