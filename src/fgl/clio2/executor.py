"""Deterministic execution of CLIO2 query plans."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from fgl.clio2.ledger import FactIndex, SemanticLedger, content_tokens, normalized_value
from fgl.clio2.model import EvidenceItem, ExecutionResult, QueryOperator, QueryPlan

_GENERIC_VALUE_TOKENS = {
    "a",
    "an",
    "art",
    "drawing",
    "of",
    "painting",
    "photo",
    "picture",
    "the",
    "this",
}

_GENERIC_QUERY_TOKENS = {
    "both",
    "go",
    "gone",
    "many",
    "much",
    "recent",
    "recently",
    "subject",
    "subjects",
    "time",
    "times",
}


def _root_token(token: str) -> str:
    for suffix in ("ing", "ied", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            root = token[: -len(suffix)]
            return f"{root}y" if suffix == "ied" else root
    return token


def _root_tokens(tokens: set[str]) -> set[str]:
    return {_root_token(token) for token in tokens}


def _concept_key(value: str) -> str:
    tokens = _concept_tokens(value)
    return " ".join(sorted(tokens)) or normalized_value(value)


def _concept_tokens(value: str) -> set[str]:
    tokens = set()
    for token in normalized_value(value).split():
        if token in _GENERIC_VALUE_TOKENS:
            continue
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _semantic_terms(plan: QueryPlan) -> set[str]:
    """Return terms safe to use as a precision gate.

    The planner may preserve conversational scaffolding such as ``how many
    times``.  Treating those words as constraints recreates broad RAG, so the
    executor removes subjects, years, and operator vocabulary before gating.
    """

    terms = content_tokens(" ".join(plan.constraints.terms))
    terms -= content_tokens(" ".join(plan.subjects))
    return {
        term
        for term in terms
        if term not in _GENERIC_QUERY_TOKENS and not term.isdigit()
    }


def _fact_semantic_tokens(fact) -> set[str]:
    return content_tokens(
        f"{fact.object_name} {fact.span} {fact.episode_text}"
    )


def _focus_by_terms(candidates, plan: QueryPlan):
    """Prefer literal semantic matches when the ledger contains any.

    This is intentionally a conditional gate.  If extraction paraphrased the
    question and no literal match exists, dense/lexical ranking remains the
    recall fallback instead of returning an empty answer.
    """

    terms = _root_tokens(_semantic_terms(plan))
    if not terms:
        return candidates
    matching = [
        pair
        for pair in candidates
        if terms & _root_tokens(_fact_semantic_tokens(pair[0]))
    ]
    return matching or candidates


def _fact_date(fact) -> datetime:
    return fact.t_valid.start or fact.episode_ts


def _deduplicate_facts(scored_facts):
    best = {}
    for fact, score in scored_facts:
        key = fact.proposition_id
        if key not in best or score > best[key][1]:
            best[key] = (fact, score)
    return sorted(
        best.values(), key=lambda pair: (-pair[1], -_fact_date(pair[0]).timestamp())
    )


def _group_values(scored_facts) -> list[EvidenceItem]:
    groups: dict[str, EvidenceItem] = {}
    for fact, score in scored_facts:
        key = _concept_key(fact.object_name)
        item = groups.get(key)
        if item is None:
            item = EvidenceItem(
                value=fact.object_name,
                score=score,
                subject=fact.subject_name,
                relation=fact.relation,
                object_type=fact.object_type,
                t_valid=fact.t_valid,
            )
            groups[key] = item
        elif score > item.score:
            item.value = fact.object_name
            item.score = score
            item.t_valid = fact.t_valid
        if fact.episode_id not in item.episode_ids:
            item.episode_ids.append(fact.episode_id)
        if fact.proposition_id not in item.proposition_ids:
            item.proposition_ids.append(fact.proposition_id)
    return sorted(groups.values(), key=lambda item: (-item.score, item.value.casefold()))


class QueryExecutor:
    def __init__(self, memory, ledger: SemanticLedger, fact_index: FactIndex):
        self.memory = memory
        self.ledger = ledger
        self.fact_index = fact_index

    def _candidates(self, question: str, plan: QueryPlan):
        subject_ids = self.ledger.resolve_subjects(plan.subjects)
        candidates = self.fact_index.search(
            question,
            subject_ids=subject_ids,
            relations=plan.relations,
            constraints=plan.constraints,
            limit=None,
        )
        if plan.constraints.companions:
            companion_tokens = content_tokens(" ".join(plan.constraints.companions))
            constrained = [
                pair
                for pair in candidates
                if companion_tokens & content_tokens(pair[0].episode_text)
            ]
            if constrained:
                candidates = constrained
        return _deduplicate_facts(candidates)

    def _direct_episodes(self, question: str, limit: int = 12) -> list[str]:
        return [
            episode.id
            for episode, _ in self.memory.episode_index.search_scored(
                question, k=limit, min_score=0.0
            )
        ]

    def execute(self, question: str, plan: QueryPlan) -> ExecutionResult:
        candidates = self._candidates(question, plan)
        result = ExecutionResult(plan=plan)
        result.candidate_facts = [fact for fact, _ in candidates]
        operator = plan.operator
        if operator == QueryOperator.INTERSECTION:
            self._intersection(result, candidates)
        elif operator == QueryOperator.COUNT_DISTINCT:
            self._count(result, candidates)
        elif operator == QueryOperator.LATEST:
            self._latest(result, candidates)
        elif operator == QueryOperator.TEMPORAL_LOOKUP:
            self._temporal(result, candidates)
        elif operator == QueryOperator.PREMISE_CHECK:
            self._premise(result, candidates)
        else:
            result.items = _group_values(candidates)
        # Raw episodes are a recall backstop, not automatically evidence for
        # a value. The answerer may use them for an inference or a fact that
        # has not yet been normalized into the ledger.
        result.candidate_episode_ids = self._direct_episodes(question)
        if not candidates:
            result.diagnostics.append("no structured fact matched the compiled plan")
        return result

    def _intersection(self, result: ExecutionResult, candidates) -> None:
        subject_ids = self.ledger.resolve_subjects(result.plan.subjects)
        candidates = _focus_by_terms(candidates, result.plan)
        query_roots = _root_tokens(content_tokens(" ".join(result.plan.constraints.terms)))
        if "paint" in query_roots or "draw" in query_roots:
            # The old extractor represents an artwork as ``owns(object)`` and
            # the act as a sibling ``practices(painting)`` fact.  Join those
            # roles inside the same episode so an owned gift with "painted" in
            # its name is not mistaken for something the person painted.
            creation_episodes = {
                fact.episode_id
                for fact, _ in candidates
                if fact.relation == "created"
                or (
                    fact.relation == "practices"
                    and _root_tokens(_fact_semantic_tokens(fact))
                    & {"paint", "draw"}
                )
            }
            candidates = [
                pair
                for pair in candidates
                if pair[0].relation == "created"
                or (
                    pair[0].relation == "owns"
                    and pair[0].episode_id in creation_episodes
                )
            ]
        by_subject: dict[str, list[tuple]] = defaultdict(list)
        for fact, score in candidates:
            # Pronouns such as "this painting" are useful evidence spans but
            # not stable concepts.  Letting them into set algebra creates a
            # false intersection between every person's generic artwork.
            if _concept_tokens(fact.object_name):
                by_subject[fact.subject_id].append((fact, score))
        if len(subject_ids) < 2:
            result.diagnostics.append("intersection needs at least two resolved subjects")
            return
        if any(not by_subject[subject_id] for subject_id in subject_ids):
            return

        items: list[EvidenceItem] = []
        seen: set[str] = set()
        for anchor, anchor_score in by_subject[subject_ids[0]]:
            anchor_tokens = _concept_tokens(anchor.object_name)
            matches = [(anchor, anchor_score)]
            common = set(anchor_tokens)
            for subject_id in subject_ids[1:]:
                compatible = []
                for fact, score in by_subject[subject_id]:
                    tokens = _concept_tokens(fact.object_name)
                    overlap = len(anchor_tokens & tokens) / max(
                        1, min(len(anchor_tokens), len(tokens))
                    )
                    if overlap >= 0.5:
                        compatible.append((fact, score, overlap))
                if not compatible:
                    matches = []
                    break
                fact, score, _ = max(
                    compatible, key=lambda row: (row[2], row[1])
                )
                matches.append((fact, score))
                common &= _concept_tokens(fact.object_name)
            if not matches:
                continue
            key = " ".join(sorted(common or anchor_tokens))
            if key in seen:
                continue
            seen.add(key)
            representative = min(
                (fact for fact, _ in matches),
                key=lambda fact: (len(normalized_value(fact.object_name)), fact.object_name),
            )
            item = EvidenceItem(
                value=representative.object_name,
                score=max(score for _, score in matches),
                subject=" & ".join(result.plan.subjects),
                relation=representative.relation,
                object_type=representative.object_type,
                t_valid=representative.t_valid,
            )
            for fact, _ in matches:
                if fact.episode_id not in item.episode_ids:
                    item.episode_ids.append(fact.episode_id)
                if fact.proposition_id not in item.proposition_ids:
                    item.proposition_ids.append(fact.proposition_id)
            items.append(item)
        result.items = sorted(items, key=lambda item: (-item.score, item.value))

    def _count(self, result: ExecutionResult, candidates) -> None:
        if not candidates:
            return
        focused = _focus_by_terms(candidates, result.plan)
        max_score = focused[0][1]
        # A count must not include every vaguely related fact. Structural
        # filtering happens first; this band keeps paraphrases of the best
        # event while rejecting the subject hub's unrelated history.
        focused = [pair for pair in focused if pair[1] >= max_score - 0.16]
        values = _group_values(focused)
        event_relations = SemanticLedger.EVENT_RELATIONS
        if any(fact.relation in event_relations for fact, _ in focused):
            # Adjacent turns commonly restate one occurrence.  A semantic
            # concept on the same calendar date is one event; the same concept
            # on another date is another event.  This avoids both mention
            # counting and the opposite error of collapsing a recurring event
            # into one long-lived graph edge.
            distinct = {
                fact.episode_ts
                for fact, _ in focused
            }
        else:
            distinct = {_concept_key(item.value) for item in values}
        result.items = values
        result.scalar = len(distinct)

    def _latest(self, result: ExecutionResult, candidates) -> None:
        if not candidates:
            return
        latest_date = max(_fact_date(fact) for fact, _ in candidates)
        latest = [pair for pair in candidates if _fact_date(pair[0]) == latest_date]
        result.items = _group_values(latest)

    def _temporal(self, result: ExecutionResult, candidates) -> None:
        if not candidates:
            return
        fact, score = candidates[0]
        date = _fact_date(fact)
        value = date.strftime("%d %B %Y")
        if fact.t_valid.granularity == "year":
            value = str(date.year)
        elif fact.t_valid.granularity == "month":
            value = date.strftime("%B %Y")
        result.items = [
            EvidenceItem(
                value=value,
                episode_ids=[fact.episode_id],
                proposition_ids=[fact.proposition_id],
                score=score,
                subject=fact.subject_name,
                relation=fact.relation,
                t_valid=fact.t_valid,
            )
        ]
        result.scalar = value

    def _premise(self, result: ExecutionResult, candidates) -> None:
        if not candidates:
            result.scalar = None
            return
        best_fact, _ = candidates[0]
        result.items = _group_values(candidates[:8])
        result.scalar = bool(best_fact.polarity)


__all__ = ["QueryExecutor"]
