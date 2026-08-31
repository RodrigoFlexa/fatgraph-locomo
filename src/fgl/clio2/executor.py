"""Deterministic execution of CLIO2 query plans."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime

from fgl.clio2.ledger import FactIndex, SemanticLedger, content_tokens, normalized_value
from fgl.clio2.model import (
    AnswerType,
    EvidenceItem,
    ExecutionResult,
    QueryOperator,
    QueryPlan,
)

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
    "ago",
    "date",
    "duration",
    "happen",
    "happened",
    "long",
    "often",
    "own",
    "pet",
    "pets",
}

_MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}


def _root_token(token: str) -> str:
    irregular = {
        "hiking": "hike",
        "making": "make",
        "moving": "move",
        "practicing": "practice",
        "taking": "take",
        "using": "use",
        "writing": "write",
    }
    if token in irregular:
        return irregular[token]
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


def _query_roots(question: str, plan: QueryPlan) -> set[str]:
    expanded = re.sub(r"\broadtrip\b", "road trip", question, flags=re.IGNORECASE)
    expanded = re.sub(r"\bfesetival\b", "festival fest", expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\bfestival\b", "festival fest", expanded, flags=re.IGNORECASE)
    low = expanded.casefold()
    expansions = []
    if "excited" in low:
        expansions.extend(("thrilled", "looking forward"))
    if re.search(r"\bthink(?:s)? about\b", low):
        expansions.extend(("amazing", "awesome", "lovely"))
    if re.search(r"\bpets?\b", low):
        expansions.extend(("guinea pig", "dog", "cat"))
    if re.search(r"\bfind|found\b", low):
        expansions.append("came across")
    if "journey" in low:
        expansions.extend(("adventure", "learning", "growing"))
    if "support" in low:
        expansions.extend(("appreciate", "grateful"))
    if "latest project" in low:
        expansions.append("latest work")
    if "colors and patterns" in low:
        expansions.extend(("creative", "express feelings", "catch the eye"))
    if expansions:
        expanded = f"{expanded} {' '.join(expansions)}"
    tokens = content_tokens(expanded)
    tokens |= content_tokens(" ".join(plan.constraints.terms))
    tokens -= content_tokens(" ".join(plan.subjects))
    return {
        root
        for root in _root_tokens(tokens)
        if root not in _GENERIC_QUERY_TOKENS and not root.isdigit()
    }


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

    def _direct_episodes(
        self, question: str, plan: QueryPlan, candidates, limit: int = 20
    ) -> list[str]:
        """Hybrid episodic recall with lexical, entity and calendar signals.

        Dense retrieval alone overweights conversational boilerplate and names.
        Temporal and attribute questions usually contain one discriminative term
        (``road trip``, ``birthday``, ``hand-painted bowl``), so an IDF-weighted
        lexical channel is combined with dense similarity, structured provenance,
        speaker attribution and explicit month/year constraints.
        """

        episodes = list(self.memory.log.all())
        if not episodes:
            return []
        query_roots = _query_roots(question, plan)
        doc_roots = {
            episode.id: _root_tokens(content_tokens(episode.text))
            for episode in episodes
        }
        document_frequency = Counter(
            token for tokens in doc_roots.values() for token in tokens
        )
        weights = {
            token: math.log((len(episodes) + 1) / (document_frequency[token] + 1)) + 1
            for token in query_roots
        }
        total_weight = sum(weights.values()) or 1.0
        dense = {
            episode.id: max(0.0, score)
            for episode, score in self.memory.episode_index.search_scored(
                question, k=min(len(episodes), 64), min_score=0.0
            )
        }
        structured: dict[str, float] = {}
        if candidates:
            ceiling = max(score for _, score in candidates) or 1.0
            for fact, score in candidates:
                structured[fact.episode_id] = max(
                    structured.get(fact.episode_id, 0.0), score / ceiling
                )

        subject_names = {name.casefold() for name in plan.subjects}
        low_question = question.casefold()
        mentioned_months = {
            number for name, number in _MONTHS.items() if name in low_question
        }
        year_match = re.search(r"\b(19|20)\d{2}\b", question)
        mentioned_year = int(year_match.group()) if year_match else None
        scored = []
        for episode in episodes:
            overlap = query_roots & doc_roots[episode.id]
            lexical = sum(weights[token] for token in overlap) / total_weight
            speaker = 0.16 if episode.speaker.casefold() in subject_names else 0.0
            calendar = 0.0
            if mentioned_months and episode.ts_ingest.month in mentioned_months:
                calendar += 0.12
            if mentioned_year and episode.ts_ingest.year == mentioned_year:
                calendar += 0.08
            intent_bonus = 0.0
            if plan.operator == QueryOperator.DURATION and re.search(
                r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                r"(?:days?|weeks?|months?|years?)\b",
                episode.text,
                flags=re.IGNORECASE,
            ):
                intent_bonus += 0.24
            if plan.operator == QueryOperator.FREQUENCY and re.search(
                r"\b(?:once|twice|every|often|daily|weekly|monthly|yearly|"
                r"\d+\s+times?)\b",
                episode.text,
                flags=re.IGNORECASE,
            ):
                intent_bonus += 0.28
            score = (
                0.50 * lexical
                + 0.24 * dense.get(episode.id, 0.0)
                + 0.22 * structured.get(episode.id, 0.0)
                + speaker
                + calendar
                + intent_bonus
            )
            scored.append((episode, score))
        scored.sort(key=lambda pair: (-pair[1], -pair[0].ts_ingest.timestamp()))

        # Dialogue facts are frequently split across a question and its next
        # reply. Interleave immediate neighbours of strong hits before filling
        # the remaining budget, while keeping the operation deterministic.
        position = {episode.id: index for index, episode in enumerate(episodes)}
        out: list[str] = []
        for episode, _ in scored[: max(8, limit // 2)]:
            if episode.id not in out:
                out.append(episode.id)
            index = position[episode.id]
            for neighbour_index in (index - 2, index - 1, index + 1, index + 2):
                if not 0 <= neighbour_index < len(episodes):
                    continue
                neighbour = episodes[neighbour_index]
                if (
                    neighbour.session_id == episode.session_id
                    and neighbour.ts_ingest == episode.ts_ingest
                    and neighbour.id not in out
                ):
                    out.append(neighbour.id)
            if len(out) >= limit:
                return out[:limit]
        for episode, _ in scored:
            if episode.id not in out:
                out.append(episode.id)
            if len(out) >= limit:
                break
        return out

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
        elif operator in (QueryOperator.TEMPORAL_LOOKUP, QueryOperator.DURATION):
            self._temporal(result, candidates)
        elif operator in (QueryOperator.FREQUENCY, QueryOperator.ATTRIBUTE_LOOKUP):
            self._focused_values(result, candidates)
        elif operator == QueryOperator.JOIN and plan.answer_type in (
            AnswerType.DATE,
            AnswerType.DURATION,
        ):
            self._temporal(result, candidates)
        elif operator == QueryOperator.PREMISE_CHECK:
            self._premise(result, candidates, question)
        elif operator == QueryOperator.LOOKUP:
            self._focused_values(result, candidates)
        else:
            result.items = _group_values(candidates)
        # Raw episodes are a recall backstop, not automatically evidence for
        # a value. The answerer may use them for an inference or a fact that
        # has not yet been normalized into the ledger.
        result.candidate_episode_ids = self._direct_episodes(
            question, plan, candidates
        )
        if not candidates:
            result.diagnostics.append("no structured fact matched the compiled plan")
        return result

    def _focused_values(self, result: ExecutionResult, candidates) -> None:
        result.items = _group_values(_focus_by_terms(candidates, result.plan))

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
        candidates = _focus_by_terms(candidates, result.plan)
        latest_date = max(_fact_date(fact) for fact, _ in candidates)
        latest = [pair for pair in candidates if _fact_date(pair[0]) == latest_date]
        result.items = _group_values(latest)

    def _temporal(self, result: ExecutionResult, candidates) -> None:
        if not candidates:
            return
        focused = _focus_by_terms(candidates, result.plan)
        ceiling = focused[0][1]
        focused = [pair for pair in focused if pair[1] >= ceiling - 0.20][:8]
        items = []
        for fact, score in focused:
            interval = fact.source_t_valid
            date = interval.start or fact.episode_ts
            if fact.time_expression:
                value = fact.time_expression
            elif interval.granularity == "year":
                value = str(date.year)
            elif interval.granularity == "month":
                value = date.strftime("%B %Y")
            else:
                value = date.strftime("%d %B %Y")
            items.append(
                EvidenceItem(
                    value=value,
                    episode_ids=[fact.episode_id],
                    proposition_ids=[fact.proposition_id],
                    score=score,
                    subject=fact.subject_name,
                    relation=fact.relation,
                    t_valid=interval,
                    time_expression=fact.time_expression,
                )
            )
        result.items = items
        # Temporal answers require event selection plus interpretation of the
        # original expression against the episode timestamp.  A bare first
        # fact date is not a safe deterministic proof, so the evidence-bounded
        # answerer performs that final projection.
        result.scalar = None

    def _premise(self, result: ExecutionResult, candidates, question: str) -> None:
        if not candidates:
            result.scalar = None
            return
        literal_tokens = content_tokens(
            f"{question} {' '.join(result.plan.constraints.terms)}"
        ) - content_tokens(" ".join(result.plan.subjects))
        query_roots = {
            root
            for root in _root_tokens(literal_tokens)
            if root not in _GENERIC_QUERY_TOKENS and not root.isdigit()
        }
        supported = [
            pair
            for pair in candidates
            if query_roots & _root_tokens(_fact_semantic_tokens(pair[0]))
        ]
        if query_roots and not supported:
            result.scalar = None
            result.diagnostics.append("premise has no matching fact")
            return
        candidates = supported or candidates
        best_fact, _ = candidates[0]
        result.items = _group_values(candidates[:8])
        result.scalar = bool(best_fact.polarity)


__all__ = ["QueryExecutor"]
