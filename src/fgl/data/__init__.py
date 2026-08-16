"""Dataset access. Only LoCoMo today; the loader returns framework-native objects."""

from fgl.data.locomo import (
    ABSTAIN_ANSWER,
    CATEGORY_NAMES,
    TEMPORAL_QUESTION_SUFFIX,
    Conversation,
    Question,
    Session,
    Turn,
    evidence_session,
    load_conversations,
    normalize_timestamp,
)

__all__ = [
    "ABSTAIN_ANSWER", "CATEGORY_NAMES", "TEMPORAL_QUESTION_SUFFIX", "Conversation",
    "Question", "Session", "Turn", "evidence_session", "load_conversations",
    "normalize_timestamp",
]
