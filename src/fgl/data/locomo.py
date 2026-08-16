"""LoCoMo dataset access.

Loads the *official* data file (``data/locomo10.json`` of
``snap-research/locomo``, branch ``code``) without filtering or subsampling
anything: all 10 conversations and all 1986 questions, every category
(1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

#: The string the official scorer accepts as an abstention (it looks for the
#: substrings ``'not mentioned'`` or ``'no information available'``).
ABSTAIN_ANSWER = "Not mentioned in the conversation"

#: Suffix the official pipeline appends to temporal questions
#: (``task_eval/gpt_utils.py``); replicated for comparability.
TEMPORAL_QUESTION_SUFFIX = (
    " Use DATE of CONVERSATION to answer with an approximate date."
)


# --------------------------------------------------------------------------- #
# Dataclasses                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class Turn:
    dia_id: str  # "D3:12"
    speaker: str
    text: str
    session_num: int
    img_caption: str = ""

    @property
    def rendered(self) -> str:
        """Turn as it goes into a prompt (matches the official rendering)."""
        out = f"{self.speaker}: {self.text}"
        if self.img_caption:
            out += f" [shares {self.img_caption}]"
        return out


@dataclass
class Session:
    num: int
    date_time_raw: str  # "1:56 pm on 8 May, 2023"
    timestamp: str  # ISO-8601, sortable
    turns: list[Turn] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"S{self.num}"


@dataclass
class Question:
    question: str
    answer: str  # normalised gold answer (adversarial -> ABSTAIN_ANSWER)
    category: int
    evidence: list[str]
    raw: dict = field(default_factory=dict)

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, str(self.category))

    @property
    def is_adversarial(self) -> bool:
        return self.category == 5

    def prompt_question(self) -> str:
        """Question text as fed to the model."""
        if self.category == 2:
            return self.question + TEMPORAL_QUESTION_SUFFIX
        return self.question


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    questions: list[Question]

    def turns(self) -> Iterator[Turn]:
        for s in self.sessions:
            yield from s.turns

    @property
    def n_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    def turn_by_id(self, dia_id: str) -> Optional[Turn]:
        return self._index().get(dia_id)

    def _index(self) -> dict[str, Turn]:
        if not hasattr(self, "_turn_index"):
            self._turn_index = {t.dia_id: t for t in self.turns()}  # type: ignore[attr-defined]
        return self._turn_index  # type: ignore[attr-defined]

    def render_full(self, header: bool = True) -> str:
        """The whole conversation as text, sessions in chronological order."""
        parts: list[str] = []
        if header:
            parts.append(
                f"Below is a conversation between two people: {self.speaker_a} and "
                f"{self.speaker_b}. The conversation takes place over multiple days "
                f"and the date of each conversation is written at the beginning of "
                f"the conversation.\n"
            )
        for s in self.sessions:
            parts.append(f"DATE: {s.date_time_raw}\nCONVERSATION:")
            for t in s.turns:
                parts.append(t.rendered)
            parts.append("")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

_DATE_RE = re.compile(
    r"^\s*(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ap>am|pm)\s+on\s+"
    r"(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+),?\s*(?P<y>\d{4})\s*$",
    re.IGNORECASE,
)
_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}


def normalize_timestamp(raw: str) -> str:
    """``"1:56 pm on 8 May, 2023"`` -> ``"2023-05-08T13:56:00"``.

    Falls back to the raw string when the format is unexpected, so ingestion
    never crashes on an unseen variant (sorting then degrades gracefully).
    """
    m = _DATE_RE.match(raw or "")
    if not m:
        return raw or ""
    hour = int(m.group("h")) % 12
    if m.group("ap").lower() == "pm":
        hour += 12
    month = _MONTHS.get(m.group("mon").lower()[:3] and m.group("mon").lower())
    if month is None:
        month = next(
            (v for k, v in _MONTHS.items() if k.startswith(m.group("mon").lower()[:3])),
            1,
        )
    dt = datetime(int(m.group("y")), month, int(m.group("d")), hour, int(m.group("m")))
    return dt.isoformat()


def load_conversations(data_file: str | Path) -> list[Conversation]:
    """Load every conversation from the official LoCoMo json."""
    raw = json.loads(Path(data_file).read_text(encoding="utf-8"))
    return [_parse_conversation(item) for item in raw]


def _parse_conversation(item: dict) -> Conversation:
    conv = item["conversation"]
    sessions: list[Session] = []
    for num in range(1, 100):
        key = f"session_{num}"
        if key not in conv:
            continue
        turns_raw = conv[key] or []
        if not turns_raw:
            continue
        date_raw = conv.get(f"{key}_date_time", "")
        session = Session(
            num=num, date_time_raw=date_raw, timestamp=normalize_timestamp(date_raw)
        )
        for t in turns_raw:
            session.turns.append(
                Turn(
                    dia_id=t["dia_id"],
                    speaker=t["speaker"],
                    text=t.get("text", ""),
                    session_num=num,
                    img_caption=(t.get("blip_caption") or "") if t.get("img_url") else "",
                )
            )
        sessions.append(session)
    sessions.sort(key=lambda s: (s.timestamp or "", s.num))

    questions = [_parse_question(q) for q in item.get("qa", [])]
    return Conversation(
        sample_id=item["sample_id"],
        speaker_a=conv.get("speaker_a", "Speaker A"),
        speaker_b=conv.get("speaker_b", "Speaker B"),
        sessions=sessions,
        questions=questions,
    )


def _parse_question(q: dict) -> Question:
    """Normalise the gold answer.

    Adversarial items (category 5) carry ``adversarial_answer`` and usually have
    **no** ``answer`` key at all -- the official ``eval_question_answering``
    crashes on them (see COERENCIA.md item C7).  We normalise the gold answer to
    the abstention string, which is exactly what the official category-5 rule
    scores as correct.
    """
    category = int(q.get("category", 0))
    if category == 5:
        answer = ABSTAIN_ANSWER
    else:
        answer = q.get("answer", "")
        answer = str(answer) if answer is not None else ""
    evidence = [e.replace("(", "").replace(")", "") for e in (q.get("evidence") or [])]
    return Question(
        question=q.get("question", ""),
        answer=answer,
        category=category,
        evidence=[e for e in evidence if e],
        raw=q,
    )


def evidence_session(dia_id: str) -> int:
    """``"D3:12"`` -> ``3``."""
    return int(dia_id.split(":")[0].lstrip("Dd"))
