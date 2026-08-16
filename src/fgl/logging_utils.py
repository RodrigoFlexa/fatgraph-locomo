"""Structured JSONL logging of every agent decision."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append-only JSONL sink.  One file per (condition, conversation)."""

    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.records: list[dict] = []
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        else:
            self._fh = None

    def log(self, event: str, **fields: Any) -> None:
        rec = {"ts": time.time(), "event": event, **fields}
        self.records.append(rec)
        if self._fh is not None:
            self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()

    def count(self, event: str) -> int:
        return sum(1 for r in self.records if r["event"] == event)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class NullLogger(JsonlLogger):
    def __init__(self) -> None:  # noqa: D107
        super().__init__(Path("/dev/null"), enabled=False)
