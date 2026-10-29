"""The per-run view of the replies: ``<run dir>/responses.jsonl``.

One JSON object per line, one line per request that has a success: the FIRST success of a (model, item key, condition). The truth is the journal (``nmi.api_exp.journal``), which has the same
data in the result record of the success; this file is the copy that sits in the run directory, written right after the journal, and rebuilt from the journal by the runner when a crash
left it short. A line is

``{"kind": "reply", "model", "item_key", "condition", "message_md5", "prompt_chars", "max_tokens", "decision", "first_char", "reply_chars", "reply_sha256", "finish", "returned_model",
"reasoning_present", "anomaly", "attempts", "usage", "latency_s", "t"}``

and holds no text that a server wrote: ``decision`` is answer, ask or format_failure; ``first_char`` is set only when it is a digit 1 to 5; ``reply_chars`` and ``reply_sha256`` are the
length and the sha256 of the content; ``finish`` is from a fixed list; ``returned_model`` is the model field if it has the shape of a model name, else a fixed word. A request that never
had a success has no line here: its attempts are in the journal.

The file is only ever appended to, every line is ``fsync``-ed, and a line cut off by a crash is dropped when the file is opened for writing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Self

from nmi.api_exp.durable import DurableAppender

STORE_NAME = "responses.jsonl"
REPLY = "reply"
Key = tuple[str, str, str]  # (model, item key, condition id)


class StoreError(RuntimeError):
    """The store cannot be read or a record is malformed."""


def key_of(record: Mapping[str, Any]) -> Key:
    try:
        return (str(record["model"]), str(record["item_key"]), str(record["condition"]))
    except KeyError as error:
        raise StoreError(f"a record lacks the field {error.args[0]!r}") from error


def scan_store_lines(path: str | Path) -> Iterator[dict[str, Any] | None]:
    """Every complete line of a store file as a record; ``None`` for a line that is not a reply record."""
    p = Path(path)
    if not p.is_file():
        return
    text = p.read_text(encoding="ascii", errors="replace")
    lines = text.split("\n")
    for line in lines[:-1]:  # the part after the last newline is empty or a write cut off by a crash
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            yield None
            continue
        if not isinstance(record, dict) or record.get("kind") != REPLY or not {"model", "item_key", "condition"} <= set(record):
            yield None
            continue
        yield record


class ResponseStore:
    """The replies of one run directory, loaded into memory and appended to as the run goes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._replies: dict[Key, dict[str, Any]] = {}
        self._appender: DurableAppender | None = None
        self.bad_lines = 0

    # ---- opening ---------------------------------------------------------------------------
    @classmethod
    def read(cls, path: str | Path) -> Self:
        """The store of a finished or running run, loaded for reading only (nothing is opened for appending, nothing is written)."""
        store = cls(path)
        store._load()
        return store

    def open(self) -> Self:
        self._load()
        self._appender = DurableAppender(self.path)
        self._appender.open()
        return self

    def _load(self) -> None:
        self._replies.clear()
        self.bad_lines = 0
        for record in scan_store_lines(self.path):
            if record is None:
                self.bad_lines += 1
                continue
            self._replies.setdefault(key_of(record), record)  # the first reply of a key wins

    def close(self) -> None:
        if self._appender is not None:
            self._appender.close()
            self._appender = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- queries ---------------------------------------------------------------------------
    def has_reply(self, model: str, item_key: str, condition: str) -> bool:
        return (model, item_key, condition) in self._replies

    def get_reply(self, model: str, item_key: str, condition: str) -> dict[str, Any] | None:
        return self._replies.get((model, item_key, condition))

    def replies(self, model: str | None = None) -> Iterator[dict[str, Any]]:
        for key, record in self._replies.items():
            if model is None or key[0] == model:
                yield record

    def n_replies(self, model: str | None = None) -> int:
        return sum(1 for _ in self.replies(model))

    # ---- appending -------------------------------------------------------------------------
    def append_reply(self, record: Mapping[str, Any]) -> bool:
        """Write the reply unless the key already has one (the first success is kept). Returns whether it was written."""
        if record.get("kind") != REPLY:
            raise StoreError("append_reply needs a record of kind 'reply'")
        key = key_of(record)
        if key in self._replies:
            return False
        if self._appender is None:
            raise StoreError("the store is not open")
        self._appender.append(dict(record))
        self._replies[key] = dict(record)
        return True
