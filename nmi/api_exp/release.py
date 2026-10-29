"""Release list of the API experiment: a model is started only if it has been released explicitly.

The list is a file of the state directory, ``release.json``. It is written by the operator by hand: no code of this package writes it, and tests never write the one of the real state
directory (they pass another path through ``ApiHooks.release_file``). Format 1::

    {"format": 1,
     "released": [{"model": "deepseek-v4.1-flash", "basis": "<stated basis of the release>", "t": "2026-01-01T09:00:00+08:00"}, ...]}

Each entry names a model of the configuration once, with the stated basis of the release (``basis``) and the time of the decision (``t``, an ISO 8601 time). A file that is absent is an
empty list: no model is released. A file that is not valid releases nothing and the message says which field is not as it should be. The list grows over time, so it is not part of the
identity of the experiment that every run compares with the freeze record.

Text written by a person does not leave the file. The program checks that ``basis`` is a non-empty string and keeps its sha256 and its length, nothing else. Nothing the program prints or
writes holds the words themselves: not the dry-run report, not a refusal, not the log, not the manifest. What may appear is a model id, the time, the number of entries, and the sha256 and
the length of the words. A field that is not as it should be is named by its place and a fixed word (``released[2].t``: not an ISO 8601 time); its value is never repeated, because whoever
wrote the file may have pasted a question, a reply, an error text or a key into it.

In a real run, a model that is named and not on the list ends the call with exit 2 before the key is read, and ``--model all`` stands for the models of the list and no others
(``scripts/run_api_decisions.py``); ``nmi.api_exp.runner.ApiRun`` checks again after it has taken the process lock. A dry run shows the list and, for each model, whether it is released."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

RELEASE_NAME = "release.json"


def author_text_digest(text: str) -> tuple[str, int]:
    """(sha256, number of characters) of a text that a person wrote. This is all of such a text that the program keeps or shows."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest(), len(text)


@dataclass(frozen=True)
class ReleaseEntry:
    model: str
    basis_sha256: str  # the text of the basis is not kept: its sha256 and its length are
    basis_chars: int
    t: str  # the time, as ``datetime.isoformat`` writes the one that was read


@dataclass(frozen=True)
class Release:
    path: Path
    exists: bool
    entries: tuple[ReleaseEntry, ...]
    problem: str | None  # why the file cannot be used (then nothing is released); None for a valid file and for an absent one

    def allows(self, model_id: str) -> bool:
        return self.problem is None and any(entry.model == model_id for entry in self.entries)

    def models(self) -> list[str]:
        """The released models in the order of the file (empty for a file that is not valid)."""
        return [] if self.problem is not None else [entry.model for entry in self.entries]

    def unreleased(self, model_ids: Iterable[str]) -> list[str]:
        return [m for m in model_ids if not self.allows(m)]

    def refusal(self, model_ids: Iterable[str]) -> str | None:
        """Why a real run of these models is refused, or None if every one of them is released."""
        missing = self.unreleased(model_ids)
        if not missing:
            return None
        if self.problem is not None:
            return f"the release list {self.path} cannot be used ({self.problem}); no model is released, so {missing} cannot start"
        listed = self.models()
        where = f"the release list {self.path}" if self.exists else f"the release list {self.path}, which does not exist (that is an empty list)"
        return f"{missing} {'is' if len(missing) == 1 else 'are'} not on {where}; it releases {listed if listed else 'no model'}. Only an entry in the release list releases a model"

    def as_record(self, known_models: Iterable[str]) -> dict[str, Any]:
        return {
            "file": str(self.path), "exists": self.exists, "problem": self.problem,
            "released": [{"model": e.model, "t": e.t, "basis_sha256": e.basis_sha256, "basis_chars": e.basis_chars} for e in self.entries] if self.problem is None else [],
            "n_released": len(self.entries) if self.problem is None else 0,
            "models": {m: self.allows(m) for m in known_models},
        }


def _entry(item: Any, index: int, known: Collection[str], seen: set[str]) -> ReleaseEntry:
    if not isinstance(item, dict) or set(item) != {"model", "basis", "t"}:
        raise ValueError(f"released[{index}] must be an object with exactly the keys model, basis and t")
    model, basis, t = item["model"], item["basis"], item["t"]
    if not isinstance(model, str) or model not in known:
        raise ValueError(f"released[{index}].model: not a model of the configuration")
    if model in seen:
        raise ValueError(f"released[{index}].model: {model} is listed twice")  # a model of the configuration: its id may be shown
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError(f"released[{index}].basis: the basis of the release is needed (a non-empty text)")
    if not isinstance(t, str):
        raise ValueError(f"released[{index}].t: an ISO 8601 time is needed")  # noqa: TRY004
    try:
        moment = datetime.fromisoformat(t)
    except ValueError:
        raise ValueError(f"released[{index}].t: not an ISO 8601 time") from None
    seen.add(model)
    sha256, chars = author_text_digest(basis)
    return ReleaseEntry(model, sha256, chars, moment.isoformat())


def parse_release(text: str, known_models: Collection[str]) -> tuple[ReleaseEntry, ...]:
    """The entries of a release file; raises ``ValueError`` with the first thing that is wrong."""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):  # (a file of nothing but opening brackets exhausts the depth of the reader)
        raise ValueError("it is not valid JSON") from None
    if not isinstance(data, dict) or set(data) != {"format", "released"}:
        raise ValueError("it must be a JSON object with exactly the keys format and released")
    if data["format"] != 1 or isinstance(data["format"], bool):
        raise ValueError("format must be 1")
    if not isinstance(data["released"], list):
        raise ValueError("released must be a list")  # noqa: TRY004 - every problem of the file is one kind of error for the caller
    seen: set[str] = set()
    return tuple(_entry(item, i, known_models, seen) for i, item in enumerate(data["released"]))


def load_release(path: str | Path, known_models: Collection[str]) -> Release:
    """Read the release list (never writes it). An absent file is an empty list."""
    path = Path(path)
    if not path.is_file():
        return Release(path, False, (), None)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return Release(path, True, (), f"it cannot be read ({type(error).__name__})")
    try:
        return Release(path, True, parse_release(text, known_models), None)
    except ValueError as error:
        return Release(path, True, (), str(error))
