"""The lockout after a bound violation.

A reply whose usage block has a valid count above the provable bound of its send, or (a batch with a price margin, supp4 and supp5) that is booked for more than the reserve of its send, breaks
the premise that the cap rests on. Stopping the session is not enough: the next session would work with the same bound and send again. So the violation is an experiment-level fact, and it
stays until the operator has dealt with it.

* The lockout RECORD is the result record in the ledger: ``nmi.api_exp.journal`` finds a violation by replaying every result with the one rule that also books its money
  (``book_usage``), so the ledger and the lockout cannot disagree, and a process that dies right after the result is on disk (before the session could stop) is locked out all the same.
  The identifier of a lockout record is the id of the attempt and of its result.
* While a violation is open, every real start and every resume ends with exit 2 before the key is read (``scripts/run_api_decisions.py``, and again after the process lock is taken in
  ``nmi.api_exp.runner.ApiRun``), and ``Journal.write_attempt`` refuses to write an attempt. A dry run works as usual and shows the lockout.
* The program has no command that clears the lockout. The one way out is a file of the state directory that the OPERATOR writes, ``bound_disposition.json``. Format 1::

      {"format": 1,
       "dispositions": [{"reason": "<what was found and why it may go on>", "date": "YYYY-MM-DD", "covers": [{"id": 17, "model": "deepseek-v4.1-flash"}]}]}

  Each entry has the operator's reason, the date, and the lockout records it deals with (the id of the result and the model it is of). The program lets a run go on only when every open
  violation is named by an entry, every entry names violations that are in the ledger (with the model they are of), and the file is valid. It then writes a ``disposition`` record to the
  ledger (the sha256 and the length of the reason, the date, the ids and the md5 of the file as it was read) and the violations are closed. An entry that does not correspond to the ledger
  keeps the lockout, even if another entry would close it. A later violation is a new lockout record and needs an entry of its own.

FREE TEXT DOES NOT LEAVE THE FILE. The reason is checked to be a non-empty string and then only its sha256 and its length are used: not the dry-run
report, not a refusal, not the log and not the record in the ledger holds the words. A field that is not as it should be (the date, an id, a model) is named by its place and a fixed word,
and its value is not repeated; what a message may show is what the ledger says (the ids and the models of the lockout records), a date that has been read as a date, a count and a digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nmi.api_exp.journal import Journal, Violation
from nmi.api_exp.release import author_text_digest

DISPOSITION_NAME = "bound_disposition.json"
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Disposition:
    """One entry of the disposition file. The reason is not kept: its sha256 and its length are."""

    reason_sha256: str
    reason_chars: int
    date: str  # YYYY-MM-DD, read as a date
    covers: tuple[tuple[int, str], ...]  # (id of the lockout record, model)


@dataclass(frozen=True)
class Lockout:
    path: Path
    file_exists: bool
    file_md5: str | None
    violations: tuple[Violation, ...]  # every bound violation of the ledger
    open: tuple[Violation, ...]  # those no disposition record of the ledger has closed
    problems: tuple[str, ...]  # why a real run is refused now (empty: it may go on)
    to_record: tuple[Disposition, ...]  # entries of the file that close open violations and are to be written to the ledger when the run goes on

    @property
    def locked(self) -> bool:
        return bool(self.problems)

    def refusal(self) -> str:
        """The message of the exit 2: what happened and what the operator has to do."""
        names = "; ".join(f"{v.attempt_id}: {v.model}, {v.describe()}" for v in self.open)
        tokens, money = any(v.prompt is not None or v.completion is not None for v in self.open), any(v.cost is not None for v in self.open)
        what = "used more tokens than the provable bound of its send" if not money else ("cost more than the reserve of its send" if not tokens else "used more tokens than the provable bound of its send, or cost more than the reserve of its send")
        return (
            f"the experiment is locked out: a reply {what} (lockout record(s) {names}). Every real start and resume is refused until the "
            f"operator's disposition closes the record(s), and the program has no command that clears them. Why it is still locked: {'; '.join(self.problems)}. The operator writes "
            f"{self.path}; --dry-run shows the lockout records and the format of an entry"
        )

    def as_record(self) -> dict[str, Any]:
        """The lockout block of a dry-run report."""
        return {
            "locked": self.locked, "problems": list(self.problems), "disposition_file": str(self.path), "disposition_file_exists": self.file_exists,
            "disposition_file_md5": self.file_md5, "n_violations": len(self.violations), "open": [v.as_record() for v in self.open],
            "closed": [v.as_record() for v in self.violations if v not in self.open],
            "would_be_recorded": [
                {"reason_sha256": d.reason_sha256, "reason_chars": d.reason_chars, "date": d.date, "covers": [{"id": i, "model": m} for i, m in d.covers]} for d in self.to_record
            ],
            "format": {
                "format": 1, "dispositions": [{"reason": "<what was found and why the run may go on>", "date": "YYYY-MM-DD", "covers": [{"id": v.attempt_id, "model": v.model} for v in self.open]}],
            },
        }


def _disposition(item: Any, index: int) -> Disposition:
    if not isinstance(item, dict) or set(item) != {"reason", "date", "covers"}:
        raise ValueError(f"dispositions[{index}] must be an object with exactly the keys reason, date and covers")
    reason, day, covers = item["reason"], item["date"], item["covers"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"dispositions[{index}].reason: a reason is needed (a non-empty text)")
    if not isinstance(day, str) or not _DATE.fullmatch(day):
        raise ValueError(f"dispositions[{index}].date: YYYY-MM-DD is needed")
    try:
        date.fromisoformat(day)
    except ValueError:
        raise ValueError(f"dispositions[{index}].date: not a date") from None
    if not isinstance(covers, list) or not covers:
        raise ValueError(f"dispositions[{index}].covers: a non-empty list of lockout records is needed")
    out = []
    for j, cover in enumerate(covers):
        if not isinstance(cover, dict) or set(cover) != {"id", "model"}:
            raise ValueError(f"dispositions[{index}].covers[{j}] must be an object with exactly the keys id and model")
        number, model = cover["id"], cover["model"]
        if not isinstance(number, int) or isinstance(number, bool) or number < 1 or not isinstance(model, str) or not model:
            raise ValueError(f"dispositions[{index}].covers[{j}]: id must be the number of a lockout record and model its model")
        out.append((number, model))
    if len({i for i, _ in out}) != len(out):
        raise ValueError(f"dispositions[{index}].covers: a lockout record is named twice")
    sha256, chars = author_text_digest(reason)
    return Disposition(sha256, chars, day, tuple(out))


def parse_dispositions(data: bytes) -> tuple[Disposition, ...]:
    """The entries of a disposition file; raises ``ValueError`` with the first thing that is wrong."""
    try:
        decoded = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):  # (a file of nothing but opening brackets exhausts the depth of the reader)
        raise ValueError("the disposition file is not valid JSON") from None
    if not isinstance(decoded, dict) or set(decoded) != {"format", "dispositions"}:
        raise ValueError("the disposition file must be a JSON object with exactly the keys format and dispositions")
    if decoded["format"] != 1 or isinstance(decoded["format"], bool):
        raise ValueError("the disposition file: format must be 1")
    if not isinstance(decoded["dispositions"], list):
        raise ValueError("the disposition file: dispositions must be a list")  # noqa: TRY004 - every problem of the file is one kind of error for the caller
    return tuple(_disposition(item, i) for i, item in enumerate(decoded["dispositions"]))


def assess(journal: Journal, directory: str | Path) -> Lockout:
    """Is the experiment locked out, and may the disposition file lift it? Reads the ledger as ``journal`` has replayed it and the file in ``directory``; writes nothing."""
    path = Path(directory) / DISPOSITION_NAME
    violations = tuple(journal.violations.values())
    open_ = tuple(journal.open_violations())
    if not open_:
        return Lockout(path, path.is_file(), None, violations, (), (), ())
    ids = [v.attempt_id for v in open_]
    if not path.is_file():
        return Lockout(path, False, None, violations, open_, (f"bound violation(s) {ids} are open and the disposition file does not exist",), ())
    data = path.read_bytes()
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    try:
        entries = parse_dispositions(data)
    except ValueError as error:
        return Lockout(path, True, md5, violations, open_, (str(error),), ())
    problems: list[str] = []
    for index, entry in enumerate(entries):
        for j, (number, model) in enumerate(entry.covers):  # what is shown is the place in the file and what the LEDGER says; the values of the file are not repeated
            violation = journal.violations.get(number)
            if violation is None:
                problems.append(f"dispositions[{index}].covers[{j}].id: not a bound violation of the ledger (they are {sorted(journal.violations)})")
            elif violation.model != model:
                problems.append(f"dispositions[{index}].covers[{j}].model: not the model of the violation {number}, which is of {violation.model}")
    named = {number for entry in entries for number, _ in entry.covers}
    uncovered = [i for i in ids if i not in named]
    if uncovered:
        problems.append(f"bound violation(s) {uncovered} are not named by any entry of the disposition file")
    if problems:
        return Lockout(path, True, md5, violations, open_, tuple(problems), ())
    to_record = tuple(entry for entry in entries if any(number not in journal.covered for number, _ in entry.covers))
    return Lockout(path, True, md5, violations, open_, (), to_record)
