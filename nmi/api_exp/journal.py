"""Write-ahead journal of the API experiment: the total ledger and the index of successes.

One journal in one fixed place: ``ledger.jsonl`` of the state directory (``nmi.api_exp.state``). It is append-only, and every record is on disk (``fsync``) when the call that wrote it
returns.

* Before a request leaves, ``write_attempt`` writes an ``attempt`` record: run, stage, model, item key, condition, the number of the send for that request, the UTF-8 bytes of the message, the
  provable upper bound of the send in tokens, the unit prices of the model at that moment (a snapshot with the price version) and the CNY reserved at that price. After the response,
  ``write_result_success`` or ``write_result_failure`` writes the ``result`` record of that attempt, with the CNY it is booked for (``charged_cny``) and why (``charge_basis``: ``usage``,
  ``partial``, ``bound`` or ``none``).
* Money is booked once, at the price of the day, and never worked out again: the spent CNY is the sum of ``charged_cny`` of the results plus ``reserved_cny`` of the attempts without a
  result. The cap is checked against this sum. Another price version changes only the reserve of requests not yet sent; the tokens are kept so that a report can show a second column at the
  current prices (``repriced_cny``). On replay every booked amount is checked against the tokens and the price snapshot of its attempt.
* An attempt without a result record is UNCERTAIN: the process may have been killed before the send, after it, or while the answer was on its way. It is booked at its reserved bound for ever,
  and an ordinary resume does not send that request again. ``allow_uncertain`` (``--resend-uncertain``) permits one more send, which still counts against ``max_attempts``. When the last
  allowed attempt is uncertain the request is in the final state ``uncertain_exhausted`` and is not sent again by any option.
* The number of sends of a request is the number of its attempt records over every session. A success ends the request, and so does a 4xx status other than rate limiting. A refused key
  (``auth_refused``) does not end it: the request is ``auth_blocked`` and can be sent again, if it has sends left, once the key is valid. ``write_attempt`` refuses a send for a request that
  has a success, has used its sends, or belongs to another run directory.
* Cap: ``write_attempt`` raises ``CapReached`` and writes nothing when the CNY already booked plus the reserve of this send would pass the cap.
* Quota: the attempts of a (stage, model), over every session and run, are counted, and ``write_attempt`` raises ``SendQuotaReached`` when the quota is used. The quota is the planned keys of
  the stage times 1.1, rounded up (``nmi.api_exp.ledger.send_quota``).
* A bound violation is a success whose usage block has a valid count above the bound of its send. It is found by the replay of the result record (``book_usage``, the function that also books
  the money). While one is open, ``write_attempt`` raises ``BoundViolationOpen`` and nothing more is sent. It is closed only by a ``disposition`` record, which ``nmi.api_exp.lockout`` writes
  after checking the disposition file against the violations.
* A batch with a price margin (``nmi.api_exp.price_rules``) sends every request with a ceiling of the price (``provider.max_price``, US dollars per million tokens: the unit price of the price
  file divided by 7.3 and times the margin), so a service that charges more is not asked and the call costs nothing. The reserve is the token bound at that ceiling (back in CNY at 7.3); the
  margin and the ceiling are written into the attempt record and checked on replay. A success booked for more than the reserve of its send (the larger of the tokens at the price file and the
  price the reply reports, never cut) is a bound violation of the same kind.
* The index of successes (``successes.jsonl``) records, for each (model, item key, condition), the run directory of its first success. It is written after the result record and rebuilt from
  the journal by ``repair_index`` if a crash left it short. The per-run ``responses.jsonl`` is a view that can be rebuilt (``nmi.api_exp.runner``).

Money per kind of result. A success is booked at its usage, each of the two counts on its own (thinking tokens are in the completion tokens): a valid count as the service reported it, a count
that is missing or invalid at its OWN bound (the input bound for the input count, the output bound for the output count). What a count is is decided by ``read_count`` and nowhere else, on the
EXACT value (the reply is decoded into ``int`` and ``Decimal``, never a float). The result record keeps the kind of the reading of each field and the text of the number as the reply wrote it
(``usage.read``); the replay reads that text by the same rule and must reach the same kind and number (a text longer than ``TEXT_MAX_CHARS`` is cut and is not judged a second time). A success
with one count has the basis ``partial``, with none ``bound``. A count above its bound is booked as reported and is a bound violation. A send that ended in a timeout, a damaged body or a 5xx
(``UNKNOWN_BILLING``: the service may have processed it) is booked at the bound when ``charge_unknown`` is set. A refused key, a 429 and any other 4xx are booked at zero.

A line cut off at the end of the file by a crash is dropped when the file is opened for writing; an unreadable line anywhere else stops the program (``JournalError``): the ledger is not to be
guessed at. The journal holds no text that a server wrote, only fixed names, numbers, hashes and the identifiers of our own requests.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from nmi.api_exp.client import (
    AUTH_REFUSED,
    ENDS_REQUEST,
    FAILURE_CLASSES,
    HUGE_COUNT,
    HUGE_DIGITS,
    SUCCESS,
    UNKNOWN_BILLING,
)
from nmi.api_exp.durable import DurableAppender
from nmi.api_exp.price_rules import PriceCeiling, exceeds_reserve, price_ceiling
from nmi.api_exp.prices import PriceBook, amount_cny

LEDGER_NAME = "ledger.jsonl"
INDEX_NAME = "successes.jsonl"
# how a result was booked: usage (both counts reported), partial (one count reported, the other at its own bound), bound (no count reported, or a send of unknown billing), none (zero)
CHARGE_BASES = ("usage", "partial", "bound", "none")
Key = tuple[str, str, str]  # (model, item key, condition id)
_REL = 1e-9


class JournalError(RuntimeError):
    """The journal refuses a write or cannot be read."""


class CapReached(JournalError):
    """The send would take the CNY booked past the cap: nothing was written."""


class SendQuotaReached(JournalError):
    """The (stage, model) has made all the sends its quota allows, over every session: nothing was written."""


class BoundViolationOpen(JournalError):
    """A reply used more tokens than the provable bound of its send, and no disposition has closed that: nothing more is written, so nothing more is sent."""


KIND_COUNT, KIND_OVER, KIND_INVALID, KIND_ABSENT = "count", "over", "invalid", "absent"
READ_KINDS = (KIND_COUNT, KIND_OVER, KIND_INVALID, KIND_ABSENT)
TEXT_MAX_CHARS = 200  # the text of a number that a record keeps; a longer text is cut here, and the record says so (then the kind and the number that the record holds are the decision)
_INTEGER_TEXT = re.compile(r"-?\d+")


def _is_whole(number: Decimal) -> bool:
    """Is this finite decimal a whole number (``12``, ``12.0``, ``12.000``, ``1E+3``)? It is decided on the digits, so that no exponent, however large, has to be turned into an integer."""
    _sign, digits, exponent = number.as_tuple()
    if not any(digits) or exponent >= 0:  # type: ignore[operator]
        return True
    places = -exponent  # type: ignore[operator]
    return places <= len(digits) and not any(digits[-places:])


def _over_value(number: Decimal) -> int | Decimal:
    """The count that a number above its bound is booked as: the integer itself if it is whole, else the exact decimal. A number of 401 digits or more is the largest count (10^400)."""
    if number.adjusted() >= HUGE_DIGITS:
        return HUGE_COUNT
    return int(number) if _is_whole(number) else number


def read_count(value: Any, bound: int) -> tuple[int | Decimal | None, bool]:
    """How one field of a usage block is read against the bound of its own send: ``(the count, whether it is above the bound)``, with None for the count when the field holds no count.

    The rule is applied to the EXACT value, never to a float (a float rounds ``4.0000000000000001`` to 4.0 and makes ``1e-999`` zero before the rule has seen the number). A field that holds a number
    (a boolean excepted): not finite is above every bound (a violation, booked as the largest count, 10^400); zero in any writing is the count 0; a negative number is not a count (booked at the
    bound, no violation); above the bound is a violation however it is written (``1e18``, ``12.5``, ``4.0000000000000001`` against a bound of 4), booked as it came; within the bound, a whole
    number (``12``, ``12.0``, ``12.000``) is a count and anything else (``12.5``, ``1e-999``) is not a count (booked at the bound, no violation). A string, a boolean, null, a list, a dictionary
    and an absent field are not a count either, booked at the bound. A ``float`` is read as the exact ``Decimal`` of its value."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None, False
    if isinstance(value, float):
        value = Decimal(value)
    if isinstance(value, int):
        if value > bound:
            return value, True
        return (None, False) if value < 0 else (value, False)
    if not value.is_finite():
        return HUGE_COUNT, True
    if value.is_zero():
        return 0, False
    if value.is_signed():
        return None, False
    if value > bound:
        return _over_value(value), True
    return (int(value), False) if _is_whole(value) else (None, False)


def exact_text(value: float | Decimal) -> str:
    """The exact decimal text of a number as it was written (``4.0000000000000001``, ``12.000``, ``1E-999``, ``Infinity``). For a number whose exponent the decimal type refuses (more than 18 digits) the reply was decoded into a stand-in (``nmi.api_exp.client.read_float_literal``), and this is the text of the stand-in."""
    if isinstance(value, float):
        value = Decimal(value)
    if isinstance(value, int):
        try:
            return str(value)
        except ValueError:  # an integer of more than 4,300 digits, which the interpreter refuses to write
            return str(Decimal(value))
    return str(value)


@dataclass(frozen=True)
class FieldRead:
    """What one field of a usage block held, read against the bound of its send, as the ledger keeps it: the ``kind`` of the reading (``count``: a count within the bound; ``over``: above the
    bound; ``invalid``: a number that is not a count; ``absent``: no number), the ``count`` that is booked (None for the last two), and the ``text`` of the number as the reply wrote it (see
    ``exact_text`` for the one exception; None when the field held none; cut at ``TEXT_MAX_CHARS`` with ``truncated`` set)."""

    kind: str
    count: int | Decimal | None
    text: str | None
    truncated: bool = False

    @property
    def over(self) -> bool:
        return self.kind == KIND_OVER

    def stored_number(self) -> int | float | None:
        """The JSON number that stands for the booked count in a record: the integer, or, for a number above the bound with a fraction, its float (the number the money is worked out from, see
        ``nmi.api_exp.prices``). Where that float would overflow to infinity, which JSON cannot write, it is the largest count, 10^400. None when no count is booked."""
        if isinstance(self.count, Decimal):
            number = float(self.count)
            return number if math.isfinite(number) else HUGE_COUNT
        return self.count

    def stored_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"kind": self.kind}
        if self.text is not None:
            record["text"] = self.text
        if self.truncated:
            record["truncated"] = True
        return record


def read_field(value: Any, bound: int) -> FieldRead:
    """``read_count`` with the kind and the text, for the booking and for the record."""
    count, over = read_count(value, bound)
    is_number = not isinstance(value, bool) and isinstance(value, (int, float, Decimal))
    if count is not None:
        kind = KIND_OVER if over else KIND_COUNT
    else:
        kind = KIND_INVALID if is_number else KIND_ABSENT
    if not is_number:
        return FieldRead(kind, None, None)
    text = exact_text(value)
    return FieldRead(kind, count, text[:TEXT_MAX_CHARS], len(text) > TEXT_MAX_CHARS)


def restore_field(stored: Any, info: Any, bound: int) -> FieldRead:
    """Read a field of a RECORD of the ledger: ``stored`` is the number in the record, ``info`` what the record says of the reading (kind and text). The text is read by the same rule as the reply
    was (``read_count``) and has to give the kind and the number that the record states. A text that was cut (more than ``TEXT_MAX_CHARS`` characters) cannot be read again: the kind and the
    stored number of the record are taken as they are, and only the record is checked against itself. A record with no ``info`` (one written by hand) is read as the number it holds."""
    if not isinstance(info, Mapping):
        return read_field(stored, bound)
    kind, text = info.get("kind"), info.get("text")
    if kind not in READ_KINDS:
        raise ValueError("a reading of a usage field of an unknown kind")
    if kind == KIND_ABSENT:
        return FieldRead(KIND_ABSENT, None, None)
    if not isinstance(text, str):
        raise ValueError("a reading of a number that has no text")  # noqa: TRY004
    if info.get("truncated") is True:  # the kind and the number that the record holds are all that is left of the reading, so they have to agree
        if kind == KIND_INVALID:
            if stored is not None:
                raise ValueError("the number booked in a record is not the one that its own text gives")
            return FieldRead(KIND_INVALID, None, text, True)
        if isinstance(stored, bool) or not isinstance(stored, (int, float)):
            raise ValueError("the number booked in a record is not the one that its own text gives")
        return FieldRead(kind, Decimal(stored) if isinstance(stored, float) else stored, text, True)
    try:
        exact = int(text) if _INTEGER_TEXT.fullmatch(text) else Decimal(text)
    except InvalidOperation:
        raise ValueError("a reading of a number whose text is not a number") from None
    read = read_field(exact, bound)
    if read.kind != kind:
        raise ValueError(f"a usage field that the record calls {kind!r} reads as {read.kind!r} by the same rule")
    booked = read.stored_number()
    if not (booked == stored or (isinstance(booked, float) and isinstance(stored, float) and math.isclose(booked, stored, rel_tol=1e-12))):
        raise ValueError("the number booked in a record is not the one that its own text gives")
    return read


@dataclass(frozen=True)
class UsageBooking:
    """How the usage block of a successful reply is booked, each of its two fields on its own against its own bound (``read_count`` is the rule). A field that holds a count is booked as the
    service reported it; one that holds none is booked at ITS OWN bound. The same rule books the money when the result is written and checks the booked amount when the ledger is replayed, and
    it decides what a bound violation is: a count above its bound, however large it is and however it is written."""

    prompt_tokens: int | Decimal  # the input tokens booked: the reported count, or the bound of the send
    completion_tokens: int | Decimal
    prompt_reported: bool
    completion_reported: bool
    prompt_over: bool  # a count above its bound
    completion_over: bool
    prompt_read: FieldRead
    completion_read: FieldRead

    @property
    def basis(self) -> str:
        if self.prompt_reported and self.completion_reported:
            return "usage"
        return "partial" if self.prompt_reported or self.completion_reported else "bound"

    @property
    def violated(self) -> bool:
        return self.prompt_over or self.completion_over

    def stored_usage(self, usage: Mapping[str, Any]) -> dict[str, Any]:
        """The usage block as the ledger keeps it: the other fields as they were, the two booked fields as the numbers that were booked (None where no count was), and, under ``read``, the kind and the
        text of what each of the two held (see ``exact_text`` for the one exception). It holds only numbers, strings and booleans, so JSON can write it."""
        stored = {k: v for k, v in usage.items() if k not in ("prompt_tokens", "completion_tokens")}
        stored["prompt_tokens"], stored["completion_tokens"] = self.prompt_read.stored_number(), self.completion_read.stored_number()
        stored["read"] = {"prompt_tokens": self.prompt_read.stored_record(), "completion_tokens": self.completion_read.stored_record()}
        return stored


def _booking(prompt: FieldRead, completion: FieldRead, bound_prompt: int, bound_completion: int) -> UsageBooking:
    return UsageBooking(
        prompt_tokens=bound_prompt if prompt.count is None else prompt.count, completion_tokens=bound_completion if completion.count is None else completion.count,
        prompt_reported=prompt.count is not None, completion_reported=completion.count is not None, prompt_over=prompt.over, completion_over=completion.over,
        prompt_read=prompt, completion_read=completion,
    )


def book_usage(usage: Mapping[str, Any] | None, bound_prompt: int, bound_completion: int) -> UsageBooking:
    """The booking of a usage block as the reply gave it (the numbers as the decoder made them: ``int`` and ``Decimal``) against the bound of its send (see ``UsageBooking``)."""
    source = usage or {}
    return _booking(read_field(source.get("prompt_tokens"), bound_prompt), read_field(source.get("completion_tokens"), bound_completion), bound_prompt, bound_completion)


def book_stored_usage(usage: Mapping[str, Any] | None, bound_prompt: int, bound_completion: int) -> UsageBooking:
    """The booking of a usage block as the ledger keeps it: the same decision, got from the text that the record holds, or, for a text that was cut, from the kind and the number that it holds
    (``restore_field``)."""
    source = usage or {}
    info = source.get("read") if isinstance(source.get("read"), Mapping) else {}
    return _booking(
        restore_field(source.get("prompt_tokens"), info.get("prompt_tokens"), bound_prompt), restore_field(source.get("completion_tokens"), info.get("completion_tokens"), bound_completion),
        bound_prompt, bound_completion,
    )


def _json_number(number: int | Decimal) -> int | str:
    """A reported number for a report: an integer as it is, an exact decimal as its text (JSON has no exact decimal)."""
    return str(number) if isinstance(number, Decimal) else number


@dataclass(frozen=True)
class Violation:
    """A lockout record: the result of ``attempt_id`` reported a count above the bound of its send. ``attempt_id`` is its identifier (a disposition record names it). ``prompt`` and
    ``completion`` are (reported, bound) for the count that was above its bound, else None. Nothing here is text that a server wrote."""

    attempt_id: int
    run: str
    model: str
    item_key: str
    condition: str
    send: int
    t: str
    prompt: tuple[int | Decimal, int] | None  # (reported, bound); the reported number is an int, or an exact Decimal for a number with a fraction
    completion: tuple[int | Decimal, int] | None
    cost: tuple[float, float] | None = None  # (booked CNY, reserved CNY): a batch with a price margin booked a reply for more than the reserve of its send

    def components(self) -> list[str]:
        return [name for name, value in (("prompt", self.prompt), ("completion", self.completion), ("cost", self.cost)) if value is not None]

    def describe(self) -> str:
        """For a person: ``prompt 5000 against the bound 1064``. An integer of 16 digits or more is shown as 1.234e+18, an exact decimal as it was written (``4.0000000000000001``; cut after 37
        characters when it has more than 40). The reports hold the number as the booking has it: the exact value when the reply was just read, and, for a record whose text was cut, the number
        that the record holds."""
        parts = []
        for name, value in (("prompt", self.prompt), ("completion", self.completion)):
            if value is not None:
                reported = str(value[0])
                if isinstance(value[0], Decimal):
                    shown = reported if len(reported) <= 40 else f"{reported[:37]}..."
                else:
                    shown = reported if len(reported) <= 15 else f"{reported[0]}.{reported[1:4]}e+{len(reported) - 1}"
                parts.append(f"{name} {shown} against the bound {value[1]}")
        if self.cost is not None:
            parts.append(f"cost {self.cost[0]:.6g} CNY against the reserve {self.cost[1]:.6g} CNY")
        return " and ".join(parts)

    def as_record(self) -> dict[str, Any]:
        return {
            "id": self.attempt_id, "run": self.run, "model": self.model, "item_key": self.item_key, "condition": self.condition, "send": self.send, "t": self.t,
            "components": self.components(), "prompt": None if self.prompt is None else {"reported": _json_number(self.prompt[0]), "bound": self.prompt[1]},
            "completion": None if self.completion is None else {"reported": _json_number(self.completion[0]), "bound": self.completion[1]},
            **({"cost": {"booked_cny": self.cost[0], "reserved_cny": self.cost[1]}} if self.cost is not None else {}),
        }


@dataclass
class KeyInfo:
    run: str
    n_attempts: int = 0
    last_cls: str | None = None  # the class of the result of the last attempt; None when that attempt has no result
    success: dict[str, Any] | None = None  # the reply data of the first success
    success_meta: dict[str, Any] | None = None  # its usage, latency and time (what the per-run view needs besides the reply data)
    n_bytes: int = 0  # the UTF-8 bytes of the message of the last attempt


@dataclass
class OpenAttempt:
    attempt_id: int
    key: Key
    run: str
    model: str
    n: int  # which send of the request this is
    t: str
    n_bytes: int
    bound_prompt: int
    bound_completion: int
    reserved_cny: float
    price_in: float  # CNY per million tokens, the snapshot taken when the attempt was written
    price_out: float
    price_version: int
    stage: str = ""
    margin: float | None = None  # the price margin of the batch, when it has one
    ceiling: PriceCeiling | None = None  # and the ceiling of the price that went into the request with it (provider.max_price): reserved_cny is the cost of the bound at the ceiling


@dataclass
class Bucket:
    """Tokens and booked CNY of one (run, model): the tokens to be worked out again at any price (actual, and at the bound for the open attempts and the results booked at the bound), and
    the CNY of the results as they were booked."""

    actual_prompt: int | Decimal = 0  # (a Decimal once a number above its bound with a fraction has been booked, then the run is locked out; its sum is rounded at 28 significant digits)
    actual_completion: int | Decimal = 0
    bound_prompt: int = 0
    bound_completion: int = 0
    closed_cny: float = 0.0
    versions: set[int] = field(default_factory=set)  # the price versions of the attempts of this bucket

    def prompt(self) -> int | Decimal:
        return self.actual_prompt + self.bound_prompt

    def completion(self) -> int | Decimal:
        return self.actual_completion + self.bound_completion


def _cost(price_in: float, price_out: float, prompt: float | Decimal, completion: float | Decimal) -> float:
    """The CNY of these tokens at a price in CNY per million tokens: ``nmi.api_exp.prices.amount_cny``, the function that ``Price.cost`` is, so that a booked amount reproduces exactly. The
    counts are integers of any size; see there for an amount above ``MAX_BOOKED_CNY``."""
    return amount_cny(price_in, price_out, prompt, completion)


MAX_REPORTED_CNY = 7300.0  # the most a reply may report for ONE call (1,000 US dollars at 7.3): a larger or a negative number is not a price (``nmi.api_exp.openrouter.COST_MAX_USD``)


def _reported_cny(value: Any) -> float:
    """The price a reply reported, in CNY, as the ledger keeps it: a finite number from 0 to ``MAX_REPORTED_CNY``; anything else is a record that cannot be applied."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= MAX_REPORTED_CNY:
        raise ValueError("the price a reply reported is not a number from 0 to the most a call can cost")
    return float(value)


MAX_TOKENS_PER_BYTE = 10**6  # a measured ratio is never kept above this: no estimate can use more (it is the bound of a send, at most bytes + margin, that the estimate is cut at)


def _tokens_per_byte(tokens: int | Decimal, n_bytes: int) -> float:
    """Input tokens per message byte. The count can be any non-negative integer, so the division is done only where the result fits in a float, and the ratio is kept at most
    ``MAX_TOKENS_PER_BYTE``."""
    return float(tokens) / n_bytes if tokens <= n_bytes * MAX_TOKENS_PER_BYTE else float(MAX_TOKENS_PER_BYTE)


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL, abs_tol=1e-12)


@dataclass
class Journal:
    state_dir: Path
    prices: PriceBook  # the price book of this session: the prices of the attempts written now
    max_attempts: int
    cap_cny: float
    charge_unknown: bool
    run_filter: str | None = None
    keys: dict[Key, KeyInfo] = field(default_factory=dict)
    open_attempts: dict[int, OpenAttempt] = field(default_factory=dict)
    buckets: dict[tuple[str, str], Bucket] = field(default_factory=dict)
    counts: dict[tuple[str, str], Counter] = field(default_factory=dict)
    indexed: dict[Key, str] = field(default_factory=dict)
    measured: dict[str, tuple[float, int]] = field(default_factory=dict)  # per model: the largest measured input tokens per byte and the number of measurements
    measured_completion: dict[str, tuple[int, int]] = field(default_factory=dict)  # per model: the largest reported output tokens of a reply (within its bound) and the number of replies
    stage_sends: Counter = field(default_factory=Counter)  # (stage, model) -> attempt records, over every session and run: the sends made
    violations: dict[int, Violation] = field(default_factory=dict)  # attempt id -> bound violation, in the order of the ledger
    dispositions: list[dict[str, Any]] = field(default_factory=list)  # the disposition records of the ledger
    covered: dict[int, int] = field(default_factory=dict)  # violation id -> index in ``dispositions`` of the record that closed it
    max_id: int = 0
    torn_tail: int = 0
    spent_at_open: float = 0.0

    def __post_init__(self) -> None:
        self._ledger: DurableAppender | None = None
        self._index: DurableAppender | None = None

    # ---- loading ---------------------------------------------------------------------------
    @classmethod
    def load(
        cls, state_dir: Path, prices: PriceBook, *, max_attempts: int, cap_cny: float, charge_unknown: bool, run_filter: str | None = None, writable: bool = False,
    ) -> Journal:
        """Read the journal. ``run_filter`` keeps only the records of one run (for the analysis of a run directory). ``writable`` also opens the files for appending."""
        journal = cls(Path(state_dir), prices, max_attempts, cap_cny, charge_unknown, run_filter)
        ledger_path = journal.state_dir / LEDGER_NAME
        for number, record in journal._read_records(ledger_path):
            journal._replay(record, number)
        index_path = journal.state_dir / INDEX_NAME
        for _number, record in ([] if run_filter is not None else journal._read_records(index_path)):
            key = (str(record["model"]), str(record["item_key"]), str(record["condition"]))
            journal.indexed.setdefault(key, str(record["run"]))
        journal.spent_at_open = journal.spent_cny()
        if writable:
            journal.open_writer()
        return journal

    def open_writer(self) -> None:
        """Open the ledger and the index for appending (creating them if need be). A last line cut off by a crash is dropped here."""
        if self._ledger is None:
            self._ledger = DurableAppender(self.state_dir / LEDGER_NAME)
            self._index = DurableAppender(self.state_dir / INDEX_NAME)
            self._ledger.open()
            self._index.open()

    def _read_records(self, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        """The records of a journal file. A last line without its newline is a write cut off by a crash: it was never complete on disk, so it is not read (and it is dropped when the
        file is opened for writing). A line that is not a record anywhere else stops the program."""
        if not path.is_file():
            return
        text = path.read_text(encoding="ascii", errors="replace")
        lines = text.split("\n")
        if lines and lines[-1] != "":
            self.torn_tail += 1
        lines = lines[:-1]  # the part after the last newline is empty, or is the torn line
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or "kind" not in record:
                    raise ValueError("not a record")
            except ValueError:
                raise JournalError(f"{path.name}: line {number} is not a record, and it is not a cut-off last line; the ledger cannot be guessed at") from None
            yield number, record

    def close(self) -> None:
        for appender in (self._ledger, self._index):
            if appender is not None:
                appender.close()
        self._ledger = self._index = None

    # ---- replay ----------------------------------------------------------------------------
    def _bucket(self, run: str, model: str) -> Bucket:
        return self.buckets.setdefault((run, model), Bucket())

    def _count(self, run: str, model: str) -> Counter:
        return self.counts.setdefault((run, model), Counter())

    def _replay(self, record: dict[str, Any], number: int) -> None:
        kind = record["kind"]
        try:
            if kind == "attempt":
                self._apply_attempt(record)
            elif kind == "result":
                self._apply_result(record)
            elif kind == "disposition":
                self._apply_disposition(record)
            else:
                raise KeyError(kind)
        except (KeyError, TypeError, ValueError) as error:
            raise JournalError(f"{LEDGER_NAME}: line {number} is a {kind!r} record that cannot be applied ({type(error).__name__}: {error})") from None

    def _apply_attempt(self, rec: dict[str, Any]) -> None:
        attempt_id = int(rec["id"])
        self.max_id = max(self.max_id, attempt_id)
        run = str(rec["run"])
        if self.run_filter is not None and run != self.run_filter:
            return
        key: Key = (str(rec["model"]), str(rec["item_key"]), str(rec["condition"]))
        bound, price = rec["bound"], rec["price"]
        margin, stored_ceiling = rec.get("price_margin"), rec.get("max_price")
        if (margin is None) != (stored_ceiling is None):
            raise ValueError("the price margin and the ceiling of the price of an attempt go together")
        if margin is not None and (isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 1):
            raise ValueError("the price margin of an attempt is a number of at least 1")
        ceiling = None if stored_ceiling is None else PriceCeiling.from_record(stored_ceiling)
        at = OpenAttempt(
            attempt_id, key, run, key[0], int(rec["n"]), str(rec.get("t", "")), int(rec["bytes"]), int(bound["prompt"]), int(bound["completion"]), float(rec["reserved_cny"]),
            float(price["input"]), float(price["output"]), int(price["version"]), str(rec["stage"]), None if margin is None else float(margin), ceiling,
        )
        if at.n_bytes < 1 or at.bound_prompt < at.n_bytes or at.bound_completion < 0:
            raise ValueError("the bound is not made of the bytes of the message")
        if at.ceiling is None:
            expected_reserve = _cost(at.price_in, at.price_out, at.bound_prompt, at.bound_completion)
        else:
            if at.ceiling != price_ceiling(at.price_in, at.price_out, at.margin):
                raise ValueError("the ceiling of the price of an attempt is not the one its price and its margin give")
            expected_reserve = at.ceiling.cost_cny(at.bound_prompt, at.bound_completion)
        if not _same(at.reserved_cny, expected_reserve):
            raise ValueError("the reserved amount is not the bound at the price of the snapshot (at the ceiling of the price, if the attempt has one)")
        info = self.keys.get(key)
        if info is None:
            info = self.keys[key] = KeyInfo(run=run)
        elif info.run != run:
            raise ValueError("a request belongs to two runs")
        info.n_attempts += 1
        info.last_cls = None
        info.n_bytes = at.n_bytes
        self.open_attempts[attempt_id] = at
        bucket = self._bucket(run, key[0])
        bucket.bound_prompt += at.bound_prompt
        bucket.bound_completion += at.bound_completion
        bucket.versions.add(at.price_version)
        self._count(run, key[0])["attempts"] += 1
        self.stage_sends[(at.stage, key[0])] += 1

    def _apply_result(self, rec: dict[str, Any]) -> None:
        attempt_id = int(rec["id"])
        at = self.open_attempts.pop(attempt_id, None)
        if at is None:
            if self.run_filter is not None:
                return  # the attempt belongs to another run
            raise ValueError("a result without an attempt")
        cls = str(rec["class"])
        if cls != SUCCESS and cls not in FAILURE_CLASSES:
            raise ValueError("an unknown result class")
        basis, charged = str(rec["charge_basis"]), float(rec["charged_cny"])
        if basis not in CHARGE_BASES:
            raise ValueError("an unknown charge basis")
        info = self.keys[at.key]
        info.last_cls = cls
        bucket = self._bucket(at.run, at.model)
        counts = self._count(at.run, at.model)
        bucket.bound_prompt -= at.bound_prompt  # the reservation is released ...
        bucket.bound_completion -= at.bound_completion
        counts["results"] += 1
        counts[cls] += 1
        if cls == SUCCESS:
            usage = rec.get("usage") or {}
            booking = book_stored_usage(usage, at.bound_prompt, at.bound_completion)
            if basis != booking.basis:
                raise ValueError(f"a success whose usage gives the basis {booking.basis!r} is booked as {basis!r}")
            expected = _cost(at.price_in, at.price_out, booking.prompt_tokens, booking.completion_tokens)
            if rec.get("reported_cny") is not None:  # the price that the service reported for the call: the larger of the two is booked (only a batch whose service reports one has the field)
                expected = max(expected, _reported_cny(rec["reported_cny"]))
            if booking.prompt_reported:  # ... replaced by what the service reported, count by count; a count that was not reported stays at its own bound
                bucket.actual_prompt += booking.prompt_tokens
            else:
                bucket.bound_prompt += at.bound_prompt
            if booking.completion_reported:
                bucket.actual_completion += booking.completion_tokens
            else:
                bucket.bound_completion += at.bound_completion
            if booking.basis != "usage":
                counts["charged_by_bound"] += 1
            if booking.basis == "partial":
                counts["partial_usage"] += 1
            if booking.prompt_reported:
                best, n = self.measured.get(at.model, (0.0, 0))
                self.measured[at.model] = (max(best, _tokens_per_byte(booking.prompt_tokens, at.n_bytes)), n + 1)
            if booking.completion_reported and not booking.completion_over:
                best_out, n_out = self.measured_completion.get(at.model, (0, 0))
                self.measured_completion[at.model] = (max(best_out, int(booking.completion_tokens)), n_out + 1)
            over_cost = at.margin is not None and exceeds_reserve(expected, at.reserved_cny)  # (a batch with a price margin only: booked in full all the same)
            if booking.violated or over_cost:
                counts["bound_violations"] += 1
                self.violations[attempt_id] = Violation(
                    attempt_id, at.run, at.model, at.key[1], at.key[2], at.n, str(rec.get("t", "")),
                    (booking.prompt_tokens, at.bound_prompt) if booking.prompt_over else None, (booking.completion_tokens, at.bound_completion) if booking.completion_over else None,
                    (expected, at.reserved_cny) if over_cost else None,
                )
            if info.success is None:
                info.success = dict(rec["reply"])
                info.success_meta = {"usage": dict(usage), "latency_s": rec.get("latency_s"), "t": rec.get("t")}
            else:
                counts["duplicate_success"] += 1
        elif basis == "bound":
            if cls not in UNKNOWN_BILLING:
                raise ValueError("only a send of unknown billing is booked at the bound")
            expected = at.reserved_cny
            bucket.bound_prompt += at.bound_prompt
            bucket.bound_completion += at.bound_completion
            counts["charged_by_bound"] += 1
        elif basis == "none":
            expected = 0.0
        else:
            raise ValueError("a failure is not booked by usage")
        if not _same(charged, expected):
            raise ValueError("the booked amount is not the usage or the bound at the price of the attempt")
        bucket.closed_cny += charged

    def _apply_disposition(self, rec: dict[str, Any]) -> None:
        """A disposition record closes the bound violations it names (they must be in the ledger already; a replay of one run only does not have the violations of the others)."""
        covers = [int(i) for i in rec["covers"]]
        if not covers:
            raise ValueError("a disposition that closes nothing")
        str(rec["reason_sha256"]), int(rec["reason_chars"]), str(rec["date"]), str(rec["file_md5"])  # the fields are there (the reason text is not kept, only its digest)
        unknown = [i for i in covers if i not in self.violations]
        if unknown and self.run_filter is None:
            raise ValueError(f"a disposition for {unknown}, which are not bound violations of the ledger")
        index = len(self.dispositions)
        self.dispositions.append(dict(rec))
        for i in covers:
            self.covered.setdefault(i, index)

    # ---- bound violations and the sends of a stage ------------------------------------------------------
    def open_violations(self) -> list[Violation]:
        """The bound violations that no disposition record has closed, in the order of the ledger. While there is one, nothing is sent (``write_attempt``)."""
        return [v for i, v in self.violations.items() if i not in self.covered]

    def sends_in_stage(self, stage: str, model: str) -> int:
        """The sends made to ``model`` in ``stage``, over every session and every run: its attempt records."""
        return self.stage_sends[(stage, model)]

    # ---- the state of a request ------------------------------------------------------------
    def sends_left(self, model: str, item_key: str, condition: str) -> int:
        return self.max_attempts - self.n_attempts(model, item_key, condition)

    def state(self, model: str, item_key: str, condition: str) -> str:
        """One of ``unsent``, ``success``, ``permanent`` (a 4xx that ends the request), ``exhausted`` (all sends used, none succeeded), ``retryable`` (not ended, sends left), ``uncertain`` (the last
        attempt has no result, sends left), ``uncertain_exhausted`` (the last attempt has no result and no send is left) and ``auth_blocked`` (the last send was refused for the key: not ended).
        Success and the 4xx that ends a request are final; so is the end of the sends (``exhausted``, ``uncertain_exhausted``). A refused key (``auth_blocked``) is not final, whatever the number of
        the send."""
        info = self.keys.get((model, item_key, condition))
        if info is None or info.n_attempts == 0:
            return "unsent"
        if info.success is not None:
            return "success"
        left = self.max_attempts - info.n_attempts
        if info.last_cls is None:
            return "uncertain" if left > 0 else "uncertain_exhausted"
        if info.last_cls in ENDS_REQUEST:
            return "permanent"
        if info.last_cls == AUTH_REFUSED:
            return "auth_blocked"
        if left <= 0:
            return "exhausted"
        return "retryable"

    def can_send(self, model: str, item_key: str, condition: str, *, resend_uncertain: bool = False) -> bool:
        """May this request be sent in this session? A request never sent, one to be sent again, one refused for the key (if it has sends left) and, only on request, an uncertain one."""
        state = self.state(model, item_key, condition)
        if state in ("unsent", "retryable"):
            return True
        if state == "auth_blocked":
            return self.sends_left(model, item_key, condition) > 0
        return state == "uncertain" and resend_uncertain

    def n_attempts(self, model: str, item_key: str, condition: str) -> int:
        info = self.keys.get((model, item_key, condition))
        return 0 if info is None else info.n_attempts

    def success(self, model: str, item_key: str, condition: str) -> dict[str, Any] | None:
        info = self.keys.get((model, item_key, condition))
        return None if info is None else info.success

    def successes(self, run: str | None = None, model: str | None = None) -> Iterator[tuple[Key, dict[str, Any]]]:
        for key, info in self.keys.items():
            if info.success is not None and (run is None or info.run == run) and (model is None or key[0] == model):
                yield key, info.success

    def keys_of(self, run: str, model: str) -> Iterator[tuple[Key, KeyInfo]]:
        for key, info in self.keys.items():
            if info.run == run and key[0] == model:
                yield key, info

    def uncertain(self) -> list[OpenAttempt]:
        return list(self.open_attempts.values())

    def measured_tokens_per_byte(self, model: str) -> tuple[float | None, int]:
        """The largest measured input tokens per message byte of a model (over every success with usage, in every run) and the number of measurements; (None, 0) before the first."""
        best, n = self.measured.get(model, (0.0, 0))
        return (best if n else None), n

    def measured_completion_per_request(self, model: str) -> tuple[int | None, int]:
        """The largest output token count that a reply of the model reported (within the bound of its send; over every success with usage, in every run) and the number of replies; (None, 0)
        before the first. Only the estimate of a batch whose output cap is far above what it uses (the endpoint kind openrouter) takes it (``nmi.api_exp.ledger.reserve_of``)."""
        best, n = self.measured_completion.get(model, (0, 0))
        return (best if n else None), n

    # ---- money -----------------------------------------------------------------------------
    def spent_cny(self, *, run: str | None = None, model: str | None = None) -> float:
        """The CNY booked: the results as they were booked, at the price of their day, and the attempts without a result at their reservation. Never worked out again."""
        closed = math.fsum(b.closed_cny for (r, m), b in self.buckets.items() if (run is None or r == run) and (model is None or m == model))
        opened = math.fsum(a.reserved_cny for a in self.open_attempts.values() if (run is None or a.run == run) and (model is None or a.model == model))
        return closed + opened

    def spent_by_model(self, *, run: str | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        models = {m for (r, m) in self.buckets if run is None or r == run} | {a.model for a in self.open_attempts.values() if run is None or a.run == run}
        for m in sorted(models):
            out[m] = self.spent_cny(run=run, model=m)
        return out

    def repriced_cny(self, prices: PriceBook, *, run: str | None = None, model: str | None = None) -> float:
        """All the tokens worked out again at the prices of ``prices`` (the current version): the second column of a report. The cap does not use this."""
        total = 0.0
        for (r, m), bucket in self.buckets.items():
            if (run is None or r == run) and (model is None or m == model):
                total += prices.cost(m, bucket.prompt(), bucket.completion())
        return total

    def repriced_by_model(self, prices: PriceBook, *, run: str | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        for (r, m), bucket in self.buckets.items():
            if run is None or r == run:
                out[m] = out.get(m, 0.0) + prices.cost(m, bucket.prompt(), bucket.completion())
        return dict(sorted(out.items()))

    def tokens_by_model(self, *, run: str | None = None) -> dict[str, dict[str, int | Decimal]]:
        """The tokens booked per model. A sum is a ``Decimal`` once a number above its bound with a fraction has been booked in it (then the run is locked out). It is added at the default
        precision of the decimal type, 28 significant digits, so it is NOT exact beyond that; the money does not depend on it. The JSON writer of the reports writes it as text."""
        out: dict[str, dict[str, int | Decimal]] = {}
        for (r, m), bucket in self.buckets.items():
            if run is None or r == run:
                t = out.setdefault(m, {"actual_prompt": 0, "actual_completion": 0, "bound_prompt": 0, "bound_completion": 0})
                t["actual_prompt"] += bucket.actual_prompt
                t["actual_completion"] += bucket.actual_completion
                t["bound_prompt"] += bucket.bound_prompt
                t["bound_completion"] += bucket.bound_completion
        return dict(sorted(out.items()))

    def price_versions(self, *, run: str | None = None, model: str | None = None) -> list[int]:
        """The price versions the money was booked at (the versions of the attempts written); for a report."""
        found: set[int] = set()
        for (r, m), bucket in self.buckets.items():
            if (run is None or r == run) and (model is None or m == model):
                found |= bucket.versions
        return sorted(found)

    def session_spent_cny(self) -> float:
        return self.spent_cny() - self.spent_at_open

    def uncertain_reserved_cny(self) -> float:
        return math.fsum(a.reserved_cny for a in self.open_attempts.values())

    def counters(self, run: str, model: str) -> Counter:
        return self.counts.get((run, model), Counter())

    # ---- writes ----------------------------------------------------------------------------
    def _require_writable(self) -> DurableAppender:
        if self._ledger is None:
            raise JournalError("the journal is not open for writing")
        return self._ledger

    def write_attempt(
        self, *, run: str, stage: str, model: str, item_key: str, condition: str, message_bytes: int, bound_prompt: int, bound_completion: int, now: str, allow_uncertain: bool = False,
        stage_quota: int | None = None, price_margin: float | None = None, ceiling: PriceCeiling | None = None,
    ) -> int:
        """The attempt record, on disk before the request leaves. Raises ``BoundViolationOpen`` while a bound violation is open; ``JournalError`` for a request that has a success, belongs to another
        run, has used its sends or is uncertain (unless ``allow_uncertain``); ``SendQuotaReached`` if ``stage_quota`` is given and this (stage, model) has made that many sends already; and
        ``CapReached`` if the send would pass the cap. Returns the id of the attempt.

        ``price_margin`` and ``ceiling`` (together): the margin and the price ceiling (``provider.max_price``) that the request carries. The ceiling must be the one that the price file and the
        margin give; the CNY reserved is the bound of the tokens at the ceiling, and both are kept in the record."""
        appender = self._require_writable()
        open_violations = self.open_violations()
        if open_violations:
            raise BoundViolationOpen(f"bound violation(s) {[v.attempt_id for v in open_violations]} of the ledger are open: nothing is sent until a disposition closes them")
        key: Key = (model, item_key, condition)
        info = self.keys.get(key)
        if info is not None:
            if info.success is not None:
                raise JournalError("this request already has a success: it is never sent again")
            if info.run != run:
                raise JournalError("this request belongs to another run directory")
            if info.n_attempts >= self.max_attempts:
                raise JournalError(f"this request has used its {self.max_attempts} sends")
            if info.last_cls is None and info.n_attempts > 0 and not allow_uncertain:
                raise JournalError("the last attempt of this request has no result (uncertain); it is sent again only on an explicit --resend-uncertain")
        if key in self.indexed and self.indexed[key] is not None:
            raise JournalError("this request is in the index of successes: it is never sent again")
        if message_bytes < 1 or bound_prompt < message_bytes:
            raise JournalError("the bound of a send is at least the bytes of its message")
        if stage_quota is not None and self.stage_sends[(stage, model)] >= stage_quota:
            raise SendQuotaReached(f"{model} has made the {stage_quota} sends that its quota allows in the stage {stage}")
        if (price_margin is None) != (ceiling is None):
            raise JournalError("the price margin and the ceiling of the price of a send go together")
        if price_margin is not None and not (math.isfinite(price_margin) and price_margin >= 1):
            raise JournalError("the price margin of a send is a number of at least 1")
        price = self.prices.price(model)
        if ceiling is None:
            reserved = price.cost(bound_prompt, bound_completion)
        else:
            if ceiling != price_ceiling(price.input_per_million, price.output_per_million, price_margin):
                raise JournalError("the ceiling of the price of this request is not the one that the price file and the margin give")
            reserved = ceiling.cost_cny(bound_prompt, bound_completion)
        spent = self.spent_cny()
        if spent + reserved > self.cap_cny:
            raise CapReached(f"booked {spent:.4f} + the bound of this send {reserved:.4f} CNY would pass the cap {self.cap_cny:.2f}")
        attempt_id = self.max_id + 1
        record = {
            "kind": "attempt", "id": attempt_id, "run": run, "stage": stage, "model": model, "item_key": item_key, "condition": condition, "n": (0 if info is None else info.n_attempts) + 1,
            "bytes": int(message_bytes), "bound": {"prompt": int(bound_prompt), "completion": int(bound_completion)},
            "price": {"version": self.prices.version, "input": price.input_per_million, "output": price.output_per_million}, "reserved_cny": reserved, "t": now,
        }
        if price_margin is not None:
            record["price_margin"] = float(price_margin)
            record["max_price"] = ceiling.as_request()
        appender.append(record)
        self._apply_attempt(record)
        return attempt_id

    def _open_attempt(self, attempt_id: int) -> OpenAttempt:
        at = self.open_attempts.get(attempt_id)
        if at is None:
            raise JournalError(f"attempt {attempt_id} is not open: it has a result already, or it is not in the ledger")
        return at

    def write_result_failure(self, attempt_id: int, *, cls: str, status: int | None, exc: str | None, latency_s: float, now: str) -> float:
        """The result of a send that failed. Returns the CNY booked for it: the bound for a send of unknown billing (a timeout, a damaged body, a 5xx) when ``charge_unknown`` is set,
        else zero."""
        if cls not in FAILURE_CLASSES:
            raise JournalError(f"unknown failure class {cls!r}")
        at = self._open_attempt(attempt_id)
        at_bound = cls in UNKNOWN_BILLING and self.charge_unknown
        charged = at.reserved_cny if at_bound else 0.0
        record = {
            "kind": "result", "id": attempt_id, "class": cls, "status": status, "exc": exc, "latency_s": round(latency_s, 3), "usage": None, "reply": None,
            "charge_basis": "bound" if at_bound else "none", "charged_cny": charged, "t": now,
        }
        self._require_writable().append(record)
        self._apply_result(record)
        return charged

    def write_result_success(
        self, attempt_id: int, *, status: int, reply: dict[str, Any], usage: dict[str, Any], latency_s: float, now: str, reported_cny: float | None = None,
    ) -> float:
        """The result of a send that succeeded. Returns the CNY booked for it: each field of the usage at the price of the attempt when it holds a count, at its own bound when it holds none
        (see ``read_count`` and ``book_usage``). A count above its bound, however written, is booked as reported and opens a bound violation (``open_violations``).

        ``reported_cny`` (a batch of the endpoint kind openrouter): the price that the service says the call cost, in CNY. The amount booked is the LARGER of the tokens at the price of the
        attempt and this price; without it, or with a smaller one, it is the tokens at the price file. The number is in the record, so that the replay books the same amount."""
        at = self._open_attempt(attempt_id)
        booking = book_usage(usage, at.bound_prompt, at.bound_completion)
        charged = _cost(at.price_in, at.price_out, booking.prompt_tokens, booking.completion_tokens)
        if reported_cny is not None:
            charged = max(charged, _reported_cny(reported_cny))
        # the ledger keeps the two booked fields as the numbers that were booked (None where no count was; infinity and NaN, which JSON cannot write, are the largest count) and, under
        # ``read``, the kind and the text of what each field held, so that the replay reaches the same decision
        stored = booking.stored_usage(usage)
        record = {
            "kind": "result", "id": attempt_id, "class": SUCCESS, "status": status, "exc": None, "latency_s": round(latency_s, 3), "usage": stored, "reply": reply,
            "charge_basis": booking.basis, "charged_cny": charged, "t": now,
        }
        if reported_cny is not None:
            record["reported_cny"] = _reported_cny(reported_cny)
        self._require_writable().append(record)
        self._apply_result(record)
        return charged

    def write_disposition(self, *, covers: Sequence[int], reason_sha256: str, reason_chars: int, date: str, file_name: str, file_md5: str, now: str) -> None:
        """Close bound violations: the disposition record, with the date written in the disposition file, the sha256 and the length of the reason (the text itself is not kept) and the md5 of the file as
        it was read. The caller (``nmi.api_exp.lockout``) has checked the file against the violations."""
        appender = self._require_writable()
        ids = sorted({int(i) for i in covers})
        unknown = [i for i in ids if i not in self.violations]
        if not ids or unknown:
            raise JournalError(f"a disposition closes bound violations of the ledger; {unknown or 'none'} given")
        record = {"kind": "disposition", "covers": ids, "reason_sha256": reason_sha256, "reason_chars": int(reason_chars), "date": date, "file": file_name, "file_md5": file_md5, "t": now}
        appender.append(record)
        self._apply_disposition(record)

    # ---- the index of successes -------------------------------------------------------------
    def index_success(self, key: Key, run: str) -> bool:
        """Record the first success of a request in the index (once)."""
        if key in self.indexed:
            return False
        if self._index is None:
            raise JournalError("the journal is not open for writing")
        self._index.append({"kind": "success", "model": key[0], "item_key": key[1], "condition": key[2], "run": run})
        self.indexed[key] = run
        return True

    def missing_from_index(self) -> list[tuple[Key, str]]:
        """Successes of the journal that the index does not have (a crash between the result record and the index line)."""
        return [(key, info.run) for key, info in self.keys.items() if info.success is not None and key not in self.indexed]

    def repair_index(self) -> int:
        n = 0
        for key, run in self.missing_from_index():
            n += int(self.index_success(key, run))
        return n

    def index_conflicts(self) -> list[Key]:
        """Keys whose index entry names another run than the journal does (should be none)."""
        return [key for key, run in self.indexed.items() if key in self.keys and self.keys[key].success is not None and self.keys[key].run != run]
