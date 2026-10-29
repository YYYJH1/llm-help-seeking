"""Unit prices of the API experiment, read from ``configs/api_exp_prices.yaml``.

The prices are kept in a file of their own so that checking them against the provider's console does not change the main configuration ``configs/api_exp_v1.yaml`` (whose hash is recorded
with every run), and a run can still be continued in its own directory. The file has

* ``version``: a whole number. Version 0 is the placeholder (12 CNY per million input tokens, 24 per million output tokens, ``verified: false``). When the numbers change, the version
  goes up. The program remembers the hash of every version it has run with (``nmi.api_exp.state``) and refuses a file whose hash differs from the one it knows under the same version;
* ``verified``: true only for numbers read from the console, and then ``checked_on`` (YYYY-MM-DD) and ``source`` must be given. It says what the numbers are; no stage waits for it;
* ``models``: an entry (input and output price per million tokens) for every model of the configuration and for no other. There is no default price.

Every session records the version it ran with. Every send is booked at the price version in force when it was sent (the attempt record keeps a snapshot of the model's prices) and is never worked out
again: the cap is checked against these booked amounts. A report shows a second column in which all the tokens are worked out at the current version (``PriceBook`` is the object to price with).

A token count can be any non-negative integer that a reply holds, so the money of a count can be larger than a float. ``amount_cny`` works the money out for every
count and is the one function that does it (the booking, the replay of the ledger, the reserve and the second column); an amount that does not fit in ``MAX_BOOKED_CNY`` is worked out as
``MAX_BOOKED_CNY``. That is the limit of the number type and not a policy: 10^300 CNY is 10^297 times the cap, and a float above it would be infinity, which neither the JSON of the ledger nor a sum
of booked amounts can hold.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from nmi.api_exp.config import (
    ApiConfigError,
    _mapping,
    _number,
    _string,
    _UniqueKeyLoader,
)
from nmi.provenance import bytes_md5

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_BOOKED_CNY = 1e300  # the amount that an amount above it is worked out as: see the docstring of the module


class PriceError(ValueError):
    """The price file is malformed, incomplete or contradicts what the program knows of an earlier version."""


def _part(tokens: float | Decimal, per_million: float) -> float:
    """The tokens times the price, without the division by a million. A count of 0 or a price of 0 is 0, whatever the other is; a count that does not fit in a float is infinite. A count that
    is an exact ``Decimal`` (a number above the bound with a fraction) becomes a float here, at the money, and nowhere before."""
    if not tokens or not per_million:
        return 0.0
    if isinstance(tokens, Decimal):
        tokens = float(tokens)
    try:
        return tokens * per_million
    except OverflowError:  # an integer count too large for a float (above about 10^308)
        return float("inf")


def amount_cny(input_per_million: float, output_per_million: float, prompt_tokens: float | Decimal, completion_tokens: float | Decimal) -> float:
    """The CNY of these token counts at these prices, which are CNY per million tokens: ``(prompt x input price + completion x output price) / 10^6``, as a finite float. The counts are
    integers of any size (a reply may report a count of any size, and it is booked as it came); an amount above ``MAX_BOOKED_CNY`` is worked out as ``MAX_BOOKED_CNY``. For every ordinary
    count the value is the very float that the plain expression gives."""
    total = (_part(prompt_tokens, input_per_million) + _part(completion_tokens, output_per_million)) / 1e6
    return min(MAX_BOOKED_CNY, total)


@dataclass(frozen=True)
class Price:
    """CNY per million tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, prompt_tokens: float | Decimal, completion_tokens: float | Decimal) -> float:
        return amount_cny(self.input_per_million, self.output_per_million, prompt_tokens, completion_tokens)


@dataclass(frozen=True)
class PriceBook:
    """One version of the unit prices."""

    path: Path
    md5: str
    version: int
    verified: bool
    checked_on: str | None
    source: str
    models: Mapping[str, Price]

    def price(self, model_id: str) -> Price:
        try:
            return self.models[model_id]
        except KeyError:
            raise PriceError(f"the price file has no entry for model {model_id!r}") from None

    def cost(self, model_id: str, prompt_tokens: float | Decimal, completion_tokens: float | Decimal) -> float:
        return self.price(model_id).cost(prompt_tokens, completion_tokens)

    def identity(self) -> dict[str, Any]:
        """What a session writes into its record: enough to say which prices it ran with."""
        return {"version": self.version, "md5": self.md5, "verified": self.verified, "checked_on": self.checked_on}

    def as_record(self) -> dict[str, Any]:
        return {
            **self.identity(), "currency": "CNY", "unit": "per million tokens", "source": self.source,
            "models": {m: {"input": p.input_per_million, "output": p.output_per_million} for m, p in sorted(self.models.items())},
        }


def parse_price_book(data: bytes, path: Path, model_ids: Sequence[str]) -> PriceBook:
    try:
        raw = yaml.load(data, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, ApiConfigError) as error:
        raise PriceError(f"{path.name}: not valid YAML: {error}") from error
    try:
        top = _mapping(raw, "<root>", ("version", "verified", "checked_on", "source", "models"))
        version = int(_number(top["version"], "version", lo=0, integer=True))
        verified = top["verified"]
        if not isinstance(verified, bool):
            raise ApiConfigError("verified: must be true or false")
        checked_on = top["checked_on"]
        if isinstance(checked_on, datetime.date):  # an unquoted YYYY-MM-DD is read by YAML as a date
            checked_on = checked_on.isoformat()
        source = _string(top["source"], "source")
        if checked_on is not None and (not isinstance(checked_on, str) or not _DATE.fullmatch(checked_on)):
            raise ApiConfigError("checked_on: must be null or a date written YYYY-MM-DD (in quotes)")
        if verified and checked_on is None:
            raise ApiConfigError("checked_on: verified prices need the date they were read from the console")
        if version == 0 and verified:
            raise ApiConfigError("verified: version 0 is the placeholder and cannot be verified; raise the version")
        block = top["models"]
        if not isinstance(block, dict):
            raise ApiConfigError("models: must be a mapping")
        missing = [m for m in model_ids if m not in block]
        unknown = [m for m in block if m not in model_ids]
        if missing or unknown:
            raise ApiConfigError(f"models: missing {missing}, unknown {unknown}; there is an entry for every model of the configuration and for no other")
        prices = {}
        for model_id in model_ids:
            entry = _mapping(block[model_id], f"models.{model_id}", ("input", "output"))
            prices[model_id] = Price(float(_number(entry["input"], f"models.{model_id}.input", lo=0.0)), float(_number(entry["output"], f"models.{model_id}.output", lo=0.0)))
    except ApiConfigError as error:
        raise PriceError(f"{path.name}: {error}") from error
    return PriceBook(path=path, md5=bytes_md5(data), version=version, verified=verified, checked_on=checked_on, source=source, models=prices)


def load_price_book(path: str | Path, model_ids: Sequence[str]) -> PriceBook:
    """Read and validate the price file for the given models."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise PriceError(f"the price file {path.name} cannot be read ({type(error).__name__})") from error
    return parse_price_book(data, path, model_ids)
