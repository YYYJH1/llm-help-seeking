"""The rules of the price for the batches supp4 and supp5 (configurations not distributed).

The earlier batches (the first batch and supp1 to supp3) use the price and booking rules of ``nmi.api_exp.openrouter`` and ``nmi.api_exp.journal``, which are left unchanged. Those rules leave two
gaps in the cap of 500 CNY, which this module closes for supp4 and supp5. The md5 of this file is part of the freeze record of these two batches (``batch_code_md5``,
``nmi.api_exp.runner.batch_code_md5``), so a change of one rule after their first real send is refused.

* THE CEILING OF THE PRICE AND THE RESERVE OF A SEND (``price_ceiling``, ``PriceCeiling.cost_cny``). Every request of supp4 and supp5 carries ``provider.max_price``, OpenRouter's own ceiling
  on the price of the call, in US dollars per million tokens: ``prompt`` and ``completion``. They are the unit prices of the model in the price file (CNY per million tokens) divided by 7.3
  and multiplied by the MARGIN of the batch (``endpoint.price_margin`` of the configuration, 2.0), rounded UP at the 9th significant digit. OpenRouter forwards a request only to a service
  whose price does not pass ``max_price``; when the pinned service charges more, the request fails before it is forwarded and costs nothing. So the cost of one call is at most its
  tokens times ``max_price``, and the tokens of a send are at most the provable bound (the bytes of the message plus a margin; ``max_tokens`` plus a margin): the CNY that
  a send reserves before it leaves, and that the cap is checked against, is the bound of its tokens at the ceiling that is IN the request, converted back to CNY at 7.3. The ceiling is
  written into the attempt record, and the replay of the ledger works the reserve out of it. The money of every attempt without a result, whatever its number, counts at its reserve in the
  total of the ledger, so a send that is in flight is counted at its upper bound while it flies. (In 36 test calls the price a reply reported was the list price in 33, below it in two and 1.81 times it in one.)
* THE BOOKING stays what it is for the batches of the kind openrouter: the LARGER of the tokens at the price file and the price the reply reports times 7.3
  (``nmi.api_exp.journal.Journal.write_result_success``). It is booked in full, never cut to the reserve and never hidden.
* A BOOKED AMOUNT ABOVE THE RESERVE OF ITS SEND (``exceeds_reserve``) breaks the premise that the cap rests on, as a count of tokens above its bound does, and has the same consequence:
  a lockout record in the ledger, nothing more sent in this session or any later one, until a written disposition closes it (``nmi.api_exp.lockout``). With the margin the
  cap cannot be passed in silence: before every send the booked CNY plus the reserves of all the sends in flight plus the reserve of this one is at most the cap, and a reply that costs
  more than its reserve, which only a service that does not keep to ``max_price`` or a count of tokens above the bound can bring about, ends the run at once.
* THE PRICE THE REPLY REPORTS (``reported_price``). ``nmi.api_exp.openrouter.reported_cost`` drops a price whose plain text is longer than 40 characters, which loses a legal price written
  with trailing zeros (``0.01`` followed by 40 zeros is 0.01 US dollars). Here the number that the client has decoded exactly, an ``int`` or a ``Decimal``, is turned into a
  price by fixed rules: from 0 to 1,000 US dollars, or it is not a price; the trailing zeros of its digits are removed exactly; more than 12 significant digits are rounded UP at the
  12th (towards plus infinity, so that no rounding books less than was reported). The price is kept as the text of that number, and its CNY is the number times 7.3. The runner does not
  hand ``usage.cost`` to ``nmi.api_exp.openrouter.read_router_facts`` for these batches (it takes the field out of the copy of the reply that it passes on), and every
  sum of prices that a report writes is written by ``decimal_text``: plain up to 40 characters, scientific beyond, never the plain expansion of a number such as 1E-99999999.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import MAX_EMAX, MIN_EMIN, ROUND_CEILING, Context, Decimal
from typing import Any

from nmi.api_exp.openrouter import COST_MAX_USD, USD_TO_CNY

REPORTED_DIGITS = 12  # the significant digits of a reported price that are kept; more are rounded up at the 12th
PLAIN_TEXT_MAX = 40  # a price whose plain text is longer than this (a very small one) is kept as scientific notation
CEILING_DIGITS = 9  # the significant digits of a price ceiling (``provider.max_price``): the exact value is rounded UP at the 9th
_ROUND_UP = Context(prec=REPORTED_DIGITS, rounding=ROUND_CEILING, Emin=MIN_EMIN, Emax=MAX_EMAX, traps=[])
_WIDE = Context(prec=60, Emin=MIN_EMIN, Emax=MAX_EMAX, traps=[])
_CEILING_UP = Context(prec=CEILING_DIGITS, rounding=ROUND_CEILING, Emin=MIN_EMIN, Emax=MAX_EMAX, traps=[])


@dataclass(frozen=True)
class ReportedPrice:
    """A price a reply reported, as it is kept: ``usd`` the number (no trailing zeros, at most 12 significant digits, from 0 to 1,000), ``text`` its text, ``cny`` the number times 7.3."""

    usd: Decimal
    text: str
    cny: float


@dataclass(frozen=True)
class PriceCeiling:
    """The ceiling of the price of a call, ``provider.max_price`` of the request: US dollars per million input (``prompt``) and output (``completion``) tokens."""

    prompt: Decimal
    completion: Decimal

    def as_request(self) -> dict[str, float]:
        """The two numbers as the request and the attempt record carry them (JSON numbers; at most 9 significant digits, so a float holds them exactly as they are written)."""
        return {"prompt": float(self.prompt), "completion": float(self.completion)}

    @classmethod
    def from_record(cls, value: Any) -> PriceCeiling:
        """The ceiling of an attempt record; raises ``ValueError`` for anything that is not two numbers from 0 up with at most 9 significant digits."""
        if not isinstance(value, Mapping) or set(value) != {"prompt", "completion"}:
            raise ValueError("the ceiling of the price of an attempt is two numbers, prompt and completion")
        numbers = []
        for key in ("prompt", "completion"):
            x = value[key]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError("the ceiling of the price of an attempt is two numbers from 0 up")
            d = Decimal(repr(x))
            if len(_strip_zeros(d).as_tuple().digits) > CEILING_DIGITS:
                raise ValueError("the ceiling of the price of an attempt has at most 9 significant digits")
            numbers.append(d)
        return cls(*numbers)

    def cost_cny(self, prompt_tokens: int, completion_tokens: int) -> float:
        """The CNY of these token counts at the ceiling: (prompt x P_in + completion x P_out) US dollars per million, times 7.3. The counts are integers (the bounds of a send)."""
        usd = _WIDE.add(_WIDE.multiply(Decimal(int(prompt_tokens)), self.prompt), _WIDE.multiply(Decimal(int(completion_tokens)), self.completion))
        return float(_WIDE.divide(_WIDE.multiply(usd, USD_TO_CNY), Decimal(10**6)))


def price_ceiling(input_per_million_cny: float, output_per_million_cny: float, margin: float) -> PriceCeiling:
    """The ceiling of the price of a model: its unit prices in the price file (CNY per million tokens) divided by 7.3 and times the margin, in US dollars per million tokens, rounded UP at
    the 9th significant digit (so the ceiling is never below the exact value; for the list prices times 7.3 of the price file the division is exact and nothing is rounded)."""
    ceilings = []
    for cny in (input_per_million_cny, output_per_million_cny):
        exact = _WIDE.multiply(_WIDE.divide(Decimal(repr(float(cny))), USD_TO_CNY), Decimal(repr(float(margin))))
        ceilings.append(_strip_zeros(_CEILING_UP.plus(exact)) if exact != 0 else Decimal(0))
    return PriceCeiling(*ceilings)


def exceeds_reserve(booked_cny: float, reserved_cny: float) -> bool:
    """Is this booked amount above the reserve of its send, beyond the rounding of floats (a relative 1e-9)?"""
    return booked_cny > reserved_cny and not math.isclose(booked_cny, reserved_cny, rel_tol=1e-9, abs_tol=1e-12)


def _strip_zeros(number: Decimal) -> Decimal:
    """The same number without the trailing zeros of its digits, exactly (``Decimal.normalize`` rounds at the precision of the context; this does not)."""
    _sign, digits, exponent = number.as_tuple()
    kept = list(digits)
    while len(kept) > 1 and kept[-1] == 0:
        kept.pop()
        exponent += 1
    return Decimal((0, tuple(kept), exponent))


def _text(kept: Decimal) -> str:
    """The text of a price: plain (``0.073``) when that is at most 40 characters, else scientific (``1E-60``). The length is worked out from the digits, so that the plain text of a number
    with a huge negative exponent is never built."""
    _sign, digits, exponent = kept.as_tuple()
    n = len(digits)
    plain_length = n + exponent if exponent >= 0 else (2 - exponent if -exponent >= n else n + 1)
    return format(kept, "f") if plain_length <= PLAIN_TEXT_MAX else str(kept)


def decimal_text(number: Decimal) -> str:
    """The text of a finite number for a report: plain when that is at most 40 characters, else scientific (``1E-99999999``); never the plain expansion of a number with a huge exponent."""
    return _text(number)


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float) and math.isfinite(value):  # (the client makes none; another caller's float is taken at its shortest text)
        return Decimal(repr(value))
    return None


def reported_price(value: Any) -> ReportedPrice | None:
    """The price that a reply reports (``usage.cost``, as the client decoded it: an ``int`` or a ``Decimal``), or None when it is not a price: not a number, a boolean, infinity, NaN,
    below 0 or above 1,000 US dollars. Zero is a price. The digits are kept exactly, but for the trailing zeros, which go, and the digits beyond the 12th significant one, which are rounded up."""
    number = _as_decimal(value)
    if number is None or not number.is_finite() or number < 0 or number > COST_MAX_USD:
        return None
    if number == 0:
        return ReportedPrice(Decimal(0), "0", 0.0)
    kept = _strip_zeros(number)
    if len(kept.as_tuple().digits) > REPORTED_DIGITS:
        kept = _strip_zeros(_ROUND_UP.plus(kept))
    return ReportedPrice(kept, _text(kept), float(_WIDE.multiply(kept, USD_TO_CNY)))
