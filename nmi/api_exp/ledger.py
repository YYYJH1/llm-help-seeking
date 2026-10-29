"""The budget arithmetic of the API experiment: the provable bound of one send, the estimate of a model and the quota of sends of a stage.

The cap of 500 CNY is hard, and two different numbers are used for two different purposes.

* The PROVABLE bound of ONE send decides whether that send may be made. A token is at least one byte, so the input tokens of a request are at most the UTF-8 bytes of the rendered user
  message plus a fixed margin for the chat template (``budget.template_margin_tokens``, 64 by default); the output tokens are at most ``max_tokens`` of the class, thinking tokens being
  output tokens. The batch of the endpoint kind openrouter has wider bounds, declared in its request classes and measured on the models: the output
  bound is ``max_tokens`` plus 64 (what is sent as ``max_tokens`` is not changed), and the class of ``gpt-6-luna-pro`` has an input bound of three times the bytes plus 2,048 tokens. The
  bound of a send is written into its attempt record, and a reply above it is a violation as in every batch. Before EVERY send the journal checks that the CNY already booked (an attempt without a result counts at its reserved bound) plus this bound does not pass the cap
  (``nmi.api_exp.journal.Journal.write_attempt``). A reply with a count above the bound would break the premise, so the run stops at once and nothing more is sent in any later session
  until a disposition has been recorded (``nmi.api_exp.runner``, ``nmi.api_exp.lockout``).
  The batches supp4 and supp5 reserve the cost of the bound TIMES a price margin (2.0, ``endpoint.price_margin``; ``nmi.api_exp.price_rules.reserve_cny``), and a reply booked for more than that
  reserve is the same kind of violation. The margin is for the check before each send only: the ESTIMATE of a model below does not use it.
* The ESTIMATE of a MODEL only decides whether the model starts. Its remaining requests are counted at their provable bound, or, once the model has measured usage in the ledger, at
  the largest measured tokens per byte of that model times ``budget.measured_factor`` (1.25), never above the bound; the output is ``max_tokens`` in both cases. The sum is multiplied by
  1.1. The model starts only if the CNY already booked plus the estimate is at most the cap (``nmi.api_exp.state.BudgetState.admit`` decides and remembers). An estimate that proves too
  low has one consequence, that the run stops at the cap.
  One batch counts the output differently: the batch of the endpoint kind openrouter, whose group B asks with an output cap of 256 for a reply of one
  token. Once such a model has measured usage its output is the largest output token count that a reply of the model reported, times the same factor, at least 1 and never above
  the output bound; before that it is the output bound. Without this the output bound alone would put the estimate of one such model above the whole cap. Only the estimate changes: the
  bound of each send, which is checked before the send against the cap, is still the output bound.

* The QUOTA of sends is a count, not money: a (stage, model) makes at most its planned keys times ``budget.send_quota_factor`` (1.1) sends in all, over every session, rounded up. The
  rule of at most 8 sends per request stays beside it; both must hold (``nmi.api_exp.journal.Journal.write_attempt``).

The money itself (what has been booked, per run and per model) is kept by ``nmi.api_exp.journal``; this module only turns bytes and ``max_tokens`` into token counts and prices.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from nmi.api_exp.config import RESERVE_MARGIN
from nmi.api_exp.prices import PriceBook


def message_bytes(message: str) -> int:
    """The UTF-8 bytes of a rendered user message: the number the provable bound is made of."""
    return len(message.encode("utf-8", "surrogatepass"))


def request_bound(n_bytes: int, max_tokens: int, template_margin_tokens: int, input_bytes_factor: int = 1, output_margin_tokens: int = 0) -> tuple[int, int]:
    """(input tokens, output tokens) that one send cannot exceed: the bytes of the message (times ``input_bytes_factor``, 1 unless the class says otherwise) plus the template margin, and
    ``max_tokens`` (plus ``output_margin_tokens``, 0 unless the class says otherwise). The defaults are the first batch's bounds."""
    return int(input_bytes_factor) * int(n_bytes) + int(template_margin_tokens), int(max_tokens) + int(output_margin_tokens)


def send_quota(planned_keys: int, factor: float) -> int:
    """The most sends a (stage, model) may make over every session: ``planned_keys`` times ``factor``, rounded up. The product is exact, with the factor as the decimal number the
    configuration states, so that 5,400 keys at 1.1 are 5,940 sends and not 5,941 (floating point gives 5940.000000000001; 600 happens to come out right, 100 and 6,000 do not)."""
    return math.ceil(Fraction(str(factor)) * int(planned_keys))


@dataclass(frozen=True)
class Reserve:
    n_requests: int
    bound_prompt_tokens: int  # the input tokens the estimate is made of, summed over the requests
    bound_completion_tokens: int
    cost_cny: float  # those tokens priced, without the margin
    margin: float
    reserve_cny: float  # cost_cny times (1 + margin)
    basis: str  # "byte_bound" (no measured usage yet) or "measured" (the largest measured tokens per byte times the factor)
    tokens_per_byte: float | None  # the largest measured tokens per byte of the model, when it has any
    n_measured: int
    completion_basis: str = "output_cap"  # "output_cap" (max_tokens a request) or "measured" (the largest measured output tokens of a reply times the factor; the endpoint kind openrouter)
    completion_per_request: int | None = None  # the output tokens a request is counted at

    def as_record(self) -> dict[str, Any]:
        return {
            "n_requests": self.n_requests, "bound_prompt_tokens": self.bound_prompt_tokens, "bound_completion_tokens": self.bound_completion_tokens, "cost_cny": self.cost_cny,
            "margin": self.margin, "reserve_cny": self.reserve_cny, "basis": self.basis, "tokens_per_byte": self.tokens_per_byte, "n_measured": self.n_measured,
            "completion_basis": self.completion_basis, "completion_per_request": self.completion_per_request,
        }


def reserve_of(
    prices: PriceBook, model_id: str, bytes_of_requests: Iterable[int], max_tokens: int, template_margin_tokens: int, margin: float = RESERVE_MARGIN, *,
    tokens_per_byte: float | None = None, n_measured: int = 0, measured_factor: float = 1.25, measured_completion: int | None = None,
    input_bytes_factor: int = 1, output_margin_tokens: int = 0,
) -> Reserve:
    """The estimate of a model for the requests with these message sizes (UTF-8 bytes). With ``tokens_per_byte`` (the largest measured value of the model) each request is counted at
    that times ``measured_factor``, otherwise at its provable bound; the measured count is never above the bound. The output of a request is its bound (``max_tokens`` plus
    ``output_margin_tokens``), except for a model whose batch counts it from the measurement (``measured_completion``, the largest output token count of a reply of the model): then
    ``ceil(measured_completion * measured_factor)``, at least 1 and never above the bound. ``input_bytes_factor`` and ``output_margin_tokens`` are those of the class of the model."""
    n = bound_prompt = 0
    output_bound = int(max_tokens) + int(output_margin_tokens)
    for n_bytes in bytes_of_requests:
        n += 1
        provable = request_bound(n_bytes, max_tokens, template_margin_tokens, input_bytes_factor, output_margin_tokens)[0]
        bound_prompt += provable if tokens_per_byte is None else min(provable, math.ceil(n_bytes * tokens_per_byte * measured_factor))
    per_request = output_bound if measured_completion is None else min(output_bound, max(1, math.ceil(measured_completion * measured_factor)))
    bound_completion = n * per_request
    cost = prices.cost(model_id, bound_prompt, bound_completion)
    return Reserve(
        n, bound_prompt, bound_completion, cost, margin, cost * (1.0 + margin), "byte_bound" if tokens_per_byte is None else "measured", tokens_per_byte, n_measured,
        "output_cap" if measured_completion is None else "measured", per_request,
    )
