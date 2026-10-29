"""The digit readout of the confidence follow-up.

After P-answer's answer the model is asked for a confidence as one digit from 0 to 9. The readout is the log-probability of EACH of the ten digit tokens at the first generated position, taken exactly
(vLLM ``SamplingParams(logprob_token_ids=<the ten ids>, max_tokens=1)``; no top-k, so no digit is ever missing from the readout). From the ten log-probabilities:

* ``mass``, the total probability of the ten digit tokens, not normalised (the digit quality gate: mass >= 0.9 for at least 99 % of the items);
* ``c``, the raw confidence, the expectation of the digit under the distribution renormalised over the ten digits, divided by 9 (NaN when the mass is 0).

Every digit must be exactly one token of the tokenizer and decode to itself (``resolve_digit_token_ids``); otherwise the run stops before any generation.
Nothing here sees a question or a generated text.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nmi.readout.digits import DigitTokenError, resolve_digit_token_ids

DIGITS = tuple(str(d) for d in range(10))
DIGIT_MASS_MIN = 0.9
_POSITIVE_TOLERANCE = 1e-4  # a float32 log-probability may exceed 0 by rounding only
_MASS_TOLERANCE = 1e-3


class ReadoutValueError(ValueError):
    """The ten log-probabilities cannot be raw log-probabilities of ten digit tokens."""


def resolve_digit_ids(tokenizer: Any) -> tuple[int, ...]:
    """Token ids of "0" to "9": each digit must be one token, distinct, and decode to itself (``DigitTokenError`` otherwise)."""
    return resolve_digit_token_ids(tokenizer, DIGITS)


@dataclass(frozen=True)
class Confidence:
    """The readout of one follow-up."""

    logprobs: tuple[float, ...]  # log P(digit d) at the first generated position, d = 0..9 (raw, not renormalised)
    mass: float  # sum over d of P(d)
    c: float  # sum_d d P(d) / sum_d P(d) / 9; NaN when the mass is 0
    gate_pass: bool  # mass >= DIGIT_MASS_MIN


def confidence_from_logprobs(logprobs: Sequence[float]) -> Confidence:
    """Mass and raw confidence from the ten digit log-probabilities (-inf is a probability of 0)."""
    values = [float(x) for x in logprobs]
    if len(values) != len(DIGITS):
        raise ReadoutValueError(f"need {len(DIGITS)} digit log-probabilities, got {len(values)}")
    if any(math.isnan(x) or x == math.inf for x in values):
        raise ReadoutValueError("a digit log-probability is NaN or +inf")
    if any(x > _POSITIVE_TOLERANCE for x in values):
        raise ReadoutValueError("a digit log-probability is positive: the engine is not returning raw log-probabilities")
    probs = np.exp(np.asarray(values, dtype=np.float64))
    mass = float(probs.sum())
    if mass > 1.0 + _MASS_TOLERANCE:
        raise ReadoutValueError(f"the digit probabilities sum to {mass:.4f} > 1: not raw log-probabilities")
    c = float((probs / mass * np.arange(10)).sum() / 9.0) if mass > 0.0 else math.nan
    return Confidence(logprobs=tuple(values), mass=mass, c=c, gate_pass=mass >= DIGIT_MASS_MIN)


__all__ = ["DIGITS", "DIGIT_MASS_MIN", "Confidence", "DigitTokenError", "ReadoutValueError", "confidence_from_logprobs", "resolve_digit_ids"]
