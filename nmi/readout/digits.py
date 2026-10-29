"""Digit readout over the first generated token.

The device answers by the first generated token, which is one of the digits "1".."5"
("5" asks for help). This module holds what does not depend on a backend:

* the five digit token ids (each digit must be exactly one token, otherwise an error);
* the sampling arguments of the two vLLM interfaces;
* the quantities derived from the five digit logprobs.

The backends (HF, offline vLLM, HTTP vLLM) parse their own output into five logprobs and
then call ``readout_from_logprobs``.

vLLM facts, read from the installed sources (vllm 0.30.0):

* ``vllm/sampling_params.py``: ``SamplingParams.logprob_token_ids`` returns the logprobs of
  exactly these ids plus the sampled token; ``logprobs`` may stay unset and, when set, must
  equal ``len(logprob_token_ids)`` (validation near line 879); at most 128 ids.
* ``vllm/entrypoints/openai/completion/protocol.py``: the HTTP request is rejected unless
  ``logprobs`` is set whenever ``logprob_token_ids`` is (near line 510), so the HTTP request
  carries both. The response keys ``top_logprobs`` by token STRING (parsed by the backend).
* ``vllm/outputs.py`` and ``vllm/logprobs.py``: ``RequestOutput.outputs[0].token_ids`` holds
  the generated id and ``outputs[0].logprobs[0]`` is a ``dict[int, Logprob]`` (or a
  ``FlatLogprobs`` position) with ``.logprob`` per id.

Derived quantities: digit mass = sum of exp(logprob) over the five
digits; gate = mass >= 0.5; margin m = log P(ask) - max over the other digits of log P;
m > 0 asks for help, m = 0 answers, and the self-answer is the most likely of "1".."4".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DIGITS = ("1", "2", "3", "4", "5")
DIGIT_MASS_GATE = 0.5
MAX_LOGPROB_TOKEN_IDS = 128  # vllm.sampling_params.MAX_LOGPROB_TOKEN_IDS


class DigitTokenError(ValueError):
    """A digit is not exactly one token for this tokenizer."""


def resolve_digit_token_ids(tokenizer: Any, digits: Sequence[str] = DIGITS) -> tuple[int, ...]:
    """Token id of each digit; every digit must encode to exactly one token.

    Also required: the ids are distinct and each id decodes back to the digit itself (a
    tokenizer that returns a word-initial variant would put the readout on the wrong token).
    """
    ids: list[int] = []
    for digit in digits:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise DigitTokenError(
                f"digit {digit!r} is {len(encoded)} tokens ({list(encoded)}); each digit must be one token"
            )
        token_id = int(encoded[0])
        decoded = tokenizer.decode([token_id])
        if decoded != digit:
            raise DigitTokenError(f"token {token_id} for digit {digit!r} decodes to {decoded!r}")
        ids.append(token_id)
    if len(set(ids)) != len(ids):
        raise DigitTokenError(f"digit token ids are not distinct: {ids}")
    return tuple(ids)


def _check_ids(token_ids: Sequence[int]) -> list[int]:
    ids = [int(t) for t in token_ids]
    if not ids or len(ids) > MAX_LOGPROB_TOKEN_IDS or len(set(ids)) != len(ids):
        raise ValueError(f"need 1..{MAX_LOGPROB_TOKEN_IDS} distinct token ids, got {ids}")
    return ids


def offline_sampling_kwargs(
    token_ids: Sequence[int],
    *,
    max_tokens: int = 1,
    temperature: float = 0.0,
    logprobs: int | None = None,
) -> dict[str, Any]:
    """Arguments for ``vllm.SamplingParams``; ``logprobs`` is left out unless given, and a
    given value must equal ``len(token_ids)`` (vLLM raises otherwise)."""
    ids = _check_ids(token_ids)
    if logprobs is not None and logprobs != len(ids):
        raise ValueError(f"logprobs ({logprobs}) must equal len(logprob_token_ids) ({len(ids)})")
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens, "temperature": temperature, "logprob_token_ids": ids,
    }
    if logprobs is not None:
        kwargs["logprobs"] = logprobs
    return kwargs


def http_request_fields(
    token_ids: Sequence[int], *, max_tokens: int = 1, temperature: float = 0.0
) -> dict[str, Any]:
    """Sampling fields of a ``/v1/completions`` request: both ``logprobs`` and
    ``logprob_token_ids`` are required together by the server."""
    ids = _check_ids(token_ids)
    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": len(ids),
        "logprob_token_ids": ids,
    }


def free_token_is_digit(token_id: int, token_ids: Sequence[int]) -> bool:
    """Whether the freely generated first token is one of the digit tokens."""
    return int(token_id) in {int(t) for t in token_ids}


def digit_mass(logprobs: Sequence[float]) -> float:
    """Total probability of the digit tokens (their exp-logprobs summed)."""
    return float(sum(math.exp(float(x)) for x in logprobs))


@dataclass(frozen=True)
class DigitReadout:
    """One trial's readout over the digit tokens."""

    logprobs: tuple[float, ...]
    mass: float
    gate_pass: bool
    margin: float  # log P(ask) - max log P(others); NaN when no digit has positive probability
    ask: bool  # margin > 0
    self_choice: int  # 1-based index of the most likely non-ask digit (lowest index on ties)

    def to_dict(self) -> dict[str, Any]:
        return {
            "digit_logprobs": list(self.logprobs),
            "digit_mass": self.mass,
            "gate_pass": self.gate_pass,
            "margin": self.margin,
            "ask": self.ask,
            "self_choice": self.self_choice,
        }


def readout_from_logprobs(
    logprobs: Sequence[float],
    *,
    ask_index: int = 4,
    gate: float = DIGIT_MASS_GATE,
) -> DigitReadout:
    """Derive mass, gate, margin and decision from the digit logprobs (digit order 1..5).

    ``-inf`` (probability zero) is allowed; NaN and ``+inf`` are errors. When every digit
    has probability zero the margin is NaN and the trial does not ask (it fails the gate
    anyway, and the caller counts it there).
    """
    values = tuple(float(x) for x in logprobs)
    if len(values) < 2 or not 0 <= ask_index < len(values):
        raise ValueError(f"need at least two logprobs and a valid ask index, got {len(values)} and {ask_index}")
    if any(math.isnan(x) or x == math.inf for x in values):
        raise ValueError("digit logprobs must not be NaN or +inf")
    mass = digit_mass(values)
    candidates = [i for i in range(len(values)) if i != ask_index]
    best = max(candidates, key=lambda i: (values[i], -i))  # lowest index wins a tie
    if values[ask_index] == -math.inf and values[best] == -math.inf:
        margin = math.nan
    else:
        margin = values[ask_index] - values[best]  # +inf if only the ask digit has probability
    return DigitReadout(
        logprobs=values,
        mass=mass,
        gate_pass=mass >= gate,
        margin=margin,
        ask=bool(margin > 0),
        self_choice=best + 1,
    )


@dataclass(frozen=True)
class ReadoutSpec:
    """The readout numbers of the run configuration (section ``readout``)."""

    digits: tuple[str, ...]
    ask_token: str
    digit_mass_gate: float
    min_gate_pass_rate: float
    offline_max_tokens: int
    offline_temperature: float
    http_max_tokens: int
    http_temperature: float
    http_logprobs: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ReadoutSpec":
        r = cfg["readout"]
        return cls(
            digits=tuple(r["device_tokens"]),
            ask_token=r["ask_token"],
            digit_mass_gate=float(r["digit_mass_gate"]),
            min_gate_pass_rate=float(r["min_gate_pass_rate"]),
            offline_max_tokens=int(r["vllm_offline"]["max_tokens"]),
            offline_temperature=float(r["vllm_offline"]["temperature"]),
            http_max_tokens=int(r["vllm_http"]["max_tokens"]),
            http_temperature=float(r["vllm_http"]["temperature"]),
            http_logprobs=int(r["vllm_http"]["logprobs"]),
        )

    @property
    def ask_index(self) -> int:
        return self.digits.index(self.ask_token)

    def token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        return resolve_digit_token_ids(tokenizer, self.digits)

    def offline_kwargs(self, token_ids: Sequence[int]) -> dict[str, Any]:
        return offline_sampling_kwargs(
            token_ids, max_tokens=self.offline_max_tokens, temperature=self.offline_temperature
        )

    def http_fields(self, token_ids: Sequence[int]) -> dict[str, Any]:
        fields = http_request_fields(
            token_ids, max_tokens=self.http_max_tokens, temperature=self.http_temperature
        )
        if fields["logprobs"] != self.http_logprobs:
            raise ValueError(
                f"config readout.vllm_http.logprobs is {self.http_logprobs} but there are "
                f"{fields['logprobs']} digit tokens; vLLM needs them equal"
            )
        return fields

    def readout(self, logprobs: Sequence[float]) -> DigitReadout:
        return readout_from_logprobs(logprobs, ask_index=self.ask_index, gate=self.digit_mass_gate)
