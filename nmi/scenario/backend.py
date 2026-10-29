"""The generation back end of the scenario S1: greedy generation of a batch of prompts, with the exact digit log-probabilities when asked for.

``GenerationBackend`` is the interface the three-phase protocol (``nmi.scenario.protocol``) works against: ``generate(prompts, spec)`` gives one ``GenResult`` per prompt, or ``None`` for a prompt whose
generation failed or did not come back. ``VLLMGenerationBackend`` implements it on an offline ``vllm.LLM`` (vLLM 0.30.0):

* the prompts are the rendered chat texts, tokenised on the client without special tokens (the template holds them) and sent as token ids; the echoed ``prompt_token_ids`` must equal what was sent;
* ``SamplingParams(temperature=0, max_tokens=..., skip_special_tokens=...)``; the decide phase keeps the special tokens (the default deletes them, and the call markers of Phi and Gemma-4 would vanish);
  the stop tokens are those the engine takes from the model (the template default);
* with ``logprob_token_ids`` (the ten digit ids, ``max_tokens=1``) the log-probability of exactly those ids at the first generated position is returned, in addition to the sampled token. ``logprobs`` is
  left unset. The values must be raw log-probabilities (``logprobs_mode`` of the engine is ``raw_logprobs``, checked at construction).

A batch that raises is a failed batch: the caller (``generate_with_retry``) runs it once more and then records the prompts as missing. A single output that cannot be parsed (no logprobs for a requested id,
an impossible value) is returned as ``None`` for that prompt only. Nothing here prints or logs a prompt or a generated text; exceptions carry class names and counts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from nmi.readout.hf_backend import ReadoutError, encode_prompts
from nmi.readout.vllm_backend import assert_raw_logprobs

MAX_LOGPROB_TOKEN_IDS = 128  # vllm.sampling_params.MAX_LOGPROB_TOKEN_IDS
_POSITIVE_TOLERANCE = 1e-4


class BackendError(ReadoutError):
    """The back end failed on a whole batch, or its output cannot be read."""


@dataclass(frozen=True)
class GenSpec:
    """The settings of one generation call (greedy)."""

    max_tokens: int
    skip_special_tokens: bool = True
    logprob_token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive integer, got {self.max_tokens!r}")
        if self.logprob_token_ids is not None:
            ids = tuple(int(i) for i in self.logprob_token_ids)
            if not ids or len(ids) > MAX_LOGPROB_TOKEN_IDS or len(set(ids)) != len(ids):
                raise ValueError(f"need 1..{MAX_LOGPROB_TOKEN_IDS} distinct token ids, got {len(ids)}")
            object.__setattr__(self, "logprob_token_ids", ids)

    def as_dict(self) -> dict[str, Any]:
        return {"max_tokens": self.max_tokens, "skip_special_tokens": self.skip_special_tokens, "temperature": 0.0, "logprob_token_ids": None if self.logprob_token_ids is None else list(self.logprob_token_ids)}


@dataclass(frozen=True)
class GenResult:
    """One generation: the text, the generated token ids, the finish reason, the requested log-probabilities and the engine's own service time (None where the engine reports none)."""

    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None
    logprobs: dict[int, float] | None = None
    service_s: float | None = None


class GenerationBackend(Protocol):
    def generate(self, prompts: Sequence[str], spec: GenSpec) -> list[GenResult | None]: ...


def _logprob_value(entry: Any, what: str) -> float:
    value = getattr(entry, "logprob", entry)
    try:
        x = float(value)
    except (TypeError, ValueError) as error:
        raise BackendError(f"{what}: logprob is not a number") from error
    if math.isnan(x) or x == math.inf:
        raise BackendError(f"{what}: logprob is {x}")
    if x > _POSITIVE_TOLERANCE:
        raise BackendError(f"{what}: logprob {x} is positive; the engine is not returning raw log-probabilities")
    return x


def service_time(output: Any) -> float | None:
    """The time the engine core spent on the request (last token minus scheduling, both on the engine's monotonic clock); None when the engine keeps no statistics."""
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    try:
        start, end = float(metrics.scheduled_ts), float(metrics.last_token_ts)
    except (AttributeError, TypeError, ValueError):
        return None
    return end - start if end > start > 0.0 else None


def parse_output(output: Any, spec: GenSpec) -> GenResult:
    """The ``GenResult`` of one finished ``RequestOutput`` (or any object with the same attributes)."""
    outputs = getattr(output, "outputs", None)
    if not outputs:
        raise BackendError("RequestOutput has no outputs")
    completion = outputs[0]
    text = getattr(completion, "text", None)
    if not isinstance(text, str):
        raise BackendError("outputs[0].text is not a string")
    token_ids = tuple(int(t) for t in (getattr(completion, "token_ids", None) or ()))
    values: dict[int, float] | None = None
    if spec.logprob_token_ids is not None:
        positions = getattr(completion, "logprobs", None)
        if positions is None or len(positions) < 1 or positions[0] is None:
            raise BackendError("outputs[0].logprobs is missing: logprob_token_ids was not honoured")
        first = positions[0]
        values = {}
        for tid in spec.logprob_token_ids:
            entry = first.get(int(tid)) if hasattr(first, "get") else None
            if entry is None:
                raise BackendError(f"requested token id {tid} is missing from logprobs[0]")
            values[int(tid)] = _logprob_value(entry, f"token {tid}")
    return GenResult(text=text, token_ids=token_ids, finish_reason=getattr(completion, "finish_reason", None), logprobs=values, service_s=service_time(output))


class VLLMGenerationBackend:
    """Greedy generation through an offline ``vllm.LLM`` (or any object with the same ``generate``)."""

    def __init__(self, engine: Any, tokenizer: Any, *, use_tqdm: bool = False, check_logprobs_mode: bool = True) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.use_tqdm = use_tqdm
        self.logprobs_mode = assert_raw_logprobs(engine) if check_logprobs_mode else None

    @classmethod
    def from_model(cls, model_path: str, *, tokenizer: Any = None, **llm_kwargs: Any) -> "VLLMGenerationBackend":
        """Start ``vllm.LLM(model_path, ...)``. ``logprobs_mode`` must not be passed (the default is raw log-probabilities)."""
        if "logprobs_mode" in llm_kwargs:
            raise ValueError("logprobs_mode must stay at the vLLM default (raw_logprobs); do not pass it")
        from vllm import LLM

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(LLM(model=model_path, **llm_kwargs), tokenizer)

    @staticmethod
    def sampling_params(spec: GenSpec) -> Any:
        """``vllm.SamplingParams`` of a spec (imports vLLM)."""
        from vllm import SamplingParams

        kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": spec.max_tokens, "skip_special_tokens": spec.skip_special_tokens}
        if spec.logprob_token_ids is not None:
            kwargs["logprob_token_ids"] = list(spec.logprob_token_ids)
        return SamplingParams(**kwargs)

    def generate(self, prompts: Sequence[str], spec: GenSpec) -> list[GenResult | None]:
        if len(prompts) == 0:
            return []
        id_lists = encode_prompts(self.tokenizer, prompts)
        outputs = self.engine.generate([{"prompt_token_ids": ids} for ids in id_lists], self.sampling_params(spec), use_tqdm=self.use_tqdm)
        if len(outputs) != len(id_lists):
            raise BackendError(f"the engine returned {len(outputs)} outputs for {len(id_lists)} prompts")
        results: list[GenResult | None] = []
        for sent, output in zip(id_lists, outputs, strict=True):
            echoed = getattr(output, "prompt_token_ids", None)
            if echoed is not None and [int(t) for t in echoed] != sent:
                raise BackendError("the engine returned an output for a different prompt (order or content changed)")
            try:
                results.append(parse_output(output, spec))
            except BackendError:
                results.append(None)  # this prompt only: the caller runs it once more
        return results


__all__ = ["BackendError", "GenResult", "GenSpec", "GenerationBackend", "VLLMGenerationBackend", "parse_output", "service_time"]
