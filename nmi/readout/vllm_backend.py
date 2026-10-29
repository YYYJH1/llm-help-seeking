"""Offline vLLM readout backend.

One request per prompt with ``SamplingParams(max_tokens=1, temperature=0, logprob_token_ids=<the five
digit ids>)`` (``logprobs`` stays unset; if it were set it would have to equal 5). The result is read
from ``RequestOutput.outputs[0]``:

* ``token_ids[0]`` is the freely generated token;
* ``logprobs[0]`` is one position (a ``dict[int, Logprob]``, or position 0 of a ``FlatLogprobs``) that holds
  the five requested ids and the sampled token; ``.logprob`` is the value.

The structure of the result follows ``vllm/outputs.py`` and ``vllm/logprobs.py`` of vLLM 0.30.0.

Raw logprobs: vLLM's default ``logprobs_mode`` is ``raw_logprobs`` (``vllm/config/model.py``), the values
before temperature and logit processors. This backend never sets the option, refuses a caller that
tries to, and checks the engine's mode after construction. The parser also rejects values that
cannot be log-probabilities (a positive logprob, digit probabilities summing above one), which is how a
server running in ``raw_logits`` mode would show up.

Prompts are tokenised once on the client (``encode_prompts``, no special tokens added, identical to the
HF backend) and sent as token ids, and the echoed ``prompt_token_ids`` are compared with what was sent, so a
reordered or altered prompt cannot pass unnoticed.

``import vllm`` is lazy; the parser needs no vLLM import and works on any object with the same attributes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from nmi.readout.digits import ReadoutSpec, digit_mass, free_token_is_digit
from nmi.readout.hf_backend import ReadoutError, TrialReadout, encode_prompts

EXPECTED_LOGPROBS_MODE = "raw_logprobs"
_POSITIVE_TOLERANCE = 1e-4  # a float32 log-probability may exceed 0 by rounding only
_MASS_TOLERANCE = 1e-3


class VLLMParseError(ReadoutError):
    """A vLLM ``RequestOutput`` does not have the expected content."""


def _logprob_value(entry: Any, what: str) -> float:
    value = getattr(entry, "logprob", entry)
    try:
        x = float(value)
    except (TypeError, ValueError) as error:
        raise VLLMParseError(f"{what}: logprob {value!r} is not a number") from error
    if math.isnan(x) or x == math.inf:
        raise VLLMParseError(f"{what}: logprob is {x}")
    if x > _POSITIVE_TOLERANCE:
        raise VLLMParseError(
            f"{what}: logprob {x} is positive; the engine is not returning raw log-probabilities "
            f"(logprobs_mode must stay at {EXPECTED_LOGPROBS_MODE!r})"
        )
    return x


def parse_request_output(output: Any, token_ids: Sequence[int], spec: ReadoutSpec) -> TrialReadout:
    """Readout of one finished ``RequestOutput`` for the digit ids ``token_ids``."""
    outputs = getattr(output, "outputs", None)
    if not outputs:
        raise VLLMParseError("RequestOutput has no outputs")
    completion = outputs[0]
    generated = getattr(completion, "token_ids", None)
    if generated is None or len(generated) < 1:
        raise VLLMParseError("outputs[0].token_ids is empty: no token was generated")
    free_id = int(generated[0])
    positions = getattr(completion, "logprobs", None)
    if positions is None or len(positions) < 1:
        raise VLLMParseError("outputs[0].logprobs is missing: logprob_token_ids was not honoured")
    first = positions[0]
    if first is None:
        raise VLLMParseError("outputs[0].logprobs[0] is None")
    values: list[float] = []
    for digit, tid in zip(spec.digits, token_ids):
        entry = first.get(int(tid)) if hasattr(first, "get") else None
        if entry is None:
            raise VLLMParseError(
                f"digit {digit!r} (token id {tid}) is missing from logprobs[0]; ids present: {sorted(first)}"
            )
        values.append(_logprob_value(entry, f"digit {digit!r}"))
    if digit_mass(values) > 1.0 + _MASS_TOLERANCE:
        raise VLLMParseError(
            f"digit probabilities sum to {digit_mass(values):.4f} > 1: not raw log-probabilities"
        )
    free_entry = first.get(free_id)
    free_lp = None if free_entry is None else _logprob_value(free_entry, "free token")
    return TrialReadout(
        readout=spec.readout(values),
        free_token_id=free_id,
        free_is_digit=free_token_is_digit(free_id, token_ids),
        free_token_logprob=free_lp,
        finish_reason=getattr(completion, "finish_reason", None),
        backend="vllm_offline",
    )


def assert_raw_logprobs(engine: Any) -> str:
    """Return the engine's ``logprobs_mode`` and raise unless it is ``raw_logprobs``."""
    mode = getattr(getattr(engine, "model_config", None), "logprobs_mode", None)
    if mode is None:
        raise ReadoutError("cannot read logprobs_mode from the engine (engine.model_config.logprobs_mode)")
    if mode != EXPECTED_LOGPROBS_MODE:
        raise ReadoutError(f"engine logprobs_mode is {mode!r}, the readout needs {EXPECTED_LOGPROBS_MODE!r}")
    return str(mode)


class VLLMBackend:
    """Digit readout through an offline ``vllm.LLM`` (or any object with the same ``generate``)."""

    def __init__(
        self,
        engine: Any,
        tokenizer: Any,
        spec: ReadoutSpec,
        *,
        sampling_params: Any = None,
        use_tqdm: bool = False,
        check_logprobs_mode: bool = True,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.spec = spec
        self.token_ids = spec.token_ids(tokenizer)
        self.use_tqdm = use_tqdm
        self.logprobs_mode = assert_raw_logprobs(engine) if check_logprobs_mode else None
        self.sampling_params = sampling_params if sampling_params is not None else self.make_sampling_params()

    def make_sampling_params(self) -> Any:
        """``vllm.SamplingParams`` built from the configured readout (imports vLLM)."""
        from vllm import SamplingParams

        return SamplingParams(**self.spec.offline_kwargs(self.token_ids))

    @classmethod
    def from_model(
        cls, model_path: str, spec: ReadoutSpec, *, tokenizer: Any = None, **llm_kwargs: Any
    ) -> "VLLMBackend":
        """Start ``vllm.LLM(model_path, ...)``. ``logprobs_mode`` must not be passed."""
        if "logprobs_mode" in llm_kwargs:
            raise ValueError("logprobs_mode must stay at the vLLM default (raw_logprobs); do not pass it")
        from vllm import LLM

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)
        engine = LLM(model=model_path, **llm_kwargs)
        return cls(engine, tokenizer, spec)

    def readouts(self, prompts: Sequence[str] | Sequence[Sequence[int]]) -> list[TrialReadout]:
        """Readouts in input order. ``prompts`` are rendered chat texts or ready token id lists."""
        if len(prompts) == 0:
            return []
        if isinstance(prompts[0], str):
            id_lists = encode_prompts(self.tokenizer, prompts)  # type: ignore[arg-type]
        else:
            id_lists = [[int(t) for t in p] for p in prompts]  # type: ignore[union-attr]
        outputs = self.engine.generate(
            [{"prompt_token_ids": ids} for ids in id_lists], self.sampling_params, use_tqdm=self.use_tqdm
        )
        if len(outputs) != len(id_lists):
            raise ReadoutError(f"engine returned {len(outputs)} outputs for {len(id_lists)} prompts")
        records = []
        for sent, output in zip(id_lists, outputs):
            echoed = getattr(output, "prompt_token_ids", None)
            if echoed is not None and [int(t) for t in echoed] != sent:
                raise ReadoutError("the engine returned an output for a different prompt (order or content changed)")
            records.append(parse_request_output(output, self.token_ids, self.spec))
        return records
