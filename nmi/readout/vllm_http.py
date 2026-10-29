"""HTTP (OpenAI-compatible ``/v1/completions``) readout against a vLLM server.

Request: ``max_tokens=1``, ``temperature=0``, and BOTH ``logprobs: 5`` and ``logprob_token_ids`` (the five digit
ids); the server rejects ``logprob_token_ids`` without ``logprobs``
(``vllm/entrypoints/openai/completion/protocol.py``, ``check_logprobs``). ``add_special_tokens`` is sent as false
because the rendered chat text already contains every special token (a second BOS would change the input).

Response (``CompletionResponse`` in the same file): ``choices[0].logprobs`` is a ``CompletionLogProbs`` with
``tokens``, ``token_logprobs`` and ``top_logprobs``; ``top_logprobs[0]`` maps the DECODED TOKEN STRING to its
logprob and holds the five requested ids plus the sampled token (the server builds it in
``completion/serving.py::_create_completion_logprobs``). ``-inf`` is reported as ``-9999.0`` (the server's JSON-safe
clamp); the parser turns every value at or below ``-9999.0`` back into ``-inf``. With
``return_tokens_as_token_ids`` the keys are ``token_id:<id>``, which the parser also accepts. The free token's
id is read from ``choices[0].token_ids`` when the request set ``return_token_ids``; otherwise it is known only
when the free token is one of the digits (its string is the digit).

``HTTPBackend`` is a small synchronous client for batch readouts (cross-checks, calibration runs). The
real-time loop client is separate. The server must run with the default
``--logprobs-mode raw_logprobs``; nothing here can verify that from the client side except the sanity
checks of the parser (no positive logprob, digit mass not above one).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from nmi.readout.digits import ReadoutSpec, digit_mass
from nmi.readout.hf_backend import ReadoutError, TrialReadout, encode_prompts

SERVER_NEG_INF = -9999.0  # vLLM clamps -inf to this value in JSON responses
_POSITIVE_TOLERANCE = 1e-4
_MASS_TOLERANCE = 1e-3
_TOKEN_ID_RE = re.compile(r"^token_id:(\d+)$")


class HttpParseError(ReadoutError):
    """A completion response does not have the expected content."""


def completion_request_body(
    model: str,
    prompt: str | Sequence[int],
    spec: ReadoutSpec,
    token_ids: Sequence[int],
    *,
    add_special_tokens: bool = False,
    return_token_ids: bool = False,
) -> dict[str, Any]:
    """JSON body of the readout request (``spec.http_fields`` carries the pre-specified fields)."""
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt if isinstance(prompt, str) else [int(t) for t in prompt],
        **spec.http_fields(token_ids),
        "add_special_tokens": add_special_tokens,
    }
    if return_token_ids:
        body["return_token_ids"] = True
    return body


def _logprob(value: Any, what: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as error:
        raise HttpParseError(f"{what}: logprob {value!r} is not a number") from error
    if math.isnan(x) or x == math.inf:
        raise HttpParseError(f"{what}: logprob is {x}")
    if x <= SERVER_NEG_INF:
        return -math.inf
    if x > _POSITIVE_TOLERANCE:
        raise HttpParseError(
            f"{what}: logprob {x} is positive; the server is not returning raw log-probabilities "
            "(it must run with --logprobs-mode raw_logprobs)"
        )
    return x


def parse_completion_response(
    payload: Mapping[str, Any], token_ids: Sequence[int], spec: ReadoutSpec
) -> TrialReadout:
    """Readout of one decoded ``/v1/completions`` JSON response."""
    try:
        choice = payload["choices"][0]
        logprobs = choice["logprobs"]
        top = logprobs["top_logprobs"][0]
        tokens = logprobs["tokens"]
        token_logprobs = logprobs["token_logprobs"]
    except (KeyError, IndexError, TypeError) as error:
        raise HttpParseError(f"response lacks choices[0].logprobs.top_logprobs[0] ({error!r})") from error
    if not isinstance(top, Mapping) or not top:
        raise HttpParseError("top_logprobs[0] is empty or not a mapping")
    values: list[float] = []
    for digit, tid in zip(spec.digits, token_ids):
        key = digit if digit in top else f"token_id:{int(tid)}" if f"token_id:{int(tid)}" in top else None
        if key is None:
            raise HttpParseError(
                f"digit {digit!r} (token id {tid}) is missing from top_logprobs[0]; keys present: {sorted(top)}"
            )
        values.append(_logprob(top[key], f"digit {digit!r}"))
    if digit_mass(values) > 1.0 + _MASS_TOLERANCE:
        raise HttpParseError(f"digit probabilities sum to {digit_mass(values):.4f} > 1: not raw log-probabilities")
    free_string = tokens[0] if tokens else None
    free_lp = None
    if token_logprobs and token_logprobs[0] is not None:
        free_lp = _logprob(token_logprobs[0], "free token")
    free_id: int | None = None
    ids = choice.get("token_ids")
    if ids:
        free_id = int(ids[0])
    elif isinstance(free_string, str):
        if free_string in spec.digits:
            free_id = int(token_ids[spec.digits.index(free_string)])
        else:
            match = _TOKEN_ID_RE.match(free_string)
            if match:
                free_id = int(match.group(1))
    digit_ids = {int(t) for t in token_ids}
    if free_id is not None:
        free_is_digit = free_id in digit_ids
    else:
        free_is_digit = isinstance(free_string, str) and free_string in spec.digits
    return TrialReadout(
        readout=spec.readout(values),
        free_token_id=free_id,
        free_is_digit=free_is_digit,
        free_token_logprob=free_lp,
        finish_reason=choice.get("finish_reason"),
        backend="vllm_http",
    )


class HTTPBackend:
    """Synchronous batch client: one request per prompt, a thread pool for concurrency."""

    def __init__(
        self,
        base_url: str,
        model: str,
        tokenizer: Any,
        spec: ReadoutSpec,
        *,
        client: Any = None,
        timeout: float = 120.0,
        max_workers: int = 16,
        add_special_tokens: bool = False,
        return_token_ids: bool = False,
    ) -> None:
        import httpx

        self.model = model
        self.spec = spec
        self.tokenizer = tokenizer
        self.token_ids = spec.token_ids(tokenizer)
        self.max_workers = max_workers
        self.add_special_tokens = add_special_tokens
        self.return_token_ids = return_token_ids
        # trust_env stays at its default: httpx takes its network settings from the environment.
        self.client = client or httpx.Client(
            base_url=base_url, timeout=timeout,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _one(self, prompt: str | Sequence[int]) -> TrialReadout:
        body = completion_request_body(
            self.model, prompt, self.spec, self.token_ids,
            add_special_tokens=self.add_special_tokens, return_token_ids=self.return_token_ids,
        )
        response = self.client.post("/v1/completions", json=body)
        if response.status_code != 200:
            raise ReadoutError(f"HTTP {response.status_code}: {response.text[:300]}")
        return parse_completion_response(response.json(), self.token_ids, self.spec)

    def readouts(self, prompts: Sequence[str] | Sequence[Sequence[int]]) -> list[TrialReadout]:
        """Readouts in input order (a failed request raises; it is never read as "did not ask")."""
        if len(prompts) == 0:
            return []
        if isinstance(prompts[0], str):
            texts: list[Any] = list(encode_prompts(self.tokenizer, prompts))  # type: ignore[arg-type]
        else:
            texts = [list(p) for p in prompts]  # type: ignore[union-attr]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._one, texts))
