"""The HTTP client of the API experiment.

One request is one chat-completions call with one user message. The body holds the model, the message, ``temperature`` 0, ``max_tokens`` of the model's class and, for the class that can
switch thinking off, ``enable_thinking: false`` at the top level (what the OpenAI SDK sends for ``extra_body``). It holds no system message, no assistant prefix, no ``logprobs``.

``ChatClient.send_once`` makes ONE send and classifies what came back. It does not retry: how often a request is sent, in all, over every session, and when, is the business of the runner,
which writes an attempt record to disk before each send and a result record after it (``nmi.api_exp.journal``). The classes (all fixed names; nothing a server wrote is kept):

* ``success``: a 2xx reply whose body is a JSON object. Its content may be unusable (a format failure); that is the reply, and it is never sent again;
* ``http_429``, ``http_5xx``: sent again (rate limit, server error);
* ``transport_error``: no usable response (a connection or timeout error, a body that cannot be decoded), and ``invalid_body``: a 2xx reply that is not a JSON object (a damaged
  transfer). Both are sent again, and the service may have processed the request, so their billing is unknown; so is that of ``http_5xx``;
* ``http_4xx`` (any 4xx but 429 and the abort statuses) and ``http_other`` (anything else that is not 2xx): the end of the request, never sent again;
* ``auth_refused``: a status in ``abort_statuses`` (401, 403). The key or the account is refused: that is a fact about the run, so it stops the run and does not end the request.

Nothing a server returned is stored, except what ``parse_chat_body`` extracts under fixed rules: the decision read from the content, its first character only when that is a
digit 1 to 5, the length and the sha256 of the content, the token usage (the numbers of its two booked fields, exactly as they were written), a finish reason from a fixed list, and the model name only if it is exactly the id of the requested model or one
of its configured aliases (otherwise a fixed word and the sha256 of what came). Error bodies are not read at all. The key is held in the client object only.
It is not in ``repr`` and it is never written anywhere.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Self

import httpx

from nmi.api_exp.config import ApiExpConfig, RetryPolicy
from nmi.api_exp.reader import read_reply

SUCCESS = "success"
HTTP_429 = "http_429"
HTTP_5XX = "http_5xx"
HTTP_4XX = "http_4xx"
HTTP_OTHER = "http_other"
AUTH_REFUSED = "auth_refused"
TRANSPORT_ERROR = "transport_error"
INVALID_BODY = "invalid_body"
FAILURE_CLASSES = (HTTP_429, HTTP_5XX, HTTP_4XX, HTTP_OTHER, AUTH_REFUSED, TRANSPORT_ERROR, INVALID_BODY)
ENDS_REQUEST = frozenset({HTTP_4XX, HTTP_OTHER})  # a 4xx that is not rate limiting (and anything else that is not 2xx): never sent again
UNKNOWN_BILLING = frozenset({TRANSPORT_ERROR, INVALID_BODY, HTTP_5XX})  # the service may have processed the request: booked at the upper bound of the send

MODEL_FIELD_UNLISTED = "(unlisted)"  # what is stored (with the sha256 of the field) when the model field of a response is not the requested id or one of its aliases
MODEL_FIELD_ABSENT = "(absent)"
FINISH_CLASSES = ("stop", "length", "content_filter", "tool_calls", "function_call")  # a finish reason outside this list is stored as "other"
HUGE_DIGITS = 400  # an integer in a reply that is written with more digits than this is read as 10^400 (see ``read_count_literal``)
HUGE_COUNT = 10**HUGE_DIGITS
DECIMAL_EXPONENT_LIMIT = 999999999999999999  # the decimal type refuses an exponent with more digits than this (see ``read_float_literal``)
# the names of httpx exceptions that may be stored for a transport error; the first one on the exception's class chain is used
_TRANSPORT_NAMES = (
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "ReadError", "WriteError", "CloseError", "RemoteProtocolError", "LocalProtocolError", "ProxyError",
    "UnsupportedProtocol", "DecodingError", "NetworkError", "TimeoutException", "ProtocolError", "TransportError",
)


@dataclass(frozen=True)
class SendResult:
    cls: str  # success or one of FAILURE_CLASSES
    status: int | None  # HTTP status of the response (None after a transport error)
    exc: str | None  # for transport_error: the name of the httpx exception class (a fixed name), never its message
    body: dict[str, Any] | None  # the decoded body of a successful send
    latency_s: float


@dataclass(frozen=True)
class ParsedReply:
    """What is kept of a successful reply. The content itself is not in this object."""

    decision: str  # answer, ask or format_failure
    first_char: str | None  # the first character of the stripped content, only when it is a digit 1 to 5
    reply_chars: int  # the length of the content
    reply_sha256: str  # the sha256 of the content (UTF-8)
    finish: str | None  # a finish reason from FINISH_CLASSES, "other", or None
    returned_model: str  # the model field of the response if it is the requested id or one of its aliases, else a fixed word
    returned_model_sha256: str | None  # the sha256 of the model field when it was not stored as it came
    prompt_tokens: int | Decimal | None  # the number the field held, EXACTLY as it was written (an int, or a Decimal: a decimal, infinity or NaN); None when it held no number
    completion_tokens: int | Decimal | None
    total_tokens: int | None
    reasoning_tokens: int | None
    reasoning_present: bool  # the reply carried reasoning text (never stored: only that it was there)
    anomaly: str | None  # ``no_choices`` or ``no_message`` when the body has no usable message

    def usage(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens, "reasoning_tokens": self.reasoning_tokens,
        }

    @property
    def has_usage(self) -> bool:
        """Both fields held a number. The booking does not need this: each is read against the bound of its own send (``nmi.api_exp.journal.read_count``)."""
        return self.prompt_tokens is not None and self.completion_tokens is not None


def build_request_body(cfg: ApiExpConfig, model_id: str, message: str) -> dict[str, Any]:
    """The JSON body of one request."""
    cls = cfg.class_of(model_id)
    body: dict[str, Any] = {
        "model": model_id, "messages": [{"role": "user", "content": message}], "temperature": cfg.temperature, "max_tokens": cls.max_tokens,
    }
    body.update(cls.extra_body)
    return body


def reported_number(value: Any) -> int | Decimal | None:
    """The number that a usage field of a reply holds, EXACTLY as it was written, or None when it holds none (a string, a boolean, null, a list, a dictionary, an absent field).

    The reply is decoded without a float anywhere (``send_once``): an integer is an ``int``, a decimal is a ``Decimal`` with the digits it was written with (``4.0000000000000001``, ``1e-999``), and NaN and
    the infinities are the non-finite ``Decimal`` values. The number is handed on as it is, negative or not, finite or not. Whether it is a count, and whether it is above the bound of its send, is
    decided in ONE place, on the exact value, where the money is booked and the ledger replayed (``nmi.api_exp.journal.read_count``). A ``float`` handed in by another caller is turned into the
    ``Decimal`` of its exact binary value, which changes nothing in it."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, Decimal)):
        return value
    return Decimal(value) if isinstance(value, float) else None


def valid_token_count(value: Any) -> int | None:
    """A non-negative integer, of any size, else None. The fields of a usage block that are only stored and decide nothing (``total_tokens`` and ``reasoning_tokens``) are read with it; the two
    fields that are booked (``prompt_tokens`` and ``completion_tokens``) are read by ``reported_number`` and ``nmi.api_exp.journal.read_count``."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def read_count_literal(text: str) -> int:
    """``parse_int`` of the JSON reader of a reply. An integer written with more than ``HUGE_DIGITS`` digits is read as plus or minus 10^400: Python refuses to turn a literal of more than
    4,300 digits into an integer, and then the whole reply would be read as a damaged transfer and no count in it would be seen. 10^400 is above every bound, and above every amount a float can
    hold, so a count that large is a bound violation and is booked and locked out like any other (it is stored as 10^400, which is all that is kept of it)."""
    negative = text.startswith("-")
    if len(text) - negative <= HUGE_DIGITS:
        return int(text)
    return -HUGE_COUNT if negative else HUGE_COUNT


def read_float_literal(text: str) -> Decimal:
    """``parse_float`` of the JSON reader of a reply: a decimal is read as the exact ``Decimal`` of its digits, never as a float (a float rounds ``4.0000000000000001`` to 4.0 and makes ``1e-999``
    zero, and then what the number was is lost before anyone has asked whether it is a count). The decimal type refuses an exponent of more than 18 digits; such a number is
    read as a number with the same sign, the same kind (zero, or nonzero) and the largest or smallest exponent the type holds, which is as far above every bound, or as far above zero, as the
    number was."""
    try:
        return Decimal(text)
    except InvalidOperation:
        pass
    mantissa, _, exponent = text.lower().partition("e")
    value = Decimal(mantissa)
    if value.is_zero():
        return value
    return Decimal((value.is_signed(), (1,), -DECIMAL_EXPONENT_LIMIT if exponent.startswith("-") else DECIMAL_EXPONENT_LIMIT))


_count = valid_token_count


def safe_model_field(value: Any, allowed: Collection[str]) -> tuple[str, str | None]:
    """The model name of a response as it may be stored, and the sha256 to keep beside it. Unchanged only if it is exactly one of the ``allowed`` names (the id of the requested model and
    its aliases from the configuration); anything else, whatever it looks like, becomes a fixed word and the sha256 of the value."""
    if value is None:
        return MODEL_FIELD_ABSENT, None
    if isinstance(value, str) and value in allowed:
        return value, None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, sort_keys=True, ensure_ascii=True)
        except (TypeError, ValueError):
            text = repr(type(value))
    return MODEL_FIELD_UNLISTED, hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def returned_model_key(payload: Mapping[str, Any]) -> str:
    """How a stored returned-model field is counted in a report: the name itself, or, for a name that was not listed, the fixed word with the first 12 hex digits of its sha256."""
    name, sha = str(payload.get("returned_model")), payload.get("returned_model_sha256")
    return f"{name}:{str(sha)[:12]}" if sha else name


def _finish_class(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value in FINISH_CLASSES else "other"


def parse_chat_body(body: Mapping[str, Any], allowed_models: Collection[str]) -> ParsedReply:
    """The fields of a chat-completions body that the experiment keeps. Missing or oddly typed fields read as absent; nothing here raises on a reply."""
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    message = first.get("message") if first is not None else None
    anomaly = None
    if first is None:
        anomaly = "no_choices"
    elif not isinstance(message, dict):
        anomaly = "no_message"
    content, reasoning_present = "", False
    if isinstance(message, dict):
        raw = message.get("content")
        content = raw if isinstance(raw, str) else ""
        reasoning = message.get("reasoning_content")
        reasoning_present = isinstance(reasoning, str) and bool(reasoning.strip())
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    reading = read_reply(content)
    returned_model, returned_sha = safe_model_field(body.get("model"), allowed_models)
    return ParsedReply(
        decision=reading.decision, first_char=reading.first_char if reading.valid else None, reply_chars=len(content),
        reply_sha256=hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest(), finish=_finish_class(first.get("finish_reason")) if first is not None else None,
        returned_model=returned_model, returned_model_sha256=returned_sha, prompt_tokens=reported_number(usage.get("prompt_tokens")), completion_tokens=reported_number(usage.get("completion_tokens")),
        total_tokens=_count(usage.get("total_tokens")), reasoning_tokens=_count(details.get("reasoning_tokens")), reasoning_present=reasoning_present, anomaly=anomaly,
    )


def backoff_delay(policy: RetryPolicy, retry_number: int, rng: random.Random | None = None) -> float:
    """Seconds to wait before retry number ``retry_number`` (1 for the first retry): base * factor^(n-1), capped, then jittered by up to +-``jitter``."""
    delay = min(policy.backoff_max_s, policy.backoff_base_s * policy.backoff_factor ** (retry_number - 1))
    if policy.jitter > 0 and rng is not None:
        delay *= 1.0 + policy.jitter * (2.0 * rng.random() - 1.0)
    return max(0.0, delay)


def exception_name(exc: BaseException) -> str:
    """A fixed name for a transport exception (never its message, which can carry a URL or a proxy text): the first class on its chain that is an httpx exception we know."""
    for klass in type(exc).__mro__:
        if klass.__name__ in _TRANSPORT_NAMES:
            return klass.__name__
    return "TransportError"


class ChatClient:
    """Async client; the caller keeps at most ``concurrency`` sends in flight, and the connection pool is sized to match."""

    def __init__(self, url: str, api_key: str, retry: RetryPolicy, *, concurrency: int = 8, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("the API key is empty")
        self._url = url
        self._key = api_key.strip()
        self._abort = frozenset(retry.abort_statuses)
        limits = httpx.Limits(max_connections=max(1, concurrency), max_keepalive_connections=max(1, concurrency))
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(retry.timeout_s), limits=limits, transport=transport,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )

    def __repr__(self) -> str:
        return f"ChatClient(url={self._url!r}, key=<hidden>)"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def send_once(self, body: Mapping[str, Any]) -> SendResult:
        """One send, classified. Never raises for anything the service or the network does."""
        started = time.monotonic()
        try:
            response = await self._client.post(self._url, json=body)
        except (httpx.TransportError, httpx.DecodingError) as exc:  # a body that cannot be decoded is a damaged transfer too
            return SendResult(TRANSPORT_ERROR, None, exception_name(exc), None, time.monotonic() - started)
        status = response.status_code
        elapsed = time.monotonic() - started
        if 200 <= status < 300:
            try:
                decoded = response.json(parse_int=read_count_literal, parse_float=read_float_literal, parse_constant=Decimal)  # (no float is made: see ``reported_number``)
            except (ValueError, RecursionError):  # (a body of nothing but opening brackets exhausts the depth of the reader: a damaged transfer, not a crash)
                decoded = None
            if isinstance(decoded, dict):
                return SendResult(SUCCESS, status, None, decoded, elapsed)
            return SendResult(INVALID_BODY, status, None, None, elapsed)  # a damaged transfer, not a reply
        if status == 429:
            return SendResult(HTTP_429, status, None, None, elapsed)
        if 500 <= status <= 599:
            return SendResult(HTTP_5XX, status, None, None, elapsed)
        if status in self._abort:
            return SendResult(AUTH_REFUSED, status, None, None, elapsed)
        return SendResult(HTTP_4XX if 400 <= status <= 499 else HTTP_OTHER, status, None, None, elapsed)
