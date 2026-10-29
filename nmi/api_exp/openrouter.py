"""Extra fields that a batch of the kind ``openrouter`` reads from a reply besides those of the first batch.

The request is built by ``nmi.api_exp.client.build_request_body`` from the request class of the model: the pinned service (``provider``), ``usage: {include: true}`` and the thinking
parameter (``reasoning``) are fields of the class in the configuration. The reply is read by ``nmi.api_exp.client.parse_chat_body`` (the content, its first character, the usage with
``reasoning_tokens``, the finish reason, the model field). This module adds three things that OpenRouter returns besides. Nothing that a server wrote is kept, except under fixed rules:

* the service that answered (the top-level field ``provider``). It is kept as it came only when it is exactly the pinned service (or a name the class lists for it); otherwise it is the fixed word
  ``(unlisted)`` with the sha256 of the field, and an absent field is ``(absent)``. A reply from another service than the pinned one, or from another model than the one asked for (the field
  ``model``, which ``nmi.api_exp.client`` keeps under the same rule), or with either field missing, is an identity failure, with the anomaly ``provider_mismatch``, ``provider_absent``,
  ``model_mismatch`` or ``model_absent``. Such a reply is booked, counts as one send, is kept and is never sent again; but it is not read as a decision, so its trial
  stays in the denominator and counts neither as an answer nor as a request for help (``apply_router_facts``);
* the price that OpenRouter reports for the call (``usage.cost``, US dollars), kept as the text of the number in a field of its own, ``reported_cost_usd``. The money booked for the reply is the
  larger of the tokens at the price file and this price times ``USD_TO_CNY`` (``reported_cny``, ``nmi.api_exp.journal.Journal.write_result_success``), so that a call that cost more than the list
  price is booked at what it cost; a reply that reports no price is booked from the tokens, as in every batch;
* whether the reply carried thinking text (``message.reasoning`` or ``message.reasoning_details``): only that it was there, joined with the field ``reasoning_present`` that the first batch's
  parser sets from ``reasoning_content``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from nmi.api_exp.client import MODEL_FIELD_ABSENT, MODEL_FIELD_UNLISTED
from nmi.api_exp.reader import FORMAT_FAILURE

PROVIDER_UNLISTED = "(unlisted)"  # what is stored (with the sha256 of the field) when the provider of a reply is not the pinned one
PROVIDER_ABSENT = "(absent)"
ANOMALY_PROVIDER_MISMATCH = "provider_mismatch"
ANOMALY_PROVIDER_ABSENT = "provider_absent"
ANOMALY_MODEL_MISMATCH = "model_mismatch"
ANOMALY_MODEL_ABSENT = "model_absent"
USD_TO_CNY = Decimal("7.3")  # the rate of the price file: US dollars to CNY, with no markup
COST_MAX_USD = Decimal(1000)  # a reported price above this (for ONE call of 4 to 256 output tokens) is not a price
COST_TEXT_MAX = 40  # characters of the text of a reported price that is kept


@dataclass(frozen=True)
class RouterFacts:
    """What is kept of a reply of OpenRouter besides the first batch's fields."""

    returned_provider: str  # the pinned service when the reply named it, else a fixed word
    returned_provider_sha256: str | None  # the sha256 of the field when it was not stored as it came
    anomaly: str | None  # provider_mismatch or provider_absent
    reported_cost_usd: str | None  # the text of usage.cost when it is a number from 0 to COST_MAX_USD with a text of at most COST_TEXT_MAX characters
    reasoning_text_present: bool

    def fields(self) -> dict[str, Any]:
        return {"returned_provider": self.returned_provider, "returned_provider_sha256": self.returned_provider_sha256, "reported_cost_usd": self.reported_cost_usd}


def safe_provider_field(value: Any, allowed: Collection[str]) -> tuple[str, str | None]:
    """The provider of a reply as it may be stored, and the sha256 to keep beside it: unchanged only when it is exactly the pinned service or one of the names the class lists for it
    (``allowed``), else a fixed word and the sha256 of the value."""
    if value is None:
        return PROVIDER_ABSENT, None
    if isinstance(value, str) and value in allowed:
        return value, None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, sort_keys=True, ensure_ascii=True)
        except (TypeError, ValueError):
            text = repr(type(value))
    return PROVIDER_UNLISTED, hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def provider_key(payload: Mapping[str, Any]) -> str:
    """How a stored provider is counted in a report: the name itself, or the fixed word with the first 12 hex digits of its sha256 (as ``client.returned_model_key`` does for the model)."""
    name, sha = str(payload.get("returned_provider")), payload.get("returned_provider_sha256")
    return f"{name}:{str(sha)[:12]}" if sha else name


def reported_cost(value: Any) -> str | None:
    """The text of the price that a reply reports, or None. The reply is decoded without a float (``nmi.api_exp.client``): the price is an ``int`` or a ``Decimal`` (a float, which only another caller can hand in, is read at its shortest text). It is kept when it is a
    finite number from 0 to ``COST_MAX_USD`` whose plain text is short; anything else (a string, a boolean, null, a negative or huge number, 1e-999) is not a price."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, Decimal):
        number = value
    elif isinstance(value, float) and math.isfinite(value):  # (the decoder makes none; another caller's float is taken at its shortest text)
        number = Decimal(repr(value))
    else:
        return None
    if not number.is_finite() or number < 0 or number > COST_MAX_USD:
        return None
    if number == 0:
        return "0"
    text = format(number, "f")
    return text if len(text) <= COST_TEXT_MAX else None


def _reasoning_text_present(message: Any) -> bool:
    if not isinstance(message, Mapping):
        return False
    text = message.get("reasoning")
    details = message.get("reasoning_details")
    return (isinstance(text, str) and bool(text.strip())) or (isinstance(details, list) and bool(details))


def read_router_facts(body: Mapping[str, Any], pinned: str | Collection[str] | None) -> RouterFacts:
    """The facts of a reply body of OpenRouter that the batch keeps. ``pinned``: the pinned service, or the names a reply may carry for it. Missing or oddly typed fields read as absent;
    nothing here raises on a reply."""
    allowed = () if pinned is None else ((pinned,) if isinstance(pinned, str) else pinned)
    returned, sha = safe_provider_field(body.get("provider"), allowed)
    anomaly = ANOMALY_PROVIDER_ABSENT if returned == PROVIDER_ABSENT else (ANOMALY_PROVIDER_MISMATCH if returned == PROVIDER_UNLISTED else None)
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else None
    message = first.get("message") if first is not None else None
    return RouterFacts(returned, sha, anomaly, reported_cost(usage.get("cost")), _reasoning_text_present(message))


def reported_cny(reported_cost_usd: str | None) -> float | None:
    """The price a reply reports, in CNY (the text of ``usage.cost`` in US dollars times ``USD_TO_CNY``), or None when the reply reported none (or none that ``reported_cost`` keeps)."""
    return None if reported_cost_usd is None else float(Decimal(reported_cost_usd) * USD_TO_CNY)


def identity_anomaly(payload: Mapping[str, Any], facts: RouterFacts) -> str | None:
    """Why a reply is not from the service and the model that were asked for, or None: the provider first (``facts.anomaly``), then the model field that ``nmi.api_exp.client`` has stored in
    the payload (the id or an alias of the model as it came, else a fixed word)."""
    if facts.anomaly is not None:
        return facts.anomaly
    returned_model = payload.get("returned_model")
    if returned_model == MODEL_FIELD_ABSENT:
        return ANOMALY_MODEL_ABSENT
    if returned_model == MODEL_FIELD_UNLISTED:
        return ANOMALY_MODEL_MISMATCH
    return None


def apply_router_facts(payload: dict[str, Any], facts: RouterFacts) -> str | None:
    """Add the facts to the reply data of a success (``nmi.api_exp.runner.reply_payload``): the fields of their own, the thinking flag joined, and the anomaly if the reply has none yet.

    A reply that is not from the pinned service or not from the requested model, or that lacks either field (an identity failure), is not read as a decision: whatever it says, its decision
    is the format failure and it has no first character, so the trial is in the denominator and is neither an answer nor a request for help. It is booked and counted as a send, and it is
    never sent again. Returns the identity anomaly, or None."""
    payload.update(facts.fields())
    payload["reasoning_present"] = bool(payload.get("reasoning_present")) or facts.reasoning_text_present
    identity = identity_anomaly(payload, facts)
    if payload.get("anomaly") is None:
        payload["anomaly"] = identity
    if identity is not None:
        payload["decision"] = FORMAT_FAILURE
        payload["first_char"] = None
    return identity
