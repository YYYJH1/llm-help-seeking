"""Final step of request construction for the batches supp4 and supp5.

``nmi.api_exp.client.build_request_body`` builds the body of a request from the request class of a model: the model field of that body is the ``id`` of the model, and the
``provider`` object pins the service with ``allow_fallbacks: false``. This module is the one place where that body becomes the body that is sent, with two changes and nothing else:

* Model name. A model of supp5 is queried in a cell of its own, and its ``id`` carries the cell (``openai/gpt-6-luna@a0.95``). Everything the program keeps for a model (the
  ledger, the send quotas, the run directory, the budget approvals, the prices, the release list) is keyed by that id, but the request must name the model that the service
  knows, the ``wire`` name (``openai/gpt-6-luna``).
* Price ceiling. ``provider.max_price`` (``prompt`` and ``completion``, US dollars per million tokens; ``nmi.api_exp.price_rules.PriceCeiling``) is OpenRouter's ceiling on the price
  of the call: it forwards the request only to a service whose price does not exceed it, and if the pinned service charges more, the request fails before it is forwarded. The
  reserve for a send is computed from the same two numbers (``nmi.api_exp.journal.Journal.write_attempt``), so the 500 CNY spending cap rests on a limit that the service enforces.

The module is separate so that the runner can hash it with the other batch code files (``nmi.api_exp.runner.batch_code_md5``): a change to it after the first real send is refused,
as for ``client.py``. The runner builds every request through ``request_body``.
"""

from __future__ import annotations

from typing import Any

from nmi.api_exp.client import build_request_body
from nmi.api_exp.config import ApiExpConfig, ModelSpec
from nmi.api_exp.price_rules import PriceCeiling


def request_body(cfg: ApiExpConfig, model: ModelSpec, message: str, ceiling: PriceCeiling | None = None) -> dict[str, Any]:
    """The JSON body that is sent for ``message`` to ``model``: the client's body for the id of the model, with the model field replaced by the wire name of a model that has one and,
    for a batch with a price margin, ``provider.max_price`` set to ``ceiling``. A batch with a price margin must be given its ceiling and a batch without one must not."""
    if (cfg.price_margin is None) != (ceiling is None):
        raise ValueError("the request of a batch with a price margin carries the ceiling of the price (provider.max_price), and the request of any other batch does not")
    body = build_request_body(cfg, model.id, message)
    if model.wire is not None:
        body = {**body, "model": model.wire}
    if ceiling is not None:
        provider = body.get("provider")
        if not isinstance(provider, dict):
            raise ValueError("the request class of a model names its pinned service in the provider object")
        body = {**body, "provider": {**provider, "max_price": ceiling.as_request()}}  # new dicts: the dictionaries of the request class are shared by every request
    return body
