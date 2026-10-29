"""The closed-model API experiment: the pilot on the development pool (configuration configs/api_exp_v1.yaml, prices configs/api_exp_prices.yaml).

usage:
  python scripts/run_api_decisions.py --mode pilot --stage probe --model MODEL --tag TAG --expect-config-md5 MD5 --expect-protocol-md5 MD5
  python scripts/run_api_decisions.py --mode pilot --stage pilot --model MODEL --resume RUN_DIR --expect-config-md5 MD5 --expect-protocol-md5 MD5
  (``--config`` selects the configuration file of a batch; ``--dry-run`` on any call prints the plan and the gates and sends nothing)

Closed models behind the Bailian OpenAI-compatible endpoint receive the user message of the local ask prompt (the same ``PromptBook.ask_prompt`` call; the rule section a00, the question, the four
options, the condition block B1_f2, the last line ``Answer:``; no system message, no assistant prefix). Only the first character of the reply (1 to 5) is read. Nothing a server wrote is kept.

Two stages on the development pool ``mmlu_redux_cal`` at a_H 0.80, c0 0 and the six waiting costs, one run directory per model, both stages in that directory:

* ``--mode pilot --stage probe``: the first 100 questions of the pool, 600 requests. A new run directory needs ``--tag``.
* ``--mode pilot --stage pilot``: all 1,000 questions, 6,000 requests, continuing the directory of the probe (``--resume RUN_DIR``). It starts only when the probe of every named model has ended
  completely; otherwise it exits 2. It does not wait for verified unit prices: the money is booked at the prices of ``configs/api_exp_prices.yaml`` as they are (``verified: false``), and the cap
  judges the booked money. Requests that have a success are not sent again.

``--mode full`` (the 2,291 questions of the first batch) is REFUSED, whatever else is given, with exit code 2: reading that pool needs a release marker that is not part of this repository, and this
program has no path that reads the pool.

SUPPLEMENTARY BATCH: ``--config`` with a supplementary-batch file (not distributed) runs the models added after the first batch. The file repeats every setting of ``configs/api_exp_v1.yaml`` (the loader checks that item by
item) and lists other models, which run after the first batch's in the budget order. The two batches share the ledger (the cap of 500 CNY is judged on the money booked for both), the index of
successes, the registry of run directories, the budget state, the price file, the release list and the process lock. The freeze record of the supplementary batch is ``freeze_supp1.json``, written at its
own first real send; ``freeze.json`` of the first batch is only read (the configuration the supplementary file repeats must be the one the first batch was frozen with, and its prompt, split, messages and
request and reading code must be the ones the supplementary batch is run with) and no batch writes the other's record. A model of the other batch is refused as an unknown model.

OPENROUTER BATCHES: The supplementary batches supp3, supp4 and supp5 (not distributed) are batches of the endpoint kind ``openrouter``: they define their own ``endpoint`` and ``request`` blocks
(``batch.own``) and repeat the first batch's other blocks. Each request class pins one service per model (``provider.order`` with ``allow_fallbacks: false``), asks OpenRouter for its own price
(``usage.include``) and carries the thinking parameter of its group (see the configuration files). The PROVABLE bounds of a send are declared in the class (the output cap plus 64 tokens; a wider input
bound for one model) and decide the check before each send, the estimate and the lockout. A reply that is not from the pinned service or not from the requested model, or lacks either field, is an
IDENTITY FAILURE (``provider_mismatch``, ``provider_absent``, ``model_mismatch``, ``model_absent``): it is booked, counts as a send, is kept and is not sent again, but it is not read as a decision (a
format failure in the denominator), and a run of them (``retry.stop_after_consecutive_exhausted``) stops the model for the session. The price OpenRouter reports in US dollars is recorded
(``reported_cost_usd``); the money booked for the reply is the larger of the tokens at the price file and that price times 7.3. The md5 of the code that does this (``nmi/api_exp/openrouter.py``) is in the
freeze record of the batch (``batch_code_md5``). The batches supp4 and supp5 reserve the cost of the bound of every send times a price margin of 2.0, lock the experiment out when a reply is booked for more
than its reserve, read the reported price with ``nmi/api_exp/price_rules.py`` and finish the request with ``nmi/api_exp/final_request.py`` (both files are in their ``batch_code_md5``).

THE MONEY THAT IS THERE. A model whose estimate does not fit in the remaining cap of 500 CNY is refused WHOLE when it is to start (``not started (budget)``: nothing is sent to it, the log and the
dry run give the booked money, the estimate and the shortfall, and the models after it in the run order are stopped for that stage). The cap is the only limit the program knows of its own:
the balance of an OpenRouter account it does not know. ``--available-cny X`` tells it (the balance in CNY, at 7.3 to the dollar): a model whose estimate does not fit in what is left of X in this
session (``available_money``) is not started either, its shortfall is reported, and the session STOPS there; nothing is written to the budget state for it. For a
batch of the kind openrouter the status 402 (payment required) stops the run like a refused key, and the requests refused stay to be sent again after the account has been topped up. The models of a
call run in the order of the configuration, whatever order they are named in: to run them in another order (the cheapest first by measured price), call them one at a time.

Where things go is not an option: the runs are under ``results/api_exp/`` of the repository at ``state.code_root``, and the ledger, the index of successes, the registry of run directories, the budget
state, the freeze record and the process lock under its ``_state/``, whichever working tree this code is in. A REAL run is refused (exit 2) unless this code is that repository; a copy elsewhere runs the
tests and ``--dry-run``. A model that already has a run directory cannot be given another tag: its directory is continued with
``--resume``. The experiment is frozen just before its first real send (protocol, configuration, prompt, split, messages, request and reading code); every later session compares itself with
the record and ends with exit 2 if anything differs, so another configuration file does not start a model.

Every send is written ahead: an attempt record is on disk before the request leaves and a result record after the answer. An attempt without a result is UNCERTAIN: it counts as booked at its
upper bound and an ordinary resume does not send that request again; ``--resend-uncertain`` sends it once more (a request has at most 8 sends in all sessions; after the eighth, an uncertain
send is final). Beside that, a (stage, model) makes at most its planned keys times 1.1 sends in all sessions, rounded up (600 keys: 660 sends; 120 keys: 132); at that number nothing more is
sent to the model in that stage and the stage does not count as completely ended. The cap of 500 CNY is hard. Before every send, the CNY booked plus the PROVABLE bound of that send (the UTF-8 bytes of the message plus a template margin as input tokens, the
output cap as output tokens) must be within the cap; a model starts only if the CNY booked plus its estimate (remaining requests, at the bound or at the largest measured tokens per byte times
1.25, times 1.1) is. Money is booked at the price of the day and never worked out again.

``--dry-run`` renders every message, counts the requests, reserves, applies the gates and the budget to the models in order and prints the report. It sends nothing, reads no key and writes no
file. A real run needs ``--tag`` for a new directory (or ``--resume``) and the md5 of the configuration and of the protocol that are in force (``--expect-config-md5``,
``--expect-protocol-md5``): no question is sent before the protocol takes effect, so the call states which version it believes is in force. The key is read from ``endpoint.key_file`` (or
``--key-file``) inside the process, once, and is never printed or written.

Only the models of the release list start (``release.json`` in the state directory; an absent file releases none): a real run that names another model ends with exit 2 before the
key is read, and ``--model all`` stands for the released models and no others. A reply whose usage has a count above the provable bound of its send stops the run and locks the experiment:
every later real start and resume ends with exit 2 before the key is read, until a ``bound_disposition.json`` in the state directory closes the lockout records
(``nmi.api_exp.lockout``). ``--dry-run`` shows the release list, the lockout, the quota of sends of each model and the sends it has used. The free text in those two
files (the basis of a release, the reason of a disposition) is checked to be a non-empty string and then used only as a sha256 and a length: it is in no report, message, log, manifest
or ledger record, and a field that is not as it should be is named by its place, never by its value. A token count in a reply is a non-negative integer of any size.

Models run in the order of the configuration. The first Ctrl-C (or SIGTERM) stops the sending cleanly (sends in flight finish, the manifest says why, the run can be resumed); a second one acts as
it would without this handler.

Exit codes: 0 every requested model is complete; 1 the run ended with requests missing or a model not run for budget; 2 refused or unusable input (nothing was sent); 3 the run was stopped (a
refused key or account, an interrupt, the cap, ``--max-session-cny``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import httpx

from nmi import paths
from nmi.api_exp.config import (
    STAGE_NAMES,
    ApiConfigError,
    ApiExpConfig,
    ModelSpec,
    load_api_config,
)
from nmi.api_exp.journal import Journal, JournalError
from nmi.api_exp.ledger import message_bytes, reserve_of, send_quota
from nmi.api_exp.lockout import Lockout, assess
from nmi.api_exp.price_rules import price_ceiling
from nmi.api_exp.prices import PriceBook, PriceError, load_price_book
from nmi.api_exp.release import RELEASE_NAME, Release, load_release
from nmi.api_exp.render import (
    RenderError,
    RequestPlan,
    conditions_of,
    keys_md5,
    load_pilot_items,
    load_prompt_book,
)
from nmi.api_exp.runner import (
    EXIT_ABORTED,
    EXIT_REFUSED,
    FREEZE_KEYS,
    SHOWN_STATES,
    ApiRun,
    RunRefused,
    RunSettings,
    batch_code_md5,
    code_md5,
    count_states,
    describe_exception,
    estimate_completion,
    first_batch_problem,
    freeze_refusal,
    gate_problems,
    plan_keys,
    read_api_key,
    rel,
)
from nmi.api_exp.state import (
    BudgetState,
    Freeze,
    RunRegistry,
    StateError,
    api_root,
    code_root_problem,
    freeze_file_name,
    state_dir,
)
from nmi.config import ConfigError, load_config
from nmi.data.registry import McItem, SplitError, SplitRegistry
from nmi.prompts import DisplayRule, PromptError
from nmi.provenance import file_md5, note_skipped

DEFAULT_CONFIG = paths.CONFIG_DIR / "api_exp_v1.yaml"
FULL_MODE_REFUSAL = (
    "--mode full is refused (exit 2). The full run reads the 2,291 questions of the first batch (mmlu_redux_holdout). Reading that pool needs a release marker that is not part of this repository, "
    "and this program has no path that reads the pool. "
    "Nothing was read and nothing was sent. Use --mode pilot for the development pool."
)


@dataclass
class ApiHooks:
    """The few things a test replaces: the pool loader (the real one opens the question bank), the transport (no network), the key reader, the wait between two sends, the results root
    (the one way to put the runs and the state somewhere else: there is no option and no environment variable for it), the repository root the code pretends to be in (the one
    way past the refusal of a real run from another working tree), the place of the release list (a test never writes the list of the real state directory), a fault hook called at named points (to kill a process in a test) and the random generator of the backoff."""

    item_loader: Callable[..., list[McItem]] | None = None
    transport: httpx.AsyncBaseTransport | None = None
    key_reader: Callable[[Path], str] | None = None
    sleep: Callable[[float], Any] | None = None
    install_signals: bool = True
    results_root: Path | None = None
    code_root: Path | None = None
    release_file: Path | None = None
    fault: Callable[..., None] | None = None
    rng: random.Random | None = None


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Closed-model API experiment: the pilot in two stages (the full run is refused).")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="configuration file (default: configs/api_exp_v1.yaml)")
    p.add_argument("--model", default=None, help="a model id of the configuration, a comma list of them, or 'all' = the models of the release list and no others (required except for --mode full, which is refused); they run in the order of the configuration; a real run of a model that is not released ends with exit 2")
    p.add_argument("--mode", required=True, choices=("pilot", "full"), help="pilot: the development pool; full: refused")
    p.add_argument("--stage", default=None, choices=STAGE_NAMES, help="probe: the first 100 questions; pilot: all 1,000, continuing the probe's directory")
    p.add_argument("--tag", default=None, help="run tag of a NEW run directory: results/api_exp/<time>_<tag>/ (an existing tag, or a model that already has a directory, is refused)")
    p.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue a run directory under results/api_exp/: requests that have a success are not sent again; the stage pilot needs it")
    p.add_argument("--resend-uncertain", action="store_true", help="send once more the requests whose last attempt has no result record (they count as spent at their upper bound); still at most 8 sends each")
    p.add_argument("--dry-run", action="store_true", help="render, count, reserve and apply the gates; send nothing, read no key, write no file")
    p.add_argument("--concurrency", type=int, default=None, help="requests in flight (default: runtime.concurrency of the configuration; may differ between sessions)")
    p.add_argument("--max-session-cny", type=float, default=None, help="stop sending when this session has spent this many CNY (an operator limit on top of the cap)")
    p.add_argument("--available-cny", type=float, default=None, help="the money that is really there for this session, in CNY (for example the balance of the OpenRouter account at 7.3 CNY to the dollar): a model whose estimate does not fit in what is left is not started at all, its shortfall is reported, and the session stops there")
    p.add_argument("--key-file", default=None, help="file with the API key (default: endpoint.key_file of the configuration); read only by a real run")
    p.add_argument("--expect-config-md5", default=None, help="a real run stops unless the configuration file has exactly this md5")
    p.add_argument("--expect-protocol-md5", default=None, help="a real run stops unless the protocol file has exactly this md5 (the version that is in force)")
    return p


def csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


@dataclass
class Prepared:
    cfg: ApiExpConfig
    prices: PriceBook
    base_plan: RequestPlan
    models: list[ModelSpec]
    input_md5: dict[str, Any]
    extra_manifest: dict[str, Any]
    stats: dict[str, Any]  # the sizes of the rendered messages of the whole pool and the md5 over all of them
    release: Release  # the release list as it was when the call started


def prepare(args: argparse.Namespace, hooks: ApiHooks) -> Prepared:
    """Everything a dry run and a real run share: the configuration, the prices, the md5 checks, the pool, the prompt and the plan. Raises the errors that exit 2."""
    cfg = load_api_config(args.config)
    prices = load_price_book(cfg.prices_file, cfg.experiment_model_ids())  # the ledger is shared by the batches: an entry for the models of every batch
    protocol_md5 = file_md5(cfg.protocol_file) if cfg.protocol_file.is_file() else None
    if protocol_md5 is None:  # the protocol text is not part of this release: its md5 is neither required nor compared
        note_skipped(cfg.protocol_file, "protocol")
    if not args.dry_run:
        missing = [flag for flag, value in (("--expect-config-md5", args.expect_config_md5), ("--expect-protocol-md5", args.expect_protocol_md5)) if not value and (flag != "--expect-protocol-md5" or protocol_md5 is not None)]
        if missing:
            raise RunRefused(f"a real run needs {' and '.join(missing)}: no question is sent before the protocol takes effect, so the call states the md5 of the version that is in force")
        if args.expect_config_md5 != cfg.md5:
            raise RunRefused(f"{cfg.path.name} has md5 {cfg.md5}, the call expects {args.expect_config_md5}")
        if protocol_md5 is not None and args.expect_protocol_md5 != protocol_md5:
            raise RunRefused(f"{cfg.protocol_file.name} has md5 {protocol_md5}, the call expects {args.expect_protocol_md5}")
        if not args.resume and not args.tag:
            raise RunRefused("--tag is required for a new run directory (or --resume RUN_DIR to continue one)")
    release = load_release(hooks.release_file or state_dir(api_root(hooks.results_root)) / RELEASE_NAME, cfg.experiment_model_ids())  # one list for every batch
    names = csv_list(args.model)
    if names == ["all"]:  # "all" stands for the models that are released, and no others
        models = [m for m in cfg.models if release.allows(m.id)]
        if not models:
            raise RunRefused(f"no model selected: --model all stands for the models of the release list, and {release.refusal(cfg.model_ids())}")
    else:
        models = cfg.select_models(names)
        refusal = None if args.dry_run else release.refusal([m.id for m in models])
        if refusal:
            raise RunRefused(refusal)
    if not models:
        raise RunRefused("no model selected")
    if args.concurrency is not None and not 1 <= args.concurrency <= 256:
        raise RunRefused("--concurrency must be between 1 and 256")
    if args.max_session_cny is not None and not args.max_session_cny > 0:
        raise RunRefused("--max-session-cny must be positive")
    if args.available_cny is not None and not args.available_cny > 0:
        raise RunRefused("--available-cny must be positive")

    local_cfg = load_config(cfg.local_config)
    local_prompt = Path(local_cfg["prompts"]["file"])
    if (local_prompt if local_prompt.is_absolute() else paths.REPO_ROOT / local_prompt).resolve() != cfg.prompt.file.resolve():
        raise RunRefused(f"the local configuration uses the prompt file {local_cfg['prompts']['file']}, this configuration {rel(cfg.prompt.file)}")
    book = load_prompt_book(cfg, DisplayRule.from_config(local_cfg))
    grid = cfg.grid(args.mode)
    registry = SplitRegistry.load(cfg.split_dir)
    items = load_pilot_items(local_cfg, registry, grid.pool, results_root=paths.results_root(hooks.results_root), loader=hooks.item_loader)
    if len(items) != grid.items_expected:
        raise RunRefused(f"pool {grid.pool} has {len(items)} items, the configuration expects {grid.items_expected}")
    base_plan = RequestPlan(book, items, conditions_of(grid), paraphrase=cfg.prompt.paraphrase, framing=cfg.prompt.framing, layout=cfg.prompt.layout)
    split_files = registry.manifest_md5s([grid.pool])
    stats = base_plan.length_stats()  # renders every message of the pool once: the md5 over all of them is part of the identity of the experiment
    input_md5 = {
        "protocol": protocol_md5, "config": cfg.md5, "config_frozen": cfg.frozen_md5, "prompts": book.md5, "local_config": file_md5(cfg.local_config),
        "split": hashlib.md5(json.dumps(split_files, sort_keys=True).encode("ascii"), usedforsecurity=False).hexdigest(), "split_files": split_files,
        "messages": stats["messages_md5"], "code": code_md5(),
    }
    if cfg.endpoint_kind != "bailian":  # a batch with request-and-reading code of its own (the kind openrouter) adds its md5 to the identity; the other batches have no such key
        input_md5["batch_code"] = batch_code_md5(cfg.endpoint_kind, cfg.batch.name if cfg.batch is not None else None)
    extra = {
        "config_file": rel(cfg.path), "protocol_file": rel(cfg.protocol_file), "local_config_file": rel(cfg.local_config), "split_dir": rel(cfg.split_dir), "prices_file": rel(cfg.prices_file),
        "item_source": "split_registry" if hooks.item_loader is None else "hook", "pool_keys_md5": keys_md5(base_plan.item_keys),
    }
    if cfg.batch is not None:  # only a supplementary batch has the key: the manifest of a first-batch run is as it was
        extra["batch"] = batch_record(cfg)
    return Prepared(cfg, prices, base_plan, models, input_md5, extra, stats, release)


def batch_record(cfg: ApiExpConfig) -> dict[str, Any] | None:
    """What the report and the manifest say of the batch: None for the first batch."""
    if cfg.batch is None:
        return None
    return {
        "name": cfg.batch.name, "base_config": rel(cfg.batch.base), "base_config_md5": cfg.batch.base_md5, "base_config_frozen_md5": cfg.batch.base_frozen_md5,
        "first_order": cfg.batch.first_order, "freeze_file": freeze_file_name(cfg.batch.name), "models_of_the_experiment": list(cfg.experiment_model_ids()),
        **({"own_blocks": list(cfg.batch.own), "endpoint_kind": cfg.endpoint_kind} if cfg.batch.own else {}),
    }


def dry_run_report(prep: Prepared, args: argparse.Namespace, hooks: ApiHooks) -> dict[str, Any]:
    """What a real run would do, as numbers. Sends nothing, reads no key, writes no file (the state is read, never opened for writing)."""
    cfg, prices, base_plan = prep.cfg, prep.prices, prep.base_plan
    root = api_root(hooks.results_root)
    sdir = state_dir(root)
    journal = Journal.load(sdir, prices, max_attempts=cfg.retry.max_attempts, cap_cny=cfg.budget.cap_cny, charge_unknown=cfg.budget.charge_unknown_at_bound, writable=False)
    registry = RunRegistry(sdir)
    budget = BudgetState(sdir)
    freeze = Freeze(sdir, cfg.batch.name if cfg.batch is not None else None)
    stage = args.stage
    resume_name = Path(args.resume).resolve().name if args.resume else None
    problems = gate_problems(cfg, journal, registry, base_plan, prep.models, mode=args.mode, stage=stage, resume_name=resume_name)
    identity = {FREEZE_KEYS[k]: str(prep.input_md5.get(k, "")) for k in FREEZE_KEYS}
    freeze_differs = freeze.mismatches(identity)
    if freeze_differs:
        what = "the experiment" if cfg.batch is None else f"the supplementary batch {cfg.batch.name}"
        problems = [*problems, f"{what} was frozen at its first real send and these differ now: {freeze_differs}"]
    first_batch = first_batch_problem(cfg, sdir, identity)
    if first_batch:
        problems = [*problems, first_batch]
    lockout = assess(journal, sdir)
    if lockout.locked:
        problems = [*problems, f"the experiment is locked out by a bound violation: {'; '.join(lockout.problems)}"]
    release_refusal = prep.release.refusal([m.id for m in prep.models])
    if release_refusal:
        problems = [*problems, release_refusal]
    models = []
    assumed = 0.0
    stopped_at: int | None = None  # the order of the first model the simulation refuses
    available = args.available_cny  # the money said to be there (--available-cny); the session stops at the first model that does not fit
    session_assumed, stopped_for_money = 0.0, False  # what the models that start would spend in this session, each at its whole estimate
    total_requests = 0
    for model in prep.models:
        plan = base_plan.limited(cfg.stage_items(stage))
        cls = cfg.class_of(model.id)
        resend = args.resend_uncertain

        def skip(item_key: str, condition_id: str, model_id: str = model.id, resend: bool = resend) -> bool:
            return not journal.can_send(model_id, item_key, condition_id, resend_uncertain=resend)

        sendable = [message_bytes(r.message) for r in plan.requests(skip)]
        counts = count_states(journal, model.id, plan_keys(plan))
        planned = cfg.stage_new_items(stage) * len(plan.conditions)
        quota = send_quota(planned, cfg.budget.send_quota_factor)
        sent = journal.sends_in_stage(stage, model.id)
        left = max(0, quota - sent)
        per_byte, n_measured = journal.measured_tokens_per_byte(model.id)
        reserve = reserve_of(
            prices, model.id, sendable[:left], cls.max_tokens, cls.input_margin(cfg.budget.template_margin_tokens), cfg.budget.reserve_margin, tokens_per_byte=per_byte, n_measured=n_measured,
            measured_factor=cfg.budget.measured_factor, measured_completion=estimate_completion(cfg, journal, model.id),
            input_bytes_factor=cls.input_bytes_factor, output_margin_tokens=cls.output_margin_tokens,
        )
        peek = budget.peek(stage, model.id, model.order, prices.version)
        spent = journal.spent_cny()
        shortfall = None  # CNY the estimate is short by when the model is refused for the cap or for the money that is there
        if not sendable:
            verdict, reason = "nothing_to_send", None
        elif left <= 0:
            verdict, reason = "refused", "send_quota"  # the model's sends of this stage are used up: it sends nothing and asks nothing of the budget
        elif stopped_for_money or (available is not None and session_assumed + reserve.reserve_cny > available):  # the money that is there does not carry the model to its end: not started, the session stops
            verdict, reason = "refused", "session_stopped" if stopped_for_money else "available_money"
            if not stopped_for_money:
                shortfall = session_assumed + reserve.reserve_cny - available
            stopped_for_money = True
        elif peek == "already_approved":
            verdict, reason = "start", "already_approved"
            session_assumed += reserve.reserve_cny
        elif peek == "stage_stopped" or (stopped_at is not None and model.order >= stopped_at):
            verdict, reason = "refused", "stage_stopped"
        elif spent + assumed + reserve.reserve_cny > cfg.budget.cap_cny:
            verdict, reason = "refused", "budget"
            shortfall = spent + assumed + reserve.reserve_cny - cfg.budget.cap_cny
            stopped_at = model.order if stopped_at is None else min(stopped_at, model.order)
        else:
            verdict, reason = "start", "approved"
            assumed += reserve.reserve_cny  # the models after it are judged as if it had spent its whole estimate
            session_assumed += reserve.reserve_cny
        total_requests += len(sendable)
        p = prices.price(model.id)
        models.append({
            "id": model.id, "class": model.cls, "order": model.order, "max_tokens": cls.max_tokens, "extra_body": dict(cls.extra_body),
            **({"bounds": {"output_tokens": cls.output_bound(), "input_bytes_factor": cls.input_bytes_factor, "input_margin_tokens": cls.input_margin(cfg.budget.template_margin_tokens)}} if cls.bounds_record() else {}),
            "returned_names_stored_as_they_come": sorted(model.returned_names()), **({"wire": model.wire} if model.wire else {}), **({"price_margin": cfg.price_margin, "max_price": price_ceiling(p.input_per_million, p.output_per_million, cfg.price_margin).as_request()} if cfg.price_margin is not None else {}), "run_directory": registry.run_of(args.mode, model.id), "stage_items": cfg.stage_items(stage),
            "requests_in_stage": len(plan.items) * len(plan.conditions), "key_states": {k: counts.get(k, 0) for k in SHOWN_STATES}, "requests_to_send": len(sendable), "shortfall_cny": shortfall,
            "price_cny_per_million": {"input": p.input_per_million, "output": p.output_per_million}, "reserve": reserve.as_record(), "budget_verdict": verdict, "budget_reason": reason,
            "released": prep.release.allows(model.id),
            "send_quota": {"stage": stage, "planned_keys": planned, "factor": cfg.budget.send_quota_factor, "quota": quota, "sent": sent, "left": left},
        })
    problem = code_root_problem(hooks.code_root)
    return {
        "dry_run": True, "sends_nothing": True, "reads_key": False, "writes_files": False, "mode": args.mode, "stage": stage, "protocol": rel(cfg.protocol_file), "config": rel(cfg.path),
        "prices": prices.identity(), "input_md5": prep.input_md5, "endpoint": cfg.endpoint_url, "key_file": str(Path(args.key_file) if args.key_file else cfg.key_file),
        "key_file_exists": Path(args.key_file or cfg.key_file).is_file(), "state_dir": str(sdir), "gate_problems": problems,
        "real_run_allowed_from_here": problem is None, "real_run_refusal": problem,
        "batch": batch_record(cfg), "freeze_file": freeze.path.name,
        "freeze": {"exists": freeze.exists, "frozen_at": freeze.data.get("frozen_at"), "first_run": freeze.data.get("first_run"), "differs": freeze_differs},
        "release": prep.release.as_record(cfg.model_ids()), "lockout": lockout.as_record(),
        "prompt": {"file": rel(cfg.prompt.file), "md5": base_plan.book.md5, "paraphrase": cfg.prompt.paraphrase, "block": cfg.prompt.block, "framing": cfg.prompt.framing, "layout": cfg.prompt.layout},
        "pool": {"name": cfg.grid(args.mode).pool, "n_items": len(base_plan.items), "items_expected": cfg.grid(args.mode).items_expected},
        "conditions": [c.id for c in base_plan.conditions], "rendered_pool": prep.stats, "concurrency": args.concurrency or cfg.concurrency, "available_cny": args.available_cny,
        "retry": {"max_attempts": cfg.retry.max_attempts, "backoff_base_s": cfg.retry.backoff_base_s, "backoff_max_s": cfg.retry.backoff_max_s},
        "ledger": {
            "spent_cny": journal.spent_cny(), "spent_cny_at_current_prices": journal.repriced_cny(prices), "cap_cny": cfg.budget.cap_cny, "n_uncertain_attempts": len(journal.uncertain()),
            "uncertain_reserved_cny": journal.uncertain_reserved_cny(), "prices_verified": prices.verified, "price_versions_booked": journal.price_versions(),
        },
        "uncertain_attempts": [
            {"attempt": a.attempt_id, "run": a.run, "model": a.model, "item_key": a.key[1], "condition": a.key[2], "send": a.n, "t": a.t, "reserved_cny": a.reserved_cny}
            for a in journal.uncertain()
        ],
        "estimate_note": "before a model starts, its remaining requests are counted at the provable bound of a send (the UTF-8 bytes of the message plus the template margin as input tokens, "
                         "max_tokens as output tokens; a model of the endpoint kind openrouter has the wider bounds of its class, shown under `bounds`) or, once the model has measured usage, at the largest measured tokens per byte times the measured factor, and the sum is multiplied by "
                         "1.1; the models are judged in order as if each one that starts had booked its whole estimate (a model whose quota of sends is used up asks nothing, and a model with sends "
                         "left is counted for those sends at most). Before every send the provable bound of that send is checked against the cap and the quota of sends of the stage",
        "models": models, "totals": {"requests": total_requests, "reserve_cny_of_models_that_start": assumed},
    }


def load_lockout(prep: Prepared, hooks: ApiHooks) -> Lockout:
    """The lockout as the ledger and the disposition file say it is now. Read only: no lock, no file written, no key read."""
    sdir = state_dir(api_root(hooks.results_root))
    cfg = prep.cfg
    journal = Journal.load(sdir, prep.prices, max_attempts=cfg.retry.max_attempts, cap_cny=cfg.budget.cap_cny, charge_unknown=cfg.budget.charge_unknown_at_bound, writable=False)
    return assess(journal, sdir)


def summary(outcome: Any, wall_s: float) -> dict[str, Any]:
    m = outcome.manifest
    return {
        "run_dir": str(outcome.run_dir), "exit_status": m["exit_status"], "stop_reason": outcome.stop_reason, "mode": m["mode"],
        "models": {k: {x: v.get(x) for x in ("status", "n_planned", "states", "n_attempts", "decisions", "returned_models", "tokens", "cost_cny", "stopped")} for k, v in m["models"].items()},
        "cost_cny_run": m["cost_cny_run"], "ledger": m["ledger"], "wall_s": wall_s,
    }


def main(argv: Sequence[str] | None = None, hooks: ApiHooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or ApiHooks()
    args = make_parser().parse_args(argv)
    if args.mode == "full":  # before anything is read, whatever else was given (also with --dry-run: a dry run of the full pool would read it)
        print(f"error: {FULL_MODE_REFUSAL}", file=sys.stderr)
        return EXIT_REFUSED
    if not args.model:
        make_parser().error("--model is required")
    if not args.stage:
        make_parser().error("--stage is required: probe or pilot")
    started = time.time()
    try:
        prep = prepare(args, hooks)
    except (ApiConfigError, ConfigError, PriceError, RenderError, RunRefused, SplitError, PromptError, paths.PathPolicyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_REFUSED
    if args.dry_run:
        try:
            report = dry_run_report(prep, args, hooks)
        except (JournalError, StateError, paths.PathPolicyError) as error:
            print(f"error: {error}", file=sys.stderr)
            return EXIT_REFUSED
        print(json.dumps(report, indent=2, sort_keys=True))
        return EXIT_REFUSED if report["gate_problems"] else 0
    problem = code_root_problem(hooks.code_root)  # before the key is read
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return EXIT_REFUSED
    try:  # a bound violation of any earlier session locks the experiment: nothing starts, and the key is not read, until a disposition closes it
        lockout = load_lockout(prep, hooks)
    except (JournalError, StateError, paths.PathPolicyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_REFUSED
    if lockout.locked:
        print(f"error: {lockout.refusal()}", file=sys.stderr)
        return EXIT_REFUSED
    try:  # the experiment (or the batch) was frozen with another procedure, or with other code of the batch's own: refused before the key is read (the run asks again under the lock)
        sdir = state_dir(api_root(hooks.results_root))
        frozen = freeze_refusal(prep.cfg, sdir, {FREEZE_KEYS[k]: str(prep.input_md5.get(k, "")) for k in FREEZE_KEYS})
    except (StateError, paths.PathPolicyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_REFUSED
    if frozen:
        print(f"error: {frozen}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        key_file = Path(args.key_file) if args.key_file else prep.cfg.key_file
        api_key = (hooks.key_reader or read_api_key)(key_file)
        settings = RunSettings(
            mode=args.mode, stage=args.stage, concurrency=args.concurrency or prep.cfg.concurrency, results_root=hooks.results_root, code_root=hooks.code_root,
            release_file=hooks.release_file, argv=["scripts/run_api_decisions.py", *argv], tag=args.tag,
            resume=Path(args.resume) if args.resume else None, max_session_cny=args.max_session_cny, available_cny=args.available_cny, resend_uncertain=args.resend_uncertain,
            expected_md5_checked={"config": True, "protocol": True},
        )
        run = ApiRun(
            prep.cfg, prep.prices, prep.base_plan, prep.models, settings, prep.input_md5, api_key=api_key, transport=hooks.transport, sleep=hooks.sleep,
            install_signals=hooks.install_signals, extra_manifest=prep.extra_manifest, fault=hooks.fault, rng=hooks.rng,
        )
        outcome = run.run()
    except (RunRefused, paths.PathPolicyError) as error:  # refused before the first send
        print(f"error: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as error:  # noqa: BLE001
        # the run failed: say what kind of failure and where, never its message (no text from outside reaches a log)
        print(f"error: the run stopped with {describe_exception(error)}; the manifest of the run directory says where it stands, and the run can be resumed", file=sys.stderr)
        return EXIT_ABORTED
    print(json.dumps(summary(outcome, time.time() - started), indent=2, sort_keys=True, default=str))
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
