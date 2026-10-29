"""One run of the API experiment: the state, the run directory and its manifest, the stages, the models in the configured order, the budget, and resume.

What a run touches, and where. Everything that must be shared by all runs is in ONE absolute directory, ``FIXED_STATE_DIR`` (``results/api_exp/_state`` of the checkout at ``FIXED_CODE_ROOT``,
``nmi.api_exp.state``): the process lock, the write-ahead journal and the index of successes (``nmi.api_exp.journal``), the registry of run directories, the budget state and the freeze record.
A run directory ``results/api_exp/<YYYYMMDD_HHMMSS>_<tag>/`` (in the parent of that directory) holds

* ``manifest.json``: the md5 of the protocol, the configuration (and its frozen part), the prompt file, the local configuration and the split; the conditions; per model the plan (a
  VERSIONED key file under ``plans/`` and a pointer to it), the status, the decision of the budget, the counts, the tokens, the model names the service returned and the cost; every
  session with its start and end, its stage and the price version it ran with; the token totals and the money as it was booked in the ledger, with beside it the same tokens worked out at the
  current price version (``cost_cny_at_current_prices``);
* ``plans/<model>.v<k>.keys``: the item keys of a model's plan, one per line (hashes, not question text);
* ``responses.jsonl``: the per-run view of the first successes (``nmi.api_exp.store``), a copy of data that is in the journal.

Rules enforced here. There is one run directory per (mode, model): a new tag for a model that has a run directory is refused, the directory is continued. The probe and the pilot of a model are
two stages of the SAME directory: the pilot grows the plan of the probe (first the key file of the new version is written, then the pointer in the manifest is replaced atomically).
Before each send an attempt record is on disk and after it a result record. A request has at most ``max_attempts`` sends in all sessions and a (stage, model) has at most its quota of sends in
all sessions. The cap is checked before every send, on the provable bound of that send, and before every model, on an estimate; a model that was refused for budget stops its stage for the
models after it. The money is booked at the price of the day, each count of a reply's usage on its own. A reply with a count above the bound of its send stops the run and locks the experiment
until the operator records a disposition. Only the models of the operator's release list start. The experiment is frozen just before its first real send and every session compares itself with
the record. The process lock is held for the whole run.

A SUPPLEMENTARY batch (for example supp1, configuration not distributed) is a run of this same code with another configuration file: its models are others, everything else in the configuration is the
first batch's. It shares the whole state directory with the first batch (the ledger, so the cap of 500 CNY is judged on the money booked for both batches; the index of successes; the registry
of run directories; the budget state; the release list; the lock), except for the freeze record: it freezes at its own first real send into ``freeze_<name>.json`` and compares itself with that
file only. It also READS ``freeze.json`` and refuses to run unless the configuration it repeats is the one the first batch was frozen with (the frozen md5 of the first batch's file is the
``config_frozen_md5`` of the record) and the prompt, the local configuration, the split, the rendered messages and the request and reading code are the ones the first batch recorded
(``first_batch_problem``). It never writes ``freeze.json``, and the first batch never writes ``freeze_<name>.json``.

A batch of the endpoint kind ``openrouter`` has in addition two blocks of its own, a request class for every pinned service and thinking mode, the facts of ``nmi.api_exp.openrouter`` recorded
for each reply (the md5 of that module is in its record), an estimate that counts the output of a model from the measurement, and the status 402 as one of the statuses that stop the run.
The batches supp4 and supp5 add a PRICE MARGIN of 2.0 (``endpoint.price_margin``), by which the reserve of every send is multiplied (a booked amount above that reserve is a lockout like a count
of tokens above its bound), the reported price read by ``nmi.api_exp.price_rules`` instead of the 40-character rule of ``nmi.api_exp.openrouter``, and the last step of the request in
``nmi.api_exp.final_request``. The md5 of ``price_rules`` and ``final_request`` is in their ``batch_code_md5``.

Nothing in the manifest, the journal or the log is text that a server wrote or a question or a prompt; the log holds counts, model names, money and the names of exception types.

The modules of ``nmi.api_exp`` (the package file stays empty):

* ``config``: the validated configuration (``configs/api_exp_v1.yaml``), the stages and the md5 that a resumed run must keep;
* ``prices``: the versioned price file ``configs/api_exp_prices.yaml``;
* ``render``: the conditions, the user message (the existing prompt code) and the pool;
* ``reader``: the reading of a reply;
* ``client``: one send and its classification, and what is kept of a reply;
* ``durable``: fsync-ed appends and atomic whole-file writes;
* ``journal``: the write-ahead ledger, the index of successes, the number of sends, the quota of sends of a stage and the cap;
* ``lockout``: the lockout after a bound violation and the operator's disposition file; ``release``: the operator's list of the models that may start;
* ``openrouter``, ``price_rules``, ``final_request``: what a batch of the kind openrouter has of its own (the reply's facts; for supp4 and supp5 the rules of the price and the last step of the request);
* ``state``: the fixed state directory, the process lock, the registry of run directories and the persisted budget approvals;
* ``ledger``: the upper bounds and the reserve of a model;
* ``store``: the per-run view of the successes;
* ``runner`` (this module): one run;
* ``analysis``: the quantities of the analysis from a run directory, with the state of every model.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import random
import signal
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from nmi.api_exp.client import (
    AUTH_REFUSED,
    SUCCESS,
    ChatClient,
    ParsedReply,
    backoff_delay,
    parse_chat_body,
    returned_model_key,
)
from nmi.api_exp.config import STAGE_NAMES, ApiExpConfig, ModelSpec, RetryPolicy
from nmi.api_exp.durable import write_json_atomic, write_text_atomic
from nmi.api_exp.final_request import request_body
from nmi.api_exp.journal import (
    BoundViolationOpen,
    CapReached,
    Journal,
    JournalError,
    SendQuotaReached,
)
from nmi.api_exp.ledger import message_bytes, reserve_of, send_quota
from nmi.api_exp.lockout import DISPOSITION_NAME, Lockout, assess
from nmi.api_exp.openrouter import (
    apply_router_facts,
    provider_key,
    read_router_facts,
    reported_cny,
)
from nmi.api_exp.price_rules import (
    PriceCeiling,
    decimal_text,
    price_ceiling,
    reported_price,
)
from nmi.api_exp.prices import PriceBook
from nmi.api_exp.reader import DECISION_VALUES
from nmi.api_exp.release import RELEASE_NAME, Release, load_release
from nmi.api_exp.render import RenderedRequest, RequestPlan, keys_md5, keys_text
from nmi.api_exp.state import (
    SHARED_FIELDS,
    AdmitDecision,
    BudgetState,
    Freeze,
    RunRegistry,
    StateError,
    api_root,
    code_root_problem,
    process_lock,
    state_dir,
)
from nmi.api_exp.store import REPLY, STORE_NAME, ResponseStore
from nmi.paths import REPO_ROOT
from nmi.provenance import create_run_dir, file_md5, iso, now_local, source_hash

MANIFEST_NAME = "manifest.json"
PLANS_DIR = "plans"
MANIFEST_SCHEMA = "nmi.api_exp.manifest/3"
EXIT_OK, EXIT_INCOMPLETE, EXIT_REFUSED, EXIT_ABORTED = 0, 1, 2, 3

# model statuses in the manifest
COMPLETE = "complete"  # every request of the plan has a success
FINISHED_WITH_FAILURES = "finished_with_failures"  # every request has ended for good, some without a success
INCOMPLETE = "incomplete"  # some request has still sends left, was not sent, or is uncertain
NOT_RUN_BUDGET = "not_run_budget"
NOT_STARTED = "not_started"
# the states in which a request will not be sent again: it has a reply, or it ended for good, or its sends are used up (an uncertain last send included). A request refused for the key
# (auth_blocked) is not here: it is not a property of the request, and a stage with such a request is not completely ended.
ENDED_STATES = ("success", "permanent", "exhausted", "uncertain_exhausted")
SHOWN_STATES = ("success", "permanent", "exhausted", "uncertain_exhausted", "auth_blocked", "retryable", "uncertain", "unsent")
# the inputs a resumed run must find unchanged; the plan of each model is checked on its own. The same keys (with the suffix _md5) are the
# identity of the whole experiment in the freeze record.
FROZEN_INPUTS = ("protocol", "config_frozen", "prompts", "local_config", "split", "messages", "code", "batch_code")
# the sources that build the request, read the reply and render the message: their md5 is part of the identity of the experiment
CODE_FILES = ("nmi/api_exp/client.py", "nmi/api_exp/reader.py", "nmi/api_exp/render.py")
# the sources that a batch of a kind has of its own besides the three above (the request fields come from its request classes, the reading of what OpenRouter returns besides is here). Their md5
# is part of the freeze record of that batch only (``batch_code_md5``); the first batch and the supplementary batches 1 and 2 have none
BATCH_CODE_FILES = {"openrouter": ("nmi/api_exp/openrouter.py",)}
# and the sources that supp4 and supp5 have besides openrouter.py: the last step of the request, the id of the model becoming the name that is sent, and
# the rules of the price (the reserve of a send with its margin, the exact reading of the reported price). They are in the ``batch_code_md5`` of these two batches only; that of supp3 does not
# include them
BATCH_EXTRA_CODE_FILES = {name: ("nmi/api_exp/final_request.py", "nmi/api_exp/price_rules.py") for name in ("supp4", "supp5")}
FREEZE_KEYS = {
    "protocol": "protocol_md5", "config_frozen": "config_frozen_md5", "prompts": "prompts_md5", "local_config": "local_config_md5", "split": "split_md5", "messages": "messages_md5",
    "code": "code_md5", "batch_code": "batch_code_md5",
}


def first_batch_problem(cfg: ApiExpConfig, directory: Path, identity: Mapping[str, str]) -> str | None:
    """Why a session of a SUPPLEMENTARY batch may not run with this identity, or None. The first batch's record (``freeze.json``) is read here, never written, and only once the first batch
    has been frozen. Two things must hold:

    * the configuration that the supplementary file repeats is the one the first batch was FROZEN with: the frozen md5 of the first batch's file, as the loader parsed it
      (``BatchSpec.base_frozen_md5``), is the ``config_frozen_md5`` of the record. The loader compares the supplementary file with the first batch's file as it is now, so without this a
      pair of files changed together after the first batch was frozen (another ``max_tokens``, say) would agree with each other and with nothing that was frozen. The block ``runtime`` is
      not in the frozen md5, so the concurrency may still differ;
    * the procedure of the supplementary batch (the prompt file, the local configuration, the split, all rendered messages, the request and reading code) is the one the record holds, so
      that the models of the two batches were asked in the same way. The protocol is each batch's own.

    A batch that asks in a cell of its own (``cfg.cell_batch``: the a_H of its pilot is not the first batch's) renders other messages, so it shares the other four values only; its own record holds
    its messages.

    The session is refused (``_load_state``, under the lock) and the dry run reports the same text. A session of the first batch has nothing to check here."""
    if cfg.batch is None:
        return None
    first = Freeze(directory)
    if not first.exists:
        return None
    name, when = cfg.batch.name, f"{first.data.get('frozen_at')}, run {first.data.get('first_run')}"
    problems = []
    recorded = first.data["identity"].get("config_frozen_md5")
    if recorded != cfg.batch.base_frozen_md5:
        problems.append(
            f"the supplementary batch {name} repeats the configuration of the first batch, but {cfg.batch.base.name} is not the configuration the first batch was frozen with ({when}): "
            f"its frozen md5 is {cfg.batch.base_frozen_md5} and the record holds {recorded}; no model of a supplementary batch can be started with a changed method (request, reading, pools, "
            "conditions, stages, budget, retry)"
        )
    # a batch that asks in a cell of its own renders other messages (the a_H is in the text), so the rendered messages are not shared with the first batch: its own record holds and compares them
    differs = first.mismatches(identity, tuple(f for f in SHARED_FIELDS if not (cfg.cell_batch and f == "messages_md5")))
    if differs:
        problems.append(
            f"the supplementary batch {name} is asked in the way of the first batch, which was frozen at its first real send ({when}), and these differ now: {differs}; "
            "no model of a supplementary batch can be started with another prompt, split, message or request/reading code"
        )
    return "; ".join(problems) or None


def freeze_refusal(cfg: ApiExpConfig, directory: Path, identity: Mapping[str, str]) -> str | None:
    """Why a session with this identity may not run, from the freeze records alone, or None: the experiment (or the supplementary batch) was frozen with another protocol, configuration,
    prompt, split, message, request/reading code or code of the batch's own, or a supplementary batch is not asked in the way the first batch was frozen. Read only, no lock: the script asks
    this BEFORE it reads the key, and ``ApiRun`` asks it again under the lock (a session may have written the record in between)."""
    batch = cfg.batch.name if cfg.batch is not None else None
    freeze = Freeze(directory, batch)
    differs = freeze.mismatches(identity)
    if differs:
        what = "the experiment" if batch is None else f"the supplementary batch {batch}"
        return (
            f"{what} was frozen at its first real send ({freeze.data.get('frozen_at')}, run {freeze.data.get('first_run')}) and these differ now: {differs}; "
            "no model or run directory can be started with another protocol, configuration, prompt, split, message or request/reading code"
        )
    return first_batch_problem(cfg, directory, identity)


def batch_code_files(kind: str, batch: str | None = None) -> tuple[str, ...]:
    """The sources a batch of this endpoint kind has of its own: those of the kind (``BATCH_CODE_FILES``) and, for supp4 and supp5, the two of ``BATCH_EXTRA_CODE_FILES``; () for a kind
    that has none."""
    names = BATCH_CODE_FILES.get(kind, ())
    return (*names, *BATCH_EXTRA_CODE_FILES.get(batch or "", ())) if names else ()


def batch_code_md5(kind: str, batch: str | None = None) -> str:
    """The md5 over the sources a batch of this endpoint kind has of its own (``batch_code_files``), as ``code_md5`` is over the three frozen ones; "" for a kind that has none. For every
    batch but supp4 and supp5 it is the md5 over the sources of the kind alone, as it was when the batch was frozen."""
    names = batch_code_files(kind, batch)
    if not names:
        return ""
    digest = hashlib.md5(usedforsecurity=False)
    for name in names:
        digest.update(f"{name}:{file_md5(REPO_ROOT / name)}\n".encode("ascii"))
    return digest.hexdigest()


def code_md5() -> str:
    """The md5 over the three sources of ``CODE_FILES`` (by name and md5): the request and reading code of the experiment."""
    digest = hashlib.md5(usedforsecurity=False)
    for name in CODE_FILES:
        digest.update(f"{name}:{file_md5(REPO_ROOT / name)}\n".encode("ascii"))
    return digest.hexdigest()


class RunRefused(RuntimeError):
    """The run cannot start (an input, a lock, a gate or a resume does not match): the entry prints the message and exits 2. Nothing has been sent."""


def rel(path: str | Path) -> str:
    """A repository-relative path for the manifest where the file is inside the repository, else the path as it is."""
    p = Path(path)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def leaf_exception(error: BaseException) -> BaseException:
    """The exception a TaskGroup wrapped (the first one of a group), else the exception itself."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


def describe_exception(error: BaseException) -> str:
    """The name of an exception type and the place it was raised (file and line), never its message: a message can carry text that came from outside."""
    leaf = leaf_exception(error)
    frames = traceback.extract_tb(leaf.__traceback__)
    where = f" at {Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else ""
    return f"{type(leaf).__name__}{where}"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def read_api_key(path: str | Path) -> str:
    """The key, read at run time inside the process (first line of the file). It is returned to the caller that builds the client and is never printed, logged or written."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise RunRefused(f"the key file cannot be read ({type(error).__name__}); the path is the endpoint.key_file of the configuration or --key-file") from error
    lines = text.strip().splitlines()
    key = lines[0].strip() if lines else ""
    if not key:
        raise RunRefused("the key file is empty")
    return key


# ---------------------------------------------------------------------------------------
# records and summaries
# ---------------------------------------------------------------------------------------

def reply_payload(request: RenderedRequest, parsed: ParsedReply, max_tokens: int, attempt_no: int) -> dict[str, Any]:
    """The data of a success that is kept: no text that a server wrote. ``first_char`` is set only for a digit 1 to 5."""
    return {
        "message_md5": request.message_md5, "prompt_chars": len(request.message), "max_tokens": max_tokens, "decision": parsed.decision, "first_char": parsed.first_char,
        "reply_chars": parsed.reply_chars, "reply_sha256": parsed.reply_sha256, "finish": parsed.finish, "returned_model": parsed.returned_model,
        "returned_model_sha256": parsed.returned_model_sha256, "reasoning_present": parsed.reasoning_present, "anomaly": parsed.anomaly, "attempts": attempt_no,
    }


PAYMENT_REQUIRED = 402  # OpenRouter's answer to an account without credits


def client_retry(cfg: ApiExpConfig) -> RetryPolicy:
    """The retry policy the client is built with. For the endpoint kind openrouter the status 402 (payment required: the credits of the account are used up) is one of the abort statuses, like 401
    and 403: it stops the run and does not end the request, so a request refused for want of credits is sent again when the run is resumed after the account has been topped up, and not
    given up for good (any other 4xx is the end of a request). The configuration, which is the first batch's in this block, is not changed."""
    if cfg.endpoint_kind == "openrouter" and PAYMENT_REQUIRED not in cfg.retry.abort_statuses:
        return dataclasses.replace(cfg.retry, abort_statuses=(*cfg.retry.abort_statuses, PAYMENT_REQUIRED))
    return cfg.retry


def estimate_completion(cfg: ApiExpConfig, journal: Journal, model_id: str) -> int | None:
    """The largest output token count of a reply of the model, for the estimate of a model of a batch that counts its output from the measurement (the endpoint kind openrouter);
    None for every other batch, whose estimate counts ``max_tokens``, and for a model that has no measured reply yet."""
    return journal.measured_completion_per_request(model_id)[0] if cfg.endpoint_kind == "openrouter" else None


def plan_file_name(model_id: str, version: int) -> str:
    """The key file of a version of a model's plan, ``plans/<model>.v<k>.keys``. A "/" in the id (an OpenRouter id is vendor/name) becomes "__", so that the file stays directly in ``plans/``;
    the id of every other model is as it was."""
    return f"{PLANS_DIR}/{model_id.replace('/', '__')}.v{version}.keys"


def view_record(key: tuple[str, str, str], payload: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    """The line of ``responses.jsonl`` for a success; it can be rebuilt from the journal alone."""
    model, item_key, condition = key
    return {"kind": REPLY, "model": model, "item_key": item_key, "condition": condition, **payload, "usage": meta.get("usage"), "latency_s": meta.get("latency_s"), "t": meta.get("t")}


def plan_keys(plan: RequestPlan) -> list[tuple[str, str]]:
    return [(item.item_key, cond.id) for item in plan.items for cond in plan.conditions]


def count_states(journal: Journal, model_id: str, keys: Sequence[tuple[str, str]]) -> Counter:
    counts: Counter = Counter()
    for item_key, condition in keys:
        counts[journal.state(model_id, item_key, condition)] += 1
    return counts


def status_of(counts: Counter, total: int) -> str:
    """complete, finished_with_failures (every request ended for good, some without a reply), incomplete or not_started (nothing of the plan was ever sent)."""
    if counts.get("success", 0) == total:
        return COMPLETE
    if counts.get("unsent", 0) == total:
        return NOT_STARTED
    if sum(counts.get(s, 0) for s in ENDED_STATES) == total:
        return FINISHED_WITH_FAILURES
    return INCOMPLETE


def probe_state(cfg: ApiExpConfig, journal: Journal, base_plan: RequestPlan, model: ModelSpec) -> tuple[bool, Counter]:
    """Has the probe of this model ended completely? Every request of the probe plan has ended for good (a success, a 4xx that ends it, or all its sends used); none is unsent, to be
    sent again or uncertain."""
    plan = base_plan.limited(cfg.stage_items("probe"))
    counts = count_states(journal, model.id, plan_keys(plan))
    return sum(counts.get(s, 0) for s in ENDED_STATES) == sum(counts.values()), counts


def gate_problems(
    cfg: ApiExpConfig, journal: Journal, registry: RunRegistry, base_plan: RequestPlan, models: Sequence[ModelSpec], *, mode: str, stage: str, resume_name: str | None,
) -> list[str]:
    """Why the stage cannot start, one line each. Empty when it can. Changes nothing.

    The stage pilot waits for the probe of every model it names to have ended completely, and for nothing else. It does NOT wait for verified unit prices: the money is booked at the prices of the
    price file as they are, the placeholder of version 0 included, and the cap of 500 CNY judges the money that was booked."""
    problems: list[str] = []
    for model in models:
        held = registry.run_of(mode, model.id)
        if resume_name is None and held is not None:
            problems.append(f"{model.id} already has a run directory ({held}); continue it with --resume {held} instead of starting another")
        if resume_name is not None and held is not None and held != resume_name:
            problems.append(f"{model.id} belongs to the run directory {held}, not {resume_name}")
    if stage == "pilot":
        if resume_name is None:
            problems.append("stage pilot continues the run directory of the probe: give --resume RUN_DIR")
        for model in models:
            held = registry.run_of(mode, model.id)
            if held is None:
                problems.append(f"{model.id} has no probe: the stage pilot starts only after its probe has ended completely")
                continue
            done, counts = probe_state(cfg, journal, base_plan, model)
            if not done:
                left = {k: v for k, v in sorted(counts.items()) if k not in ENDED_STATES}
                problems.append(f"the probe of {model.id} has not ended completely ({left}); the stage pilot starts only after it")
    return problems


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------

@dataclass
class ModelControl:
    """What stops the sending to one model: the cancel event of its sends and the reason."""

    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    reason: str | None = None


@dataclass
class RunSettings:
    mode: str
    stage: str
    concurrency: int
    results_root: Path | None = None  # None: the one absolute state directory; there is no option or variable for another one (a parameter for testing only)
    code_root: Path | None = None  # None: the repository this code is in (another one only for testing)
    release_file: Path | None = None  # None: release.json of the state directory (another place only for testing)
    argv: list[str] = field(default_factory=list)
    tag: str | None = None
    resume: Path | None = None
    max_session_cny: float | None = None
    available_cny: float | None = None  # the money that is really there for this session (the OpenRouter balance, in CNY): a model whose estimate does not fit in what is left is not started
    resend_uncertain: bool = False
    expected_md5_checked: dict[str, bool] = field(default_factory=dict)
    progress_every: int = 500
    progress_seconds: float = 60.0


@dataclass
class RunOutcome:
    run_dir: Path
    exit_code: int
    stop_reason: str | None
    manifest: dict[str, Any]


class ApiRun:
    def __init__(
        self, cfg: ApiExpConfig, prices: PriceBook, base_plan: RequestPlan, models: Sequence[ModelSpec], settings: RunSettings, input_md5: Mapping[str, Any], *, api_key: str,
        transport: httpx.AsyncBaseTransport | None = None, sleep: Callable[[float], Any] | None = None, install_signals: bool = True, extra_manifest: Mapping[str, Any] | None = None,
        fault: Callable[..., None] | None = None, rng: random.Random | None = None,
    ) -> None:
        if settings.stage not in STAGE_NAMES:
            raise RunRefused(f"unknown stage {settings.stage!r}; the stages are {list(STAGE_NAMES)}")
        self.cfg, self.prices, self.base_plan, self.models, self.settings = cfg, prices, base_plan, list(models), settings
        self.input_md5 = dict(input_md5)
        self._api_key, self._transport, self._sleep = api_key, transport, sleep
        self._install_signals = install_signals
        self._extra = dict(extra_manifest or {})
        self._fault = fault or (lambda point, **info: None)  # fault-injection hook for testing: it may kill the process at a named point; nothing else uses it
        self._rng = rng or random.Random()
        self.root = api_root(settings.results_root)
        self.sdir = state_dir(self.root)
        self.stop: asyncio.Event | None = None  # created inside the event loop
        self.stop_reason: str | None = None
        self.run_dir: Path | None = None
        self.run_name = ""
        self.store: ResponseStore | None = None
        self.journal: Journal | None = None
        self.registry: RunRegistry | None = None
        self.budget: BudgetState | None = None
        self.freeze: Freeze | None = None
        self.release: Release | None = None
        self.lockout: Lockout | None = None
        self._ceilings: dict[str, PriceCeiling] = {}  # per model: the ceiling of the price of its requests (supp4, supp5), worked out once per session
        self._quota: dict[str, dict[str, Any]] = {}  # per model: the quota of sends of the stage of this session (stage, planned keys, factor, quota)
        self.manifest: dict[str, Any] = {}
        self._control: ModelControl | None = None
        self._consecutive_ended = 0
        self._done_since_progress = 0
        self._last_progress = time.monotonic()
        self._stats: Counter[str] = Counter()
        self._admit: dict[str, AdmitDecision] = {}
        self._refused: set[str] = set()  # models refused for budget in this session

    # ---- starting --------------------------------------------------------------------------
    def run(self) -> RunOutcome:
        """The whole run, synchronously. Raises ``RunRefused`` (exit 2) when it cannot start; everything after the first send is reported in the outcome."""
        problem = code_root_problem(self.settings.code_root)
        if problem:
            raise RunRefused(problem)
        try:
            with process_lock(self.sdir):
                self._load_state()
                self._check_release()
                self._check_lockout()
                try:
                    self._preflight()
                    self.journal.open_writer()
                    self._record_dispositions()
                    self._open()
                    try:
                        asyncio.run(self._main())
                    finally:
                        self._close()
                finally:
                    self.journal.close()
        except (StateError, JournalError) as error:
            raise RunRefused(str(error)) from error
        return RunOutcome(self.run_dir, self._exit_code(), self.stop_reason, self.manifest)

    def _load_state(self) -> None:
        cfg = self.cfg
        self.journal = Journal.load(
            self.sdir, self.prices, max_attempts=cfg.retry.max_attempts, cap_cny=cfg.budget.cap_cny, charge_unknown=cfg.budget.charge_unknown_at_bound, writable=False,
        )
        self.registry = RunRegistry(self.sdir)
        self.budget = BudgetState(self.sdir)
        self.budget.check_price_identity(self.prices.version, self.prices.md5)
        batch = cfg.batch.name if cfg.batch is not None else None  # a supplementary batch has a freeze record of its own; the first batch's is only read by it
        self.freeze = Freeze(self.sdir, batch)
        problem = freeze_refusal(cfg, self.sdir, self._identity())
        if problem:
            raise RunRefused(problem)
        if self.journal.index_conflicts():
            raise RunRefused("the index of successes names another run than the ledger for some request; the state is not to be guessed at")
        if self.journal.torn_tail:
            log(f"the ledger had a last line cut off by a crash ({self.journal.torn_tail}); it was never complete and is dropped")

    def _check_release(self) -> None:
        """Only the models of the release list start. The script has checked before it read the key; this is the check of whoever calls ``run``."""
        self.release = load_release(self.settings.release_file or self.sdir / RELEASE_NAME, self.cfg.experiment_model_ids())
        refusal = self.release.refusal([m.id for m in self.models])
        if refusal:
            raise RunRefused(refusal)

    def _check_lockout(self) -> None:
        """A bound violation of any earlier session keeps every start and resume out until an operator's disposition closes it."""
        self.lockout = assess(self.journal, self.sdir)
        if self.lockout.locked:
            raise RunRefused(self.lockout.refusal())

    def _record_dispositions(self) -> None:
        """The entries of the operator's disposition file that close open violations go to the ledger before anything else is written (the run goes on only after that)."""
        for entry in self.lockout.to_record:
            self.journal.write_disposition(
                covers=[number for number, _ in entry.covers], reason_sha256=entry.reason_sha256, reason_chars=entry.reason_chars, date=entry.date, file_name=DISPOSITION_NAME,
                file_md5=str(self.lockout.file_md5), now=iso(now_local()),
            )
            log(f"the operator's disposition of {entry.date} closes the bound violation(s) {[number for number, _ in entry.covers]}; it is in the ledger now")
        if self.journal.open_violations():
            raise RunRefused("a bound violation is still open after the disposition was recorded; the state is not to be guessed at")

    def _identity(self) -> dict[str, str]:
        """The identity of the experiment as this session sees it (the keys of the freeze record)."""
        return {FREEZE_KEYS[k]: str(self.input_md5.get(k, "")) for k in FROZEN_INPUTS}

    def _ensure_frozen(self) -> None:
        """Just before the first real send of the experiment, write the freeze record (once); every session has compared itself with it already."""
        if self.freeze.exists:
            return
        files = {
            "protocol": rel(self.cfg.protocol_file), "config": rel(self.cfg.path), "prompts": rel(self.cfg.prompt.file), "local_config": rel(self.cfg.local_config),
            "split": rel(self.cfg.split_dir), "code": ", ".join(CODE_FILES),
        }
        if self.cfg.endpoint_kind in BATCH_CODE_FILES:
            files["batch_code"] = ", ".join(batch_code_files(self.cfg.endpoint_kind, self.cfg.batch.name if self.cfg.batch is not None else None))
        self.freeze.write(self._identity(), run=self.run_name, now=iso(now_local()), files=files)
        log(f"the experiment is frozen now: {self.freeze.path.name} holds the md5 of the protocol, the configuration, the prompt, the split, the messages and the request and reading code")

    def _resume_name(self) -> str | None:
        if self.settings.resume is None:
            return None
        run_dir = Path(self.settings.resume).resolve()
        if run_dir.parent != self.root.resolve():
            raise RunRefused(f"--resume {run_dir}: a run directory of {self.root} is expected")
        return run_dir.name

    def _preflight(self) -> None:
        """Every refusal that does not need a run directory, before anything is created or sent."""
        mode = self.settings.mode
        problems = gate_problems(
            self.cfg, self.journal, self.registry, self.base_plan, self.models, mode=mode, stage=self.settings.stage, resume_name=self._resume_name(),
        )
        if problems:
            raise RunRefused("; ".join(problems))
        if self.settings.resume is None and not self.settings.tag:
            raise RunRefused("--tag is required for a new run")

    def _open(self) -> None:
        settings = self.settings
        item_keys_by_model = {m.id: self._stage_plan(m).item_keys for m in self.models}
        if settings.resume is not None:
            self.run_dir = Path(settings.resume).resolve()
            self.run_name = self.run_dir.name
            self.manifest = self._check_resume(self.run_dir)
            if self.manifest.get("error"):  # the error of the earlier session stays with that session
                self.manifest["sessions"][-1]["error"] = self.manifest.pop("error")
            self.manifest["sessions"].append(self._session_record())
            self.manifest["exit_status"] = "running"
            self.manifest["stop_reason"] = None
            now_hash = source_hash()
            self.manifest["sessions"][-1]["nmi_source_sha256"] = now_hash
            if now_hash != self.manifest.get("nmi_source_sha256"):
                log("warning: the nmi/ source differs from the one this run started with (the prompt file and the other frozen inputs are unchanged); recorded in the manifest")
            for model in self.models:
                if model.id not in self.manifest["models"] or "plan" not in self.manifest["models"][model.id]:
                    self._new_model_entry(model, item_keys_by_model[model.id])
        else:
            self.run_dir = create_run_dir(self.root.name, settings.tag, root=self.root.parent)
            self.run_name = self.run_dir.name
            self.manifest = self._new_manifest()
            for model in self.models:
                self._new_model_entry(model, item_keys_by_model[model.id])
        self.registry.register(settings.mode, [m.id for m in self.models], self.run_name)
        self.store = ResponseStore(self.run_dir / STORE_NAME).open()
        self._sync_view()
        self.journal.repair_index()
        self._write_manifest()

    def _close(self) -> None:
        if self.store is not None:
            self.store.close()

    def _stage_plan(self, model: ModelSpec) -> RequestPlan:
        return self.base_plan.limited(self.cfg.stage_items(self.settings.stage))

    def _session_record(self) -> dict[str, Any]:
        s = self.settings
        return {
            "start_time": iso(now_local()), "end_time": None, "argv": list(s.argv), "stage": s.stage, "concurrency": s.concurrency, "max_session_cny": s.max_session_cny, "available_cny": s.available_cny,
            "resend_uncertain": s.resend_uncertain, "models_asked": [m.id for m in self.models], "price": self.prices.identity(), "exit_status": "running",
        }

    def _new_manifest(self) -> dict[str, Any]:
        cfg, plan, s = self.cfg, self.base_plan, self.settings
        book = plan.book
        return {
            "schema": MANIFEST_SCHEMA, "kind": "api_exp_run", "mode": s.mode, "development_data": s.mode == "pilot", "tag": s.tag, "run_id": self.run_name, "exit_status": "running",
            "stop_reason": None, "start_time": iso(now_local()), "end_time": None, "sessions": [self._session_record()],
            "input_md5": dict(self.input_md5), "expected_md5_checked": dict(s.expected_md5_checked),
            "prompt": {
                "file": rel(cfg.prompt.file), "md5": book.md5, "paraphrase": cfg.prompt.paraphrase, "block": cfg.prompt.block, "framing": cfg.prompt.framing, "layout": cfg.prompt.layout,
                "assistant_prefix_used": False, "system_message": False,
            },
            "pool": {"name": cfg.grid(s.mode).pool, "n_items_in_pool": len(plan.items), "pool_keys_md5": keys_md5(plan.item_keys)},
            "conditions": [{"id": c.id, "a_H": c.a_H, "c0": c.c0, "w": c.w} for c in plan.conditions],
            "endpoint": {"url": cfg.endpoint_url, **({"kind": cfg.endpoint_kind} if cfg.endpoint_kind != "bailian" else {})}, "request": {"temperature": cfg.temperature, "classes": {n: {"max_tokens": c.max_tokens, "extra_body": dict(c.extra_body), **c.bounds_record()} for n, c in cfg.classes.items()}},
            "retry": dataclasses.asdict(cfg.retry), "runtime": {"concurrency": s.concurrency},
            "budget": {"cap_cny": cfg.budget.cap_cny, "reserve_margin": cfg.budget.reserve_margin, "template_margin_tokens": cfg.budget.template_margin_tokens,
                       "measured_factor": cfg.budget.measured_factor, "send_quota_factor": cfg.budget.send_quota_factor, "charge_unknown_at_bound": cfg.budget.charge_unknown_at_bound},
            "state_dir": str(self.sdir), "models": {}, "nmi_source_sha256": source_hash(), **self._extra,
        }

    # ---- plans --------------------------------------------------------------------
    def _new_model_entry(self, model: ModelSpec, item_keys: Sequence[str]) -> None:
        """A model new to this run directory: version 1 of its plan. The key file is written before the manifest names it."""
        file = plan_file_name(model.id, 1)
        write_text_atomic(self.run_dir / file, keys_text(item_keys))
        self.manifest["models"][model.id] = {
            "class": model.cls, "order": model.order, "status": NOT_STARTED,
            "plan": {"version": 1, "stage": self.settings.stage, "n_items": len(item_keys), "file": file, "keys_md5": keys_md5(item_keys)},
        }

    def _extend_plan(self, model: ModelSpec, plan: RequestPlan) -> None:
        """Grow the plan of a model to the first ``len(plan.items)`` questions (the probe becomes the pilot). First the key file of the next version is written (an atomic rename), then
        the pointer in the manifest is replaced atomically. A crash between the two leaves the old pointer and an unused file, which the next extension overwrites."""
        entry = self.manifest["models"][model.id]
        current = entry["plan"]
        if len(plan.items) <= current["n_items"]:
            return
        version = current["version"] + 1
        file = plan_file_name(model.id, version)
        keys = plan.item_keys
        write_text_atomic(self.run_dir / file, keys_text(keys))
        self._fault("plan_keys_written", model=model.id, version=version)
        entry["plan"] = {"version": version, "stage": self.settings.stage, "n_items": len(keys), "file": file, "keys_md5": keys_md5(keys)}
        self._write_manifest()
        self._fault("plan_pointer_swapped", model=model.id, version=version)

    def _check_resume(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / MANIFEST_NAME
        if not path.is_file():
            raise RunRefused(f"--resume {run_dir}: no {MANIFEST_NAME} there")
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("kind") != "api_exp_run" or old.get("mode") != self.settings.mode:
            raise RunRefused(f"--resume {run_dir}: that run is {old.get('kind')!r} in mode {old.get('mode')!r}, this call is mode {self.settings.mode!r}")
        recorded = old.get("input_md5", {})
        changed = sorted(k for k in FROZEN_INPUTS if recorded.get(k) != self.input_md5.get(k))
        if changed:
            raise RunRefused(f"--resume {run_dir}: inputs changed since the run started: {changed} (only the concurrency and the unit prices may change)")
        if [c["id"] for c in old.get("conditions", [])] != [c.id for c in self.base_plan.conditions]:
            raise RunRefused(f"--resume {run_dir}: the conditions of the run are not those of this call")
        for model in self.models:
            entry = old.get("models", {}).get(model.id)
            if entry is None or "plan" not in entry:
                continue
            plan = entry["plan"]
            keys_path = run_dir / plan["file"]
            keys = keys_path.read_text(encoding="ascii").splitlines() if keys_path.is_file() else []
            if keys_md5(keys) != plan["keys_md5"] or len(keys) != plan["n_items"] or keys != self.base_plan.item_keys[: len(keys)]:
                raise RunRefused(f"--resume {run_dir}: the planned items of {model.id} are not the first {plan['n_items']} questions of this pool (planned_items changed)")
        return old

    def _sync_view(self) -> None:
        """Make the per-run view hold every success of this run that the journal has (a crash can leave the view short, never the journal)."""
        n = 0
        for key, payload in self.journal.successes(run=self.run_name):
            if not self.store.has_reply(*key):
                meta = self.journal.keys[key].success_meta or {}
                n += int(self.store.append_reply(view_record(key, payload, meta)))
        if n:
            log(f"{n} successes were in the ledger but not in {STORE_NAME}; copied")

    # ---- the event loop --------------------------------------------------------------------
    async def _main(self) -> None:
        self.stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        handled: list[int] = []
        if self._install_signals:

            def on_signal(sig: int) -> None:
                self.request_stop("interrupted")
                loop.remove_signal_handler(sig)  # a second signal of the same kind acts as it would without us (SIGINT raises KeyboardInterrupt)
                handled.remove(sig)

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, on_signal, sig)
                    handled.append(sig)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass
        try:
            async with ChatClient(self.cfg.endpoint_url, self._api_key, client_retry(self.cfg), concurrency=self.settings.concurrency, transport=self._transport) as client:
                await self._run_models(client)
        except BaseException as error:
            self.stop_reason = self.stop_reason or "error"
            self.manifest["error"] = type(leaf_exception(error)).__name__  # the name of the exception type, never its message
            self._finish("error")
            raise
        finally:
            for sig in handled:
                loop.remove_signal_handler(sig)
        self._finish(None)

    def request_stop(self, reason: str) -> None:
        """Stop sending: sends in flight finish, no request is started, nothing is recorded for a request that was not sent."""
        if self.stop_reason is None:
            self.stop_reason = reason
        if self.stop is not None:
            self.stop.set()
        if self._control is not None:
            self._control.cancel.set()

    def _sendable(self, model_id: str) -> Callable[[str, str], bool]:
        """The predicate for ``RequestPlan.requests(skip)``: True for a request that is NOT to be sent in this session (it has a success, has ended for good, has used its sends, or is
        uncertain and --resend-uncertain was not given)."""
        resend = self.settings.resend_uncertain

        def skip(item_key: str, condition_id: str) -> bool:
            return not self.journal.can_send(model_id, item_key, condition_id, resend_uncertain=resend)

        return skip

    async def _run_models(self, client: ChatClient) -> None:
        cfg, journal = self.cfg, self.journal
        for model in self.models:
            entry = self.manifest["models"][model.id]
            if self.stop.is_set():  # a signal, a refused key or the cap: the models after it are not started
                continue
            plan = self._stage_plan(model)
            cls = cfg.class_of(model.id)
            max_tokens = cls.max_tokens
            sendable = list(plan.requests(self._sendable(model.id)))
            if not sendable:
                self._refresh_entry(model.id, entry)
                self._write_manifest()
                continue
            stage = self.settings.stage
            planned = cfg.stage_new_items(stage) * len(plan.conditions)
            self._quota[model.id] = {"stage": stage, "planned_keys": planned, "factor": cfg.budget.send_quota_factor, "quota": send_quota(planned, cfg.budget.send_quota_factor)}
            left = self._quota[model.id]["quota"] - journal.sends_in_stage(stage, model.id)
            if left <= 0:  # the quota is used up: the model asks nothing of the budget (it must not stop the models after it) and sends nothing
                entry["stopped"] = "send_quota"
                self._refresh_entry(model.id, entry)
                log(f"{model.id}: not started: its {self._quota[model.id]['quota']} sends of the stage {stage} are used up ({len(sendable)} requests are left as they are)")
                self._write_manifest()
                continue
            per_byte, n_measured = journal.measured_tokens_per_byte(model.id)
            reserve = reserve_of(
                self.prices, model.id, (message_bytes(r.message) for r in sendable[:left]), max_tokens, cls.input_margin(cfg.budget.template_margin_tokens),
                cfg.budget.reserve_margin, tokens_per_byte=per_byte, n_measured=n_measured, measured_factor=cfg.budget.measured_factor, measured_completion=estimate_completion(cfg, journal, model.id),
                input_bytes_factor=cls.input_bytes_factor, output_margin_tokens=cls.output_margin_tokens,
            )
            available = self.settings.available_cny
            if available is not None and journal.session_spent_cny() + reserve.reserve_cny > available:  # the money that is there does not carry this model to its end: it is not started at all
                short = journal.session_spent_cny() + reserve.reserve_cny - available
                entry["start_decision"] = {
                    "admitted": False, "reason": "available_money", "session_spent_cny": journal.session_spent_cny(), "reserve_cny": reserve.reserve_cny, "available_cny": available,
                    "shortfall_cny": short, "reserve": reserve.as_record(),
                }
                self._refused.add(model.id)
                entry["status"] = NOT_RUN_BUDGET
                log(
                    f"{model.id}: not started (available_money): this session has spent {journal.session_spent_cny():.2f} + estimate {reserve.reserve_cny:.2f} CNY against the {available:.2f} CNY "
                    f"that are there: short by {short:.2f} CNY. Nothing is sent to it and the session stops here, so that the operator can be asked; the models after it are not started"
                )
                self._write_manifest()
                self.request_stop("available_money")
                continue
            decision = self.budget.admit(
                stage=self.settings.stage, model=model.id, order=model.order, price_version=self.prices.version, spent_cny=journal.spent_cny(), reserve_cny=reserve.reserve_cny,
                cap_cny=cfg.budget.cap_cny, now=iso(now_local()),
            )
            self._admit[model.id] = decision
            entry["start_decision"] = {**decision.as_record(), "reserve": reserve.as_record()}
            if not decision.admitted:
                self._refused.add(model.id)
                entry["status"] = NOT_RUN_BUDGET
                log(
                    f"{model.id}: not started ({decision.reason}): booked {decision.spent_cny:.2f} + estimate {decision.reserve_cny:.2f} CNY against the cap {decision.cap_cny:.0f}"
                    + (f": short by {decision.spent_cny + decision.reserve_cny - decision.cap_cny:.2f} CNY, nothing is sent to it" if decision.reason == "budget" else "")
                    + (f"; the stage {self.settings.stage} stopped at {decision.stop_model}" if decision.reason == "stage_stopped" else "")
                )
                self._write_manifest()
                continue
            self._extend_plan(model, plan)
            log(
                f"{model.id}: {reserve.n_requests} requests to send ({decision.reason}), estimate {reserve.reserve_cny:.2f} CNY ({reserve.basis}), booked so far {decision.spent_cny:.2f} of "
                f"{decision.cap_cny:.0f}"
            )
            entry["status"] = "running"
            self._write_manifest()
            self._ensure_frozen()  # the record is on disk before the first send of the experiment
            await self._send_model(client, model, plan.requests(self._sendable(model.id)))
            self._refresh_entry(model.id, entry)
            if self._control is not None and self._control.reason:
                entry["stopped"] = self._control.reason
            self._write_manifest()

    async def _send_model(self, client: ChatClient, model: ModelSpec, todo: Iterator[RenderedRequest]) -> None:
        n_workers = self.settings.concurrency
        queue: asyncio.Queue[RenderedRequest | None] = asyncio.Queue(maxsize=2 * n_workers)
        self._control = ctl = ModelControl()
        self._consecutive_ended = 0
        if self.stop.is_set():
            ctl.cancel.set()

        async def producer() -> None:
            for request in todo:
                if ctl.cancel.is_set():
                    break
                await queue.put(request)
            for _ in range(n_workers):
                await queue.put(None)

        async def worker() -> None:
            while True:
                request = await queue.get()
                if request is None:
                    return
                if ctl.cancel.is_set():
                    continue  # drain: nothing more is sent to this model
                await self._run_key(client, model, request, ctl)

        async with asyncio.TaskGroup() as group:
            group.create_task(producer())
            for _ in range(n_workers):
                group.create_task(worker())
        self._progress(model.id, force=True)

    async def _wait(self, delay: float, cancel: asyncio.Event) -> None:
        """The wait between two sends of one request; it ends at once if the model is cancelled (with the default sleep)."""
        if cancel.is_set():
            return
        if self._sleep is None:
            try:
                await asyncio.wait_for(cancel.wait(), timeout=delay)
            except TimeoutError:
                pass
            return
        await self._sleep(delay)

    def _ceiling(self, model_id: str) -> PriceCeiling | None:
        """The ceiling of the price (``provider.max_price``) of the requests to ``model_id``, from the price file of this session and the margin of the batch; None for a batch without a margin."""
        if self.cfg.price_margin is None:
            return None
        if model_id not in self._ceilings:
            price = self.prices.price(model_id)
            self._ceilings[model_id] = price_ceiling(price.input_per_million, price.output_per_million, self.cfg.price_margin)
        return self._ceilings[model_id]

    async def _run_key(self, client: ChatClient, model: ModelSpec, request: RenderedRequest, ctl: ModelControl) -> None:
        """All the sends of one request in this session, each one written ahead and counted."""
        journal = self.journal
        cfg = self.cfg
        cls_spec = cfg.class_of(model.id)
        key = (model.id, request.item_key, request.condition.id)
        ceiling = self._ceiling(model.id)  # supp4, supp5: the ceiling of the price (provider.max_price) of the request, which the reserve of the send is made of too
        body = request_body(cfg, model, request.message, ceiling)  # (a model asked in a cell of its own: its id, the key of everything the program keeps, carries the cell, and the request names the model; nmi.api_exp.final_request)
        n_bytes = message_bytes(request.message)
        bound_prompt = cls_spec.input_bound(n_bytes, cfg.budget.template_margin_tokens)  # the provable bound of this send: the first batch's unless the class declares its own
        bound_completion = cls_spec.output_bound()
        allow_uncertain = self.settings.resend_uncertain and journal.state(*key) == "uncertain"
        failures = 0
        while True:
            if not journal.can_send(*key, resend_uncertain=allow_uncertain) or ctl.cancel.is_set():
                return  # it has a reply, ended for good, used its sends, or is uncertain and was not asked for
            try:
                attempt_id = journal.write_attempt(
                    run=self.run_name, stage=self.settings.stage, model=model.id, item_key=request.item_key, condition=request.condition.id, message_bytes=n_bytes,
                    bound_prompt=bound_prompt, bound_completion=bound_completion, now=iso(now_local()), allow_uncertain=allow_uncertain,
                    stage_quota=self._quota[model.id]["quota"] if model.id in self._quota else None, price_margin=cfg.price_margin, ceiling=ceiling,
                )
            except CapReached:
                self.request_stop("budget_cap")
                return
            except SendQuotaReached:  # this model has made all the sends its quota allows in this stage: nothing more goes to it; the others go on
                if ctl.reason is None:
                    ctl.reason = "send_quota"
                    log(f"{model.id}: its {self._quota[model.id]['quota']} sends of the stage {self.settings.stage} are used up: no more requests are sent to this model, the rest stay as they are")
                ctl.cancel.set()
                return
            except BoundViolationOpen:  # defence: a violation is open in the ledger, so nothing is written (the run has stopped for it already, or must)
                self.request_stop("bound_violated")
                return
            allow_uncertain = False  # only the first send after an uncertain attempt is the one that was asked for
            self._fault("after_attempt_record", model=model.id, item_key=request.item_key, condition=request.condition.id, n=journal.n_attempts(*key))
            result = await client.send_once(body)
            self._fault("after_send", model=model.id, item_key=request.item_key, condition=request.condition.id, n=journal.n_attempts(*key))
            now = iso(now_local())
            if result.cls == SUCCESS:
                parsed = parse_chat_body(result.body, cfg.returned_names(model.id))
                payload = reply_payload(request, parsed, cls_spec.max_tokens, journal.n_attempts(*key))
                reported, identity = None, None
                if cfg.endpoint_kind == "openrouter":  # the service that answered, the price it reports and its thinking text: kept under fixed rules, see nmi.api_exp.openrouter
                    facts_body = result.body
                    if cfg.price_margin is not None and isinstance(facts_body, Mapping) and isinstance(facts_body.get("usage"), Mapping):  # supp4, supp5: only nmi.api_exp.price_rules reads the price
                        facts_body = {**facts_body, "usage": {k: v for k, v in facts_body["usage"].items() if k != "cost"}}
                    facts = read_router_facts(facts_body, cfg.provider_names(model.id))
                    identity = apply_router_facts(payload, facts)
                    if cfg.price_margin is None:
                        reported = reported_cny(payload.get("reported_cost_usd"))  # booked if it is larger than the tokens at the price file
                    else:  # supp4, supp5: the price is read from the number the client decoded, by the rules of nmi.api_exp.price_rules (no text-length rule), and the reply keeps that text
                        usage_block = result.body.get("usage") if isinstance(result.body, Mapping) else None
                        price = reported_price(usage_block.get("cost") if isinstance(usage_block, Mapping) else None)
                        payload["reported_cost_usd"] = None if price is None else price.text
                        reported = None if price is None else price.cny
                    if identity is not None:  # not the pinned service or not the requested model: booked, counted as a send, kept and never sent again, but not read as a decision
                        self._stats[identity] += 1
                        if self._stats[identity] == 1:
                            log(
                                f"{model.id}: a reply is not from the pinned service {cfg.pinned_provider(model.id)} and the requested model ({identity}): it is booked and kept, never sent again, "
                                "and does not count as an answer or a request for help (an identity failure)"
                            )
                journal.write_result_success(attempt_id, status=result.status, reply=payload, usage=parsed.usage(), latency_s=result.latency_s, now=now, reported_cny=reported)
                self._fault("after_result_record", model=model.id, item_key=request.item_key, condition=request.condition.id)
                meta = journal.keys[key].success_meta or {}
                self.store.append_reply(view_record(key, payload, meta))
                journal.index_success(key, self.run_name)
                self._stats[payload["decision"]] += 1
                if identity is None:
                    self._consecutive_ended = 0
                else:  # a run of replies that are not from the pinned service or the requested model (a spelling that does not fit, a service that is routed elsewhere) wastes the model: stop it early
                    self._consecutive_ended += 1
                    limit_n = cfg.retry.stop_after_consecutive_exhausted
                    if self._consecutive_ended >= limit_n and ctl.reason is None:
                        ctl.reason = "consecutive_identity_failures"
                        ctl.cancel.set()
                        log(f"{model.id}: {self._consecutive_ended} replies in a row were not from the pinned service and the requested model: no more requests are sent to this model in this session")
                violation = journal.violations.get(attempt_id)
                if violation is not None:
                    # the premise of the cap is broken (the input is at most the bound of the send, the bytes plus the margin of its class; the output is at most the output bound; a batch with a price margin
                    # books at most the reserve of the send): booked as it came, nothing more is sent, and the ledger now holds the lockout record that keeps every later session out until an operator's disposition
                    tokens, money = violation.prompt is not None or violation.completion is not None, violation.cost is not None
                    what = "used more tokens than the provable bound of the send" if not money else ("cost more than the reserve of the send" if not tokens else "used more tokens than the provable bound of the send and cost more than its reserve")
                    log(
                        f"{model.id}: a reply {what} ({violation.describe()}); the run stops, and so does every later session until the operator's "
                        f"disposition (lockout record {attempt_id})"
                    )
                    self._stats["bound_violations"] += 1
                    self.request_stop("bound_violated")
                limit = self.settings.max_session_cny
                if limit is not None and journal.session_spent_cny() >= limit:
                    self.request_stop("session_limit")
                self._progress(model.id)
                return
            journal.write_result_failure(attempt_id, cls=result.cls, status=result.status, exc=result.exc, latency_s=result.latency_s, now=now)
            self._stats[f"failed_{result.cls}"] += 1
            if result.cls == AUTH_REFUSED:
                if result.status == PAYMENT_REQUIRED:
                    log(f"{model.id}: HTTP 402 (payment required): the credits of the account are used up. The run stops; the requests refused are sent again when it is resumed after the account has been topped up")
                self.manifest["error"] = f"HTTP {result.status} ({AUTH_REFUSED})"  # the status and a fixed class name, nothing the service wrote
                self.request_stop("fatal_api_error")
                return
            if journal.state(*key) in ENDED_STATES:  # a 4xx that ends the request, or the last of its sends
                self._consecutive_ended += 1
                limit_n = cfg.retry.stop_after_consecutive_exhausted
                if self._consecutive_ended >= limit_n and ctl.reason is None:
                    ctl.reason = "consecutive_ended_without_reply"
                    ctl.cancel.set()
                    log(f"{model.id}: {self._consecutive_ended} requests in a row ended without a reply: no more requests are sent to this model in this session")
                self._progress(model.id)
                return
            failures += 1
            await self._wait(backoff_delay(cfg.retry, failures, self._rng), ctl.cancel)

    def _progress(self, model_id: str, *, force: bool = False) -> None:
        self._done_since_progress += 1
        now = time.monotonic()
        if not force and self._done_since_progress < self.settings.progress_every and now - self._last_progress < self.settings.progress_seconds:
            return
        self._done_since_progress, self._last_progress = 0, now
        s = self._stats
        failed = sum(v for k, v in s.items() if k.startswith("failed_"))
        log(
            f"{model_id}: successes {self.journal.counters(self.run_name, model_id)['success']} (this session: answer {s['answer']}, ask {s['ask']}, format_failure {s['format_failure']}, "
            f"failed sends {failed}); session {self.journal.session_spent_cny():.2f} CNY, ledger {self.journal.spent_cny():.2f} of {self.cfg.budget.cap_cny:.0f}"
        )

    # ---- summaries and finishing -----------------------------------------------------------
    def _refresh_entry(self, model_id: str, entry: dict[str, Any]) -> None:
        """The counts, tokens, returned model names and cost of one model in this run, from the journal: the money as it was booked, and the tokens worked out again at the current prices."""
        journal = self.journal
        plan = entry["plan"]
        keys = [(k, c.id) for k in self.base_plan.item_keys[: plan["n_items"]] for c in self.base_plan.conditions]
        counts = count_states(journal, model_id, keys)
        decisions: Counter[str] = Counter()
        returned: Counter[str] = Counter()
        providers: Counter[str] = Counter()
        reported_usd, n_reported = Decimal(0), 0
        for _key, payload in journal.successes(run=self.run_name, model=model_id):
            decisions[payload.get("decision")] += 1
            returned[returned_model_key(payload)] += 1
            if self.cfg.endpoint_kind == "openrouter":
                providers[provider_key(payload)] += 1
                if payload.get("reported_cost_usd") is not None:
                    reported_usd, n_reported = reported_usd + Decimal(payload["reported_cost_usd"]), n_reported + 1
        tokens = journal.tokens_by_model(run=self.run_name).get(model_id, {"actual_prompt": 0, "actual_completion": 0, "bound_prompt": 0, "bound_completion": 0})
        c = journal.counters(self.run_name, model_id)
        status = NOT_RUN_BUDGET if model_id in self._refused else status_of(counts, len(keys))
        entry.update({
            "status": status, "n_planned": len(keys), "states": {s: counts.get(s, 0) for s in SHOWN_STATES},
            "n_attempts": c["attempts"], "n_results": c["results"], "decisions": {d: decisions.get(d, 0) for d in DECISION_VALUES}, "returned_models": dict(sorted(returned.items())),
            "tokens": tokens, "cost_cny": journal.spent_cny(run=self.run_name, model=model_id), "cost_cny_at_current_prices": journal.repriced_cny(self.prices, run=self.run_name, model=model_id),
            "price_versions_booked": journal.price_versions(run=self.run_name, model=model_id), "n_charged_by_bound": c["charged_by_bound"], "n_bound_violations": c["bound_violations"],
            "n_partial_usage": c["partial_usage"],
        })
        if self.cfg.endpoint_kind == "openrouter":
            entry["returned_providers"] = dict(sorted(providers.items()))
            entry["reported_cost_usd"] = {"sum": decimal_text(reported_usd), "n_replies": n_reported}  # what OpenRouter says the calls cost; the money booked is the tokens at the price file
        info = self._quota.get(model_id)
        if info is not None:
            sent = journal.sends_in_stage(info["stage"], model_id)
            entry["send_quota"] = {**info, "sent": sent, "left": max(0, info["quota"] - sent)}

    def _exit_code(self) -> int:
        if self.stop_reason is not None:
            return EXIT_ABORTED
        statuses = [self.manifest["models"][m.id]["status"] for m in self.models]
        return EXIT_OK if all(s == COMPLETE for s in statuses) and not self._refused else EXIT_INCOMPLETE

    def _finish(self, forced: str | None) -> None:
        m = self.manifest
        for model_id, entry in m["models"].items():  # ALL models of the run, not only the ones this session named, so that every count is current
            if "plan" in entry:
                self._refresh_entry(model_id, entry)
        code = self._exit_code()
        status = forced or ("ok" if code == EXIT_OK else ("incomplete" if code == EXIT_INCOMPLETE else "aborted"))
        end = iso(now_local())
        m["exit_status"], m["stop_reason"], m["end_time"] = status, self.stop_reason, end
        m["sessions"][-1].update({
            "end_time": end, "exit_status": status, "stop_reason": self.stop_reason, "session_cny": self.journal.session_spent_cny(), "session_stats": dict(self._stats),
            "error": m.get("error"), "decisions": {k: v.as_record() for k, v in self._admit.items()},
        })
        self._write_manifest()

    def _write_manifest(self) -> None:
        m, journal = self.manifest, self.journal
        for model_id, entry in m["models"].items():
            if "cost_cny" in entry:  # the money as booked, and, beside it, the tokens at the current prices
                entry["cost_cny"] = journal.spent_cny(run=self.run_name, model=model_id)
                entry["cost_cny_at_current_prices"] = journal.repriced_cny(self.prices, run=self.run_name, model=model_id)
                entry["price_versions_booked"] = journal.price_versions(run=self.run_name, model=model_id)
        run_models = [e for e in m["models"].values() if "tokens" in e]
        m["tokens_total"] = {
            "prompt": sum(e["tokens"]["actual_prompt"] + e["tokens"]["bound_prompt"] for e in run_models), "completion": sum(e["tokens"]["actual_completion"] + e["tokens"]["bound_completion"] for e in run_models),
        }
        m["cost_cny_run"] = journal.spent_cny(run=self.run_name)
        m["cost_cny_run_at_current_prices"] = journal.repriced_cny(self.prices, run=self.run_name)
        uncertain = [a for a in journal.uncertain() if a.run == self.run_name]
        m["ledger"] = {
            "spent_cny": journal.spent_cny(), "spent_cny_at_current_prices": journal.repriced_cny(self.prices), "cap_cny": self.cfg.budget.cap_cny,
            "remaining_cny": self.cfg.budget.cap_cny - journal.spent_cny(), "spent_by_model": journal.spent_by_model(), "price": self.prices.identity(),
            "price_versions_booked": journal.price_versions(), "n_uncertain_attempts_in_run": len(uncertain),
            "note": "spent_cny is the money as it was booked, each send at the price version of its day (the cap is checked against it); spent_cny_at_current_prices works all the tokens out "
                    "again at the price version in force now; the ledger is the file ledger.jsonl of the state directory",
        }
        write_json_atomic(self.run_dir / MANIFEST_NAME, m)
