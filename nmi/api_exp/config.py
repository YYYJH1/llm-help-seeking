"""Loader and validator of ``configs/api_exp_v1.yaml``.

``load_api_config`` reads the file, refuses a repeated key and a key it does not know, checks every value (types, ranges, the pools, the models and their classes, the stages, the budget
rule) and returns a read-only ``ApiExpConfig`` that carries two md5 values:

* ``md5``: the md5 of the file as it is;
* ``frozen_md5``: the md5 of the parsed configuration WITHOUT the block ``runtime``. The number of concurrent requests and the unit prices may change between sessions. The unit prices
  are not in this file at all (``configs/api_exp_prices.yaml``, see ``nmi.api_exp.prices``), so neither md5 includes them. A resumed run must have the same ``frozen_md5`` as the run it
  continues, and so must every session of the experiment (``nmi.api_exp.state.Freeze``).

The pools are fixed in the code, not only in the file: the pilot reads ``mmlu_redux_cal`` and nothing else, and the pool of the full run is the holdout pool, which no code of this package
reads (``FULL_POOL`` is a name, not a path to the data). Several values are fixed by the rules and the loader refuses any other value, so that another file cannot loosen them: the cap
is at most 500 CNY, a request is sent at most 8 times, a (stage, model) makes at most 1.1 times its planned keys in sends, the reserve margin is 10%, the state directory and the
repository that may send real requests are the two absolute paths below, and the reading rule (block ``reading``) is the one in ``nmi.api_exp.reader``.

A SUPPLEMENTARY BATCH is a further configuration file for models added to the experiment after the first batch (for example ``api_exp_v1_supp1.yaml``, not distributed). It has the optional top-level block ``batch``
(``name``, ``base``, ``first_order``) and repeats every other block of the first batch's file. The loader reads the file named by ``base`` (in the directory of this file) and refuses the
configuration unless every top-level block other than ``version``, ``batch``, ``models`` and ``runtime`` is, item by item, the same JSON as in the base. The models must be other models than
the base's (an id or an alias belongs to one batch), and their run order begins at ``first_order``, after the base's, so that a stop of the budget at a model of this batch does not stop a
model of the first batch. A file ``api_exp_batches.yaml`` next to the configuration files lists the batches; ``load_api_config`` then adds the models of all of them to
``ApiExpConfig.experiment_models``, which the price file and the release list are checked against (the ledger, the price file and the release list are shared by every batch).

A batch of the endpoint kind ``openrouter`` (supp3, not distributed) declares two more blocks of its own in ``batch.own``, ``endpoint`` and ``request`` (see ``_openrouter_classes``
for the request classes). The batches supp4 and supp5 have one more key in their block ``endpoint``, ``price_margin`` (2.0; ``PRICE_MARGIN_BATCHES``): the reserve of every send is the cost of
its provable bound at the price file times it (``nmi.api_exp.price_rules``). No other configuration has the key, and the loader refuses any other value. The key is in the frozen md5 of the
configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from nmi.api_exp.reader import ANSWER_DIGITS, ASK_DIGIT
from nmi.paths import REPO_ROOT
from nmi.provenance import bytes_md5

PILOT_POOLS = ("mmlu_redux_cal",)  # the development pool: the only pool this package reads
FULL_POOL = "mmlu_redux_holdout"  # the held-out pool (2,291 questions): named here, never read
CLASS_NAMES = ("thinking_off",)  # one class: every model is asked with thinking switched off; the request code looks its parameters up by model
FROZEN_EXCLUDES = ("runtime",)  # the concurrency may change between sessions (the unit prices are in a file of their own)
STAGE_NAMES = ("probe", "pilot")  # the two stages of the pilot: the small trial, then the whole pool in the same run directory
RESERVE_MARGIN = 0.10  # the reserve of a model is its remaining requests at their estimates, times 1 + this
CAP_CEILING_CNY = 500.0  # upper limit of the spending cap: no configuration may raise it
MAX_ATTEMPTS_CEILING = 8  # a request is sent at most 8 times in all
SEND_QUOTA_FACTOR_CEILING = 1.1  # the sends a (stage, model) may make in all sessions are its planned keys times at most this, rounded up
TEMPLATE_MARGIN_MIN = 16  # tokens: the provable bound of the input tokens is the UTF-8 bytes of the message plus this margin for the chat template
FIXED_CODE_ROOT = "/path/to/NMI"  # absolute path of the one checkout that may send real requests; set it, and state.code_root of every API configuration, to your checkout
FIXED_STATE_DIR = "/path/to/NMI/results/api_exp/_state"  # the ledger, the index, the budget state, the freeze record and the lock; the runs are in its parent
TOP_KEYS = (
    "version", "protocol", "endpoint", "local", "prompt", "request", "reading", "models", "retry", "prices", "budget", "state", "runtime", "pilot", "full", "analysis",
)
OPTIONAL_TOP_KEYS = ("batch",)  # present in the configuration of a supplementary batch only
BATCH_OWN_KEYS = ("version", "batch", "models", "runtime")  # the top-level keys in which a supplementary batch may differ from the configuration of the first batch
BATCH_OWN_OPTIONAL = ("endpoint", "request", "pilot")  # the blocks a batch of the kind openrouter may declare in ``batch.own`` as its own (``pilot`` only a batch of ``OPENROUTER_CELL_BATCHES``); no other block can be declared its own
ENDPOINT_KINDS = ("bailian", "openrouter")  # ``endpoint.kind``; the first batch and the supplementary batches 1 and 2 are bailian (the default)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"  # the only address an openrouter configuration may name: the key of OpenRouter goes nowhere else
KEY_FILE_WORD = "openrouter"  # the key file of a configuration of the kind openrouter has this word in its name, and the key file of any other configuration does not: a key does not go to the other service
OPENROUTER_MAX_TOKENS = {"off": 4, "minimal": 256}  # group A (thinking switched off, or a model without the parameter) and group B (the lowest thinking): the output cap of the request
OPENROUTER_BATCH_NAMES = ("supp3", "supp4", "supp5")  # the batches that may be of the kind openrouter, and so may declare blocks of their own (``batch.own``); no other file is, whatever it says
OPENROUTER_CELL_BATCHES = {"supp5": 0.95}  # the batches that ask in another cell than the common one (a_H 0.80, c0 0): name -> the a_H of their pilot; the rest of the grid is the first batch's
PRICE_MARGIN_BATCHES = {"supp4": 2.0, "supp5": 2.0}  # the batches that reserve the list price times this margin for every send (``endpoint.price_margin``; nmi/api_exp/price_rules.py): name -> the margin; no other batch has the key
_CELL_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*@a[01]\.\d\d$")  # vendor/name@a0.95: a model asked in a cell of its own (its ``wire`` name, a tag for the cell)
OPENROUTER_OUTPUT_MARGIN = 64  # tokens added to the output cap of every class of the kind openrouter for the PROVABLE output bound of a send (68 in group A, 320 in group B)
OPENROUTER_WIDE_INPUT_BOUND = (3, 2048)  # (bytes factor, margin tokens) of the one class whose model reports far more input tokens than the message has bytes: the input bound is factor * bytes + margin
BATCH_REGISTRY_NAME = "api_exp_batches.yaml"  # in the directory of the configuration files: the first batch's file, then the files of the supplementary batches
_BATCH_NAME = re.compile(r"[a-z][a-z0-9]{0,15}")  # part of the version and of the name of the freeze record (freeze_<name>.json)
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_OPENROUTER_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*$")  # vendor/name, the form of an OpenRouter model slug
_CLASS_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_RESERVED_BODY_KEYS = frozenset({"model", "messages", "temperature", "max_tokens", "stream", "n", "logprobs", "top_logprobs"})


class ApiConfigError(ValueError):
    """The configuration file is malformed or inconsistent."""


# ---------------------------------------------------------------------------------------
# YAML without silent key overrides
# ---------------------------------------------------------------------------------------

class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key (PyYAML keeps the last one)."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ApiConfigError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


# ---------------------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------------------

def _mapping(value: Any, path: str, required: Sequence[str] = (), optional: Sequence[str] = ()) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ApiConfigError(f"{path}: must be a mapping")
    missing = [k for k in required if k not in value]
    unknown = [k for k in value if k not in required and k not in optional]
    if missing:
        raise ApiConfigError(f"{path}: missing keys {missing}")
    if unknown:
        raise ApiConfigError(f"{path}: unknown keys {unknown}")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value: Any, path: str, *, lo: float | None = None, hi: float | None = None, positive: bool = False, integer: bool = False) -> float:
    if not _is_number(value) or (integer and not isinstance(value, int)):
        raise ApiConfigError(f"{path}: must be {'an integer' if integer else 'a finite number'}, got {value!r}")
    if positive and not value > 0:
        raise ApiConfigError(f"{path}: must be positive, got {value!r}")
    if lo is not None and value < lo:
        raise ApiConfigError(f"{path}: must be at least {lo}, got {value!r}")
    if hi is not None and value > hi:
        raise ApiConfigError(f"{path}: must be at most {hi}, got {value!r}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiConfigError(f"{path}: must be a non-empty string")
    return value


def _two_decimals(value: Any, path: str, *, upper: float | None = None) -> float:
    x = _number(value, path, lo=0.0, hi=upper)
    if abs(round(x, 2) - x) > 1e-9:
        raise ApiConfigError(f"{path}: {value!r} cannot be shown with two decimals without changing it")
    return float(x)


def _levels(value: Any, path: str, *, upper: float | None = None) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ApiConfigError(f"{path}: must be a non-empty list")
    out = tuple(_two_decimals(v, f"{path}[{i}]", upper=upper) for i, v in enumerate(value))
    if len(set(out)) != len(out):
        raise ApiConfigError(f"{path}: repeated value")
    return out


# ---------------------------------------------------------------------------------------
# the parts
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassSpec:
    """A class of models: the request parameters. There is one class, ``thinking_off``, and every model names it; the request code (``nmi.api_exp.client``) looks the
    parameters up through ``ApiExpConfig.class_of``."""

    name: str
    max_tokens: int
    extra_body: Mapping[str, Any]
    provider_aliases: tuple[str, ...] = ()  # the other names under which the pinned service may name itself in a reply (the kind openrouter); the reply is not an anomaly if it carries one of them
    output_margin_tokens: int = 0  # the PROVABLE output bound of a send is ``max_tokens`` plus this (0: the output cap itself, as in the first batch); what is sent as ``max_tokens`` is unchanged
    input_bytes_factor: int = 1  # the PROVABLE input bound of a send is this times the UTF-8 bytes of the message ...
    input_margin_tokens: int | None = None  # ... plus this (None: ``budget.template_margin_tokens``, as in the first batch)

    def output_bound(self) -> int:
        """The output tokens one send of this class cannot exceed: ``max_tokens`` plus the margin of the class. Thinking tokens are output tokens."""
        return int(self.max_tokens) + int(self.output_margin_tokens)

    def input_margin(self, template_margin_tokens: int) -> int:
        """The margin of the input bound: the class's own, or ``template_margin_tokens`` (``budget.template_margin_tokens``) when it has none."""
        return int(template_margin_tokens) if self.input_margin_tokens is None else int(self.input_margin_tokens)

    def input_bound(self, n_bytes: int, template_margin_tokens: int) -> int:
        """The input tokens one send of this class cannot exceed: the bytes of the message times the factor of the class plus its margin."""
        return int(self.input_bytes_factor) * int(n_bytes) + self.input_margin(template_margin_tokens)

    def bounds_record(self) -> dict[str, int]:
        """The bound fields of the class that differ from the first batch's, for the manifest and the dry run; empty for a class that has none (every class of the first batch)."""
        out: dict[str, int] = {}
        if self.output_margin_tokens != 0:
            out["output_margin_tokens"] = int(self.output_margin_tokens)
        if self.input_bytes_factor != 1:
            out["input_bytes_factor"] = int(self.input_bytes_factor)
        if self.input_margin_tokens is not None:
            out["input_margin_tokens"] = int(self.input_margin_tokens)
        return out


@dataclass(frozen=True)
class ModelSpec:
    id: str
    cls: str
    order: int  # position in the run order
    aliases: tuple[str, ...] = ()  # other names under which the service may return this model; the model field of a response is stored as it came only if it is the id or one of these
    wire: str | None = None  # the model as the request names it, for a model asked in a cell of its own (the kind openrouter, ``OPENROUTER_CELL_BATCHES``): the id is then ``wire@a<a_H>``, the key of everything the
    # program keeps for the model (ledger, quotas, run directory, budget approvals, prices, release list), so that the same model asked in two cells is two models there; None: the id is what is sent

    def returned_names(self) -> frozenset[str]:
        """The model names a response may carry for this model: what the request names (the id, or the wire name of a model asked in another cell) and the aliases."""
        return frozenset((self.wire or self.id, *self.aliases))

    def state_names(self) -> frozenset[str]:
        """The names that belong to this model alone in the whole experiment: the id and the aliases; for a model asked in another cell only the id (its wire name and aliases are those of the model of
        the common cell as well)."""
        return frozenset((self.id,)) if self.wire is not None else frozenset((self.id, *self.aliases))


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int  # sends of one request (model, question, condition) in all, over every session: the first and the retries
    backoff_base_s: float
    backoff_factor: float
    backoff_max_s: float
    jitter: float
    timeout_s: float
    abort_statuses: tuple[int, ...]
    stop_after_consecutive_exhausted: int  # a model gets no more requests in a session after this many requests in a row that ended for good without a reply


@dataclass(frozen=True)
class Budget:
    cap_cny: float
    reserve_margin: float
    template_margin_tokens: int  # the provable input bound of a send is the UTF-8 bytes of the message plus this many tokens
    measured_factor: float  # the estimate before a model starts uses the largest measured tokens per byte of that model times this
    send_quota_factor: float  # a (stage, model) makes at most its planned keys times this many sends, over every session, rounded up (a count of requests, not money)
    charge_unknown_at_bound: bool  # a send that ended in a timeout, a damaged body or a 5xx (it may have been processed) is charged its upper bound


@dataclass(frozen=True)
class StageSpec:
    """How many questions of the pool a stage takes (the first ones, in pool order); the same for every model."""

    items: int


@dataclass(frozen=True)
class GridSpec:
    """The conditions of a mode, in the order of ``itertools.product(a_H, c0, w)`` (the order of ``scripts/run_decisions.grid_conditions``)."""

    pool: str
    items_expected: int
    a_H: tuple[float, ...]
    c0: tuple[float, ...]
    w: tuple[float, ...]
    stages: Mapping[str, StageSpec] = field(default_factory=dict)  # the pilot only

    @property
    def n_conditions(self) -> int:
        return len(self.a_H) * len(self.c0) * len(self.w)


@dataclass(frozen=True)
class PromptSpec:
    file: Path
    paraphrase: str
    block: str
    framing: str
    layout: str


@dataclass(frozen=True)
class AnalysisSpec:
    resamples: int
    seed: int
    delta_eq: float
    dominated_lower: float
    min_valid_share: float


@dataclass(frozen=True)
class BatchSpec:
    """The block ``batch`` of the configuration of a supplementary batch: models added to the experiment after the first batch. The batch has its own configuration file and its
    own freeze record (``freeze_<name>.json``); the ledger, the index of successes, the budget state, the process lock, the cap, the price file and the release list are those of the first batch."""

    name: str  # e.g. supp1: the suffix of the version (api_exp_v1_supp1) and of the freeze record
    base: Path  # the configuration file of the first batch, in the directory of this file
    base_md5: str  # its md5 as the loader read it
    base_frozen_md5: str  # its frozen md5 (the md5 of the configuration without the block runtime) as the loader parsed it: what the first batch's freeze record must hold (``first_batch_problem``)
    first_order: int  # the run order of the first model of this batch (the models of the first batch come before)
    own: tuple[str, ...] = ()  # the blocks, besides ``BATCH_OWN_KEYS``, that this batch has of its own (``BATCH_OWN_OPTIONAL``); empty for a batch that is the first batch with other models


@dataclass(frozen=True)
class ApiExpConfig:
    """A validated, read-only configuration of the API experiment."""

    path: Path
    md5: str
    frozen_md5: str
    version: str
    protocol_file: Path
    endpoint_url: str
    key_file: Path
    local_config: Path
    split_dir: Path
    prompt: PromptSpec
    temperature: float
    classes: Mapping[str, ClassSpec]
    models: tuple[ModelSpec, ...]
    retry: RetryPolicy
    prices_file: Path
    budget: Budget
    state_dir: Path
    code_root: Path
    concurrency: int
    pilot: GridSpec
    full: GridSpec
    analysis: AnalysisSpec
    raw: Mapping[str, Any] = field(repr=False)
    batch: BatchSpec | None = None  # None: the first batch
    experiment_models: tuple[str, ...] = ()  # the models of every batch (see ``experiment_model_ids``); empty: this configuration alone
    endpoint_kind: str = "bailian"  # ``endpoint.kind``: which service the requests go to (bailian, or openrouter with its own request classes, pinned services and recorded replies)
    price_margin: float | None = None  # ``endpoint.price_margin``: the batches supp4 and supp5 reserve the list price times this for every send and read the reported price by the rules of nmi/api_exp/price_rules.py; None for the others

    @property
    def cell_batch(self) -> bool:
        """A batch that asks in a cell of its own: its block ``pilot`` is its own (the a_H of ``OPENROUTER_CELL_BATCHES``), so the messages it renders are not the first batch's."""
        return self.batch is not None and "pilot" in self.batch.own

    def model_ids(self) -> tuple[str, ...]:
        """The models of THIS batch: the ones ``--model`` can name and a run of this configuration can ask."""
        return tuple(m.id for m in self.models)

    def experiment_model_ids(self) -> tuple[str, ...]:
        """The models of every batch of the experiment. The batches share the ledger, the price file and the release list, so the price file has an entry for each of these models and the
        release list may name each of them; a batch asks only its own (``model_ids``)."""
        return self.experiment_models or self.model_ids()

    def model(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)

    def class_of(self, model_id: str) -> ClassSpec:
        return self.classes[self.model(model_id).cls]

    def pinned_provider(self, model_id: str) -> str | None:
        """The one service that the request of this model pins (``extra_body.provider.order``), or None when the class pins none (every class of the kind bailian)."""
        return pinned_provider_of(self.class_of(model_id).extra_body)

    def provider_names(self, model_id: str) -> frozenset[str]:
        """The names under which a reply may name the service of this model for the reply not to be an anomaly: the pinned name and the ``provider_aliases`` of the class."""
        pinned = self.pinned_provider(model_id)
        return frozenset(() if pinned is None else (pinned, *self.class_of(model_id).provider_aliases))

    def returned_names(self, model_id: str) -> frozenset[str]:
        """The model names a response may carry for this model for them to be stored as they came."""
        return self.model(model_id).returned_names()

    def stage_items(self, stage: str) -> int:
        """The number of questions (the first ones of the pool) that ``stage`` takes, for every model."""
        if stage not in self.pilot.stages:
            raise ApiConfigError(f"unknown stage {stage!r}; the stages are {list(STAGE_NAMES)}")
        return self.pilot.stages[stage].items

    def stage_new_items(self, stage: str) -> int:
        """The questions that ``stage`` adds to the plan of a model: all of them for the probe, the rest for the pilot (the pilot grows the plan of the probe). The quota of sends of a stage is
        counted on the keys it adds."""
        items = self.stage_items(stage)
        position = STAGE_NAMES.index(stage)
        before = self.stage_items(STAGE_NAMES[position - 1]) if position > 0 else 0
        return max(0, items - before)

    def grid(self, mode: str) -> GridSpec:
        if mode == "pilot":
            return self.pilot
        if mode == "full":
            return self.full
        raise ApiConfigError(f"unknown mode {mode!r}")

    def select_models(self, names: Sequence[str]) -> list[ModelSpec]:
        """The models asked for, in the RUN ORDER of the configuration whatever order they were named in; ``["all"]`` is every model."""
        if list(names) == ["all"]:
            return list(self.models)
        unknown = [n for n in names if n not in self.model_ids()]
        if unknown:
            raise ApiConfigError(f"unknown model(s) {unknown}; the configuration lists {list(self.model_ids())}")
        if len(set(names)) != len(names):
            raise ApiConfigError("a model is named twice")
        return [m for m in self.models if m.id in set(names)]


# ---------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------

def _anchored(path: str | Path, root: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def frozen_part(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if k not in FROZEN_EXCLUDES}


def frozen_md5_of(raw: Mapping[str, Any]) -> str:
    text = json.dumps(frozen_part(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.md5(text.encode("ascii"), usedforsecurity=False).hexdigest()


def _stages(value: Any, items_expected: int) -> dict[str, StageSpec]:
    block = _mapping(value, "pilot.stages", STAGE_NAMES)
    out: dict[str, StageSpec] = {}
    for name in STAGE_NAMES:
        spec = _mapping(block[name], f"pilot.stages.{name}", ("items",))
        out[name] = StageSpec(items=int(_number(spec["items"], f"pilot.stages.{name}.items", lo=1, hi=items_expected, integer=True)))
    if out["pilot"] != StageSpec(items_expected):
        raise ApiConfigError(f"pilot.stages.pilot: the stage pilot takes the whole pool ({items_expected} questions) for every model")
    return out


def _grid(value: Any, path: str, pools: Sequence[str], *, stages: bool = False) -> GridSpec:
    block = _mapping(value, path, ("pool", "items_expected", "a_H", "c0", "w"), ("stages",) if stages else ())
    pool = _string(block["pool"], f"{path}.pool")
    if pool not in pools:
        raise ApiConfigError(f"{path}.pool: {pool!r} is not allowed here; the only pool is {list(pools)}")
    items_expected = int(_number(block["items_expected"], f"{path}.items_expected", lo=1, integer=True))
    if stages and "stages" not in block:
        raise ApiConfigError(f"{path}: missing keys ['stages']")
    return GridSpec(
        pool=pool, items_expected=items_expected, a_H=_levels(block["a_H"], f"{path}.a_H", upper=1.0), c0=_levels(block["c0"], f"{path}.c0"), w=_levels(block["w"], f"{path}.w"),
        stages=_stages(block["stages"], items_expected) if stages else {},
    )


def _classes(value: Any) -> dict[str, ClassSpec]:
    block = _mapping(value, "request.classes", CLASS_NAMES)
    out: dict[str, ClassSpec] = {}
    for name in CLASS_NAMES:
        spec = _mapping(block[name], f"request.classes.{name}", ("max_tokens", "extra_body"))
        extra = spec["extra_body"]
        if not isinstance(extra, dict):
            raise ApiConfigError(f"request.classes.{name}.extra_body: must be a mapping")
        clash = sorted(set(extra) & _RESERVED_BODY_KEYS)
        if clash:
            raise ApiConfigError(f"request.classes.{name}.extra_body: {clash} are set by the client and cannot be given here")
        out[name] = ClassSpec(name, int(_number(spec["max_tokens"], f"request.classes.{name}.max_tokens", lo=1, integer=True)), dict(extra))
    if out["thinking_off"].extra_body != {"enable_thinking": False}:
        raise ApiConfigError("request.classes.thinking_off.extra_body must be exactly {enable_thinking: false}")
    return out


def pinned_provider_of(extra_body: Mapping[str, Any]) -> str | None:
    """The service that a request body pins: the one name of ``provider.order``. None for a body that has no ``provider`` field."""
    provider = extra_body.get("provider")
    order = provider.get("order") if isinstance(provider, Mapping) else None
    return order[0] if isinstance(order, list) and len(order) == 1 and isinstance(order[0], str) else None


def _openrouter_classes(value: Any) -> dict[str, ClassSpec]:
    """The request classes of a batch of the kind openrouter. A class is a name, an output cap and the extra fields of the body, and it is checked field by field, because it is the whole recipe
    of a request to a pinned service:

    * ``provider``: ``{order: [<one service>], allow_fallbacks: false}``. The request names the service and OpenRouter may not send it anywhere else;
    * ``usage``: ``{include: true}``: OpenRouter then reports its own price of the call, which is recorded beside the booked money;
    * ``reasoning`` (optional): ``{enabled: false}`` with ``max_tokens`` 4 (group A), or ``{effort: minimal}`` with ``max_tokens`` 256 (group B: the lowest thinking, and room for the hidden
      thinking that would otherwise use up the 4 tokens and leave the content empty). A class without it (a model that does not accept the field, which then finds no service if it is sent)
      has ``max_tokens`` 4;
    * ``output_margin_tokens``: exactly 64. The provable bound of the output of a send is ``max_tokens`` plus this (68 in group A, 320 in group B): the OpenAI models report 5 output tokens
      for a ``max_tokens`` of 4 and ``gpt-6-luna-pro`` reports 25. What is sent as ``max_tokens`` is not changed;
    * ``input_bytes_factor`` and ``input_margin_tokens`` (optional, both or neither): exactly 3 and 2048. The provable bound of the input of a send is then three times the bytes of the
      message plus 2,048 tokens instead of the bytes plus ``budget.template_margin_tokens``: ``gpt-6-luna-pro`` reports about 1,450 input tokens more than its message needs. Only the class of
      that model has them.

    Nothing else may be in ``extra_body``; the keys the client sets itself are refused as in the first batch."""
    if not isinstance(value, dict) or not value:
        raise ApiConfigError("request.classes: must be a non-empty mapping")
    out: dict[str, ClassSpec] = {}
    for name, spec_value in value.items():
        if not isinstance(name, str) or not _CLASS_NAME.fullmatch(name):
            raise ApiConfigError(f"request.classes: the class name {name!r} must match [a-z][a-z0-9_]{{0,31}}")
        path = f"request.classes.{name}"
        spec = _mapping(spec_value, path, ("max_tokens", "extra_body", "output_margin_tokens"), ("provider_aliases", "input_bytes_factor", "input_margin_tokens"))
        max_tokens = int(_number(spec["max_tokens"], f"{path}.max_tokens", lo=1, integer=True))
        if not isinstance(spec["output_margin_tokens"], int) or isinstance(spec["output_margin_tokens"], bool) or spec["output_margin_tokens"] != OPENROUTER_OUTPUT_MARGIN:
            raise ApiConfigError(f"{path}.output_margin_tokens: must be {OPENROUTER_OUTPUT_MARGIN}, got {spec['output_margin_tokens']!r} (the provable output bound of a send is max_tokens plus this)")
        if ("input_bytes_factor" in spec) != ("input_margin_tokens" in spec):
            raise ApiConfigError(f"{path}: input_bytes_factor and input_margin_tokens are given together or not at all")
        input_pair: tuple[int, int | None] = (1, None)
        if "input_bytes_factor" in spec:
            given = (spec["input_bytes_factor"], spec["input_margin_tokens"])
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in given) or given != OPENROUTER_WIDE_INPUT_BOUND:
                raise ApiConfigError(f"{path}: input_bytes_factor and input_margin_tokens must be exactly {OPENROUTER_WIDE_INPUT_BOUND[0]} and {OPENROUTER_WIDE_INPUT_BOUND[1]}, got {given!r}")
            input_pair = (OPENROUTER_WIDE_INPUT_BOUND[0], OPENROUTER_WIDE_INPUT_BOUND[1])
        aliases = spec.get("provider_aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases) or len(set(aliases)) != len(aliases):
            raise ApiConfigError(f"{path}.provider_aliases: must be a list of names, each once")
        extra = spec["extra_body"]
        if not isinstance(extra, dict):
            raise ApiConfigError(f"{path}.extra_body: must be a mapping")
        clash = sorted(set(extra) & _RESERVED_BODY_KEYS)
        if clash:
            raise ApiConfigError(f"{path}.extra_body: {clash} are set by the client and cannot be given here")
        _mapping(extra, f"{path}.extra_body", ("provider", "usage"), ("reasoning",))
        provider = _mapping(extra["provider"], f"{path}.extra_body.provider", ("order", "allow_fallbacks"))
        order = provider["order"]
        if not (isinstance(order, list) and len(order) == 1 and isinstance(order[0], str) and order[0].strip()):
            raise ApiConfigError(f"{path}.extra_body.provider.order: must be a list of exactly one service name")
        if provider["allow_fallbacks"] is not False:
            raise ApiConfigError(f"{path}.extra_body.provider.allow_fallbacks: must be false (the request is pinned to the service)")
        usage = extra["usage"]
        if not (isinstance(usage, dict) and set(usage) == {"include"} and usage["include"] is True):
            raise ApiConfigError(f"{path}.extra_body.usage: must be exactly {{include: true}}")
        if "reasoning" not in extra:
            wanted = OPENROUTER_MAX_TOKENS["off"]
        else:
            reasoning = extra["reasoning"]
            if isinstance(reasoning, dict) and set(reasoning) == {"enabled"} and reasoning["enabled"] is False:
                wanted = OPENROUTER_MAX_TOKENS["off"]
            elif isinstance(reasoning, dict) and set(reasoning) == {"effort"} and reasoning["effort"] == "minimal":
                wanted = OPENROUTER_MAX_TOKENS["minimal"]
            else:
                raise ApiConfigError(f"{path}.extra_body.reasoning: must be {{enabled: false}} or {{effort: minimal}}")
        if max_tokens != wanted:
            raise ApiConfigError(f"{path}.max_tokens: {max_tokens} but this class needs {wanted} (group A: thinking switched off or no thinking parameter, 4; group B: minimal thinking, 256)")
        out[name] = ClassSpec(name, max_tokens, json.loads(json.dumps(extra)), tuple(aliases), OPENROUTER_OUTPUT_MARGIN, input_pair[0], input_pair[1])
    return out


def _models(value: Any, classes: Mapping[str, ClassSpec], first_order: int = 0, kind: str = "bailian", cell_a_h: float | None = None) -> tuple[ModelSpec, ...]:
    """The models of a configuration. ``cell_a_h``: the a_H of the pilot of a batch that asks in a cell of its own; every model of such a batch names the model it sends (``wire``) and its id is that name
    and the cell (``vendor/name@a0.95``), and a batch that does not ask in another cell has no ``wire``."""
    if not isinstance(value, list) or not value:
        raise ApiConfigError("models: must be a non-empty list")
    name_form = _OPENROUTER_MODEL_ID if kind == "openrouter" else _MODEL_ID
    shape = "vendor/name" if kind == "openrouter" else "[A-Za-z0-9][A-Za-z0-9_.-]*"
    out = []
    for i, entry in enumerate(value):
        item = _mapping(entry, f"models[{i}]", ("id", "class"), ("aliases", "wire"))
        model_id = _string(item["id"], f"models[{i}].id")
        wire = item.get("wire")
        if (wire is not None) != (cell_a_h is not None):
            raise ApiConfigError(f"models[{i}].wire: a model has a wire name if and only if its batch asks in a cell of its own (the batches {sorted(OPENROUTER_CELL_BATCHES)})")
        if wire is not None:
            wire = _string(wire, f"models[{i}].wire")
            if not name_form.fullmatch(wire):
                raise ApiConfigError(f"models[{i}].wire: {wire!r} must match {shape}")
            if model_id != f"{wire}@a{cell_a_h:.2f}" or not _CELL_MODEL_ID.fullmatch(model_id):
                raise ApiConfigError(f"models[{i}].id: {model_id!r} must be {wire}@a{cell_a_h:.2f} (the model that is sent and the a_H of the cell)")
        elif not name_form.fullmatch(model_id):
            raise ApiConfigError(f"models[{i}].id: {model_id!r} must match {shape}")
        cls = _string(item["class"], f"models[{i}].class")
        if cls not in classes:
            raise ApiConfigError(f"models[{i}].class: {cls!r} is not one of {list(classes)}")
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            raise ApiConfigError(f"models[{i}].aliases: must be a list of names")
        names = tuple(_string(a, f"models[{i}].aliases[{j}]") for j, a in enumerate(aliases))
        for name in names:
            if not name_form.fullmatch(name):
                raise ApiConfigError(f"models[{i}].aliases: {name!r} must match {shape}")
        if len(set(names)) != len(names) or model_id in names or wire in names:
            raise ApiConfigError(f"models[{i}].aliases: a name is listed twice or repeats the id")
        out.append(ModelSpec(model_id, cls, first_order + i, names, wire))
    ids = [m.id for m in out]
    if len(set(ids)) != len(ids):
        raise ApiConfigError("models: a model id is listed twice")
    every_name = [name for m in out for name in m.returned_names()]
    if len(set(every_name)) != len(every_name):
        raise ApiConfigError("models: a name (an id or an alias) belongs to two models")
    return tuple(out)


def _prices_file(value: Any, repo_root: Path) -> Path:
    block = _mapping(value, "prices", ("file",))
    return _anchored(_string(block["file"], "prices.file"), repo_root)


def _retry(value: Any) -> RetryPolicy:
    block = _mapping(
        value, "retry", ("max_attempts", "backoff_base_s", "backoff_factor", "backoff_max_s", "jitter", "timeout_s", "abort_statuses", "stop_after_consecutive_exhausted"),
    )
    statuses = block["abort_statuses"]
    if not isinstance(statuses, list) or not all(isinstance(s, int) and not isinstance(s, bool) and 400 <= s <= 499 and s != 429 for s in statuses):
        raise ApiConfigError("retry.abort_statuses: must be a list of 4xx statuses other than 429 (those are retried)")
    return RetryPolicy(
        max_attempts=int(_number(block["max_attempts"], "retry.max_attempts", lo=1, hi=MAX_ATTEMPTS_CEILING, integer=True)),
        backoff_base_s=float(_number(block["backoff_base_s"], "retry.backoff_base_s", lo=0.0)),
        backoff_factor=float(_number(block["backoff_factor"], "retry.backoff_factor", lo=1.0)),
        backoff_max_s=float(_number(block["backoff_max_s"], "retry.backoff_max_s", lo=0.0)),
        jitter=float(_number(block["jitter"], "retry.jitter", lo=0.0, hi=1.0)),
        timeout_s=float(_number(block["timeout_s"], "retry.timeout_s", positive=True)),
        abort_statuses=tuple(int(s) for s in statuses),
        stop_after_consecutive_exhausted=int(_number(block["stop_after_consecutive_exhausted"], "retry.stop_after_consecutive_exhausted", lo=1, integer=True)),
    )


def _budget(value: Any) -> Budget:
    block = _mapping(value, "budget", ("cap_cny", "reserve_margin", "template_margin_tokens", "measured_factor", "send_quota_factor", "charge_unknown_at_bound"))
    if _number(block["reserve_margin"], "budget.reserve_margin", lo=0.0) != RESERVE_MARGIN:
        raise ApiConfigError(f"budget.reserve_margin: fixed by the rule at {RESERVE_MARGIN}")
    if not isinstance(block["charge_unknown_at_bound"], bool):
        raise ApiConfigError("budget.charge_unknown_at_bound: must be true or false")
    return Budget(
        cap_cny=float(_number(block["cap_cny"], "budget.cap_cny", positive=True, hi=CAP_CEILING_CNY)), reserve_margin=float(RESERVE_MARGIN),
        template_margin_tokens=int(_number(block["template_margin_tokens"], "budget.template_margin_tokens", lo=TEMPLATE_MARGIN_MIN, integer=True)),
        measured_factor=float(_number(block["measured_factor"], "budget.measured_factor", lo=1.0)),
        send_quota_factor=float(_number(block["send_quota_factor"], "budget.send_quota_factor", lo=1.0, hi=SEND_QUOTA_FACTOR_CEILING)),
        charge_unknown_at_bound=block["charge_unknown_at_bound"],
    )


def _state(value: Any) -> tuple[Path, Path]:
    """The state directory and the code root: the two absolute paths of the rules. Another value is an error, so that another file cannot move the state."""
    block = _mapping(value, "state", ("dir", "code_root"))
    state_dir, code_root = _string(block["dir"], "state.dir"), _string(block["code_root"], "state.code_root")
    if state_dir != FIXED_STATE_DIR:
        raise ApiConfigError(f"state.dir: must be {FIXED_STATE_DIR} (one absolute directory for every working tree), got {state_dir!r}")
    if code_root != FIXED_CODE_ROOT:
        raise ApiConfigError(f"state.code_root: must be {FIXED_CODE_ROOT} (the one repository that may send real requests), got {code_root!r}")
    return Path(state_dir), Path(code_root)


def _reading(value: Any) -> None:
    """The reading rule, written in the configuration so that it is part of the frozen md5; the code in ``nmi.api_exp.reader`` is what runs, and they must agree."""
    block = _mapping(value, "reading", ("strip", "answer_digits", "ask_digit"))
    if block["strip"] != "both":
        raise ApiConfigError("reading.strip: the reply is stripped at both ends")
    if block["answer_digits"] != ANSWER_DIGITS or block["ask_digit"] != ASK_DIGIT:
        raise ApiConfigError(f"reading: answer_digits must be {ANSWER_DIGITS!r} and ask_digit {ASK_DIGIT!r} (see nmi/api_exp/reader.py)")


def _analysis(value: Any) -> AnalysisSpec:
    block = _mapping(value, "analysis", ("resamples", "seed", "delta_eq", "dominated_lower", "min_valid_share"))
    return AnalysisSpec(
        resamples=int(_number(block["resamples"], "analysis.resamples", lo=1, integer=True)),
        seed=int(_number(block["seed"], "analysis.seed", lo=0, integer=True)),
        delta_eq=float(_number(block["delta_eq"], "analysis.delta_eq", positive=True)),
        dominated_lower=float(_number(block["dominated_lower"], "analysis.dominated_lower", lo=0.0, hi=1.0)),
        min_valid_share=float(_number(block["min_valid_share"], "analysis.min_valid_share", positive=True, hi=1.0)),
    )


def _canonical(value: Any) -> str:
    """The JSON text that ``frozen_md5_of`` hashes, for one block: two blocks are the same when this text is the same."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _batch_block(value: Any) -> tuple[str, str, int, tuple[str, ...]]:
    """The block ``batch`` as (name, file name of the base, first order, the blocks of its own); the base is read by ``_check_batch``."""
    block = _mapping(value, "batch", ("name", "base", "first_order"), ("own",))
    name = _string(block["name"], "batch.name")
    if not _BATCH_NAME.fullmatch(name):
        raise ApiConfigError(f"batch.name: {name!r} must match [a-z][a-z0-9]{{0,15}} (it is part of the version and of the name of the freeze record)")
    base = _string(block["base"], "batch.base")
    if Path(base).name != base:
        raise ApiConfigError(f"batch.base: {base!r} must be the file name of the first batch's configuration, which is in the directory of this file")
    own = block.get("own", [])
    if not isinstance(own, list) or not all(isinstance(b, str) for b in own) or len(set(own)) != len(own) or not set(own) <= set(BATCH_OWN_OPTIONAL):
        raise ApiConfigError(f"batch.own: must be a list of blocks, each once, from {list(BATCH_OWN_OPTIONAL)}")
    return name, base, int(_number(block["first_order"], "batch.first_order", lo=1, integer=True)), tuple(own)


def _check_batch(
    raw: Mapping[str, Any], version: str, models: Sequence[ModelSpec], parts: tuple[str, str, int, tuple[str, ...]], path: Path, repo_root: Path,
) -> tuple[BatchSpec, ApiExpConfig]:
    """A supplementary batch is the first batch with other models: read the file named by ``batch.base`` and refuse this configuration unless every
    top-level block other than ``BATCH_OWN_KEYS`` and the blocks the batch declares in ``batch.own`` (``endpoint`` and ``request`` of a batch of the kind openrouter) is, item by item, the
    same JSON, the version is the base's and the name of the batch, the models are other models than the base's and their run order comes after the base's. Returns the block as a
    ``BatchSpec`` and the base.

    This compares with the base file as it is NOW. What the first batch was frozen with is another matter: the frozen md5 of the base is kept in the ``BatchSpec`` so that a session can
    check it against the first batch's freeze record (``nmi.api_exp.runner.first_batch_problem``), and a base and a supplementary file changed together after that freeze are refused."""
    name, base_name, first_order, own = parts
    base_path = path.parent / base_name
    if base_path.resolve() == path.resolve():
        raise ApiConfigError("batch.base: a configuration cannot be its own base")
    try:
        base_data = base_path.read_bytes()
    except OSError as error:
        raise ApiConfigError(f"batch.base: {base_name} cannot be read ({type(error).__name__}); it is the configuration of the first batch, in the directory of {path.name}") from error
    base = parse_api_config(base_data, base_path, repo_root=repo_root, _as_base=True)
    if version != f"{base.version}_{name}":
        raise ApiConfigError(f"version: {version!r} must be {base.version}_{name} (the version of the first batch, an underscore and the name of the batch)")
    open_keys = (*BATCH_OWN_KEYS, *own)
    differing = sorted(k for k in set(raw) | set(base.raw) if k not in open_keys and _canonical(raw.get(k)) != _canonical(base.raw.get(k)))
    if differing:
        raise ApiConfigError(f"{path.name} differs from the first batch ({base_name}) in {differing}; a supplementary batch differs from it in {list(open_keys)} only")
    if "pilot" in own:  # a batch that asks in another cell: its grid is the first batch's but for the a_H
        a_h = OPENROUTER_CELL_BATCHES[name]
        pilot, base_pilot = raw.get("pilot"), base.raw.get("pilot")
        if not isinstance(pilot, dict) or pilot.get("a_H") != [a_h]:
            raise ApiConfigError(f"pilot.a_H: the batch {name} asks in the cell a_H {a_h}, so it is [{a_h}]")
        if _canonical({k: v for k, v in pilot.items() if k != "a_H"}) != _canonical({k: v for k, v in base_pilot.items() if k != "a_H"}):
            raise ApiConfigError(f"pilot: the grid of the batch {name} differs from the first batch's in other items than a_H; a batch that asks in another cell differs in a_H only")
    if first_order < len(base.models):
        raise ApiConfigError(f"batch.first_order: {first_order} puts the models before those of the first batch ({len(base.models)} models, run orders 0 to {len(base.models) - 1})")
    shared = sorted({n for m in models for n in m.returned_names()} & {n for m in base.models for n in m.returned_names()})
    if shared:
        raise ApiConfigError(f"models: {shared} belong to the first batch; a model (an id or an alias) is asked in one batch only")
    return BatchSpec(name, base_path, base.md5, base.frozen_md5, first_order, own), base


def parse_api_config(data: bytes, path: Path, *, repo_root: Path = REPO_ROOT, _as_base: bool = False) -> ApiExpConfig:
    try:
        raw = yaml.load(data, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ApiConfigError(f"{path.name}: not valid YAML: {error}") from error
    top = _mapping(raw, "<root>", TOP_KEYS, OPTIONAL_TOP_KEYS)
    if _as_base and "batch" in top:
        raise ApiConfigError(f"batch.base: {path.name} is itself a supplementary batch; the base is the configuration of the first batch")
    version = _string(top["version"], "version")
    batch_parts = _batch_block(top["batch"]) if "batch" in top else None
    if batch_parts is None and not re.fullmatch(r"api_exp_v\d+", version):
        raise ApiConfigError("version: must look like api_exp_v1")
    if batch_parts is not None and not re.fullmatch(r"api_exp_v\d+_[a-z][a-z0-9]{0,15}", version):
        raise ApiConfigError("version: the version of a supplementary batch looks like api_exp_v1_supp1")
    protocol = _mapping(top["protocol"], "protocol", ("file",))
    endpoint = _mapping(top["endpoint"], "endpoint", ("url", "key_file"), ("kind", "price_margin"))
    kind = endpoint.get("kind", "bailian")
    if kind not in ENDPOINT_KINDS:
        raise ApiConfigError(f"endpoint.kind: must be one of {list(ENDPOINT_KINDS)}, got {kind!r}")
    url = _string(endpoint["url"], "endpoint.url")
    if not url.startswith("https://"):
        raise ApiConfigError("endpoint.url: must be an https URL")
    if kind == "openrouter" and url != OPENROUTER_URL:
        raise ApiConfigError(f"endpoint.url: a configuration of the kind openrouter names {OPENROUTER_URL}, got {url!r}")
    if (KEY_FILE_WORD in Path(_string(endpoint["key_file"], "endpoint.key_file")).name.lower()) != (kind == "openrouter"):
        raise ApiConfigError(f"endpoint.key_file: the name of the key file of a configuration of the kind openrouter has {KEY_FILE_WORD!r} in it, and no other configuration names such a file; a key is not sent to a service it does not belong to")
    if batch_parts is not None and batch_parts[3] and kind != "openrouter":
        raise ApiConfigError("batch.own: only a batch of the kind openrouter has blocks of its own")
    if (kind == "openrouter" or (batch_parts is not None and batch_parts[3])) and (batch_parts is None or batch_parts[0] not in OPENROUTER_BATCH_NAMES):
        raise ApiConfigError(f"endpoint.kind: only the supplementary batches {list(OPENROUTER_BATCH_NAMES)} are of the kind openrouter and have blocks of their own (the first batch and the other supplementary batches are not)")
    expected_margin = PRICE_MARGIN_BATCHES.get(batch_parts[0]) if batch_parts is not None else None
    price_margin = endpoint.get("price_margin")
    if (price_margin is None) != (expected_margin is None) or (
        price_margin is not None and (isinstance(price_margin, bool) or not isinstance(price_margin, (int, float)) or float(price_margin) != expected_margin)
    ):
        raise ApiConfigError(
            f"endpoint.price_margin: the batches {sorted(PRICE_MARGIN_BATCHES)} reserve the list price times {PRICE_MARGIN_BATCHES} for every send, and "
            "no other configuration has the key"
        )
    if batch_parts is not None and ("pilot" in batch_parts[3]) != (batch_parts[0] in OPENROUTER_CELL_BATCHES):
        raise ApiConfigError(f"batch.own: the block pilot is the batch's own if and only if the batch asks in a cell of its own (the batches {sorted(OPENROUTER_CELL_BATCHES)})")
    local = _mapping(top["local"], "local", ("config", "split_dir"))
    prompt = _mapping(top["prompt"], "prompt", ("file", "paraphrase", "block", "framing", "layout"))
    request = _mapping(top["request"], "request", ("temperature", "classes"))
    if request["temperature"] != 0:
        raise ApiConfigError("request.temperature: must be 0")
    classes = _openrouter_classes(request["classes"]) if kind == "openrouter" else _classes(request["classes"])
    _reading(top["reading"])
    cell_a_h = OPENROUTER_CELL_BATCHES.get(batch_parts[0]) if batch_parts is not None and "pilot" in batch_parts[3] else None
    models = _models(top["models"], classes, batch_parts[2] if batch_parts is not None else 0, kind, cell_a_h)
    runtime = _mapping(top["runtime"], "runtime", ("concurrency",))
    pilot = _grid(top["pilot"], "pilot", PILOT_POOLS, stages=True)
    full = _grid(top["full"], "full", (FULL_POOL,))
    if pilot.n_conditions != 6:
        raise ApiConfigError(f"pilot: {pilot.n_conditions} conditions; the pilot is the 6 waiting costs at one (a_H, c0)")
    if full.n_conditions != 54:
        raise ApiConfigError(f"full: {full.n_conditions} conditions; the full run is the 54 conditions of the local grid")
    spec = PromptSpec(
        file=_anchored(_string(prompt["file"], "prompt.file"), repo_root), paraphrase=_string(prompt["paraphrase"], "prompt.paraphrase"),
        block=_string(prompt["block"], "prompt.block"), framing=_string(prompt["framing"], "prompt.framing"), layout=_string(prompt["layout"], "prompt.layout"),
    )
    if (spec.framing, spec.layout) != ("f2", "standard"):
        raise ApiConfigError("prompt: framing f2 and layout standard are the pre-specified readout")
    state_dir, code_root = _state(top["state"])
    batch: BatchSpec | None = None
    experiment = tuple(m.id for m in models)
    if batch_parts is not None:
        batch, base = _check_batch(raw, version, models, batch_parts, path, repo_root)
        experiment = (*base.model_ids(), *experiment)
    return ApiExpConfig(
        path=path, md5=bytes_md5(data), frozen_md5=frozen_md5_of(raw), version=version, protocol_file=_anchored(_string(protocol["file"], "protocol.file"), repo_root),
        endpoint_url=url, key_file=Path(_string(endpoint["key_file"], "endpoint.key_file")), local_config=_anchored(_string(local["config"], "local.config"), repo_root),
        split_dir=_anchored(_string(local["split_dir"], "local.split_dir"), repo_root), prompt=spec, temperature=0.0, classes=classes, models=models, retry=_retry(top["retry"]),
        prices_file=_prices_file(top["prices"], repo_root), budget=_budget(top["budget"]), state_dir=state_dir, code_root=code_root, concurrency=int(_number(runtime["concurrency"], "runtime.concurrency", lo=1, hi=256, integer=True)),
        pilot=pilot, full=full, analysis=_analysis(top["analysis"]), raw=raw, batch=batch, experiment_models=experiment, endpoint_kind=kind,
        price_margin=None if price_margin is None else float(price_margin),
    )


def _registered_models(cfg: ApiExpConfig, repo_root: Path) -> tuple[str, ...] | None:
    """The models of every batch, from ``api_exp_batches.yaml`` in the directory of ``cfg`` (None when there is no such file: the configuration stands alone). The file lists the
    configuration file of the first batch and then those of the supplementary batches, by file name::

        format: 1
        batches: [api_exp_v1.yaml, api_exp_v1_supp1.yaml]

    ``cfg`` must be one of them. The first is a first batch, every other one is a supplementary batch of it with a name of its own, and no model (id or alias) and no run order belongs to
    two batches: the models share one ledger, one price file, one release list and one budget state, in which the run order of a model is its place."""
    registry = cfg.path.parent / BATCH_REGISTRY_NAME
    if not registry.is_file():
        return None
    try:
        raw = yaml.load(registry.read_bytes(), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ApiConfigError(f"{registry.name}: cannot be read as YAML ({type(error).__name__})") from error
    top = _mapping(raw, registry.name, ("format", "batches"))
    names = top["batches"]
    if top["format"] != 1 or isinstance(top["format"], bool):
        raise ApiConfigError(f"{registry.name}: format must be 1")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n and Path(n).name == n for n in names) or len(set(names)) != len(names):
        raise ApiConfigError(f"{registry.name}: batches must be a list of configuration file names (no paths), each once")
    configs: list[ApiExpConfig] = []
    for name in names:
        file = registry.parent / name
        if file.resolve() == cfg.path.resolve():
            configs.append(cfg)
            continue
        try:
            data = file.read_bytes()
        except OSError as error:
            raise ApiConfigError(f"{registry.name}: {name} cannot be read ({type(error).__name__})") from error
        configs.append(parse_api_config(data, file, repo_root=repo_root))
    if not any(c is cfg for c in configs):
        raise ApiConfigError(f"{cfg.path.name} is not listed in {registry.name}; a configuration of the experiment is listed there, or the file is removed")
    first, *others = configs
    if first.batch is not None:
        raise ApiConfigError(f"{registry.name}: the first file must be the configuration of the first batch, {first.path.name} is a supplementary batch")
    for other in others:
        if other.batch is None or other.batch.base.resolve() != first.path.resolve():
            raise ApiConfigError(f"{registry.name}: {other.path.name} must be a supplementary batch of {first.path.name}")
    batch_names = [c.batch.name for c in others if c.batch is not None]
    models = [m for c in configs for m in c.models]
    every_name = [n for m in models for n in m.state_names()]
    if len(set(batch_names)) != len(batch_names) or len(set(every_name)) != len(every_name) or len({m.order for m in models}) != len(models):
        raise ApiConfigError(f"{registry.name}: two batches have the same name, the same model (an id or an alias) or the same run order")
    return tuple(m.id for m in models)


def load_api_config(path: str | Path, *, expected_md5: str | None = None, repo_root: Path = REPO_ROOT) -> ApiExpConfig:
    """Read, validate and freeze the configuration; optionally check the md5 of the file first (the frozen value). If ``api_exp_batches.yaml`` is next to it, the models of all the batches
    are in ``experiment_models`` (``_registered_models``)."""
    path = Path(path)
    data = path.read_bytes()
    if expected_md5 is not None and bytes_md5(data) != expected_md5:
        raise ApiConfigError(f"{path.name} has md5 {bytes_md5(data)}, the expected value is {expected_md5}; the configuration changed after it was frozen")
    cfg = parse_api_config(data, path, repo_root=repo_root)
    models = _registered_models(cfg, repo_root)
    return cfg if models is None else replace(cfg, experiment_models=models)
