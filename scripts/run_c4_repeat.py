#!/usr/bin/env python
"""C4 for the new main models that have no float32 reference: the repeated readout.

One model, one condition, the 300 items of the cross-check (pool ``mmlu_redux_eval``, key ``SeedNamespace.hf_cross_check_key()`` = (9, 2, 9001, 0xA0), the first ``--n-items`` entries of
``draw_cross_check_order``). The P-ask prompt is a00, F2, standard layout, a_H 0.80, c0 0, w 0 (all of them options, with the pre-specified values as defaults). It is read twice, by two
independent offline vLLM engines:

* The PARENT (this script without ``--child-pass``) never opens an engine, never loads a tokenizer and never initialises CUDA. It draws the item KEYS (no question text), opens the run
  (``common.open_run``: manifest, source guard, SIGTERM), and starts one child process per pass. Before the second child starts it checks that the first one has exited, that no live process of its process
  group remains and none of the processes it reported below itself (the engine-core process holds the GPU memory), and that ``nvidia-smi --query-compute-apps=pid`` lists neither the child's PID nor
  one of those. Nothing is killed: if the check fails the run stops with exit code 1 and names the PIDs. After both children it pairs the two tables by item key, checks them, judges and writes the report.
* A CHILD (``--child-pass 1|2``, internal) renders the prompts of its pass, tokenises them (the digest of every prompt's token ids is recorded), starts one engine
  (``common.open_backend``), reads the engine settings back, makes ONE ``readouts()`` call with all the prompts of the pass (one condition, no mixing, no warm-up call, no explicit cache reset: the
  prefix cache is empty because the process is new), closes the engine, writes its table (``parts/pass<k>.parquet``) and a record (``pass<k>.json``), and exits. Pass 1 submits the items in the
  draw order, pass 2 in the order of a shuffle drawn from ``SeedNamespace.c4_repeat_order_key()`` = (9, 9, 9001, 0xA0).

The verdict is the "valid and equal" gate. For each drawn item i: S_i = 1 when both passes have a usable readout, the digit mass of both is at least the gate of the
configuration (0.5), and the ask decisions are the same (margin > 0 asks, margin <= 0 answers). The denominator is the number of drawn items. The run passes when sum S_i >= n - ``--max-failures``
(300 items, one failure allowed: 299). ``--threshold`` (0.995) is recorded and compared with the agreement, but the count decides. An item with a low-quality readout in either pass is ONE quality
failure; a decision flip is counted only among items whose two readouts are valid, so the two numbers do not overlap. A missing record, a duplicated item key, a readout that cannot be computed
(NaN or +inf in a digit log-probability), a different set of items, a different input token sequence of an item, or engine settings that differ between the passes are TECHNICAL FAILURES: exit 1,
no verdict, nothing counted. Reported and deciding nothing: the option choices of items both passes answer, the largest difference of the digit log-probabilities, the digit-mass and margin
distributions.

Output ``results/c4/<time>_<tag>/``: ``manifest.json`` (with ``c4_repeat``: the planned protocol, the engine settings read back from each engine, the quantisation and dtype of the weights),
``c4_repeat.json``, ``c4_repeat_items.parquet`` (item keys, digests and numbers; no text), ``pass1.json``, ``pass2.json``, ``pass<k>_child.json`` (PID and process group of the child, written when it
starts), ``pass<k>_processes.json`` (the processes below the child once its engine is up), ``parts/pass<k>.parquet``, ``logs/pass<k>.log`` (the output of the child, which carries the engine's own start-up lines) and, unless the hooks say otherwise, ``llm/model.json``.
``--resume RUN_DIR`` skips a pass that is finished.

Examples (one idle card each, ``CUDA_VISIBLE_DEVICES`` chosen by the caller)::

    python scripts/run_c4_repeat.py --config configs/v6_confirm.yaml --expect-config-md5 <md5> --model qwen3-14b --gpu-mem-util 0.80 --llm-kwargs '{"quantization": "fp8"}' --max-model-len 4096 --tag <tag>
    python scripts/run_c4_repeat.py --config configs/v6_confirm.yaml --expect-config-md5 <md5> --model qwen3-32b --gpu-mem-util 0.85 --max-model-len 4096 --tag <tag>

``--pilot`` reads the first ``--pilot-items`` (default and maximum 6) items in each pass: at most 12 readouts in all. A pilot only shows that the entry runs; its role is
``pilot``. A run with another condition, item count, threshold or failure allowance than the pre-specified ones has the role ``description`` (for example the w 0.2 bridge on the same items): the
report lists ``non_default_options``. Only a run with every pre-specified value has the role ``primary``.

Exit codes: 0 passed; 1 not passed or technical failure; 2 arguments or inputs; 4 ``nmi/`` changed during the run (``run.check_source()``); SIGTERM is recorded by ``common.open_run`` (``aborted.json``).
``--dry-run`` checks the inputs and prints the plan: it loads no model and no tokenizer, reads no question text and starts no process.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _c2_common as common  # noqa: E402
import numpy as np  # noqa: E402

from nmi import paths, seeds  # noqa: E402
from nmi.data.registry import SplitError  # noqa: E402
from nmi.provenance import LabelExistsError, note_skipped, source_hash  # noqa: E402
from nmi.readout.digits import ReadoutSpec  # noqa: E402
from nmi.readout.hf_backend import CrossCheckSpec, draw_cross_check_order, encode_prompts  # noqa: E402

SCRIPT = "run_c4_repeat"
STAGE = "c4"
REPORT_SCHEMA = "nmi.c4_repeat/1"
PASS_SCHEMA = "nmi.c4_repeat_pass/1"
POOL = common.MAIN_EVAL_POOL  # mmlu_redux_eval, the pool of the cross-check
PARAPHRASE, FRAMING, LAYOUT = "a00", "f2", "standard"
PASSES = (1, 2)
# the pre-specified values; every one of them is an option
REGISTERED = {"n_items": 300, "a_H": 0.80, "c0": 0.0, "w": 0.0, "threshold": 0.995, "max_failures": 1}
PILOT_ITEMS_DEFAULT = PILOT_ITEMS_MAX = 6  # the pre-flight makes at most 12 readouts in all: two passes of at most 6 prompts
RELEASE_WAIT_S, RELEASE_POLL_S = 30.0, 1.0  # how long the driver may take to let go of an exited child
GPU_QUERY = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
GPU_INFO_QUERY = ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"]
SPEC_FILES = {  # bound in the manifest (input_md5) besides the files that ``common.manifest_inputs`` binds; documents that are not part of this release are skipped
    "prereg_v2_draft": common.REPO_ROOT / "protocol" / "preregistration_v2_draft.md",
    "spec_c4": common.REPO_ROOT / "protocol" / "repeat_readout_spec.md",
}
SCOPE = (
    "The result covers the condition and the way of submission that were checked (a00, F2, standard layout, a_H {a_H:g}, c0 {c0:g}, w {w:g}; {n} prompts of one condition in one readouts() call per "
    "pass, one engine process per pass). It is not extended to the other conditions of the confirmation grid."
)
ROLE_NOTES = {
    "primary": "every registered value (items, condition, threshold, allowance): this is the C4 determination for the model",
    "description": "another condition, item count, threshold or allowance than the registered ones (for example the w 0.2 bridge on the same items): a description; it does not replace the primary determination and is not merged with it",
    "pilot": "a pre-flight that only shows that the entry runs: not the frozen protocol; no condition and no threshold is chosen from it",
}
SUBMISSION = {
    "readouts_calls_per_pass": 1,
    "prompts_in_the_call": "all the prompts of the pass",
    "mixed_conditions": False,
    "input": "token ids encoded on the client with the tokenizer of the model, no special tokens added (nmi.readout.hf_backend.encode_prompts); the digest of every prompt's ids is recorded",
    "prefix_cache_reset": "none: the entry never resets the prefix cache; it is empty because every pass starts a new engine process",
    "warmup": "none: the first readouts() call is the evaluation call; the engine's own start-up profiling and CUDA-graph capture use dummy inputs",
    "first_pass_order": "the draw order of draw_cross_check_order",
    "second_pass_order": "the order of seeds.rng(SeedNamespace.c4_repeat_order_key()).permutation(n)",
}
# what the engine record must hold (the others may be unreadable without stopping the run): these are read back from the engine
REQUIRED_SETTINGS = (
    "model.quantization", "model.dtype", "prefix_caching.enabled", "scheduler.max_num_seqs", "scheduler.max_num_batched_tokens", "scheduler.enable_chunked_prefill", "attention.backend_requested",
)
# vllm_config section -> (attribute of VllmConfig, {name in the record: attribute of that config}); the names are those of vllm 0.30.0 (vllm/config/*.py)
SETTINGS_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "model": ("model_config", {"quantization": "quantization", "dtype": "dtype", "max_model_len": "max_model_len", "seed": "seed", "enforce_eager": "enforce_eager", "logprobs_mode": "logprobs_mode"}),
    "prefix_caching": ("cache_config", {"enabled": "enable_prefix_caching", "hash_algo": "prefix_caching_hash_algo", "block_size": "block_size"}),
    "kv_cache": ("cache_config", {"cache_dtype": "cache_dtype", "gpu_memory_utilization": "gpu_memory_utilization"}),
    "scheduler": ("scheduler_config", {
        "max_num_seqs": "max_num_seqs", "max_num_batched_tokens": "max_num_batched_tokens", "enable_chunked_prefill": "enable_chunked_prefill", "policy": "policy",
        "async_scheduling": "async_scheduling", "long_prefill_token_threshold": "long_prefill_token_threshold", "max_num_scheduled_tokens": "max_num_scheduled_tokens",
    }),
    "attention": ("attention_config", {"backend_requested": "backend", "flash_attn_version": "flash_attn_version"}),
    "compilation": ("compilation_config", {"mode": "mode", "cudagraph_mode": "cudagraph_mode", "cudagraph_capture_sizes": "cudagraph_capture_sizes"}),
    "parallel": ("parallel_config", {"tensor_parallel_size": "tensor_parallel_size", "pipeline_parallel_size": "pipeline_parallel_size", "data_parallel_size": "data_parallel_size",
                                     "distributed_executor_backend": "distributed_executor_backend"}),
}
VOLATILE_MAP = {"kv_cache": ("cache_config", {"num_gpu_blocks": "num_gpu_blocks", "kv_cache_size_tokens": "kv_cache_size_tokens"})}  # recorded, not compared between the passes
CHECKPOINT_QUANT_KEYS = ("quant_method", "bits", "group_size", "zero_point", "version", "sym", "desc_act")
SOFTWARE_PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "triton", "flash-attn", "xformers", "numpy", "safetensors", "compressed-tensors", "pyarrow")
SOFTWARE_PREFIXES = ("nvidia-", "flashinfer", "vllm")
ENV_PREFIXES = ("VLLM_", "CUDA_", "PYTORCH_", "CUBLAS", "NVIDIA_")
# the lines in which vLLM 0.30 (vllm/platforms/cuda.py) reports the attention backend it selected inside the engine-core process; the log of the child carries them
ATTENTION_LOG_PATTERNS = (re.compile(r"Using (\S+) attention backend out of potential backends"), re.compile(r"\[cuda\.py:\d+\] Using (\S+) backend\."))  # automatic selection; selection by the user
FAILURE_CODES = {  # the column ``failure_code`` of the item table: why an item is not valid and equal (0: it is)
    0: "valid and equal", 1: "low quality in pass 1 only", 2: "low quality in pass 2 only", 3: "low quality in both passes", 4: "decision flip (both readings valid, the ask decisions differ)",
}
class TechnicalFailure(RuntimeError):
    """The two readings cannot be compared or the process that makes them did not behave: no verdict, exit code 1, nothing is counted."""


# ---------------------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    model: str
    n_items: int
    pilot: bool
    a_H: float
    c0: float
    w: float
    threshold: float
    max_failures: int
    llm_kwargs: Mapping[str, Any]  # the keyword arguments of vllm.LLM as the entry builds them (common.vllm_llm_kwargs)
    non_default: tuple[str, ...]  # protocol options whose value is not the pre-specified one

    @property
    def role(self) -> str:
        if self.pilot:
            return "pilot"
        return "description" if self.non_default else "primary"

    @property
    def condition(self) -> dict[str, Any]:
        return {"paraphrase": PARAPHRASE, "framing": FRAMING, "layout": LAYOUT, "a_H": self.a_H, "c0": self.c0, "w": self.w}

    @property
    def required_valid_and_equal(self) -> int:
        return self.n_items - self.max_failures


def resolve_parameters(ctx: common.Context, args: argparse.Namespace) -> Params:
    """The options as one validated record. ``EntryError`` (exit 2) for anything the entry cannot run with."""
    if args.pilot_items is not None and not args.pilot:
        raise common.EntryError("--pilot-items needs --pilot")
    if args.pilot:
        if args.n_items is not None:
            raise common.EntryError("--pilot reads --pilot-items items per pass; do not combine it with --n-items")
        n = PILOT_ITEMS_DEFAULT if args.pilot_items is None else int(args.pilot_items)
        if not 1 <= n <= PILOT_ITEMS_MAX:
            raise common.EntryError(f"--pilot-items must be in 1..{PILOT_ITEMS_MAX} (a pilot makes at most {2 * PILOT_ITEMS_MAX} readouts in all), got {n}")
    else:
        n = REGISTERED["n_items"] if args.n_items is None else int(args.n_items)
        if n < 1:
            raise common.EntryError(f"--n-items must be positive, got {n}")
    for name, value in (("--a-h", args.a_H), ("--c0", args.c0), ("--w", args.w), ("--threshold", args.threshold)):
        if not math.isfinite(float(value)):
            raise common.EntryError(f"{name} must be a finite number, got {value}")
    if not 0.0 < args.a_H <= 1.0:
        raise common.EntryError(f"--a-h must be in (0, 1], got {args.a_H}")
    cap = float(ctx.cfg["grid"]["displayed_w"]["cap"])
    if args.c0 < 0.0 or not 0.0 <= args.w <= cap:
        raise common.EntryError(f"--c0 must not be negative and --w must be in [0, {cap:g}] (the displayed cap), got c0 {args.c0}, w {args.w}")
    if not 0.0 < args.threshold <= 1.0:
        raise common.EntryError(f"--threshold must be in (0, 1], got {args.threshold}")
    if not 0 <= args.max_failures < n:
        raise common.EntryError(f"--max-failures must be in 0..{n - 1} for {n} items, got {args.max_failures}")
    if PARAPHRASE not in ctx.book.paraphrases:
        raise common.EntryError(f"the prompt file has no paraphrase {PARAPHRASE!r}")
    cross_check_pool = CrossCheckSpec.from_config(ctx.cfg).pool
    if cross_check_pool != POOL:
        raise common.EntryError(f"the cross-check of the configuration draws from {cross_check_pool!r}; the repeated readout reads {POOL!r}, the same items")
    llm_kwargs = parse_llm_kwargs(args)
    values = {"n_items": n, "a_H": float(args.a_H), "c0": float(args.c0), "w": float(args.w), "threshold": float(args.threshold), "max_failures": int(args.max_failures)}
    # a pilot always reads fewer items than the pre-specified number: that is its role, not a deviation to list
    non_default = tuple(name for name, registered in REGISTERED.items() if not (args.pilot and name == "n_items") and values[name] != registered)
    return Params(
        model=args.model, n_items=n, pilot=bool(args.pilot), a_H=values["a_H"], c0=values["c0"], w=values["w"], threshold=values["threshold"], max_failures=values["max_failures"],
        llm_kwargs=llm_kwargs, non_default=non_default,
    )


def parse_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``common.vllm_llm_kwargs(args)`` after the checks that keep ``--llm-kwargs`` from overriding the dedicated options or the logprob mode."""
    if args.llm_kwargs:
        try:
            extra = json.loads(args.llm_kwargs)
        except ValueError as error:
            raise common.EntryError(f"--llm-kwargs is not valid JSON: {error}") from error
        if not isinstance(extra, dict):
            raise common.EntryError("--llm-kwargs must be a JSON object")
        clash = sorted(set(extra) & {"gpu_memory_utilization", "max_model_len", "logprobs_mode"})
        if clash:
            raise common.EntryError(f"--llm-kwargs must not set {clash}: use --gpu-mem-util and --max-model-len; logprobs_mode must stay at the vLLM default (raw_logprobs)")
    return dict(common.vllm_llm_kwargs(args))


# ---------------------------------------------------------------------------------------
# the draw: keys only in the parent, the same function in the children
# ---------------------------------------------------------------------------------------

def sha256_text(parts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def token_digest(ids: Sequence[int]) -> str:
    """sha256 of one prompt's token ids (decimal, comma separated): equal digests mean equal input sequences; no text is stored."""
    return hashlib.sha256(",".join(str(int(t)) for t in ids).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class DrawPlan:
    """The items of the run and the two submission orders, as item KEYS."""

    n_pool: int
    draw_key: tuple[int, int, int, int]
    shuffle_key: tuple[int, int, int, int]
    positions: tuple[int, ...]  # positions in the pool of the drawn items, in the draw order
    keys: tuple[str, ...]  # item keys in the draw order (the order of the first pass)
    second_order: tuple[int, ...]  # row j of the second pass is the item at draw position second_order[j]

    @property
    def second_keys(self) -> tuple[str, ...]:
        return tuple(self.keys[j] for j in self.second_order)

    @property
    def keys_sha256(self) -> str:
        return sha256_text(self.keys)

    @property
    def second_keys_sha256(self) -> str:
        return sha256_text(self.second_keys)

    def order_of(self, pass_index: int) -> tuple[int, ...]:
        """Draw positions in the order the pass submits them."""
        return tuple(range(len(self.keys))) if pass_index == 1 else self.second_order


def build_plan(ctx: common.Context, n_items: int) -> DrawPlan:
    """The draw from the key list of the pool (``SplitRegistry.keys``: the order of ``load_pool_items``): the parent needs no question text."""
    pool_keys = tuple(ctx.registry().keys(POOL))
    if n_items > len(pool_keys):
        raise common.EntryError(f"--n-items {n_items} is more than the {len(pool_keys)} items of {POOL}")
    draw_key, shuffle_key = ctx.ns.hf_cross_check_key(), ctx.ns.c4_repeat_order_key()
    order = [int(i) for i in draw_cross_check_order(len(pool_keys), draw_key, n_items)]
    keys = tuple(pool_keys[i] for i in order)
    if len(set(keys)) != len(keys):
        raise common.EntryError("the drawn items hold a repeated item key")
    second = tuple(int(j) for j in seeds.rng(shuffle_key).permutation(len(keys)))
    return DrawPlan(n_pool=len(pool_keys), draw_key=tuple(draw_key), shuffle_key=tuple(shuffle_key), positions=tuple(order), keys=keys, second_order=second)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# what the engine says about itself
# ---------------------------------------------------------------------------------------

_MISSING = object()


def plain(value: Any) -> Any:
    """JSON-ready: enums by name, ``torch.dtype`` without its prefix, containers recursively, anything else as text."""
    if isinstance(value, enum.Enum):
        return value.name
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in value]
    return str(value).removeprefix("torch.")


def checkpoint_quantization(model_config: Any) -> dict[str, Any] | None:
    """The quantisation block of the checkpoint's own ``config.json`` (AWQ and the like), reduced to its descriptive keys; None when the checkpoint has none."""
    hf_config = getattr(model_config, "hf_config", None)
    block = getattr(hf_config, "quantization_config", None)
    if block is None:
        return None
    if not isinstance(block, Mapping) and hasattr(block, "to_dict"):
        block = block.to_dict()
    if not isinstance(block, Mapping):
        return {"text": str(block)}
    return {k: plain(block[k]) for k in CHECKPOINT_QUANT_KEYS if k in block}


def read_engine_settings(engine: Any) -> dict[str, Any]:
    """The settings the engine WORKS with, read back from ``engine.llm_engine.vllm_config`` (nothing here is an argument that was passed in): the weights' quantisation and dtype, prefix caching,
    scheduling and batching, the attention backend that was requested (``None``: chosen by the engine; the choice itself is made inside the engine-core process and is in the engine's log),
    the compilation mode and the environment variables that vLLM reads. A value the engine does not have is ``None`` and its name is in ``unreadable``; ``volatile`` is recorded but not compared
    between the passes."""
    vllm_config = getattr(getattr(engine, "llm_engine", None), "vllm_config", None)
    unreadable: list[str] = []

    def pull(section: str, source: str, names: Mapping[str, str]) -> dict[str, Any]:
        node = getattr(vllm_config, source, _MISSING) if vllm_config is not None else _MISSING
        out: dict[str, Any] = {}
        for label, attribute in names.items():
            value = getattr(node, attribute, _MISSING) if node is not _MISSING else _MISSING
            if value is _MISSING:
                unreadable.append(f"{section}.{label}")
                out[label] = None
            else:
                out[label] = plain(value)
        return out

    settings = {section: pull(section, source, names) for section, (source, names) in SETTINGS_MAP.items()}
    model_config = getattr(vllm_config, "model_config", None)
    settings["model"]["checkpoint_quantization"] = checkpoint_quantization(model_config) if model_config is not None else None
    volatile = {section: pull(f"volatile.{section}", source, names) for section, (source, names) in VOLATILE_MAP.items()}
    environment = {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}
    return {"settings": settings, "volatile": volatile, "environment": environment, "unreadable": unreadable}


def engine_software_versions() -> dict[str, Any]:
    """Interpreter and the versions of vLLM, torch, the CUDA wheels (``nvidia-*``) and the kernel libraries, read from the package metadata of the engine process."""
    from importlib import metadata

    found: dict[str, str] = {}
    for dist in metadata.distributions():
        name = str(dist.metadata["Name"] or "").lower().replace("_", "-")
        if name and (name in SOFTWARE_PACKAGES or name.startswith(SOFTWARE_PREFIXES)):
            found[name] = dist.version
    out: dict[str, Any] = {"python": platform.python_version(), "executable": sys.executable, "packages": dict(sorted(found.items()))}
    torch = sys.modules.get("torch")
    if torch is not None:  # the engine process has imported it; nothing is imported here
        out["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    return out


def requested_against_actual(requested: Mapping[str, Any], settings: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """For the keyword arguments that were passed to ``vllm.LLM`` and have a counterpart in the settings read back: whether the engine works with what was asked."""
    pairs = (("quantization", "model", "quantization"), ("dtype", "model", "dtype"), ("max_model_len", "model", "max_model_len"), ("seed", "model", "seed"),
             ("gpu_memory_utilization", "kv_cache", "gpu_memory_utilization"))
    out: dict[str, Any] = {}
    for key, section, label in pairs:
        if key not in requested:
            continue
        asked, actual = requested[key], settings.get(section, {}).get(label)
        if isinstance(asked, float) or isinstance(actual, float):
            same = actual is not None and math.isclose(float(asked), float(actual), rel_tol=1e-9, abs_tol=1e-12)
        else:
            same = str(plain(asked)).removeprefix("torch.") == str(actual)
        out[key] = {"requested": plain(asked), "actual": actual, "same": bool(same)}
    return out


# ---------------------------------------------------------------------------------------
# the child: one pass, one engine, one readouts() call
# ---------------------------------------------------------------------------------------

@dataclass
class C4Hooks:
    """Replaceable parts, for testing. ``opener`` and ``tokenizer_loader`` are the child's; ``launcher``, ``gpu_probe``, ``gpu_info``, ``group_members``, ``alive`` and ``sleep`` are the parent's."""

    opener: Callable[..., common.BackendHandle] | None = None
    tokenizer_loader: Callable[[Path], Any] | None = None
    launcher: Callable[[ChildSpec], ChildOutcome] | None = None
    gpu_probe: Callable[[], list[int]] | None = None
    gpu_info: Callable[[], list[dict[str, str]]] | None = None
    group_members: Callable[[int], list[int]] | None = None
    alive: Callable[[list[int]], list[int]] | None = None
    sleep: Callable[[float], None] | None = None
    model_files: bool = True


def child_arguments(ctx: common.Context, args: argparse.Namespace, params: Params, run_dir: Path, pass_index: int, source_sha256: str) -> list[str]:
    """The command line of one child, from the parent's resolved parameters: the configuration is named by path AND md5 (the child refuses a changed file), the item count is the effective one."""
    out = [
        "--child-pass", str(pass_index), "--child-run-dir", str(run_dir), "--child-source-sha256", source_sha256,
        "--config", str(ctx.cfg.path), "--expect-config-md5", ctx.cfg.md5, "--results-root", str(ctx.results_root), "--split-dir", str(ctx.split_dir),
        "--model", params.model, "--gpu-mem-util", repr(float(args.gpu_mem_util)), "--max-model-len", str(int(args.max_model_len)),
        "--n-items", str(params.n_items), "--a-h", repr(params.a_H), "--c0", repr(params.c0), "--w", repr(params.w), "--threshold", repr(params.threshold),
        "--max-failures", str(params.max_failures),
    ]
    if args.llm_kwargs:
        out += ["--llm-kwargs", args.llm_kwargs]
    return out


def child_command(child_args: Sequence[str]) -> list[str]:
    """The command line of a child process: this interpreter (with ``-B`` when this process does not write bytecode), this script, the child arguments."""
    interpreter = [sys.executable, *(["-B"] if sys.flags.dont_write_bytecode else [])]
    return [*interpreter, str(Path(__file__).resolve()), *child_args]


def execute_child(ctx: common.Context, args: argparse.Namespace, hooks: C4Hooks) -> int:
    """One pass in this process: render, tokenise, start the engine, read back its settings, read once, close the engine, write the table and the record."""
    pass_index = args.child_pass
    run_dir = Path(args.child_run_dir or "")
    if not args.child_run_dir or not run_dir.is_dir():
        raise common.EntryError(f"--child-run-dir {args.child_run_dir!r} is not a run directory")
    paths.assert_off_root_partition(run_dir)
    if args.child_source_sha256 and source_hash() != args.child_source_sha256:
        raise common.SourceChangedError("nmi/ is not the source the parent started with")
    guard = common.SourceHashGuard()
    params = resolve_parameters(ctx, args)
    common.require_gpu_memory_fraction(args)
    entry = common.device_entry(ctx.cfg, args.model)
    spec = ReadoutSpec.from_config(ctx.cfg)
    plan = build_plan(ctx, params.n_items)
    items = common.load_pool_items(ctx, POOL)
    if [i.item_key for i in items] != list(ctx.registry().keys(POOL)):
        raise common.EntryError(f"{POOL}: the loaded items are not in the order of the frozen key list")
    order = plan.order_of(pass_index)
    drawn = [items[p] for p in plan.positions]
    ordered = [drawn[j] for j in order]
    submit_keys = [i.item_key for i in ordered]
    if submit_keys != ([*plan.keys] if pass_index == 1 else list(plan.second_keys)):
        raise common.EntryError("the items of this pass are not the ones the parent planned")
    started = common.now_local()
    t_start = time.time()
    tokenizer = (hooks.tokenizer_loader or common.load_tokenizer)(Path(entry["path"]))
    spec.token_ids(tokenizer)  # every digit is one token, or DigitTokenError
    texts = common.ask_texts(ctx.book, tokenizer, ordered, common.chat_kwargs_of(entry), paraphrase=PARAPHRASE, framing=FRAMING, a_H=params.a_H, c0=params.c0, w=params.w, layout=LAYOUT)
    id_lists = encode_prompts(tokenizer, texts)
    digests = [token_digest(ids) for ids in id_lists]
    opener = hooks.opener or (lambda c, kind, model, sp: common.open_backend(c, kind, model, sp))
    guard.check()
    t_engine = time.time()
    handle = opener(ctx, "vllm_offline", args.model, spec)
    engine_start_s = time.time() - t_engine
    below = descendant_pids(os.getpid())  # the engine-core process holds the GPU memory; its PID is written down now, so that the parent can look for it after this process has gone
    common.write_json(run_dir / f"pass{pass_index}_processes.json", {"pass": pass_index, "pid": os.getpid(), "descendants": below, "written": common.iso(common.now_local())})
    try:
        numeric_actual = handle.numeric_actual()
        engine_record = read_engine_settings(handle.backend.engine)
        engine_record["requested_vs_actual"] = requested_against_actual(params.llm_kwargs, engine_record["settings"])
        software = engine_software_versions()
        guard.check()
        t_read = time.time()
        records = handle.backend.readouts(id_lists)  # the one call of the pass: every prompt, one condition
        read_s = time.time() - t_read
        guard.check()
    finally:
        handle.close()
    if len(records) != len(id_lists):
        raise common.EntryError(f"pass {pass_index}: the engine returned {len(records)} readouts for {len(id_lists)} prompts")
    n = len(records)
    columns: dict[str, Any] = {
        "item_key": submit_keys, "submit_pos": np.arange(n), "draw_pos": np.asarray(order, dtype=np.int64), "input_sha256": digests,
        "n_input_tokens": np.asarray([len(ids) for ids in id_lists], dtype=np.int64), **common.trial_columns(records, free_logprob=True),
    }
    table_path = common.PartStore(run_dir / "parts").write(f"pass{pass_index}", columns)
    ended = common.now_local()
    condition = params.condition
    info = {
        "schema": PASS_SCHEMA, "pass": pass_index, "status": "ok", "order": "draw" if pass_index == 1 else "shuffled", "pid": os.getpid(), "started": common.iso(started), "ended": common.iso(ended),
        "engine_start_s": engine_start_s, "read_s": read_s, "wall_s": time.time() - t_start, "n_prompts": n, "item_keys_sha256": plan.keys_sha256, "submission_keys_sha256": sha256_text(submit_keys),
        "table": f"parts/{table_path.name}", "table_md5": common.file_md5(table_path), "condition": condition,
        "readouts_calls": [{
            "index": 1, "n_prompts": n, "mixed_conditions": False, "condition": condition, "order": "draw" if pass_index == 1 else "shuffled", "seconds": read_s,
            "warmup_call_before": False, "prefix_cache_reset_before": False,
        }],
        "submission": dict(SUBMISSION), "requested": {"llm_kwargs": dict(params.llm_kwargs)}, "numeric_actual": numeric_actual, "engine": engine_record, "software": software,
    }
    common.write_json(run_dir / f"pass{pass_index}.json", info)
    common.log(f"pass {pass_index}: {n} prompts read in {read_s:.1f} s (engine start {engine_start_s:.1f} s)")
    return 0


# ---------------------------------------------------------------------------------------
# the parent: starting a child and checking that it let go of the GPU
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ChildSpec:
    pass_index: int
    args: list[str]  # the arguments of the child entry (what ``main`` parses)
    command: list[str]  # the whole command line of the subprocess launcher
    log_path: Path
    cwd: Path
    on_start: Callable[[int], None]  # called with the PID as soon as the process exists


@dataclass(frozen=True)
class ChildOutcome:
    pid: int | None
    pgid: int | None
    returncode: int | None


def subprocess_launcher(spec: ChildSpec) -> ChildOutcome:
    """The child is a new process in a new session (its process group is its PID, and so are the groups of everything it starts: the engine-core process, too), its output goes to the log
    file. The parent waits for it and never kills it."""
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(spec.log_path, "ab") as log:
        log.write(f"--- pass {spec.pass_index}: {shlex.join(spec.command)}\n".encode("utf-8"))
        log.flush()
        process = subprocess.Popen(spec.command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=str(spec.cwd), env=env, start_new_session=True)  # noqa: S603
        spec.on_start(process.pid)
        returncode = process.wait()
    return ChildOutcome(pid=process.pid, pgid=process.pid, returncode=returncode)


def process_table(proc_root: Path = Path("/proc")) -> dict[int, tuple[str, int, int, int]]:
    """PID -> (state, parent PID, process group, session) of every process that ``/proc`` shows."""
    table: dict[int, tuple[str, int, int, int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii", errors="replace")
            fields = stat[stat.rindex(")") + 2:].split()  # the command name is in parentheses and may hold spaces: split after the last ')'
            table[int(entry.name)] = (fields[0], int(fields[1]), int(fields[2]), int(fields[3]))
        except (OSError, ValueError, IndexError):
            continue
    return table


def process_group_members(pgid: int, proc_root: Path = Path("/proc")) -> list[int]:
    """PIDs of the live (not zombie) processes whose process group or session is ``pgid``: what is left of a child that has exited, such as an engine-core process that outlived it."""
    return sorted(pid for pid, (state, _, group, session) in process_table(proc_root).items() if state != "Z" and pgid in (group, session))


def descendant_pids(root_pid: int, proc_root: Path = Path("/proc")) -> list[int]:
    """PIDs of every process below ``root_pid`` (children, their children, ...): in an engine process that is the engine-core process, the one that holds the GPU memory."""
    children: dict[int, list[int]] = {}
    for pid, (_, parent, _, _) in process_table(proc_root).items():
        children.setdefault(parent, []).append(pid)
    found: list[int] = []
    pending = [root_pid]
    while pending:
        for child in children.get(pending.pop(), []):
            found.append(child)
            pending.append(child)
    return sorted(found)


def alive_pids(pids: Sequence[int], proc_root: Path = Path("/proc")) -> list[int]:
    """Those of ``pids`` that are live (not zombie) processes now."""
    table = process_table(proc_root)
    return sorted(pid for pid in pids if pid in table and table[pid][0] != "Z")


def attention_backends_in_log(path: Path) -> list[str]:
    """The attention backends that the engine's own log line names, in the order they appear (empty when the log is missing or has no such line). The backend is chosen inside the engine-core process
    and cannot be read from the front end without the insecure-serialisation switch of vLLM, so the engine's report of its choice is taken from its log."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found: list[str] = []
    for line in text.splitlines():
        for pattern in ATTENTION_LOG_PATTERNS:
            match = pattern.search(line)
            if match:
                if match.group(1) not in found:
                    found.append(match.group(1))
                break
    return found


def parse_compute_pids(text: str) -> list[int]:
    """PIDs from the output of ``nvidia-smi --query-compute-apps=pid --format=csv,noheader`` (lines that are not a number, such as '[Not Supported]', are ignored)."""
    return sorted({int(line.strip()) for line in text.splitlines() if line.strip().isdigit()})


def nvidia_smi_compute_pids(run: Callable[..., Any] = subprocess.run) -> list[int]:
    try:
        result = run(GPU_QUERY, capture_output=True, text=True, timeout=60, check=False)  # noqa: S603
    except (OSError, subprocess.SubprocessError) as error:
        raise TechnicalFailure(f"cannot run `{' '.join(GPU_QUERY)}`: {error}") from error
    if result.returncode != 0:
        raise TechnicalFailure(f"`{' '.join(GPU_QUERY)}` exited with {result.returncode}: {str(result.stderr).strip()[:200]}")
    return parse_compute_pids(result.stdout)


def nvidia_smi_gpu_info(run: Callable[..., Any] = subprocess.run) -> list[dict[str, str]]:
    """Index, name, driver version and total memory (MiB) of the GPUs, for the record of the software stack. Best effort: an empty list when ``nvidia-smi`` does not answer."""
    try:
        result = run(GPU_INFO_QUERY, capture_output=True, text=True, timeout=60, check=False)  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    rows = []
    for line in str(result.stdout).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            rows.append({"index": parts[0], "name": parts[1], "driver_version": parts[2], "memory_total_mib": parts[3]})
    return rows


def verify_release(outcome: ChildOutcome, hooks: C4Hooks, engine_pids: Sequence[int] = ()) -> dict[str, Any]:
    """The child has exited, nothing of its process group is alive (nor any process it reported below itself, ``engine_pids``: the engine-core process), and ``nvidia-smi`` lists neither its
    PID nor a PID of its group nor an engine PID, dead or alive (the driver may list a PID for a moment after the process has gone). The driver may need a moment to let go, so the check
    repeats for up to ``RELEASE_WAIT_S`` seconds; a check that still fails is reported, not repaired."""
    probe = hooks.gpu_probe or nvidia_smi_compute_pids
    members_of = hooks.group_members or process_group_members
    alive_of = hooks.alive or alive_pids
    sleep = hooks.sleep or time.sleep
    waited = 0.0
    while True:
        members = sorted(set(members_of(outcome.pgid) if outcome.pgid else []) | set(alive_of(list(engine_pids))))
        listed = sorted(set(probe()) & ({outcome.pid} | set(members) | set(engine_pids))) if outcome.pid else []
        exited = outcome.returncode is not None
        ok = exited and not members and not listed
        if ok or waited >= RELEASE_WAIT_S:
            break
        sleep(RELEASE_POLL_S)
        waited += RELEASE_POLL_S
    return {
        "child_pid": outcome.pid, "child_pgid": outcome.pgid, "engine_pids": sorted(engine_pids), "returncode": outcome.returncode, "exited": exited, "group_members_alive": members,
        "gpu_compute_pids_listed": listed, "waited_s": waited, "query": " ".join(GPU_QUERY), "ok": bool(ok),
    }


# ---------------------------------------------------------------------------------------
# the two tables, the pairing and the verdict
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PassTable:
    """What one pass left in its table, in the order of its rows (the order it submitted)."""

    item_keys: list[str]
    logprobs: np.ndarray  # [n, 5]
    input_sha256: list[str]
    n_input_tokens: np.ndarray
    free_token_id: list[int | None]
    draw_pos: np.ndarray | None = None


def read_pass_table(path: Path) -> PassTable:
    table = common.read_parquet(path)
    missing = [c for c in ("item_key", "digit_logprobs", "input_sha256", "n_input_tokens", "free_token_id") if c not in table.column_names]
    if missing:
        raise TechnicalFailure(f"{path.name}: the columns {missing} are missing")
    return PassTable(
        item_keys=table.column("item_key").to_pylist(), logprobs=common.column_array(table, "digit_logprobs").reshape(table.num_rows, -1), input_sha256=table.column("input_sha256").to_pylist(),
        n_input_tokens=np.asarray(table.column("n_input_tokens").to_pylist(), dtype=np.int64), free_token_id=table.column("free_token_id").to_pylist(),
        draw_pos=np.asarray(table.column("draw_pos").to_pylist(), dtype=np.int64) if "draw_pos" in table.column_names else None,
    )


@dataclass(frozen=True)
class Pairing:
    """The two readings of every drawn item, in the draw order (the pairing is made by item key)."""

    keys: tuple[str, ...]
    lp1: np.ndarray
    lp2: np.ndarray
    submit1: np.ndarray
    submit2: np.ndarray
    input_sha256: list[str]
    n_input_tokens: np.ndarray
    free1: list[int | None]
    free2: list[int | None]


def _structure_failures(label: str, table: PassTable, expected_keys: Sequence[str], expected_order: Sequence[str]) -> list[str]:
    n = len(expected_keys)
    failures: list[str] = []
    keys = list(table.item_keys)
    seen: set[str] = set()
    repeated: set[str] = set()
    for key in keys:
        (repeated if key in seen else seen).add(key)
    if len(keys) != n:
        failures.append(f"{label}: {len(keys)} records for {n} drawn items")
    if repeated:
        failures.append(f"{label}: {len(repeated)} item keys occur more than once")
    missing, extra = set(expected_keys) - set(keys), set(keys) - set(expected_keys)
    if missing:
        failures.append(f"{label}: {len(missing)} drawn items have no record")
    if extra:
        failures.append(f"{label}: {len(extra)} records are for items that were not drawn")
    if not failures and keys != list(expected_order):
        failures.append(f"{label}: the rows are not in the submission order the entry fixed")
    lp = np.asarray(table.logprobs, dtype=np.float64)
    if lp.ndim != 2 or lp.shape[0] != len(keys) or lp.shape[1] != 5:
        failures.append(f"{label}: the digit log-probabilities have the shape {lp.shape}, not [{len(keys)}, 5]")
    else:
        with np.errstate(invalid="ignore"):
            bad = np.flatnonzero(np.isnan(lp).any(axis=1) | np.isposinf(lp).any(axis=1))
        if bad.size:
            failures.append(f"{label}: {bad.size} readouts cannot be computed (a digit log-probability is NaN or +inf; rows {bad[:5].tolist()})")
    return failures


def pair_passes(plan: DrawPlan, first: PassTable, second: PassTable) -> tuple[Pairing | None, list[str]]:
    """Check both tables against the plan and restore the pairing by item key. The failures (all of them) are technical: a different set of items, a repeated key, a missing record,
    a readout that cannot be computed, rows that are not in the fixed order, or an item whose input token sequence differs between the passes."""
    failures = _structure_failures("pass 1", first, plan.keys, plan.keys) + _structure_failures("pass 2", second, plan.keys, plan.second_keys)
    if failures:
        return None, failures
    row1 = {k: i for i, k in enumerate(first.item_keys)}
    row2 = {k: i for i, k in enumerate(second.item_keys)}
    r1 = np.asarray([row1[k] for k in plan.keys], dtype=np.int64)
    r2 = np.asarray([row2[k] for k in plan.keys], dtype=np.int64)
    different = [pos for pos in range(len(plan.keys)) if first.input_sha256[r1[pos]] != second.input_sha256[r2[pos]]]
    if different:
        failures.append(f"the input token sequences of {len(different)} items differ between the passes (draw positions {different[:10]})")
    if not np.array_equal(first.n_input_tokens[r1], second.n_input_tokens[r2]):
        failures.append("the number of input tokens of an item differs between the passes")
    if failures:
        return None, failures
    return Pairing(
        keys=plan.keys, lp1=np.asarray(first.logprobs, dtype=np.float64)[r1], lp2=np.asarray(second.logprobs, dtype=np.float64)[r2], submit1=r1, submit2=r2,
        input_sha256=[first.input_sha256[i] for i in r1], n_input_tokens=first.n_input_tokens[r1], free1=[first.free_token_id[i] for i in r1], free2=[second.free_token_id[i] for i in r2],
    ), []


def abs_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| where both are finite; 0 where both are -inf; inf where only one is."""
    with np.errstate(invalid="ignore"):
        d = np.abs(a - b)
    d[np.isneginf(a) & np.isneginf(b)] = 0.0
    return d


@dataclass(frozen=True)
class Judgement:
    """The verdict and the numbers behind it (per-item arrays are in the draw order)."""

    n: int
    required: int
    max_failures: int
    threshold: float
    gate: float
    mass1: np.ndarray
    mass2: np.ndarray
    ok1: np.ndarray  # digit mass at least the gate
    ok2: np.ndarray
    margin1: np.ndarray
    margin2: np.ndarray
    ask1: np.ndarray
    ask2: np.ndarray
    choice1: np.ndarray
    choice2: np.ndarray
    valid_and_equal: np.ndarray  # S_i
    quality_failure: np.ndarray  # a low-quality readout in either pass (an item with two counts once)
    decision_flip: np.ndarray  # both readouts valid, the ask decisions differ
    ask_differs: np.ndarray  # raw: the ask decisions differ, whatever the quality
    both_answer: np.ndarray  # both readouts valid, neither asks
    option_differs: np.ndarray  # ... and the chosen options differ (reported, not judged)
    logprob_diff: np.ndarray  # per item, the largest |difference| of the five digit log-probabilities
    margin_diff: np.ndarray  # per item, |margin 1 - margin 2| (NaN where a margin is undefined)

    @property
    def n_valid_and_equal(self) -> int:
        return int(self.valid_and_equal.sum())

    @property
    def n_quality_failures(self) -> int:
        return int(self.quality_failure.sum())

    @property
    def n_decision_flips(self) -> int:
        return int(self.decision_flip.sum())

    @property
    def agreement(self) -> float:
        return self.n_valid_and_equal / self.n

    @property
    def passed(self) -> bool:
        return self.n_valid_and_equal >= self.required

    @property
    def failure_code(self) -> np.ndarray:
        """Per item, a key of ``FAILURE_CODES``: the reason the item is not valid and equal."""
        code = np.zeros(self.n, dtype=np.int8)
        code[~self.ok1 & self.ok2] = 1
        code[self.ok1 & ~self.ok2] = 2
        code[~self.ok1 & ~self.ok2] = 3
        code[self.decision_flip] = 4
        return code

    def positions(self, mask: np.ndarray) -> list[int]:
        return [int(i) for i in np.flatnonzero(mask)]

    def result(self) -> dict[str, Any]:
        low1, low2 = ~self.ok1, ~self.ok2
        return {
            "passed": self.passed, "n_items": self.n, "n_valid_and_equal": self.n_valid_and_equal, "required_valid_and_equal": self.required, "max_failures": self.max_failures,
            "agreement": self.agreement, "threshold_recorded": self.threshold, "agreement_at_least_threshold": bool(self.agreement + 1e-12 >= self.threshold),
            "n_failures": self.n - self.n_valid_and_equal, "n_decision_flips": self.n_decision_flips, "n_quality_failures": self.n_quality_failures,
            "decision_flip_positions": self.positions(self.decision_flip), "quality_failure_positions": self.positions(self.quality_failure),
            "quality": {
                "digit_mass_gate": self.gate, "n_low_quality_pass1": int(low1.sum()), "n_low_quality_pass2": int(low2.sum()), "n_low_quality_both": int((low1 & low2).sum()),
                "n_low_quality_either": int((low1 | low2).sum()),
            },
            "n_ask_differs_any_quality": int(self.ask_differs.sum()), "failure_codes": {str(k): v for k, v in FAILURE_CODES.items()},
            "rule": ("S_i = 1 when both passes have a usable readout, the digit mass of both is at least the gate and the ask decisions are the same (margin > 0 asks, otherwise answers); "
                     "the denominator is the number of drawn items; passed when sum S_i >= n - max_failures; a low-quality item counts once as a quality failure, a flip is counted only "
                     "among items whose two readouts are valid; --threshold is recorded and compared, the count decides"),
        }

    def descriptive(self) -> dict[str, Any]:
        finite = np.isfinite(self.logprob_diff)
        diff = self.logprob_diff[finite]
        margin = self.margin_diff[np.isfinite(self.margin_diff)]
        return {
            "ask_rate": {"pass1": float(self.ask1.mean()), "pass2": float(self.ask2.mean()), "n_ask_pass1": int(self.ask1.sum()), "n_ask_pass2": int(self.ask2.sum())},
            "options": {
                "n_both_answer": int(self.both_answer.sum()), "n_option_differs": int(self.option_differs.sum()), "option_differs_positions": self.positions(self.option_differs),
                "note": "among the items whose two readouts are valid and answer; reported, not part of the verdict",
            },
            "digit_logprob_diff": {
                "max_abs_finite": float(diff.max()) if diff.size else None, "n_items_nonfinite_difference": int((~finite).sum()), "n_items_identical": int((self.logprob_diff == 0).sum()),
                "n_items_with_any_difference": int((self.logprob_diff > 0).sum()), "summary": common.describe_quantiles(diff.tolist()),
            },
            "digit_mass": {"pass1": common.describe_quantiles(self.mass1.tolist()), "pass2": common.describe_quantiles(self.mass2.tolist())},
            "margin_abs_diff": {"max": float(margin.max()) if margin.size else None, "summary": common.describe_quantiles(margin.tolist())},
        }


def judge(pairing: Pairing, spec: ReadoutSpec, *, max_failures: int, threshold: float) -> Judgement:
    """The composite gate on a pairing. The readouts are derived from the stored digit log-probabilities with the same code as every other entry (``ReadoutSpec.readout``)."""
    n = len(pairing.keys)
    first = [spec.readout([float(x) for x in row]) for row in pairing.lp1]
    second = [spec.readout([float(x) for x in row]) for row in pairing.lp2]

    def column(readouts: Sequence[Any], name: str, dtype: Any) -> np.ndarray:
        return np.asarray([getattr(r, name) for r in readouts], dtype=dtype)

    ok1, ok2 = column(first, "gate_pass", bool), column(second, "gate_pass", bool)
    ask1, ask2 = column(first, "ask", bool), column(second, "ask", bool)
    choice1, choice2 = column(first, "self_choice", np.int64), column(second, "self_choice", np.int64)
    both_valid = ok1 & ok2
    flip = both_valid & (ask1 != ask2)
    both_answer = both_valid & ~ask1 & ~ask2
    margin1, margin2 = column(first, "margin", np.float64), column(second, "margin", np.float64)
    with np.errstate(invalid="ignore"):
        margin_diff = np.abs(margin1 - margin2)
    return Judgement(
        n=n, required=n - max_failures, max_failures=max_failures, threshold=threshold, gate=spec.digit_mass_gate, mass1=column(first, "mass", np.float64), mass2=column(second, "mass", np.float64),
        ok1=ok1, ok2=ok2, margin1=margin1, margin2=margin2, ask1=ask1, ask2=ask2, choice1=choice1, choice2=choice2, valid_and_equal=both_valid & ~flip, quality_failure=~both_valid, decision_flip=flip,
        ask_differs=ask1 != ask2, both_answer=both_answer, option_differs=both_answer & (choice1 != choice2),
        logprob_diff=abs_difference(pairing.lp1, pairing.lp2).max(axis=1), margin_diff=margin_diff,
    )


def item_columns(pairing: Pairing, j: Judgement) -> dict[str, Any]:
    """The table of the report: item keys, digests of the input token sequences and numbers. Nothing of any question or option text."""
    n = j.n
    return {
        "pos": np.arange(n), "item_key": list(pairing.keys), "input_sha256": list(pairing.input_sha256), "n_input_tokens": pairing.n_input_tokens,
        "submit_pos_pass1": pairing.submit1, "submit_pos_pass2": pairing.submit2, "digit_logprobs_pass1": pairing.lp1, "digit_logprobs_pass2": pairing.lp2,
        "digit_mass_pass1": j.mass1, "digit_mass_pass2": j.mass2, "gate_pass_pass1": j.ok1, "gate_pass_pass2": j.ok2, "margin_pass1": j.margin1, "margin_pass2": j.margin2,
        "ask_pass1": j.ask1, "ask_pass2": j.ask2, "self_choice_pass1": j.choice1, "self_choice_pass2": j.choice2,
        "free_token_id_pass1": [-1 if t is None else int(t) for t in pairing.free1], "free_token_id_pass2": [-1 if t is None else int(t) for t in pairing.free2],
        "valid_and_equal": j.valid_and_equal, "quality_failure": j.quality_failure, "decision_flip": j.decision_flip, "failure_code": j.failure_code, "ask_differs": j.ask_differs,
        "option_differs": j.option_differs,
        "max_abs_logprob_diff": j.logprob_diff, "margin_abs_diff": j.margin_diff,
    }


def engine_differences(info1: Mapping[str, Any], info2: Mapping[str, Any]) -> list[str]:
    """The settings that must be the same in the two engines and are not (one line each). What is compared: the arguments passed to vLLM, the settings read back from the engine (the
    volatile ones such as the number of KV-cache blocks excluded), the software versions and the numeric record."""
    def view(info: Mapping[str, Any]) -> dict[str, Any]:
        return {"requested": info["requested"], "settings": info["engine"]["settings"], "environment": info["engine"]["environment"], "software": info["software"], "numeric": info["numeric_actual"],
                "condition": info["condition"]}

    return common.numeric_differences(view(info1), view(info2))


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------

def spec_file_md5s() -> dict[str, str]:
    for key, path in SPEC_FILES.items():
        if not path.is_file():  # not part of this release: one line, and the file is left out of the manifest
            note_skipped(path, f"specification file {key}")
    return {key: common.file_md5(path) for key, path in SPEC_FILES.items() if path.is_file()}


def planned_block(params: Params, plan: DrawPlan, ctx: common.Context, gpu: Sequence[Mapping[str, str]] = ()) -> dict[str, Any]:
    return {
        "kind": "repeated_readout", "model": params.model, "pool": POOL, "n_items": params.n_items, "pilot": params.pilot, "role": params.role, "role_note": ROLE_NOTES[params.role],
        "non_default_options": list(params.non_default),
        "condition": params.condition, "draw_key": list(plan.draw_key), "shuffle_key": list(plan.shuffle_key), "item_keys_sha256": plan.keys_sha256, "second_pass_keys_sha256": plan.second_keys_sha256,
        "rule": {"max_failures": params.max_failures, "required_valid_and_equal": params.required_valid_and_equal, "threshold_recorded": params.threshold, "digit_mass_gate": ReadoutSpec.from_config(ctx.cfg).digit_mass_gate},
        "submission": dict(SUBMISSION), "engine_requested": {"llm_kwargs": dict(params.llm_kwargs)}, "spec_files": {k: str(p.relative_to(common.REPO_ROOT)) for k, p in SPEC_FILES.items()},
        "engine_actual": {}, "scope": SCOPE.format(a_H=params.a_H, c0=params.c0, w=params.w, n=params.n_items), "gpu": [dict(row) for row in gpu],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def read_engine_pids(run_dir: Path, pass_index: int) -> list[int]:
    """The PIDs of the processes below the child that the child wrote down once its engine was up (empty when it never got that far)."""
    try:
        doc = common.read_json(Path(run_dir) / f"pass{pass_index}_processes.json")
        return sorted(int(pid) for pid in doc["descendants"])
    except (OSError, ValueError, KeyError, TypeError):
        return []


def pass_complete(run: common.RunHandle, pass_index: int) -> bool:
    """A pass is finished when its table is there and its record names the md5 of that very table."""
    info_path = run.run_dir / f"pass{pass_index}.json"
    if not run.parts.done(f"pass{pass_index}") or not info_path.is_file():
        return False
    try:
        info = common.read_json(info_path)
        return info.get("status") == "ok" and info.get("table_md5") == common.file_md5(run.parts.path(f"pass{pass_index}"))
    except (OSError, ValueError):
        return False


@dataclass
class RunState:
    """What the parent knows so far; the failure report is made from it."""

    infos: dict[int, dict[str, Any]] = field(default_factory=dict)
    releases: dict[str, dict[str, Any]] = field(default_factory=dict)
    reused: list[int] = field(default_factory=list)
    children: dict[int, dict[str, Any]] = field(default_factory=dict)
    logged_backends: dict[int, list[str]] = field(default_factory=dict)
    report_written: bool = False


def run_pass(run: common.RunHandle, ctx: common.Context, args: argparse.Namespace, params: Params, hooks: C4Hooks, state: RunState, pass_index: int) -> None:
    """Start the child of one pass (or reuse the finished pass of a resumed run), check it, and keep its record."""
    if pass_complete(run, pass_index):
        common.log(f"pass {pass_index}: finished in an earlier session, not run again")
        state.reused.append(pass_index)
    else:
        run.check_source()
        run_dir = run.run_dir
        child_args = child_arguments(ctx, args, params, run_dir, pass_index, run.record.manifest["nmi_source_sha256"])
        command = child_command(child_args)
        pidfile = run_dir / f"pass{pass_index}_child.json"

        def started(pid: int) -> None:  # the PID is on disk before the parent waits: whoever has to stop the child finds it
            common.write_json(pidfile, {"pass": pass_index, "pid": pid, "pgid": pid, "started": common.iso(common.now_local()), "command": command, "returncode": None})
            common.log(f"pass {pass_index}: child process {pid} (its own process group); log {run_dir / 'logs' / f'pass{pass_index}.log'}")

        spec = ChildSpec(pass_index=pass_index, args=child_args, command=command, log_path=run_dir / "logs" / f"pass{pass_index}.log", cwd=common.REPO_ROOT, on_start=started)
        t0 = time.time()
        try:
            outcome = (hooks.launcher or subprocess_launcher)(spec)
        except OSError as error:
            raise TechnicalFailure(f"pass {pass_index}: the child process could not be started: {error}") from error
        except (KeyboardInterrupt, SystemExit):  # SIGTERM arrives here as SystemExit (common.open_run): the child is not stopped, whoever stops it finds its PID in the file
            common.log(f"pass {pass_index}: the parent is stopping; the child may still be running (PID and process group in {pidfile}); check it with ps -p before stopping anything")
            raise
        state.children[pass_index] = {"pid": outcome.pid, "pgid": outcome.pgid, "returncode": outcome.returncode, "wall_s": time.time() - t0, "log": f"logs/pass{pass_index}.log"}
        if pidfile.is_file():
            common.write_json(pidfile, {**common.read_json(pidfile), "returncode": outcome.returncode, "ended": common.iso(common.now_local())})
        run.check_source()
        release = verify_release(outcome, hooks, read_engine_pids(run_dir, pass_index))
        state.releases[f"after_pass{pass_index}"] = release
        common.log(f"pass {pass_index}: child exited with {outcome.returncode}; released: {release['ok']}")
        if outcome.returncode == common.EXIT_STOPPED:
            raise common.SourceChangedError(f"pass {pass_index}: the child found that nmi/ changed")
        if outcome.returncode != 0:
            raise TechnicalFailure(f"pass {pass_index}: the child process exited with code {outcome.returncode} (see logs/pass{pass_index}.log)")
        if not release["ok"] and pass_index == 1:
            raise TechnicalFailure(
                f"pass 1: the first engine process is not gone (still alive in its group: {release['group_members_alive']}; still listed by nvidia-smi: {release['gpu_compute_pids_listed']}); "
                "pass 2 was not started; stop those processes by hand (check each PID with ps -p first)"
            )
        if not pass_complete(run, pass_index):
            raise TechnicalFailure(f"pass {pass_index}: the child exited with 0 but its table and record are not complete (parts/pass{pass_index}.parquet, pass{pass_index}.json)")
        if not release["ok"]:
            common.log(f"WARNING after the last pass: processes of the child's group are still alive {release['group_members_alive']} or listed by nvidia-smi {release['gpu_compute_pids_listed']}: stop them by hand")
    info = common.read_json(run.run_dir / f"pass{pass_index}.json")
    state.infos[pass_index] = info
    state.logged_backends[pass_index] = attention_backends_in_log(run.run_dir / "logs" / f"pass{pass_index}.log")
    run.confirm_numeric(f"vllm_offline_pass{pass_index}", info["numeric_actual"])
    block = run.record.manifest.setdefault("c4_repeat", {})
    block.setdefault("engine_actual", {})[f"pass{pass_index}"] = {
        "settings": info["engine"]["settings"], "volatile": info["engine"]["volatile"], "environment": info["engine"]["environment"], "unreadable": info["engine"]["unreadable"],
        "requested_vs_actual": info["engine"]["requested_vs_actual"], "software": info["software"], "readouts_calls": info["readouts_calls"], "pid": info["pid"],
        "attention_selected_from_log": state.logged_backends[pass_index], "log": f"logs/pass{pass_index}.log",
    }
    run.record.write_manifest()


def check_engines(infos: Mapping[int, Mapping[str, Any]], logged_backends: Mapping[int, Sequence[str]] | None = None) -> list[str]:
    """Failures of the engine records: settings that are required to be read back from the engine and that it did not give, a quantisation that is not the one asked for, and settings that differ
    between the two passes (the attention backends that the two logs name included, when both name one)."""
    failures: list[str] = []
    for k, info in sorted(infos.items()):
        missing = sorted(set(info["engine"]["unreadable"]) & set(REQUIRED_SETTINGS))
        if missing:
            failures.append(f"pass {k}: the engine did not give {missing}")
        quantization = info["engine"]["requested_vs_actual"].get("quantization")
        if quantization is not None and not quantization["same"]:
            failures.append(f"pass {k}: quantization {quantization['requested']!r} was asked for, the engine works with {quantization['actual']!r}")
    if len(infos) == 2:
        differences = engine_differences(infos[1], infos[2])
        if differences:
            failures.append(f"the engines of the two passes do not have the same settings: {differences[:8]}")
        logged = logged_backends or {}
        if logged.get(1) and logged.get(2) and list(logged[1]) != list(logged[2]):
            failures.append(f"the engines of the two passes selected other attention backends (their logs: {list(logged[1])} and {list(logged[2])})")
    return failures


def parent_cuda_state() -> dict[str, Any]:
    """Whether this (parent) process has imported torch and initialised CUDA: it must not have."""
    torch = sys.modules.get("torch")
    initialised = bool(torch.cuda.is_initialized()) if torch is not None and hasattr(torch, "cuda") else False
    return {"torch_imported": torch is not None, "cuda_initialized": initialised}


def write_failure_report(run: common.RunHandle, params: Params, plan: DrawPlan, state: RunState, failures: Sequence[str], t0: float) -> None:
    report = {
        "schema": REPORT_SCHEMA, "status": "technical_failure", "passed": False, "role": params.role, "role_note": ROLE_NOTES[params.role], "pilot": params.pilot, "model": params.model,
        "pool": POOL, "n_items": params.n_items, "condition": params.condition, "non_default_options": list(params.non_default), "failures": list(failures), "result": None,
        "item_keys_sha256": plan.keys_sha256, "second_pass_keys_sha256": plan.second_keys_sha256, "passes_finished": sorted(state.infos), "passes_reused": state.reused,
        "children": state.children, "release": state.releases, "parent": parent_cuda_state(), "wall_s": time.time() - t0,
        "note": "no verdict: the readings cannot be compared or the process that makes them did not behave; nothing is counted (specification section 3)",
    }
    common.write_json(run.run_dir / "c4_repeat.json", report)
    state.report_written = True


def read_tables(run: common.RunHandle) -> dict[int, PassTable]:
    tables: dict[int, PassTable] = {}
    for k in PASSES:
        try:
            tables[k] = read_pass_table(run.parts.path(f"pass{k}"))
        except TechnicalFailure:
            raise
        except Exception as error:  # noqa: BLE001 - a table that cannot be read is a failure of the readings, whatever the reader says
            raise TechnicalFailure(f"pass {k}: the table cannot be read: {type(error).__name__}: {error}") from error
    return tables


def judge_and_report(run: common.RunHandle, ctx: common.Context, params: Params, plan: DrawPlan, state: RunState, t0: float) -> dict[str, Any]:
    tables = read_tables(run)
    failures = check_engines(state.infos, state.logged_backends)
    pairing, pair_failures = pair_passes(plan, tables[1], tables[2])
    failures += pair_failures
    if failures or pairing is None:
        write_failure_report(run, params, plan, state, failures, t0)
        raise TechnicalFailure("; ".join(failures))
    spec = ReadoutSpec.from_config(ctx.cfg)
    verdict = judge(pairing, spec, max_failures=params.max_failures, threshold=params.threshold)
    common.write_parquet(run.run_dir / "c4_repeat_items.parquet", item_columns(pairing, verdict))
    first = state.infos[1]
    weights = {"quantization": first["engine"]["settings"]["model"]["quantization"], "dtype": first["engine"]["settings"]["model"]["dtype"],
               "checkpoint_quantization": first["engine"]["settings"]["model"].get("checkpoint_quantization")}
    report = {
        "schema": REPORT_SCHEMA, "status": "ok", "passed": verdict.passed, "role": params.role, "role_note": ROLE_NOTES[params.role], "pilot": params.pilot, "model": params.model, "pool": POOL,
        "n_items": params.n_items, "condition": params.condition, "registered_protocol": params.role == "primary", "non_default_options": list(params.non_default), "options": {"threshold": params.threshold, "max_failures": params.max_failures},
        "draw_key": list(plan.draw_key), "shuffle_key": list(plan.shuffle_key), "item_keys_sha256": plan.keys_sha256, "second_pass_keys_sha256": plan.second_keys_sha256,
        "second_pass_order": list(plan.second_order), "second_order_is_identity": plan.second_order == tuple(range(len(plan.keys))), "result": verdict.result(), "descriptive": verdict.descriptive(),
        "weights": weights, "engines_identical": True,
        "passes": {f"pass{k}": {key: info[key] for key in ("pid", "started", "ended", "engine_start_s", "read_s", "wall_s", "n_prompts", "order", "table_md5")} for k, info in sorted(state.infos.items())},
        "release": state.releases, "children": state.children, "passes_reused": state.reused, "parent": parent_cuda_state(),
        "scope": SCOPE.format(a_H=params.a_H, c0=params.c0, w=params.w, n=params.n_items), "submission": dict(SUBMISSION),
        "files": [name for name in ("manifest.json", "c4_repeat.json", "c4_repeat_items.parquet", "pass1.json", "pass2.json", "pass1_child.json", "pass2_child.json", "pass1_processes.json",
                                    "pass2_processes.json", "parts/pass1.parquet", "parts/pass2.parquet", "logs/pass1.log", "logs/pass2.log", "llm/model.json") if name == "c4_repeat.json" or name == "c4_repeat_items.parquet" or (run.run_dir / name).is_file()],
        "wall_s": time.time() - t0,
    }
    common.write_json(run.run_dir / "c4_repeat.json", report)
    run.record.manifest.setdefault("c4_repeat", {})["weights"] = weights
    run.record.manifest["c4_repeat"]["result"] = {k: report["result"][k] for k in ("passed", "n_valid_and_equal", "required_valid_and_equal", "agreement", "n_decision_flips", "n_quality_failures")}
    run.record.write_manifest()
    result = report["result"]
    return {
        "run_dir": str(run.run_dir), "model": params.model, "role": params.role, "passed": verdict.passed, "n_items": params.n_items, "n_valid_and_equal": result["n_valid_and_equal"],
        "required_valid_and_equal": result["required_valid_and_equal"], "agreement": result["agreement"], "n_decision_flips": result["n_decision_flips"],
        "n_quality_failures": result["n_quality_failures"], "quantization": weights["quantization"], "dtype": weights["dtype"], "wall_s": time.time() - t0,
    }


def checked_model(ctx: common.Context, model: str) -> Mapping[str, Any]:
    """The device entry of the model, after the checks that a child would otherwise fail on: it has the chat-template arguments of the configuration and its directory has the configuration and
    the tokenizer files (``EntryError``, exit 2, before any run directory is made)."""
    entry = common.device_entry(ctx.cfg, model)
    common.chat_kwargs_of(entry)
    (report,) = common.check_model_dirs(ctx.cfg, [("device", model)])
    if not (report["is_dir"] and report["has_config"] and report["has_tokenizer"]):
        raise common.EntryError(f"the model directory of {model!r} is not complete: {report}")
    return entry


def execute(ctx: common.Context, args: argparse.Namespace, hooks: C4Hooks) -> dict[str, Any]:
    params = resolve_parameters(ctx, args)
    common.require_gpu_memory_fraction(args)
    entry = checked_model(ctx, args.model)
    plan = build_plan(ctx, params.n_items)
    extra_md5 = spec_file_md5s()
    numeric = common.numeric_plan({
        f"vllm_offline_pass{k}": common.vllm_backend_plan(args, unit={"pass": k, "n_prompts": params.n_items, "order": "draw" if k == 1 else "shuffled"}) for k in PASSES
    })
    gpu = (hooks.gpu_info or nvidia_smi_gpu_info)()
    t0 = time.time()
    with common.open_run(
        ctx, pools=[POOL], seed_keys=[plan.draw_key, plan.shuffle_key], numeric=numeric, extra_manifest={"c4_repeat": planned_block(params, plan, ctx, gpu)}, extra_md5=extra_md5,
    ) as run:
        state = RunState()
        try:
            (hooks.gpu_probe or nvidia_smi_compute_pids)()  # the release check must work BEFORE the first engine is started
            for k in PASSES:
                run_pass(run, ctx, args, params, hooks, state, k)
            if hooks.model_files and not (run.run_dir / "llm" / "model.json").is_file():
                first = state.infos[1]["engine"]["settings"]["model"]
                common.write_model_files(run, entry, None, model_dir=Path(entry["path"]), dtype=str(first["dtype"]),
                                         serve_params={"llm_kwargs": dict(params.llm_kwargs), "quantization": first["quantization"], "process_per_pass": True})
            return judge_and_report(run, ctx, params, plan, state, t0)
        except TechnicalFailure as failure:
            if not state.report_written:  # the failures of the comparison are in the report already; the others (a child that failed, a release that did not happen) are written here
                write_failure_report(run, params, plan, state, [str(failure)], t0)
            raise


# ---------------------------------------------------------------------------------------
# dry run and command line
# ---------------------------------------------------------------------------------------

def dry_run(ctx: common.Context, args: argparse.Namespace) -> int:
    dry = common.DryRun(ctx)
    try:
        params = resolve_parameters(ctx, args)
        plan = build_plan(ctx, params.n_items)
    except (common.EntryError, SplitError, seeds.SeedKeyError) as error:
        dry.check("arguments and inputs", False, str(error))
        return dry.emit()
    common.dry_run_basics(dry, pools=[POOL], models=[("device", args.model)], seed_keys=[plan.draw_key, plan.shuffle_key])
    try:
        dry.check("chat template arguments", True, common.chat_kwargs_of(common.device_entry(ctx.cfg, args.model)))
    except common.EntryError as error:
        dry.check("chat template arguments", False, str(error))
    dry.check("specification files bound in the manifest", True,
              {k: {"path": str(p), "md5": common.file_md5(p) if p.is_file() else None, "skipped": not p.is_file()} for k, p in SPEC_FILES.items()})
    dry.check("items", params.n_items <= plan.n_pool, {"pool": POOL, "n_pool": plan.n_pool, "n_items": params.n_items})
    if getattr(args, "gpu_mem_util", None) is not None:
        child = child_command(child_arguments(ctx, args, params, Path("<run_dir>"), 1, "<nmi source sha256>"))
    else:
        child = None
    dry.note("workload", {
        "pool": POOL, "n_pool": plan.n_pool, "n_items": params.n_items, "role": params.role, "non_default_options": list(params.non_default), "condition": params.condition,
        "draw_key": list(plan.draw_key), "shuffle_key": list(plan.shuffle_key), "item_keys_sha256": plan.keys_sha256, "second_pass_keys_sha256": plan.second_keys_sha256,
        "second_order_is_identity": plan.second_order == tuple(range(len(plan.keys))),
        "passes": [{"pass": 1, "prompts": params.n_items, "order": "the draw order"}, {"pass": 2, "prompts": params.n_items, "order": "the shuffled order of the second key"}],
        "readouts": 2 * params.n_items, "engine_kwargs": dict(params.llm_kwargs), "child_command_pass_1": child,
        "release_check": (f"between the passes: the first child has exited, no live process of its group or below it, and neither its PID nor a PID of its group or below it in `{' '.join(GPU_QUERY)}`"),
        "rule": {"max_failures": params.max_failures, "required_valid_and_equal": params.required_valid_and_equal, "threshold_recorded": params.threshold},
        "outputs": f"{ctx.results_root}/{STAGE}/<time>_{ctx.tag or '<tag>'}/",
    })
    return dry.emit()


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C4: the repeated readout of one model in two independent vLLM engine processes.")
    common.add_common_arguments(p)
    g = p.add_argument_group("the repeated readout")
    g.add_argument("--model", required=True, help="end-side model id of the configuration (models.devices[].id)")
    g.add_argument("--gpu-mem-util", type=float, default=None, help="vLLM gpu_memory_utilization (required: a fraction of the card's total memory, at most the free memory that nvidia-smi shows now / total)")
    g.add_argument("--llm-kwargs", default=None, help="JSON object merged into the keyword arguments of vllm.LLM (for example '{\"quantization\": \"fp8\"}' for the 14B model)")
    g.add_argument("--max-model-len", type=int, default=4096, help="vLLM max_model_len (default 4096)")
    g.add_argument("--n-items", type=int, default=None, help=f"items read in each pass: the first n of the cross-check draw (default {REGISTERED['n_items']})")
    g.add_argument("--a-h", dest="a_H", type=float, default=REGISTERED["a_H"], help=f"a_H of the P-ask prompt (default {REGISTERED['a_H']})")
    g.add_argument("--c0", type=float, default=REGISTERED["c0"], help=f"c0 of the P-ask prompt (default {REGISTERED['c0']:g})")
    g.add_argument("--w", type=float, default=REGISTERED["w"], help=f"displayed waiting cost w of the P-ask prompt (default {REGISTERED['w']:g}; 0.2 is the bridge to the cross-check condition, a description)")
    g.add_argument("--threshold", type=float, default=REGISTERED["threshold"], help=f"agreement threshold, recorded and compared with the agreement; the count decides (default {REGISTERED['threshold']})")
    g.add_argument("--max-failures", type=int, default=REGISTERED["max_failures"], help=f"items that may fail the valid-and-equal gate (default {REGISTERED['max_failures']}: 299 of 300)")
    g.add_argument("--pilot", action="store_true", help=f"pre-flight: read only --pilot-items items per pass (at most {PILOT_ITEMS_MAX}); the role of the run is 'pilot'")
    g.add_argument("--pilot-items", type=int, default=None, help=f"items per pass of a pilot (default and maximum {PILOT_ITEMS_MAX})")
    p.add_argument("--child-pass", type=int, choices=PASSES, default=None, help=argparse.SUPPRESS)  # internal: one pass in this process
    p.add_argument("--child-run-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--child-source-sha256", default=None, help=argparse.SUPPRESS)
    p.set_defaults(backend="vllm_offline")  # the offline engine: the shared dry-run checks name the memory fraction for it
    return p


def main(argv: Sequence[str] | None = None, hooks: C4Hooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or C4Hooks()
    args = make_parser().parse_args(argv)
    try:
        ctx = common.build_context(args, script=SCRIPT, stage=STAGE, argv=argv)
        if args.child_pass is not None:
            if args.dry_run or args.resume:
                raise common.EntryError("--child-pass is internal: it takes neither --dry-run nor --resume")
            return execute_child(ctx, args, hooks)
        if args.dry_run:
            return dry_run(ctx, args)
        outcome = execute(ctx, args, hooks)
    except (common.EntryError, SplitError, paths.PathPolicyError, seeds.SeedKeyError, LabelExistsError) as error:  # a tag that is in use is an argument the user can fix
        print(f"error: {error}", file=sys.stderr)
        return 2
    except common.SourceChangedError as error:
        return common.stopped_by_source_change(error)
    except TechnicalFailure as error:
        print(f"technical failure: {error}", file=sys.stderr)
        return 1
    print(json.dumps(common.json_safe(outcome), indent=2, sort_keys=True))
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
