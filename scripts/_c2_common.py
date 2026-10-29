"""Shared code of the GPU-side ``run_*`` entry points (for example ``run_calib`` and ``run_decisions``).

Nothing here runs a model by itself. Heavy libraries (torch, transformers, vllm) are imported inside the
functions that need them, so ``--dry-run`` stays light. What this module holds:

* the common command-line arguments (``add_common_arguments``) and the ``Context`` every entry builds
  from them (configuration, prompt book, seed namespace, split registry, results root);
* ``DryRun``: the report of ``--dry-run`` (configuration and prompt checks, the split, input
  hashes, model directories, seed keys, the output directory, the planned workload). It never loads a
  model, never touches a GPU and never puts a question text into memory or output: the input files are only
  hashed (``SplitRegistry.verify_input``), and the key lists are hashes;
* the run lifecycle (``open_run``): ``nmi.provenance.begin_run`` with the md5 of the two yaml files, the
  pre-registration (and the amendment files that the configuration lists) and the split, the seed keys, the
  ``SourceHashGuard`` that stops the dispatch of work when ``nmi/`` changes, the SIGTERM handler that writes
  ``aborted.json``, and resuming an interrupted run. The manifest also gets a ``numeric`` block (the dtype, device and
  back end, the attention implementation, the versions of torch, transformers and vllm, autocast, TF32, the batch size
  and the batch plan: what the numbers of the run were computed with); a resumed run must match it, as it must match
  the input hashes;
* ``PartStore`` and ``run_jobs``: work is cut into units, each unit is written atomically as one Parquet
  part, a resumed run skips the finished units;
* readout back ends (``open_backend``: Hugging Face, offline vLLM, HTTP) and the table columns of the
  decision readouts;
* the readers of the artifacts of other runs (``read_screening_run`` and the others) and the run ids of those upstream
  runs (``parent_run_ids``), which an entry passes to ``open_run`` as the parents of its run;
* small file helpers (strict JSON, Parquet).

The Parquet tables carry the specified columns in a fixed order, followed by a few extra columns that are named in
the entry that writes them.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (REPO_ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from nmi import paths, seeds  # noqa: E402
from nmi.config import Config, ConfigError, load_config  # noqa: E402
from nmi.data.registry import (  # noqa: E402
    BANK_MMLU,
    BANK_SIMPLEQA,
    POOL_BANK,
    POOL_NAMES,
    InputHashMismatchError,
    McItem,
    SplitError,
    SplitRegistry,
    is_holdout_pool,
    load_items_from_config,
)
from nmi.prompts import DisplayRule, PromptBook, render_chat  # noqa: E402
from nmi.provenance import (  # noqa: E402
    RunRecord,
    SourceChangedError,
    SourceHashGuard,
    begin_run,
    file_md5,
    iso,
    note_skipped,
    now_local,
    source_hash,
    write_model_json,
)
from nmi.readout.digits import ReadoutSpec  # noqa: E402
from nmi.readout.hf_backend import TrialReadout  # noqa: E402
from nmi.seeds import SeedNamespace  # noqa: E402

# scratch state and the model-hash cache live under ``paths.TMP_ROOT``; ``paths`` refuses writes to the root partition for
# the directories it hands out, and this module checks the few paths that come from the command line
HASH_CACHE_PATH = paths.TMP_ROOT / "model_hash_cache.json"

CAL_POOLS = ("mmlu_redux_cal", "simpleqa_mc_cal")
EVAL_POOLS = ("mmlu_redux_eval", "simpleqa_mc_eval")
# MMLU-Redux is the main bank (its evaluation pool is the main one, SimpleQA is the robustness check); the prompt screening and the decision
# grid default to its pools, and the SimpleQA pools are opt-in (``--pools``), each in a run of its own: the CPU-side readers take one bank per decisions table.
MAIN_CAL_POOL = "mmlu_redux_cal"
MAIN_EVAL_POOL = "mmlu_redux_eval"


# exit codes of the entries: 0 done, 1 the run completed but the check it makes did not pass (for example the loop validity), 2 unusable input or arguments
# (nothing was computed), 4 the source guard stopped the dispatch (the code under nmi/ changed while the batch ran)
EXIT_STOPPED = 4


class EntryError(RuntimeError):
    """A failure the user can fix (bad argument, missing input, refused request); the entry prints it and exits 2."""


def json_safe(obj: Any) -> Any:
    """Non-finite floats become ``None`` (strict JSON); containers are converted recursively."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    return obj


def stopped_by_source_change(error: SourceChangedError) -> int:
    """What the ``main`` of an entry returns when ``SourceHashGuard`` fired. The run record has been closed as ``error`` by then and keeps the finished units."""
    print(
        f"stopped: {error}\nThe run directory keeps what was finished. Restore nmi/ and continue with --resume RUN_DIR (the code must be identical to that of the start), "
        "or start the run again under a new --tag.", file=sys.stderr,
    )
    return EXIT_STOPPED


# ---------------------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------------------

def log(message: str) -> None:
    """One progress line on stderr with the local time (the stdout of an entry is reserved for its report)."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def csv_list(text: str | None) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()] if text else []


def write_json(path: str | Path, obj: Any) -> Path:
    """Strict JSON (non-finite floats become null), atomic, sorted keys. Refuses paths on the root partition."""
    target = paths.assert_off_root_partition(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(json_safe(obj), fh, indent=2, sort_keys=True, allow_nan=False, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    return target


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def to_arrow(values: Any) -> pa.Array:
    """Arrow column of a python list or a numpy array; a 2-D array becomes a fixed-size list column."""
    if isinstance(values, np.ndarray) and values.ndim == 2:
        flat = pa.array(np.ascontiguousarray(values).reshape(-1))
        return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return pa.array(values)
    return pa.array(list(values))


def table_from_columns(columns: Mapping[str, Any]) -> pa.Table:
    n = {len(v) for v in columns.values()}
    if len(n) > 1:
        raise EntryError(f"table columns have different lengths: { {k: len(v) for k, v in columns.items()} }")
    return pa.table({name: to_arrow(values) for name, values in columns.items()})


def write_parquet(path: str | Path, columns: Mapping[str, Any] | pa.Table) -> Path:
    target = paths.assert_off_root_partition(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = columns if isinstance(columns, pa.Table) else table_from_columns(columns)
    tmp = target.with_name(target.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, target)
    return target


def read_parquet(path: str | Path) -> pa.Table:
    return pq.read_table(path)


def column_array(table: pa.Table, name: str) -> np.ndarray:
    """A column as a numpy array (a list column becomes 2-D)."""
    col = table.column(name)
    if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
        return np.asarray(col.to_pylist(), dtype=np.float64)
    return col.to_numpy(zero_copy_only=False)


# ---------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("run (common to every entry)")
    group.add_argument("--config", default=None, help="configuration file (default: configs/v6_confirm.yaml)")
    group.add_argument("--expect-config-md5", default=None, help="stop unless the configuration has exactly this md5 (the frozen value)")
    group.add_argument("--results-root", default=None, help="default: paths.results_root of the configuration, relative to the repository")
    group.add_argument("--split-dir", default=None, help="frozen split directory (default: data/splits/v1)")
    group.add_argument("--tag", default=None, help="run tag: results/<stage>/<time>_<tag>/ (required unless --dry-run; an existing tag is an error)")
    group.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue an interrupted run in its own directory; finished units are skipped")
    group.add_argument("--dry-run", action="store_true", help="check the configuration, the input hashes, the seeds, the output directory and the arguments; print the plan; load no model")


def add_readout_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("readout back end")
    group.add_argument("--model", required=True, help="end-side model id of the configuration (models.devices[].id)")
    group.add_argument("--backend", choices=("vllm_offline", "hf", "vllm_http"), default="vllm_offline")
    group.add_argument("--hf-batch-size", type=int, default=16)
    group.add_argument("--gpu-mem-util", type=float, default=None, help="vLLM gpu_memory_utilization (offline); set it from the free memory measured before the run")
    group.add_argument("--max-model-len", type=int, default=4096)
    group.add_argument("--llm-kwargs", default=None, help="JSON object merged into the keyword arguments of vllm.LLM (offline)")
    group.add_argument("--http-url", default=None, help="base URL of a running vLLM server (backend vllm_http)")
    group.add_argument("--http-model", default=None, help="served model name of that server (default: the model id)")
    group.add_argument("--limit-items", type=int, default=None, help="pilot: a seeded subset of this many items per pool")
    group.add_argument("--pilot-rep", type=int, default=None, help="registered development pilot seed (9001-9006) of the subset")


@dataclass
class Context:
    args: argparse.Namespace
    cfg: Config
    book: PromptBook
    ns: SeedNamespace
    results_root: Path
    split_dir: Path
    stage: str
    script: str
    argv: list[str]
    _registry: SplitRegistry | None = field(default=None, repr=False)

    def registry(self) -> SplitRegistry:
        if self._registry is None:
            self._registry = SplitRegistry.load(self.split_dir)
        return self._registry

    @property
    def tag(self) -> str | None:
        return getattr(self.args, "tag", None)

    @property
    def prompts_path(self) -> Path:
        return _anchored(self.cfg["prompts"]["file"])


def _anchored(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def build_context(args: argparse.Namespace, *, script: str, stage: str, argv: Sequence[str]) -> Context:
    try:
        cfg = load_config(args.config, expected_md5=args.expect_config_md5)
    except (ConfigError, OSError) as error:
        raise EntryError(f"configuration: {error}") from error
    book = PromptBook.load(_anchored(cfg["prompts"]["file"]), rule=DisplayRule.from_config(cfg))
    results_root = paths.results_root(args.results_root or cfg["paths"]["results_root"])
    split_dir = Path(args.split_dir) if args.split_dir else paths.split_dir()
    return Context(
        args=args, cfg=cfg, book=book, ns=SeedNamespace.from_config(cfg), results_root=results_root,
        split_dir=split_dir, stage=stage, script=script, argv=[f"scripts/{script}.py", *argv],
    )


# ---------------------------------------------------------------------------------------
# pools and items
# ---------------------------------------------------------------------------------------

def bank_of(pool: str) -> str:
    return POOL_BANK[pool]


def split_of(pool: str) -> str:
    """``cal`` or ``eval`` (the shuffled evaluation view counts as ``eval``)."""
    return "cal" if pool.endswith("_cal") else "eval"


def validate_pools(pools: Iterable[str]) -> list[str]:
    out = list(pools)
    for pool in out:
        if pool not in POOL_NAMES:
            raise EntryError(f"unknown pool {pool!r}; known: {sorted(POOL_NAMES)}")
        if is_holdout_pool(pool):
            raise EntryError(f"pool {pool!r} is a holdout pool: no kill-test entry reads it (it stays sealed)")
    if len(set(out)) != len(out):
        raise EntryError("a pool is listed twice")
    return out


def pilot_subset(ctx: Context, n_items: int, limit: int | None, pilot_rep: int | None) -> np.ndarray:
    """Positions of a seeded subset of ``limit`` items (sorted, so the pool order is kept)."""
    if limit is None:
        return np.arange(n_items)
    if pilot_rep is None:
        raise EntryError("--limit-items needs --pilot-rep (a registered development pilot seed, 9001-9006)")
    if limit < 1:
        raise EntryError("--limit-items must be positive")
    key = ctx.ns.development_pilot_key(pilot_rep)
    pick = seeds.rng(key).permutation(n_items)[: min(limit, n_items)]
    return np.sort(pick)


def load_pool_items(ctx: Context, pool: str, *, limit: int | None = None, pilot_rep: int | None = None) -> list[McItem]:
    """Items of one calibration or evaluation pool, through the frozen locator map (never a holdout pool)."""
    validate_pools([pool])
    items = load_items_from_config(ctx.cfg, pool, registry=ctx.registry(), results_root=ctx.results_root)
    ctx.registry().assert_no_holdout([item.item_key for item in items], context=pool)
    keep = pilot_subset(ctx, len(items), limit, pilot_rep)
    return [items[i] for i in keep]


def pool_sizes(ctx: Context, pools: Iterable[str], limit: int | None = None) -> dict[str, int]:
    reg = ctx.registry()
    return {pool: min(len(reg.keys(pool)), limit) if limit else len(reg.keys(pool)) for pool in pools}


# ---------------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------------

def device_entry(cfg: Config, model_id: str) -> Mapping[str, Any]:
    try:
        return cfg.device(model_id)
    except KeyError as error:
        ids = [d["id"] for d in cfg["models"]["devices"]]
        raise EntryError(f"unknown end-side model {model_id!r}; the configuration lists {ids}") from error


def helper_entry(cfg: Config, helper_id: str) -> Mapping[str, Any]:
    try:
        return cfg.helper(helper_id)
    except KeyError as error:
        ids = [h["id"] for h in cfg["models"]["helpers"]]
        raise EntryError(f"unknown helper {helper_id!r}; the configuration lists {ids}") from error


def chat_kwargs_of(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The ``chat_template_kwargs`` a model entry carries in the configuration (Qwen3 and Gemma-4: ``enable_thinking: false``).

    A model entry without them is refused: the Qwen3 template thinks by default, so a missing key would silently run every prompt in thinking mode."""
    if "chat_template_kwargs" not in entry:
        raise EntryError(f"model {entry.get('id')!r} has no chat_template_kwargs in the configuration (the kill test runs every model with enable_thinking false)")
    return dict(entry["chat_template_kwargs"])


def answer_texts(book: PromptBook, tokenizer: Any, items: Sequence[McItem], chat_kwargs: Mapping[str, Any]) -> list[str]:
    """P-answer prompts of an END-SIDE model through the chat template (one user message) and, when the prompt file has one, the assistant prefix
    after the start of the assistant turn (``PromptBook.answer_chat_text``; a prompt file without a prefix gives the plain chat text).
    The helper's direct answer renders the prompt itself and takes no prefix."""
    return [book.answer_chat_text(tokenizer, i.question, i.choices, chat_kwargs) for i in items]


def ask_texts(
    book: PromptBook,
    tokenizer: Any,
    items: Sequence[McItem],
    chat_kwargs: Mapping[str, Any],
    *,
    paraphrase: str,
    framing: str,
    a_H: float,
    c0: float,
    w: float | None = None,
    D_hat_s: float | None = None,
    kappa: float | None = None,
    layout: str = "standard",
) -> list[str]:
    """P-ask prompts of one condition for every item."""
    return [
        render_chat(
            tokenizer,
            book.ask_prompt(
                paraphrase, i.question, i.choices, framing=framing, a_H=a_H, c0=c0, w=w, D_hat_s=D_hat_s,
                kappa=kappa, layout=layout,
            ),
            chat_kwargs,
        )
        for i in items
    ]


def load_tokenizer(path: str | Path, *, left_padded: bool = False) -> Any:
    """The tokenizer files of a model directory (offline). ``left_padded`` for the steering forward."""
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"padding_side": "left"} if left_padded else {}
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, **kwargs)


# ---------------------------------------------------------------------------------------
# numeric settings of a run
# ---------------------------------------------------------------------------------------

NUMERIC_PACKAGES = ("torch", "transformers", "vllm")
HF_BATCH_RULE = "token ids sorted by (length, position), consecutive chunks of batch_size, left padded by hand"
VLLM_BATCH_RULE = "continuous batching of the offline engine (the batch composition is the engine's)"


def installed_versions() -> dict[str, str | None]:
    """Versions of torch, transformers and vllm from the package metadata (nothing is imported)."""
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in NUMERIC_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def torch_numerics() -> dict[str, Any]:
    """The numerical switches of torch as they are NOW, read back (``{}`` where torch cannot be imported): TF32 for matrix products and for cuDNN, the float32
    matmul precision, the new-style ``fp32_precision`` switches where this torch has them, and whether autocast is on (CUDA, CPU). ``tf32_effective`` is true if any of
    them lets a float32 product run in TF32."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return {}

    def autocast(device_type: str) -> bool:
        try:
            return bool(torch.is_autocast_enabled(device_type))
        except TypeError:  # an older torch: no argument, CUDA only
            return bool(torch.is_autocast_enabled()) if device_type == "cuda" else False

    new_style = {
        "matmul": getattr(torch.backends.cuda.matmul, "fp32_precision", None),
        "cudnn": getattr(torch.backends.cudnn, "fp32_precision", None),
        "cudnn_conv": getattr(getattr(torch.backends.cudnn, "conv", None), "fp32_precision", None),
        "cudnn_rnn": getattr(getattr(torch.backends.cudnn, "rnn", None), "fp32_precision", None),
    }
    out: dict[str, Any] = {
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp32_precision": new_style,
        "autocast_cuda": autocast("cuda"),
        "autocast_cpu": autocast("cpu"),
    }
    out["tf32_effective"] = bool(out["tf32_matmul"] or out["tf32_cudnn"] or out["float32_matmul_precision"] != "highest" or "tf32" in {v for v in new_style.values() if v is not None})
    return out


def apply_strict_float32() -> dict[str, Any]:
    """Switch TF32 off for matrix products and cuDNN and set the float32 matmul precision to ``highest``, and return ``torch_numerics()`` read back afterwards (what is
    in force counts, not what was asked). The entries never use autocast; the read-back says so too."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch_numerics()


def assert_strict_float32(numerics: Mapping[str, Any]) -> dict[str, Any]:
    """The read-back of ``apply_strict_float32``, or ``EntryError`` if TF32 or autocast is still in force (the float32 reference would not be one)."""
    if numerics.get("tf32_effective") or numerics.get("autocast_cuda") or numerics.get("autocast_cpu"):
        raise EntryError(f"TF32 or autocast is still on after the switches were set: {dict(numerics)}")
    return dict(numerics)


def hf_batch_plan(backend: Any) -> dict[str, Any]:
    """The batch plan of an ``HFBackend``: the rule and the batch size."""
    return {"rule": HF_BATCH_RULE, "batch_size": getattr(backend, "batch_size", None), "sort_by_length": getattr(backend, "sort_by_length", None)}


def hf_numeric_actual(model: Any, *, batch_size: int | None, batch_plan: Mapping[str, Any]) -> dict[str, Any]:
    """The ``actual`` numeric record of a loaded Hugging Face model: dtype and device of its parameters, the device name and compute capability on a GPU, the attention
    implementation of its configuration (and of its text configuration where it has one), the torch switches as they are now, the batch size and the batch plan."""
    import torch

    param = next(model.parameters())
    device = param.device
    name: str | None = None
    capability: list[int] | None = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name, capability = torch.cuda.get_device_name(index), list(torch.cuda.get_device_capability(index))
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None)
    numerics = torch_numerics()
    return {
        "kind": "hf", "dtype": str(param.dtype).removeprefix("torch."), "device": str(device), "device_name": name, "compute_capability": capability,
        "attention_implementation": getattr(cfg, "_attn_implementation", None), "attention_implementation_text": getattr(text_cfg, "_attn_implementation", None),
        "autocast": {"cuda": numerics.get("autocast_cuda"), "cpu": numerics.get("autocast_cpu")},
        "tf32": {k: numerics.get(k) for k in ("tf32_matmul", "tf32_cudnn", "float32_matmul_precision", "tf32_effective")}, "fp32_precision": numerics.get("fp32_precision"),
        "batch_size": batch_size, "batch_plan": dict(batch_plan), "versions": installed_versions(),
    }


def vllm_numeric_actual(engine: Any, serve_params: Mapping[str, Any]) -> dict[str, Any]:
    """The ``actual`` numeric record of an offline vLLM engine, as far as the front-end process can see it: the dtype of the model configuration, the keyword arguments the engine was
    started with (without the memory fraction, which changes no number), the attention backend that was REQUESTED (the one selected is chosen inside the engine-core process
    and is not read here: ``selected`` is None) and the environment variable that can set it."""
    kwargs = {k: v for k, v in dict(serve_params.get("llm_kwargs", {})).items() if k != "gpu_memory_utilization"}
    dtype = None
    try:
        dtype = str(engine.llm_engine.model_config.dtype).removeprefix("torch.")
    except Exception:  # noqa: BLE001
        dtype = None
    return {
        "kind": "vllm_offline", "dtype": dtype or kwargs.get("dtype"), "dtype_requested": kwargs.get("dtype"), "device": "cuda",
        "attention_implementation": {"requested": kwargs.get("attention_config", kwargs.get("attention_backend")), "environment": os.environ.get("VLLM_ATTENTION_BACKEND"), "selected": None},
        "autocast": None, "tf32": None, "batch_size": None, "batch_plan": {"rule": VLLM_BATCH_RULE, "engine_kwargs": kwargs}, "versions": installed_versions(),
    }


def hf_backend_plan(*, dtype: str, device: str, batch_size: int, strict_float32: bool, unit: Mapping[str, Any] | None = None, rule: str = HF_BATCH_RULE, max_batch_tokens: int | None = None) -> dict[str, Any]:
    """What an entry intends before it loads a Hugging Face model (the ``planned`` numeric record): dtype, device, batch size, batch plan, and whether TF32 is switched off
    (``"strict float32"``) or left as torch has it. ``unit`` names how the entry cuts its work into units, when that changes what is read together."""
    plan_rule: dict[str, Any] = {"rule": rule, "batch_size": batch_size}
    if max_batch_tokens is not None:
        plan_rule["max_batch_tokens"] = max_batch_tokens
    if unit:
        plan_rule["unit"] = dict(unit)
    return {"kind": "hf", "dtype": dtype, "device": device, "batch_size": batch_size, "batch_plan": plan_rule, "numerics": "strict float32: no TF32, no autocast" if strict_float32 else "torch defaults"}


def vllm_backend_plan(args: argparse.Namespace, *, unit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The same for offline vLLM: the engine keyword arguments (without the memory fraction) and the unit of work."""
    kwargs = {k: v for k, v in vllm_llm_kwargs(args).items() if k != "gpu_memory_utilization"}
    plan_rule: dict[str, Any] = {"rule": VLLM_BATCH_RULE, "engine_kwargs": kwargs}
    if unit:
        plan_rule["unit"] = dict(unit)
    return {"kind": "vllm_offline", "dtype": kwargs.get("dtype"), "device": "cuda", "batch_size": None, "batch_plan": plan_rule, "numerics": "engine"}


def http_backend_plan(args: argparse.Namespace) -> dict[str, Any]:
    """A server that the entry only talks to: its numerics are the server's, the record says so."""
    return {"kind": "vllm_http", "dtype": None, "device": None, "batch_size": None, "batch_plan": {"rule": "the server's"}, "numerics": "the server's", "url": getattr(args, "http_url", None)}


def readout_backend_plan(args: argparse.Namespace, kind: str | None = None, *, unit: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """The planned numeric record of the back end that ``--backend`` (or ``kind``) names, keyed by its kind, for ``numeric_plan``."""
    kind = kind or getattr(args, "backend", "vllm_offline")
    if kind == "hf":
        return {"hf": hf_backend_plan(dtype="bfloat16", device="cuda", batch_size=args.hf_batch_size, strict_float32=False, unit=unit)}
    if kind == "vllm_http":
        return {"vllm_http": http_backend_plan(args)}
    return {"vllm_offline": vllm_backend_plan(args, unit=unit)}


def numeric_plan(backends: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The ``planned`` part of the manifest's ``numeric`` block: the installed versions of torch, transformers and vllm, and one record per back end the entry will open."""
    return {"versions": installed_versions(), "backends": {name: dict(plan) for name, plan in (backends or {}).items()}}


def numeric_differences(recorded: Any, current: Any, prefix: str = "") -> list[str]:
    """The settings in which two numeric records differ, one line each (``dtype: 'float32' -> 'bfloat16'``); empty when they agree. Lists and tuples compare alike."""
    if isinstance(recorded, Mapping) and isinstance(current, Mapping):
        out: list[str] = []
        for key in sorted(set(recorded) | set(current), key=str):
            if key not in recorded:
                out.append(f"{prefix}{key}: not recorded -> {current[key]!r}")
            elif key not in current:
                out.append(f"{prefix}{key}: {recorded[key]!r} -> not given")
            else:
                out.extend(numeric_differences(recorded[key], current[key], f"{prefix}{key}."))
        return out
    return [] if json_safe(recorded) == json_safe(current) else [f"{prefix.rstrip('.')}: {recorded!r} -> {current!r}"]


# ---------------------------------------------------------------------------------------
# readout back ends
# ---------------------------------------------------------------------------------------

@dataclass
class BackendHandle:
    """What an entry needs from an opened back end: the readout object, the tokenizer, the model directory that
    ``llm/model.json`` describes, and the serving parameters to record."""

    backend: Any
    tokenizer: Any
    model_dir: Path
    kind: str
    serve_params: dict[str, Any] = field(default_factory=dict)
    dtype: str = "bfloat16"
    closer: Callable[[], None] | None = None

    def numeric_actual(self) -> dict[str, Any]:
        """What the opened back end really computes with (the ``actual`` entry of the manifest's ``numeric`` block): for a Hugging Face model the dtype of its
        parameters, its device, the attention implementation it chose and the switches of torch as they are now; for offline vLLM what the engine reports of its
        configuration; for the others (a server, a stand-in) the kind and the dtype the handle names."""
        backend = self.backend
        if self.kind == "hf" and getattr(backend, "model", None) is not None:
            return hf_numeric_actual(backend.model, batch_size=getattr(backend, "batch_size", None), batch_plan=hf_batch_plan(backend))
        if self.kind == "vllm_offline" and getattr(backend, "engine", None) is not None:
            return vllm_numeric_actual(backend.engine, self.serve_params)
        return {"kind": self.kind, "dtype": self.dtype, "versions": installed_versions()}

    def close(self) -> None:
        """Run the closer, drop the handle's own reference to the back end and give the GPU memory back. The cross-check opens offline vLLM in the process that just
        held the HF model: as long as this handle (or the caller's variable) kept the back end, the weights would still sit on the card and the engine would not fit."""
        closer, self.closer = self.closer, None
        if closer is not None:
            closer()
        self.backend = None
        free_gpu_memory()


def free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def load_hf_model(path: str | Path, *, dtype: str = "bfloat16", device: str = "cuda") -> tuple[Any, str]:
    """A causal LM for the readout, and the loader that worked. ``AutoModelForCausalLM`` is tried first (it maps
    ``gemma4`` to ``Gemma4ForConditionalGeneration``, whose text path the readout uses), then
    ``AutoModelForImageTextToText``. The class must load and its digit readout must agree with vLLM item by item
    (a cross-check of the entries); the loader name goes into the report of the entry."""
    import torch
    import transformers

    torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
    errors: list[str] = []
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        loader = getattr(transformers, name, None)
        if loader is None:
            errors.append(f"{name}: not available in transformers {transformers.__version__}")
            continue
        try:
            model = loader.from_pretrained(str(path), dtype=torch_dtype, local_files_only=True)
            model.to(device)
            model.eval()
            return model, f"{name} -> {type(model).__name__}"
        except Exception as error:  # try the next loader; report all of them when none works
            errors.append(f"{name}: {type(error).__name__}: {str(error)[:200]}")
    raise EntryError(f"no loader could read {path}: " + "; ".join(errors))


def require_gpu_memory_fraction(args: argparse.Namespace) -> None:
    """The offline engine claims a fraction of the card's TOTAL memory and refuses to start when that much is not free; vLLM's own default (0.9) fails on a
    card that is shared with other processes. The fraction must be chosen from the free memory that ``nvidia-smi`` shows just before the run."""
    if getattr(args, "gpu_mem_util", None) is None:
        raise EntryError("--gpu-mem-util is required for the offline vLLM back end (a fraction of the card's total memory, at most the free memory that nvidia-smi shows now / total)")
    if not 0.0 < float(args.gpu_mem_util) <= 1.0:
        raise EntryError(f"--gpu-mem-util must be in (0, 1], got {args.gpu_mem_util}")


def vllm_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The keyword arguments of ``vllm.LLM`` of the offline back end: bfloat16, the context length, seed 0, the memory fraction, and whatever ``--llm-kwargs`` adds."""
    llm_kwargs: dict[str, Any] = {"dtype": "bfloat16", "max_model_len": args.max_model_len, "seed": 0, "gpu_memory_utilization": args.gpu_mem_util}
    if args.llm_kwargs:
        llm_kwargs.update(json.loads(args.llm_kwargs))
    return llm_kwargs


def open_backend(
    ctx: Context, kind: str, model_id: str, spec: ReadoutSpec, *, device: str = "cuda", dtype: str = "bfloat16", batch_size: int | None = None, strict_float32: bool | None = None,
) -> BackendHandle:
    """Open the readout back end ``kind`` for an end-side model (nothing is loaded before this call).

    The Hugging Face back end defaults to bfloat16 and ``--hf-batch-size``. ``dtype``, ``batch_size`` and ``strict_float32`` are for the float32 reference:
    ``dtype="float32"`` with the default ``strict_float32`` switches TF32 off before the model is built (``apply_strict_float32``), and the
    handle then names the dtype it really holds. Offline vLLM is bfloat16 whatever is asked here (it is the production setting)."""
    entry = device_entry(ctx.cfg, model_id)
    model_dir = Path(entry["path"])
    args = ctx.args
    if kind == "vllm_offline":
        require_gpu_memory_fraction(args)  # before anything is read
    tokenizer = load_tokenizer(model_dir)
    if kind == "hf":
        from nmi.readout.hf_backend import HFBackend

        strict = (dtype == "float32") if strict_float32 is None else strict_float32
        numerics = assert_strict_float32(apply_strict_float32()) if strict else None
        bs = args.hf_batch_size if batch_size is None else batch_size
        model, loader = load_hf_model(model_dir, dtype=dtype, device=device)
        backend = HFBackend(model, tokenizer, spec, device=device, batch_size=bs)

        def close_hf() -> None:
            nonlocal model, backend
            del backend, model
            free_gpu_memory()

        serve_params: dict[str, Any] = {"loader": loader, "batch_size": bs}
        if dtype != "bfloat16" or strict:  # the bfloat16 default adds nothing to the serving record
            serve_params.update(dtype=dtype, device=device, numerics=numerics)
        return BackendHandle(backend, tokenizer, model_dir, kind, serve_params, dtype=dtype, closer=close_hf)
    if kind == "vllm_offline":
        from nmi.readout.vllm_backend import VLLMBackend

        llm_kwargs = vllm_llm_kwargs(args)
        backend = VLLMBackend.from_model(str(model_dir), spec, tokenizer=tokenizer, **llm_kwargs)

        def close_vllm() -> None:
            shutdown_offline_engine(backend.engine)

        return BackendHandle(backend, tokenizer, model_dir, kind, {"llm_kwargs": llm_kwargs}, closer=close_vllm)
    if kind == "vllm_http":
        from nmi.readout.vllm_http import HTTPBackend

        if not args.http_url:
            raise EntryError("--backend vllm_http needs --http-url")
        backend = HTTPBackend(args.http_url, args.http_model or model_id, tokenizer, spec)
        return BackendHandle(backend, tokenizer, model_dir, kind, {"http_url": args.http_url}, closer=backend.close)
    raise EntryError(f"unknown backend {kind!r}")


def shutdown_offline_engine(engine: Any) -> None:
    """Stop an offline vLLM engine and its engine-core process (an orphaned ``VLLM::EngineCore`` keeps ~28 GB)."""
    try:
        core = getattr(getattr(engine, "llm_engine", None), "engine_core", None)
        if core is not None:
            core.shutdown()
    except Exception as error:  # pragma: no cover
        log(f"engine shutdown: {type(error).__name__}: {error}")
    del engine
    free_gpu_memory()


def trial_columns(records: Sequence[TrialReadout], *, free_logprob: bool = False) -> dict[str, Any]:
    """The readout columns of a batch of trials (the columns of the decisions table).

    ``free_logprob`` adds the column ``free_token_logprob`` (the log-probability of the most likely first token over the whole vocabulary, NaN where a back end gave
    none) behind the others. It is used for the P-answer, whose token id and probability are recorded beside the digit mass from the same forward pass;
    the other tables do not have it."""
    n = len(records)
    logprobs = np.asarray([r.readout.logprobs for r in records], dtype=np.float64).reshape(n, -1)
    columns: dict[str, Any] = {
        "digit_logprobs": logprobs,
        "digit_mass": np.asarray([r.readout.mass for r in records], dtype=np.float64),
        "gate_pass": np.asarray([r.readout.gate_pass for r in records], dtype=bool),
        "free_token_id": [r.free_token_id for r in records],
        "free_is_digit": np.asarray([r.free_is_digit for r in records], dtype=bool),
        "margin": np.asarray([r.readout.margin for r in records], dtype=np.float64),
        "ask": np.asarray([r.readout.ask for r in records], dtype=bool),
        "self_choice": np.asarray([r.readout.self_choice for r in records], dtype=np.int64),
    }
    if free_logprob:
        columns["free_token_logprob"] = np.asarray([math.nan if r.free_token_logprob is None else r.free_token_logprob for r in records], dtype=np.float64)
    return columns


# ---------------------------------------------------------------------------------------
# run lifecycle
# ---------------------------------------------------------------------------------------

class PartStore:
    """One Parquet file per finished unit of work, written atomically; a resumed run skips them."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path(self, unit_id: str) -> Path:
        return self.directory / f"{paths.validate_name('unit', unit_id)}.parquet"

    def done(self, unit_id: str) -> bool:
        return self.path(unit_id).is_file()

    def write(self, unit_id: str, columns: Mapping[str, Any] | pa.Table) -> Path:
        return write_parquet(self.path(unit_id), columns)

    def read(self, unit_id: str) -> pa.Table:
        return read_parquet(self.path(unit_id))

    def units(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.parquet")) if self.directory.is_dir() else []

    def concat(self, unit_ids: Sequence[str] | None = None) -> pa.Table:
        ids = list(unit_ids) if unit_ids is not None else self.units()
        if not ids:
            raise EntryError("no finished units to assemble")
        return pa.concat_tables([self.read(u) for u in ids], promote_options="permissive")


@dataclass
class RunHandle:
    record: RunRecord
    guard: SourceHashGuard
    parts: PartStore
    resumed: bool
    run_dir: Path
    ctx: Context

    def check_source(self) -> None:
        """Stop the dispatch of work when ``nmi/`` changed since the batch started."""
        self.guard.check()

    def confirm_numeric(self, name: str, actual: Mapping[str, Any]) -> None:
        """Record what the opened back end ``name`` really computes with (``BackendHandle.numeric_actual()``) as ``numeric.actual[name]`` of the manifest. A resumed run
        has the record of its first session: the new one must be the same, else ``EntryError`` (the run would continue with other numbers than it started with)."""
        numeric = self.record.manifest.setdefault("numeric", {"planned": {}, "actual": {}})
        recorded = numeric.setdefault("actual", {})
        if recorded.get(name) is not None:
            differences = numeric_differences(recorded[name], actual)
            if differences:
                raise EntryError(f"--resume {self.run_dir}: the numeric settings of the back end {name!r} differ from those the run started with: " + "; ".join(differences))
            return
        recorded[name] = json_safe(actual)
        self.record.write_manifest()


def tag_in_use(results_root: Path, stage: str, tag: str) -> str | None:
    """The existing run directory that already uses ``tag`` in ``stage`` (``begin_run`` refuses to reuse a tag)."""
    stage_path = results_root / paths.validate_name("stage", stage)
    if not stage_path.is_dir():
        return None
    for existing in sorted(stage_path.iterdir()):
        match = re.fullmatch(r"\d{8}_\d{6}_(.+)", existing.name)
        if match and match.group(1) == tag:
            return existing.name
    return None


def manifest_inputs(ctx: Context, pools: Iterable[str]) -> tuple[dict[str, Path], dict[str, str]]:
    """Files whose md5 enters the manifest (the two yaml files, the pre-registration, and every amendment file that the configuration lists, keyed
    ``prereg_amendment_<tag>``) and the split md5s. A resumed run compares all of them."""
    files = {"config": ctx.cfg.path, "prompts": ctx.prompts_path, "prereg": paths.PREREG, **ctx.cfg.prereg_amendment_files()}
    for name in ("prereg", *[k for k in files if k.startswith("prereg_amendment_")]):  # documents that are not part of this release are not bound
        if not Path(files[name]).is_file():
            note_skipped(files[name], "pre-registration binding")
            del files[name]
    pool_list = list(pools)
    split_md5 = ctx.registry().manifest_md5s(pool_list) if pool_list else {}
    return files, split_md5


def _resume_record(
    ctx: Context, run_dir: Path, stage: str, files: Mapping[str, Path], split_md5: Mapping[str, str], numeric: Mapping[str, Any] | None = None,
) -> RunRecord:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise EntryError(f"--resume {run_dir}: no manifest.json there")
    manifest = read_json(manifest_path)
    if manifest.get("stage") != stage:
        raise EntryError(f"--resume {run_dir}: the run is stage {manifest.get('stage')!r}, this entry writes {stage!r}")
    if manifest.get("exit_status") == "ok":
        raise EntryError(f"--resume {run_dir}: the run finished normally; nothing to resume")
    current = {name: file_md5(path) for name, path in files.items()}
    current.update(split_md5)
    recorded = manifest.get("input_md5", {})
    changed = sorted(k for k, v in current.items() if recorded.get(k) != v)
    if changed:
        raise EntryError(f"--resume {run_dir}: inputs changed since the run started: {changed}")
    if manifest.get("nmi_source_sha256") != source_hash():
        raise EntryError(f"--resume {run_dir}: the nmi/ source changed since the run started (code must be identical)")
    if numeric is not None:  # the numeric settings are bound like the input files
        recorded_numeric = manifest.get("numeric")
        if not isinstance(recorded_numeric, dict) or "planned" not in recorded_numeric:
            raise EntryError(f"--resume {run_dir}: the manifest has no numeric record, so the numeric settings cannot be compared; start the run again under a new --tag")
        differences = numeric_differences(recorded_numeric["planned"], numeric)
        if differences:
            raise EntryError(f"--resume {run_dir}: the numeric settings changed since the run started: " + "; ".join(differences))
    start = now_local()
    manifest["exit_status"] = "running"
    manifest["end_time"] = None
    manifest["exit_code"] = None
    manifest["error"] = None
    manifest.setdefault("resumes", []).append({"start_time": iso(start), "command_line": list(ctx.argv)})
    aborted = run_dir / "aborted.json"
    if aborted.is_file():
        aborted.rename(run_dir / f"aborted.{len(manifest['resumes'])}.json")
    record = RunRecord(run_dir=run_dir, manifest=manifest, _start=start)
    record.write_manifest()
    return record


@contextlib.contextmanager
def open_run(
    ctx: Context,
    *,
    pools: Iterable[str] = (),
    seed_keys: Iterable[Sequence[int]] = (),
    parents: Sequence[str] = (),
    install_sigterm: bool = True,
    extra_manifest: Mapping[str, Any] | None = None,
    extra_md5: Mapping[str, str] | None = None,
    numeric: Mapping[str, Any] | None = None,
) -> Iterator[RunHandle]:
    """Open (or resume) the run directory of the entry and guard the batch.

    The manifest gets the md5 of the configuration, the prompt file, the pre-registration (and the amendment files of the configuration) and the split, the full
    seed keys, the ``nmi/`` source hash and the ``numeric`` block: ``planned`` is ``numeric`` (``numeric_plan(...)``; the installed versions only when the entry
    gives none), ``actual`` is filled by ``RunHandle.confirm_numeric`` once a back end is open. A resumed run must match the input hashes and ``planned``. SIGTERM writes
    ``aborted.json``. The ``with`` block closes the record (``ok``, ``error`` or ``aborted``).
    """
    files, split_md5 = manifest_inputs(ctx, pools)
    split_md5 = {**split_md5, **(extra_md5 or {})}
    keys = [tuple(k) for k in seed_keys]
    seeds.assert_unique(keys)
    planned = json_safe(numeric if numeric is not None else numeric_plan())
    resumed = bool(getattr(ctx.args, "resume", None))
    if resumed:
        record = _resume_record(ctx, Path(ctx.args.resume), ctx.stage, files, split_md5, planned)
    else:
        if not ctx.tag:
            raise EntryError("--tag is required for a real run")
        record = begin_run(
            stage=ctx.stage, tag=ctx.tag, protocol_version=ctx.cfg.protocol_version, config_revision=ctx.cfg.revision,
            results=ctx.results_root, argv=ctx.argv, input_files=files, input_md5=split_md5, seed_keys=keys,
            parents=list(parents),
        )
        record.manifest["numeric"] = {"planned": planned, "actual": {}}
        record.write_manifest()
    if extra_manifest:
        record.manifest.update(extra_manifest)
        record.write_manifest()
    restore = record.install_sigterm_handler() if install_sigterm else None
    handle = RunHandle(
        record=record, guard=SourceHashGuard(), parts=PartStore(record.run_dir / "parts"), resumed=resumed,
        run_dir=record.run_dir, ctx=ctx,
    )
    try:
        with record:
            yield handle
    finally:
        if restore is not None:
            restore()


def write_model_files(
    handle: RunHandle,
    entry: Mapping[str, Any],
    bh: BackendHandle | None,
    *,
    model_dir: Path | None = None,
    serve_params: Mapping[str, Any] | None = None,
    dtype: str = "bfloat16",
    filename: str = "model.json",
) -> dict[str, Any]:
    """``llm/<filename>`` (weight and tokenizer hashes, dtype, vLLM version, serving parameters, thinking switch). The dtype is that of the back end handle
    (``bh.dtype``: the dtype it really holds), or ``dtype`` when there is no handle: an entry that loads a model itself passes the dtype of the loaded model.

    ``model.json`` is the default file name; a run with two services (the loop) writes the second one under
    another name (``model_decision.json``)."""
    cache = read_json(HASH_CACHE_PATH) if HASH_CACHE_PATH.is_file() else {}
    directory = model_dir or (bh.model_dir if bh is not None else Path(entry["path"]))
    params = dict(serve_params or (bh.serve_params if bh is not None else {}))
    target = handle.run_dir / "llm" / filename
    if filename == "model.json":
        content = write_model_json(
            handle.run_dir, model_id=str(entry["id"]), model_dir=directory, dtype=dtype if bh is None else bh.dtype,
            serve_params=params, chat_template_kwargs=chat_kwargs_of(entry), hash_cache=cache,
        )
    else:  # write_model_json always names its file model.json: let it write into a scratch directory, then move
        scratch = handle.run_dir / ".llm_scratch"
        content = write_model_json(
            scratch, model_id=str(entry["id"]), model_dir=directory, dtype=dtype if bh is None else bh.dtype,
            serve_params=params, chat_template_kwargs=chat_kwargs_of(entry), hash_cache=cache,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch / "llm" / "model.json", target)
        (scratch / "llm").rmdir()
        scratch.rmdir()
    write_json(HASH_CACHE_PATH, cache)
    return content


@dataclass
class Job:
    """One unit of work: ``build()`` returns the rendered texts and the per-row columns of its output part."""

    unit_id: str
    build: Callable[[], tuple[list[str], dict[str, Any]]]
    free_logprob: bool = False  # also write the column ``free_token_logprob`` (see ``trial_columns``)


def run_jobs(handle: RunHandle, backend: Any, jobs: Iterable[Job], *, label: str = "") -> dict[str, Any]:
    """Run the units that are not finished yet. Returns counts and wall time for the report."""
    t0 = time.time()
    done = skipped = rows = 0
    for job in jobs:
        if handle.parts.done(job.unit_id):
            skipped += 1
            continue
        handle.check_source()
        started = time.time()
        texts, meta = job.build()
        records = backend.readouts(texts)
        if len(records) != len(texts):
            raise EntryError(f"{job.unit_id}: the back end returned {len(records)} readouts for {len(texts)} prompts")
        handle.parts.write(job.unit_id, {**meta, **trial_columns(records, free_logprob=job.free_logprob)})
        done += 1
        rows += len(texts)
        log(f"{label}{job.unit_id}: {len(texts)} prompts in {time.time() - started:.1f} s")
    return {"units_done": done, "units_skipped": skipped, "rows": rows, "wall_s": time.time() - t0}


# ---------------------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------------------

class DryRun:
    """Collects the checks and the planned workload of ``--dry-run`` and prints one JSON report."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.checks: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}

    def check(self, name: str, ok: bool, detail: Any = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def note(self, key: str, value: Any) -> None:
        self.plan[key] = value

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)

    def emit(self) -> int:
        report = {
            "script": self.ctx.script, "stage": self.ctx.stage, "dry_run": True, "ok": self.ok,
            "checks": self.checks, "plan": self.plan,
        }
        print(json.dumps(json_safe(report), indent=2, sort_keys=True, default=str))
        return 0 if self.ok else 1


def check_model_dirs(cfg: Config, entries: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    """Existence of the model directories (config, tokenizer, weight files by name only; nothing is read)."""
    out = []
    for kind, model_id in entries:
        entry = device_entry(cfg, model_id) if kind == "device" else helper_entry(cfg, model_id)
        directory = Path(entry["path"])
        weights = sorted(p.name for p in directory.glob("*.safetensors")) if directory.is_dir() else []
        out.append({
            "id": model_id, "kind": kind, "path": str(directory), "is_dir": directory.is_dir(),
            "has_config": (directory / "config.json").is_file(), "n_weight_files": len(weights),
            "has_tokenizer": (directory / "tokenizer.json").is_file() or (directory / "tokenizer.model").is_file(),
        })
    return out


def verify_bank_inputs(ctx: Context, banks: Iterable[str]) -> dict[str, Any]:
    """Hash the input files a pool is loaded from and compare with the sealed locator map (bytes pass through the hash only)."""
    reg = ctx.registry()
    data = ctx.cfg["data"]
    sources = {BANK_SIMPLEQA: Path(data["simpleqa_mc"]["source"]), BANK_MMLU: Path(data["mmlu_redux"]["source"])}
    out: dict[str, Any] = {}
    for bank in sorted(set(banks)):
        problems: list[str] = []
        verified = 0
        recorded = reg.recorded_inputs(bank)
        for name in sorted(recorded):
            path = sources[bank] / name[len(bank) + 1:]
            if not path.is_file():
                problems.append(f"missing input file {name}")
                continue
            try:
                reg.verify_input(name, path)
                verified += 1
            except InputHashMismatchError as error:
                problems.append(str(error))
        out[bank] = {"n_registered": len(recorded), "n_verified": verified, "problems": problems}
    return out


def dry_run_basics(
    dry: DryRun,
    *,
    pools: Sequence[str] = (),
    models: Sequence[tuple[str, str]] = (),
    seed_keys: Iterable[Sequence[int]] = (),
    check_inputs: bool = True,
) -> None:
    """The checks every entry shares (see the module docstring)."""
    ctx = dry.ctx
    cfg = ctx.cfg
    dry.check("configuration", True, {
        "path": str(cfg.path), "md5": cfg.md5, "revision": cfg.revision, "protocol_version": cfg.protocol_version,
        "expected_md5": ctx.args.expect_config_md5,
    })
    dry.check("prompt file", True, {"path": str(ctx.book.path), "md5": ctx.book.md5, "paraphrases": list(ctx.book.paraphrase_ids)})
    if not paths.PREREG.is_file():
        note_skipped(paths.PREREG, "pre-registration")
    dry.check("pre-registration", True, {"path": str(paths.PREREG), "md5": file_md5(paths.PREREG) if paths.PREREG.is_file() else None, "skipped": not paths.PREREG.is_file()})
    amendments = cfg.prereg_amendment_files()
    if amendments:  # the amendment files are bound in the manifest beside the pre-registration
        for key, path in amendments.items():
            if not path.is_file():
                note_skipped(path, "pre-registration amendment")
        dry.check("pre-registration amendments", True,
                  {key: {"path": str(path), "md5": file_md5(path) if path.is_file() else None, "skipped": not path.is_file()} for key, path in amendments.items()})
    try:
        reg = ctx.registry()
        dry.check("split registry", True, {
            "dir": str(ctx.split_dir), "version": reg.version, "file_md5": reg.file_md5s,
            "pools": {p: {"n": len(reg.keys(p)), "md5": reg.pool_md5(p)} for p in pools},
        })
        if check_inputs and pools:
            result = verify_bank_inputs(ctx, {bank_of(p) for p in pools})
            dry.check("input file hashes", all(not v["problems"] for v in result.values()), result)
    except SplitError as error:
        dry.check("split registry", False, str(error))
    if models:
        report = check_model_dirs(cfg, models)
        dry.check("model directories", all(m["is_dir"] and m["has_config"] and m["has_tokenizer"] for m in report), report)
        if getattr(ctx.args, "backend", None) == "vllm_offline":  # the entries that open an offline engine
            try:
                require_gpu_memory_fraction(ctx.args)
                dry.check("offline vLLM memory fraction", True, {"gpu_mem_util": ctx.args.gpu_mem_util, "max_model_len": ctx.args.max_model_len})
            except EntryError as error:
                dry.check("offline vLLM memory fraction", False, str(error))
    keys = [tuple(k) for k in seed_keys]
    try:
        n_keys = seeds.assert_unique(keys)
        dry.check("seed keys", True, {"n": n_keys, "keys": [list(k) for k in keys[:12]]})
    except seeds.SeedKeyError as error:
        dry.check("seed keys", False, str(error))
    off_root = True
    try:
        paths.assert_off_root_partition(ctx.results_root)
    except paths.PathPolicyError as error:
        off_root = False
        dry.check("output directory (off the root partition)", False, str(error))
    if off_root:
        resume = ctx.args.resume
        if resume:
            run_dir = Path(resume)
            ok = (run_dir / "manifest.json").is_file()
            dry.check("resume directory", ok, str(run_dir))
        else:
            used = tag_in_use(ctx.results_root, ctx.stage, ctx.tag or "dryrun-tag")
            dry.check("output directory", used is None, {
                "results_root": str(ctx.results_root), "stage": ctx.stage, "tag": ctx.tag,
                "tag_already_used_by": used,
            })
    dry.note("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))


# ---------------------------------------------------------------------------------------
# artifacts of other runs: the contract between the entries
# ---------------------------------------------------------------------------------------

SCREENING = "screening.json"
CALIB_SUMMARY = "calib_summary.json"
CALIB_TABLE = "calib.parquet"
RHAT = "rhat.json"


def _need_json(run_dir: str | Path, name: str) -> Any:
    path = Path(run_dir) / name
    if not path.is_file():
        raise EntryError(f"{path} is missing: is {run_dir} a finished run of the entry that writes {name}?")
    return read_json(path)


def _field(doc: Mapping[str, Any], dotted: str, where: str) -> Any:
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise EntryError(f"{where}: the field {dotted!r} is missing")
        node = node[part]
    return node


@dataclass(frozen=True)
class ScreeningInputs:
    run_dir: Path
    primary: str | None
    retained: tuple[str, ...]
    md5: str


def read_screening_run(run_dir: str | Path) -> ScreeningInputs:
    run_dir = Path(run_dir)
    doc = _need_json(run_dir, SCREENING)
    where = f"{run_dir}/{SCREENING}"
    return ScreeningInputs(
        run_dir=run_dir, primary=_field(doc, "result.primary", where),
        retained=tuple(_field(doc, "result.retained", where)), md5=file_md5(run_dir / SCREENING),
    )


@dataclass(frozen=True)
class CalibInputs:
    """The answer-phase calibration run: the temperature of every bank and the calibrated confidence of every item."""

    run_dir: Path
    model: str
    temperature_by_bank: Mapping[str, float]
    md5: str

    def z_by_item(self, pool: str) -> dict[str, float]:
        table = read_parquet(self.run_dir / CALIB_TABLE)
        pools = table.column("pool").to_pylist()
        keys = table.column("item_key").to_pylist()
        z = table.column("z").to_pylist()
        return {k: float(v) for k, p, v in zip(keys, pools, z, strict=True) if p == pool}


def read_calib_run(run_dir: str | Path) -> CalibInputs:
    run_dir = Path(run_dir)
    doc = _need_json(run_dir, CALIB_SUMMARY)
    where = f"{run_dir}/{CALIB_SUMMARY}"
    banks = _field(doc, "banks", where)
    return CalibInputs(
        run_dir=run_dir, model=str(_field(doc, "model", where)),
        temperature_by_bank={b: float(v["temperature"]["temperature"]) for b, v in banks.items()},
        md5=file_md5(run_dir / CALIB_SUMMARY),
    )


# ---------------------------------------------------------------------------------------
# the upstream runs of a run, named as its parents
# ---------------------------------------------------------------------------------------

MANIFEST_FILE = "manifest.json"


def upstream_run_id(directory: str | Path) -> str | None:
    """The ``run_id`` in the ``manifest.json`` of an upstream run directory; None when the directory has no manifest, the manifest cannot be read or it names no run (the directory is then
    not a run of ``nmi.provenance``)."""
    path = Path(directory) / MANIFEST_FILE
    if not path.is_file():
        return None
    try:
        doc = read_json(path)
    except (OSError, ValueError):
        return None
    run_id = doc.get("run_id") if isinstance(doc, Mapping) else None
    return run_id if isinstance(run_id, str) and run_id else None


def parent_run_ids(inputs: Sequence[tuple[str, str | Path]]) -> list[str]:
    """The run ids for ``parent_run_ids`` of the manifest: one for every ``(label, run directory)`` of ``inputs``, in this order, each run once (a run named by two inputs counts once).

    The id is read from the ``manifest.json`` of the upstream run directory, which is where the parents of a run are found again. A directory without a
    readable manifest that names a run cannot be named as a parent: ``EntryError`` (the entry exits 2 before it creates a run directory), and the message lists every such input."""
    ids: list[str] = []
    missing: list[str] = []
    for label, directory in inputs:
        run_id = upstream_run_id(directory)
        if run_id is None:
            missing.append(f"{label} {directory}")
        elif run_id not in ids:
            ids.append(run_id)
    if missing:
        raise EntryError(
            f"{'; '.join(missing)}: no readable {MANIFEST_FILE} with a run_id there, so the run cannot be named as a parent of this run (the inputs of the later steps are runs of nmi.provenance)"
        )
    return ids


# ---------------------------------------------------------------------------------------
# misc numbers
# ---------------------------------------------------------------------------------------

def describe_quantiles(values: Sequence[float], qs: Sequence[float] = (5, 25, 50, 75, 95, 99)) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    out: dict[str, Any] = {"n": int(arr.size), "mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}
    out.update({f"p{int(q)}": float(np.percentile(arr, q)) for q in qs})
    return out
