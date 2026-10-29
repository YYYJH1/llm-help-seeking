#!/usr/bin/env python
"""Runner of the explicit-number confirmation test.

The test has no questions: every trial is a two-option prompt built from three numbers (p, a_H, w) in four arms (R field identification, C compare two numbers, A explicit
terms, B explicit terms plus the two expected scores). The model's answer is the first generated token; its exact log-probabilities of the label tokens "1" and "2" are read,
m = logprob_1 - logprob_2 is the margin, the choice is the label with the larger log-probability. Templates, sampler and plans live in ``_explicit_numbers_prompts.py``.

Subcommands (one model per call; the models are those of ``configs/v6_confirm.yaml``):

``build --out DIR``  (CPU, no model)  writes ``conditions.json``, ``plans.json`` (every trial with its exact user text), their ``.sha256`` files and ``build_report.json``. Deterministic:
    the same code gives the same bytes; the expected counts are asserted (785,555 kept points; 400 / 48 / 100 / 90 conditions; 16,800 / 2,016 / 100 / 3,780 trials).
``inspect --model ID --plans DIR``  (CPU, tokenizer only)  renders every trial through the model's chat template, encodes it as the readout does, checks that "1" and "2" are one token
    each and that for every trial ``encode(text + label) == encode(text) + [label_id]``, reports the token counts (a prompt longer than 4095 tokens fails) and writes
    ``inspect_<model>.json``.
``read --model ID --plans DIR --set {pilot,main,grid,calib} --out RUN --gpu-mem-util X``  (GPU)  offline vLLM, bfloat16 activations, seed 0, ``max_model_len`` 4096, raw log-probabilities
    (``nmi.readout.vllm_backend.VLLMBackend`` with a two-label ``ReadoutSpec``); writes ``rows.jsonl`` (one row per planned trial, schema ``ROW_FIELDS``), ``manifest.json`` and
    ``format_summary.json`` (gate-pass rate by arm and version stratum; it holds no accuracy, so the format pilot cannot select a template by performance).
``crosscheck --model ID --plans DIR --pilot-run RUN [--calib-run RUN] --out DIR [--reference-device DEV]``  (GPU or CPU)  the numerical check:
    bf16 models: Hugging Face float32 (eval mode, batch size 1, no padding, no autocast, no TF32) reads the 100 calibration prompts, which fixes delta_m = 0.25 * max(1, ceil(6 * max_j
    |m_V - m_F|)); then the pilot prompts are scanned arm by arm in the order of ``default_rng(20261023)`` (at most 75 per arm), the prompts with |m_F| >= delta_m are included
    (at least 30 per arm and 150 in all), and the check passes only with zero decision disagreements. Quantized models (qwen3-14b FP8, qwen3-32b AWQ): the same 300 pilot prompts
    (75 per arm) are read a second time in a new, independent engine process, submitted in the reverse of their plan order, as ``scripts/run_c4_repeat.py`` reads its second pass;
    the first reading is the pilot run. The check passes with at least 299 of 300 prompts that pass the gate in both readings with the same choice; delta_q is
    0.25 * max(1, ceil(6 * max_j |m_1 - m_2|)). ``crosscheck.json`` holds every number, every disagreement and the verdict.

Quality rule: a trial passes the gate when exp(logprob_1) + exp(logprob_2) >= 0.5; a failing trial is still a valid reading, decided by its margin; |margin| < 1e-6 is a tie
(no choice). A planned trial whose reading failed is written with ``missing: true`` and null numbers; it stays in the denominator. More than 1% missing stops the run.

Examples (the numeric settings of the confirmation run; one idle card per ``read``, chosen by the caller with ``CUDA_VISIBLE_DEVICES``)::

    python scripts/run_explicit_numbers.py build --out PLANS
    python scripts/run_explicit_numbers.py inspect --model qwen3-4b --plans PLANS
    python scripts/run_explicit_numbers.py read --model qwen3-4b --plans PLANS --set pilot --out RUNS/qwen3-4b/pilot --gpu-mem-util 0.5
    python scripts/run_explicit_numbers.py read --model qwen3-4b --plans PLANS --set calib --out RUNS/qwen3-4b/calib --gpu-mem-util 0.5
    python scripts/run_explicit_numbers.py crosscheck --model qwen3-4b --plans PLANS --calib-run RUNS/qwen3-4b/calib --pilot-run RUNS/qwen3-4b/pilot --out CHECKS/qwen3-4b --reference-device cpu
    python scripts/run_explicit_numbers.py read --model qwen3-14b --plans PLANS --set pilot --out RUNS/qwen3-14b/pilot --gpu-mem-util 0.80 --llm-kwargs '{"quantization": "fp8"}'
    python scripts/run_explicit_numbers.py read --model qwen3-32b --plans PLANS --set pilot --out RUNS/qwen3-32b/pilot --gpu-mem-util 0.85
    python scripts/run_explicit_numbers.py crosscheck --model qwen3-32b --plans PLANS --pilot-run RUNS/qwen3-32b/pilot --out CHECKS/qwen3-32b

Files the other steps read: ``plans.json`` (object with ``trials``, one trial per line; each trial has the row fields known before the reading, ``trial_id`` and ``text``), ``rows.jsonl``
(one row per planned trial, ``ROW_FIELDS``), ``manifest.json`` of a run (``status``, ``plans_sha256``, ``engine.llm_kwargs``, ``code.sha256``, ...) and ``crosscheck.json`` (``passed``,
``kind``, ``delta`` = ``{name, value}`` with ``delta_m`` or ``delta_q`` repeated at the top, every number of the check).

Exit codes: 0 done; 1 a check did not pass (inspect, crosscheck) or a technical failure; 2 bad arguments or inputs; 4 ``nmi/`` changed during a ``read`` (the source guard).
Nothing under ``nmi/`` is modified by this script. Every output path must be off the root partition.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import re
import signal
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _c2_common as common  # noqa: E402
import _explicit_numbers_prompts as prompts  # noqa: E402
import numpy as np  # noqa: E402
import run_c4_repeat as c4  # noqa: E402
import yaml  # noqa: E402

from nmi import paths  # noqa: E402
from nmi.prompts import render_chat  # noqa: E402
from nmi.provenance import SourceChangedError, SourceHashGuard, file_md5, file_sha256, iso, now_local, source_hash  # noqa: E402
from nmi.readout.digits import ReadoutSpec  # noqa: E402
from nmi.readout.hf_backend import ReadoutError, TrialReadout, encode_prompts  # noqa: E402
from nmi.readout.vllm_backend import VLLMParseError  # noqa: E402

SCRIPT = Path(__file__).resolve()
REPO_ROOT = common.REPO_ROOT
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v6_confirm.yaml"
RUN_SCHEMA = "nmi.explicit_numbers.run/1"
FORMAT_SCHEMA = "nmi.explicit_numbers.format_summary/1"
INSPECT_SCHEMA = "nmi.explicit_numbers.inspect/1"
CROSSCHECK_SCHEMA = "nmi.explicit_numbers.crosscheck/1"
LABELS = ("1", "2")
MAX_MODEL_LEN = 4096
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - 1  # one generated token must fit
MIN_GATE_PASS_RATE = 0.95  # per version stratum (0.95 rather than the 0.99 of the confirmation run, because the prompts are new)
MAX_MISSING_FRACTION = 0.01
DEFAULT_CHUNK_SIZE = 2048
ALLOWED_PREFIXES = ("", "Answer: ")  # the pre-specified fallback is the P-answer prefix of configs/prompts/ask_v1_1.yaml
PROTECTED_LLM_KEYS = ("dtype", "max_model_len", "seed", "gpu_memory_utilization", "logprobs_mode")
# the numeric settings of the confirmation run (configs/v6_confirm.yaml); "crosscheck" names the kind of numerical check
REGISTERED_RUN: dict[str, dict[str, Any]] = {
    "qwen3-14b": {"llm_kwargs": {"quantization": "fp8"}, "gpu_mem_util": 0.80, "crosscheck": "repeat"},
    "qwen3-32b": {"llm_kwargs": {}, "gpu_mem_util": 0.85, "crosscheck": "repeat"},
}
DEFAULT_RUN: dict[str, Any] = {"llm_kwargs": {}, "gpu_mem_util": None, "crosscheck": "float32"}
ENV_DEFAULTS = {  # set only when the launching shell has not set them; nothing may be written to the root partition
    "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_CONFIG_ROOT": str(Path(os.environ.get("NMI_TMP_ROOT", Path(__file__).resolve().parents[1] / "tmp")) / "vllm" / "config"),
}
ENV_RECORDED = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONNOUSERSITE", *ENV_DEFAULTS)
# the numerical check
CAL_N = 100
SCAN_PER_ARM = 75
MIN_INCLUDED_PER_ARM = 30
MIN_INCLUDED_TOTAL = 150
REPEAT_REQUIRED = 299
DELTA_STEP = 0.25
DELTA_MULTIPLIER = 6  # 1.5 / 0.25: delta = 0.25 * max(1, ceil(6 * e_max))
PASS_FILE = "pass2_rows.jsonl"


class TechnicalFailure(RuntimeError):
    """A reading or a file the check needs is missing, not comparable or not finite: no verdict (exit code 1)."""


# ---------------------------------------------------------------------------------------
# configuration, models, settings
# ---------------------------------------------------------------------------------------

def load_model_entries(config_path: str | Path) -> tuple[dict[str, dict[str, Any]], str]:
    """``models.devices`` of the configuration (read with a YAML loader, nothing hard-coded) by id, and the md5 of the file."""
    path = Path(config_path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        devices = doc["models"]["devices"]
        entries = {str(d["id"]): dict(d) for d in devices}
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise common.EntryError(f"cannot read models.devices from {path}: {error}") from error
    return entries, file_md5(path)


def model_entry(entries: Mapping[str, Mapping[str, Any]], model_id: str) -> dict[str, Any]:
    if model_id not in entries:
        raise common.EntryError(f"unknown model {model_id!r}; the configuration lists {sorted(entries)}")
    entry = dict(entries[model_id])
    common.chat_kwargs_of(entry)  # a model without chat_template_kwargs would run in thinking mode
    if "path" not in entry:
        raise common.EntryError(f"model {model_id!r} has no path in the configuration")
    return entry


def check_prefix(prefix: str) -> str:
    if prefix not in ALLOWED_PREFIXES:
        raise common.EntryError(f"--assistant-prefix must be '' (the default) or 'Answer: ' (the pre-specified fallback of the format pilot), got {prefix!r}")
    return prefix


def check_off_root(path: str | Path) -> Path:
    try:
        return paths.assert_off_root_partition(path)
    except paths.PathPolicyError as error:
        raise common.EntryError(str(error)) from error


def two_label_spec() -> ReadoutSpec:
    """The readout of this test: the label tokens "1" and "2", the margin logprob_1 - logprob_2 (``ask_token`` "1" makes the spec subtract the other label), gate 0.5."""
    return ReadoutSpec(
        digits=LABELS, ask_token="1", digit_mass_gate=prompts.DIGIT_MASS_GATE, min_gate_pass_rate=MIN_GATE_PASS_RATE, offline_max_tokens=1, offline_temperature=0.0,
        http_max_tokens=1, http_temperature=0.0, http_logprobs=len(LABELS),
    )


def resolve_llm_kwargs(model_id: str, gpu_mem_util: float | None, llm_kwargs_json: str | None, max_model_len: int, allow_nonregistered: bool) -> tuple[dict[str, Any], list[str]]:
    """The keyword arguments of ``vllm.LLM`` and the list of settings that differ from the pre-specified run settings (``REGISTERED_RUN``; empty when none differs).

    bfloat16 activations, seed 0, ``max_model_len`` 4096 and the memory fraction are fixed here; ``--llm-kwargs`` may add engine arguments but never override these or the log-prob
    mode. Qwen3-14B must run with online FP8 weights at 0.80, Qwen3-32B (the AWQ checkpoint) at 0.85; a different value is an error unless ``--allow-nonregistered``
    is given, and then it is recorded."""
    if gpu_mem_util is None or not 0.0 < float(gpu_mem_util) <= 1.0:
        raise common.EntryError(f"--gpu-mem-util must be in (0, 1] (a fraction of the card's total memory, chosen from the free memory just before the run), got {gpu_mem_util!r}")
    extra: dict[str, Any] = {}
    if llm_kwargs_json:
        try:
            extra = json.loads(llm_kwargs_json)
        except ValueError as error:
            raise common.EntryError(f"--llm-kwargs is not valid JSON: {error}") from error
        if not isinstance(extra, dict):
            raise common.EntryError("--llm-kwargs must be a JSON object")
    clash = sorted(set(extra) & set(PROTECTED_LLM_KEYS))
    if clash:
        raise common.EntryError(f"--llm-kwargs must not set {clash}: bfloat16, seed 0, --max-model-len and --gpu-mem-util have their own options; logprobs_mode stays at raw_logprobs")
    registered = REGISTERED_RUN.get(model_id, DEFAULT_RUN)
    differences: list[str] = []
    if extra != registered["llm_kwargs"]:
        differences.append(f"llm_kwargs {extra} (registered {registered['llm_kwargs']})")
    if registered["gpu_mem_util"] is not None and not math.isclose(float(gpu_mem_util), registered["gpu_mem_util"]):
        differences.append(f"gpu_mem_util {gpu_mem_util} (registered {registered['gpu_mem_util']})")
    if max_model_len != MAX_MODEL_LEN:
        differences.append(f"max_model_len {max_model_len} (registered {MAX_MODEL_LEN})")
    if differences and not allow_nonregistered:
        raise common.EntryError(f"{model_id}: settings that differ from the registered run: {'; '.join(differences)}. Use the registered values or pass --allow-nonregistered (the run is then marked)")
    kwargs = {"dtype": "bfloat16", "max_model_len": int(max_model_len), "seed": 0, "gpu_memory_utilization": float(gpu_mem_util), **extra}
    return kwargs, differences


# ---------------------------------------------------------------------------------------
# tokenization shared by all subcommands
# ---------------------------------------------------------------------------------------

def template_uses_date(tokenizer: Any) -> bool:
    """Whether the chat template of the tokenizer calls ``strftime_now`` (SmolLM3: ``Today Date: <today>`` in its system prompt)."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, Mapping):
        template = "\n".join(str(v) for v in template.values())
    return isinstance(template, str) and "strftime_now" in template


@contextlib.contextmanager
def pinned_template_date(tokenizer: Any, day: str = prompts.TEMPLATE_DATE):
    """Inside the block the ``strftime_now`` of the chat templates returns ``day`` (noon); yields whether the template of this tokenizer uses the date at all.

    ``transformers`` renders it with ``datetime.now()`` of ``transformers.utils.chat_template_utils``; without the
    pin a run on another day would put another date into the system prompt of SmolLM3, and the pilot, the calibration, the main run and the cross-check would not read the same
    token ids. Nothing is patched for a template that does not use the date. An installed ``transformers`` without that attribute is an error."""
    if not template_uses_date(tokenizer):
        yield False
        return
    from transformers.utils import chat_template_utils as module

    original = getattr(module, "datetime", None)
    if original is None or not hasattr(original, "now"):
        raise common.EntryError("this version of transformers does not render chat templates with chat_template_utils.datetime: the template date cannot be pinned")
    year, month, day_of_month = (int(x) for x in day.split("-"))
    fixed = dt.datetime(year, month, day_of_month, 12, 0, 0)

    class Pinned(original):  # type: ignore[misc, valid-type]
        @classmethod
        def now(cls, tz: Any = None) -> dt.datetime:  # noqa: ARG003 - the signature of datetime.now
            return fixed

    module.datetime = Pinned
    try:
        yield True
    finally:
        module.datetime = original


def render_trials(tokenizer: Any, trials: Sequence[Mapping[str, Any]], chat_kwargs: Mapping[str, Any], prefix: str) -> list[str]:
    """The chat texts the model sees: one user message through the chat template with the configured parameters, ending at the generation prompt, plus the assistant prefix.
    The date a template may write into its system prompt is pinned to ``prompts.TEMPLATE_DATE``."""
    with pinned_template_date(tokenizer):
        return [render_chat(tokenizer, t["text"], chat_kwargs, assistant_prefix=prefix) for t in trials]


def tokenize_trials(tokenizer: Any, trials: Sequence[Mapping[str, Any]], chat_kwargs: Mapping[str, Any], prefix: str) -> list[list[int]]:
    """Token ids of the trials, encoded as the readout does (``encode_prompts``: no special tokens added, the template carries them)."""
    return encode_prompts(tokenizer, render_trials(tokenizer, trials, chat_kwargs, prefix))


def load_tokenizer(entry: Mapping[str, Any], hooks: "Hooks") -> Any:
    return (hooks.tokenizer_loader or common.load_tokenizer)(Path(entry["path"]))


def code_hashes(extra: Sequence[Path] = ()) -> dict[str, str]:
    """sha256 of this script, of ``_explicit_numbers_prompts.py`` and of the ``nmi/readout`` modules in use (those imported by now), plus ``nmi/prompts.py`` (the chat rendering)."""
    out = {
        "scripts/run_explicit_numbers.py": file_sha256(SCRIPT),
        "scripts/_explicit_numbers_prompts.py": file_sha256(HERE / "_explicit_numbers_prompts.py"),
        "nmi/prompts.py": file_sha256(REPO_ROOT / "nmi" / "prompts.py"),
    }
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if name.startswith("nmi.readout.") and file:
            out[f"nmi/readout/{Path(file).name}"] = file_sha256(file)
    for path in extra:
        out[str(Path(path).resolve().relative_to(REPO_ROOT))] = file_sha256(path)
    return out


# ---------------------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------------------

def build_row(model: str, trial: Mapping[str, Any], n_tokens: int, digest: str, record: TrialReadout | None) -> dict[str, Any]:
    """One row of ``rows.jsonl`` in the order of ``ROW_FIELDS``.

    ``record`` is the ``TrialReadout`` of the two-label spec: mass, gate and margin are the spec's (``readout_from_logprobs``), tie and choice follow ``choice_from_margin``. No
    record, or a log-probability that is not finite (a reading that cannot give a margin), is a missing reading: ``missing`` true and the readout fields null."""
    row: dict[str, Any] = {"model": model, **{k: trial[k] for k in prompts.PLAN_ROW_FIELDS}, "prompt_sha256": digest, "n_tokens": int(n_tokens)}
    usable = record is not None and all(math.isfinite(x) for x in record.readout.logprobs)
    if usable:
        assert record is not None
        lp1, lp2 = (float(x) for x in record.readout.logprobs)
        tie, choice = prompts.choice_from_margin(float(record.readout.margin))
        row.update(
            logprob_1=lp1, logprob_2=lp2, mass=float(record.readout.mass), gate_pass=bool(record.readout.gate_pass), margin=float(record.readout.margin), tie=tie, choice=choice,
            free_token_id=None if record.free_token_id is None else int(record.free_token_id), missing=False,
        )
    else:
        row.update({k: None for k in prompts.READOUT_ROW_FIELDS}, missing=True)
    if tuple(row) != prompts.ROW_FIELDS:
        raise AssertionError(f"row fields {tuple(row)} differ from the schema {prompts.ROW_FIELDS}")
    return row


def dump_row(row: Mapping[str, Any]) -> str:
    return json.dumps(row, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def load_run_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise TechnicalFailure(f"{path} line {number}: not valid JSON ({error})") from error
            if tuple(row) != prompts.ROW_FIELDS:
                raise TechnicalFailure(f"{path} line {number}: the fields {tuple(row)} are not the row schema")
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------
# format summary: gate-pass rate by arm and version stratum
# ---------------------------------------------------------------------------------------

def version_label(arm: str, row: Mapping[str, Any]) -> str:
    """The version stratum of a row: R field x wording x label order; C label order x rewrite; A and B wording x label order x rewrite."""
    parts = {
        "R": (("field", row["field"]), ("wording", row["wording"]), ("label_order", row["label_order"])),
        "C": (("label_order", row["label_order"]), ("rewrite", row["rewrite"])),
    }.get(arm, (("wording", row["wording"]), ("label_order", row["label_order"]), ("rewrite", row["rewrite"])))
    return " ".join(f"{k}={v}" for k, v in parts)


def format_summary(rows: Sequence[Mapping[str, Any]], label_ids: Sequence[int], tokenizer: Any | None = None, *, model: str = "", prefix: str = "") -> dict[str, Any]:
    """Gate-pass rate by set, arm and version stratum. The denominator is every planned trial (a missing reading is not a pass). Accuracy is deliberately absent.

    A stratum is ``ok`` when its rate is at least 0.95; the calibration set has one to five trials per stratum and is reported, not judged. For a stratum below 0.95 the most
    frequent first tokens are listed (decoded when a tokenizer is given), which shows what the model wrote instead of a label."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["set"], row["arm"], version_label(row["arm"], row)), []).append(row)
    label_set = {int(i) for i in label_ids}

    def stats(rs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        n = len(rs)
        passed = sum(1 for r in rs if r["gate_pass"])
        masses = [r["mass"] for r in rs if r["mass"] is not None]
        free = [r["free_token_id"] for r in rs if r["free_token_id"] is not None]
        return {
            "n": n, "n_gate_pass": passed, "gate_pass_rate": passed / n if n else None, "n_missing": sum(1 for r in rs if r["missing"]), "n_tie": sum(1 for r in rs if r["tie"]),
            "mean_mass": float(np.mean(masses)) if masses else None, "min_mass": float(np.min(masses)) if masses else None,
            "free_token_is_label_rate": sum(1 for t in free if t in label_set) / n if n else None,
        }

    out_sets: dict[str, Any] = {}
    failing: list[dict[str, Any]] = []
    for set_name in prompts.SETS:
        set_rows = [r for r in rows if r["set"] == set_name]
        if not set_rows:
            continue
        judged = set_name != "calib"
        arms: dict[str, Any] = {}
        for arm in prompts.ARMS:
            arm_rows = [r for r in set_rows if r["arm"] == arm]
            if not arm_rows:
                continue
            strata = []
            for (s, a, label), rs in sorted(groups.items()):
                if s != set_name or a != arm:
                    continue
                entry = {"version": label, **stats(rs)}
                entry["ok"] = bool(entry["gate_pass_rate"] >= MIN_GATE_PASS_RATE) if judged else None
                if judged and not entry["ok"]:
                    counts = Counter(r["free_token_id"] for r in rs if not r["gate_pass"] and r["free_token_id"] is not None).most_common(5)
                    entry["top_first_tokens_of_failures"] = [
                        {"token_id": int(t), "n": int(n), "decoded": tokenizer.decode([int(t)]) if tokenizer is not None else None} for t, n in counts
                    ]
                    failing.append({"set": set_name, "arm": arm, **entry})
                strata.append(entry)
            arms[arm] = {**stats(arm_rows), "strata": strata}
        out_sets[set_name] = {**stats(set_rows), "judged": judged, "arms": arms}
    return {
        "schema": FORMAT_SCHEMA, "model": model, "assistant_prefix": prefix, "gate": prompts.DIGIT_MASS_GATE, "min_gate_pass_rate": MIN_GATE_PASS_RATE, "sets": out_sets,
        "all_judged_strata_pass": not failing, "n_failing_strata": len(failing), "failing_strata": failing,
        "note": "the gate-pass rate has every planned trial in the denominator; no accuracy is computed here (design 2.4: the format pilot checks the format only)",
    }


# ---------------------------------------------------------------------------------------
# hooks (replaceable for testing)
# ---------------------------------------------------------------------------------------

@dataclass
class Hooks:
    tokenizer_loader: Callable[[Path], Any] | None = None
    vllm_opener: Callable[..., Any] | None = None  # (model_path, tokenizer, spec, llm_kwargs) -> backend with readouts(id_lists) and, optionally, engine
    hf_opener: Callable[..., Any] | None = None  # (entry, tokenizer, spec, device) -> (backend with readouts_from_ids(id_lists), info dict)
    launcher: Callable[[Any], Any] | None = None  # c4.ChildSpec -> c4.ChildOutcome
    gpu_info: Callable[[], list[dict[str, str]]] | None = None
    # the four hooks of c4.verify_release (duck-typed by that function)
    gpu_probe: Callable[[], list[int]] | None = None
    group_members: Callable[[int], list[int]] | None = None
    alive: Callable[[list[int]], list[int]] | None = None
    sleep: Callable[[float], None] | None = None


def apply_env_defaults() -> dict[str, str | None]:
    """Set the environment variables that keep vLLM from writing outside the repository tmp directory or phoning home, where the launching shell has not set them; return the values in force."""
    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    return {key: os.environ.get(key) for key in ENV_RECORDED}


def install_sigterm() -> Callable[[], None]:
    """Turn SIGTERM into ``SystemExit`` so that the ``finally`` blocks run: the engine is closed and the manifest says how the run ended. Returns the function that restores the old handler."""
    def handler(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, handler)
    return lambda: signal.signal(signal.SIGTERM, previous)


def open_vllm(model_path: Path, tokenizer: Any, spec: ReadoutSpec, llm_kwargs: Mapping[str, Any]) -> Any:
    """Start offline vLLM (``VLLMBackend.from_model``: raw log-probabilities, token ids sent, the echoed prompt checked)."""
    from nmi.readout.vllm_backend import VLLMBackend

    return VLLMBackend.from_model(str(model_path), spec, tokenizer=tokenizer, **dict(llm_kwargs))


def close_vllm(backend: Any) -> None:
    engine = getattr(backend, "engine", None)
    if engine is not None:
        common.shutdown_offline_engine(engine)


def engine_record(backend: Any, llm_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """What the engine says about itself (``run_c4_repeat.read_engine_settings``) and whether it works with what was asked; a stand-in engine gives unreadable fields only."""
    engine = getattr(backend, "engine", None)
    record = c4.read_engine_settings(engine) if engine is not None else {"settings": {}, "volatile": {}, "environment": {}, "unreadable": ["no engine"]}
    record["requested_vs_actual"] = c4.requested_against_actual(llm_kwargs, record["settings"]) if record["settings"] else {}
    unreadable = set(record["unreadable"])
    wrong = {k: v for k, v in record["requested_vs_actual"].items() if k in ("quantization", "dtype", "max_model_len", "seed") and f"model.{k}" not in unreadable and not v["same"]}
    if wrong:
        raise common.EntryError(f"the engine works with other settings than asked for: {wrong}")
    return record


def read_chunk(backend: Any, id_lists: Sequence[Sequence[int]]) -> tuple[list[TrialReadout | None], list[str | None]]:
    """Readouts of a chunk. When the whole chunk is refused because some output cannot be parsed (``VLLMParseError``), the prompts are read one by one: the ones that still fail
    are returned as None (missing readings) with the error text; the others keep their readout. Any other error propagates."""
    try:
        records = list(backend.readouts([list(x) for x in id_lists]))
        if len(records) != len(id_lists):
            raise ReadoutError(f"the backend returned {len(records)} readouts for {len(id_lists)} prompts")
        return records, [None] * len(records)
    except VLLMParseError:
        out: list[TrialReadout | None] = []
        errors: list[str | None] = []
        for ids in id_lists:
            try:
                (record,) = backend.readouts([list(ids)])
                out.append(record)
                errors.append(None)
            except VLLMParseError as error:
                out.append(None)
                errors.append(f"{type(error).__name__}: {error}")
        return out, errors


# ---------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace, hooks: Hooks) -> int:
    out = check_off_root(args.out)
    built = prompts.build_all()
    try:
        files = prompts.write_built(built, out, overwrite=args.overwrite)
    except prompts.PlansError as error:
        raise common.EntryError(str(error)) from error
    report = {
        **built.report, "out": str(out), "files_sha256": files, "conditions_sha256": built.conditions_sha256, "plans_sha256": built.plans_sha256,
        "trial_counts": prompts.trial_counts(built.trials), "code": code_hashes(),
    }
    common.write_json(out / "build_report.json", report)
    print(json.dumps(common.json_safe(report), indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------------------

def variant_key(trial: Mapping[str, Any]) -> str:
    return f"{trial['arm']} field={trial['field']} wording={trial['wording']} label_order={trial['label_order']} rewrite={trial['rewrite']}"


def inspect_trials(model_id: str, entry: Mapping[str, Any], trials: Sequence[Mapping[str, Any]], tokenizer: Any, prefix: str, plans_sha256: str = "") -> dict[str, Any]:
    """The tokenizer checks of ``inspect`` for a list of trials. ``passed`` is true when both labels are one token and every trial ends in a way that a label appends exactly
    one token (``encode(text + label) == encode(text) + [label_id]``), and no prompt is longer than 4095 tokens."""
    spec = two_label_spec()
    chat_kwargs = common.chat_kwargs_of(entry)
    label_ids = spec.token_ids(tokenizer)  # raises DigitTokenError when "1" or "2" is not one clean token
    texts = render_trials(tokenizer, trials, chat_kwargs, prefix)
    id_lists = encode_prompts(tokenizer, texts)
    lengths = np.asarray([len(x) for x in id_lists])
    failures: list[dict[str, Any]] = []
    variants: dict[str, dict[str, Any]] = {}
    for trial, text, ids in zip(trials, texts, id_lists, strict=True):
        entry_v = variants.setdefault(variant_key(trial), {"n_trials": 0, "n_failed": 0})
        entry_v["n_trials"] += 1
        for label, label_id in zip(LABELS, label_ids, strict=True):
            extended = encode_prompts(tokenizer, [text + label])[0]
            if extended != ids + [int(label_id)]:
                entry_v["n_failed"] += 1
                if len(failures) < 20:
                    failures.append({
                        "trial_id": trial["trial_id"], "label": label, "expected_tail": [int(x) for x in (ids + [int(label_id)])[-6:]], "found_tail": [int(x) for x in extended[-6:]],
                        "found_tail_tokens": [str(x) for x in tokenizer.convert_ids_to_tokens(extended[-6:])] if hasattr(tokenizer, "convert_ids_to_tokens") else None,
                    })
    arms: dict[str, Any] = {}
    for arm in prompts.ARMS:
        mask = np.asarray([t["arm"] == arm for t in trials])
        if mask.any():
            arms[arm] = {"n": int(mask.sum()), "max": int(lengths[mask].max()), "mean": float(lengths[mask].mean()), "min": int(lengths[mask].min())}
    longest = int(np.argmax(lengths)) if len(lengths) else 0
    examples = {}
    for arm in prompts.ARMS:
        index = next((i for i, t in enumerate(trials) if t["arm"] == arm), None)
        if index is not None:
            examples[arm] = {
                "trial_id": trials[index]["trial_id"], "rendered_text": texts[index], "n_tokens": len(id_lists[index]),
                "last_tokens": [str(x) for x in tokenizer.convert_ids_to_tokens(id_lists[index][-12:])] if hasattr(tokenizer, "convert_ids_to_tokens") else None,
            }
    too_long = int((lengths > MAX_PROMPT_TOKENS).sum())
    n_failed = sum(v["n_failed"] for v in variants.values())
    return {
        "schema": INSPECT_SCHEMA, "model": model_id, "model_path": str(entry["path"]), "tokenizer_class": type(tokenizer).__name__, "chat_template_kwargs": dict(chat_kwargs),
        "assistant_prefix": prefix, "plans_sha256": plans_sha256, "n_trials": len(trials),
        "template_date": {"pinned": prompts.TEMPLATE_DATE, "template_uses_date": template_uses_date(tokenizer)},
        "labels": {label: {"id": int(i), "decoded": tokenizer.decode([int(i)])} for label, i in zip(LABELS, label_ids, strict=True)},
        "n_tokens": {"max": int(lengths.max()), "mean": float(lengths.mean()), "min": int(lengths.min()), "longest_trial": trials[longest]["trial_id"], "limit": MAX_PROMPT_TOKENS,
                     "n_over_limit": too_long, "by_arm": arms},
        "label_append": {"rule": "encode(text + label) == encode(text) + [label_id], for both labels on every trial", "n_checks": 2 * len(trials), "n_failed": n_failed,
                         "variants": variants, "failures": failures},
        "examples": examples, "passed": bool(too_long == 0 and n_failed == 0),
    }


def cmd_inspect(args: argparse.Namespace, hooks: Hooks) -> int:
    prefix = check_prefix(args.assistant_prefix)
    entries, cfg_md5 = load_model_entries(args.config)
    entry = model_entry(entries, args.model)
    plans = load_plans_checked(args.plans)
    out_dir = check_off_root(args.out or args.plans)
    tokenizer = load_tokenizer(entry, hooks)
    started = time.time()
    report = inspect_trials(args.model, entry, plans.trials, tokenizer, prefix, plans.sha256)
    report.update(config={"path": str(Path(args.config).resolve()), "md5": cfg_md5}, code=code_hashes(), wall_s=time.time() - started)
    target = common.write_json(out_dir / f"inspect_{args.model}.json", report)
    n = report["n_tokens"]
    print(json.dumps(common.json_safe({
        "model": args.model, "passed": report["passed"], "tokenizer": report["tokenizer_class"], "labels": report["labels"], "n_trials": report["n_trials"],
        "n_tokens": {k: n[k] for k in ("max", "mean", "min", "longest_trial", "n_over_limit")}, "by_arm": n["by_arm"],
        "label_append": {k: report["label_append"][k] for k in ("n_checks", "n_failed")}, "written": str(target),
    }), indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def load_plans_checked(directory: str | Path) -> prompts.Plans:
    try:
        return prompts.load_plans(directory)
    except prompts.PlansError as error:
        raise common.EntryError(str(error)) from error


# ---------------------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------------------

def execute_read(args: argparse.Namespace, hooks: Hooks, argv: Sequence[str] = ()) -> dict[str, Any]:
    prefix = check_prefix(args.assistant_prefix)
    entries, cfg_md5 = load_model_entries(args.config)
    entry = model_entry(entries, args.model)
    llm_kwargs, non_registered = resolve_llm_kwargs(args.model, args.gpu_mem_util, args.llm_kwargs, args.max_model_len, args.allow_nonregistered)
    if args.chunk_size < 1:
        raise common.EntryError("--chunk-size must be positive")
    plans = load_plans_checked(args.plans)
    trials = plans.of_set(args.set)
    out = check_off_root(args.out)
    for name in ("rows.jsonl", "rows.jsonl.tmp"):
        if (out / name).exists():
            raise common.EntryError(f"{out / name} exists: every attempt needs a directory of its own (choose another --out)")
    out.mkdir(parents=True, exist_ok=True)
    chat_kwargs = common.chat_kwargs_of(entry)
    tokenizer = load_tokenizer(entry, hooks)
    spec = two_label_spec()
    label_ids = spec.token_ids(tokenizer)
    id_lists = tokenize_trials(tokenizer, trials, chat_kwargs, prefix)
    lengths = [len(x) for x in id_lists]
    if max(lengths) > MAX_PROMPT_TOKENS:
        raise common.EntryError(f"{sum(1 for n in lengths if n > MAX_PROMPT_TOKENS)} prompts are longer than {MAX_PROMPT_TOKENS} tokens (longest {max(lengths)})")
    digests = [prompts.token_digest(ids) for ids in id_lists]
    guard = SourceHashGuard()
    started = now_local()
    t0 = time.time()
    chunks = [(i, min(i + args.chunk_size, len(trials))) for i in range(0, len(trials), args.chunk_size)]
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA, "status": "running", "model": args.model, "model_path": str(entry["path"]), "chat_template_kwargs": dict(chat_kwargs), "set": args.set,
        "assistant_prefix": prefix, "template_date": {"pinned": prompts.TEMPLATE_DATE, "template_uses_date": template_uses_date(tokenizer)},
        "plans_dir": str(Path(args.plans).resolve()), "plans_sha256": plans.sha256, "conditions_sha256": plans.conditions_sha256, "n_planned": len(trials),
        "readout": {
            "labels": list(LABELS), "label_token_ids": [int(i) for i in label_ids], "digit_mass_gate": spec.digit_mass_gate, "max_tokens": spec.offline_max_tokens,
            "temperature": spec.offline_temperature, "logprobs_mode": "raw_logprobs", "tie_eps": prompts.TIE_EPS, "margin": "logprob_1 - logprob_2",
            "min_gate_pass_rate": MIN_GATE_PASS_RATE,
        },
        "engine": {"llm_kwargs": llm_kwargs, "non_registered": non_registered, "launch_environment": {}},
        "submission": {"order": "plan order of the set", "chunk_size": args.chunk_size, "n_chunks": len(chunks)},
        "n_tokens": {"max": max(lengths), "mean": float(np.mean(lengths)), "min": min(lengths)},
        "code": {"sha256": code_hashes(), "nmi_source_sha256": source_hash()}, "config": {"path": str(Path(args.config).resolve()), "md5": cfg_md5},
        "command_line": [str(SCRIPT), *argv], "python": sys.version.split()[0], "versions": common.installed_versions(), "start_time": iso(started), "end_time": None,
    }
    manifest["vllm_version"] = manifest["versions"].get("vllm")
    manifest_path = out / "manifest.json"
    common.write_json(manifest_path, manifest)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    tmp_path = out / "rows.jsonl.tmp"
    max_missing = int(math.ceil(MAX_MISSING_FRACTION * len(trials)))
    backend: Any = None
    status, failure = "error", None
    restore_sigterm = install_sigterm()
    try:
        manifest["engine"]["launch_environment"] = apply_env_defaults()
        common.log(f"{args.model} {args.set}: {len(trials)} trials, {len(chunks)} chunks; starting the engine with {llm_kwargs}")
        opener = hooks.vllm_opener or open_vllm
        backend = opener(Path(entry["path"]), tokenizer, spec, llm_kwargs)
        manifest["engine"].update(engine_record(backend, llm_kwargs))
        manifest["engine"]["software"] = c4.engine_software_versions()
        manifest["engine"]["gpu"] = (hooks.gpu_info or c4.nvidia_smi_gpu_info)()
        common.write_json(manifest_path, manifest)
        with open(tmp_path, "w", encoding="utf-8") as sink:
            for number, (lo, hi) in enumerate(chunks, start=1):
                guard.check()
                t_chunk = time.time()
                records, chunk_errors = read_chunk(backend, id_lists[lo:hi])
                for k, (record, error) in enumerate(zip(records, chunk_errors, strict=True)):
                    i = lo + k
                    row = build_row(args.model, trials[i], lengths[i], digests[i], record)
                    if row["missing"]:
                        errors.append({"trial_id": trials[i]["trial_id"], "error": error or "non-finite log-probability"})
                    rows.append(row)
                    sink.write(dump_row(row) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                common.log(f"chunk {number}/{len(chunks)}: {hi - lo} prompts in {time.time() - t_chunk:.1f} s; missing so far {len(errors)}")
                if len(errors) > max_missing:
                    raise common.EntryError(f"{len(errors)} readings are missing (allowed {max_missing} = {MAX_MISSING_FRACTION:.0%} of {len(trials)}); first: {errors[0]}")
        guard.check()
        os.replace(tmp_path, out / "rows.jsonl")
        status = "ok"
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        restore_sigterm()
        try:
            if backend is not None:
                close_vllm(backend)
        finally:
            ended = now_local()
            manifest.update(status=status, error=failure, end_time=iso(ended), wall_s=time.time() - t0, n_rows=len(rows), n_missing=len(errors), missing_trials=errors[:50])
            if status == "ok":
                manifest["rows_sha256"] = file_sha256(out / "rows.jsonl")
            common.write_json(manifest_path, manifest)
    summary = format_summary(rows, label_ids, tokenizer, model=args.model, prefix=prefix)
    common.write_json(out / "format_summary.json", summary)
    manifest["format_summary_sha256"] = file_sha256(out / "format_summary.json")
    common.write_json(manifest_path, manifest)
    return {
        "run_dir": str(out), "model": args.model, "set": args.set, "n_rows": len(rows), "n_missing": len(errors), "all_judged_strata_pass": summary["all_judged_strata_pass"],
        "n_failing_strata": summary["n_failing_strata"], "gate_pass_rate": summary["sets"][args.set]["gate_pass_rate"], "non_registered": non_registered, "wall_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------------------
# crosscheck: pure rules
# ---------------------------------------------------------------------------------------

def delta_from_max_error(e_max: float) -> float:
    """delta = 0.25 * max(1, ceil(6 * e_max)): 1.5 times the largest deviation, rounded up to a multiple of 0.25, at least 0.25. A deviation that is not finite
    or is negative is a technical failure."""
    if not math.isfinite(e_max) or e_max < 0:
        raise TechnicalFailure(f"the largest deviation must be finite and not negative, got {e_max!r}")
    return DELTA_STEP * max(1, math.ceil(DELTA_MULTIPLIER * e_max))


def scan_selection(pilot_trials: Sequence[Mapping[str, Any]], *, seed: int = prompts.SEEDS["scan"], per_arm: int = SCAN_PER_ARM) -> dict[str, list[int]]:
    """Positions (in ``pilot_trials``) of the prompts to scan, per arm, in scan order: one generator ``default_rng(20261023)``; for the arms R, C, A, B in turn a permutation of
    the arm's pilot trials in plan order, the first ``per_arm`` of it. Every arm must have at least ``per_arm`` pilot trials."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[int]] = {}
    for arm in prompts.ARMS:
        positions = [i for i, t in enumerate(pilot_trials) if t["arm"] == arm]
        if len(positions) < per_arm:
            raise TechnicalFailure(f"arm {arm} has {len(positions)} pilot trials, fewer than the {per_arm} to scan")
        order = rng.permutation(len(positions))[:per_arm]
        out[arm] = [positions[int(j)] for j in order]
    return out


def margin_of(row: Mapping[str, Any]) -> float | None:
    """Margin of a row from its stored log-probabilities (None for a missing reading)."""
    if row["missing"] or row["logprob_1"] is None or row["logprob_2"] is None:
        return None
    return float(row["logprob_1"]) - float(row["logprob_2"])


def judge_float32_scan(items: Sequence[Mapping[str, Any]], delta_m: float, *, min_per_arm: int = MIN_INCLUDED_PER_ARM, min_total: int = MIN_INCLUDED_TOTAL) -> dict[str, Any]:
    """The verdict of the bf16 check on the scanned prompts.

    ``items`` hold, in scan order, ``arm``, ``trial_id``, ``m_ref`` (float32 reference margin, finite) and ``m_vllm`` (None for a missing vLLM reading). A prompt is included when
    ``|m_ref| >= delta_m``. A disagreement is a different choice (the sign of the margin); a missing or tied vLLM reading on an included prompt is a disagreement. Passed
    when every arm has at least ``min_per_arm`` included prompts, all arms together at least ``min_total``, and there is no disagreement."""
    per_arm = {arm: {"scanned": 0, "included": 0, "disagreements": 0, "excluded_near_tie": 0} for arm in prompts.ARMS}
    disagreements: list[dict[str, Any]] = []
    included: list[str] = []
    for item in items:
        m_ref = item["m_ref"]
        if m_ref is None or not math.isfinite(m_ref):
            raise TechnicalFailure(f"{item['trial_id']}: the float32 reference margin is not finite ({m_ref!r})")
        stats = per_arm[item["arm"]]
        stats["scanned"] += 1
        if abs(m_ref) < delta_m:
            stats["excluded_near_tie"] += 1
            continue
        stats["included"] += 1
        included.append(item["trial_id"])
        choice_ref = "1" if m_ref > 0 else "2"
        m_v = item["m_vllm"]
        if m_v is None:
            reason, choice_v = "vllm reading missing", None
        else:
            tie, choice_v = prompts.choice_from_margin(m_v)
            reason = "vllm reading tied" if tie else ("different choice" if choice_v != choice_ref else "")
        if reason:
            stats["disagreements"] += 1
            disagreements.append({"trial_id": item["trial_id"], "arm": item["arm"], "m_ref": m_ref, "m_vllm": m_v, "choice_ref": choice_ref, "choice_vllm": choice_v, "reason": reason})
    total = sum(s["included"] for s in per_arm.values())
    reasons = []
    for arm, s in per_arm.items():
        if s["included"] < min_per_arm:
            reasons.append(f"arm {arm}: {s['included']} included prompts, fewer than {min_per_arm}")
    if total < min_total:
        reasons.append(f"{total} included prompts in all, fewer than {min_total}")
    if disagreements:
        reasons.append(f"{len(disagreements)} decision disagreements among the included prompts (zero allowed)")
    return {
        "delta_m": delta_m, "per_arm": per_arm, "n_scanned": sum(s["scanned"] for s in per_arm.values()), "n_included": total, "n_disagreements": len(disagreements),
        "disagreements": disagreements, "min_per_arm": min_per_arm, "min_total": min_total, "passed": not reasons, "reasons": reasons,
    }


def judge_repeat(items: Sequence[Mapping[str, Any]], *, required: int = REPEAT_REQUIRED) -> dict[str, Any]:
    """The verdict of the quantized-model check on two readings of the same prompts.

    ``items`` hold ``trial_id``, ``arm`` and the two readings ``r1`` and ``r2`` as rows of ``rows.jsonl``. A prompt is valid and equal when neither reading is missing, both pass
    the gate, neither is a tie and the choices are the same; a tie counts as no choice. Passed when at least ``required`` prompts are valid and equal. delta_q comes from the
    largest |m_1 - m_2| over the prompts where both margins exist (a technical failure when there is none)."""
    n = len(items)
    n_ok = n_quality = n_tie = n_flip = 0
    flips: list[dict[str, Any]] = []
    quality_failures: list[dict[str, Any]] = []
    diffs: list[float] = []
    per_arm = {arm: {"n": 0, "valid_and_equal": 0} for arm in prompts.ARMS}
    for item in items:
        r1, r2 = item["r1"], item["r2"]
        per_arm[item["arm"]]["n"] += 1
        m1, m2 = margin_of(r1), margin_of(r2)
        if m1 is not None and m2 is not None:
            diffs.append(abs(m1 - m2))
        if r1["missing"] or r2["missing"] or not (r1["gate_pass"] and r2["gate_pass"]):
            n_quality += 1
            quality_failures.append({"trial_id": item["trial_id"], "arm": item["arm"], "missing": [bool(r1["missing"]), bool(r2["missing"])], "gate_pass": [r1["gate_pass"], r2["gate_pass"]]})
        elif r1["tie"] or r2["tie"]:
            n_tie += 1
        elif r1["choice"] != r2["choice"]:
            n_flip += 1
            flips.append({"trial_id": item["trial_id"], "arm": item["arm"], "m_1": m1, "m_2": m2, "choice_1": r1["choice"], "choice_2": r2["choice"]})
        else:
            n_ok += 1
            per_arm[item["arm"]]["valid_and_equal"] += 1
    if not diffs:
        raise TechnicalFailure("no prompt has a margin in both readings: delta_q cannot be computed")
    e_max = float(max(diffs))
    delta_q = delta_from_max_error(e_max)
    passed = n_ok >= required
    m_first = [m for m in (margin_of(i["r1"]) for i in items) if m is not None]
    m_second = [m for m in (margin_of(i["r2"]) for i in items) if m is not None]
    return {
        "n": n, "required_valid_and_equal": required, "n_valid_and_equal": n_ok, "n_quality_failures": n_quality, "n_ties": n_tie, "n_decision_flips": n_flip, "per_arm": per_arm,
        "decision_flips": flips, "quality_failures": quality_failures, "e_max": e_max, "delta_q": delta_q, "n_margin_pairs": len(diffs),
        "margin_abs_diff": common.describe_quantiles(diffs), "m_first_reading": common.describe_quantiles(m_first), "m_second_reading": common.describe_quantiles(m_second), "passed": passed,
        "reasons": [] if passed else [f"{n_ok} of {n} prompts are valid and equal, {required} needed"],
    }


# ---------------------------------------------------------------------------------------
# crosscheck: inputs (the runs of ``read``)
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RunData:
    directory: Path
    manifest: dict[str, Any]
    rows: dict[tuple[Any, ...], dict[str, Any]]


def load_run(directory: str | Path, *, model: str, set_name: str, plans: prompts.Plans) -> RunData:
    """A finished ``read`` run: manifest ok, the same model, set and plans, ``rows.jsonl`` with the hash the manifest names, one row per planned trial whose plan fields equal the
    plan. Anything else is a technical failure."""
    directory = Path(directory)
    try:
        manifest = common.read_json(directory / "manifest.json")
    except (OSError, ValueError) as error:
        raise TechnicalFailure(f"{directory}: cannot read manifest.json ({error})") from error
    problems = []
    if manifest.get("schema") != RUN_SCHEMA or manifest.get("status") != "ok":
        problems.append(f"manifest schema/status {manifest.get('schema')!r}/{manifest.get('status')!r}")
    for key, expected in (("model", model), ("set", set_name), ("plans_sha256", plans.sha256)):
        if manifest.get(key) != expected:
            problems.append(f"manifest {key} is {manifest.get(key)!r}, expected {expected!r}")
    pinned = (manifest.get("template_date") or {}).get("pinned")
    if pinned != prompts.TEMPLATE_DATE:
        problems.append(f"the run pinned the template date {pinned!r}, this code pins {prompts.TEMPLATE_DATE!r}")
    if problems:
        raise TechnicalFailure(f"{directory}: " + "; ".join(problems))
    rows_path = directory / "rows.jsonl"
    if file_sha256(rows_path) != manifest.get("rows_sha256"):
        raise TechnicalFailure(f"{rows_path} does not have the sha256 that manifest.json recorded")
    rows = load_run_rows(rows_path)
    by_key = {prompts.trial_key(r): r for r in rows}
    if len(by_key) != len(rows):
        raise TechnicalFailure(f"{rows_path}: a trial occurs more than once")
    planned = plans.of_set(set_name)
    if len(by_key) != len(planned):
        raise TechnicalFailure(f"{rows_path}: {len(by_key)} rows for {len(planned)} planned trials")
    for trial in planned:
        row = by_key.get(prompts.trial_key(trial))
        if row is None:
            raise TechnicalFailure(f"{rows_path}: no row for {trial['trial_id']}")
        if row["model"] != model or any(row[k] != trial[k] for k in prompts.PLAN_ROW_FIELDS):
            raise TechnicalFailure(f"{rows_path}: the row of {trial['trial_id']} does not match the plan")
    return RunData(directory, manifest, by_key)


def same_engine(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Differences between two runs that must have the same model, prefix, plans and engine arguments."""
    out = []
    for key in ("model_path", "assistant_prefix", "plans_sha256", "template_date"):
        if a.get(key) != b.get(key):
            out.append(f"{key}: {a.get(key)!r} vs {b.get(key)!r}")
    if a["engine"]["llm_kwargs"] != b["engine"]["llm_kwargs"]:
        out.append(f"llm_kwargs: {a['engine']['llm_kwargs']} vs {b['engine']['llm_kwargs']}")
    if a["versions"].get("vllm") != b["versions"].get("vllm"):
        out.append(f"vllm version: {a['versions'].get('vllm')} vs {b['versions'].get('vllm')}")
    return out


def check_digests(trials: Sequence[Mapping[str, Any]], id_lists: Sequence[Sequence[int]], rows: Mapping[tuple[Any, ...], Mapping[str, Any]], where: str) -> None:
    """The prompts of the check have exactly the token ids of the reading they are compared with (``prompt_sha256`` of the row)."""
    bad = [t["trial_id"] for t, ids in zip(trials, id_lists, strict=True) if prompts.token_digest(ids) != rows[prompts.trial_key(t)]["prompt_sha256"]]
    if bad:
        raise TechnicalFailure(f"{where}: {len(bad)} prompts have other token ids than in the run they are compared with (first: {bad[:5]}); same tokenizer, chat parameters and prefix are needed")


# ---------------------------------------------------------------------------------------
# crosscheck: bf16 models, float32 reference
# ---------------------------------------------------------------------------------------

def open_hf_reference(entry: Mapping[str, Any], tokenizer: Any, spec: ReadoutSpec, device: str) -> tuple[Any, dict[str, Any]]:
    """The float32 reference side: strict float32 (TF32 off, no autocast), eval mode, batch size 1 (no padding); the read-back of the switches is returned."""
    from nmi.readout.hf_backend import HFBackend

    numerics = common.assert_strict_float32(common.apply_strict_float32())
    model, loader = common.load_hf_model(entry["path"], dtype="float32", device=device)
    backend = HFBackend(model, tokenizer, spec, device=device, batch_size=1)
    info = {"loader": loader, "numerics": numerics, "numeric_actual": common.hf_numeric_actual(model, batch_size=1, batch_plan=common.hf_batch_plan(backend))}
    return backend, info


def crosscheck_float32(args: argparse.Namespace, hooks: Hooks, entry: Mapping[str, Any], plans: prompts.Plans, tokenizer: Any, calib: RunData, pilot: RunData) -> dict[str, Any]:
    chat_kwargs = common.chat_kwargs_of(entry)
    prefix = calib.manifest["assistant_prefix"]
    spec = two_label_spec()
    calib_trials = plans.of_set("calib")
    if len(calib_trials) != CAL_N:
        raise TechnicalFailure(f"the calibration set has {len(calib_trials)} prompts, the design needs {CAL_N}")
    pilot_trials = plans.of_set("pilot")
    scan = scan_selection(pilot_trials)
    scan_trials = [pilot_trials[i] for arm in prompts.ARMS for i in scan[arm]]
    cal_ids = tokenize_trials(tokenizer, calib_trials, chat_kwargs, prefix)
    scan_ids = tokenize_trials(tokenizer, scan_trials, chat_kwargs, prefix)
    check_digests(calib_trials, cal_ids, calib.rows, "calibration prompts")
    check_digests(scan_trials, scan_ids, pilot.rows, "pilot prompts")
    t0 = time.time()
    opener = hooks.hf_opener or open_hf_reference
    reference, info = opener(entry, tokenizer, spec, args.reference_device)
    try:
        common.log(f"float32 reference on {args.reference_device}: {len(cal_ids)} calibration and {len(scan_ids)} pilot prompts")
        ref_cal = reference.readouts_from_ids(cal_ids)
        ref_scan = reference.readouts_from_ids(scan_ids)
    finally:
        reference = None  # the weights must leave the card (or the memory) before anything else is loaded
        common.free_gpu_memory()
    m_ref_cal = [float(r.readout.margin) for r in ref_cal]
    m_v_cal = [margin_of(calib.rows[prompts.trial_key(t)]) for t in calib_trials]
    if any(m is None or not math.isfinite(m) for m in m_v_cal) or any(not math.isfinite(m) for m in m_ref_cal):
        bad = [calib_trials[i]["trial_id"] for i, (a, b) in enumerate(zip(m_ref_cal, m_v_cal, strict=True)) if b is None or not math.isfinite(a) or not math.isfinite(b)]
        raise TechnicalFailure(f"calibration: a margin is missing or not finite ({bad[:5]}); no delta_m, no item dropped, none replaced")
    e_cal = [abs(v - f) for v, f in zip(m_v_cal, m_ref_cal, strict=True)]
    e_max = float(max(e_cal))
    delta_m = delta_from_max_error(e_max)
    items = []
    for trial, record in zip(scan_trials, ref_scan, strict=True):
        items.append({"arm": trial["arm"], "trial_id": trial["trial_id"], "m_ref": float(record.readout.margin), "m_vllm": margin_of(pilot.rows[prompts.trial_key(trial)])})
    verdict = judge_float32_scan(items, delta_m)
    scanned_e = [abs(i["m_vllm"] - i["m_ref"]) for i in items if i["m_vllm"] is not None]
    return {
        "kind": "float32_reference", "passed": verdict["passed"], "reasons": verdict["reasons"], "delta_m": delta_m,
        "delta": {"name": "delta_m", "value": delta_m, "e_max": e_max, "rule": "0.25 * max(1, ceil(6 * max_j |m_V - m_F|)) over the calibration prompts"},
        "calibration": {
            "n": len(calib_trials), "e_m": common.describe_quantiles(e_cal), "m_reference": common.describe_quantiles(m_ref_cal), "m_vllm": common.describe_quantiles([float(m) for m in m_v_cal]),
            "items": [{"trial_id": t["trial_id"], "arm": t["arm"], "m_ref": a, "m_vllm": b, "e_m": e} for t, a, b, e in zip(calib_trials, m_ref_cal, m_v_cal, e_cal, strict=True)],
        },
        "scan": {
            "seed": prompts.SEEDS["scan"], "per_arm": SCAN_PER_ARM, "order": "one generator; for R, C, A, B in turn a permutation of the arm's pilot trials in plan order, the first 75",
            "trial_ids": {arm: [pilot_trials[i]["trial_id"] for i in scan[arm]] for arm in prompts.ARMS}, **verdict,
        },
        "descriptive": {
            "scan_e_m_all_scanned": common.describe_quantiles(scanned_e), "scan_m_reference": common.describe_quantiles([i["m_ref"] for i in items]),
            "n_scan_above_delta_m": int(sum(1 for e in scanned_e if e > delta_m)),
            "note": "the scan is not used to change delta_m; the sensitivity analysis drops the trials with |m| < delta_m",
        },
        "reference": {
            "device": args.reference_device, "dtype": "float32", "batch_size": 1, "padding": "none", "autocast": False, "tf32": False, "loader": info.get("loader"),
            "numerics": info.get("numerics"), "numeric_actual": info.get("numeric_actual"), "wall_s": time.time() - t0,
        },
        "items": items,
    }


# ---------------------------------------------------------------------------------------
# crosscheck: FP8 and AWQ models, second reading in a new engine process
# ---------------------------------------------------------------------------------------

def child_command(child_args: Sequence[str]) -> list[str]:
    interpreter = [sys.executable, *(["-B"] if sys.flags.dont_write_bytecode else [])]
    return [*interpreter, str(SCRIPT), *child_args]


def crosscheck_repeat(args: argparse.Namespace, hooks: Hooks, entry: Mapping[str, Any], plans: prompts.Plans, tokenizer: Any, pilot: RunData, out: Path) -> dict[str, Any]:
    chat_kwargs = common.chat_kwargs_of(entry)
    prefix = pilot.manifest["assistant_prefix"]
    pilot_trials = plans.of_set("pilot")
    scan = scan_selection(pilot_trials)
    scan_positions = [i for arm in prompts.ARMS for i in scan[arm]]
    scan_trials = [pilot_trials[i] for i in scan_positions]
    ids = tokenize_trials(tokenizer, scan_trials, chat_kwargs, prefix)
    check_digests(scan_trials, ids, pilot.rows, "pilot prompts")
    order2 = sorted(scan_positions, reverse=True)  # the pilot run submitted the pilot trials in plan order: the second reading submits the same prompts in the reverse order
    selection = {
        "scan_trial_ids": {arm: [pilot_trials[i]["trial_id"] for i in scan[arm]] for arm in prompts.ARMS},
        "pass2_order": [pilot_trials[i]["trial_id"] for i in order2], "pass2_order_rule": "reverse of the plan order (the pilot run submitted the pilot trials in plan order)",
    }
    request = {
        "model": args.model, "model_path": str(entry["path"]), "chat_template_kwargs": dict(chat_kwargs), "assistant_prefix": prefix, "llm_kwargs": pilot.manifest["engine"]["llm_kwargs"],
        "plans_sha256": plans.sha256, "nmi_source_sha256": source_hash(),
    }
    common.write_json(out / "selection.json", selection)
    common.write_json(out / "engine_request.json", request)
    child_args = [
        "crosscheck-pass", "--model", args.model, "--plans", str(Path(args.plans).resolve()), "--config", str(Path(args.config).resolve()), "--run-dir", str(out),
        "--expect-plans-sha256", plans.sha256, "--expect-nmi-source-sha256", request["nmi_source_sha256"],
    ]
    pidfile = out / "pass2_child.json"

    def started(pid: int) -> None:  # the PID is on disk before the parent waits: whoever has to stop the child finds it
        common.write_json(pidfile, {"pid": pid, "pgid": pid, "started": iso(now_local()), "command": child_command(child_args), "returncode": None})
        common.log(f"second reading: child process {pid} (its own process group); log {out / 'logs' / 'pass2.log'}")

    spec = c4.ChildSpec(pass_index=2, args=child_args, command=child_command(child_args), log_path=out / "logs" / "pass2.log", cwd=REPO_ROOT, on_start=started)
    t0 = time.time()
    outcome = (hooks.launcher or c4.subprocess_launcher)(spec)
    if pidfile.is_file():
        common.write_json(pidfile, {**common.read_json(pidfile), "returncode": outcome.returncode, "ended": iso(now_local())})
    release = c4.verify_release(outcome, hooks, c4.read_engine_pids(out, 2))
    common.log(f"second reading: child exited with {outcome.returncode}; released: {release['ok']}")
    if outcome.returncode != 0:
        raise TechnicalFailure(f"the child process of the second reading exited with {outcome.returncode} (see {out / 'logs' / 'pass2.log'})")
    rows2 = load_run_rows(out / PASS_FILE)
    by_key2 = {prompts.trial_key(r): r for r in rows2}
    if len(by_key2) != len(rows2) or {prompts.trial_key(t) for t in scan_trials} != set(by_key2):
        raise TechnicalFailure(f"{out / PASS_FILE}: the rows are not the {len(scan_trials)} scanned prompts, each once")
    check_digests(scan_trials, ids, by_key2, "second reading")
    info2 = common.read_json(out / "pass2.json")
    items = [{"trial_id": t["trial_id"], "arm": t["arm"], "r1": pilot.rows[prompts.trial_key(t)], "r2": by_key2[prompts.trial_key(t)]} for t in scan_trials]
    verdict = judge_repeat(items)
    differences = common.numeric_differences(
        {"llm_kwargs": pilot.manifest["engine"]["llm_kwargs"], "settings": pilot.manifest["engine"].get("settings")}, {"llm_kwargs": info2["engine"]["llm_kwargs"], "settings": info2["engine"]["settings"]},
    )
    return {
        "kind": "repeat_reading", "passed": verdict["passed"], "reasons": verdict["reasons"], "delta_q": verdict["delta_q"],
        "delta": {"name": "delta_q", "value": verdict["delta_q"], "e_max": verdict["e_max"], "rule": "0.25 * max(1, ceil(6 * max_j |m_1 - m_2|)) over the 300 prompts"},
        "readings": {
            "first": {"source": str(pilot.directory), "engine_process": "the pilot run of `read`", "submission": pilot.manifest["submission"]},
            "second": {
                "source": str(out / PASS_FILE), "engine_process": "a new, independent process", "order": selection["pass2_order_rule"], "n_prompts": len(scan_trials), "calls": 1,
                "child": common.read_json(pidfile) if pidfile.is_file() else None, "release": release, "wall_s": time.time() - t0, "engine": info2["engine"],
            },
            "engine_differences_vs_pilot_run": differences,
        },
        "repeat": verdict,
        "scan": {"seed": prompts.SEEDS["scan"], "per_arm": SCAN_PER_ARM, "trial_ids": selection["scan_trial_ids"]},
        "items": [{"trial_id": i["trial_id"], "arm": i["arm"], "m_1": margin_of(i["r1"]), "m_2": margin_of(i["r2"]), "choice_1": i["r1"]["choice"], "choice_2": i["r2"]["choice"],
                   "gate_pass_1": i["r1"]["gate_pass"], "gate_pass_2": i["r2"]["gate_pass"]} for i in items],
    }


def cmd_crosscheck_pass(args: argparse.Namespace, hooks: Hooks) -> int:
    """Internal: the second reading of a quantized model, in a process of its own: renders the 300 prompts in the order of ``selection.json``, starts one engine with the arguments of the
    pilot run, makes ONE ``readouts()`` call, closes the engine and writes ``pass2_rows.jsonl`` and ``pass2.json``. The parent checks and judges."""
    run_dir = check_off_root(args.run_dir)
    if args.expect_nmi_source_sha256 and source_hash() != args.expect_nmi_source_sha256:
        raise SourceChangedError("nmi/ is not the source the parent started with")
    guard = SourceHashGuard()
    try:
        request = common.read_json(run_dir / "engine_request.json")
        selection = common.read_json(run_dir / "selection.json")
    except (OSError, ValueError) as error:
        raise common.EntryError(f"{run_dir}: the parent's request files cannot be read ({error})") from error
    plans = load_plans_checked(args.plans)
    if args.expect_plans_sha256 and plans.sha256 != args.expect_plans_sha256:
        raise common.EntryError("the plans are not the ones the parent started with")
    by_id = {t["trial_id"]: t for t in plans.of_set("pilot")}
    unknown = [i for i in selection["pass2_order"] if i not in by_id]
    if unknown:
        raise common.EntryError(f"selection.json names trials that are not in the pilot plans: {unknown[:3]}")
    trials = [by_id[i] for i in selection["pass2_order"]]
    entries, _ = load_model_entries(args.config)
    entry = model_entry(entries, args.model)
    tokenizer = load_tokenizer(entry, hooks)
    spec = two_label_spec()
    label_ids = spec.token_ids(tokenizer)
    id_lists = tokenize_trials(tokenizer, trials, request["chat_template_kwargs"], request["assistant_prefix"])
    digests = [prompts.token_digest(ids) for ids in id_lists]
    started = now_local()
    t_engine = time.time()
    apply_env_defaults()
    restore_sigterm = install_sigterm()
    backend = (hooks.vllm_opener or open_vllm)(Path(request["model_path"]), tokenizer, spec, request["llm_kwargs"])
    engine_start_s = time.time() - t_engine
    below = c4.descendant_pids(os.getpid())  # the engine-core process holds the GPU memory: its PID goes on disk now, so that the parent can look for it after this process has gone
    common.write_json(run_dir / "pass2_processes.json", {"pass": 2, "pid": os.getpid(), "descendants": below, "written": iso(now_local())})
    try:
        record = engine_record(backend, request["llm_kwargs"])
        guard.check()
        t_read = time.time()
        records = backend.readouts(id_lists)  # the one call of the pass: every prompt, no warm-up call, no explicit cache reset (the prefix cache is empty in a new process)
        read_s = time.time() - t_read
        guard.check()
    finally:
        restore_sigterm()
        close_vllm(backend)
    if len(records) != len(id_lists):
        raise common.EntryError(f"the engine returned {len(records)} readouts for {len(id_lists)} prompts")
    rows = [build_row(args.model, t, len(ids), d, r) for t, ids, d, r in zip(trials, id_lists, digests, records, strict=True)]
    with open(run_dir / PASS_FILE, "w", encoding="utf-8") as sink:
        sink.write("".join(dump_row(r) + "\n" for r in rows))
    common.write_json(run_dir / "pass2.json", {
        "pid": os.getpid(), "started": iso(started), "ended": iso(now_local()), "engine_start_s": engine_start_s, "read_s": read_s, "n_prompts": len(rows), "calls": 1,
        "order": selection["pass2_order_rule"], "engine": {"llm_kwargs": request["llm_kwargs"], **record, "software": c4.engine_software_versions()},
        "label_token_ids": [int(i) for i in label_ids], "rows_sha256": file_sha256(run_dir / PASS_FILE),
    })
    common.log(f"second reading: {len(rows)} prompts in {read_s:.1f} s (engine start {engine_start_s:.1f} s)")
    return 0


def cmd_crosscheck(args: argparse.Namespace, hooks: Hooks) -> int:
    entries, cfg_md5 = load_model_entries(args.config)
    entry = model_entry(entries, args.model)
    kind = REGISTERED_RUN.get(args.model, DEFAULT_RUN)["crosscheck"]
    out = check_off_root(args.out)
    plans = load_plans_checked(args.plans)
    notes: list[str] = []
    if kind == "float32":
        if not args.calib_run:
            raise common.EntryError(f"{args.model} is a bf16 model: --calib-run (the output of `read --set calib`) is required")
        if not args.reference_device or not re.fullmatch(r"cpu|cuda(:[0-9]+)?", args.reference_device):
            raise common.EntryError("--reference-device must be 'cpu', 'cuda' or 'cuda:N' for a bf16 model")
    else:
        if args.calib_run:
            notes.append("--calib-run is not used for a quantized model (the check is a repeated reading)")
        if args.reference_device:
            notes.append("--reference-device is not used for a quantized model")
        for note in notes:
            common.log(note)
    if out.exists() and any(out.iterdir()):
        raise common.EntryError(f"{out} is not empty: every check needs a directory of its own")
    pilot = load_run(args.pilot_run, model=args.model, set_name="pilot", plans=plans)
    calib = load_run(args.calib_run, model=args.model, set_name="calib", plans=plans) if kind == "float32" else None
    if calib is not None:
        problems = same_engine(calib.manifest, pilot.manifest)
        if problems:
            raise TechnicalFailure("the calibration run and the pilot run differ: " + "; ".join(problems))
    tokenizer = load_tokenizer(entry, hooks)
    out.mkdir(parents=True, exist_ok=True)
    started = now_local()
    if kind == "float32":
        assert calib is not None
        result = crosscheck_float32(args, hooks, entry, plans, tokenizer, calib, pilot)
    else:
        result = crosscheck_repeat(args, hooks, entry, plans, tokenizer, pilot, out)
    report = {
        "schema": CROSSCHECK_SCHEMA, "model": args.model, "plans_sha256": plans.sha256, "assistant_prefix": pilot.manifest["assistant_prefix"], "start_time": iso(started),
        "end_time": iso(now_local()), "notes": notes,
        "inputs": {
            "pilot_run": {"dir": str(pilot.directory), "rows_sha256": pilot.manifest["rows_sha256"], "code": pilot.manifest["code"]["sha256"], "engine_llm_kwargs": pilot.manifest["engine"]["llm_kwargs"],
                          "non_registered": pilot.manifest["engine"]["non_registered"], "vllm_version": pilot.manifest.get("vllm_version")},
            "calib_run": None if calib is None else {"dir": str(calib.directory), "rows_sha256": calib.manifest["rows_sha256"], "code": calib.manifest["code"]["sha256"]},
        },
        "config": {"path": str(Path(args.config).resolve()), "md5": cfg_md5}, "code": code_hashes([HERE / "run_c4_repeat.py"]), **result,
    }
    items = report.pop("items")
    common.write_json(out / "crosscheck.json", report)
    (out / "crosscheck_items.jsonl").write_text("".join(json.dumps(common.json_safe(i), separators=(",", ":"), allow_nan=False) + "\n" for i in items), encoding="utf-8")
    print(json.dumps(common.json_safe({
        "model": args.model, "kind": report["kind"], "passed": report["passed"], "reasons": report["reasons"], "delta": report["delta"], "out": str(out),
        **({"per_arm": report["scan"]["per_arm"], "n_included": report["scan"]["n_included"], "n_disagreements": report["scan"]["n_disagreements"]} if report["kind"] == "float32_reference"
           else {k: report["repeat"][k] for k in ("n", "n_valid_and_equal", "n_quality_failures", "n_ties", "n_decision_flips", "required_valid_and_equal")}),
    }), indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


# ---------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Runner of the explicit-number confirmation test.")
    sub = p.add_subparsers(dest="command", required=True, metavar="{build,inspect,read,crosscheck}")  # crosscheck-pass is internal and stays out of the usage line

    def config_arg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", default=str(DEFAULT_CONFIG), help="configuration with models.devices (default configs/v6_confirm.yaml)")

    b = sub.add_parser("build", help="write conditions.json, plans.json and their hashes (CPU, no model)")
    b.add_argument("--out", required=True, help="output directory")
    b.add_argument("--overwrite", action="store_true", help="replace existing plan files with other bytes (the plans are frozen: normally an error)")

    i = sub.add_parser("inspect", help="tokenizer checks of every trial for one model (CPU)")
    i.add_argument("--model", required=True)
    i.add_argument("--plans", required=True, help="directory made by `build`")
    i.add_argument("--assistant-prefix", default="", help="'' (default) or 'Answer: ' (the pre-specified fallback)")
    i.add_argument("--out", default=None, help="where inspect_<model>.json goes (default: the plans directory)")
    config_arg(i)

    r = sub.add_parser("read", help="read one set with offline vLLM (GPU)")
    r.add_argument("--model", required=True)
    r.add_argument("--plans", required=True, help="directory made by `build`")
    r.add_argument("--set", required=True, choices=prompts.SETS)
    r.add_argument("--out", required=True, help="run directory (must not hold rows.jsonl yet)")
    r.add_argument("--gpu-mem-util", type=float, required=True, help="vLLM gpu_memory_utilization: a fraction of the card's total memory, chosen from the free memory just before the run")
    r.add_argument("--llm-kwargs", default=None, help="JSON object merged into the keyword arguments of vllm.LLM (qwen3-14b: '{\"quantization\": \"fp8\"}')")
    r.add_argument("--assistant-prefix", default="", help="'' (default) or 'Answer: ' (the pre-specified fallback)")
    r.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    r.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="prompts per readouts() call (recorded in the manifest)")
    r.add_argument("--allow-nonregistered", action="store_true", help="accept engine settings that differ from the registered ones (recorded in the manifest)")
    config_arg(r)

    c = sub.add_parser("crosscheck", help="numerical check of the readings")
    c.add_argument("--model", required=True)
    c.add_argument("--plans", required=True)
    c.add_argument("--calib-run", default=None, help="output directory of `read --set calib` (bf16 models)")
    c.add_argument("--pilot-run", required=True, help="output directory of `read --set pilot`")
    c.add_argument("--out", required=True, help="empty or new directory for crosscheck.json")
    c.add_argument("--reference-device", default=None, help="device of the float32 reference: cpu, cuda or cuda:N (bf16 models)")
    config_arg(c)

    cp = sub.add_parser("crosscheck-pass")  # internal (no help text, so it is not listed): the second reading of a quantized model, started by `crosscheck`
    cp.add_argument("--model", required=True)
    cp.add_argument("--plans", required=True)
    cp.add_argument("--run-dir", required=True)
    cp.add_argument("--expect-plans-sha256", default=None)
    cp.add_argument("--expect-nmi-source-sha256", default=None)
    config_arg(cp)
    return p


def main(argv: Sequence[str] | None = None, hooks: Hooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or Hooks()
    args = make_parser().parse_args(argv)
    try:
        if args.command == "build":
            return cmd_build(args, hooks)
        if args.command == "inspect":
            return cmd_inspect(args, hooks)
        if args.command == "read":
            outcome = execute_read(args, hooks, argv)
            print(json.dumps(common.json_safe(outcome), indent=2, sort_keys=True))
            return 0
        if args.command == "crosscheck":
            return cmd_crosscheck(args, hooks)
        if args.command == "crosscheck-pass":
            return cmd_crosscheck_pass(args, hooks)
    except (common.EntryError, paths.PathPolicyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except SourceChangedError as error:
        return common.stopped_by_source_change(error)
    except TechnicalFailure as error:
        print(f"technical failure: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
