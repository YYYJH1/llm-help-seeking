#!/usr/bin/env python
"""C1 confirmation run of one end-side model on the holdout pool. Stage ``c1confirm``.

One run reads, for one model, P-answer (every item once, with the assistant prefix ``Answer: `` and the prompt file of the configuration) and the 54-cell P-ask grid (paraphrase a00,
framing F2, standard layout; a_H in {0.60, 0.80, 0.95} x c0 in {0, 0.1, 0.2} x w in {0, 0.1, 0.2, 0.45, 0.8, 1.5}, every item once in every cell). It writes ``answer.parquet`` (the columns of the
``calib.parquet`` of the calibration run; pool ``mmlu_redux_holdout``, split ``confirm``), ``decisions.parquet`` (the columns of the ``decisions.parquet`` of ``run_decisions``) and ``c1_confirm_summary.json``.

    python -I scripts/run_c1_confirm.py --config configs/v6_confirm.yaml --results-root RESULTS --split-dir SPLITS --tag TAG --model qwen3-1.7b --gpu-mem-util 0.80 \\
        --batch-manifest BATCH.json --w5-run RUN --expect-w5-run-id calib/RUNID --expect-w5-summary-md5 MD5 --expect-w5-calib-md5 MD5

Frozen temperature. z = ``calibration_columns(log_p[:, :4], gt - 1, T)``: only the four answer options are normalised (not the digit "5" that asks), and T is the temperature of the calibration (answer-phase) run that the
development mechanism run of this model is bound to (``banks["mmlu_redux"]`` of its ``calib_summary.json``), never refitted. The run refuses (exit 2, nothing read) unless ``--w5-run`` is that run
and its ``calib_summary.json`` and ``calib.parquet`` have the expected md5 values (``--expect-w5-run-id``, ``--expect-w5-summary-md5``, ``--expect-w5-calib-md5``).

Order. Every check that can fail comes first and the pool is loaded last: arguments and tag, the model directory, the memory fraction of the offline engine, the frozen temperature and its md5 values,
the batch (``scripts/_confirm_common.py``: the batch id, the pool, the protocol and description md5, the models and the arms; the md5 of the frozen preregistration and of the files the batch
manifest binds; the unseal marker and the registration), and the identity of a run that is resumed. Only then is the run directory made, the run bound to the batch (one run directory for every batch,
model and arm: a new tag cannot start again), and the pool loaded through the holdout access of the batch, which is the one place where the ledger consumes keys. A failure before that consumes nothing.

Units and failures. The work is cut into units (P-answer in chunks of ``--chunk-size`` items, P-ask in groups of ``--conditions-per-unit`` conditions), each written as one part once it is complete. Two kinds
of failure are retried and nothing else: a call failure (the back end raised, or returned fewer readouts than prompts) and missing rows (a readout is empty, or the unit lacks cells of its expected
item x condition set). A readout whose digit mass is low is a valid readout: it is not retried, it stays in the tables and it counts in the denominator of the quality gate. A unit has at most two attempts,
the original and one resume, counted in the central registry (``confirm_runs.jsonl``; ``attempts.jsonl`` of the run directory is a copy). The first complete attempt is the unit's result and a finished
unit is never run again, so no result is picked from several. The partial output of a failed attempt, with a record of what failed, is kept apart in ``failed_attempts/`` and never merged. After a failed attempt the run stops
(exit 3) so that the other units keep their attempts; ``--resume RUN_DIR`` continues in the run directory that the batch is bound to. A unit that failed twice is exhausted: its cells are missing and the
model's C1 is not testable (exit 1 when the run ends so). The quality gate is worked out at the end and recorded in the summary.

``--rehearsal`` runs the same code on the development pool that stands for the holdout pool (``mmlu_redux_cal``), marks the run exploratory and touches no ledger, no registry and no batch.
``--dry-run`` checks everything above, prints the plan and loads no model and no pool.

Exit codes: 0 done (the quality gate and testability are in the summary); 1 done, but a unit failed twice (not testable); 2 unusable input or arguments (nothing was consumed); 3 a unit failed and the
run can be resumed; 4 the source guard stopped the run.

A chain script runs the entry, and on exit 3 runs it again with ``--resume RUN_DIR`` (no ``--tag`` is needed; the registry stops a unit at its second failed attempt, so the loop ends); on 0 and 1 the run
is finished and the next model follows; any other code stops the chain.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _c2_common as common  # noqa: E402
import _confirm_common as cc  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import run_decisions  # noqa: E402

from nmi import c1_confirm_stats as stats  # noqa: E402
from nmi import paths  # noqa: E402
from nmi.data.holdout import HoldoutLedgerError, HoldoutSealedError  # noqa: E402
from nmi.data.registry import SplitError, load_items_from_config  # noqa: E402
from nmi.measure import calibration as cal  # noqa: E402
from nmi.provenance import file_md5, note_skipped  # noqa: E402
from nmi.readout.digits import ReadoutSpec  # noqa: E402

SCRIPT = "run_c1_confirm"
STAGE = "c1confirm"
POOL = "mmlu_redux_holdout"  # the holdout pool of the confirmatory batch (C1 and the mechanism group)
BANK = "mmlu_redux"
ARMS = ("answer", "ask")  # the arms of the batch that this run reads; the mechanism group's arm is another entry's
SPLIT_LABEL = "confirm"
PARAPHRASE, FRAMING, LAYOUT, GRID = stats.PARAPHRASE, stats.FRAMING, stats.LAYOUT, stats.GRID_TAG
ANSWER_TABLE = "answer.parquet"
DECISION_TABLE = "decisions.parquet"
SUMMARY_FILE = "c1_confirm_summary.json"
SUMMARY_SCHEMA = "nmi.c1confirm_summary/1"
ATTEMPTS_FILE = "attempts.jsonl"
FAILED_DIR = "failed_attempts"
MAX_ATTEMPTS = 2  # the original attempt and one resume
EXIT_NOT_TESTABLE = 1
EXIT_UNIT_FAILED = 3
DEFAULT_CONFIG = paths.CONFIG_DIR / "v6_confirm.yaml"
DEFAULT_PREREG = paths.PROTOCOL_DIR / "preregistration_v2.md"
ASSISTANT_PREFIX = "Answer: "  # the assistant prefix of the answer readout, as in the calibration run
FORBIDDEN_NAME_PARTS = ("holdout", "question", "benchmark")  # no file name of a confirmatory run says one of these
W5_STAGE = "calib"
REHEARSAL_REASON = "rehearsal of the C1 confirmation entry on the development pool that stands for the holdout pool: the same code, no ledger, no registry; development evidence that enters no gate"
OUTCOMES = ("complete", "call_failure", "missing_rows")


class UnitFailure(RuntimeError):
    """A unit failed an attempt: the run stops so that the other units keep their attempts, and it can be resumed (exit 3)."""


class UnitExhausted(Exception):
    """A unit has used its two attempts (internal)."""


@dataclass
class ConfirmHooks:
    """Replaceable parts, for the tests: the function that opens the readout back end, and the model-hash switch."""

    opener: Callable[[common.Context, str, str, ReadoutSpec], common.BackendHandle] | None = None
    model_files: bool = True


@dataclass(frozen=True)
class W5Pin:
    """The calibration (answer-phase) run that gives the frozen temperature, with the md5 values that were checked against the expected ones."""

    run_dir: Path
    run_id: str
    model: str
    temperature: float
    summary_md5: str
    calib_md5: str


@dataclass
class Unit:
    """One unit of work: ``build(bh)`` returns the prompts and the per-row columns, ``expected()`` the set of row keys the finished unit must hold exactly."""

    unit_id: str
    arm: str
    n_rows: int
    build: Callable[[common.BackendHandle], tuple[list[str], dict[str, Any]]]
    expected: Callable[[], set]
    free_logprob: bool = False


def chunks(n: int, size: int) -> list[range]:
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


# ---------------------------------------------------------------------------------------
# arguments and the checks that come before the pool
# ---------------------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C1 confirmation run of one end-side model: P-answer and the 54-cell P-ask grid on the holdout pool, through the holdout access of a registered batch.")
    common.add_common_arguments(p)
    common.add_readout_arguments(p)
    p.set_defaults(config=str(DEFAULT_CONFIG))
    g = p.add_argument_group("confirmation run")
    g.add_argument("--rehearsal", action="store_true", help="the same code on the development pool that stands for the holdout pool; exploratory; no ledger, no registry, no batch")
    g.add_argument("--batch-manifest", default=None, help="the batch manifest (JSON) of the registered batch; required unless --rehearsal")
    g.add_argument("--prereg-frozen", default=str(getattr(cc, "FROZEN_PREREG_PATH", DEFAULT_PREREG)), help="the frozen preregistration document (default: the preregistration file of the repository); its md5 must be the one the batch manifest names")
    g.add_argument("--w5-run", required=True, help="the calibration (answer-phase) run of this model that the development mechanism run is bound to: it gives the frozen temperature")
    g.add_argument("--expect-w5-run-id", required=True, help="the expected run id of --w5-run, <stage>/<run directory> as its manifest names it")
    g.add_argument("--expect-w5-summary-md5", required=True, help="the expected md5 of calib_summary.json of --w5-run")
    g.add_argument("--expect-w5-calib-md5", required=True, help="the expected md5 of calib.parquet of --w5-run")
    g.add_argument("--chunk-size", type=int, default=1000, help="items of P-answer in one unit of work (default 1000)")
    g.add_argument("--conditions-per-unit", type=int, default=6, help="conditions of P-ask in one unit of work (default 6)")
    return p


def check_arguments(args: argparse.Namespace) -> bool:
    """The checks on the arguments alone. Returns whether the run is a rehearsal."""
    if args.limit_items is not None or args.pilot_rep is not None:
        raise common.EntryError("--limit-items and --pilot-rep belong to the development entries: a confirmation run reads every item of its pool")
    rehearsal = bool(args.rehearsal)
    if rehearsal and args.batch_manifest:
        raise common.EntryError("a rehearsal touches no ledger, no registry and no batch: it takes no --batch-manifest")
    if not rehearsal and not args.batch_manifest:
        raise common.EntryError("--batch-manifest is required (the registered batch of the run); use --rehearsal for a run on the development pool")
    if args.chunk_size < 1 or args.conditions_per_unit < 1:
        raise common.EntryError("--chunk-size and --conditions-per-unit must be positive numbers")
    if not args.resume and not args.tag:
        raise common.EntryError("--tag is required for a real run")
    if args.tag and any(part in args.tag.lower() for part in FORBIDDEN_NAME_PARTS):
        raise common.EntryError(f"the tag {args.tag!r} names a word that no output of a confirmation run may carry ({list(FORBIDDEN_NAME_PARTS)}): the run directory carries the tag")
    return rehearsal


def check_prompts(ctx: common.Context) -> None:
    """The P-answer of a confirmation run is that of the calibration run: the assistant prefix ``Answer: `` after the start of the assistant turn (an older prompt file has none, and a run
    without it would read the digit at another position). P-ask is the paraphrase a00."""
    if ctx.book.assistant_prefix != ASSISTANT_PREFIX:
        raise common.EntryError(f"the prompt file {ctx.book.path.name} has the assistant prefix {ctx.book.assistant_prefix!r}; the P-answer of a confirmation run needs {ASSISTANT_PREFIX!r}, as in the calibration run")
    if PARAPHRASE not in ctx.book.paraphrases:
        raise common.EntryError(f"the prompt file has no paraphrase {PARAPHRASE!r}")


def psychometric_conditions(cfg: Any) -> list[run_decisions.Condition]:
    """The 54 conditions of the pre-specified grid; the grid of the configuration must be exactly it."""
    conds = [c for c, _ in run_decisions.grid_conditions(cfg, ["psychometric"])]
    if [(c.a_H, c.c0, c.w) for c in conds] != list(stats.CELLS):
        raise common.EntryError("the psychometric grid of the configuration is not the expected 54 cells (a_H 0.60, 0.80, 0.95 x c0 0, 0.1, 0.2 x w 0, 0.1, 0.2, 0.45, 0.8, 1.5)")
    return conds


def read_w5(args: argparse.Namespace) -> W5Pin:
    """The frozen temperature: the calibration (answer-phase) run of ``--model`` whose ``calib_summary.json`` and ``calib.parquet`` have the expected md5 values. Exit 2 otherwise."""
    run_dir = Path(args.w5_run)
    if not run_dir.is_dir():
        raise common.EntryError(f"--w5-run {run_dir}: not a directory")
    calib = common.read_calib_run(run_dir)  # exit 2 when calib_summary.json is missing
    if calib.model != args.model:
        raise common.EntryError(f"--w5-run {run_dir} is the run of {calib.model!r}, not of {args.model!r}")
    run_id = common.parent_run_ids([("calibration run", run_dir)])[0]
    if run_id.split("/", 1)[0] != W5_STAGE:
        raise common.EntryError(f"--w5-run {run_dir} is a run of stage {run_id.split('/', 1)[0]!r}, not of {W5_STAGE!r} (the answer phase of run_calib)")
    if run_id != args.expect_w5_run_id:
        raise common.EntryError(f"--w5-run {run_dir} is the run {run_id!r}, the expected calibration run is {args.expect_w5_run_id!r}; the frozen temperature does not match")
    table = run_dir / common.CALIB_TABLE
    if not table.is_file():
        raise common.EntryError(f"--w5-run {run_dir}: {common.CALIB_TABLE} is missing")
    summary_md5, calib_md5 = file_md5(run_dir / common.CALIB_SUMMARY), file_md5(table)
    if summary_md5 != args.expect_w5_summary_md5:
        raise common.EntryError(f"--w5-run {run_dir}: {common.CALIB_SUMMARY} has md5 {summary_md5}, the frozen value is {args.expect_w5_summary_md5}; the frozen temperature does not match")
    if calib_md5 != args.expect_w5_calib_md5:
        raise common.EntryError(f"--w5-run {run_dir}: {common.CALIB_TABLE} has md5 {calib_md5}, the frozen value is {args.expect_w5_calib_md5}; the frozen temperature does not match")
    temperature = calib.temperature_by_bank.get(BANK)
    if temperature is None or not (temperature > 0.0) or temperature != temperature or temperature == float("inf"):
        raise common.EntryError(f"--w5-run {run_dir}: {common.CALIB_SUMMARY} has no usable temperature for the bank {BANK!r}")
    return W5Pin(run_dir=run_dir, run_id=run_id, model=calib.model, temperature=float(temperature), summary_md5=summary_md5, calib_md5=calib_md5)


def load_batch(args: argparse.Namespace) -> Any:
    try:
        return cc.load_batch_manifest(args.batch_manifest)
    except OSError as error:
        raise cc.ConfirmError(f"--batch-manifest {args.batch_manifest}: {error}") from error


def verify_bound_files(batch: Any) -> dict[str, str]:
    """The files that the batch manifest binds by their path in the repository must still have the md5 values it froze (the entry, the configuration, the prompts, the code). Returns the md5 of each
    checked file. Keys that are not the path of a file of the repository are not files and are left to the common layer."""
    checked: dict[str, str] = {}
    for rel, frozen in sorted(dict(getattr(batch, "files_md5", {}) or {}).items()):
        path = paths.REPO_ROOT / rel
        if not path.is_file():
            continue
        actual = file_md5(path)
        if actual != frozen:
            raise cc.ConfirmError(f"the batch manifest binds {rel} with md5 {frozen}, the file has {actual}; the code or configuration changed after the batch was frozen")
        checked[rel] = actual
    return checked


def check_numeric_settings(batch: Any, args: argparse.Namespace) -> dict[str, Any] | None:
    """When the batch manifest froze numeric settings for the model (``numeric``: ``gpu_mem_util``, ``max_model_len``, ``llm_kwargs``, set model by model: 14B with 0.80 and
    online FP8, 32B with 0.85), the run's own must be exactly those. Returns the frozen settings (None when the manifest froze none)."""
    frozen = (getattr(batch, "numeric", None) or {}).get(args.model)
    if frozen is None:
        return None
    try:
        given = json.loads(args.llm_kwargs) if args.llm_kwargs else {}
    except ValueError as error:
        raise cc.ConfirmError(f"--llm-kwargs is not valid JSON: {error}") from error
    differences = []
    if args.gpu_mem_util is None or abs(float(args.gpu_mem_util) - float(frozen["gpu_mem_util"])) > 1e-9:
        differences.append(f"--gpu-mem-util {args.gpu_mem_util} (frozen {frozen['gpu_mem_util']})")
    if "max_model_len" in frozen and args.max_model_len != frozen["max_model_len"]:
        differences.append(f"--max-model-len {args.max_model_len} (frozen {frozen['max_model_len']})")
    if given != dict(frozen.get("llm_kwargs") or {}):
        differences.append(f"--llm-kwargs {given} (frozen {dict(frozen.get('llm_kwargs') or {})})")
    if differences:
        raise cc.ConfirmError(f"the numeric settings of {args.model!r} differ from those the batch manifest froze: " + "; ".join(differences))
    return dict(frozen)


def precheck_batch(ctx: common.Context, args: argparse.Namespace, batch: Any) -> dict[str, Any]:
    """Everything about the batch that can be checked without touching the ledger: the pool, the model, the arms, the identity recorded for the batch and each arm, the frozen preregistration, the
    bound files, the numeric settings. ``ConfirmError`` (exit 2) at the first failure; nothing is consumed."""
    if batch.pool != POOL:
        raise cc.ConfirmError(f"this entry reads the pool {POOL!r}; the batch {batch.batch_id!r} is for the pool {batch.pool!r}")
    if args.model not in batch.models:
        raise cc.ConfirmError(f"the model {args.model!r} is not in the batch {batch.batch_id!r} ({list(batch.models)})")
    absent = [arm for arm in ARMS if arm not in batch.arms]
    if absent:
        raise cc.ConfirmError(f"the batch {batch.batch_id!r} has no arm {absent} ({list(batch.arms)})")
    for arm in ARMS:
        cc.check_batch_identity(ctx.results_root, batch, model=args.model, arm=arm)
    prereg = Path(args.prereg_frozen)
    if not prereg.is_file():  # the document is not part of this release
        note_skipped(prereg, "frozen pre-registration")
    elif file_md5(prereg) != batch.prereg_md5:
        raise cc.ConfirmError(f"--prereg-frozen {prereg} has md5 {file_md5(prereg)}, the batch manifest names {batch.prereg_md5}: it is not the frozen preregistration of the batch")
    return {"prereg_md5": batch.prereg_md5, "bound_files": verify_bound_files(batch), "numeric_settings_frozen": check_numeric_settings(batch, args)}


def unit_plan(n_items: int, n_conds: int, chunk_size: int, conditions_per_unit: int) -> list[tuple[str, str, int]]:
    """The units of a run as (unit id, arm, number of rows), in the order they are run. It depends on the number of items and conditions and on the chunking only."""
    plan = [(f"answer__c{ci:03d}", "answer", len(rows)) for ci, rows in enumerate(chunks(n_items, chunk_size))]
    plan += [(f"ask__g{gi:02d}", "ask", n_items * len(rows)) for gi, rows in enumerate(chunks(n_conds, conditions_per_unit))]
    return plan


def run_identity(
    args: argparse.Namespace, ctx: common.Context, *, rehearsal: bool, pool: str, batch: Any, w5: W5Pin, conds: Sequence[run_decisions.Condition], n_items: int, pool_md5: str,
) -> dict[str, Any]:
    """What a run is, beyond what the md5 values of the manifest hold: the manifest keeps it (``c1confirm_run``) and a resumed run must have the same. The unit names do not hold the chunking."""
    cells = [[c.a_H, c.c0, c.w] for c in conds]
    return {
        "entry": SCRIPT, "model": args.model, "rehearsal": rehearsal, "pool": pool, "batch_id": None if batch is None else batch.batch_id, "arms": list(ARMS), "split": SPLIT_LABEL,
        "paraphrase": PARAPHRASE, "framing": FRAMING, "layout": LAYOUT, "grid": GRID, "n_cells": len(cells), "cells_md5": hashlib.md5(json.dumps(cells).encode(), usedforsecurity=False).hexdigest(),
        "n_items": n_items, "pool_keys_md5": pool_md5, "chunk_size": args.chunk_size, "conditions_per_unit": args.conditions_per_unit,
        "units": [list(u) for u in unit_plan(n_items, len(cells), args.chunk_size, args.conditions_per_unit)],
        "prompts_md5": ctx.book.md5, "assistant_prefix": ctx.book.assistant_prefix,
        "w5": {"run_id": w5.run_id, "calib_summary_md5": w5.summary_md5, "calib_parquet_md5": w5.calib_md5, "temperature": w5.temperature},
    }


def check_resume_identity(args: argparse.Namespace, identity: Mapping[str, Any]) -> None:
    """The unit names hold the arm and the chunk number only, and a finished unit is never run again. A run resumed with another model, pool, grid, chunking, temperature or calibration run would take the units
    of the first invocation for its own. So the identity of the call is compared with that of the run being resumed BEFORE anything is read or touched: ``EntryError`` (exit 2) when they differ."""
    resume = getattr(args, "resume", None)
    if not resume:
        return
    path = Path(resume) / common.MANIFEST_FILE
    if not path.is_file():
        return  # (open_run says so itself)
    manifest = common.read_json(path)
    recorded = manifest.get("c1confirm_run") if isinstance(manifest, dict) else None
    if not isinstance(recorded, dict):
        raise common.EntryError(f"--resume {resume}: that is not a run of {SCRIPT} (its manifest has no c1confirm_run, the record of what it was started with)")
    differences = common.numeric_differences(recorded, json.loads(json.dumps(identity)))
    if differences:
        raise common.EntryError(f"--resume {resume}: the model, pool, batch, grid, chunking, calibration run and temperature must be those the run was started with: " + "; ".join(differences))


# ---------------------------------------------------------------------------------------
# the units
# ---------------------------------------------------------------------------------------

def ask_key(item_key: str, a_h: float, c0: float, w: float) -> tuple:
    return (item_key, *stats.cell_key(a_h, c0, w))


def plan_units(ctx: common.Context, args: argparse.Namespace, items: Sequence[Any], pool: str, conds: Sequence[run_decisions.Condition], chat_kwargs: Mapping[str, Any]) -> list[Unit]:
    """The units of the run, in the order they are run: P-answer first (the z of the mechanism group), then P-ask. Nothing is rendered here; ``Unit.build`` renders when the unit is run."""
    model = args.model
    units: list[Unit] = []
    for ci, rows in enumerate(chunks(len(items), args.chunk_size)):
        part = list(items[rows.start: rows.stop])

        def build_answer(bh: common.BackendHandle, part: list = part) -> tuple[list[str], dict[str, Any]]:
            texts = common.answer_texts(ctx.book, bh.tokenizer, part, chat_kwargs)
            meta = {"item_key": [i.item_key for i in part], "model": [model] * len(part), "pool": [pool] * len(part), "gt_position": np.asarray([i.gt_position for i in part], dtype=np.int64)}
            return texts, meta

        units.append(Unit(f"answer__c{ci:03d}", "answer", len(part), build_answer, lambda part=part: {i.item_key for i in part}, free_logprob=bool(ctx.book.assistant_prefix)))
    for gi, rows in enumerate(chunks(len(conds), args.conditions_per_unit)):
        group = [(conds[i], (GRID,)) for i in rows]

        def build_ask(bh: common.BackendHandle, group: list = group) -> tuple[list[str], dict[str, Any]]:
            return run_decisions.condition_job(ctx, bh, items, pool, PARAPHRASE, group, chat_kwargs, model, LAYOUT)()

        def expected_ask(group: list = group) -> set:
            return {ask_key(i.item_key, c.a_H, c.c0, c.w) for c, _ in group for i in items}

        units.append(Unit(f"ask__g{gi:02d}", "ask", len(items) * len(group), build_ask, expected_ask))
    if [(u.unit_id, u.arm, u.n_rows) for u in units] != unit_plan(len(items), len(conds), args.chunk_size, args.conditions_per_unit):
        raise cc.ConfirmError("the units planned are not the units of the plan of the identity")
    return units


def row_keys(table: pa.Table, arm: str) -> list:
    """The keys of the rows of a unit's table: the item key for P-answer, the item key and the cell for P-ask."""
    keys = table.column("item_key").to_pylist()
    if arm == "answer":
        return keys
    a_h, c0, w = (table.column(n).to_numpy(zero_copy_only=False) for n in ("a_H", "c0", "w_displayed"))
    return [ask_key(k, x, y, z) for k, x, y, z in zip(keys, a_h, c0, w, strict=True)]


def take(meta: Mapping[str, Any], positions: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, values in meta.items():
        out[name] = np.asarray(values)[list(positions)] if isinstance(values, np.ndarray) else [values[i] for i in positions]
    return out


class AttemptBook:
    """The attempts of the units of one run. A formal run counts them in the central registry (``confirm_runs.jsonl`` through ``_confirm_common.record_attempt``, with the binding of the unit's arm),
    which is what the limit of two attempts rests on, and the common layer copies every record to ``attempts.jsonl`` of the run directory. A rehearsal has no registry: it counts in the run directory
    alone, in a file of the same records (``kind``, ``unit_id``, ``attempt``, ``outcome``)."""

    def __init__(self, path: Path, *, model: str, results_root: Path | None = None, bindings: Mapping[str, Any] | None = None) -> None:
        self.path = path
        self.model = model
        self.results_root = results_root
        self.bindings = bindings

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def started(self, unit_id: str) -> int:
        """The attempts started, as the copy in the run directory has them (the registry decides in a formal run; see ``start``)."""
        return sum(1 for r in self.rows() if r.get("unit_id") == unit_id and r.get("outcome") == "started")

    def exhausted(self, unit: Unit) -> bool:
        return self.started(unit.unit_id) >= MAX_ATTEMPTS

    def _append(self, unit: Unit, attempt: int, outcome: str) -> None:
        record = {"kind": "attempt", "batch_id": None, "model": self.model, "arm": unit.arm, "run_dir": str(self.path.parent), "unit_id": unit.unit_id, "attempt": attempt, "outcome": outcome,
                  "utc": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()

    def _registry_started(self, unit: Unit) -> int:
        rows = cc.attempts(self.results_root, self.bindings[unit.arm], unit.unit_id)
        return sum(1 for r in rows if r.get("outcome") == "started")

    def start(self, unit: Unit) -> int:
        """Begin an attempt of ``unit``; ``UnitExhausted`` when the two attempts are used."""
        local = self.started(unit.unit_id)
        if local >= MAX_ATTEMPTS:
            raise UnitExhausted(unit.unit_id)
        if self.bindings is None:
            attempt = local + 1
            self._append(unit, attempt, "started")
            return attempt
        try:
            return int(cc.record_attempt(self.results_root, self.bindings[unit.arm], unit.unit_id, "started"))
        except cc.ConfirmError as error:
            if self._registry_started(unit) >= MAX_ATTEMPTS:  # the registry is the authority: the unit has used its attempts, whatever the copy in the run directory says
                raise UnitExhausted(unit.unit_id) from error
            raise

    def finish(self, unit: Unit, attempt: int, outcome: str) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        if self.bindings is None:
            self._append(unit, attempt, outcome)
        else:
            cc.record_attempt(self.results_root, self.bindings[unit.arm], unit.unit_id, outcome)


def run_unit(run: common.RunHandle, bh: common.BackendHandle, unit: Unit, book: AttemptBook) -> str:
    """One attempt of a unit. Returns "complete", "exhausted" (nothing was run), "call_failure" or "missing_rows". The unit's part is written only when the attempt is complete and holds exactly the
    expected rows; a failed attempt leaves its partial output in ``failed_attempts/`` and nothing in ``parts/``."""
    texts, meta = unit.build(bh)
    if len(texts) != unit.n_rows:
        raise cc.ConfirmError(f"{unit.unit_id}: the unit renders {len(texts)} prompts, its plan says {unit.n_rows}")
    try:
        attempt = book.start(unit)
    except UnitExhausted:
        return "exhausted"
    started = time.time()
    outcome: str | None = None
    detail: str | None = None
    records: Sequence[Any] = ()
    try:
        records = bh.backend.readouts(texts)
    except Exception as error:  # the back end raised (the card was taken, an out-of-memory error, ...): a call failure
        outcome, detail = "call_failure", f"{type(error).__name__}: {str(error)[:300]}"
    if outcome is None:
        if len(records) != len(texts):
            outcome, detail = "call_failure", f"the back end returned {len(records)} readouts for {len(texts)} prompts"
        elif any(r is None for r in records):
            outcome, detail = "missing_rows", f"{sum(1 for r in records if r is None)} of {len(texts)} readouts are empty"
    table = None
    if outcome is None:
        table = common.table_from_columns({**meta, **common.trial_columns(records, free_logprob=unit.free_logprob)})
        keys = row_keys(table, unit.arm)
        expected = unit.expected()
        if len(set(keys)) != len(keys) or not set(keys) <= expected:
            raise cc.ConfirmError(f"{unit.unit_id}: the unit holds rows that are repeated or not expected (a defect of the unit, not a failure to retry)")
        if set(keys) != expected:
            outcome, detail = "missing_rows", f"{len(expected - set(keys))} of the {len(expected)} expected rows are missing"
    if outcome is None:
        assert table is not None
        run.parts.write(unit.unit_id, table)
        book.finish(unit, attempt, "complete")
        common.log(f"c1confirm {unit.unit_id}: {len(texts)} prompts in {time.time() - started:.1f} s (attempt {attempt})")
        return "complete"
    kept = [i for i, r in enumerate(records) if r is not None][: len(texts)]
    if kept:
        partial = common.table_from_columns({**take(meta, kept), **common.trial_columns([records[i] for i in kept], free_logprob=unit.free_logprob)})
        common.write_parquet(run.run_dir / FAILED_DIR / f"{unit.unit_id}__attempt{attempt}.parquet", partial)
    common.write_json(run.run_dir / FAILED_DIR / f"{unit.unit_id}__attempt{attempt}.json", {
        "unit_id": unit.unit_id, "arm": unit.arm, "attempt": attempt, "outcome": outcome, "detail": detail, "n_prompts": len(texts), "n_readouts": len(records), "n_partial_rows": len(kept),
        "partial_output": None if not kept else f"{unit.unit_id}__attempt{attempt}.parquet", "merged": False,
    })
    book.finish(unit, attempt, outcome)
    common.log(f"c1confirm {unit.unit_id}: attempt {attempt} of {MAX_ATTEMPTS} failed ({outcome}): {detail}")
    return outcome


# ---------------------------------------------------------------------------------------
# the tables and the summary
# ---------------------------------------------------------------------------------------

def assemble_answer(raw: pa.Table, model: str, temperature: float) -> pa.Table:
    """``answer.parquet`` from the P-answer parts: the columns of the calibration ``calib.parquet``. z is the calibrated confidence of the FOUR option log-probabilities at the frozen temperature
    (``calibration_columns(log_p[:, :4], gt - 1, T)``); the digit that asks is not normalised with them, and the temperature is not fitted."""
    log_p = common.column_array(raw, "digit_logprobs")
    gt = raw.column("gt_position").to_numpy(zero_copy_only=False).astype(np.int64)
    cols = cal.calibration_columns(log_p[:, :4], gt - 1, temperature)
    n = len(gt)
    columns: dict[str, Any] = {
        "item_key": raw.column("item_key").to_pylist(), "model": [model] * n, "pool": raw.column("pool").to_pylist(), "split": [SPLIT_LABEL] * n, "p_options_raw": log_p[:, :4],
        "temperature": np.full(n, float(temperature)), "z": cols["z"], "pred": cols["pred"], "correct": cols["correct"], "digit_logprobs": log_p,
        "digit_mass": raw.column("digit_mass").to_numpy(zero_copy_only=False), "gate_pass": raw.column("gate_pass").to_numpy(zero_copy_only=False),
        "free_token_id": raw.column("free_token_id").to_pylist(), "free_is_digit": raw.column("free_is_digit").to_numpy(zero_copy_only=False), "gt_position": gt,
    }
    if "free_token_logprob" in raw.column_names:
        columns["free_token_logprob"] = raw.column("free_token_logprob").to_numpy(zero_copy_only=False)
    return common.table_from_columns(columns)


def assemble_decisions(raw: pa.Table) -> pa.Table:
    """``decisions.parquet`` from the P-ask parts, as ``run_decisions`` writes its table (``self_correct`` added, the columns in the order of the ``run_decisions`` table)."""
    choice = raw.column("self_choice").to_numpy(zero_copy_only=False)
    gt = raw.column("gt_position").to_numpy(zero_copy_only=False)
    table = raw.append_column("self_correct", pa.array(np.asarray(choice == gt, dtype=bool)))
    return table.select(run_decisions.DECISION_FIRST + [c for c in table.column_names if c not in run_decisions.DECISION_FIRST])


def ask_counts(decisions: pa.Table | None) -> tuple[list[int], list[int]]:
    """Per cell of the grid (the order of ``stats.CELLS``): the rows present and the rows that pass the digit-mass gate."""
    rows, passes = [0] * len(stats.CELLS), [0] * len(stats.CELLS)
    if decisions is None:
        return rows, passes
    a_h, c0, w = (decisions.column(n).to_numpy(zero_copy_only=False) for n in ("a_H", "c0", "w_displayed"))
    gate = decisions.column("gate_pass").to_numpy(zero_copy_only=False)
    for x, y, z, g in zip(a_h, c0, w, gate, strict=True):
        k = stats.CELL_INDEX[stats.cell_key(x, y, z)]
        rows[k] += 1
        passes[k] += int(bool(g))
    return rows, passes


def finalize(
    run: common.RunHandle, args: argparse.Namespace, *, rehearsal: bool, pool: str, batch: Any, w5: W5Pin, units: Sequence[Unit], status: Mapping[str, str], book: AttemptBook, registered: Sequence[str],
    pool_md5: str, spec: ReadoutSpec, session: Mapping[str, Any],
) -> dict[str, Any]:
    """The tables, the quality gate and the summary of a run whose units are all complete or exhausted."""
    answer_ids = [u.unit_id for u in units if u.arm == "answer" and status[u.unit_id] == "complete"]
    ask_ids = [u.unit_id for u in units if u.arm == "ask" and status[u.unit_id] == "complete"]
    exhausted = [u.unit_id for u in units if status[u.unit_id] == "exhausted"]
    answer = decisions = None
    tables: dict[str, Any] = {}
    if answer_ids:
        answer = assemble_answer(run.parts.concat(answer_ids), args.model, w5.temperature)
        common.write_parquet(run.run_dir / ANSWER_TABLE, answer)
        tables[ANSWER_TABLE] = {"md5": file_md5(run.run_dir / ANSWER_TABLE), "n_rows": answer.num_rows}
    if ask_ids:
        decisions = assemble_decisions(run.parts.concat(ask_ids))
        common.write_parquet(run.run_dir / DECISION_TABLE, decisions)
        tables[DECISION_TABLE] = {"md5": file_md5(run.run_dir / DECISION_TABLE), "n_rows": decisions.num_rows}
    reasons: list[str] = []
    if exhausted:
        reasons.append(f"a unit failed twice and its cells are missing: {exhausted}")
    # exactly once: every registered item once in P-answer, every (item, cell) once in P-ask, for the arms that are complete
    complete_answer = not any(u.arm == "answer" and status[u.unit_id] != "complete" for u in units)
    complete_ask = not any(u.arm == "ask" and status[u.unit_id] != "complete" for u in units)
    exactly_once = {"answer": None, "ask": None}
    if answer is not None and complete_answer:
        exactly_once["answer"] = bool(sorted(answer.column("item_key").to_pylist()) == sorted(registered))
    if decisions is not None and complete_ask:
        keys = row_keys(decisions, "ask")
        exactly_once["ask"] = bool(len(keys) == len(registered) * len(stats.CELLS) and len(set(keys)) == len(keys) and {k[0] for k in keys} == set(registered))
    for arm, ok in exactly_once.items():
        if ok is False:
            raise cc.ConfirmError(f"the {arm} table does not hold every registered item exactly once")
    rows, passes = ask_counts(decisions)
    gate = stats.quality_gate_report(
        n_registered=len(registered), answer_rows=0 if answer is None else answer.num_rows, answer_pass=0 if answer is None else int(np.sum(answer.column("gate_pass").to_numpy(zero_copy_only=False))),
        ask_rows=rows, ask_pass=passes, threshold=spec.min_gate_pass_rate,
    )
    if not gate["passed"]:
        reasons.append(f"the quality gate is not passed: P-answer {gate['p_answer']['rate']:.4f}, lowest P-ask cell {gate['p_ask']['min_rate']:.4f} ({gate['p_ask']['min_cell']}), threshold {spec.min_gate_pass_rate}")
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA, "model": args.model, "stage": STAGE, "rehearsal": rehearsal, "pool": pool, "batch_id": None if batch is None else batch.batch_id, "split": SPLIT_LABEL,
        "n_items": len(registered), "pool_keys_md5": pool_md5, "paraphrase": PARAPHRASE, "framing": FRAMING, "layout": LAYOUT, "n_cells": len(stats.CELLS), "backend": args.backend,
        "temperature": {"value": w5.temperature, "bank": BANK, "w5_run_id": w5.run_id, "w5_summary_md5": w5.summary_md5, "w5_calib_md5": w5.calib_md5, "refitted": False},
        "units": {
            "planned": [u.unit_id for u in units], "complete": [u.unit_id for u in units if status[u.unit_id] == "complete"], "exhausted": exhausted,
            "attempts": {u.unit_id: book.started(u.unit_id) for u in units}, "max_attempts": MAX_ATTEMPTS,
        },
        "exactly_once": exactly_once, "quality_gate": gate, "testable": not reasons, "testable_reasons": reasons, "tables": tables, "session": dict(session),
        "failed_attempt_files": sorted(p.name for p in (run.run_dir / FAILED_DIR).glob("*")) if (run.run_dir / FAILED_DIR).is_dir() else [],
    }
    common.write_json(run.run_dir / SUMMARY_FILE, summary)
    return summary


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------

def execute(ctx: common.Context, args: argparse.Namespace, hooks: ConfirmHooks) -> dict[str, Any]:
    cfg = ctx.cfg
    rehearsal = check_arguments(args)
    entry = common.device_entry(cfg, args.model)
    spec = ReadoutSpec.from_config(cfg)
    chat_kwargs = common.chat_kwargs_of(entry)
    check_prompts(ctx)
    conds = psychometric_conditions(cfg)
    if not args.resume:
        used = common.tag_in_use(ctx.results_root, STAGE, ctx.tag)
        if used is not None:
            raise common.EntryError(f"the tag {ctx.tag!r} is in use by {used}")
    if args.backend == "vllm_offline":
        common.require_gpu_memory_fraction(args)
    models = common.check_model_dirs(cfg, [("device", args.model)])
    if not all(m["is_dir"] and m["has_config"] and m["has_tokenizer"] for m in models):
        raise common.EntryError(f"the model directory of {args.model!r} is incomplete: {models}")
    w5 = read_w5(args)
    batch = access = None
    pool = cc.REHEARSAL_POOLS[POOL] if rehearsal else POOL
    extra_md5: dict[str, str] = {"input.entry": file_md5(Path(__file__).resolve()), "input.run_decisions": file_md5(HERE / "run_decisions.py"), "input.c2_common": file_md5(HERE / "_c2_common.py"),
                                 "input.w5_summary": w5.summary_md5, "input.w5_calib": w5.calib_md5}
    if getattr(cc, "__file__", None):
        extra_md5["input.confirm_common"] = file_md5(Path(cc.__file__))
    extra_manifest: dict[str, Any] = {}
    if rehearsal:
        extra_manifest.update({"exploratory": True, "development_reason": REHEARSAL_REASON, "rehearsal": True})
    else:
        batch = load_batch(args)
        precheck_batch(ctx, args, batch)
        access = cc.open_confirm_access(ctx.results_root, batch, ctx.registry())  # exit 2 here while the pool is sealed or the batch is not registered: nothing is read yet
        block = dict(cc.confirm_manifest_block(batch, prereg_frozen_path=args.prereg_frozen))
        extra_md5.update(block.pop("input_md5", {}))  # the frozen preregistration replaces the default one under the key "prereg"; the batch manifest is bound beside it
        extra_manifest.update(block)
    registry = ctx.registry()
    registered = list(registry.keys(pool))
    pool_md5 = registry.pool_md5(pool)
    identity = run_identity(args, ctx, rehearsal=rehearsal, pool=pool, batch=batch, w5=w5, conds=conds, n_items=len(registered), pool_md5=pool_md5)
    check_resume_identity(args, identity)
    extra_manifest["c1confirm_run"] = identity
    opener = hooks.opener or (lambda c, kind, model, sp: common.open_backend(c, kind, model, sp))
    plan = common.numeric_plan(common.readout_backend_plan(args, unit={"chunk_size": args.chunk_size, "conditions_per_unit": args.conditions_per_unit}))
    with common.open_run(ctx, pools=[pool], parents=[w5.run_id], extra_md5=extra_md5, numeric=plan, extra_manifest=extra_manifest) as run:
        bindings = None
        if batch is not None:  # one run directory for every (batch, model, arm); another one is refused here, before the pool is read
            canonical = Path(run.run_dir).resolve()  # (a relative --resume names the same directory as the absolute path the first invocation recorded)
            bindings = {arm: cc.bind_run(ctx.results_root, batch, model=args.model, arm=arm, run_dir=canonical) for arm in ARMS}
        if access is not None:
            items = load_items_from_config(cfg, pool, registry=registry, results_root=access.results_root, holdout_access=access)
        else:
            items = common.load_pool_items(ctx, pool)
        if [i.item_key for i in items] != registered:
            raise cc.ConfirmError("the items loaded are not the frozen key list of the pool")
        units = plan_units(ctx, args, items, pool, conds, chat_kwargs)
        book = AttemptBook(run.run_dir / ATTEMPTS_FILE, model=args.model, results_root=ctx.results_root, bindings=bindings)
        status: dict[str, str] = {}
        bh: common.BackendHandle | None = None
        started = time.time()
        try:
            for unit in units:
                if run.parts.done(unit.unit_id):
                    status[unit.unit_id] = "complete"
                    continue
                if book.exhausted(unit):
                    status[unit.unit_id] = "exhausted"
                    continue
                if bh is None:  # the back end is opened when the first unit needs it
                    bh = opener(ctx, args.backend, args.model, spec)
                    run.confirm_numeric(args.backend, bh.numeric_actual())
                    if hooks.model_files and not (run.run_dir / "llm" / "model.json").is_file():
                        common.write_model_files(run, entry, bh)
                run.check_source()
                outcome = run_unit(run, bh, unit, book)
                status[unit.unit_id] = outcome
                if outcome in ("call_failure", "missing_rows"):
                    last = book.started(unit.unit_id) >= MAX_ATTEMPTS
                    raise UnitFailure(
                        f"unit {unit.unit_id} ({unit.arm}) failed its attempt {book.started(unit.unit_id)} of {MAX_ATTEMPTS} ({outcome}); the run stops so that the other units keep their attempts. "
                        f"Resume with --resume {run.run_dir}" + (" (this was the last attempt of the unit: the model's C1 will not be testable)" if last else "")
                    )
        finally:
            if bh is not None:
                bh.close()
        summary = finalize(
            run, args, rehearsal=rehearsal, pool=pool, batch=batch, w5=w5, units=units, status=status, book=book, registered=registered, pool_md5=pool_md5, spec=spec,
            session={"wall_s": time.time() - started, "units_run_by_the_last_session": sum(1 for u in units if status[u.unit_id] == "complete")},
        )
        return {
            "run_dir": str(run.run_dir), "model": args.model, "rehearsal": rehearsal, "pool": pool, "testable": summary["testable"], "testable_reasons": summary["testable_reasons"],
            "quality_gate_passed": summary["quality_gate"]["passed"], "exhausted_units": summary["units"]["exhausted"], "n_items": len(registered),
        }


def dry_run(ctx: common.Context, args: argparse.Namespace) -> int:
    dry = common.DryRun(ctx)
    try:
        rehearsal = check_arguments(args)
        check_prompts(ctx)
        conds = psychometric_conditions(ctx.cfg)
        w5 = read_w5(args)
        batch = None if rehearsal else load_batch(args)
        pool = cc.REHEARSAL_POOLS[POOL] if rehearsal else POOL
    except (common.EntryError, cc.ConfirmError) as error:
        dry.check("arguments and inputs", False, str(error))
        return dry.emit()
    common.dry_run_basics(dry, pools=[pool], models=[("device", args.model)], seed_keys=[])
    dry.check("frozen temperature: the calibration run and its md5 values", True, {"run_id": w5.run_id, "temperature": w5.temperature, "calib_summary_md5": w5.summary_md5, "calib_parquet_md5": w5.calib_md5})
    dry.check("prompts: the assistant prefix of the calibration run and the paraphrase a00", True, {"assistant_prefix": ctx.book.assistant_prefix, "paraphrase": PARAPHRASE, "md5": ctx.book.md5})
    if batch is not None:
        try:
            info = precheck_batch(ctx, args, batch)
            dry.check("batch: pool, model, arms, registered identity, frozen preregistration, bound files", True, {"batch_id": batch.batch_id, "pool": batch.pool, "arms": list(ARMS), **info})
        except (cc.ConfirmError, common.EntryError) as error:
            dry.check("batch: pool, model, arms, registered identity, frozen preregistration, bound files", False, str(error))
        dry.note("unseal_marker_exists", paths.holdout_marker_path(ctx.results_root).is_file())
    sizes = common.pool_sizes(ctx, [pool])
    n_items = sizes[pool]
    n_answer, n_ask = len(chunks(n_items, args.chunk_size)), len(chunks(len(conds), args.conditions_per_unit))
    dry.note("workload", {
        "pool": pool, "rehearsal": rehearsal, "n_items": n_items, "p_answer_prompts": n_items, "p_ask_cells": len(conds), "p_ask_prompts": n_items * len(conds),
        "units": {"answer": n_answer, "ask": n_ask}, "max_attempts_per_unit": MAX_ATTEMPTS, "backend": args.backend, "stage": STAGE,
    })
    if getattr(args, "resume", None):
        dry.check("resume directory", (Path(args.resume) / common.MANIFEST_FILE).is_file(), args.resume)
    return dry.emit()


def main(argv: Sequence[str] | None = None, hooks: ConfirmHooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or ConfirmHooks()
    args = make_parser().parse_args(argv)
    t0 = time.time()
    try:
        ctx = common.build_context(args, script=SCRIPT, stage=STAGE, argv=argv)
        ctx = dataclasses.replace(ctx, results_root=ctx.results_root.resolve())  # one canonical results root for the registry, the ledger, the loader, the access and the run directory
        if args.dry_run:
            return dry_run(ctx, args)
        outcome = execute(ctx, args, hooks)
    except UnitFailure as failure:
        print(f"stopped: {failure}", file=sys.stderr)
        return EXIT_UNIT_FAILED
    except (common.EntryError, cc.ConfirmError, SplitError, paths.PathPolicyError, HoldoutSealedError, HoldoutLedgerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except common.SourceChangedError as error:
        return common.stopped_by_source_change(error)
    outcome["wall_s"] = time.time() - t0
    print(json.dumps(common.json_safe(outcome), indent=2, sort_keys=True))
    return EXIT_NOT_TESTABLE if outcome["exhausted_units"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
