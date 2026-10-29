#!/usr/bin/env python
"""Confirmatory run of the mechanism group M of one end-side model on the MMLU-Redux HOLDOUT pool.

Usage: python scripts/run_mechanism_confirm.py --model MODEL --tag TAG --c1-run RUN_DIR --batch-manifest FILE [--backend B] [--resume RUN_DIR | --rehearsal | --dry-run]

What is read: the 43 conditions of the group (the seven variants with a level, M1, M1w, M2, M3, M0, F3 and OBJ, at the six levels of ``configs/prompts/mech_v1.yaml``, and M0n,
which has none) for every item of the holdout pool, one prompt per (item, condition), through the readout of ``scripts/run_mechanism.py`` (the same back end, chat template, single-token
digit readout, digit-mass gate and conditions). This file imports the side-effect-free pieces of that script (the prompt book, the condition list, the job builder, the summary) and does
not change it. The run itself uses no confidence: z is taken at analysis time from the P-answer of the C1 confirmation run of the same model
(``--c1-run``), which this run is bound to. A C1 run in which every P-answer unit failed twice has no P-answer table (``scripts/run_c1_confirm.py``
writes a table only for an arm with a complete unit); that is a legal end of a C1 run, and the run is bound to it as one WITHOUT a P-answer (``p_answer_available`` is false in the
record): the group does not need z to be read, and the claim that is judged against z is then not testable at analysis time. Any other missing table is refused.

Differences from the development entry:

* the pool is the holdout pool, loaded WITH a ``HoldoutAccess`` of the registered batch (``--batch-manifest``; the access carries all the models and arms of the batch, because the ledger keeps
  the access of the first read only). Every check runs before the pool is loaded and nothing is consumed when one fails (exit 2): the label, the model directory, the memory fraction, the prompt
  files, the md5 values the batch manifest binds, the identity of the batch against the ledger, the frozen pre-registration, the marker and the C1 run. The run is
  bound to its (batch, model, arm) in ``confirm_runs.jsonl`` before the pool is read: a second label for the same model cannot start the batch over;
* the manifest carries no exploratory mark; ``input_md5.prereg`` is the frozen pre-registration file and the batch manifest is bound as well (``confirm_manifest_block``); the stage is ``mechconfirm``;
* failure rules. A unit (a chunk of six conditions) has at most two attempts, the original and one resume. Only two failures are retried: a CALL FAILURE (the back end raises, or
  returns fewer readouts than prompts) and MISSING ROWS (the finished unit does not hold exactly its (item, condition) cells). A readout whose digit mass is below the gate is a valid readout:
  it is neither retried nor dropped. The first attempt that completes is the result of the unit; a failed attempt leaves its partial output under ``failed_attempts/`` and never enters the table. The
  attempt counts live in ``confirm_runs.jsonl`` (across processes and labels) and nowhere else: the number of attempts, whether a unit is used up and the kind of its last failure are read from
  there, and ``attempts.jsonl`` in the run directory is the shared layer's copy of those records (only the shared layer writes it). What this entry knows of an attempt beyond them (the error, the
  rows, the wall time) is written to ``mech_attempt_details.jsonl``, which is never counted. A rehearsal has no registry: it counts in ``attempts.jsonl`` alone. A unit that fails twice leaves the
  group untestable for the model (the summary says so, the exit code is 1). ``--resume`` continues the bound run directory only, and the unit-level skip checks the finished parts cell by cell;
* output file names carry none of the words in ``cc.FORBIDDEN_NAME_PARTS`` (the unit names do not hold the pool name).

``--rehearsal`` runs the same code on the calibration pool (``REHEARSAL_POOLS``), stage ``mechconfirm_rehearsal``, with ``exploratory: true`` in the manifest and the summary; it writes no ledger,
makes no binding and needs no registered batch (a batch manifest, if given, is used for the md5, pool, model and arm checks only). ``--dry-run`` checks everything that does not need the loader
and loads nothing. Exit codes: 0 the group is complete; 1 an attempt at a unit failed (``--resume`` continues while an attempt is left) or the group ended untestable; 2 refused before anything was
read (nothing consumed); 4 the source of ``nmi/`` changed during the run.

Output: ``mechanism.parquet`` (the columns of ``run_mechanism.MECH_COLUMNS``) and ``mechanism_summary.json`` (the digit-mass gate pass rate, the share of digit free tokens and the ask rate per
variant, level, and variant and level; the unit table with the attempts; the binding to the batch and the C1 run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (REPO_ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _c2_common as common  # noqa: E402
import _confirm_common as cc  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import run_mechanism as dev  # noqa: E402

from nmi import mech_prompts, paths  # noqa: E402
from nmi.data.holdout import HoldoutLedgerError, HoldoutSealedError  # noqa: E402
from nmi.data.keys import keylist_digests  # noqa: E402
from nmi.data.registry import SplitError, load_items_from_config  # noqa: E402
from nmi.provenance import file_md5, iso, note_skipped, now_local  # noqa: E402
from nmi.readout.digits import ReadoutSpec  # noqa: E402

SCRIPT = "run_mechanism_confirm"
STAGE = "mechconfirm"
REHEARSAL_STAGE = "mechconfirm_rehearsal"
HOLDOUT_POOL = "mmlu_redux_holdout"
REHEARSAL_POOL = cc.REHEARSAL_POOLS[HOLDOUT_POOL]
ARM = "mech"
C1_STAGE_PREFIX = "c1confirm"
C1_TABLE = "answer.parquet"
C1_SUMMARY = "c1_confirm_summary.json"
C1_ANSWER_UNITS = "answer__"  # the ids of the P-answer units of a C1 run (``run_c1_confirm.unit_plan``)
C1_COLUMNS = ("item_key", "model", "pool")
CONDITIONS_PER_UNIT = 6  # as the development runs: the unit is also the batch composition of the engine, and the unit of the retry rule
MAX_ATTEMPTS = 2  # the original attempt and one resume
N_CONDITIONS = 43
UNIT_PREFIX = "confirm"  # unit names must not hold the pool name: the pool of this run is a holdout pool and no file name may carry that word
TABLE_FILE = dev.TABLE_FILE
SUMMARY_FILE = dev.SUMMARY_FILE
ATTEMPTS_FILE = "attempts.jsonl"  # a confirmation run: the shared layer's copy of the registry records (only ``cc.record_attempt`` writes it); a rehearsal: the record the rehearsal counts in
DETAILS_FILE = "mech_attempt_details.jsonl"  # a confirmation run only: what this entry knows of an attempt beyond the registry (error, rows, wall time); never counted
ATTEMPT_FAILURES = ("call_failure", "missing_rows")  # the outcomes that close an attempt as failed (``cc.OUTCOMES`` without ``started`` and ``complete``)
FAILED_DIR = "failed_attempts"
DEFAULT_PREREG_FROZEN = cc.FROZEN_PREREG_PATH
REHEARSAL_REASON = (
    "rehearsal of the confirmation entry scripts/run_mechanism_confirm.py on the calibration pool: the same code as the "
    "confirmation run, no ledger, no binding; it is not evidence"
)


class UnitFailed(RuntimeError):
    """An attempt at a unit failed (a call failure or missing rows). The attempt has been recorded; ``--resume`` continues while an attempt is left."""

    def __init__(self, unit_id: str, attempt: int, outcome: str, reason: str) -> None:
        super().__init__(f"unit {unit_id}: attempt {attempt} failed ({outcome}): {reason}")
        self.unit_id, self.attempt, self.outcome, self.reason = unit_id, attempt, outcome, reason


class AttemptsExhausted(RuntimeError):
    """Both attempts at a unit are used and it has no finished part."""


@dataclass
class ConfirmHooks:
    opener: Callable[[common.Context, str, str, ReadoutSpec], common.BackendHandle] | None = None
    model_files: bool = True


@dataclass(frozen=True)
class Unit:
    """A unit of work: a chunk of conditions that one back end call reads together, and the unit of the retry rule."""

    unit_id: str
    conds: tuple[mech_prompts.Condition, ...]

    def labels(self) -> list[str]:
        return [c.label() for c in self.conds]


@dataclass(frozen=True)
class C1Input:
    """The C1 confirmation run of the same model that this run is bound to: its run id and the md5 of its P-answer table and of the set of its item keys."""

    run_dir: str
    run_id: str
    stage: str
    answer_md5: str | None  # None: the C1 run has no P-answer table (every P-answer unit failed twice, ``p_answer_legally_missing``)
    item_keys_md5: str | None
    n_items: int
    complete: bool  # the table holds exactly the frozen item set, once each (the analysis decides about z; the binding only needs the run to be finished)
    summary_md5: str | None = None  # only when the table is absent: the md5 of the C1 summary that shows why

    @property
    def p_answer_available(self) -> bool:
        return self.answer_md5 is not None

    def record(self) -> dict[str, Any]:
        out = {"run_id": self.run_id, "answer_parquet_md5": self.answer_md5, "item_keys_md5": self.item_keys_md5, "n_items": self.n_items, "complete_item_set": self.complete}
        if not self.p_answer_available:  # (the key exists only for a run without a P-answer)
            out["p_answer_available"] = False
        return out


@dataclass
class Plan:
    """Everything a run needs, settled before the first item is read."""

    rehearsal: bool
    stage: str
    pool: str
    model: str
    mbook: mech_prompts.MechPromptBook
    conds: list[mech_prompts.Condition]
    units: list[Unit]
    frozen_keys: tuple[str, ...]
    registry: Any
    c1: C1Input
    batch: Any  # cc.BatchManifest or None (rehearsal without a batch manifest)
    access: Any  # HoldoutAccess or None
    bound_files: dict[str, str]
    identity: dict[str, Any]
    extra_md5: dict[str, str]
    extra_manifest: dict[str, Any]
    chat_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Confirmatory run of the mechanism group M: the 43 conditions of one end-side model on the MMLU-Redux holdout pool, "
        "in the registered batch, bound to the C1 confirmation run of the same model."
    )
    common.add_common_arguments(p)
    common.add_readout_arguments(p)
    p.add_argument("--c1-run", required=True, help="the finished C1 confirmation run of the same model (stage c1confirm): its P-answer table gives z at analysis time; its run id and key set are bound here")
    p.add_argument("--batch-manifest", default=None, help="the frozen batch manifest (batch id, pool, models, arms, md5 of the bound files); required unless --rehearsal")
    p.add_argument("--prereg-frozen", default=str(DEFAULT_PREREG_FROZEN), help="the frozen pre-registration file (default: DEFAULT_PREREG_FROZEN); its md5 must be the one the batch manifest names")
    p.add_argument("--rehearsal", action="store_true", help=f"run the same code on the calibration pool ({REHEARSAL_POOL}), exploratory, with no ledger and no binding")
    return p


def refuse_unusable_arguments(args: argparse.Namespace) -> None:
    if args.limit_items is not None or args.pilot_rep is not None:
        raise common.EntryError("--limit-items and --pilot-rep are refused: the group reads every item of the pool, a subset would leave cells that no analysis may drop")
    if args.backend == "vllm_http" and not args.http_url:
        raise common.EntryError("--backend vllm_http needs --http-url")


def pool_of(args: argparse.Namespace) -> str:
    return REHEARSAL_POOL if args.rehearsal else HOLDOUT_POOL


def stage_of(args: argparse.Namespace) -> str:
    return REHEARSAL_STAGE if args.rehearsal else STAGE


def normalise_results_root(ctx: common.Context) -> None:
    """One canonical results root for the registration, the loader, the access, the binding and the resume."""
    ctx.results_root = Path(os.path.realpath(ctx.results_root))


def real_path(path: str | Path) -> Path:
    return Path(os.path.realpath(path))


# ---------------------------------------------------------------------------------------
# conditions and units
# ---------------------------------------------------------------------------------------

def build_units(mbook: mech_prompts.MechPromptBook, conds: Sequence[mech_prompts.Condition]) -> list[Unit]:
    units = []
    for gi, rows in enumerate(dev.chunks(len(conds), CONDITIONS_PER_UNIT)):
        units.append(Unit(f"{UNIT_PREFIX}__{mbook.version}__g{gi:02d}", tuple(conds[i] for i in rows)))
    return units


def cell_key(variant: str, x: float | None, item_key: str) -> tuple[str, float | None, str]:
    return (variant, None if x is None or x != x else round(float(x), 9), item_key)


def expected_cells(unit: Unit, keys: Sequence[str]) -> set[tuple[str, float | None, str]]:
    return {cell_key(c.variant, c.x, k) for c in unit.conds for k in keys}


def unit_row_problems(table: pa.Table, unit: Unit, keys: Sequence[str]) -> list[str]:
    """What is wrong with the rows of a finished unit, compared with the (item, condition) cells it should hold; empty when it holds exactly those, once each."""
    required = {"variant", "x", "item_key"}
    if not required <= set(table.column_names):
        return [f"the table has no columns {sorted(required - set(table.column_names))}"]
    cells = [cell_key(v, x, k) for v, x, k in zip(table.column("variant").to_pylist(), table.column("x").to_pylist(), table.column("item_key").to_pylist(), strict=True)]
    expected = expected_cells(unit, keys)
    problems: list[str] = []
    if len(cells) != len(expected):
        problems.append(f"{len(cells)} rows, the unit has {len(expected)} cells")
    if len(set(cells)) != len(cells):
        problems.append(f"{len(cells) - len(set(cells))} repeated cells")
    missing, extra = expected - set(cells), set(cells) - expected
    if missing:
        problems.append(f"{len(missing)} cells are missing")
    if extra:
        problems.append(f"{len(extra)} cells are not cells of the unit")
    return problems


# ---------------------------------------------------------------------------------------
# the pre-checks (none of them reads the pool)
# ---------------------------------------------------------------------------------------

def rel_key(path: Path) -> str:
    """The key under which the batch manifest binds a file: its path relative to the repository, or the absolute path of a file outside it."""
    resolved = real_path(path)
    try:
        return resolved.relative_to(real_path(REPO_ROOT)).as_posix()
    except ValueError:
        return resolved.as_posix()


def bound_file_paths(ctx: common.Context) -> dict[str, Path]:
    """The files whose md5 the batch manifest must bind for this entry: itself and the code it runs, the prompt files and the configuration."""
    files = [
        Path(__file__).resolve(), Path(dev.__file__).resolve(), Path(common.__file__).resolve(), Path(cc.__file__).resolve(), Path(mech_prompts.__file__).resolve(),
        Path(dev.MECH_PROMPTS).resolve(), Path(ctx.cfg.path).resolve(), Path(ctx.prompts_path).resolve(),
    ]
    return {rel_key(p): p for p in files}


def check_bound_files(batch: Any, ctx: common.Context) -> dict[str, str]:
    """Every file in ``bound_file_paths`` is listed in the batch manifest with the md5 it has now."""
    bound = {key: file_md5(path) for key, path in bound_file_paths(ctx).items()}
    problems = []
    for key, md5 in sorted(bound.items()):
        listed = batch.files_md5.get(key)
        if listed is None:
            problems.append(f"{key}: the batch manifest binds no md5")
        elif listed != md5:
            problems.append(f"{key}: md5 {md5}, the batch manifest binds {listed}")
    if problems:
        raise cc.ConfirmError("the files of this run are not those the batch manifest binds: " + "; ".join(problems))
    return bound


def load_batch(args: argparse.Namespace, model: str) -> Any:
    """The batch manifest, with the checks that need no ledger: its pool, and the model and the arm of this run being in it."""
    if not args.batch_manifest:
        if args.rehearsal:
            return None
        raise cc.ConfirmError("--batch-manifest is required for a confirmation run (the registered batch the holdout pool is opened for)")
    path = Path(args.batch_manifest)
    if not path.is_file():
        raise cc.ConfirmError(f"--batch-manifest {path}: not a file")
    batch = cc.load_batch_manifest(path)
    if batch.pool != HOLDOUT_POOL:
        raise cc.ConfirmError(f"batch {batch.batch_id!r} is a batch of the pool {batch.pool!r}; this entry runs the mechanism group on {HOLDOUT_POOL!r}")
    if model not in batch.models:
        raise cc.ConfirmError(f"model {model!r} is not in the batch manifest of {batch.batch_id!r} (models: {list(batch.models)})")
    if ARM not in batch.arms:
        raise cc.ConfirmError(f"arm {ARM!r} is not in the batch manifest of {batch.batch_id!r} (arms: {list(batch.arms)})")
    return batch


def check_prereg(args: argparse.Namespace, batch: Any, rehearsal: bool) -> Path | None:
    path = Path(args.prereg_frozen)
    if not path.is_file():  # the document is not part of this release: one line, and the check against the batch manifest is skipped
        note_skipped(path, "frozen pre-registration")
        return None if rehearsal else path
    if batch is not None and file_md5(path) != batch.prereg_md5:
        raise cc.ConfirmError(f"--prereg-frozen {path} has md5 {file_md5(path)}, the batch manifest names {batch.prereg_md5}")
    return path


def split_block(block: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """The manifest block of the shared layer as (extra input md5 values, the rest). It carries no exploratory mark."""
    rest = dict(block)
    md5 = dict(rest.pop("input_md5", {}) or {})
    if "exploratory" in rest or "development_reason" in rest:
        raise cc.ConfirmError("the manifest block of a confirmation run must carry no exploratory mark")
    return {str(k): str(v) for k, v in md5.items()}, rest


def p_answer_legally_missing(run_dir: str | Path, manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """Is the P-answer table of a finished C1 run absent for the one reason that the end of a C1 run allows? ``scripts/run_c1_confirm.py`` (``finalize``) writes ``answer.parquet`` only when at least one
    P-answer unit is complete, so a run in which every P-answer unit failed twice ends without it (exit code 1, ``exit_status`` ok). All of these must hold: the manifest says ``exit_status`` ok; the summary
    lists P-answer units (``units.planned``, ids ``answer__...``); none of them is complete and every one is in ``units.exhausted``; the summary's ``tables`` does not list ``answer.parquet``; and the file
    does not exist."""
    if manifest.get("exit_status") != "ok":
        return False
    units = summary.get("units")
    if not isinstance(units, Mapping):
        return False
    planned, complete, exhausted = units.get("planned"), units.get("complete"), units.get("exhausted")
    if not all(isinstance(v, list) for v in (planned, complete, exhausted)):
        return False
    mine = [u for u in planned if isinstance(u, str) and u.startswith(C1_ANSWER_UNITS)]
    if not mine or any(isinstance(u, str) and u.startswith(C1_ANSWER_UNITS) for u in complete) or any(u not in exhausted for u in mine):
        return False
    return C1_TABLE not in (summary.get("tables") or {}) and not (Path(run_dir) / C1_TABLE).exists()


def read_c1(
    c1_dir: str | Path, *, model: str, pool: str, frozen_keys: Sequence[str], results_root: Path, rehearsal: bool, block: Mapping[str, Any] | None,
) -> C1Input:
    """The C1 run of the same model, finished, in the same results root, and in real mode bound to the same batch (it carries the same manifest block as this run will). A run that has no P-answer table
    because every P-answer unit failed twice (``p_answer_legally_missing``) is accepted as a run WITHOUT a P-answer: the checks that need the table are made on the summary of the run instead (its model
    and pool), the md5 values of the table and of its key set are None, and the item count is 0."""
    run_dir = Path(c1_dir)
    if not run_dir.is_dir():
        raise cc.ConfirmError(f"--c1-run {run_dir}: not a directory (the C1 run of {model} must be finished before this run can be bound to it)")
    real = real_path(run_dir)
    try:
        real.relative_to(results_root)
    except ValueError:
        raise cc.ConfirmError(f"--c1-run {run_dir} is not under the results root {results_root}") from None
    manifest_path = run_dir / common.MANIFEST_FILE
    try:
        manifest = common.read_json(manifest_path)
    except (OSError, ValueError) as error:
        raise cc.ConfirmError(f"--c1-run {run_dir}: no readable {common.MANIFEST_FILE} ({error})") from error
    run_id = manifest.get("run_id") if isinstance(manifest, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise cc.ConfirmError(f"--c1-run {run_dir}: its manifest names no run_id")
    stage = str(manifest.get("stage"))
    if not stage.startswith(C1_STAGE_PREFIX):
        raise cc.ConfirmError(f"--c1-run {run_dir} is a run of stage {stage!r}, not of the C1 confirmation ({C1_STAGE_PREFIX})")
    if manifest.get("exit_status") != "ok":
        raise cc.ConfirmError(f"--c1-run {run_dir} is not finished (exit_status {manifest.get('exit_status')!r}): the mechanism run binds a completed C1 run")
    if not rehearsal and manifest.get("exploratory"):
        raise cc.ConfirmError(f"--c1-run {run_dir} is marked exploratory (a rehearsal): a confirmation run binds a confirmation run")
    if not rehearsal and block is not None:
        for key, value in block.items():
            if key == "input_md5":
                recorded = manifest.get("input_md5", {})
                for name, md5 in value.items():
                    if recorded.get(name) != md5:
                        raise cc.ConfirmError(f"--c1-run {run_dir} is not bound to this batch: its input_md5[{name!r}] is {recorded.get(name)!r}, this run binds {md5!r}")
            elif manifest.get(key) != value:
                raise cc.ConfirmError(f"--c1-run {run_dir} is not bound to this batch: its manifest key {key!r} differs from the batch block of this run")
    table_path = run_dir / C1_TABLE
    if not table_path.is_file():
        summary_path = run_dir / C1_SUMMARY
        try:
            summary = common.read_json(summary_path)
        except (OSError, ValueError):
            summary = None
        if not isinstance(summary, dict) or not p_answer_legally_missing(run_dir, manifest, summary):
            raise cc.ConfirmError(f"--c1-run {run_dir}: {C1_TABLE} is missing")
        if summary.get("model") != model:
            raise cc.ConfirmError(f"--c1-run {run_dir}: its summary is of the model {summary.get('model')!r}, this run is of {model!r}")
        if summary.get("pool") != pool:
            raise cc.ConfirmError(f"--c1-run {run_dir}: its summary is of the pool {summary.get('pool')!r}, this run reads {pool!r}")
        return C1Input(run_dir=str(real), run_id=run_id, stage=stage, answer_md5=None, item_keys_md5=None, n_items=0, complete=False, summary_md5=file_md5(summary_path))
    names = set(pq.read_schema(table_path).names)
    if not set(C1_COLUMNS) <= names:
        raise cc.ConfirmError(f"--c1-run {run_dir}: {C1_TABLE} has no columns {sorted(set(C1_COLUMNS) - names)}")
    table = pq.read_table(table_path, columns=list(C1_COLUMNS))
    models, pools = set(table.column("model").to_pylist()), set(table.column("pool").to_pylist())
    if models != {model}:
        raise cc.ConfirmError(f"--c1-run {run_dir}: {C1_TABLE} holds the models {sorted(models)}, this run is of {model!r}")
    if pools != {pool}:
        raise cc.ConfirmError(f"--c1-run {run_dir}: {C1_TABLE} holds the pools {sorted(pools)}, this run reads {pool!r}")
    keys = table.column("item_key").to_pylist()
    if len(set(keys)) != len(keys):
        raise cc.ConfirmError(f"--c1-run {run_dir}: {C1_TABLE} holds {len(keys) - len(set(keys))} repeated items (a confirmation table has each item once)")
    outside = set(keys) - set(frozen_keys)
    if outside:
        raise cc.ConfirmError(f"--c1-run {run_dir}: {len(outside)} items of {C1_TABLE} are not in the frozen key list of {pool!r}")
    return C1Input(
        run_dir=str(real), run_id=run_id, stage=stage, answer_md5=file_md5(table_path), item_keys_md5=keylist_digests(sorted(keys))[0], n_items=len(keys),
        complete=set(keys) == set(frozen_keys),
    )


def mechanism_identity(
    args: argparse.Namespace, mbook: mech_prompts.MechPromptBook, conds: Sequence[mech_prompts.Condition], pool: str, batch: Any, c1: C1Input, bound: Mapping[str, str],
) -> dict[str, Any]:
    """What a run is beyond what the md5 values of the manifest hold; the manifest keeps it (``mechconfirm_run``) and a resumed run must have the same."""
    return {
        "model": args.model, "pool": pool, "arm": ARM, "rehearsal": bool(args.rehearsal), "layout": dev.LAYOUT, "framing": dev.FRAMING, "variants": list(mbook.variants),
        "levels": list(mbook.levels), "a_H": mbook.a_H, "c0": mbook.c0, "delay_rule": mech_prompts.DELAY_RULE, "delay_seconds": {f"{x:.2f}": mbook.delay_seconds(x) for x in mbook.levels},
        "conditions": [dev.condition_record(mbook, c) for c in conds], "conditions_per_unit": CONDITIONS_PER_UNIT,
        "mech_prompts": {"version": mbook.version, "md5": mbook.md5, "base_prompts_md5": mbook.base.md5},
        "batch": None if batch is None else {"batch_id": batch.batch_id, "manifest_md5": batch.md5, "prereg_md5": batch.prereg_md5, "pool": batch.pool},
        "c1": c1.record(), "bound_files": dict(sorted(bound.items())),
    }


def check_resume_identity(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    """The identity of the call is compared with that of the run being resumed BEFORE any finished unit is reused, before the manifest of that run is touched and before the pool is read."""
    resume = getattr(args, "resume", None)
    if not resume:
        return
    path = Path(resume) / common.MANIFEST_FILE
    if not path.is_file():
        return  # open_run says so itself
    manifest = common.read_json(path)
    recorded = manifest.get("mechconfirm_run") if isinstance(manifest, dict) else None
    if not isinstance(recorded, dict):
        raise common.EntryError(f"--resume {resume}: that is not a run of {SCRIPT} (its manifest has no mechconfirm_run, the record of what it was started with)")
    differences = dev.identity_differences(recorded, identity)
    if differences:
        raise common.EntryError(
            f"--resume {resume}: the model, batch, C1 run, bound files, prompt files and conditions of the run must be those it was started with (a finished unit of another setting would be "
            "reused as if it were this run's): " + "; ".join(differences)
        )


def prepare(ctx: common.Context, args: argparse.Namespace) -> Plan:
    """Every check of the run, in a fixed order; nothing is loaded, consumed, bound or created. Raises ``EntryError`` or ``ConfirmError`` (exit 2)."""
    refuse_unusable_arguments(args)
    rehearsal = bool(args.rehearsal)
    pool = pool_of(args)
    # 1. the label
    if not args.resume:
        if not args.tag:
            raise common.EntryError("--tag is required for a real run")
        paths.validate_name("tag", args.tag)
        used = common.tag_in_use(ctx.results_root, ctx.stage, args.tag)
        if used:
            raise common.EntryError(f"tag {args.tag!r} is already used in stage {ctx.stage!r} ({used}); a confirmation run does not start again under another label of the same name")
    # 2. the model, its directory and the memory fraction
    entry = common.device_entry(ctx.cfg, args.model)
    chat_kwargs = common.chat_kwargs_of(entry)
    report = common.check_model_dirs(ctx.cfg, [("device", args.model)])
    if not all(m["is_dir"] and m["has_config"] and m["has_tokenizer"] for m in report):
        raise common.EntryError(f"the model directory of {args.model!r} is not usable: {report}")
    if args.backend == "vllm_offline":
        common.require_gpu_memory_fraction(args)
    # 3. the prompt files, the conditions and the pinned files
    mbook = dev.load_mech_prompts(ctx)
    conds = dev.condition_list(mbook, list(mbook.variants))
    if len(conds) != N_CONDITIONS:
        raise common.EntryError(f"the group has {len(conds)} conditions, the pre-registration has {N_CONDITIONS}")
    units = build_units(mbook, conds)
    # 4. the split, the pool and the frozen key list
    registry = ctx.registry()
    frozen_keys = tuple(registry.keys(pool))
    # 5. the batch: its manifest, the md5 values it binds, the pre-registration, and (real mode) its identity in the ledger and the unseal marker
    batch = load_batch(args, args.model)
    bound = check_bound_files(batch, ctx) if batch is not None else {key: file_md5(path) for key, path in bound_file_paths(ctx).items()}
    prereg = check_prereg(args, batch, rehearsal)
    access, block = None, None
    if not rehearsal:
        cc.check_batch_identity(ctx.results_root, batch, model=args.model, arm=ARM)
        access = cc.open_confirm_access(ctx.results_root, batch, registry)
        block = cc.confirm_manifest_block(batch, prereg_frozen_path=prereg)
    # 6. the C1 run of the same model
    c1 = read_c1(args.c1_run, model=args.model, pool=pool, frozen_keys=frozen_keys, results_root=ctx.results_root, rehearsal=rehearsal, block=block)
    identity = mechanism_identity(args, mbook, conds, pool, batch, c1, bound)
    check_resume_identity(args, identity)
    extra_md5 = {"input.mech_prompts": mbook.md5, "input.entry": file_md5(Path(__file__).resolve()), "input.dev_entry": file_md5(Path(dev.__file__).resolve())}
    if c1.p_answer_available:
        extra_md5.update({"input.c1_answer": c1.answer_md5, "input.c1_item_keys": c1.item_keys_md5})
    else:  # a C1 run without a P-answer table: the summary that shows why is bound in place of the table
        extra_md5["input.c1_summary"] = c1.summary_md5
    extra_manifest: dict[str, Any] = {"mechconfirm_run": identity}  # (the identity holds the arm, the batch and the rehearsal flag; the keys of the shared layer's block are not overwritten)
    if batch is not None:
        extra_md5["input.batch_manifest"] = batch.md5
    if rehearsal:
        extra_manifest.update(exploratory=True, development_reason=REHEARSAL_REASON)
        if prereg is not None:
            extra_md5["input.prereg_frozen"] = file_md5(prereg)
    else:
        block_md5, block_rest = split_block(block)
        extra_md5.update(block_md5)
        extra_manifest.update(block_rest)
    return Plan(
        rehearsal=rehearsal, stage=ctx.stage, pool=pool, model=args.model, mbook=mbook, conds=conds, units=units, frozen_keys=frozen_keys, registry=registry, c1=c1, batch=batch, access=access,
        bound_files=bound, identity=identity, extra_md5=extra_md5, extra_manifest=extra_manifest, chat_kwargs=chat_kwargs,
    )


# ---------------------------------------------------------------------------------------
# attempts
# ---------------------------------------------------------------------------------------

class AttemptTracker:
    """The attempts at the units of one run.

    A confirmation run (``binding`` is set) counts them in the shared registry and nowhere else. ``cc.attempts`` (``confirm_runs.jsonl`` of the results root, across processes and labels) gives the
    number of attempts started, whether a unit is used up and the kind of its last failure, and ``cc.record_attempt`` is the only writer of ``attempts.jsonl`` in the run directory: it copies each
    registry record there (the records carry ``unit_id``). What this entry knows of an attempt beyond the registry (the error, the rows, the wall time) goes to ``mech_attempt_details.jsonl``, a file that
    is never read for a count. A rehearsal has no registry: it counts in ``attempts.jsonl`` alone, in records keyed by ``unit``."""

    def __init__(self, run_dir: Path, results_root: Path, binding: Any) -> None:
        self.path = run_dir / ATTEMPTS_FILE
        self.details_path = run_dir / DETAILS_FILE
        self.results_root = results_root
        self.binding = binding

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _write(path: Path, record: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(common.json_safe(record), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def records(self, unit_id: str) -> list[dict[str, Any]]:
        """The attempt records of one unit in the order written: those of the registry in a confirmation run, those of the run directory in a rehearsal."""
        if self.binding is not None:
            return cc.attempts(self.results_root, self.binding, unit_id)
        return [r for r in self._read(self.path) if r["unit"] == unit_id]

    def count(self, unit_id: str) -> int:
        return sum(1 for r in self.records(unit_id) if r["outcome"] == "started")

    def detail_of(self, unit_id: str, attempt: int, outcome: str) -> Any:
        """What the entry noted when it closed the attempt (the last note for that attempt and outcome), or None."""
        notes = []
        for line in (self.details_path.read_text(encoding="utf-8").splitlines() if self.details_path.is_file() else []):
            try:  # (the notes are informative: a line that a kill cut short is skipped, it never decides a count)
                note = json.loads(line)
            except ValueError:
                continue
            if isinstance(note, dict) and (note.get("unit_id"), note.get("attempt"), note.get("outcome")) == (unit_id, attempt, outcome):
                notes.append(note)
        return notes[-1].get("detail") if notes else None

    def last_failure(self, unit_id: str) -> dict[str, Any] | None:
        failures = [r for r in self.records(unit_id) if r["outcome"] in ATTEMPT_FAILURES]
        if not failures:
            return None
        last = failures[-1]
        if self.binding is None:
            return last
        return {"unit": unit_id, "attempt": last["attempt"], "outcome": last["outcome"], "time": last.get("utc"), "detail": self.detail_of(unit_id, last["attempt"], last["outcome"])}

    def start(self, unit_id: str) -> int:
        """Record that an attempt at the unit starts and return its number. Raises ``AttemptsExhausted`` when both attempts are used (in a confirmation run: when the registry says so), and the registry's
        ``ConfirmError`` when it disagrees for another reason."""
        if self.binding is None:
            done = self.count(unit_id)
            if done >= MAX_ATTEMPTS:
                raise AttemptsExhausted(unit_id)
            number = done + 1
            self._write(self.path, {"unit": unit_id, "attempt": number, "outcome": "started", "time": iso(now_local())})
            return number
        try:
            return int(cc.record_attempt(self.results_root, self.binding, unit_id, "started"))
        except cc.ConfirmError as error:
            rows = self.records(unit_id)  # the registry decides: two attempts started and no complete one (another process, or an earlier invocation, used them)
            if sum(1 for r in rows if r["outcome"] == "started") >= MAX_ATTEMPTS and not any(r["outcome"] == "complete" for r in rows):
                raise AttemptsExhausted(unit_id) from error
            raise

    def finish(self, unit_id: str, outcome: str, **detail: Any) -> int:
        """Close the open attempt at the unit with ``outcome`` and return its number. A confirmation run records it in the registry (whose copy in the run directory is the shared layer's), then notes the
        detail in ``mech_attempt_details.jsonl``; a rehearsal writes one record to ``attempts.jsonl``."""
        if self.binding is None:
            number = self.count(unit_id)
            self._write(self.path, {"unit": unit_id, "attempt": number, "outcome": outcome, "time": iso(now_local()), "detail": detail})
            return number
        number = int(cc.record_attempt(self.results_root, self.binding, unit_id, outcome))
        self._write(self.details_path, {"kind": "attempt_detail", "unit_id": unit_id, "attempt": number, "outcome": outcome, "time": iso(now_local()), "detail": detail})
        return number


# ---------------------------------------------------------------------------------------
# the units
# ---------------------------------------------------------------------------------------

def save_failed_output(run_dir: Path, unit_id: str, attempt: int, data: pa.Table | Mapping[str, Any]) -> Path:
    """A failed attempt's partial output, apart from the parts; it never enters the table."""
    return common.write_parquet(run_dir / FAILED_DIR / f"{unit_id}__a{attempt}.parquet", data)


def run_attempt(
    run: common.RunHandle, tracker: AttemptTracker, backend: Any, job: common.Job, unit: Unit, keys: Sequence[str],
) -> int:
    """One attempt at one unit. Success writes the part and records ``complete``; a failure records its kind and raises ``UnitFailed``."""
    run.check_source()  # a changed source stops the dispatch and does not use up an attempt
    texts, meta = job.build()
    attempt = tracker.start(unit.unit_id)
    started = time.time()
    try:
        records = backend.readouts(texts)
    except Exception as error:  # noqa: BLE001
        reason = f"{type(error).__name__}: {error}"
        tracker.finish(unit.unit_id, "call_failure", error=reason)
        raise UnitFailed(unit.unit_id, attempt, "call_failure", reason) from error
    if len(records) != len(texts):
        reason = f"the back end returned {len(records)} readouts for {len(texts)} prompts"
        if records:
            save_failed_output(run.run_dir, unit.unit_id, attempt, common.trial_columns(records))
        tracker.finish(unit.unit_id, "call_failure", error=reason, readouts=len(records), prompts=len(texts))
        raise UnitFailed(unit.unit_id, attempt, "call_failure", reason)
    table = common.table_from_columns({**meta, **common.trial_columns(records)})
    problems = unit_row_problems(table, unit, keys)
    if problems:
        save_failed_output(run.run_dir, unit.unit_id, attempt, table)
        tracker.finish(unit.unit_id, "missing_rows", problems=problems)
        raise UnitFailed(unit.unit_id, attempt, "missing_rows", "; ".join(problems))
    run.parts.write(unit.unit_id, table)
    tracker.finish(unit.unit_id, "complete", rows=table.num_rows, wall_s=round(time.time() - started, 3))
    common.log(f"mechanism confirm {unit.unit_id}: {len(texts)} prompts in {time.time() - started:.1f} s (attempt {attempt})")
    return table.num_rows


@dataclass
class UnitsReport:
    done: list[str] = field(default_factory=list)  # finished in this invocation
    skipped: list[str] = field(default_factory=list)  # finished before, verified cell by cell
    dead: dict[str, Any] = field(default_factory=dict)  # unit -> the record of its last failure; the group is untestable
    rows: int = 0


def run_units(
    run: common.RunHandle, tracker: AttemptTracker, backend: Any, jobs: Mapping[str, common.Job], plan: Plan,
) -> UnitsReport:
    """The units in order. A finished unit is verified and kept, never run again; a unit with an attempt left is attempted; a unit whose attempts are used up ends the run as untestable."""
    report = UnitsReport()
    for unit in plan.units:
        if run.parts.done(unit.unit_id):
            problems = unit_row_problems(run.parts.read(unit.unit_id), unit, plan.frozen_keys)
            if problems:
                raise cc.ConfirmError(f"{unit.unit_id}: the finished part does not hold exactly the cells of the unit ({'; '.join(problems)}); it is not reused and not replaced")
            report.skipped.append(unit.unit_id)
            continue
        if tracker.count(unit.unit_id) >= MAX_ATTEMPTS:  # both attempts are used (an earlier invocation) and no part came of them: the group is untestable, nothing more is run
            report.dead[unit.unit_id] = tracker.last_failure(unit.unit_id) or {"outcome": "started", "detail": "the attempts were used up without a recorded failure"}
            break
        try:
            report.rows += run_attempt(run, tracker, backend, jobs[unit.unit_id], unit, plan.frozen_keys)
        except AttemptsExhausted:  # (the check above makes this a backstop)
            report.dead[unit.unit_id] = tracker.last_failure(unit.unit_id) or {"outcome": "started", "detail": "the attempts were used up without a recorded failure"}
            break
        except UnitFailed as failure:
            if failure.attempt >= MAX_ATTEMPTS:
                report.dead[unit.unit_id] = {"outcome": failure.outcome, "detail": failure.reason, "attempt": failure.attempt}
                break
            raise
        report.done.append(unit.unit_id)
    return report


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------

def load_pool(ctx: common.Context, plan: Plan) -> list[Any]:
    """The items of the pool: the holdout pool WITH the access of the registered batch (this is the call that consumes the keys in the ledger), a rehearsal's calibration pool without."""
    if plan.rehearsal:
        items = load_items_from_config(ctx.cfg, plan.pool, registry=plan.registry, results_root=ctx.results_root)
        plan.registry.assert_no_holdout([i.item_key for i in items], context=plan.pool)
    else:
        items = load_items_from_config(ctx.cfg, plan.pool, registry=plan.registry, results_root=ctx.results_root, holdout_access=plan.access)
    if tuple(i.item_key for i in items) != plan.frozen_keys:
        raise cc.ConfirmError(f"the items loaded for {plan.pool!r} are not the frozen key list")
    return items


def execute(ctx: common.Context, args: argparse.Namespace, hooks: ConfirmHooks) -> dict[str, Any]:
    plan = prepare(ctx, args)
    cfg = ctx.cfg
    entry = common.device_entry(cfg, args.model)
    spec = ReadoutSpec.from_config(cfg)
    opener = hooks.opener or (lambda c, kind, model, sp: common.open_backend(c, kind, model, sp))
    numeric = common.numeric_plan(common.readout_backend_plan(args, unit={"conditions_per_unit": CONDITIONS_PER_UNIT}))
    t0 = time.time()
    with common.open_run(ctx, pools=[plan.pool], parents=[plan.c1.run_id], extra_md5=plan.extra_md5, numeric=numeric, extra_manifest=plan.extra_manifest) as run:
        binding = None
        if not plan.rehearsal:
            binding = cc.bind_run(ctx.results_root, plan.batch, model=args.model, arm=ARM, run_dir=real_path(run.run_dir))  # before the pool is read
        tracker = AttemptTracker(run.run_dir, ctx.results_root, binding)
        items = load_pool(ctx, plan)
        bh = opener(ctx, args.backend, args.model, spec)
        try:
            run.confirm_numeric(args.backend, bh.numeric_actual())
            if hooks.model_files and not run.resumed:
                common.write_model_files(run, entry, bh)
            jobs = {u.unit_id: common.Job(u.unit_id, dev.condition_job(bh, plan.mbook, items, plan.pool, list(u.conds), plan.chat_kwargs, args.model)) for u in plan.units}
            report = run_units(run, tracker, bh.backend, jobs, plan)
        finally:
            bh.close()
        outcome = finish_run(run, plan, args, spec, items, report, tracker, time.time() - t0)
    return outcome


def finish_run(
    run: common.RunHandle, plan: Plan, args: argparse.Namespace, spec: ReadoutSpec, items: Sequence[Any], report: UnitsReport, tracker: AttemptTracker, wall_s: float,
) -> dict[str, Any]:
    """The table of the finished units and the summary. A run with a dead unit is untestable: its table holds the finished units only and the summary says why."""
    finished = [u for u in plan.units if run.parts.done(u.unit_id)]
    complete = len(finished) == len(plan.units) and not report.dead
    status = "complete" if complete else "untestable"
    summary: dict[str, Any] = {}
    n_rows = 0
    gate_rate = ask_rate = None
    if finished:
        table = run.parts.concat([u.unit_id for u in finished])
        choice = table.column("self_choice").to_numpy(zero_copy_only=False)
        gt = table.column("gt_position").to_numpy(zero_copy_only=False)
        table = table.append_column("self_correct", pa.array(np.asarray(choice == gt, dtype=bool))).select(dev.MECH_COLUMNS)
        common.write_parquet(run.run_dir / TABLE_FILE, table)
        summary = dev.summarize_mechanism(table, [c for u in finished for c in u.conds])
        n_rows, gate_rate, ask_rate = int(table.num_rows), summary["gate_pass_rate"], summary["ask_rate"]
    units_record = {
        u.unit_id: {
            "conditions": u.labels(), "status": "complete" if run.parts.done(u.unit_id) else ("failed" if u.unit_id in report.dead else "not_run"), "attempts": tracker.count(u.unit_id),
        }
        for u in plan.units
    }
    summary.update({
        "model": args.model, "backend": args.backend, "pool": plan.pool, "layout": dev.LAYOUT, "variants": list(plan.mbook.variants), "n_conditions": len(plan.conds),
        "min_gate_pass_rate": spec.min_gate_pass_rate, "mech_prompts": plan.identity["mech_prompts"], "conditions": plan.identity["conditions"],
        "jobs": {"units_done": len(report.done), "units_skipped": len(report.skipped), "rows": report.rows, "wall_s": wall_s},
        "confirm": {
            "status": status, "arm": ARM, "rehearsal": plan.rehearsal, "batch": plan.identity["batch"], "c1": plan.identity["c1"], "n_items": len(items), "n_rows_expected": len(plan.conds) * len(items),
            "units": units_record, "failed_units": {u: r for u, r in report.dead.items()}, "attempts_file": ATTEMPTS_FILE,
            "untestable_reason": None if complete else "a unit failed twice or did not run: the grid is incomplete, so neither claim of the group can be judged for this model",
        },
    })
    if plan.rehearsal:
        summary.update(exploratory=True, development_reason=REHEARSAL_REASON)
    else:  # (the ``attempts`` of the units are the registry's counts; the notes of the entry on the attempts are in this file, which no count reads.)
        summary["confirm"]["attempt_details_file"] = DETAILS_FILE
    common.write_json(run.run_dir / SUMMARY_FILE, summary)
    if not complete:  # finished, so that --resume refuses it; the exit code of the manifest says that the group is untestable
        run.record.finish("ok", exit_code=1, extra={"confirm_status": status})
    return {
        "run_dir": str(run.run_dir), "model": args.model, "status": status, "rehearsal": plan.rehearsal, "n_rows": n_rows, "gate_pass_rate": gate_rate, "ask_rate": ask_rate,
        "units": {k: v["status"] for k, v in units_record.items()}, "failed_units": sorted(report.dead),
    }


# ---------------------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------------------

def dry_run(ctx: common.Context, args: argparse.Namespace) -> int:
    """Everything that needs no loader: the same checks as the run, each reported on its own; no pool, no model, no ledger write, no binding, no run directory."""
    dry = common.DryRun(ctx)
    rehearsal = bool(args.rehearsal)

    def step(name: str, fn: Callable[[], Any], ok_detail: Callable[[Any], Any] = lambda v: "") -> Any:
        try:
            value = fn()
        except (common.EntryError, cc.ConfirmError, SplitError, paths.PathPolicyError, HoldoutSealedError, HoldoutLedgerError) as error:
            dry.check(name, False, str(error))
            return None
        dry.check(name, True, ok_detail(value))
        return value

    if step("arguments", lambda: refuse_unusable_arguments(args) or True) is None:
        return dry.emit()
    pool = pool_of(args)
    common.dry_run_basics(dry, pools=[pool], models=[("device", args.model)], seed_keys=[])
    mbook = step("mechanism prompts", lambda: dev.load_mech_prompts(ctx), lambda b: {"path": str(b.path), "md5": b.md5, "version": b.version, "base_prompts_md5": b.base.md5})
    conds = None
    if mbook is not None:
        conds = step("conditions", lambda: dev.condition_list(mbook, list(mbook.variants)), lambda c: {"n": len(c)})
    keys = step("frozen key list", lambda: ctx.registry().keys(pool), lambda k: {"pool": pool, "n_items": len(k)})
    batch = step("batch manifest", lambda: load_batch(args, args.model),
                 lambda b: None if b is None else {"batch_id": b.batch_id, "pool": b.pool, "models": list(b.models), "arms": list(b.arms), "md5": b.md5})
    bound = None
    if batch is not None:
        bound = step("files bound by the batch manifest", lambda: check_bound_files(batch, ctx), lambda v: v)
    else:
        bound = {key: file_md5(path) for key, path in bound_file_paths(ctx).items()}
    prereg = step("frozen pre-registration", lambda: check_prereg(args, batch, rehearsal), lambda p: None if p is None else {"path": str(p), "md5": file_md5(p) if p.is_file() else None})
    block = None
    if not rehearsal and batch is not None:
        step("batch identity in the ledger", lambda: cc.check_batch_identity(ctx.results_root, batch, model=args.model, arm=ARM) or True)
        step("holdout access (marker and registered batch)", lambda: cc.open_confirm_access(ctx.results_root, batch, ctx.registry()) and True)
        block = step("manifest block", lambda: cc.confirm_manifest_block(batch, prereg_frozen_path=prereg))
    c1 = None
    if keys is not None:
        c1 = step("C1 run", lambda: read_c1(args.c1_run, model=args.model, pool=pool, frozen_keys=keys, results_root=ctx.results_root, rehearsal=rehearsal, block=block), lambda c: c.record())
    if mbook is not None and conds is not None:
        units = build_units(mbook, conds)
        sizes = common.pool_sizes(ctx, [pool])
        dry.note("workload", {
            "pool": sizes, "conditions": len(conds), "mechanism_prompts": sizes[pool] * len(conds), "units": len(units), "conditions_per_unit": CONDITIONS_PER_UNIT, "backend": args.backend,
            "stage": ctx.stage, "max_attempts_per_unit": MAX_ATTEMPTS,
        })
        if args.resume and c1 is not None and bound is not None:  # the run being resumed must have been started with what this call has
            step("resume: the run being resumed has this identity", lambda: check_resume_identity(args, mechanism_identity(args, mbook, conds, pool, batch, c1, bound)) or {"resume": args.resume})
    dry.note("confirmation", {"rehearsal": rehearsal, "arm": ARM, "exploratory": rehearsal, "batch_id": None if batch is None else batch.batch_id})
    return dry.emit()


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None, hooks: ConfirmHooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or ConfirmHooks()
    args = make_parser().parse_args(argv)
    t0 = time.time()
    try:
        ctx = common.build_context(args, script=SCRIPT, stage=stage_of(args), argv=argv)
        normalise_results_root(ctx)
        if args.dry_run:
            return dry_run(ctx, args)
        outcome = execute(ctx, args, hooks)
    except (common.EntryError, cc.ConfirmError, SplitError, paths.PathPolicyError, HoldoutSealedError, HoldoutLedgerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except UnitFailed as failure:
        print(
            f"error: {failure}\nThe unit has {'no attempt left: the group is untestable for this model' if failure.attempt >= MAX_ATTEMPTS else 'one attempt left'}; "
            "continue with --resume RUN_DIR (the same run directory, never a new label).", file=sys.stderr,
        )
        return 1
    except common.SourceChangedError as error:
        return common.stopped_by_source_change(error)
    outcome["wall_s"] = time.time() - t0
    print(json.dumps(common.json_safe(outcome), indent=2, sort_keys=True))
    return 0 if outcome["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
