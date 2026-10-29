#!/usr/bin/env python
"""Confirmatory batch of the real scenario S1: the three phases of ``scripts/run_scenario.py`` (expert, answer, decide)
on the TriviaQA SEALED pool, through a registered batch. Stage ``scenario``, label ``s1seal_...``.

The development entry ``scripts/run_scenario.py`` refuses the sealed pool. This entry is the only one that reads the sealed pool. It imports the development entry for
what the two share without change (the units of work, the rendering, the numeric plan, the run record, the raw-output files) and writes the rest itself. The steps of the opening come from the shared layer
``scripts/_confirm_common.py`` (``run_preflight``, ``confirm_manifest_block`` with ``open_run_kwargs``, ``bind_run``, ``load_confirm_items``, ``record_attempt``); this file adds what is the scenario's.

What it does differently from the development entry:

* Configuration: ``configs/scenario_s1_confirm_v1.yaml`` (``nmi.scenario.config.load_confirm_scenario_config``): ``exploratory: false`` and the frozen ``a_pct`` (64, the expert's accuracy on the DEVELOPMENT
  pool; nothing on the sealed pool changes it). ``--results-root`` is named explicitly (the shared layer never falls back to a default root).
* Order. Every check that can fail comes first and the pool is loaded last: the arguments and the label (``s1seal_``), then the preflight of the shared layer (the pool of the batch, the label, the model
  directory, the batch identity, the md5 of every file the batch manifest binds, the numeric settings, the unique run of the (batch, model, arm) or the identity of the run that is resumed), the md5 of the
  frozen pre-registration, the unseal marker, the frozen fit and its pins, the source file, the chat template of the model, the files this entry binds against the batch manifest and the identity of a resumed
  run. Only then the run directory is made and bound to (batch, model, phase) in ``confirm_runs.jsonl``, the engine is started and the pool is loaded through the access of the batch, the one place where the
  ledger consumes keys. A failure before that consumes nothing.
* Answer phase: nothing is fitted. z is the FINAL full-pool fit of the development answer run (``--dev-answer-run``, pinned by ``--expect-dev-run-id`` and ``--expect-dev-summary-md5``, the frozen values)
  applied to the raw confidences of the sealed pool by ``nmi.scenario.analysis.batch_z``; ``nmi.scenario.calibration.calibrate`` is never called. The summary judges the QUALITY GATE
  of the batch (``nmi.scenario.sealed``): digit mass at least 0.9 for at least 99 % of ALL registered items, and at most 1 % of the answer-phase results missing.
* Expert phase: the summary suggests no ``a_pct``.
* No ``--limit-items``, no ``--conditions`` in a confirmatory run: it covers every registered item and all 18 conditions.

Units and attempts. One UNIT is the unit of work of the development entry (a chunk of items; for decide a chunk under one condition). One ATTEMPT is one execution of
the frozen protocol function of a unit (``nmi.scenario.protocol``), which itself runs a generation that raised or did not come back once more with the same settings (``generate_with_retry``);
what still has no result after that is a missing cell (``status = missing``; it is not a format failure, and the share of a condition is gated at 1 %). The attempts of a unit are counted in the central registry
``confirm_runs.jsonl`` across processes and labels (``record_attempt``: ``started``, then ``complete``, ``call_failure`` or ``missing_rows``; the shared layer copies every record to ``attempts.jsonl`` of the run
directory): at most two, the original and one after a resume. A unit that completes is never run again, so nothing is picked from several attempts: its table is that of the one attempt that completed it.

An attempt COMPLETES when its table holds exactly the expected cells of the unit and at least one of them has a result. Cells that are missing after the protocol's own rerun are cells of that table (they
are the missing cells of the 1 % gates, per phase and per condition; a rerun of the whole unit would make a 1 % gate depend on 250 other items, and the greedy generation fails the same way
again). Two things FAIL an attempt: a CALL FAILURE (an exception leaves the protocol, or no cell of the unit has a result and the back end raised or returned the wrong number of results) and MISSING ROWS (the
table is not the expected cells of the unit, or no cell has a result and nothing was raised). A failed attempt keeps its partial output apart in ``failed_attempts/`` (the table, the generated texts and a record of
what failed), is never merged, and stops the run (exit 3) so that the other units keep their attempts; ``--resume RUN_DIR`` continues in the bound run directory. A unit that failed twice is exhausted: its cells
are written as missing rows (exit 1 when the run ends so; the analysis then finds the condition or the phase not testable by the 1 % gates). A digit mass below 0.9, a format failure and a self-answer are
readings, never failures, and are never retried.

``--rehearsal`` runs the same code on the development pool that stands for the sealed pool (``REHEARSAL_POOLS``), marks the run exploratory (``rehearsal_manifest_block``) and touches no ledger, no registry and no
batch; the label starts with ``s1seal_rehearsal_``. ``--dry-run`` checks everything above, prints the plan and loads no pool, no tokenizer and no model (``--check-template`` adds the chat template).

Output (``<results-root>/scenario/<time>_s1seal_<tag>/``): ``<phase>.parquet`` (the tables of ``nmi.scenario.protocol``, with the extra column ``z_frozen`` for answer), ``<phase>_summary.json``,
``raw_outputs.jsonl.gz``, ``attempts.jsonl``, ``failed_attempts/`` (only after a failed attempt), ``llm/model.json``, ``manifest.json``. Nothing the entry prints or logs contains a question or a generated text.

Typical use (one GPU per run):
    python -I scripts/run_scenario_confirm.py --phase expert --model qwen3-32b-awq --tag s1seal_expert --gpu-mem-util 0.85 --batch-manifest BATCH.json --results-root results
    python -I scripts/run_scenario_confirm.py --phase answer --model qwen3-4b --tag s1seal_answer_qwen3-4b --gpu-mem-util 0.45 --batch-manifest BATCH.json --results-root ... \\
        --dev-answer-run RESULTS/scenario/<dev answer run> --expect-dev-run-id scenario/<dev answer run> --expect-dev-summary-md5 MD5
    python -I scripts/run_scenario_confirm.py --phase decide --model qwen3-4b --tag s1seal_decide_qwen3-4b --gpu-mem-util 0.45 --batch-manifest BATCH.json --results-root ...
    python -I scripts/run_scenario_confirm.py --rehearsal --phase expert --model qwen3-32b-awq --tag s1seal_rehearsal_expert --limit-items 20 --gpu-mem-util 0.85 --results-root ...

Exit codes: 0 done; 1 done, but a unit failed twice and its cells are missing (not testable there), or a dry run found a problem; 2 unusable input or arguments (nothing was consumed); 3 a unit failed an
attempt and the run can be resumed; 4 the source guard stopped the dispatch. A chain script runs the entry and, on exit 3, again with ``--resume RUN_DIR`` (no ``--tag`` is needed); the registry stops a unit at
its second failed attempt, so the loop ends.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _c2_common as common  # noqa: E402
import pyarrow as pa  # noqa: E402
import run_scenario as dev  # noqa: E402

# (after run_scenario: the shared layer puts its own directory at the front of sys.path, and the development entry must be the one next to this file)
import _confirm_common as cc  # noqa: E402

from nmi import paths  # noqa: E402
from nmi.data import triviaqa  # noqa: E402
from nmi.data.registry import InputHashMismatchError, SplitError  # noqa: E402
from nmi.prompts import PromptError  # noqa: E402
from nmi.provenance import file_md5, iso, now_local  # noqa: E402
from nmi.scenario import analysis, model_files, protocol, readout, render, sealed, summary  # noqa: E402
from nmi.scenario import config as scenario_config  # noqa: E402
from nmi.scenario.config import DEFAULT_CONFIRM_CONFIG, ModelEntry, ScenarioConfigError, load_confirm_scenario_config  # noqa: E402
from nmi.scenario.prompts import Condition, ScenarioPromptBook  # noqa: E402
from nmi.scenario.task import TASKS  # noqa: E402

SCRIPT = "run_scenario_confirm"
STAGE = dev.STAGE  # the confirmatory runs are made in the stage of the scenario; their label starts with s1seal_
TAG_PREFIX = "s1seal_"
REHEARSAL_PREFIX = "s1seal_rehearsal_"
REPO_ROOT = paths.REPO_ROOT  # the files the batch manifest binds are named relative to it
MAX_ATTEMPTS = cc.MAX_ATTEMPTS
ATTEMPTS_FILE = cc.ATTEMPTS_COPY_NAME
FAILED_DIR = "failed_attempts"
OUTCOMES = ("complete", "call_failure", "missing_rows")
UNIT_FAILED_REASON = "unit_failed_twice"  # the ``missing_reason`` of the cells of a unit whose two attempts failed
UNLISTED_OK = frozenset({"dev_answer_summary"})  # pinned by --expect-dev-summary-md5 and --expect-dev-run-id (a result file, not a frozen code or configuration file)
EXIT_NOT_TESTABLE = 1
EXIT_UNIT_FAILED = 3
ScenarioHooks = dev.ScenarioHooks
USER_ERRORS = (*dev.USER_ERRORS, cc.ConfirmError)


class UnitFailure(RuntimeError):
    """A unit failed an attempt: its partial output is kept apart and the run stops so that the other units keep their attempts; ``--resume`` continues (exit 3)."""


# ---------------------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Confirmatory batch of the real scenario S1: the phases expert, answer and decide of one model on the sealed pool of TriviaQA.")
    p.add_argument("--phase", required=True, choices=protocol.PHASES, help="expert (the expert answers), answer (P-answer, the confidence follow-up and z from the frozen development fit) or decide (P-decide, all 18 conditions)")
    p.add_argument("--model", required=True, help="model id of the run configuration: the expert for --phase expert, an end-side model for the others")
    p.add_argument("--config", default=str(DEFAULT_CONFIRM_CONFIG), help="confirmatory run configuration (default: configs/scenario_s1_confirm_v1.yaml)")
    p.add_argument("--expect-config-md5", default=None, help="stop unless the configuration has exactly this md5 (the frozen value)")
    p.add_argument("--results-root", default=None, help="REQUIRED: the one results root of the ledger, the unseal marker, the registry of runs and the runs (relative to the repository when it is relative; there is no default)")
    p.add_argument("--split-dir", default=None, help="registry directory of the pools (default: data.split_dir of the configuration)")
    p.add_argument("--source", default=None, help="the TriviaQA validation parquet (default: data.source of the configuration; its sha256 must be the one the registry was sealed on)")
    p.add_argument("--tag", default=None, help=f"run tag, must start with {TAG_PREFIX} ({REHEARSAL_PREFIX} for --rehearsal): <results-root>/scenario/<time>_<tag>/ (required unless --dry-run or --resume; an existing tag is an error)")
    p.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue an interrupted run in its own bound directory; finished units are skipped, a unit with a failed attempt gets its second")
    p.add_argument("--dry-run", action="store_true", help="check the arguments, the configuration, the batch, the files, the pool registry and the model directory; print the plan; load no pool, no tokenizer and no model")
    p.add_argument("--check-template", action="store_true", help="with --dry-run: also load the tokenizer of the model and check the chat template, the tool injection and the digits")
    p.add_argument("--rehearsal", action="store_true", help="the same code on the development pool that stands for the sealed pool: exploratory; no ledger, no registry, no batch")
    p.add_argument("--batch-manifest", default=None, metavar="PATH", help="the frozen batch manifest (JSON) of the batch (required unless --rehearsal)")
    p.add_argument("--prereg-frozen", default=None, help="the frozen pre-registration (default: FROZEN_PREREG_PATH of the shared layer); its md5 must be the one the batch manifest names")
    p.add_argument("--dev-answer-run", default=None, metavar="RUN_DIR", help="answer phase: the development answer run of this model, whose answer_summary.json holds the frozen fit")
    p.add_argument("--expect-dev-run-id", default=None, help="answer phase: the frozen run id of --dev-answer-run, <stage>/<run directory> as its manifest names it; required unless --rehearsal")
    p.add_argument("--expect-dev-summary-md5", default=None, help="answer phase: the frozen md5 of answer_summary.json of --dev-answer-run; required unless --rehearsal")
    p.add_argument("--conditions", default=None, help="rehearsal only (decide): comma list of condition labels or group names; a confirmatory run covers all 18")
    p.add_argument("--limit-items", type=int, default=None, help="rehearsal only: the first N items of the pool (part of the identity of the run)")
    p.add_argument("--backend", choices=("vllm_offline",), default="vllm_offline")
    p.add_argument("--gpu-mem-util", type=float, default=None, help="vLLM gpu_memory_utilization (offline); set it from the free memory measured before the run (the batch manifest may freeze it)")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--llm-kwargs", default=None, help="JSON object merged last into the keyword arguments of vllm.LLM (after the model's llm_kwargs of the configuration; the batch manifest may freeze it)")
    return p


def check_arguments(args: argparse.Namespace) -> None:
    """The label and the combinations of arguments, before the configuration or anything else is read (EntryError, exit 2)."""
    tag = args.tag
    if tag is None:
        if not args.dry_run and not args.resume:
            raise common.EntryError("--tag is required for a real run")
    else:
        try:
            paths.validate_name("tag", tag)
            cc.check_output_name(tag)
        except (paths.PathPolicyError, cc.ConfirmError) as error:
            raise common.EntryError(str(error)) from error
        if not args.resume:  # (a resumed run keeps the label it was started with; the argument is not used)
            if args.rehearsal and not tag.startswith(REHEARSAL_PREFIX):
                raise common.EntryError(f"the label of a rehearsal must start with {REHEARSAL_PREFIX}, got {tag!r}")
            if not args.rehearsal and (not tag.startswith(TAG_PREFIX) or tag.startswith(REHEARSAL_PREFIX)):
                raise common.EntryError(f"the label of a confirmatory run must start with {TAG_PREFIX} (and not {REHEARSAL_PREFIX}, which is for --rehearsal), got {tag!r}")
    if args.check_template and not args.dry_run:
        raise common.EntryError("--check-template belongs to --dry-run")
    if args.rehearsal:
        if args.batch_manifest:
            raise common.EntryError("a rehearsal touches no ledger, no registry and no batch: it takes no --batch-manifest")
    else:
        if args.limit_items is not None or args.conditions:
            raise common.EntryError("--limit-items and --conditions are for --rehearsal: a confirmatory run covers every registered item and all 18 conditions")
        if not args.batch_manifest:
            raise common.EntryError("--batch-manifest is required for a confirmatory run (the frozen manifest of the batch); use --rehearsal for a run on the development pool")
    if args.limit_items is not None and args.limit_items < 1:
        raise common.EntryError("--limit-items must be positive")
    pins = (args.expect_dev_run_id, args.expect_dev_summary_md5)
    if args.phase == "answer":
        if not args.dev_answer_run:
            raise common.EntryError("--phase answer needs --dev-answer-run: the development answer run of this model whose final fit gives z (nothing is fitted on the sealed pool)")
        if not args.rehearsal and not all(pins):
            raise common.EntryError("a confirmatory answer run needs --expect-dev-run-id and --expect-dev-summary-md5, the frozen values of the development answer run")
    elif args.dev_answer_run or any(pins):
        raise common.EntryError("--dev-answer-run and its pins belong to --phase answer")


def build_context(args: argparse.Namespace, *, argv: Sequence[str]) -> dev.ScenarioContext:
    results_root = cc.canonical_results_root(args.results_root)  # one normalised directory for the ledger, the registry, the access and the runs; none given: ConfirmError
    try:
        cfg = load_confirm_scenario_config(args.config, expected_md5=args.expect_config_md5)
    except (ScenarioConfigError, OSError) as error:
        raise common.EntryError(f"configuration: {error}") from error
    try:
        book = ScenarioPromptBook.load(cfg.prompts_path)
    except (PromptError, OSError) as error:
        raise common.EntryError(f"prompts: {error}") from error
    task = TASKS[cfg.task]
    try:
        split = task.load_split(args.split_dir or cfg.split_dir)
    except (SplitError, OSError, ValueError, KeyError) as error:
        raise common.EntryError(f"pool registry: {error}") from error
    return dev.ScenarioContext(
        args=args, cfg=cfg, book=book, task=task, split=split, results_root=results_root, source=Path(args.source) if args.source else cfg.source, script=SCRIPT, stage=STAGE, argv=[f"scripts/{SCRIPT}.py", *argv],
    )


def pool_of(args: argparse.Namespace) -> str:
    """The sealed pool for a confirmatory run, the development pool that stands for it (``REHEARSAL_POOLS``) for a rehearsal."""
    return cc.rehearsal_pool(triviaqa.SEALED_POOL) if args.rehearsal else triviaqa.SEALED_POOL


def prereg_path(args: argparse.Namespace) -> Path:
    return Path(args.prereg_frozen) if args.prereg_frozen else Path(cc.FROZEN_PREREG_PATH)


def label_for_checks(args: argparse.Namespace) -> str | None:
    """The label the preflight checks: the one given, or (for a dry run without one) a placeholder with the right prefix."""
    if args.tag or args.resume:
        return args.tag
    return (REHEARSAL_PREFIX if args.rehearsal else TAG_PREFIX) + "dryrun"


def repo_key(path: str | Path) -> str | None:
    """How the batch manifest names a file: its path relative to the repository (POSIX); None for a file outside the repository (it cannot be named there)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------
# the batch, the frozen fit and the bound files
# ---------------------------------------------------------------------------------------

def preflight(ctx: dev.ScenarioContext, batch: Any, entry: ModelEntry) -> cc.PreflightReport:
    """The preflight of the shared layer for this (batch, model, phase): every check that precedes the pool, each reported with its outcome (none stops the others)."""
    args = ctx.args
    return cc.run_preflight(
        ctx.results_root, batch, model=args.model, arm=args.phase, pool=triviaqa.SEALED_POOL, stage=STAGE, tag=label_for_checks(args), model_dir=entry.path, gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len, llm_kwargs=args.llm_kwargs, resume_dir=args.resume, repo_root=REPO_ROOT, tag_prefix=TAG_PREFIX,
    )


def check_files_listed(batch: Any, files: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """The other direction of the check of the bound files (``run_preflight`` checks that every file the manifest lists still has its md5): every file this entry binds that lies inside the repository is listed in
    the batch manifest, with the md5 it has now. A file outside the repository cannot be named there; it is pinned by an argument instead (``--expect-config-md5``). The summary of the frozen fit is a result
    file and is pinned by ``--expect-dev-run-id`` and ``--expect-dev-summary-md5``."""
    report: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for name, path in files.items():
        key, actual = repo_key(path), file_md5(path)
        frozen = None if key is None else batch.files_md5.get(key)
        report[name] = {"path": key, "outside_repository": key is None, "md5": actual, "frozen_md5": frozen}
        if key is None or name in UNLISTED_OK:
            continue
        if frozen is None:
            problems.append(f"{name} ({key}) is not listed in the batch manifest")
        elif frozen != actual:
            problems.append(f"{name} ({key}) has md5 {actual[:12]}, the batch manifest froze {frozen[:12]}")
    if problems:
        raise cc.ConfirmError("the files this entry binds are not those the batch manifest froze: " + "; ".join(problems) + " (the code that runs is not the code that was frozen)")
    return report


@dataclass(frozen=True)
class FrozenFit:
    """The development answer run whose final full-pool fit gives z of the sealed batch: its directory, its run id, its summary and the md5 of that file."""

    run_dir: Path
    run_id: str
    summary_path: Path
    summary_md5: str
    summary: dict[str, Any]

    def source(self) -> dict[str, Any]:
        return {"run_dir": str(self.run_dir), "run_id": self.run_id, "file": self.summary_path.name, "md5": self.summary_md5}

    def as_dict(self) -> dict[str, Any]:
        final = (self.summary.get("calibration") or {}).get("final") or {}
        return {**self.source(), "z_valid": self.summary.get("z_valid"), "fit": {k: final.get(k) for k in ("a", "b", "n", "ok")}}


def load_frozen_fit(args: argparse.Namespace, model_id: str) -> FrozenFit:
    """The development answer run of the model (``--dev-answer-run``): a finished answer run of this model on the development pool, whose run id and ``answer_summary.json`` are the frozen ones (the pins; a
    rehearsal may omit them). Its summary is the frozen fit. EntryError (exit 2) otherwise."""
    run_dir = Path(args.dev_answer_run)
    manifest_path, summary_path = run_dir / common.MANIFEST_FILE, run_dir / "answer_summary.json"
    for needed in (manifest_path, summary_path):
        if not needed.is_file():
            raise common.EntryError(f"--dev-answer-run {run_dir}: {needed.name} is missing (nothing is fitted on the sealed pool; the frozen fit of the development run is needed)")
    manifest = common.read_json(manifest_path)
    run = manifest.get("scenario_run") if isinstance(manifest, dict) else None
    if not isinstance(run, dict):
        raise common.EntryError(f"--dev-answer-run {run_dir}: not a run of the scenario (no scenario_run in its manifest)")
    for what, got, want in (
        ("stage", manifest.get("stage"), STAGE), ("exit_status", manifest.get("exit_status"), "ok"), ("phase", run.get("phase"), "answer"), ("model", (run.get("model") or {}).get("id"), model_id),
        ("pool", run.get("pool"), triviaqa.DEV_POOL),
    ):
        if got != want:
            raise common.EntryError(f"--dev-answer-run {run_dir}: its {what} is {got!r}, the frozen fit must come from {want!r}")
    run_id, md5 = str(manifest.get("run_id")), file_md5(summary_path)
    if args.expect_dev_run_id is not None and run_id != args.expect_dev_run_id:
        raise common.EntryError(f"--dev-answer-run {run_dir} is the run {run_id!r}, the frozen run is {args.expect_dev_run_id!r}; the frozen fit does not match")
    if args.expect_dev_summary_md5 is not None and md5 != args.expect_dev_summary_md5:
        raise common.EntryError(f"--dev-answer-run {run_dir}: answer_summary.json has md5 {md5}, the frozen value is {args.expect_dev_summary_md5}; the frozen fit does not match")
    data = common.read_json(summary_path)
    if not isinstance(data, dict) or "z_valid" not in data:
        raise common.EntryError(f"--dev-answer-run {run_dir}: answer_summary.json is not the summary of an answer run")
    return FrozenFit(run_dir=run_dir, run_id=run_id, summary_path=summary_path, summary_md5=md5, summary=data)


def bound_files(ctx: dev.ScenarioContext, frozen: FrozenFit | None) -> dict[str, Path]:
    """The files whose md5 the manifest binds: those of the development entry (configuration, prompts, adapter, scoring, readout, protocol, calibration, summary, analysis), the development entry itself and
    the module of the run lifecycle (this entry imports them), this entry, the sealed summary, the configuration module, the shared layer, and the summary of the frozen fit."""
    files = dev.bound_files(ctx.cfg)
    files["dev_entry"] = files.pop("entry")
    files["c2_common"] = Path(common.__file__).resolve()  # the run lifecycle, the part store and the dry run of the entries
    files["entry"] = Path(__file__).resolve()
    files["sealed"] = Path(sealed.__file__).resolve()
    files["scenario_config"] = Path(scenario_config.__file__).resolve()
    shared = getattr(cc, "__file__", None)
    if shared:
        files["confirm_common"] = Path(shared).resolve()
    if frozen is not None:
        files["dev_answer_summary"] = frozen.summary_path
    return files


# ---------------------------------------------------------------------------------------
# the attempts of a unit
# ---------------------------------------------------------------------------------------

def exhausted_of(records: Sequence[Mapping[str, Any]]) -> bool:
    """Two attempts were started and the unit is not complete (an attempt that was started and never closed counts, as in the shared layer)."""
    return cc.state_of_records(records) != "complete" and sum(1 for r in records if r.get("outcome") == "started") >= MAX_ATTEMPTS


class AttemptBook:
    """The attempts of the units of one run. A confirmatory run counts them in the central registry (``cc.record_attempt``, with the binding of the run); the shared layer copies every record to ``attempts.jsonl``
    of the run directory. A rehearsal has no registry: it counts in ``attempts.jsonl`` alone, in records of the same shape."""

    def __init__(self, run_dir: Path, *, model: str, arm: str, results_root: Path | None = None, binding: Any = None) -> None:
        self.path = run_dir / ATTEMPTS_FILE
        self.run_dir, self.model, self.arm = run_dir, model, arm
        self.results_root, self.binding = results_root, binding

    def local_rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def records(self, unit_id: str) -> list[dict[str, Any]]:
        if self.binding is not None:
            return cc.attempts(self.results_root, self.binding, unit_id)
        return [r for r in self.local_rows() if r.get("unit_id") == unit_id]

    def started(self, unit_id: str) -> int:
        return sum(1 for r in self.records(unit_id) if r.get("outcome") == "started")

    def exhausted(self, unit_id: str) -> bool:
        return exhausted_of(self.records(unit_id))

    def _append(self, unit_id: str, attempt: int, outcome: str) -> None:
        record = {"kind": "attempt", "batch_id": None, "model": self.model, "arm": self.arm, "run_dir": str(self.run_dir), "unit_id": unit_id, "attempt": attempt, "outcome": outcome, "utc": iso(now_local())}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def start(self, unit_id: str) -> int:
        """Open the next attempt (the number, 1 or 2). ``AttemptLimitError`` when the unit has had two; ``ConfirmError`` when it is complete."""
        if self.binding is not None:
            return int(cc.record_attempt(self.results_root, self.binding, unit_id, "started"))
        records = self.records(unit_id)
        if cc.state_of_records(records) == "complete":
            raise cc.ConfirmError(f"unit {unit_id!r} is complete; a complete unit is not run again")
        if exhausted_of(records):
            raise cc.AttemptLimitError(f"unit {unit_id!r} has had {MAX_ATTEMPTS} attempts")
        number = self.started(unit_id) + 1
        self._append(unit_id, number, "started")
        return number

    def finish(self, unit_id: str, attempt: int, outcome: str) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        if self.binding is not None:
            cc.record_attempt(self.results_root, self.binding, unit_id, outcome)
        else:
            self._append(unit_id, attempt, outcome)

    def overview(self, unit_ids: Sequence[str]) -> dict[str, Any]:
        """What the records say of the units of the run: how many were attempted, which needed a second attempt, every attempt that did not complete, and the exhausted units."""
        attempted: list[str] = []
        second: list[str] = []
        failed: list[dict[str, Any]] = []
        exhausted: list[str] = []
        for unit_id in unit_ids:
            rows = self.records(unit_id)
            if not rows:
                continue
            attempted.append(unit_id)
            if sum(1 for r in rows if r.get("outcome") == "started") >= 2:
                second.append(unit_id)
            failed += [{"unit": unit_id, "attempt": r["attempt"], "outcome": r["outcome"]} for r in rows if r["outcome"] in ("call_failure", "missing_rows")]
            if exhausted_of(rows):
                exhausted.append(unit_id)
        return {"n_units_attempted": len(attempted), "units_with_a_second_attempt": second, "attempts_not_complete": failed, "exhausted_units": exhausted}


def missing_unit_table(spec: dev.UnitSpec, items: Sequence[Any], schema: pa.Schema) -> pa.Table:
    """The cells of a unit whose two attempts are used up: every expected row of the unit as a missing row (``status = missing``, ``missing_reason = unit_failed_twice``), made without running the protocol
    (a defect that raised in the protocol would raise again there)."""
    keys = [items[i].item_key for i in spec.rows]
    empty = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    table, _ = sealed.complete_table(empty, schema, keys, conditions=[spec.condition] if spec.condition is not None else None)
    index = schema.get_field_index("missing_reason")
    return table.set_column(index, schema.field(index), pa.array([UNIT_FAILED_REASON] * table.num_rows, type=pa.string()))


def classify(out: protocol.UnitOutput, spec: dev.UnitSpec, items: Sequence[Any], schema: pa.Schema) -> tuple[str, dict[str, Any], pa.Table | None]:
    """What an attempt came to: (outcome, info, table).

    * ``complete``: the table holds exactly the expected cells of the unit and at least one has a result. Cells that are missing after the protocol's own rerun are cells of the table (``info["n_missing"]``);
    * ``call_failure``: no cell has a result and the back end raised or returned the wrong number of results (the call never came back);
    * ``missing_rows``: the table is not the expected cells of the unit (a cell lacking, repeated or foreign), or no cell has a result although nothing was raised."""
    n_rows = len(spec.rows)
    info: dict[str, Any] = {"n_rows": n_rows, "batch_errors": out.stats.batch_errors, "retried": out.stats.retried, "error_classes": dict(out.stats.error_classes), "seconds": out.seconds}
    try:
        table = out.table(schema)
    except (ValueError, KeyError, TypeError, pa.ArrowException) as error:
        return "missing_rows", {**info, "n_missing": None, "detail": f"the columns do not make the table of the phase ({type(error).__name__})"}, None
    expected = [items[i].item_key for i in spec.rows]
    cells_ok = table.column("item_key").to_pylist() == expected
    if spec.condition is not None:
        cells_ok = cells_ok and set(table.column("condition").to_pylist()) == {spec.condition.label}
    if not cells_ok:
        return "missing_rows", {**info, "n_missing": None, "detail": "the rows are not the expected cells of the unit"}, table
    n_missing = int(analysis.missing_flags(table).sum())
    info["n_missing"] = n_missing
    if n_missing == n_rows and n_rows > 0:
        if out.stats.batch_errors > 0:
            return "call_failure", {**info, "detail": "no cell came back and the back end raised or returned the wrong number of results"}, table
        return "missing_rows", {**info, "detail": "no cell of the unit has a result"}, table
    return "complete", info, table


def save_failed_attempt(run_dir: Path, spec: dev.UnitSpec, number: int, outcome: str, info: Mapping[str, Any], out: protocol.UnitOutput | None, table: pa.Table | None) -> None:
    """The partial output of a failed attempt, kept apart and never merged: the table (if it could be built), the generated texts, and a record of what failed."""
    directory = run_dir / FAILED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{paths.validate_name('unit', spec.unit_id)}__attempt{number}"
    parts: dict[str, str | None] = {"table": None, "texts": None}
    if table is not None:
        common.write_parquet(directory / f"{stem}.parquet", table)
        parts["table"] = f"{stem}.parquet"
    if out is not None and out.raw:
        (directory / f"{stem}.jsonl.gz").write_bytes(dev.gzip_bytes([{**line} for line in out.raw]))
        parts["texts"] = f"{stem}.jsonl.gz"
    common.write_json(directory / f"{stem}.json", {"unit_id": spec.unit_id, "attempt": number, "outcome": outcome, **dict(info), "partial_output": parts, "merged": False})


def write_unit(run: common.RunHandle, spec: dev.UnitSpec, out: protocol.UnitOutput, schema: pa.Schema, number: int, outcome: str) -> None:
    dev.write_raw_part(run.run_dir, spec.unit_id, [{**line} for line in out.raw])
    common.write_json(run.run_dir / dev.UNIT_STATS / f"{spec.unit_id}.json", {"unit": spec.unit_id, "seconds": out.seconds, "generation": out.stats.as_dict(), "attempt": number, "outcome": outcome})
    run.parts.write(spec.unit_id, out.table(schema))  # the part is written last: a unit is done when its table is there


def write_missing_unit(ctx: dev.ScenarioContext, run: common.RunHandle, spec: dev.UnitSpec, items: Sequence[Any]) -> None:
    """The part of an exhausted unit: every cell a missing row, no raw texts, generation statistics that say so."""
    n = len(spec.rows)
    dev.write_raw_part(run.run_dir, spec.unit_id, [])
    stats = protocol.GenStats(n=n, missing=n)
    common.write_json(run.run_dir / dev.UNIT_STATS / f"{spec.unit_id}.json", {"unit": spec.unit_id, "seconds": 0.0, "generation": stats.as_dict(), "attempt": MAX_ATTEMPTS, "outcome": "exhausted"})
    run.parts.write(spec.unit_id, missing_unit_table(spec, items, protocol.SCHEMAS[ctx.args.phase]))


def run_one_unit(
    ctx: dev.ScenarioContext, run: common.RunHandle, bh: common.BackendHandle, renderer: protocol.Renderer, spec: dev.UnitSpec, items: Sequence[Any], digit_ids: Sequence[int] | None,
    book: AttemptBook, counters: dict[str, int],
) -> None:
    """One unit through the attempt rule (see the module docstring). Returns when the unit has its part; raises ``UnitFailure`` after a failed attempt (after writing the cells as missing when it was the second)."""
    unit, schema = spec.unit_id, protocol.SCHEMAS[ctx.args.phase]
    if book.exhausted(unit):  # both attempts were used and no part was written (the run was stopped after the second): the cells of the unit are missing
        write_missing_unit(ctx, run, spec, items)
        counters["written_missing"] += 1
        return
    try:
        number = book.start(unit)
    except cc.AttemptLimitError:  # (the registry decides: another process used the attempts)
        write_missing_unit(ctx, run, spec, items)
        counters["written_missing"] += 1
        return
    out: protocol.UnitOutput | None = None
    table: pa.Table | None = None
    try:
        out = dev.run_unit(ctx, bh, renderer, spec, items, digit_ids)
    # an exception that leaves the protocol (rendering, parsing, scoring) fails the attempt; KeyboardInterrupt and SystemExit pass: the attempt stays started, which counts as used
    except Exception as error:  # noqa: BLE001
        outcome, info = "call_failure", {"n_rows": len(spec.rows), "n_missing": None, "detail": f"{type(error).__name__} left the protocol", "error_classes": {type(error).__name__: 1}}
    else:
        outcome, info, table = classify(out, spec, items, schema)
    if outcome == "complete":
        assert out is not None
        write_unit(run, spec, out, schema, number, outcome)
        book.finish(unit, number, outcome)
        counters["attempt_2" if number == 2 else "attempt_1"] += 1
        return
    save_failed_attempt(run.run_dir, spec, number, outcome, info, out, table)
    book.finish(unit, number, outcome)
    counters["failed_attempts"] += 1
    last = number >= MAX_ATTEMPTS
    if last:  # the second attempt failed as well: the cells of the unit are missing
        write_missing_unit(ctx, run, spec, items)
        counters["written_missing"] += 1
    raise UnitFailure(
        f"unit {unit} failed its attempt {number} of {MAX_ATTEMPTS} ({outcome}: {info.get('detail')}); its partial output is in {FAILED_DIR}/ and the run stops so that the other units keep their attempts. "
        f"Resume with --resume {run.run_dir}" + (" (this was the last attempt of the unit: its cells are written as missing and the model is not testable on them)" if last else "")
    )


# ---------------------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------------------

def finish_tables(
    ctx: dev.ScenarioContext, run: common.RunHandle, unit_ids: Sequence[str], conds: Sequence[Condition], keys: Sequence[str], frozen: FrozenFit | None,
) -> tuple[pa.Table, dict[str, Any]]:
    """Assemble the units into the table of the phase, check that every cell is there exactly once, write it with its summary. The answer phase gets z from the frozen fit; nothing is calibrated."""
    phase, cfg = ctx.args.phase, ctx.cfg
    schema = protocol.SCHEMAS[phase]
    table = run.parts.concat(unit_ids).select(schema.names).cast(schema)
    expected = len(keys) * (len(conds) if phase == "decide" else 1)
    if table.num_rows != expected:
        raise common.EntryError(f"the {phase} table has {table.num_rows} rows, expected {expected}")
    _, report = sealed.complete_table(table, schema, keys, conditions=list(conds) if phase == "decide" else None)
    if report["n_rows_added"]:
        raise common.EntryError(f"the {phase} table lacks {report['n_rows_added']} of the registered cells")
    if phase == "expert":
        result = sealed.expert_summary(table, cfg.a_pct)
    elif phase == "answer":
        assert frozen is not None
        batch = analysis.batch_z(frozen.summary, table, registered_keys=keys)
        table = sealed.add_z_column(table, batch)
        result = sealed.answer_summary(table.select(schema.names), frozen.summary, batch, frozen_source=frozen.source())
    else:
        result = summary.decide_summary(table, [c.label for c in conds])
    common.write_parquet(run.run_dir / f"{phase}.parquet", table)
    return table, result


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------

@dataclass
class Prepared:
    """Everything the checks before the pool is loaded produce."""

    entry: ModelEntry
    pool: str
    keys: list[str]
    conds: list[Condition]
    units: list[dev.UnitSpec]
    tokenizer: Any
    renderer: protocol.Renderer
    digit_ids: Sequence[int] | None
    identity: dict[str, Any]
    files: dict[str, Path]
    input_md5: dict[str, str]
    extra_manifest: dict[str, Any]
    plan: dict[str, Any]
    batch: Any = None
    report: cc.PreflightReport | None = None
    frozen: FrozenFit | None = None


def prepare(ctx: dev.ScenarioContext, hooks: ScenarioHooks) -> Prepared:
    """Every check that needs no question, in this order: the model and the phase, the pool's key list, the preflight of the shared layer, the frozen pre-registration and the unseal marker (a confirmatory run),
    the frozen fit, the source file, the chat template, the files this entry binds, the identity of a run being resumed. Nothing is consumed and no run directory is made; a failure here exits 2."""
    args, cfg = ctx.args, ctx.cfg
    phase, real = args.phase, not args.rehearsal
    entry = cfg.model(args.model, phase)
    pool = pool_of(args)
    conds = dev.condition_list(args, ctx.book)
    keys = list(ctx.split.keys(pool))
    if args.limit_items:
        keys = keys[: args.limit_items]
    batch = report = block = None
    if real:
        batch = cc.load_batch_manifest(args.batch_manifest)
        report = preflight(ctx, batch, entry)
        report.require()  # exit 2 here: nothing was loaded or consumed
        block = cc.confirm_manifest_block(batch, prereg_frozen_path=prereg_path(args))  # the frozen pre-registration must be the file the batch manifest names
        marker = paths.holdout_marker_path(ctx.results_root)
        if not marker.is_file():
            raise cc.ConfirmError(f"the unseal marker {marker} does not exist: it is created by hand, after the batches are registered")
    else:
        cc.check_label(ctx.results_root, STAGE, args.tag, prefix=REHEARSAL_PREFIX, resume_dir=args.resume)
        cc.check_model_dir(entry.path)
        if hooks.opener is None:
            common.require_gpu_memory_fraction(args)
    frozen = load_frozen_fit(args, args.model) if phase == "answer" else None
    source_sha = ctx.split.verify_input(triviaqa.INPUT_NAME, ctx.source)["sha256"]
    tokenizer = dev.load_tokenizer_for(hooks, entry)
    renderer = dev.renderer_of(ctx, entry, tokenizer)
    digit_ids = readout.resolve_digit_ids(tokenizer) if phase == "answer" else None
    fingerprints = dev.rendering_fingerprints(ctx, renderer, phase, conds, cfg.a_pct)
    units = dev.plan_units(phase, len(keys), conds, cfg.unit_items)
    identity = dev.scenario_identity(ctx, entry, phase, pool, keys, conds, units, renderer, fingerprints, digit_ids, source_sha, dev.model_fingerprint_of(entry))
    identity["calibration"] = None  # nothing is calibrated on the sealed pool
    files = bound_files(ctx, frozen)
    if real:
        bound_report = check_files_listed(batch, files)
    else:
        bound_report = {name: {"path": repo_key(path), "outside_repository": repo_key(path) is None, "md5": file_md5(path), "frozen_md5": None} for name, path in files.items()}
    confirm = {
        "batch_id": batch.batch_id if real else "rehearsal", "pool": pool, "model": args.model, "arm": phase, "rehearsal": not real, "label_prefix": TAG_PREFIX if real else REHEARSAL_PREFIX,
        "frozen_fit": None if frozen is None else frozen.as_dict(), "bound_files": bound_report, "binding": None,
    }
    identity["confirm"] = {k: confirm[k] for k in ("batch_id", "pool", "model", "arm", "rehearsal", "frozen_fit")}
    identity["confirm"]["bound_files"] = {name: v["md5"] for name, v in bound_report.items()}
    dev.check_resume_identity(args, identity)  # exit 2 BEFORE any finished unit of the run being resumed is reused, and before the pool is loaded
    input_md5 = ctx.split.manifest_md5s([pool])
    extra: dict[str, Any] = {"scenario_run": identity, "scenario_config": cfg.as_dict(), "source_sha256": source_sha, "scenario_confirm": confirm}
    if real:
        kwargs = cc.open_run_kwargs(block, extra_md5=input_md5, extra_manifest=extra)  # the block is not an extra_manifest as it is: its input_md5 would replace the whole md5 map
        input_md5, extra = kwargs["extra_md5"], kwargs["extra_manifest"]
    else:
        extra.update(cc.rehearsal_manifest_block(triviaqa.SEALED_POOL))
    if not args.resume and common.tag_in_use(ctx.results_root, STAGE, args.tag):
        raise common.EntryError(f"the label {args.tag!r} is already used in {STAGE} by {common.tag_in_use(ctx.results_root, STAGE, args.tag)}; nothing is overwritten")
    plan = common.numeric_plan(dev.backend_plan(ctx, entry, {"unit_items": cfg.unit_items, "phase": phase}))
    return Prepared(
        entry=entry, pool=pool, keys=keys, conds=conds, units=units, tokenizer=tokenizer, renderer=renderer, digit_ids=digit_ids, identity=identity, files=files, input_md5=input_md5, extra_manifest=extra,
        plan=plan, batch=batch, report=report, frozen=frozen,
    )


def load_items(ctx: dev.ScenarioContext, pre: Prepared) -> list[Any]:
    """The pool, last. A confirmatory run reads it through the access of the batch (``load_confirm_items`` checks the preflight, the identity, the key list and that the ledger holds every key under this batch);
    a rehearsal reads the development pool without any access."""
    args = ctx.args
    if pre.batch is not None:
        return cc.load_confirm_items(
            ctx.results_root, pre.batch, ctx.split, model=args.model, arm=args.phase, preflight=pre.report,
            loader=lambda access: ctx.task.load_pool(ctx.split, pre.pool, ctx.source, results_root=access.results_root, holdout_access=access),
        )
    items = list(ctx.task.load_pool(ctx.split, pre.pool, ctx.source, results_root=ctx.results_root))
    if args.limit_items:
        items = items[: args.limit_items]
    if [i.item_key for i in items] != pre.keys:
        raise common.EntryError("the items that were loaded are not the registered keys of the pool")
    ctx.split.assert_no_sealed(pre.keys, context=pre.pool)
    return items


def execute(ctx: dev.ScenarioContext, hooks: ScenarioHooks) -> dict[str, Any]:
    args, cfg = ctx.args, ctx.cfg
    phase, real = args.phase, not args.rehearsal
    pre = prepare(ctx, hooks)
    counters = {"attempt_1": 0, "attempt_2": 0, "failed_attempts": 0, "written_missing": 0, "skipped": 0}
    with dev.open_scenario_run(ctx, files=pre.files, input_md5=pre.input_md5, numeric=pre.plan, extra_manifest=pre.extra_manifest) as run:
        binding = None
        if real:  # the run directory exists now; it is bound before the pool is loaded (a new label for a bound triple was refused by the preflight)
            binding = cc.bind_run(ctx.results_root, pre.batch, model=args.model, arm=phase, run_dir=Path(run.run_dir).resolve())
            run.record.manifest["scenario_confirm"]["binding"] = dataclasses.asdict(binding)
            run.record.write_manifest()
        book = AttemptBook(run.run_dir, model=args.model, arm=phase, results_root=ctx.results_root if real else None, binding=binding)
        bh = (hooks.opener or dev.open_vllm)(ctx, pre.entry, pre.tokenizer)  # the engine first: when it cannot start (the card is shared) no question has been consumed yet
        t0 = time.time()
        try:
            run.confirm_numeric(args.backend, bh.numeric_actual())
            if hooks.model_files and not run.resumed:
                common.write_model_files(run, {"id": pre.entry.id, "path": pre.entry.path, "chat_template_kwargs": dict(pre.entry.chat_template_kwargs)}, bh)
            items = load_items(ctx, pre)
            for spec in pre.units:
                if run.parts.done(spec.unit_id):
                    counters["skipped"] += 1
                    continue
                run.check_source()
                run_one_unit(ctx, run, bh, pre.renderer, spec, items, pre.digit_ids, book, counters)
                common.log(f"{spec.unit_id}: {len(spec.rows)} items")
            wall = time.time() - t0
        finally:
            bh.close()
        unit_ids = [u.unit_id for u in pre.units]
        overview = book.overview(unit_ids)
        totals = protocol.GenStats()
        for unit_id in unit_ids:  # all units of the run, also those of an earlier session
            totals = totals.merge(protocol.GenStats(**dict(common.read_json(run.run_dir / dev.UNIT_STATS / f"{unit_id}.json")["generation"].items())))
        table, result = finish_tables(ctx, run, unit_ids, pre.conds, pre.keys, pre.frozen)
        raw = dev.assemble_raw(run.run_dir, unit_ids)
        result.update({
            "phase": phase, "model": pre.entry.id, "pool": pre.pool, "task": cfg.task, "n_items": len(items), "condition_labels": [c.label for c in pre.conds], "a_pct": pre.identity["a_pct"], "backend": args.backend,
            "batch_id": pre.extra_manifest["scenario_confirm"]["batch_id"], "rehearsal": not real, "settings": cfg.settings.as_dict(), "units": {"n": len(pre.units), **counters, "wall_s": wall},
            "generation": totals.as_dict(), "raw_outputs": raw, "attempts": overview, "exhausted_units": overview["exhausted_units"],
        })
        if not real:
            result.update(exploratory=True, development_reason=pre.extra_manifest["development_reason"])
        result = summary.jsonable(result)
        common.write_json(run.run_dir / f"{phase}_summary.json", result)
        run.record.manifest["raw_outputs"] = raw
        run.record.manifest["generation"] = totals.as_dict()
        run.record.manifest["phase_summary"] = {k: result[k] for k in ("n_registered", "n_missing", "missing_rate") if k in result}
        run.record.manifest["attempts"] = {**counters, **overview}
        run.record.write_manifest()
        for unit_id in unit_ids:  # the parts of the raw file are inside it now
            with contextlib.suppress(OSError):
                (run.run_dir / dev.RAW_PARTS / f"{unit_id}.jsonl.gz").unlink()
        with contextlib.suppress(OSError):
            (run.run_dir / dev.RAW_PARTS).rmdir()
        report = {
            "run_dir": str(run.run_dir), "phase": phase, "model": pre.entry.id, "pool": pre.pool, "batch_id": result["batch_id"], "rehearsal": not real, "n_rows": int(table.num_rows),
            "n_missing": result.get("n_missing"), "missing_rate": result.get("missing_rate"), "generation": totals.as_dict(), "raw_outputs": {k: raw[k] for k in ("md5", "n_lines")}, "units": result["units"],
            "exhausted_units": overview["exhausted_units"],
        }
        if phase == "answer":
            report.update({"z_valid": result["z_valid"], "z_invalid_reasons": result["z_invalid_reasons"], "quality_gate_pass": result["quality_gate"]["pass"], "n_with_z": result["n_with_z"]})
        if phase == "decide":
            report["n_untestable"] = result["n_untestable"]
        return report


# ---------------------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------------------

def dry_run(ctx: dev.ScenarioContext, hooks: ScenarioHooks) -> int:
    """The checks of a run that need neither a pool nor a model: the report says what a run would do and what is wrong. Nothing is loaded, consumed, bound or written (``--check-template`` loads the tokenizer)."""
    dry = common.DryRun(ctx)
    args, cfg = ctx.args, ctx.cfg
    real = not args.rehearsal
    try:
        phase = args.phase
        entry = cfg.model(args.model, phase)
        pool = pool_of(args)
        conds = dev.condition_list(args, ctx.book)
    except USER_ERRORS as error:
        dry.check("arguments and inputs", False, str(error))
        return dry.emit()
    dry.check("configuration", True, {
        "path": str(cfg.path), "md5": cfg.md5, "version": cfg.version, "revision": cfg.revision, "exploratory": cfg.exploratory, "a_pct": cfg.a_pct, "expected_md5": args.expect_config_md5,
    })
    dry.check("prompt file", True, ctx.book.describe())
    batch = None
    frozen = None
    if real:
        try:
            batch = cc.load_batch_manifest(args.batch_manifest)
            dry.check("batch manifest", True, {"path": str(batch.path), "md5": batch.md5, "batch_id": batch.batch_id, "pool": batch.pool, "models": list(batch.models), "arms": list(batch.arms)})
        except (*USER_ERRORS, OSError, ValueError, KeyError) as error:
            dry.check("batch manifest", False, f"{type(error).__name__}: {error}")
        if batch is not None:
            for check in preflight(ctx, batch, entry).checks:  # (each check of the shared layer is reported with its own outcome; none stops the others)
                dry.check(f"preflight: {check['name']}", bool(check["ok"]), check["detail"])
            try:
                block = cc.confirm_manifest_block(batch, prereg_frozen_path=prereg_path(args))
                dry.check("pre-registration", True, block["confirmatory"]["prereg_frozen"])
            except (cc.ConfirmError, OSError) as error:
                dry.check("pre-registration", False, str(error))
        else:
            dry.check("pre-registration", False, "not checked: the batch manifest cannot be read")
        marker = paths.holdout_marker_path(ctx.results_root)
        dry.note("unseal marker", {"path": str(marker), "exists": marker.is_file(), "note": "a real run needs it (it is created by hand, after the batches are registered); a dry run does not"})
    else:
        dry.check("pre-registration", True, {"used": False, "note": "a rehearsal binds no batch"})
        for name, check in (
            ("label", lambda: cc.check_label(ctx.results_root, STAGE, label_for_checks(args), prefix=REHEARSAL_PREFIX, resume_dir=args.resume)), ("model directory", lambda: cc.check_model_dir(entry.path)),
        ):
            try:
                check()
                dry.check(name, True)
            except cc.ConfirmError as error:
                dry.check(name, False, str(error))
        dry.note("rehearsal", {"pool": pool, "note": "the same code on the development pool: no access, no ledger, no registry, no binding; the label must start with " + REHEARSAL_PREFIX})
        if hooks.opener is None:
            try:
                common.require_gpu_memory_fraction(args)
                dry.check("offline vLLM memory fraction", True, {"gpu_mem_util": args.gpu_mem_util, "max_model_len": args.max_model_len})
            except common.EntryError as error:
                dry.check("offline vLLM memory fraction", False, str(error))
    if phase == "answer":
        try:
            frozen = load_frozen_fit(args, args.model)
            dry.check("frozen fit", True, frozen.as_dict())
        except USER_ERRORS as error:
            dry.check("frozen fit", False, str(error))
    dry.check("pool registry", True, {"dir": str(ctx.split.root), "version": ctx.split.version, "file_md5": ctx.split.file_md5s, "pool": {"name": pool, "n": len(ctx.split.keys(pool)), "md5": ctx.split.pool_md5(pool)}})
    try:
        record = ctx.split.verify_input(triviaqa.INPUT_NAME, ctx.source)
        dry.check("source file hash", True, {"path": str(ctx.source), "sha256": record["sha256"], "n_rows": record["n_rows"]})
    except (InputHashMismatchError, OSError) as error:
        dry.check("source file hash", False, str(error))
    try:
        files = bound_files(ctx, frozen)
        report = check_files_listed(batch, files) if batch is not None else {name: {"path": repo_key(p), "outside_repository": repo_key(p) is None, "md5": file_md5(p)} for name, p in files.items()}
        dry.check("bound files", True, report)
    except (*USER_ERRORS, OSError) as error:
        dry.check("bound files", False, str(error))
    model_dir = Path(entry.path)
    detail: dict[str, Any] = {"id": entry.id, "path": str(model_dir), "family": entry.family}
    readable = True
    if model_dir.is_dir():
        try:
            fp = model_files.model_fingerprint(model_dir)
            detail.update(fingerprint_md5=fp["md5"], declared=fp["declared"], n_weight_files=len(fp["weight_files"]), n_tokenizer_files=len(fp["tokenizer_files"]))
        except OSError as error:
            readable = False
            detail["fingerprint_error"] = f"{type(error).__name__}: {error}"
    dry.check("model directory", readable and model_dir.is_dir() and (model_dir / "config.json").is_file(), detail)
    n_items = len(ctx.split.keys(pool))
    if args.limit_items:
        n_items = min(n_items, args.limit_items)
    units = dev.plan_units(phase, n_items, conds, cfg.unit_items)
    if args.check_template:
        try:
            tokenizer = dev.load_tokenizer_for(hooks, entry)
            renderer = dev.renderer_of(ctx, entry, tokenizer)
            digit_ids = readout.resolve_digit_ids(tokenizer) if phase == "answer" else None
            fingerprints = dev.rendering_fingerprints(ctx, renderer, phase, conds, cfg.a_pct)
            dry.check("chat template", True, {
                "tool_injection": renderer.injection, "system_role": render.system_role(tokenizer, renderer.chat_kwargs), "digit_token_ids": None if digit_ids is None else list(digit_ids),
                "n_renderings": len(fingerprints), "rendering_md5": fingerprints,
            })
        except (*USER_ERRORS, KeyError, TypeError) as error:
            dry.check("chat template", False, f"{type(error).__name__}: {error}")
    if args.resume:
        used = Path(args.resume)
        dry.check("resume directory", (used / common.MANIFEST_FILE).is_file(), str(used))
    else:
        used_tag = common.tag_in_use(ctx.results_root, STAGE, args.tag or "dryrun-tag")
        dry.check("output directory", used_tag is None, {"results_root": str(ctx.results_root), "stage": STAGE, "tag": args.tag, "tag_already_used_by": used_tag})
    n_prompts = n_items * (len(conds) if phase == "decide" else (2 if phase == "answer" else 1))
    dry.note("workload", {
        "phase": phase, "model": entry.id, "pool": pool, "items": n_items, "conditions": [c.label for c in conds], "prompts": n_prompts, "units": len(units), "unit_items": cfg.unit_items,
        "max_attempts_per_unit": MAX_ATTEMPTS, "settings": cfg.settings.as_dict(), "backend": args.backend, "stage": STAGE, "rehearsal": not real,
    })
    dry.note("numeric_plan", common.numeric_plan(dev.backend_plan(ctx, entry, {"unit_items": cfg.unit_items, "phase": phase})))
    return dry.emit()


def main(argv: Sequence[str] | None = None, hooks: ScenarioHooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or ScenarioHooks()
    args = make_parser().parse_args(argv)
    t0 = time.time()
    try:
        check_arguments(args)  # the label and the combinations of arguments, before the configuration, the registry or a model is read
        ctx = build_context(args, argv=argv)
        if args.dry_run:
            return dry_run(ctx, hooks)
        outcome = execute(ctx, hooks)
    except UnitFailure as failure:
        print(f"stopped: {failure}", file=sys.stderr)
        return EXIT_UNIT_FAILED
    except USER_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except common.SourceChangedError as error:
        return common.stopped_by_source_change(error)
    outcome["wall_s"] = time.time() - t0
    print(json.dumps(common.json_safe(outcome), indent=2, sort_keys=True))
    return EXIT_NOT_TESTABLE if outcome["exhausted_units"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
