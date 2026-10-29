#!/usr/bin/env python
"""Mechanism group M of one end-side model (prompts: configs/prompts/mech_v1.yaml, nmi/mech_prompts.py).

The exact digit readout of the P-ask prompt of the group on the MMLU-Redux CALIBRATION pool, for the 43 conditions of the group: the seven variants with a level (M1, M1w, M2, M3, M0, F3, OBJ)
at the six levels x in {0, 0.1, 0.2, 0.45, 0.8, 1.5}, and M0n, which has none. The readout is that of the decision grid (``run_decisions.py``): the same back end (``common.open_backend``),
the same chat template (``render_chat`` with the ``chat_template_kwargs`` of the model: no thinking), the same single-token readout over "1" to "5", the same digit-mass gate, the same jobs
(``common.run_jobs``), one prompt per (item, condition). Only the prompt differs: rules paragraph, "Current conditions" block and, for OBJ, the objective note come from ``mech_prompts``.

Every run of this entry is development evidence: ``--pools`` takes ``mmlu_redux_cal`` only (the default; anything else exits 2), the manifest carries
``exploratory: true`` and a ``development_reason``, the stage is ``mechanism`` (``results/mechanism/<time>_<tag>/``) and ``mechanism_summary.json`` says ``exploratory: true`` as well.
The psychometric analysis refuses a decisions run whose manifest says ``exploratory: true`` before it reads a table, so such a run reaches no formal analysis.

Usage: python scripts/run_mechanism.py --model MODEL --w5-run CALIB_RUN --tag TAG [--variants M1,M0,...] [--conditions-per-unit N] [--limit-items N --pilot-rep K] [--dry-run]

Options. ``--w5-run`` (required) is the answer-phase calibration run of the same model (``run_calib --phase answer``, stage ``calib``): its run id goes into ``parent_run_ids``, the md5 of its
``calib.parquet`` into ``input_md5`` (``input.w5_calib``), and the run must have the calibration of ``--model`` for every item the group reads (the normative rates of the later analysis are
worked out from its z). ``--variants`` is a comma list of M1, M1w, M2, M3, M0, M0n, F3, OBJ (default: all; run in the canonical order whatever the order given). ``--conditions-per-unit``
cuts the flat list of conditions into units of work as ``run_decisions`` does (default 6). ``--limit-items`` with ``--pilot-rep`` is the seeded pilot subset of the items.

Output: ``mechanism.parquet`` (the columns of ``decisions.parquet``: item_key, model, pool, paraphrase, layout, framing, a_H, c0, w_displayed, kappa, D_hat_s, digit_logprobs[5], digit_mass,
gate_pass, free_token_id, free_is_digit, margin, ask, self_choice, self_correct, gt_position, grid; then ``variant`` and ``x``, NaN for M0n). ``paraphrase`` holds the prompt family
(``mech_v1``), ``layout`` ``standard``, ``framing`` ``mech``, ``grid`` ``mechanism``, ``a_H`` 0.8 and ``c0`` 0; ``w_displayed`` is ``x`` for M1, M1w, F3 and OBJ (NaN for the others),
``D_hat_s`` is D = round(100 x) for M0 (None for the others). ``mechanism_summary.json`` holds the digit-mass gate pass rate, the share of digit free tokens and the ask rate of the whole run,
per variant, per level (all variants together) and per variant and level.

Binding and resume. The manifest binds, besides the inputs of every entry (configuration, prompt file, protocol documents, split), the md5 of ``mech_v1.yaml`` (``input.mech_prompts``),
of the calibration ``calib.parquet`` (``input.w5_calib``) and of this file (``input.entry``); a resumed run must have the same (``open_run``), and the same code of ``nmi/``. The part of the identity that
the md5 values do not hold is the manifest key ``mechanism_run``: the model, the pool, the variants, the levels, the conditions with what they show, the size of a unit, the layout, a_H, c0, the delay
rule and its seconds, the prompt files, the calibration run and the item subset. ``--resume`` compares it with the call BEFORE any finished unit is reused (the unit names hold the pool, the prompt family and the chunk
number only): a difference exits 2 and touches nothing. ``--dry-run`` checks the configuration, the prompts, the conditions and the inputs, prints the plan and loads no model.
"""

from __future__ import annotations

import argparse
import json
import math
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

import _c2_common as common  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from nmi import mech_prompts, paths  # noqa: E402
from nmi.data.registry import SplitError  # noqa: E402
from nmi.prompts import PromptError, render_chat  # noqa: E402
from nmi.readout.digits import ReadoutSpec  # noqa: E402

SCRIPT = "run_mechanism"
STAGE = "mechanism"
POOL = common.MAIN_CAL_POOL  # the MMLU-Redux calibration pool: the one pool of the group (the evaluation pool has been read already and takes no diagnostics)
MECH_PROMPTS = mech_prompts.DEFAULT_MECH_PROMPTS
LAYOUT = "standard"
FRAMING = "mech"
GRID = "mechanism"
W5_STAGE = "calib"
TABLE_FILE = "mechanism.parquet"
SUMMARY_FILE = "mechanism_summary.json"
DEVELOPMENT_REASON = (
    "mechanism group M: a development experiment on the MMLU-Redux calibration pool; it enters no gate and no formal analysis"
)
# the columns of decisions.parquet (scripts/run_decisions.py DECISION_FIRST, then gt_position and grid), then the two of the group
DECISION_COLUMNS = [
    "item_key", "model", "pool", "paraphrase", "layout", "framing", "a_H", "c0", "w_displayed", "kappa", "D_hat_s", "digit_logprobs", "digit_mass", "gate_pass",
    "free_token_id", "free_is_digit", "margin", "ask", "self_choice", "self_correct", "gt_position", "grid",
]
MECH_COLUMNS = [*DECISION_COLUMNS, "variant", "x"]
META_KEYS = ("item_key", "model", "pool", "paraphrase", "layout", "framing", "a_H", "c0", "w_displayed", "kappa", "D_hat_s", "gt_position", "grid", "variant", "x")


@dataclass
class MechanismHooks:
    opener: Callable[[common.Context, str, str, ReadoutSpec], common.BackendHandle] | None = None
    model_files: bool = True


@dataclass(frozen=True)
class W5Input:
    """The answer-phase calibration run (``--w5-run``) the group is bound to: its run id (the parent), its model and the md5 of its calibration table."""

    run_dir: Path
    run_id: str
    model: str
    calib_path: Path
    calib_md5: str


def chunks(n: int, size: int) -> list[range]:
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


# ---------------------------------------------------------------------------------------
# arguments and inputs
# ---------------------------------------------------------------------------------------

def pool_list(args: argparse.Namespace) -> list[str]:
    """The one pool of the group, ``mmlu_redux_cal`` (the default). An evaluation pool, another bank, a holdout pool, several pools or a pool listed twice exit 2."""
    requested = common.csv_list(args.pools) or [POOL]
    if requested != [POOL]:
        raise common.EntryError(
            f"the mechanism group runs on the one pool {POOL} (the MMLU-Redux calibration pool: development evidence that no gate takes) and takes --pools {POOL} or none; "
            f"--pools {args.pools} was given: an evaluation pool, another pool or several pools are refused"
        )
    return common.validate_pools(requested)


def units_size(args: argparse.Namespace) -> int:
    size = args.conditions_per_unit
    if size < 1:
        raise common.EntryError(f"--conditions-per-unit must be a positive number, got {size}")
    return size


def load_mech_prompts(ctx: common.Context) -> mech_prompts.MechPromptBook:
    """``mech_v1.yaml`` on the prompt book of the configuration (``ctx.book``, the one the decision grid reads): a prompt file that is invalid, or a base file other than the one it pins, exits 2."""
    try:
        return mech_prompts.MechPromptBook.load(MECH_PROMPTS, base=ctx.book)
    except (PromptError, OSError) as error:  # (the message of ``load`` names the file)
        raise common.EntryError(str(error)) from error


def variant_list(args: argparse.Namespace, mbook: mech_prompts.MechPromptBook) -> list[str]:
    requested = common.csv_list(args.variants)
    if not requested:
        return list(mbook.variants)
    unknown = [v for v in requested if v not in mbook.variants]
    if unknown:
        raise common.EntryError(f"unknown variant(s) {unknown}; known: {list(mbook.variants)}")
    if len(set(requested)) != len(requested):
        raise common.EntryError(f"a variant is listed twice in --variants {args.variants}")
    return [v for v in mbook.variants if v in requested]  # the canonical order, whatever the order given


def condition_list(mbook: mech_prompts.MechPromptBook, variants: Sequence[str]) -> list[mech_prompts.Condition]:
    """The conditions of the chosen variants. Each is rendered once with a placeholder question BEFORE anything is loaded: a level that cannot be shown, a template that does not fill, or two
    conditions that would be read from the same prompt exit 2 here and not in the middle of a batch."""
    conds = mbook.conditions(variants)
    seen: dict[str, str] = {}
    try:
        for cond in conds:
            message = mbook.user_message(cond.variant, cond.x, "<question>", ("<option 1>", "<option 2>", "<option 3>", "<option 4>"))
            if message in seen:
                raise common.EntryError(f"the conditions {seen[message]} and {cond.label()} would be read from the same prompt")
            seen[message] = cond.label()
    except PromptError as error:
        raise common.EntryError(f"{MECH_PROMPTS.name}: {error}") from error
    return conds


def read_w5(args: argparse.Namespace) -> W5Input:
    """The answer-phase calibration run of ``--model``: a directory with a manifest that names a run of stage ``calib``, ``calib_summary.json`` of this model and ``calib.parquet``."""
    run_dir = Path(args.w5_run)
    if not run_dir.is_dir():
        raise common.EntryError(f"--w5-run {run_dir}: not a directory")
    calib = common.read_calib_run(run_dir)  # calib_summary.json (exit 2 when it is missing: the rhat phase and other runs have none)
    if calib.model != args.model:
        raise common.EntryError(f"--w5-run {run_dir} is the run of {calib.model!r}, not of {args.model!r}")
    run_id = common.parent_run_ids([("calibration run", run_dir)])[0]  # exit 2 when there is no manifest.json with a run_id
    stage = run_id.split("/", 1)[0]
    if stage != W5_STAGE:
        raise common.EntryError(f"--w5-run {run_dir} is a run of stage {stage!r}, not of {W5_STAGE!r} (the answer phase of run_calib)")
    table = run_dir / common.CALIB_TABLE
    if not table.is_file():
        raise common.EntryError(f"--w5-run {run_dir}: {common.CALIB_TABLE} is missing: is it a finished answer-phase run of run_calib?")
    return W5Input(run_dir=run_dir, run_id=run_id, model=calib.model, calib_path=table, calib_md5=common.file_md5(table))


def check_w5_coverage(w5: W5Input, pool: str, item_keys: Sequence[str]) -> None:
    """The calibration table must hold the calibration of every item of the run in ``pool`` (only the item keys and the pool names are read)."""
    names = set(pq.read_schema(w5.calib_path).names)
    if not {"item_key", "pool"} <= names:
        raise common.EntryError(f"--w5-run {w5.run_dir}: {common.CALIB_TABLE} has no columns item_key and pool")
    table = pq.read_table(w5.calib_path, columns=["item_key", "pool"])
    have = {k for k, p in zip(table.column("item_key").to_pylist(), table.column("pool").to_pylist(), strict=True) if p == pool}
    missing = sum(1 for k in item_keys if k not in have)
    if missing:
        raise common.EntryError(
            f"--w5-run {w5.run_dir}: {common.CALIB_TABLE} holds no calibration of {pool} for {missing} of the {len(item_keys)} items of this run "
            "(the calibration run must cover the items the group reads)"
        )


# ---------------------------------------------------------------------------------------
# identity of a run
# ---------------------------------------------------------------------------------------

def condition_record(mbook: mech_prompts.MechPromptBook, cond: mech_prompts.Condition) -> dict[str, Any]:
    """A condition with what its prompt shows: the level, the waiting cost or deduction (None where it shows none) and the delay in seconds (M0 only)."""
    w = mbook.w_displayed(cond.variant, cond.x)
    return {"variant": cond.variant, "x": cond.x, "w_displayed": None if math.isnan(w) else w, "D_hat_s": mbook.delay_shown(cond.variant, cond.x)}


def mechanism_identity(
    args: argparse.Namespace, mbook: mech_prompts.MechPromptBook, variants: Sequence[str], conds: Sequence[mech_prompts.Condition], w5: W5Input, pool: str, size: int,
) -> dict[str, Any]:
    """What a run of the group is, beyond what the md5 values of the manifest hold: the manifest keeps it (``mechanism_run``) and a resumed run must have the same
    (``check_resume_identity``). The unit names hold the pool, the prompt family and the chunk number only."""
    return {
        "model": args.model, "pool": pool, "layout": LAYOUT, "framing": FRAMING, "variants": list(variants), "levels": list(mbook.levels), "a_H": mbook.a_H, "c0": mbook.c0,
        "delay_rule": mech_prompts.DELAY_RULE, "delay_seconds": {f"{x:.2f}": mbook.delay_seconds(x) for x in mbook.levels},
        "conditions": [condition_record(mbook, c) for c in conds], "conditions_per_unit": size,
        "mech_prompts": {"version": mbook.version, "md5": mbook.md5, "base_prompts_md5": mbook.base.md5},
        "w5": {"run_id": w5.run_id, "calib_parquet_md5": w5.calib_md5},
        "items": {"limit_items": args.limit_items, "pilot_rep": args.pilot_rep},
    }


def identity_differences(recorded: dict[str, Any], identity: dict[str, Any]) -> list[str]:
    """The ways in which two identities differ, one line each (the long list of conditions is summarised, as ``run_decisions`` does)."""
    before, after = dict(recorded), json.loads(json.dumps(identity))
    cond_before, cond_after = before.pop("conditions", None), after.pop("conditions", None)
    differences = common.numeric_differences(before, after)
    if cond_before != cond_after:
        first = 0
        if isinstance(cond_before, list) and isinstance(cond_after, list):
            first = next((i for i, (x, y) in enumerate(zip(cond_before, cond_after, strict=False)) if x != y), min(len(cond_before), len(cond_after)))
            differences.append(f"conditions: {len(cond_before)} then, {len(cond_after)} now, the first difference at position {first}")
        else:
            differences.append("conditions: not recorded")
    return differences


def check_resume_identity(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    """The unit names hold the pool, the prompt family and the chunk number only, and ``run_jobs`` skips a unit whose file is there. A run that is resumed with other variants, levels, inputs or items
    would take the chunks written by the first execution for its own. So the identity of the call is compared with that of the run being resumed BEFORE any chunk is reused (and before the manifest of that run is touched):
    ``EntryError`` (exit 2) when they differ. Nothing is checked without ``--resume`` or when the run being resumed has no ``manifest.json`` (``open_run`` says so itself)."""
    resume = getattr(args, "resume", None)
    if not resume:
        return
    path = Path(resume) / common.MANIFEST_FILE
    if not path.is_file():
        return
    manifest = common.read_json(path)
    recorded = manifest.get("mechanism_run") if isinstance(manifest, dict) else None
    if not isinstance(recorded, dict):
        raise common.EntryError(f"--resume {resume}: that is not a run of {SCRIPT} (its manifest has no mechanism_run, the record of the variants, levels and inputs it was started with)")
    differences = identity_differences(recorded, identity)
    if differences:
        raise common.EntryError(
            f"--resume {resume}: the variants, levels, layout, a_H, c0, delay rule, prompt files, calibration run and items of the run must be those it was started with (a finished unit of another condition set "
            "would be reused as if it were this run's): " + "; ".join(differences)
        )


# ---------------------------------------------------------------------------------------
# the jobs and the summary
# ---------------------------------------------------------------------------------------

def mech_texts(mbook: mech_prompts.MechPromptBook, tokenizer: Any, items: Sequence[Any], chat_kwargs: dict[str, Any], cond: mech_prompts.Condition) -> list[str]:
    """P-ask prompts of one condition for every item, through the chat template (no assistant prefix, as the P-ask of the decision grid)."""
    return [render_chat(tokenizer, mbook.user_message(cond.variant, cond.x, i.question, i.choices), chat_kwargs) for i in items]


def condition_job(
    bh: common.BackendHandle, mbook: mech_prompts.MechPromptBook, items: Sequence[Any], pool: str, conds: Sequence[mech_prompts.Condition], chat_kwargs: dict[str, Any], model: str,
):
    def build() -> tuple[list[str], dict[str, Any]]:
        texts: list[str] = []
        meta: dict[str, list[Any]] = {k: [] for k in META_KEYS}
        n = len(items)
        for cond in conds:
            texts += mech_texts(mbook, bh.tokenizer, items, chat_kwargs, cond)
            meta["item_key"] += [i.item_key for i in items]
            meta["model"] += [model] * n
            meta["pool"] += [pool] * n
            meta["paraphrase"] += [mbook.version] * n
            meta["layout"] += [LAYOUT] * n
            meta["framing"] += [FRAMING] * n
            meta["a_H"] += [mbook.a_H] * n
            meta["c0"] += [mbook.c0] * n
            meta["w_displayed"] += [mbook.w_displayed(cond.variant, cond.x)] * n
            meta["kappa"] += [None] * n
            meta["D_hat_s"] += [mbook.delay_shown(cond.variant, cond.x)] * n
            meta["gt_position"] += [i.gt_position for i in items]
            meta["grid"] += [GRID] * n
            meta["variant"] += [cond.variant] * n
            meta["x"] += [math.nan if cond.x is None else cond.x] * n
        cols: dict[str, Any] = dict(meta)
        for name in ("a_H", "c0", "w_displayed", "x"):
            cols[name] = np.asarray(meta[name], dtype=np.float64)
        cols["gt_position"] = np.asarray(meta["gt_position"], dtype=np.int64)
        return texts, cols

    return build


def _rates(mask: np.ndarray, gate: np.ndarray, free: np.ndarray, ask: np.ndarray, mass: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(mask.sum()), "gate_pass_rate": float(gate[mask].mean()), "free_digit_rate": float(free[mask].mean()), "ask_rate": float(ask[mask].mean()),
        "mean_digit_mass": float(mass[mask].mean()),
    }


def summarize_mechanism(table: pa.Table, conds: Sequence[mech_prompts.Condition]) -> dict[str, Any]:
    """The digit-mass gate pass rate, the share of digit free tokens and the ask rate of the whole run, per variant, per level (the variants with a level together) and per variant and level.
    These are totals of the run: the verdict on a variant's gate is the analysis's."""
    variant = np.asarray(table.column("variant").to_pylist())
    x = table.column("x").to_numpy(zero_copy_only=False).astype(np.float64)
    gate = table.column("gate_pass").to_numpy(zero_copy_only=False).astype(bool)
    free = table.column("free_is_digit").to_numpy(zero_copy_only=False).astype(bool)
    ask = table.column("ask").to_numpy(zero_copy_only=False).astype(bool)
    mass = table.column("digit_mass").to_numpy(zero_copy_only=False)

    def at(level: float | None) -> np.ndarray:
        return np.isnan(x) if level is None else np.isclose(x, level, rtol=0.0, atol=1e-9)

    def key(level: float | None) -> str:
        return "none" if level is None else f"{level:.2f}"

    everything = np.ones(len(variant), dtype=bool)
    out: dict[str, Any] = {"n_rows": int(table.num_rows), **_rates(everything, gate, free, ask, mass)}
    out["by_variant"], out["by_level"], out["by_variant_level"] = {}, {}, {}
    for v in dict.fromkeys(c.variant for c in conds):
        out["by_variant"][v] = _rates(variant == v, gate, free, ask, mass)
        out["by_variant_level"][v] = {key(c.x): _rates((variant == v) & at(c.x), gate, free, ask, mass) for c in conds if c.variant == v}
    for level in dict.fromkeys(c.x for c in conds if c.x is not None):
        out["by_level"][key(level)] = _rates(at(level), gate, free, ask, mass)
    return out


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------

def execute(ctx: common.Context, args: argparse.Namespace, hooks: MechanismHooks) -> dict[str, Any]:
    cfg = ctx.cfg
    entry = common.device_entry(cfg, args.model)
    spec = ReadoutSpec.from_config(cfg)
    size = units_size(args)
    pool = pool_list(args)[0]
    mbook = load_mech_prompts(ctx)
    variants = variant_list(args, mbook)
    conds = condition_list(mbook, variants)
    w5 = read_w5(args)
    chat_kwargs = common.chat_kwargs_of(entry)
    items = common.load_pool_items(ctx, pool, limit=args.limit_items, pilot_rep=args.pilot_rep)
    check_w5_coverage(w5, pool, [i.item_key for i in items])
    identity = mechanism_identity(args, mbook, variants, conds, w5, pool, size)
    check_resume_identity(args, identity)  # exit 2 BEFORE any finished unit of the run being resumed is reused
    keys = [ctx.ns.development_pilot_key(args.pilot_rep)] if args.limit_items else []
    extra_md5 = {"input.mech_prompts": mbook.md5, "input.w5_calib": w5.calib_md5, "input.entry": common.file_md5(Path(__file__).resolve())}
    parents = [w5.run_id]
    opener = hooks.opener or (lambda c, kind, model, sp: common.open_backend(c, kind, model, sp))
    plan = common.numeric_plan(common.readout_backend_plan(args, unit={"conditions_per_unit": size}))
    extra_manifest: dict[str, Any] = {"exploratory": True, "development_reason": DEVELOPMENT_REASON, "mechanism_run": identity}
    with common.open_run(ctx, pools=[pool], seed_keys=keys, parents=parents, extra_md5=extra_md5, numeric=plan, extra_manifest=extra_manifest) as run:
        bh = opener(ctx, args.backend, args.model, spec)
        try:
            run.confirm_numeric(args.backend, bh.numeric_actual())
            if hooks.model_files and not run.resumed:
                common.write_model_files(run, entry, bh)
            jobs, unit_ids = [], []
            for gi, rows in enumerate(chunks(len(conds), size)):
                unit = f"{pool}__{mbook.version}__g{gi:02d}"
                unit_ids.append(unit)
                jobs.append(common.Job(unit, condition_job(bh, mbook, items, pool, [conds[i] for i in rows], chat_kwargs, args.model)))
            stats = common.run_jobs(run, bh.backend, jobs, label="mechanism ")
        finally:
            bh.close()
        table = run.parts.concat(unit_ids)
        choice = table.column("self_choice").to_numpy(zero_copy_only=False)
        gt = table.column("gt_position").to_numpy(zero_copy_only=False)
        table = table.append_column("self_correct", pa.array(np.asarray(choice == gt, dtype=bool)))
        table = table.select(MECH_COLUMNS)
        common.write_parquet(run.run_dir / TABLE_FILE, table)
        summary = summarize_mechanism(table, conds)
        summary.update({
            "model": args.model, "backend": args.backend, "pool": pool, "layout": LAYOUT, "variants": variants, "n_conditions": len(conds), "min_gate_pass_rate": spec.min_gate_pass_rate,
            "mech_prompts": identity["mech_prompts"], "w5": identity["w5"], "jobs": stats, "conditions": identity["conditions"],
            "exploratory": True, "development_reason": DEVELOPMENT_REASON,
        })
        if args.limit_items:
            summary["pilot"] = {"limit_items": args.limit_items, "pilot_rep": args.pilot_rep}
        common.write_json(run.run_dir / SUMMARY_FILE, summary)
        return {"run_dir": str(run.run_dir), "model": args.model, "n_rows": int(table.num_rows), "gate_pass_rate": summary["gate_pass_rate"], "ask_rate": summary["ask_rate"]}


def dry_run(ctx: common.Context, args: argparse.Namespace) -> int:
    dry = common.DryRun(ctx)
    try:
        size = units_size(args)
        pools = [pool_list(args)[0]]
        mbook = load_mech_prompts(ctx)
        variants = variant_list(args, mbook)
        conds = condition_list(mbook, variants)
        w5 = read_w5(args)
    except common.EntryError as error:
        dry.check("arguments and inputs", False, str(error))
        return dry.emit()
    pool = pools[0]
    keys = [ctx.ns.development_pilot_key(args.pilot_rep)] if args.limit_items and args.pilot_rep else []
    common.dry_run_basics(dry, pools=pools, models=[("device", args.model)], seed_keys=keys)
    dry.check("parent runs: the calibration run has a manifest with a run id", True, [w5.run_id])
    dry.note("parent_run_ids", [w5.run_id])
    first_positive = lambda v: next((x for x in mbook.levels_of(v) if x), mbook.levels_of(v)[0])  # noqa: E731
    dry.check("mechanism prompts", True, {
        "path": str(mbook.path), "md5": mbook.md5, "version": mbook.version, "base_prompts": {"path": str(mbook.base.path), "md5": mbook.base.md5}, "rules_m": mbook.rules_m,
        "conditions_blocks": {v: mbook.conditions_block(v, first_positive(v)) for v in variants}, "levels": list(mbook.levels), "a_H": mbook.a_H, "c0": mbook.c0,
    })
    try:
        reg_keys = list(ctx.registry().keys(pool))
        wanted = [reg_keys[i] for i in common.pilot_subset(ctx, len(reg_keys), args.limit_items, args.pilot_rep)]
        check_w5_coverage(w5, pool, wanted)
        dry.check("calibration run: the calibration of every item of the run", True, {"calib_parquet_md5": w5.calib_md5, "items": len(wanted)})
    except (common.EntryError, SplitError) as error:
        dry.check("calibration run: the calibration of every item of the run", False, str(error))
    sizes = common.pool_sizes(ctx, pools, args.limit_items)
    dry.note("workload", {
        "pool": sizes, "variants": variants, "conditions": len(conds), "mechanism_prompts": sizes[pool] * len(conds), "units": len(chunks(len(conds), size)),
        "conditions_per_unit": size, "backend": args.backend, "stage": STAGE,
    })
    dry.note("exploratory", {"exploratory": True, "development_reason": DEVELOPMENT_REASON, "pools": pools, "layout": LAYOUT})
    if getattr(args, "resume", None):  # the variants, levels and inputs of the run being resumed
        try:
            check_resume_identity(args, mechanism_identity(args, mbook, variants, conds, w5, pool, size))
            dry.check("resume: the run being resumed has these variants, levels and inputs", True, {"resume": args.resume})
        except common.EntryError as error:
            dry.check("resume: the run being resumed has these variants, levels and inputs", False, str(error))
    return dry.emit()


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mechanism group M: the P-ask variants M1, M1w, M2, M3, M0, M0n, F3 and OBJ of one end-side model on the calibration pool.")
    common.add_common_arguments(p)
    common.add_readout_arguments(p)
    p.add_argument("--w5-run", required=True, help="answer-phase calibration run directory of the same model (run_calib --phase answer): its run id is the parent of this run, the md5 of its calib.parquet is bound")
    p.add_argument("--pools", default=None, help=f"only {POOL} (the default); anything else exits 2")
    p.add_argument("--variants", default=None, help="comma list of M1, M1w, M2, M3, M0, M0n, F3, OBJ (default: all eight; run in the order of the design)")
    p.add_argument("--conditions-per-unit", type=int, default=6, help="conditions in one unit of work (default 6: 6,000 prompts of 1,000 items)")
    return p


def main(argv: Sequence[str] | None = None, hooks: MechanismHooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or MechanismHooks()
    args = make_parser().parse_args(argv)
    t0 = time.time()
    try:
        ctx = common.build_context(args, script=SCRIPT, stage=STAGE, argv=argv)
        if args.dry_run:
            return dry_run(ctx, args)
        outcome = execute(ctx, args, hooks)
    except (common.EntryError, SplitError, paths.PathPolicyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except common.SourceChangedError as error:
        return common.stopped_by_source_change(error)
    outcome["wall_s"] = time.time() - t0
    print(json.dumps(common.json_safe(outcome), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
