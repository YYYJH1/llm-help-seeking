#!/usr/bin/env python
"""Temperature scaling, ECE, AUROC (``--phase answer``) and the correctness map r_hat in both versions (``--phase rhat``) of one end-side model.

Usage: python scripts/run_calib.py --phase answer --model MODEL --tag TAG [--pools POOLS] ...
       python scripts/run_calib.py --phase rhat --model MODEL --tag TAG --calib-run RUN_DIR --screen-run RUN_DIR [RUN_DIR ...]

``--phase answer`` (needs a readout back end: offline vLLM by default)
    P-answer (Kumaran's prompt, through the model's chat template with the configured ``chat_template_kwargs`` and, after the start of the assistant turn, the assistant prefix of the
    prompt file: ``common.answer_texts``; with the prefix ``calib.parquet`` also has ``free_token_logprob``, the log-probability of the
    most likely first token over the whole vocabulary, from the same forward pass as the digit mass; ``free_token_id`` is that token) on the calibration and
    evaluation pools of both banks. The digit log-probabilities of the options 1-4 are the raw option log-probabilities. Per bank the
    temperature is fitted on the calibration pool (minimum negative log-likelihood of the correct option, ``fit_temperature``) and
    applied to the calibration and the evaluation pool: z is the largest scaled option probability. Output: ``calib.parquet`` (item_key,
    model, pool, split, p_options_raw[4], temperature, z, pred, correct, then digit_logprobs[5], digit_mass, gate_pass, free_token_id,
    free_is_digit, gt_position) and ``calib_summary.json`` (temperature fit, accuracy, ECE with ``calibration.ece_bins`` bins, AUROC per
    pool, and the P-answer digit-mass gate pass rate). The K0.1 gate takes the P-answer trials of the
    MMLU-Redux evaluation pool alone, one per item: ``k0_1_p_answer`` names that pool and gives its rate and count, and lists every pool of
    the run (the calibration pools, SimpleQA) with its own rate for diagnosis; those do not count.

``--phase rhat`` (CPU only; needs the answer-phase run and the prompt-screening run)
    r_hat(z, w) is fitted on the calibration pool from the P-ask self-answer correctness at a_H = 0.8, c0 = 0 and the fitted w of the
    configuration, for the primary paraphrase of the screening. Those trials are the ones the screening already ran (its diagnostic
    conditions are exactly these four), so this phase reads them from the screening run instead of running the model again. Both versions are
    fitted (``CalibrationSamples.r_hats``): the logistic regression (main) and the isotonic family (sensitivity), with the shape check of
    the logistic fit and the normative thresholds at the fitted conditions as a preview. The long table ``calibration_samples.parquet``
    (bank, pool, item_key, w, z, self_correct) is for inspection and plots (the bootstrap analysis builds its samples from the screening trials itself).
    ``--screen-run`` may name several screening runs: the first one gives the primary paraphrase, the trials of all of them are used (the
    main bank is screened in the first run, the SimpleQA calibration pool, for its exploratory r_hat, in a run of its own); a pool may be in one run only.

The upstream runs of a run are named as its parents (``common.parent_run_ids``): the answer-phase run and the screening runs for ``--phase rhat``. An input directory without a
readable ``manifest.json`` that names a run exits 2 before a run directory is made, and the dry run reports it.
"""

from __future__ import annotations

import argparse
import json
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

from nmi import paths  # noqa: E402
from nmi.data.registry import POOL_BANK, SplitError  # noqa: E402
from nmi.measure import calibration as cal  # noqa: E402
from nmi.measure.calibration import CalibrationSamples  # noqa: E402
from nmi.readout.digits import ReadoutSpec  # noqa: E402

SCRIPT = "run_calib"
STAGE = "calib"
K0_1_POOL = common.MAIN_EVAL_POOL  # the one pool whose P-answer trials make the denominator of K0.1


@dataclass
class CalibHooks:
    """Replaceable parts, for the tests: a function that opens the readout back end, and the model-hash switch."""

    opener: Callable[[common.Context, str, str, ReadoutSpec], common.BackendHandle] | None = None
    model_files: bool = True


def pool_list(args: argparse.Namespace) -> list[str]:
    pools = common.validate_pools(common.csv_list(args.pools) or [*common.CAL_POOLS, *common.EVAL_POOLS])
    banks = {POOL_BANK[p] for p in pools}
    for bank in banks:
        if not any(common.bank_of(p) == bank and common.split_of(p) == "cal" for p in pools):
            raise common.EntryError(f"the temperature of bank {bank!r} is fitted on its calibration pool: add it to --pools")
    return pools


# ---------------------------------------------------------------------------------------
# answer phase
# ---------------------------------------------------------------------------------------

def p_answer_gate_report(pools: np.ndarray, gate: np.ndarray) -> dict[str, Any]:
    """The P-answer digit-mass gate of the run, by pool, with the one pool that K0.1 counts named.

    K0.1 takes the P-answer trials of the MMLU-Redux evaluation pool, one per item: ``gate_pass_rate`` and ``n`` are that pool's
    (None and 0 when the run did not read it). ``by_pool`` has the rate and the count of every pool of the run; the calibration pools and the SimpleQA pools are there for
    diagnosis and do not count. The verdict is not given here: it needs the P-ask part too."""
    by_pool: dict[str, Any] = {}
    for pool in sorted(set(pools.tolist())):
        m = pools == pool
        by_pool[pool] = {"gate_pass_rate": float(gate[m].mean()), "n": int(m.sum()), "counts_in_k0_1": pool == K0_1_POOL}
    main = by_pool.get(K0_1_POOL)
    return {
        "pool": K0_1_POOL, "gate_pass_rate": None if main is None else main["gate_pass_rate"], "n": 0 if main is None else main["n"], "by_pool": by_pool,
        "note": f"digit mass >= readout.digit_mass_gate over the P-answer trials of {K0_1_POOL}, one per item; the other pools are listed for diagnosis and do not count; the P-ask part of K0.1 "
                "is the W7 trials of that pool (W6 and the other pools do not count there either) and the verdict is run_psychometrics'",
    }


def assemble_calibration(table: pa.Table, model: str, ece_bins: int) -> tuple[pa.Table, dict[str, Any]]:
    """The ``calib`` table and its summary from the P-answer readouts of every pool (``table``: one row per item, in pool order)."""
    pools = np.asarray(table.column("pool").to_pylist())
    item_key = table.column("item_key").to_pylist()
    log_p = common.column_array(table, "digit_logprobs")
    gt = table.column("gt_position").to_numpy(zero_copy_only=False).astype(np.int64)
    temperature = np.zeros(len(gt))
    z = np.zeros(len(gt))
    pred = np.zeros(len(gt), dtype=np.int64)
    correct = np.zeros(len(gt), dtype=bool)
    banks: dict[str, Any] = {}
    for bank in sorted({POOL_BANK[p] for p in set(pools)}):
        in_bank = np.asarray([POOL_BANK[p] == bank for p in pools])
        cal_mask = in_bank & np.asarray([common.split_of(p) == "cal" for p in pools])
        fit = cal.fit_temperature(log_p[cal_mask, :4], gt[cal_mask] - 1)
        cols = cal.calibration_columns(log_p[in_bank, :4], gt[in_bank] - 1, fit.temperature)
        temperature[in_bank] = fit.temperature
        z[in_bank], pred[in_bank], correct[in_bank] = cols["z"], cols["pred"], cols["correct"]
        entry: dict[str, Any] = {"temperature": fit.to_dict(), "pools": {}}
        for pool in sorted({p for p in pools[in_bank]}):
            m = pools == pool
            entry["pools"][pool] = {
                "report": cal.confidence_report(log_p[m, :4], gt[m] - 1, fit.temperature, n_bins=ece_bins),
                "gate_pass_rate": float(table.column("gate_pass").to_numpy(zero_copy_only=False)[m].mean()),
                "free_digit_rate": float(table.column("free_is_digit").to_numpy(zero_copy_only=False)[m].mean()),
                "mean_digit_mass": float(table.column("digit_mass").to_numpy(zero_copy_only=False)[m].mean()),
            }
        banks[bank] = entry
    columns: dict[str, Any] = {
        "item_key": item_key, "model": [model] * len(gt), "pool": pools.tolist(), "split": [common.split_of(p) for p in pools],
        "p_options_raw": log_p[:, :4], "temperature": temperature, "z": z, "pred": pred, "correct": correct,
        "digit_logprobs": log_p, "digit_mass": table.column("digit_mass").to_numpy(zero_copy_only=False),
        "gate_pass": table.column("gate_pass").to_numpy(zero_copy_only=False), "free_token_id": table.column("free_token_id").to_pylist(),
        "free_is_digit": table.column("free_is_digit").to_numpy(zero_copy_only=False), "gt_position": gt,
    }
    if "free_token_logprob" in table.column_names:  # the log-probability of the most likely first token, from the forward pass of the digits
        columns["free_token_logprob"] = table.column("free_token_logprob").to_numpy(zero_copy_only=False)
    out = common.table_from_columns(columns)
    gate = table.column("gate_pass").to_numpy(zero_copy_only=False).astype(bool)
    summary = {
        "model": model, "banks": banks, "ece_bins": ece_bins,
        "k0_1_p_answer": p_answer_gate_report(pools, gate),
    }
    return out, summary


def execute_answer(ctx: common.Context, args: argparse.Namespace, hooks: CalibHooks) -> dict[str, Any]:
    cfg = ctx.cfg
    entry = common.device_entry(cfg, args.model)
    spec = ReadoutSpec.from_config(cfg)
    pools = pool_list(args)
    chat_kwargs = common.chat_kwargs_of(entry)
    keys = [ctx.ns.development_pilot_key(args.pilot_rep)] if args.limit_items else []
    items_by_pool = {p: common.load_pool_items(ctx, p, limit=args.limit_items, pilot_rep=args.pilot_rep) for p in pools}
    opener = hooks.opener or (lambda c, kind, model, sp: common.open_backend(c, kind, model, sp))
    plan = common.numeric_plan(common.readout_backend_plan(args, unit={"chunk_size": args.chunk_size}))
    with common.open_run(ctx, pools=pools, seed_keys=keys, numeric=plan) as run:
        bh = opener(ctx, args.backend, args.model, spec)
        try:
            run.confirm_numeric(args.backend, bh.numeric_actual())
            if hooks.model_files and not run.resumed:
                common.write_model_files(run, entry, bh)
            jobs, unit_ids = [], []
            for pool in pools:
                items = items_by_pool[pool]
                for ci, rows in enumerate(_chunks(len(items), args.chunk_size)):
                    unit = f"{pool}__{ci:04d}"
                    unit_ids.append(unit)
                    part = [items[i] for i in rows]
                    jobs.append(common.Job(unit, _answer_job(ctx, bh, part, pool, chat_kwargs, args.model), free_logprob=bool(ctx.book.assistant_prefix)))
            stats = common.run_jobs(run, bh.backend, jobs, label="answer ")
        finally:
            bh.close()
        table, summary = assemble_calibration(run.parts.concat(unit_ids), args.model, int(cfg["calibration"]["ece_bins"]))
        common.write_parquet(run.run_dir / common.CALIB_TABLE, table)
        summary.update({"backend": args.backend, "jobs": stats, "pools": pools})
        if args.limit_items:
            summary["pilot"] = {"limit_items": args.limit_items, "pilot_rep": args.pilot_rep}
        common.write_json(run.run_dir / common.CALIB_SUMMARY, summary)
        return {"run_dir": str(run.run_dir), "phase": "answer", "model": args.model,
                "temperature": {b: v["temperature"]["temperature"] for b, v in summary["banks"].items()}, "k0_1_p_answer": summary["k0_1_p_answer"]}


def _chunks(n: int, size: int) -> list[range]:
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


def _answer_job(ctx: common.Context, bh: common.BackendHandle, items: Sequence[Any], pool: str, chat_kwargs: dict[str, Any], model: str):
    def build() -> tuple[list[str], dict[str, Any]]:
        texts = common.answer_texts(ctx.book, bh.tokenizer, items, chat_kwargs)
        meta = {"item_key": [i.item_key for i in items], "model": [model] * len(items), "pool": [pool] * len(items),
                "gt_position": np.asarray([i.gt_position for i in items], dtype=np.int64)}
        return texts, meta

    return build


# ---------------------------------------------------------------------------------------
# rhat phase
# ---------------------------------------------------------------------------------------

def _clean(obj: Any) -> Any:
    return common.json_safe(obj)


def fit_rhat_for_bank(
    bank: str, pool: str, z_by_item: dict[str, float], trials: pa.Table, paraphrase: str, spec: cal.RHatSpec, fitted_w: Sequence[float],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Both r_hat versions of one bank from the screening trials of the primary paraphrase (rows: item order of the calibration table)."""
    cols = {name: trials.column(name).to_pylist() for name in ("pool", "paraphrase", "layout", "framing", "item_key", "self_correct")}
    a_H, c0, w = (common.column_array(trials, n) for n in ("a_H", "c0", "w_displayed"))
    base = np.asarray([(p == pool and q == paraphrase and lay == "standard" and f == "f2") for p, q, lay, f in
                       zip(cols["pool"], cols["paraphrase"], cols["layout"], cols["framing"], strict=True)])
    order = list(z_by_item)
    z = np.asarray([z_by_item[k] for k in order], dtype=np.float64)
    self_correct: dict[float, np.ndarray] = {}
    for fw in fitted_w:
        m = base & (np.abs(a_H - spec.fixed_a_H) < 1e-9) & (np.abs(c0 - spec.fixed_c0) < 1e-9) & (np.abs(w - fw) < 1e-9)
        by_key = {k: bool(c) for k, c, keep in zip(cols["item_key"], cols["self_correct"], m, strict=True) if keep}
        if set(by_key) != set(order):
            raise common.EntryError(f"{pool}: the screening trials at w = {fw:g} cover {len(by_key)} of the {len(order)} calibration items")
        self_correct[fw] = np.asarray([by_key[k] for k in order], dtype=np.float64)
    samples = CalibrationSamples(z=z, self_correct=self_correct)
    fits = samples.r_hats(spec)
    logistic, isotonic = fits["logistic"], fits["isotonic"]
    shape = samples.shape_check(logistic)
    preview: dict[str, Any] = {}
    for fw in fitted_w:
        preview[f"{fw:g}"] = {}
        for name, fit in (("logistic", logistic), ("isotonic", isotonic)):
            try:
                preview[f"{fw:g}"][name] = fit.normative(spec.fixed_a_H, spec.fixed_c0, fw).to_dict()
            except cal.CalibrationError as error:  # e.g. no common support between two isotonic curves
                preview[f"{fw:g}"][name] = {"status": "error", "reason": str(error)}
    entry = {
        "pool": pool, "n_items": len(order), "paraphrase": paraphrase, "fitted_w": list(fitted_w), "main_method": spec.main_method, "sensitivity_method": spec.sensitivity_method,
        "logistic": logistic.to_dict(), "shape_check": shape.to_dict(),
        "isotonic": {
            "r_w_test": isotonic.r_w_test.to_dict(), "r_w_is_zero": isotonic.r_w_is_zero,
            "curves": {f"{w_:g}": {"z_knots": c.z_knots.tolist(), "r_knots": c.r_knots.tolist()} for w_, c in zip(isotonic.fitted_w, isotonic.curves, strict=True)},
        },
        "preview_normative_at_fitted_conditions": preview,
        "preview_note": "descriptive: theta* at (a_H, c0) = (0.8, 0) of the fit and each fitted w; the rationality ratio at the operating a_H is W8 (run_psychometrics)",
    }
    long = {"bank": [bank] * (len(order) * len(fitted_w)), "pool": [pool] * (len(order) * len(fitted_w)), "item_key": [k for _ in fitted_w for k in order],
            "w": [float(fw) for fw in fitted_w for _ in order], "z": [float(v) for _ in fitted_w for v in z],
            "self_correct": [bool(v) for fw in fitted_w for v in self_correct[fw]]}
    return entry, long


def execute_rhat(ctx: common.Context, args: argparse.Namespace) -> dict[str, Any]:
    cfg = ctx.cfg
    if not (args.calib_run and args.screen_run):
        raise common.EntryError("--phase rhat needs --calib-run and --screen-run")
    screen_runs = [Path(d) for d in args.screen_run]
    calib = common.read_calib_run(args.calib_run)
    screening = common.read_screening_run(screen_runs[0])
    paraphrase = args.paraphrase or screening.primary
    if paraphrase is None:
        raise common.EntryError(f"{screen_runs[0]}: no paraphrase was retained; r_hat needs one (--paraphrase ID overrides)")
    if calib.model != args.model:
        raise common.EntryError(f"{args.calib_run} is the run of {calib.model!r}, not of {args.model!r}")
    pools = common.validate_pools(common.csv_list(args.pools) or list(common.CAL_POOLS))
    if any(common.split_of(p) != "cal" for p in pools):
        raise common.EntryError("r_hat is fitted on calibration pools only")
    spec = cal.RHatSpec.from_config(cfg)
    fitted_w = list(spec.fitted_w)
    tables = []
    seen_pools: dict[str, Path] = {}
    for d in screen_runs:
        if common.read_json(d / common.SCREENING).get("model") != args.model:
            raise common.EntryError(f"{d} is not a screening run of {args.model!r}")
        table = common.read_parquet(d / "screening_trials.parquet")
        for pool_in_run in sorted(set(table.column("pool").to_pylist())):
            if pool_in_run in seen_pools:
                raise common.EntryError(f"the trials of {pool_in_run} are in two screening runs ({seen_pools[pool_in_run]} and {d})")
            seen_pools[pool_in_run] = d
        tables.append(table)
    trials = pa.concat_tables(tables, promote_options="permissive")
    parents = common.parent_run_ids([("answer-phase run", args.calib_run), *[("screening run", d) for d in screen_runs]])
    with common.open_run(ctx, pools=pools, seed_keys=[], parents=parents) as run:
        banks: dict[str, Any] = {}
        long_parts: list[dict[str, list[Any]]] = []
        for pool in pools:
            bank = common.bank_of(pool)
            if pool not in set(trials.column("pool").to_pylist()):
                banks[bank] = {"skipped": f"the screening run has no trials of {pool}"}
                continue
            z_by_item = calib.z_by_item(pool)
            entry, long = fit_rhat_for_bank(bank, pool, z_by_item, trials, paraphrase, spec, fitted_w)
            banks[bank] = entry
            long_parts.append(long)
        if not long_parts:
            raise common.EntryError("no bank could be fitted")
        merged = {k: [v for part in long_parts for v in part[k]] for k in long_parts[0]}
        common.write_parquet(run.run_dir / "calibration_samples.parquet", merged)
        doc = {"model": args.model, "paraphrase": paraphrase, "calib_run": str(args.calib_run), "screen_run": str(screen_runs[0]), "screen_runs": [str(d) for d in screen_runs], "banks": banks,
               "rule": f"r_hat(z, w): the P-ask self-answer correctness at a_H = {spec.fixed_a_H:g}, c0 = {spec.fixed_c0:g} and w in {fitted_w}; logistic (main) and isotonic (sensitivity)"}
        common.write_json(run.run_dir / common.RHAT, _clean(doc))
        return {"run_dir": str(run.run_dir), "phase": "rhat", "model": args.model, "paraphrase": paraphrase,
                "banks": {b: ({"logistic": v["logistic"]["beta"], "n_items": v["n_items"]} if "logistic" in v else v) for b, v in banks.items()}}


# ---------------------------------------------------------------------------------------
# dry run and command line
# ---------------------------------------------------------------------------------------

def dry_run(ctx: common.Context, args: argparse.Namespace) -> int:
    cfg = ctx.cfg
    dry = common.DryRun(ctx)
    keys = [ctx.ns.development_pilot_key(args.pilot_rep)] if args.limit_items and args.pilot_rep else []
    try:
        pools = pool_list(args) if args.phase == "answer" else common.validate_pools(common.csv_list(args.pools) or list(common.CAL_POOLS))
    except common.EntryError as error:
        dry.check("pools", False, str(error))
        return dry.emit()
    common.dry_run_basics(dry, pools=pools, models=[("device", args.model)] if args.phase == "answer" else [], seed_keys=keys, check_inputs=args.phase == "answer")
    if args.phase == "answer":
        sizes = common.pool_sizes(ctx, pools, args.limit_items)
        dry.note("workload", {"pool_sizes": sizes, "p_answer_prompts": sum(sizes.values()), "backend": args.backend, "ece_bins": cfg["calibration"]["ece_bins"],
                              "temperature_fitted_on": {b: f"{b}_cal" for b in sorted({POOL_BANK[p] for p in pools})}})
    else:
        try:
            calib = common.read_calib_run(args.calib_run) if args.calib_run else None
            screening = common.read_screening_run(args.screen_run[0]) if args.screen_run else None
            dry.check("input runs", calib is not None and screening is not None,
                      {"calib_model": getattr(calib, "model", None), "primary_paraphrase": getattr(screening, "primary", None)})
        except common.EntryError as error:
            dry.check("input runs", False, str(error))
        if args.calib_run and args.screen_run:
            try:
                parents = common.parent_run_ids([("answer-phase run", args.calib_run), *[("screening run", d) for d in args.screen_run]])
                dry.check("parent runs: the input runs have manifests with a run id", True, parents)
                dry.note("parent_run_ids", parents)
            except common.EntryError as error:
                dry.check("parent runs: the input runs have manifests with a run id", False, str(error))
        spec = cal.RHatSpec.from_config(cfg)
        dry.note("workload", {"fits": ["logistic", "isotonic"], "fitted_w": list(spec.fitted_w), "fixed_a_H": spec.fixed_a_H, "fixed_c0": spec.fixed_c0, "gpu": False})
    return dry.emit()


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Temperature scaling, ECE, AUROC and the correctness map r_hat of one end-side model.")
    common.add_common_arguments(p)
    common.add_readout_arguments(p)
    p.add_argument("--phase", choices=("answer", "rhat"), default="answer")
    p.add_argument("--pools", default=None, help="comma list (answer: the four calibration and evaluation pools; rhat: the two calibration pools)")
    p.add_argument("--chunk-size", type=int, default=1000, help="prompts per unit of work")
    p.add_argument("--calib-run", default=None, help="rhat phase: the answer-phase run")
    p.add_argument("--screen-run", nargs="+", default=None, help="rhat phase: the prompt-screening run; more runs add the trials of other pools (the first gives the primary paraphrase)")
    p.add_argument("--paraphrase", default=None, help="rhat phase: override the primary paraphrase of the screening")
    return p


def main(argv: Sequence[str] | None = None, hooks: CalibHooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or CalibHooks()
    args = make_parser().parse_args(argv)
    t0 = time.time()
    try:
        ctx = common.build_context(args, script=SCRIPT, stage=STAGE, argv=argv)
        if args.dry_run:
            return dry_run(ctx, args)
        outcome = execute_answer(ctx, args, hooks) if args.phase == "answer" else execute_rhat(ctx, args)
    except (common.EntryError, SplitError, paths.PathPolicyError, cal.CalibrationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except common.SourceChangedError as error:
        return common.stopped_by_source_change(error)
    outcome["wall_s"] = time.time() - t0
    print(json.dumps(common.json_safe(outcome), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
