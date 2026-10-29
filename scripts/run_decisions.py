#!/usr/bin/env python
"""The decision grid of one end-side model.

The exact digit readout (offline vLLM by default; ``--backend hf`` or ``vllm_http`` read the same fields) of the P-ask prompt on the evaluation
pool for every condition of the grids of the configuration (``grid``), for each kept paraphrase of the prompt screening. The default pool is the main
one (``mmlu_redux_eval``); the robustness pool (``--pools simpleqa_mc_eval``) is a run of its own, because the downstream readers take one bank per decisions table:

* ``psychometric``: a_H in {0.60, 0.80, 0.95} x c0 in {0, 0.1, 0.2} x w in {0, 0.1, 0.2, 0.45, 0.8, 1.5} (54 conditions). It gives theta, R.
* ``f1``: the F1 framing (the expected wait in seconds and the cost per second) at a_H = 0.80, c0 = 0, kappa in {0.01, 0.03}, D_hat in {0, 10, 20, 40},
  and the F2 framing at the matching w (those the other grids lack: 0.3, 0.4, 0.6, 1.2). It gives the F1-F2 comparison.

A condition that two grids share is run once (the column ``grid`` lists the grids it belongs to). Every pool and paraphrase is its own set of units
of work; ``--resume`` continues an interrupted run.

``--exploratory`` marks a diagnostic run: development evidence that never enters a gate. With it ``--pools mmlu_redux_cal`` is required and is the only pool taken (an evaluation pool,
another pool or several pools exit 2), ``--grids`` must be given; besides ``psychometric`` and ``f1`` the grid
``screening`` is available there: the four diagnostic conditions of the prompt screening (``prompts.screening.diagnostic_conditions``, F2), merged with the other grids like any other grid.
``--layout block_position`` runs every condition in the variant layout of the prompt file (the conditions block before the question; the layout of the block-position check of the prompt
screening); the ``layout`` column says so. Without ``--exploratory`` the grid ``screening`` and ``--layout`` exit 2. The manifest of an exploratory run carries ``exploratory: true``, a
``development_reason`` and ``exploratory_run`` (the layout and the conditions that the run was started with), and ``decisions_summary.json`` carries ``exploratory: true``. ``--resume``
compares the mode, the layout and the actual conditions with those of the run being resumed BEFORE any finished unit is reused (the unit names hold the pool, the paraphrase and the chunk
number only, so a unit of another layout or of another condition set would be taken for finished): a difference in either direction exits 2. The psychometric analysis refuses such a run as
an input.

Output: ``decisions.parquet`` (the decisions table: item_key, model, pool, paraphrase, layout, framing, a_H, c0, w_displayed, kappa, D_hat_s,
digit_logprobs[5], digit_mass, gate_pass, free_token_id, free_is_digit, margin, ask, self_choice, self_correct; then gt_position, grid) and
``decisions_summary.json`` (a diagnostic of this run: the digit-mass gate pass rate of the whole run and per pool, paraphrase and framing, the share of free first
tokens that are digits, the ask rate; the conditions that were run). Those pass rates are totals of the run, not the K0.1 verdict: the psychometric analysis decides K0.1 on the
pre-specified denominator (the decision trials on the MMLU-Redux evaluation pool, the kept paraphrases, every condition of the grids, one trial per (item, paraphrase, condition)).

Usage:
    python scripts/run_decisions.py --model MODEL --screen-run DIR [--pools POOLS] [--grids GRIDS] --tag TAG [--dry-run]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
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
from nmi.data.registry import SplitError  # noqa: E402
from nmi.prompts import LAYOUTS, quantize_w  # noqa: E402
from nmi.readout.digits import ReadoutSpec  # noqa: E402

SCRIPT = "run_decisions"
STAGE = "decisions"
GRIDS = ("psychometric", "f1")
# --exploratory: the grids it can take, the one pool it reads, and what the run says about itself
EXPLORATORY_GRIDS = ("psychometric", "f1", "screening")
EXPLORATORY_POOL = common.MAIN_CAL_POOL  # the MMLU-Redux calibration pool: the evaluation pool is read in full by the answer phase and takes no post-hoc diagnostics
EXPLORATORY_REASON = "exploratory diagnostic"
DEFAULT_LAYOUT = "standard"
DECISION_FIRST = [
    "item_key", "model", "pool", "paraphrase", "layout", "framing", "a_H", "c0", "w_displayed", "kappa", "D_hat_s", "digit_logprobs", "digit_mass", "gate_pass",
    "free_token_id", "free_is_digit", "margin", "ask", "self_choice", "self_correct",
]


@dataclass
class DecisionHooks:
    opener: Callable[[common.Context, str, str, ReadoutSpec], common.BackendHandle] | None = None
    model_files: bool = True


@dataclass(frozen=True)
class Condition:
    """One condition of the decision grid. F2 shows ``w``; F1 shows ``D_hat_s`` and ``kappa`` (and ``w`` is their product)."""

    framing: str
    a_H: float
    c0: float
    w: float | None = None
    D_hat_s: int | None = None
    kappa: float | None = None

    def key(self) -> tuple[Any, ...]:
        r = lambda x: None if x is None else round(float(x), 9)  # noqa: E731
        return (self.framing, r(self.a_H), r(self.c0), r(self.w), None if self.D_hat_s is None else int(self.D_hat_s), r(self.kappa))

    def shown_w(self, rule: Any) -> float:
        """The waiting cost the device effectively sees: the quantised ``w`` of F2, or ``kappa x D_hat`` of F1."""
        value = self.w if self.framing == "f2" else float(self.kappa) * float(self.D_hat_s)
        return float(quantize_w(value, rule))


def grid_conditions(cfg: Any, grids: Sequence[str]) -> list[tuple[Condition, tuple[str, ...]]]:
    """Every condition of the chosen grids of the configuration, each once, with the grids it belongs to."""
    g = cfg["grid"]
    found: dict[tuple[Any, ...], tuple[Condition, set[str]]] = {}

    def add(cond: Condition, name: str) -> None:
        entry = found.setdefault(cond.key(), (cond, set()))
        entry[1].add(name)

    if "psychometric" in grids:
        ps = g["psychometric"]
        for a_H, c0, w in itertools.product(ps["a_H"], ps["c0"], ps["w"]):
            add(Condition("f2", float(a_H), float(c0), w=float(w)), "psychometric")
    if "f1" in grids:
        f1, f2 = g["f1_paired"]["f1"], g["f1_paired"]["f2_matches"]
        for a_H, c0, kappa, d in itertools.product(f1["a_H"], f1["c0"], f1["kappa"], f1["D_hat_s"]):
            add(Condition("f1", float(a_H), float(c0), D_hat_s=int(d), kappa=float(kappa)), "f1")
        for a_H, c0, w in itertools.product(f2["a_H"], f2["c0"], f2["w"]):
            add(Condition("f2", float(a_H), float(c0), w=float(w)), "f1")
    if "screening" in grids:  # --exploratory only (``grid_list``): the diagnostic conditions of the prompt screening (F2), merged with the other grids like any other
        for a_H, c0, w in cfg["prompts"]["screening"]["diagnostic_conditions"]:
            add(Condition("f2", float(a_H), float(c0), w=float(w)), "screening")
    return [(c, tuple(sorted(names))) for c, names in found.values()]


def chunks(n: int, size: int) -> list[range]:
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


def condition_job(
    ctx: common.Context, bh: common.BackendHandle, items: Sequence[Any], pool: str, paraphrase: str, conds: Sequence[tuple[Condition, tuple[str, ...]]], chat_kwargs: dict[str, Any], model: str,
    layout: str = DEFAULT_LAYOUT,
):
    def build() -> tuple[list[str], dict[str, Any]]:
        texts: list[str] = []
        meta: dict[str, list[Any]] = {k: [] for k in ("item_key", "model", "pool", "paraphrase", "layout", "framing", "a_H", "c0", "w_displayed", "kappa", "D_hat_s", "gt_position", "grid")}
        n = len(items)
        for cond, grids in conds:
            texts += common.ask_texts(
                ctx.book, bh.tokenizer, items, chat_kwargs, paraphrase=paraphrase, framing=cond.framing, a_H=cond.a_H, c0=cond.c0, w=cond.w, D_hat_s=cond.D_hat_s,
                kappa=cond.kappa, layout=layout,
            )
            meta["item_key"] += [i.item_key for i in items]
            meta["model"] += [model] * n
            meta["pool"] += [pool] * n
            meta["paraphrase"] += [paraphrase] * n
            meta["layout"] += [layout] * n
            meta["framing"] += [cond.framing] * n
            meta["a_H"] += [cond.a_H] * n
            meta["c0"] += [cond.c0] * n
            meta["w_displayed"] += [cond.shown_w(ctx.book.rule)] * n
            meta["kappa"] += [cond.kappa] * n
            meta["D_hat_s"] += [cond.D_hat_s] * n
            meta["gt_position"] += [i.gt_position for i in items]
            meta["grid"] += ["|".join(grids)] * n
        cols: dict[str, Any] = dict(meta)
        for name in ("a_H", "c0", "w_displayed"):
            cols[name] = np.asarray(meta[name], dtype=np.float64)
        cols["gt_position"] = np.asarray(meta["gt_position"], dtype=np.int64)
        return texts, cols

    return build


def summarize_decisions(table: pa.Table) -> dict[str, Any]:
    """A diagnostic of the run: the P-ask digit-mass gate pass rate of the whole run and per pool, paraphrase and framing, the share of digit free tokens and the ask rate.
    These are totals of the run, not the K0.1 verdict, which the psychometric analysis decides on the pre-specified denominator."""
    pool = np.asarray(table.column("pool").to_pylist())
    para = np.asarray(table.column("paraphrase").to_pylist())
    framing = np.asarray(table.column("framing").to_pylist())
    gate = table.column("gate_pass").to_numpy(zero_copy_only=False).astype(bool)
    free = table.column("free_is_digit").to_numpy(zero_copy_only=False).astype(bool)
    ask = table.column("ask").to_numpy(zero_copy_only=False).astype(bool)
    mass = table.column("digit_mass").to_numpy(zero_copy_only=False)
    out: dict[str, Any] = {"n_rows": int(table.num_rows), "gate_pass_rate": float(gate.mean()), "free_digit_rate": float(free.mean()), "ask_rate": float(ask.mean()), "groups": {}}
    for p, q, f in sorted(set(zip(pool.tolist(), para.tolist(), framing.tolist(), strict=True))):
        m = (pool == p) & (para == q) & (framing == f)
        out["groups"][f"{p}/{q}/{f}"] = {
            "n": int(m.sum()), "gate_pass_rate": float(gate[m].mean()), "free_digit_rate": float(free[m].mean()), "ask_rate": float(ask[m].mean()), "mean_digit_mass": float(mass[m].mean()),
        }
    return out


def is_exploratory(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "exploratory", False))


def pool_list(args: argparse.Namespace) -> list[str]:
    if is_exploratory(args):  # the one door to a calibration pool, and to this pool only
        requested = common.csv_list(args.pools)
        if requested != [EXPLORATORY_POOL]:
            raise common.EntryError(
                f"--exploratory runs on the one pool {EXPLORATORY_POOL} (the MMLU-Redux calibration pool) and needs --pools {EXPLORATORY_POOL}; "
                f"{'no --pools was given (the default pool is an evaluation pool)' if not requested else f'--pools {args.pools} was given'}: "
                "an evaluation pool, another pool or several pools are refused"
            )
        return common.validate_pools(requested)
    pools = common.validate_pools(common.csv_list(args.pools) or [common.MAIN_EVAL_POOL])
    if any(common.split_of(p) != "eval" for p in pools):
        raise common.EntryError("the decision grid runs on the evaluation pools")
    return pools


def grid_list(args: argparse.Namespace) -> list[str]:
    if is_exploratory(args):
        grids = common.csv_list(args.grids)
        if not grids:
            raise common.EntryError(f"--exploratory needs --grids (one or more of {list(EXPLORATORY_GRIDS)})")
        bad = [g for g in grids if g not in EXPLORATORY_GRIDS]
        if bad:
            raise common.EntryError(f"unknown grid(s) {bad}; with --exploratory the known grids are {list(EXPLORATORY_GRIDS)}")
        return grids
    grids = common.csv_list(args.grids) or list(GRIDS)
    if "screening" in grids:
        raise common.EntryError("the grid 'screening' (the four diagnostic conditions of the prompt screening) belongs to the exploratory diagnostics: it needs --exploratory")
    bad = [g for g in grids if g not in GRIDS]
    if bad:
        raise common.EntryError(f"unknown grid(s) {bad}; known: {list(GRIDS)}")
    return grids


def layout_of(args: argparse.Namespace) -> str:
    """The prompt layout of the run: ``standard`` unless ``--exploratory --layout block_position``. ``--layout`` without ``--exploratory`` exits 2, whatever its value."""
    given = getattr(args, "layout", None)
    if not is_exploratory(args):
        if given is not None:
            raise common.EntryError("--layout belongs to the exploratory diagnostics: it needs --exploratory (the registered decision grid runs in the standard layout)")
        return DEFAULT_LAYOUT
    layout = DEFAULT_LAYOUT if given is None else given
    if layout not in LAYOUTS:
        raise common.EntryError(f"unknown layout {layout!r}; known: {list(LAYOUTS)}")
    return layout


def conditions_record(conds_by_pool: dict[str, Sequence[tuple[Condition, tuple[str, ...]]]], pools: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """The conditions of every pool in the order they are run (the order decides which condition is in which unit), with the grids each belongs to: the record of
    ``decisions_summary.json`` and, for an exploratory run, the part of the run identity that ``check_resume_identity`` compares."""
    return {p: [{"framing": c.framing, "a_H": c.a_H, "c0": c.c0, "w": c.w, "D_hat_s": c.D_hat_s, "kappa": c.kappa, "grids": list(g)} for c, g in conds_by_pool[p]] for p in pools}


def exploratory_identity(layout: str, conds_by_pool: dict[str, Sequence[tuple[Condition, tuple[str, ...]]]], pools: Sequence[str]) -> dict[str, Any]:
    """What an exploratory run is, beyond the pool, the paraphrase and the chunk number that the unit names hold: the layout and the actual conditions. The manifest keeps it
    (``exploratory_run``), and a resumed run must have the same (``check_resume_identity``)."""
    return {"layout": layout, "conditions": conditions_record(conds_by_pool, pools)}


def check_resume_identity(args: argparse.Namespace, identity: dict[str, Any] | None) -> None:
    """The unit names of this entry hold the pool, the paraphrase and the chunk number only (``execute``), and ``run_jobs`` skips a unit whose file is there. A run that is
    resumed in another mode, in another layout or with other conditions would take the chunks of the first call for its own. So the mode, the layout and the actual conditions
    are compared with those of the run being resumed BEFORE any chunk is reused (and before the manifest of that run is touched): ``EntryError`` (exit 2) when they differ, in either direction.

    ``identity`` is ``exploratory_identity(...)`` of this call, or None for the standard run (no ``--exploratory``). Nothing is checked without ``--resume`` or when the run being resumed has
    no ``manifest.json`` (``open_run`` says so itself)."""
    resume = getattr(args, "resume", None)
    if not resume:
        return
    path = Path(resume) / common.MANIFEST_FILE
    if not path.is_file():
        return
    manifest = common.read_json(path)
    was_exploratory = isinstance(manifest, dict) and manifest.get("exploratory") not in (None, False)
    if identity is None:
        if was_exploratory:
            raise common.EntryError(f"--resume {resume}: that run was started with --exploratory (its manifest says exploratory); resume it with --exploratory and the options it was started with, not as a registered run")
        return
    if not was_exploratory:
        raise common.EntryError(f"--resume {resume}: that run was not started with --exploratory, so it cannot be resumed with it (its units are those of the registered decision grid)")
    recorded = manifest.get("exploratory_run")
    if not isinstance(recorded, dict) or "layout" not in recorded or "conditions" not in recorded:
        raise common.EntryError(f"--resume {resume}: the manifest of that exploratory run does not record its layout and conditions (exploratory_run), so they cannot be compared")
    now = json.loads(json.dumps(identity))
    differences: list[str] = []
    if recorded["layout"] != now["layout"]:
        differences.append(f"layout {recorded['layout']!r} -> {now['layout']!r}")
    if recorded["conditions"] != now["conditions"]:
        for pool in sorted(set(recorded["conditions"]) | set(now["conditions"])):
            before, after = recorded["conditions"].get(pool), now["conditions"].get(pool)
            if before == after:
                continue
            if before is None or after is None:
                differences.append(f"pool {pool}: {'not in the first session' if before is None else 'not in this call'}")
                continue
            first = next((i for i, (x, y) in enumerate(zip(before, after, strict=False)) if x != y), min(len(before), len(after)))
            differences.append(f"conditions of {pool}: {len(before)} then, {len(after)} now, the first difference at position {first}")
    if differences:
        raise common.EntryError(
            f"--resume {resume}: the layout and the conditions of the run must be those it was started with (a finished unit of another layout or of another condition set would be reused as if it were this run's): "
            + "; ".join(differences)
        )


def paraphrases_of(ctx: common.Context, args: argparse.Namespace) -> list[str]:
    if args.paraphrases:
        ids = common.csv_list(args.paraphrases)
    elif args.screen_run:
        screening = common.read_screening_run(args.screen_run)
        if not screening.retained:
            raise common.EntryError(f"{args.screen_run}: no paraphrase was retained")
        ids = list(screening.retained)
    else:
        raise common.EntryError("--screen-run (the kept paraphrases) or --paraphrases is required")
    unknown = [i for i in ids if i not in ctx.book.paraphrases]
    if unknown:
        raise common.EntryError(f"unknown paraphrase(s) {unknown}")
    return ids


def execute(ctx: common.Context, args: argparse.Namespace, hooks: DecisionHooks) -> dict[str, Any]:
    cfg = ctx.cfg
    entry = common.device_entry(cfg, args.model)
    spec = ReadoutSpec.from_config(cfg)
    pools, grids = pool_list(args), grid_list(args)
    layout = layout_of(args)
    paraphrases = paraphrases_of(ctx, args)
    chat_kwargs = common.chat_kwargs_of(entry)
    items_by_pool = {p: common.load_pool_items(ctx, p, limit=args.limit_items, pilot_rep=args.pilot_rep) for p in pools}
    conds_by_pool = {p: grid_conditions(cfg, grids) for p in pools}
    identity = exploratory_identity(layout, conds_by_pool, pools) if is_exploratory(args) else None
    check_resume_identity(args, identity)  # exit 2 BEFORE any finished unit of the run being resumed is reused
    keys = [ctx.ns.development_pilot_key(args.pilot_rep)] if args.limit_items else []
    extra_md5: dict[str, str] = {}
    inputs = [(label, name, run_dir) for label, name, run_dir in (("screening run", common.SCREENING, args.screen_run),) if run_dir]
    parents = common.parent_run_ids([(label, run_dir) for label, _, run_dir in inputs])  # exit 2 (before a run directory is made) when an input run has no manifest with a run id
    for label, name, run_dir in inputs:
        extra_md5["input.screening"] = common.file_md5(Path(run_dir) / name)
    opener = hooks.opener or (lambda c, kind, model, sp: common.open_backend(c, kind, model, sp))
    plan = common.numeric_plan(common.readout_backend_plan(args, unit={"conditions_per_unit": args.conditions_per_unit}))
    extra_manifest: dict[str, Any] = {"decision_grid": {"grids": grids, "paraphrases": paraphrases}}
    if identity is not None:  # an exploratory run says so in its manifest, with the layout and the conditions it was started with (the identity that a --resume must match)
        extra_manifest.update({"exploratory": True, "development_reason": EXPLORATORY_REASON, "exploratory_run": identity})
    with common.open_run(ctx, pools=pools, seed_keys=keys, parents=parents, extra_md5=extra_md5, numeric=plan, extra_manifest=extra_manifest) as run:
        bh = opener(ctx, args.backend, args.model, spec)
        try:
            run.confirm_numeric(args.backend, bh.numeric_actual())
            if hooks.model_files and not run.resumed:
                common.write_model_files(run, entry, bh)
            jobs, unit_ids = [], []
            for pool in pools:
                for para in paraphrases:
                    for gi, rows in enumerate(chunks(len(conds_by_pool[pool]), args.conditions_per_unit)):
                        unit = f"{pool}__{para}__g{gi:02d}"
                        unit_ids.append(unit)
                        jobs.append(common.Job(unit, condition_job(ctx, bh, items_by_pool[pool], pool, para, [conds_by_pool[pool][i] for i in rows], chat_kwargs, args.model, layout)))
            stats = common.run_jobs(run, bh.backend, jobs, label="decisions ")
        finally:
            bh.close()
        table = run.parts.concat(unit_ids)
        choice = table.column("self_choice").to_numpy(zero_copy_only=False)
        gt = table.column("gt_position").to_numpy(zero_copy_only=False)
        table = table.append_column("self_correct", pa.array(np.asarray(choice == gt, dtype=bool)))
        table = table.select(DECISION_FIRST + [c for c in table.column_names if c not in DECISION_FIRST])
        common.write_parquet(run.run_dir / "decisions.parquet", table)
        summary = summarize_decisions(table)
        summary.update({
            "model": args.model, "backend": args.backend, "pools": pools, "grids": grids, "paraphrases": paraphrases, "jobs": stats,
            "conditions_per_pool": conditions_record(conds_by_pool, pools),
        })
        if args.limit_items:
            summary["pilot"] = {"limit_items": args.limit_items, "pilot_rep": args.pilot_rep}
        if identity is not None:
            summary.update({"exploratory": True, "layout": layout, "development_reason": EXPLORATORY_REASON})
        common.write_json(run.run_dir / "decisions_summary.json", summary)
        return {"run_dir": str(run.run_dir), "model": args.model, "n_rows": int(table.num_rows), "gate_pass_rate": summary["gate_pass_rate"], "ask_rate": summary["ask_rate"]}


def dry_run(ctx: common.Context, args: argparse.Namespace) -> int:
    cfg = ctx.cfg
    dry = common.DryRun(ctx)
    try:
        pools, grids = pool_list(args), grid_list(args)
        layout = layout_of(args)
        paraphrases = paraphrases_of(ctx, args)
    except common.EntryError as error:
        dry.check("arguments and inputs", False, str(error))
        return dry.emit()
    keys = [ctx.ns.development_pilot_key(args.pilot_rep)] if args.limit_items and args.pilot_rep else []
    common.dry_run_basics(dry, pools=pools, models=[("device", args.model)], seed_keys=keys)
    if args.screen_run:
        try:
            parents = common.parent_run_ids([("screening run", args.screen_run)])
            dry.check("parent runs: the input runs have manifests with a run id", True, parents)
            dry.note("parent_run_ids", parents)
        except common.EntryError as error:
            dry.check("parent runs: the input runs have manifests with a run id", False, str(error))
    sizes = common.pool_sizes(ctx, pools, args.limit_items)
    per_pool = {p: grid_conditions(cfg, grids) for p in pools}
    counts: dict[str, dict[str, int]] = {}
    for p, conds in per_pool.items():
        by_grid: dict[str, int] = defaultdict(int)
        for _, names in conds:
            for name in names:
                by_grid[name] += 1
        counts[p] = {"conditions_run_once": len(conds), **by_grid}
    prompts = sum(sizes[p] * len(per_pool[p]) * len(paraphrases) for p in pools)
    dry.note("workload", {
        "pools": sizes, "grids": grids, "paraphrases": paraphrases, "conditions": counts, "decision_prompts": prompts,
        "units": sum(len(chunks(len(per_pool[p]), args.conditions_per_unit)) * len(paraphrases) for p in pools), "backend": args.backend,
        "note": "a condition shared by two grids is run once; the nominal upper bound (81 conditions x 3 paraphrases x 3,000 items) counts it twice",
    })
    identity = exploratory_identity(layout, per_pool, pools) if is_exploratory(args) else None
    if identity is not None:  # the dry run says that the run is exploratory, in which layout, and how many conditions it has
        dry.note("exploratory", {
            "exploratory": True, "layout": layout, "development_reason": EXPLORATORY_REASON, "pools": pools, "grids": grids, "conditions": {p: len(per_pool[p]) for p in pools},
        })
    if getattr(args, "resume", None):  # the mode, the layout and the conditions of the run being resumed (the standard run reports a difference only)
        try:
            check_resume_identity(args, identity)
            if identity is not None:
                dry.check("resume: the run being resumed has this mode, layout and conditions", True, {"resume": args.resume})
        except common.EntryError as error:
            dry.check("resume: the run being resumed has this mode, layout and conditions", False, str(error))
    return dry.emit()


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="The decision grid (psychometric grid, F1 pairs) of one end-side model.")
    common.add_common_arguments(p)
    common.add_readout_arguments(p)
    p.add_argument("--screen-run", default=None, help="prompt-screening run directory: its retained paraphrases are the ones run")
    p.add_argument("--paraphrases", default=None, help="comma list of paraphrase ids (overrides --screen-run)")
    p.add_argument("--pools", default=None, help="comma list of evaluation pools (default: mmlu_redux_eval, the main pool; simpleqa_mc_eval, the robustness pool, in a run of its own)")
    p.add_argument("--grids", default=None, help="comma list of psychometric, f1 (default: both); with --exploratory a comma list of psychometric, f1, screening (required)")
    p.add_argument("--conditions-per-unit", type=int, default=6)
    p.add_argument("--exploratory", action="store_true", help="exploratory diagnostics (development evidence that no gate and no formal analysis takes): the one pool mmlu_redux_cal (--pools mmlu_redux_cal), "
                   "--grids from psychometric, f1, screening, --layout; the manifest and decisions_summary.json say exploratory: true; --resume must match the mode, the layout and the conditions")
    p.add_argument("--layout", default=None, help="only with --exploratory: standard (default) or block_position, the variant layout of the prompt file (the conditions block before the question)")
    return p


def main(argv: Sequence[str] | None = None, hooks: DecisionHooks | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hooks = hooks or DecisionHooks()
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
