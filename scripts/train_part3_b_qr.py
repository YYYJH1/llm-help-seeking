"""Part 3, angle B, third objective ``qr`` (graded target): the T file, LoRA training and checkpoint selection.

Everything not stated here is the B program ``scripts/train_part3_b.py``, which this file imports and composes. It never edits, wraps or monkeypatches it: ``qr`` has its own ``train`` (the B ``train`` rejects the
objective at its tau whitelist, has no loss description for it and logs ``main`` as -util) and its own ``select`` (the B ``select`` accepts only routing and sft, and a qr manifest is never relabelled as routing).
Like B, this file never prints or stores question text.

Commands
    t        T_model = max(0.01, median over the 12,800 ``train`` decision rows of |d|), d = A - z0, from the fixed plan and the fixed z0, on the CPU, with no model forward pass.
             Stops (and writes nothing) if T differs at the third decimal from the reference values (Qwen3-4B 0.237, Gemma-4-E4B-it 0.266, SmolLM3-3B 0.236). A T file is never overwritten.
    train    LoRA training of one qr run (one model, one seed, one (lr, beta)); the arguments of ``train_part3_b.py train`` minus ``--objective`` (fixed to "qr") plus ``--t-file``. Needs ``--tau-file``
             (the same tau file that routing uses; the checks are B's own ``load_tau_file``) and ``--t-file`` (model id, plan hash, z0 hash and the hash of the definition of T).
    select   the qr selection adapter. Every check of the B selection (complete grid, seed, base model files, tau binding, the new T binding, readout provenance, the quality gate,
             the one-point fidelity rule, the ranking and the ties) with the B ``select_checkpoint`` as the core. An incomplete search is "pending", never "selected" or
             "no selectable checkpoint". Exit codes: 0 selected, 3 no selectable checkpoint (as B), 5 pending.

The qr objective. ``m`` is the routing margin of the readout, ``x = m / tau``, ``q = sigmoid(x)``, ``d = A - z0`` (A of the displayed terms, the fixed z0) and
``q* = sigmoid(d / T)``. Per row, in float32 and in this stable form,

    qr term      softplus(x) - q* x,   whose derivative with respect to m is (q - q*) / tau; it equals the binary cross-entropy BCE(q*, q)
    full loss    mean(qr term) + beta (KL1 + KL2) + lambda_M (L_M,decision + L_M,P-answer),   KL1, KL2 and L_M exactly as for routing, lambda_M = 1

Merge, validation reads, evaluation and crosscheck are the B commands, unchanged.

Usage (all CPU except ``train`` with an explicit ``--device cuda:N``, exactly as in B; every command is run with ``CUDA_VISIBLE_DEVICES=""`` unless it trains on a GPU)
    python -B scripts/train_part3_b_qr.py t --model qwen3-4b --out <qr root>/t/qwen3-4b.json
    python -B scripts/train_part3_b_qr.py train --model qwen3-4b --seed 0 --lr 1e-4 --beta 0.1 --steps 1000 --save-at 250,500,1000 --device cuda:0 \
        --tau-file <B root>/tau/qwen3-4b.json --t-file <qr root>/t/qwen3-4b.json --out <qr root>/qwen3-4b/runs/qr_lr1e-4_b0.1_s0
    python -B scripts/train_part3_b_qr.py select --model qwen3-4b --runs <the 4 seed-0 run directories> --val-out <root> --out <root>/selection.json \
        --tau-file <B root>/tau/qwen3-4b.json --t-file <qr root>/t/qwen3-4b.json
``select`` opens only ``<root>/base/{val,val_panswer}/rows.jsonl`` and ``<root>/<run name>_step<N>/{val,val_panswer}/rows.jsonl`` (plus the ``manifest.json`` of each readout directory); the record
``selection.json`` has the schema of B's (``part3_b.selection/1``, objective ``qr``) and the status ``selected``, ``no_selectable_checkpoint`` or ``pending``; ``retrain`` lists seeds 1 and 2 and there is
no ``controls`` entry. A run that ``train`` finished has the manifest of B's runs (``schema`` part3_b.train/1, ``status`` finished, ``objective`` qr, ``tau``, ``t``, ``adapters``) and the adapter meta
files of B with the T it was trained with.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_part3_b as B  # noqa: E402

PROGRAM = "train_part3_b_qr/1"
OBJECTIVE = "qr"  # the only objective of this program
SPEC_PATH = REPO / "protocol" / "fine_tuning_qr_spec.md"
SPEC_MD5 = "7dc62c941d5f7cdaf087083d79828670"  # md5 of the specification file; recorded next to the observed md5 in every manifest, not enforced
DEFAULT_PLANS_DIR = REPO / "results" / "part3_b_20261020" / "plans"  # the B plans: read, never written

T_SCHEMA = "part3_b_qr.t/1"
T_FLOOR = 0.01  # T_model = max(0.01, median |d|)
T_EXPECTED_DECIMALS = 3
T_EXPECTED = {"qwen3-4b": 0.237, "gemma-4-e4b-it": 0.266, "smollm3-3b": 0.236}  # reference values of T from an independent computation; the program stops if its own value differs at the third decimal
T_QUANTILES = (5, 25, 50, 75, 95)  # the quantiles of |d| written to the T file
LOSS_TEXT = ("mean(softplus(x) - q* x) + beta (KL1 + KL2) + lambda_M (L_M,decision + L_M,P-answer), x = m / tau, q* = sigmoid(d / T), d = A - z0 "
             "(the binary cross-entropy BCE(q*, sigmoid(x)); derivative with respect to m: (q - q*) / tau)")
FIRST_TERM_TEXT = "softplus(x) - q* x"

EXIT_NO_SELECTABLE = B.EXIT_NO_SELECTABLE  # 3, as in B's select
EXIT_PENDING = 5  # new: the search, a merge or a validation read is incomplete or failed for technical reasons
STATUS_SELECTED, STATUS_NONE, STATUS_PENDING = "selected", "no_selectable_checkpoint", "pending"
SELECTION_SCHEMA = "part3_b.selection/1"  # the schema of B's selection.json; the qr record uses it with objective "qr" and extra keys
SELECTION_RULE = ("lowest validation routing regret among candidates whose forced P-answer accuracy on cal-val drops by at most 1 percentage point from the base model and whose validation "
                  "readouts pass the quality gate; ties: smaller lr, smaller beta, earlier step; an incomplete or technically failed search is pending, not a result")


class TrainPlanError(B.TrainPlanError):
    """A rule of the qr program is violated. It is a ``train_part3_b.TrainPlanError`` too, so that the callers of the B commands handle it in the same way."""


# ---------------------------------------------------------------------------------------
# the objective: identity and the first loss term
# ---------------------------------------------------------------------------------------

def check_objective(objective: str) -> str:
    """The qr program trains one objective. The B objectives are refused: routing and sft have their own program, and the controls (fixed_cost, shuffled) are never run for qr."""
    if objective != OBJECTIVE:
        raise TrainPlanError(f"this program trains the objective {OBJECTIVE!r} only; {objective!r} is refused"
                             + (" (the controls fixed_cost and shuffled are not run for qr)" if objective in B.CONTROL_OBJECTIVES else ""))
    return objective


def qr_terms(m: torch.Tensor, A: torch.Tensor, z0: torch.Tensor, tau: float, T: float) -> dict[str, torch.Tensor]:
    """Per row. ``m`` (B,) is the routing margin in float32, ``A`` and ``z0`` are float64. x = m / tau; d = A - z0 and q* = sigmoid(d / T) are formed in float64 (the targets carry no gradient) and
    q* is then cast to the dtype of x, so that the term softplus(x) - q* x is evaluated in float32 in this stable form (softplus never overflows). Returns x, d, q_star and qr (the per-row term)."""
    x = m / tau
    d = A.to(torch.float64) - z0.to(torch.float64)
    # The term is evaluated in fp32; the target q* is formed in float64 from the exact A and the fixed z0 and cast to fp32 once,
    # which differs from forming it in fp32 by at most one fp32 rounding of q*.
    q_star = torch.sigmoid(d / T).to(x.dtype)
    return {"x": x, "d": d, "q_star": q_star, "qr": torch.nn.functional.softplus(x) - q_star * x}


def row_loss_qr(terms: Mapping[str, torch.Tensor], qr: Mapping[str, torch.Tensor], beta: float, lambda_m: float = B.LAMBDA_M) -> torch.Tensor:
    """Per row: the qr term + beta (KL1 + KL2) + lambda_M (L_M,decision + L_M,P-answer). ``terms`` come from ``row_terms`` of B, so that every term but the first is routing's."""
    return qr["qr"] + beta * (terms["kl1"] + terms["kl2"]) + lambda_m * (terms["lm_dec"] + terms["lm_pa"])


# ---------------------------------------------------------------------------------------
# T
# ---------------------------------------------------------------------------------------

def payoff_gap(rows: Sequence[B.Row]) -> np.ndarray:
    """d_i = A_i - z0_i of the decision rows in float64: A of the displayed two-decimal terms (exact decimal arithmetic, converted once) and the fixed z0."""
    return np.asarray([r.A - r.z0 for r in rows], dtype=np.float64)


def t_from_gaps(d: np.ndarray) -> float:
    """T_model = max(0.01, median over the rows of |d_i|), at full precision (``numpy.median``: the mean of the two middle values for an even number of rows)."""
    return max(T_FLOOR, float(np.median(np.abs(np.asarray(d, dtype=np.float64)))))


def t_definition_sha256() -> str:
    """The hash that a T file and a train run share: the source of the two functions that define T, and the floor (the "code hash" of the T binding). A hash of the whole code files would make
    every T file stale after any later edit of this file (the T files are never overwritten), so only the definition of T (these two functions and the floor) is checked, and the hashes of the
    whole files are recorded beside it."""
    text = inspect.getsource(payoff_gap) + inspect.getsource(t_from_gaps) + f"T_FLOOR={T_FLOOR!r}\n"
    return B.sha256_bytes(text.encode())


def check_expected_t(model_id: str, t: float) -> dict:
    """Stop if T differs from the reference value at the third decimal."""
    expected = T_EXPECTED.get(model_id)
    if expected is None:
        raise TrainPlanError(f"no expected T is registered for model {model_id!r}")
    if abs(round(t, T_EXPECTED_DECIMALS) - expected) > 1e-9:
        raise TrainPlanError(f"T = {t!r} for {model_id} rounds to {round(t, T_EXPECTED_DECIMALS)}, the expected value is {expected}; stopping, nothing is written")
    return {"value": expected, "decimals": T_EXPECTED_DECIMALS, "matches": True}


def z0_file_info(info: Mapping[str, Any] | None) -> dict | None:
    """The file that the fixed z0 comes from (B's ``load_z0``: the calibration table of the model) and its sha256; None for synthetic data."""
    if not info or "parquet" not in info:
        return None
    return {"path": info["parquet"], "sha256": info.get("parquet_sha256"), "run": info.get("run"), "frozen_temperature": info.get("frozen_temperature")}


def plan_file_check(plans_dir: Path, model_id: str, rows: Sequence[B.Row], d: np.ndarray, plan_hash: str) -> dict:
    """The train plan FILE of B against the plan that the loader samples (the one that ``train`` uses): the same rows, bit for bit the same d, the same plan hash, and the file hash
    that ``plans_manifest.json`` lists. Only item keys, conditions and payoffs are read (no question text exists in the file)."""
    path = plans_dir / model_id / "train.jsonl"
    file_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    digest = B.sha256_file(path)
    manifest = B.read_json(plans_dir / "plans_manifest.json")
    listed = manifest["models"][model_id]["files"]["train"]["sha256"]
    if listed != digest:
        raise TrainPlanError(f"{path} differs from the hash listed in plans_manifest.json")
    if len(file_rows) != len(rows):
        raise TrainPlanError(f"{path} has {len(file_rows)} rows, the sampled train plan has {len(rows)}")
    d_file = np.asarray([r["payoff_help"] - r["payoff_self"] for r in file_rows], dtype=np.float64)
    if not np.array_equal(d_file, d):
        raise TrainPlanError(f"{path}: d = payoff_help - payoff_self differs from A - z0 of the sampled train plan")
    lite = [SimpleNamespace(item_key=r["item_key"], cond=SimpleNamespace(a_h=r["a_h"], c0=r["c0"], w=r["w"], frame=r["frame"], domain=r["domain"], mapping_id=r["mapping_id"])) for r in file_rows]
    if B.plan_sha256(lite) != plan_hash:
        raise TrainPlanError(f"{path}: the plan hash of the file rows differs from the plan hash of the sampled train plan")
    return {"path": str(path), "sha256": digest, "listed_in_plans_manifest": True, "n_rows": len(file_rows), "d_identical_to_sampled_plan": True, "plan_sha256_identical": True}


def compute_t(args: argparse.Namespace, deps: B.Deps) -> dict:
    """The train rows come from B's own loader (``deps.load_split``, the same sampler and the same fixed z0 as ``train``), the plan file is cross-checked, T is computed, the value
    is checked against the reference value, and the T file is written (JSON, never overwritten). No model is loaded and no GPU is visible."""
    out = B.assert_off_root(args.out)
    if out.exists():
        raise TrainPlanError(f"{out} exists; a T file is never overwritten")
    t0 = time.time()
    split = deps.load_split(args.model, "train")
    rows = B.build_rows(split)
    B.check_training_support(rows, OBJECTIVE)
    n_items = len(split.items)
    if not deps.synthetic and (n_items != 800 or len(rows) != 16 * n_items):
        raise TrainPlanError(f"expected 800 training items with 16 conditions each, got {n_items} items and {len(rows)} rows")
    plan_hash, z0_hash = B.plan_sha256(rows), B.z0_sha256(rows)
    d = payoff_gap(rows)
    if not np.isfinite(d).all():
        raise TrainPlanError("a non-finite payoff gap")
    plan_file = None if deps.synthetic else plan_file_check(Path(args.plans_dir), args.model, rows, d, plan_hash)
    t = t_from_gaps(d)
    expected = None if deps.synthetic else check_expected_t(args.model, t)
    abs_d = np.abs(d)
    median = float(np.median(abs_d))
    quantiles = {str(q): float(v) for q, v in zip(T_QUANTILES, np.percentile(abs_d, T_QUANTILES))}
    doc = {
        "schema": T_SCHEMA, "model_id": args.model, "T": t, "median_abs_d": median, "floor": T_FLOOR, "floor_active": bool(median < T_FLOOR),
        "rule": "T = max(0.01, median over the train decision rows of |d|), d = A - z0 (A = a_H - c0 - w of the displayed terms, z0 the frozen calibrated confidence); CPU only, no model forward pass",
        "abs_d_quantiles": quantiles, "share_d_positive": float((d > 0).mean()), "n_rows": int(len(d)), "n_d_zero": int((d == 0).sum()), "n_items": n_items,
        "expected": expected,
        "hashes": {"plan_sha256": plan_hash, "z0_sha256": z0_hash, "plan_file": plan_file, "z0_file": z0_file_info(split.source.get("z0")), "t_definition_sha256": t_definition_sha256(),
                   "code": B.code_sources([Path(__file__), *([Path(deps.prompts.__file__)] if getattr(deps.prompts, "__file__", None) else []), SPEC_PATH])},
        "z0_source": split.source.get("z0"), "split_source": {k: v for k, v in split.source.items() if k != "z0"},
        "spec": {"path": str(SPEC_PATH), "frozen_md5": SPEC_MD5, "observed_md5": _md5(SPEC_PATH)},
        "settings": {"device": "cpu", "model_forward_pass": False, "precision": "float64 (numpy median of |d|)", "synthetic": bool(deps.synthetic)},
        "environment": B.environment(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_sec": round(time.time() - t0, 2),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    B.write_json(out, doc)
    return doc


def _md5(path: Path) -> str | None:
    try:
        return hashlib.md5(Path(path).read_bytes(), usedforsecurity=False).hexdigest()
    except OSError:
        return None


def load_t_file(path: str | Path, *, model_id: str, plan_hash: str, z0_hash: str, check_expected: bool = True, recomputed: float | None = None) -> dict:
    """The T binding of a train run: the schema, the model id, the plan hash, the z0 hash and the hash of the definition of T must be those of this run, and T must follow the rule.
    ``check_expected`` (real data only) repeats the check against the reference values. ``recomputed`` is T computed by the run itself from its own rows (``t_from_gaps(payoff_gap(rows))``): the
    value in the file must be that value, so that a file whose T was edited together with its median, under the hashes of the right plan, is refused.
    Returns what the manifest copies."""
    p = Path(path)
    doc = B.read_json(p)
    if doc.get("schema") != T_SCHEMA:
        raise TrainPlanError(f"{p} is not a T file of this program (schema {doc.get('schema')!r})")
    if doc["model_id"] != model_id:
        raise TrainPlanError(f"the T file belongs to model {doc['model_id']!r}, not {model_id!r}")
    hashes = doc["hashes"]
    if hashes["plan_sha256"] != plan_hash:
        raise TrainPlanError("the T file was computed on another train plan than this run's (plan hashes differ)")
    if hashes["z0_sha256"] != z0_hash:
        raise TrainPlanError("the T file was computed with another frozen z0 than this run's (z0 hashes differ)")
    if hashes["t_definition_sha256"] != t_definition_sha256():
        raise TrainPlanError("the T file was computed by another definition of T than this program's (code hashes differ)")
    t, median = float(doc["T"]), float(doc["median_abs_d"])
    if not (math.isfinite(t) and t >= T_FLOOR and abs(t - max(T_FLOOR, median)) <= 1e-12 * max(1.0, t)):
        raise TrainPlanError(f"the T file is inconsistent: T {t}, median |d| {median}; the rule is T = max({T_FLOOR}, median)")
    if recomputed is not None and abs(t - recomputed) > 1e-12:
        raise TrainPlanError(f"the T of the file ({t!r}) is not the T that this run computes from its own train rows ({recomputed!r})")
    if check_expected:
        check_expected_t(model_id, t)
    return {"value": t, "median_abs_d": median, "floor": doc["floor"], "file": str(p.resolve()), "file_sha256": B.sha256_file(p), "model_id": doc["model_id"], "abs_d_quantiles": doc["abs_d_quantiles"],
            "share_d_positive": doc["share_d_positive"], "hashes": hashes}


# ---------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------

@dataclass
class QrStepContext(B.StepContext):
    """B's step context plus the temperature T of the model."""

    T: float = 1.0


def forward_backward_qr(model, batch: B.Batch, beta: float, ctx: QrStepContext, backward: bool = True) -> dict[str, float]:
    """One logical step of qr: the reference (adapter disabled, no gradient) and the policy forward on every micro-batch, and the backward of ``sum(row losses) / B``. The structure, the readouts and
    every term but the first are B's (``row_terms`` of the routing objective); the first term is the qr term. Logged: ``main`` = the qr term, and for comparison the smoothed utility ``util`` (the
    routing first term is -util), q, KL1, KL2, both L_M, the P-answer accuracy of the policy and of the reference, and the qr quantities ``q_star``, ``q_gap`` (mean |q - q*|), ``grad_weight``
    (mean |q - q*| / tau, the size of the derivative of the first term with respect to m).
    The log keys are B's (``util``, ``q``, ``kl1``, ``kl2``, ``lm_dec``, ``lm_pa``, ``pa_acc`` for the policy and ``pa_acc_ref`` for the reference model with the adapter disabled, both against the answer key); the
    qr fields are named ``q_star``, ``q_gap`` and ``grad_weight``."""
    n = len(batch)
    acc: dict[str, float] = collections.defaultdict(float)
    mins = {"M_dec": 1.0, "M_pa": 1.0, "mass5_pa": 1.0}
    m_all: list[torch.Tensor] = []
    order = sorted(range(n), key=lambda i: len(batch.dec_ids[i]) + len(batch.pa_ids[i]))  # similar lengths share a micro-batch (as B)
    for lo in range(0, n, ctx.micro):
        take = order[lo:lo + ctx.micro]
        sel = torch.tensor(take, dtype=torch.long)
        part = B.Batch([batch.dec_ids[i] for i in take], [batch.pa_ids[i] for i in take], batch.help_idx[sel], batch.ans_idx[sel], batch.A[sel], batch.z0[sel], batch.gt_idx[sel])
        dev = ctx.device
        ids = ctx.digit_ids.to(dev)
        with torch.no_grad(), model.disable_adapter():
            ref_dec = B.readout_from_logits(B.last_logits(model, part.dec_ids, ctx.pad_id, dev, ctx.amp), ids, B.DECISION_SET)
            ref_pa = B.readout_from_logits(B.last_logits(model, part.pa_ids, ctx.pad_id, dev, ctx.amp), ids, B.PANSWER_SET)
        pol_dec = B.readout_from_logits(B.last_logits(model, part.dec_ids, ctx.pad_id, dev, ctx.amp), ids, B.DECISION_SET)
        pol_pa = B.readout_from_logits(B.last_logits(model, part.pa_ids, ctx.pad_id, dev, ctx.amp), ids, B.PANSWER_SET)
        on_dev = dataclasses.replace(part, help_idx=part.help_idx.to(dev), ans_idx=part.ans_idx.to(dev), A=part.A.to(dev), z0=part.z0.to(dev), gt_idx=part.gt_idx.to(dev))
        terms = B.row_terms("routing", pol_dec, ref_dec, pol_pa, ref_pa, on_dev, ctx.tau)  # m, q, util, KL1, KL2, L_M of both prompt types: routing's own definitions
        qr = qr_terms(terms["m"], on_dev.A, on_dev.z0, ctx.tau, ctx.T)
        losses = row_loss_qr(terms, qr, beta, ctx.lambda_m)
        if backward:
            (losses.sum() / n).backward()
        with torch.no_grad():
            acc["loss"] += float(losses.sum())
            acc["main"] += float(qr["qr"].sum())
            for k in ("kl1", "kl2", "q", "m", "util", "lm_dec", "lm_pa"):
                acc[k] += float(terms[k].sum())
            gap = (terms["q"] - qr["q_star"]).abs()
            acc["q_star"] += float(qr["q_star"].sum())
            acc["q_gap"] += float(gap.sum())
            acc["grad_weight"] += float(gap.sum()) / ctx.tau
            acc["ask"] += float((terms["m"] > 0).sum())
            m_all.append(terms["m"].detach().cpu())
            acc["slope"] += float((terms["q"] * (1 - terms["q"])).sum()) / ctx.tau  # d q / d m, as in B's log (the weight of the routing gradient)
            mass_dec, mass_pa, mass5_pa = pol_dec.mass, pol_pa.mass, pol_pa.lp5.exp().sum(1)
            acc["M_dec"] += float(mass_dec.sum())
            acc["M_pa"] += float(mass_pa.sum())
            acc["mass5_pa"] += float(mass5_pa.sum())
            for key, value in (("M_dec", mass_dec), ("M_pa", mass_pa), ("mass5_pa", mass5_pa)):
                mins[key] = min(mins[key], float(value.min()))
            gt = on_dev.gt_idx
            acc["pa_acc"] += float((pol_pa.lp5[:, :4].argmax(1) == gt).sum())
            acc["pa_acc_ref"] += float((ref_pa.lp5[:, :4].argmax(1) == gt).sum())
    out = {k: v / n for k, v in acc.items()}
    m_abs = torch.cat(m_all).abs()
    out["m_abs_median"], out["unsat_frac"] = float(m_abs.median()), float((m_abs / ctx.tau <= 4.0).float().mean())
    out["M_dec_min"], out["M_pa_min"], out["mass5_pa_min"] = mins["M_dec"], mins["M_pa"], mins["mass5_pa"]
    return out


def validate_train_args(args: argparse.Namespace) -> None:
    """B's checks (``validate_train_args`` of ``train_part3_b.py``) for the one objective qr: the fixed models, lr and beta unless ``--allow-offgrid``; a tau file and a T file are required.
    The seed is not restricted (the search runs seed 0, the finals seeds 1 and 2), as in B."""
    if args.time_only is not None:  # timing run: N steps of the formal configuration, one adapter, a throw-away output
        if args.time_only < 1:
            raise TrainPlanError("--time-only needs a positive number of steps")
        args.steps, args.save_at = args.time_only, (args.time_only,)
    if not args.allow_offgrid:
        if args.model not in B.MODEL_IDS:
            raise TrainPlanError(f"model {args.model!r} is not one of the frozen models {B.MODEL_IDS}")
        if not any(math.isclose(args.lr, x, rel_tol=1e-9) for x in B.GRID_LR):
            raise TrainPlanError(f"lr {args.lr} is not a frozen candidate {B.GRID_LR}")
        if not any(math.isclose(args.beta, x, rel_tol=1e-9) for x in B.GRID_BETA):
            raise TrainPlanError(f"beta {args.beta} is not a frozen candidate {B.GRID_BETA}")
    if args.steps < 1 or max(args.save_at) > args.steps:
        raise TrainPlanError("--save-at must lie inside 1..--steps")
    if args.batch < 2 or args.micro_batch < 1:
        raise TrainPlanError("--batch needs at least 2 rows and --micro-batch at least 1")
    if not args.tau_file:
        raise TrainPlanError("qr needs --tau-file (the tau file of this model that routing uses): run the tau command of this model first")
    if not args.t_file:
        raise TrainPlanError("qr needs --t-file: run the t command of this model first")


def train(args: argparse.Namespace, deps: B.Deps) -> dict:
    """One qr run. The structure is that of ``train_part3_b.train`` (same data, batching, model loading, optimiser, schedule, saving and logging schema) with the qr loss."""
    objective = check_objective(OBJECTIVE)
    device = B.resolve_device(args.device)  # first: nothing is read, written or loaded before the device is settled
    if deps.synthetic and device.type == "cuda":
        raise TrainPlanError("synthetic data is for the CPU dry run only")
    validate_train_args(args)
    out = B.assert_off_root(args.out)
    if out.exists() and any(out.iterdir()):
        raise TrainPlanError(f"{out} exists and is not empty; a run directory is never overwritten")
    requested = device
    device = B.prepare_device(requested)
    print(f"[device] {requested}" + (" (all GPUs hidden)" if device.type == "cpu" else f" -> {device} of CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}: {torch.cuda.get_device_name(device)}"), flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    amp = dtype == torch.bfloat16  # bf16 weights run under bf16 autocast (CPU autocast included, so that the CPU tests exercise the path of the GPU run)
    model_path = Path(args.model_path) if args.model_path else Path(B.model_entry(args.model)["path"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # data and its checks (qr trains on the canonical train plan, as routing)
    train_split = deps.load_split(args.model, "train")
    val_split = deps.load_split(args.model, "val")
    rows = B.build_rows(train_split)
    B.check_training_support(rows, objective)
    n_items = len(train_split.items)
    if not deps.synthetic and (n_items != 800 or len(val_split.items) != 200 or len(rows) != 16 * n_items):
        raise TrainPlanError(f"expected 800 training items with 16 conditions each and 200 validation items, got {n_items} items, {len(rows)} rows, {len(val_split.items)} validation items")
    per_item = collections.Counter(r.item_key for r in rows)
    if len(set(per_item.values())) != 1:
        raise TrainPlanError("every training item must have the same number of conditions")
    forbidden = deps.forbidden()
    B.assert_pairwise_disjoint({"train": set(train_split.items), "val": set(val_split.items), **forbidden})

    # tokenizer, renderer, digit ids, hashes
    from transformers import AutoTokenizer

    from nmi.readout.digits import resolve_digit_token_ids

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    digit_ids = torch.tensor(resolve_digit_token_ids(tokenizer, B.DIGITS), dtype=torch.long)
    renderer = B.Renderer(deps.prompts, tokenizer, args.model)
    plan_hash, z0_hash = B.plan_sha256(rows), B.z0_sha256(rows)
    ids_hash = B.prompt_ids_sha256(rows, renderer)
    canonical_hash = B.decision_ids_sha256(rows, renderer)  # the canonical hash that the tau file carries; qr trains on the canonical plan, so it is also the run's own hash
    model_files = B.model_sources(model_path)
    tau_info = B.load_tau_file(args.tau_file, model_id=args.model, model_path=model_path, model_files=model_files, decision_hash=canonical_hash)  # B's own checks
    t_info = load_t_file(args.t_file, model_id=args.model, plan_hash=plan_hash, z0_hash=z0_hash, check_expected=not deps.synthetic, recomputed=t_from_gaps(payoff_gap(rows)))
    tau, T = tau_info["value"], t_info["value"]
    t_prepared = time.time()

    # model
    model = B.load_base_model(model_path, dtype, device, args.attn_implementation, gradient_checkpointing=not args.no_grad_checkpointing)
    n_targets = len(B.lora_target_names(model))
    model = B.add_lora(model, args.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    t_model = time.time()

    manifest = {
        "schema": "part3_b.train/1",  # the schema of B's train manifests: select, merge and the scheduler read the same fields
        "program": PROGRAM,
        "status": "started",
        "timing_only": args.time_only is not None,
        "model_id": args.model,
        "model_path": str(model_path),
        "objective": objective,
        "seed": args.seed,
        "lr": args.lr,
        "beta": args.beta,
        "steps": args.steps,
        "save_at": sorted(set(args.save_at) | {args.steps}),
        "tau": tau_info,
        "t": t_info,
        "settings": {
            "batch_decision": args.batch, "batch_panswer": args.batch, "micro_batch": args.micro_batch, "tau": tau, "T": T, "lambda_m": B.LAMBDA_M, "warmup_steps": args.warmup, "weight_decay": 0.0,
            "adamw_betas": [0.9, 0.999], "adamw_eps": 1e-8, "max_grad_norm": args.max_grad_norm, "lr_schedule": "linear warmup, then constant", "dtype": args.dtype, "autocast_bf16": amp,
            "gradient_checkpointing": not args.no_grad_checkpointing, "attn_implementation": args.attn_implementation, "device": str(requested), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "lora": {"r": B.LORA_R, "alpha": B.LORA_ALPHA, "dropout": B.LORA_DROPOUT, "targets": "language-model attention and MLP projections", "n_target_modules": n_targets, "n_params": n_lora},
            "loss": LOSS_TEXT,
            "first_term": FIRST_TERM_TEXT,
            "mass_terms": "L_M = KL(Bern(M_ref) || Bern(M_theta)), M = total probability of the five digits (decision prompt) or of the four answer digits (P-answer prompt); logsumexp, never 1 - M",
            "kl1_in_loss": True,
        },
        "data": {
            "source": "synthetic (CPU dry run)" if deps.synthetic else "B1 split loader",
            "n_items": n_items, "n_rows": len(rows), "conditions_per_item": next(iter(per_item.values())), "n_val_items": len(val_split.items),
            "plan_sha256": plan_hash, "train_plan_sha256": plan_hash, "z0_sha256": z0_hash, "prompt_ids_sha256": ids_hash, "canonical_decision_ids_sha256": canonical_hash,
            "run_decision_ids_sha256": canonical_hash, "split_source": train_split.source, "disjoint_from": sorted(forbidden),
        },
        "sources": {"code": B.code_sources([Path(__file__), SPEC_PATH, *([Path(deps.prompts.__file__)] if getattr(deps.prompts, "__file__", None) else [])]), "model_files": model_files,
                    "spec": {"path": str(SPEC_PATH), "frozen_md5": SPEC_MD5, "observed_md5": _md5(SPEC_PATH)}},
        "nmi_source_sha256": None,
        "environment": B.environment(),
        "command": " ".join(sys.argv),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "adapters": [],
    }
    try:
        from nmi.provenance import read_git_head, source_hash

        manifest["nmi_source_sha256"] = source_hash()
        manifest["git_head"] = read_git_head().commit
    except Exception:  # pragma: no cover
        pass
    B.write_json(out / "manifest.json", manifest)

    stream = B.BatchStream(len(rows), args.batch, args.seed)
    ctx = QrStepContext(pad_id=B.pad_token_id(tokenizer), device=device, amp=amp, digit_ids=digit_ids, micro=args.micro_batch, tau=tau, T=T)
    log_path = out / "train_log.jsonl"
    last_stats: dict[str, float] = {}
    step_seconds: list[float] = []
    shown_hash, n_shown = hashlib.sha256(), 0  # the decision prompts actually shown, in training order
    with open(log_path, "w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            t_step = time.time()
            idx = stream.indices(step - 1)
            batch_rows = [rows[int(i)] for i in idx]
            batch = B.make_batch(batch_rows, [r.cond for r in batch_rows], renderer)
            for ids_row in batch.dec_ids:
                shown_hash.update(np.int64(len(ids_row)).tobytes())
                shown_hash.update(np.asarray(ids_row, dtype=np.int32).tobytes())
            n_shown += len(batch.dec_ids)
            lr = args.lr * B.lr_factor(step, args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            stats = forward_backward_qr(model, batch, args.beta, ctx)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm if args.max_grad_norm > 0 else float("inf")))
            if not (math.isfinite(stats["loss"]) and math.isfinite(grad_norm)):
                raise TrainPlanError(f"non-finite loss or gradient at step {step}; the run is stopped, nothing is repaired silently")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sec = time.time() - t_step
            step_seconds.append(sec)
            record = {"step": step, "lr": lr, "grad_norm": grad_norm, "clipped": bool(args.max_grad_norm > 0 and grad_norm > args.max_grad_norm), "sec": round(sec, 3), **{k: B.sig(v) for k, v in stats.items()}}
            if device.type == "cuda":
                record["peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            last_stats = stats
            if step % 25 == 0 or step == 1 or step == args.steps:
                print(f"[{args.model} {objective} s{args.seed}] step {step}/{args.steps} loss {stats['loss']:.5f} qr {stats['main']:.5f} q {stats['q']:.3f} q* {stats['q_star']:.3f} kl1 {stats['kl1']:.5f} "
                      f"kl2 {stats['kl2']:.5f} LM {stats['lm_dec']:.3g}/{stats['lm_pa']:.3g} M {stats['M_dec']:.4g}/{stats['M_pa']:.4g} (min {stats['M_dec_min']:.4g}/{stats['M_pa_min']:.4g}) "
                      f"tau {tau:.3f} T {T:.4f} grad {grad_norm:.3f} {record['sec']}s" + (f" peak {record['peak_gb']}GB" if "peak_gb" in record else ""), flush=True)
            if step in manifest["save_at"]:
                t_save = time.time()
                entry = B.save_adapter(model, out, step, manifest)
                meta_path = out / entry["dir"] / "part3_b_meta.json"
                meta = B.read_json(meta_path)  # the adapter meta of B, plus the qr program and the T it was trained with
                meta.update(program=PROGRAM, t=T, t_file_sha256=t_info["file_sha256"])
                B.write_json(meta_path, meta)
                entry["save_seconds"] = round(time.time() - t_save, 3)
                manifest["adapters"].append(entry)
                B.write_json(out / "manifest.json", manifest)
    manifest["data"]["shown_decision_ids_sha256"], manifest["data"]["n_shown_decision_prompts"] = shown_hash.hexdigest(), n_shown
    manifest["status"] = "finished"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["elapsed_sec"] = round(time.time() - t0, 1)
    manifest["last_step_stats"] = {k: B.sig(v) for k, v in last_stats.items()}
    B.write_json(out / "manifest.json", manifest)
    if args.time_only is not None:
        stable = step_seconds[B.TIMING_FIRST_STABLE_STEP - 1:]
        timing = {
            "schema": "part3_b.timing/1", "timing_only": True, "model_id": args.model, "objective": objective, "seed": args.seed, "lr": args.lr, "beta": args.beta, "n_steps": len(step_seconds),
            "formal_configuration": {"batch_decision": args.batch, "batch_panswer": args.batch, "micro_batch": args.micro_batch, "gradient_checkpointing": not args.no_grad_checkpointing, "dtype": args.dtype,
                                     "autocast_bf16": amp, "device": str(requested), "reference_forward": True, "mass_terms": True, "tau": tau, "T": T},
            "data_and_hash_seconds": round(t_prepared - t0, 3), "model_load_seconds": round(t_model - t_prepared, 3), "first_step_seconds": round(step_seconds[0], 3),
            "initialisation_seconds": round(t_model - t_prepared + step_seconds[0], 3),
            "stable_steps": [B.TIMING_FIRST_STABLE_STEP, len(step_seconds)], "median_step_seconds": round(statistics.median(stable), 4) if stable else None,
            "step_seconds": [round(x, 4) for x in step_seconds], "adapter_save_seconds": manifest["adapters"][-1]["save_seconds"],
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3) if device.type == "cuda" else None,
        }
        B.write_json(out / "timing.json", timing)
        manifest["timing"] = {"file": "timing.json", "median_step_seconds": timing["median_step_seconds"]}
        B.write_json(out / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------------------
# select: the qr selection adapter
# ---------------------------------------------------------------------------------------

def discover_candidates(run_dirs: Sequence[str | Path]) -> list[dict]:
    """B's ``discover_candidates`` (finished, not timing-only runs; one candidate per saved adapter), with the T entry and the data hashes of each run's manifest added."""
    cands = B.discover_candidates(run_dirs)
    for c in cands:
        man = B.read_json(Path(c["run_dir"]) / "manifest.json")
        c["t_info"] = man.get("t")
        c["t_file_sha256"] = (man.get("t") or {}).get("file_sha256")
        c["plan_sha256"] = (man.get("data") or {}).get("plan_sha256")
        c["z0_sha256"] = (man.get("data") or {}).get("z0_sha256")
        c["program"] = man.get("program")
    return cands


def check_t_binding(cands: Sequence[Mapping], t_file: str | Path) -> dict:
    """The T binding, the counterpart of B's tau binding: every candidate was trained with the T file given here (its hash, as recorded in the run manifest, is the hash of the file now),
    the file belongs to the model, follows the rule for T, and its plan hash, z0 hash and hash of the definition of T are those the runs recorded."""
    p = Path(t_file)
    if not p.is_file():
        raise TrainPlanError(f"the T file {p} does not exist")
    doc = B.read_json(p)
    digest = B.sha256_file(p)
    model_id = cands[0]["model_id"]
    if doc.get("schema") != T_SCHEMA:
        raise TrainPlanError(f"{p} is not a T file of this program (schema {doc.get('schema')!r})")
    if doc["model_id"] != model_id:
        raise TrainPlanError(f"the T file {p} belongs to model {doc['model_id']!r}, not {model_id!r}")
    t, median = float(doc["T"]), float(doc["median_abs_d"])
    if not (math.isfinite(t) and t >= T_FLOOR and abs(t - max(T_FLOOR, median)) <= 1e-12 * max(1.0, t)):
        raise TrainPlanError(f"the T file {p} is inconsistent: T {t}, median |d| {median}")
    for c in cands:
        info = c["t_info"]
        if not info:
            raise TrainPlanError(f"{c['tag']}: the run manifest records no T (it is not a qr run of this program)")
        if info["file_sha256"] != digest:
            raise TrainPlanError(f"{c['tag']}: the T file given to select is not the one the run used (or it changed since): the run recorded {info['file_sha256'][:12]}, the file has {digest[:12]}")
        if float(info["value"]) != t or info["model_id"] != model_id:
            raise TrainPlanError(f"{c['tag']}: the T of the run differs from the T file")
        if c["plan_sha256"] != doc["hashes"]["plan_sha256"] or c["z0_sha256"] != doc["hashes"]["z0_sha256"]:
            raise TrainPlanError(f"{c['tag']}: the run was trained on another plan or z0 than the T file was computed on")
        if info["hashes"]["t_definition_sha256"] != doc["hashes"]["t_definition_sha256"]:
            raise TrainPlanError(f"{c['tag']}: the run and the T file carry different hashes of the definition of T")
    return {"t_file": str(p.resolve()), "t_file_sha256": digest, "checked": True, "T": t, "model_id": model_id, "t_definition_sha256": doc["hashes"]["t_definition_sha256"]}


def merge_problem(readout_dir: Path) -> str | None:
    """A reason (text) if the merged checkpoint that produced this readout is on disk and its merge was not accepted (the bitwise weight check failed); None otherwise, also when the checkpoint was pruned after
    its readout was made (the provenance check of B reports that case as "not verified")."""
    mp = readout_dir / "manifest.json"
    if not mp.is_file():
        return None
    checkpoint = B.read_json(mp).get("checkpoint")
    if checkpoint in (None, "base"):
        return None
    mm = Path(checkpoint) / "merge_manifest.json"
    if mm.is_file() and B.read_json(mm).get("accepted") is False:
        return "the merge of this candidate was not accepted (bitwise weight check failed)"
    return None


def run_select(model_id: str, run_dirs: Sequence[str | Path], val_out: str | Path, tau_file: str | Path, t_file: str | Path, out_file: str | Path, allow_partial: bool = False,
               expect_rows: tuple[int, int] | None = (B.EXPECTED_VAL_ROWS, B.EXPECTED_VAL_ITEMS)) -> dict:
    """Select among the seed-0 runs of the qr search of one model.

    * Errors (nothing is written): not the qr objective, an unfinished or timing-only run, runs that differ in model, seed, tau file, T file or base model files, an incomplete grid (unless
      ``allow_partial``, tests only), a tau or T file that does not bind the runs, a base readout that is incomplete or fails the quality gate, a readout filed under the wrong tag, trials that differ
      from the base model's.
    * ``pending`` (the record is written, ``selected`` is None, no candidate is ranked): a candidate lacks its ``val`` or ``val_panswer`` readout (its read or its merge is not done), a readout
      holds trials without a readout (a technical failure: the statistic is undetermined), or the merge of a candidate was not accepted. Pending is not a performance result.
    * Otherwise the ``select_checkpoint`` of B decides: ``selected`` or ``no_selectable_checkpoint``.
    ``expect_rows`` = (decision trials of ``val``, items of ``val_panswer``) that the base readout must hold; ``None`` skips the count."""
    val_out, out_file = Path(val_out), Path(out_file)
    cands = discover_candidates(run_dirs)
    if not cands:
        raise TrainPlanError("no candidate checkpoint")
    if cands[0]["model_id"] != model_id:
        raise TrainPlanError(f"the runs belong to model {cands[0]['model_id']!r}, not {model_id!r}")
    objectives = sorted({c["objective"] for c in cands})
    if objectives != [OBJECTIVE]:
        raise TrainPlanError(f"this selection is for the objective {OBJECTIVE!r} only, the runs are {objectives}; a manifest of another objective is never relabelled")
    for key in ("model_id", "seed", "tau_file_sha256", "t_file_sha256"):
        if len({c[key] for c in cands}) != 1:
            raise TrainPlanError(f"select compares the runs of one qr search at seed 0 with one tau file and one T file; the runs differ in {key}")
    if len({json.dumps(c["model_files"], sort_keys=True) for c in cands}) != 1:
        raise TrainPlanError("the runs of one search were trained on different base model files")
    tau_check = B.check_tau_binding(cands, tau_file)
    t_check = check_t_binding(cands, t_file)
    if cands[0]["seed"] != 0:
        raise TrainPlanError("the search runs with seed 0")
    expected = {(lr, beta, step) for lr in B.GRID_LR for beta in B.GRID_BETA for step in B.GRID_STEPS}
    got = {(c["lr"], c["beta"], c["step"]) for c in cands}
    if not allow_partial and (got != expected or len(got) != len(cands)):
        raise TrainPlanError(f"the search needs exactly the frozen grid (lr x beta x step); missing {sorted(expected - got)}, extra {sorted(got - expected)}")
    inputs: dict[str, str] = {}

    def present(tag: str, name: str) -> bool:
        return (val_out / tag / name / "rows.jsonl").is_file()

    def load(tag: str, name: str) -> list[dict]:
        if name not in B.READ_SETS:
            raise AssertionError(f"select reads only {B.READ_SETS}, not {name!r}")
        p = val_out / tag / name / "rows.jsonl"
        if not p.is_file():
            raise TrainPlanError(f"missing readout {p}")
        inputs[str(p)] = B.sha256_file(p)
        return B.read_jsonl(p)

    provenance = {"base": B.check_readout_provenance(val_out / "base", "base", cands[0]["model_id"], None)}
    for name in B.READ_SETS:
        if not present("base", name):
            raise TrainPlanError(f"missing readout of the base model: {val_out / 'base' / name / 'rows.jsonl'}")
    base_dec, base_pa = B.routing_regret(load("base", "val")), B.forced_accuracy(load("base", "val_panswer"))
    if base_dec["n_missing"] or base_pa["n_missing"]:
        raise TrainPlanError("the base model's validation readout has missing rows")
    bad_base = B.describe_failed_units("val", base_dec["gate_units"]) + B.describe_failed_units("val_panswer", base_pa["gate_units"])
    if bad_base:
        raise TrainPlanError("the base model's validation readout fails the quality gate: " + "; ".join(bad_base))
    if expect_rows is not None and (base_dec["n_rows"], base_pa["n"]) != tuple(expect_rows):
        raise TrainPlanError(f"the base readout holds {base_dec['n_rows']} decision trials and {base_pa['n']} P-answer items, expected {tuple(expect_rows)}")

    results: list[dict] = []
    pending: list[dict] = []
    for c in cands:
        head = {k: c[k] for k in ("tag", "lr", "beta", "step", "run_name", "adapter_dir")}
        absent = [name for name in B.READ_SETS if not present(c["tag"], name)]
        if absent:  # the merge or the read of this candidate is not done: nothing to check, nothing to rank
            reason = "no " + " and no ".join(f"{name} readout" for name in absent) + " (its merge or its validation read is not done)"
            pending.append({"tag": c["tag"], "reason": reason, "kind": "readout absent"})
            results.append({**head, "regret": None, "k": None, "n": None, "n_missing": None, "gate_pass_rate": None, "pa_gate_pass_rate": None, "readout_valid": None, "invalid_units": [],
                            "gate_units": None, "eligible": None, "reason": "pending: " + reason})
            continue
        provenance[c["tag"]] = B.check_readout_provenance(val_out / c["tag"], c["tag"], c["model_id"], c["adapter_sha256"])
        dec, pa = B.routing_regret(load(c["tag"], "val")), B.forced_accuracy(load(c["tag"], "val_panswer"))
        if dec["keys"] != base_dec["keys"] or pa["keys"] != base_pa["keys"]:
            raise TrainPlanError(f"{c['tag']}: the trials differ from the base model's trials (every planned trial must appear once)")
        invalid = B.describe_failed_units("val", dec["gate_units"]) + B.describe_failed_units("val_panswer", pa["gate_units"])
        problem = merge_problem(val_out / c["tag"])  # a merge that is on disk with accepted = false is pending; an absent merge shows as an absent readout
        n_missing = dec["n_missing"] + pa["n_missing"]
        row = {**head, "regret": dec["regret"], "k": pa["k"], "n": pa["n"], "n_missing": n_missing, "gate_pass_rate": dec["gate_pass_rate"], "pa_gate_pass_rate": pa["gate_pass_rate"],
               "readout_valid": not invalid, "invalid_units": invalid, "gate_units": {"val": dec["gate_units"], "val_panswer": pa["gate_units"]}}
        if problem:
            pending.append({"tag": c["tag"], "reason": problem, "kind": "merge not accepted"})
        if n_missing:  # trials without a readout are a technical failure of the read, so the candidate is undetermined and the search is pending; this holds also when the quality gate would
            # pass (a few missing trials; the core of B would exclude the candidate and still return a status) and also when the missing trials make the gate fail (B would call it "readout invalid")
            pending.append({"tag": c["tag"], "reason": f"{n_missing} trials without a readout (val {dec['n_missing']}, val_panswer {pa['n_missing']}): the read failed technically", "kind": "missing trials"})
        if problem or n_missing:
            row.update(eligible=None, reason="pending: " + "; ".join(p["reason"] for p in pending if p["tag"] == c["tag"]))
        results.append(row)

    common = {
        "schema": SELECTION_SCHEMA, "program": PROGRAM, "model_id": cands[0]["model_id"], "objective": OBJECTIVE, "search_seed": 0, "rule": SELECTION_RULE,
        "base": {"regret": base_dec["regret"], "accuracy_k": base_pa["k"], "n_items": base_pa["n"], "n_val_rows": base_dec["n_rows"]},
        "inputs": inputs, "readout_provenance": provenance, "base_gate_units": {"val": base_dec["gate_units"], "val_panswer": base_pa["gate_units"]},
        "tau": cands[0]["tau"], "tau_file_sha256": cands[0]["tau_file_sha256"], "tau_binding": tau_check,
        "t": t_check["T"], "t_file_sha256": t_check["t_file_sha256"], "t_binding": t_check,
        "spec": {"path": str(SPEC_PATH), "frozen_md5": SPEC_MD5, "observed_md5": _md5(SPEC_PATH)},
    }
    if pending:  # the record is written with status "pending" (exit code 5, a code that B's select does not use) so that a caller can read it; nothing is ranked or rejected,
        # and the numbers of the complete candidates are listed for the record only
        for r in results:
            if "eligible" not in r:
                r.update(eligible=None, reason="not ranked: the search is pending")
        out = {**common, "status": STATUS_PENDING, "candidates": results, "selected": None, "pending": pending, "n_candidates": len(cands),
               "excluded_for_missing_rows": [], "readout_invalid": [r["tag"] for r in results if r.get("readout_valid") is False]}
    else:
        verdict = B.select_checkpoint({"k": base_pa["k"], "n": base_pa["n"]}, results)  # the core of B: eligibility, the one-point rule, the ranking and the ties
        sel = verdict["selected"]
        out = {**common, "status": verdict["status"], "candidates": verdict["table"], "selected": sel, "pending": [], "n_candidates": len(cands),
               "excluded_for_missing_rows": [c["tag"] for c in verdict["table"] if c["reason"] == "missing readout rows"],  # empty: a candidate with missing trials makes the search pending
               "readout_invalid": [c["tag"] for c in verdict["table"] if not c.get("readout_valid", True)]}
        if sel:
            out["retrain"] = [{"seed": s, "lr": sel["lr"], "beta": sel["beta"], "steps": sel["step"], "save_at": [sel["step"]]} for s in (1, 2)]  # no controls for qr
    out_file.parent.mkdir(parents=True, exist_ok=True)
    B.write_json(out_file, out)
    return out


def format_selection(sel: Mapping) -> str:
    base = sel["base"]
    lines = [f"{sel['model_id']} {sel['objective']}: {sel['status']}   base regret {base['regret']}, base correct {base['accuracy_k']}/{base['n_items']}   tau {sel['tau']}, T {sel['t']}"]
    lines.append(f"{'tag':48s} {'lr':>8s} {'beta':>5s} {'step':>5s} {'regret':>9s} {'drop_pp':>8s} {'gate':>6s} {'paGate':>6s} eligible")
    nan = float("nan")
    for c in sel["candidates"]:
        regret = "-" if c.get("regret") is None else f"{c['regret']:.5f}"
        gate = c.get("gate_pass_rate")
        pa_gate = c.get("pa_gate_pass_rate")
        lines.append(f"{c['tag']:48s} {c['lr']:8.0e} {c['beta']:5.2f} {c['step']:5d} {regret:>9s} {c.get('drop_pp', nan):8.2f} {nan if gate is None else gate:6.3f} {nan if pa_gate is None else pa_gate:6.3f} "
                     f"{c.get('eligible')} {c.get('reason', '')}")
    if sel.get("pending"):
        lines.append(f"PENDING (not a performance result, nothing is ranked): {len(sel['pending'])} problems")
        lines += [f"  {p['tag']}: {p['reason']}" for p in sel["pending"]]
    if sel.get("readout_invalid"):
        lines.append(f"readout invalid (fails the quality gate, not ranked): {sel['readout_invalid']}")
    if sel.get("selected"):
        s = sel["selected"]
        lines.append(f"selected: {s['tag']} (lr {s['lr']}, beta {s['beta']}, step {s['step']}); retrain seeds 1 and 2 to step {s['step']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = p.add_subparsers(dest="command", required=True)

    u = sub.add_parser("t", help="T = max(0.01, median |d|) of the train plan; writes the T file (never overwritten)", allow_abbrev=False)
    u.add_argument("--model", required=True)
    u.add_argument("--out", required=True, help="the T file (JSON); never overwritten")
    u.add_argument("--plans-dir", default=str(DEFAULT_PLANS_DIR), help="the plans of B (read only): the train plan file is cross-checked against the sampled plan")
    u.add_argument("--dry-run-synthetic", type=int, default=0, metavar="N_ITEMS", help="CPU dry run on N fake items instead of the real data; never for results")

    t = sub.add_parser("train", help="train one qr LoRA run (the objective is fixed to qr)", allow_abbrev=False)
    t.add_argument("--model", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--beta", type=float, required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--save-at", type=B.parse_save_at, default=B.parse_save_at("250,500,1000"))
    t.add_argument("--out", required=True)
    t.add_argument("--model-path", default=None, help="override the model directory of the configuration (the tiny test model)")
    t.add_argument("--device", default="cpu", help="cpu (default) or an explicit cuda:N; a bare 'cuda' is refused. A CPU run hides all GPUs from the process.")
    t.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    t.add_argument("--batch", type=int, default=B.BATCH, help="decision prompts and P-answer prompts per step")
    t.add_argument("--micro-batch", type=int, default=B.DEFAULT_MICRO_BATCH, help="rows per forward and backward pass; the optimiser step is over the whole batch")
    t.add_argument("--warmup", type=int, default=B.WARMUP_STEPS)
    t.add_argument("--max-grad-norm", type=float, default=B.DEFAULT_MAX_GRAD_NORM, help="0 disables clipping")
    t.add_argument("--no-grad-checkpointing", action="store_true")
    t.add_argument("--attn-implementation", default="sdpa")
    t.add_argument("--allow-offgrid", action="store_true", help="allow an lr, beta or model outside the frozen candidates (tests and dry runs only)")
    t.add_argument("--tau-file", default=None, help="the tau file of this model (B's tau command; the same file that routing uses); required")
    t.add_argument("--t-file", default=None, help="the T file of this model (the t command); required")
    t.add_argument("--time-only", type=int, default=None, metavar="N", help="N steps of the formal configuration and one adapter save, timed; writes timing.json, marks the manifest timing_only; select refuses the directory")
    t.add_argument("--dry-run-synthetic", type=int, default=0, metavar="N_ITEMS", help="CPU dry run on N fake items instead of the real data; never for results")

    s = sub.add_parser("select", help="choose a checkpoint of the qr search of one model from the validation readouts of B (pending if the search is incomplete)", allow_abbrev=False)
    s.add_argument("--model", required=True)
    s.add_argument("--runs", nargs="+", required=True, help="the seed-0 qr run directories of the 4 (lr, beta) cells")
    s.add_argument("--val-out", required=True, help="the read output root of B: <val-out>/<tag>/val/rows.jsonl and <tag>/val_panswer/rows.jsonl, tag = base | <run name>_step<N>")
    s.add_argument("--tau-file", required=True, help="the tau file of the model: the base model files and the tau of every candidate are bound to it")
    s.add_argument("--t-file", required=True, help="the T file of the model: every candidate must have been trained with it")
    s.add_argument("--out", required=True, help="where selection.json is written (the record of the search: selected, no_selectable_checkpoint or pending)")
    s.add_argument("--allow-partial", action="store_true", help="accept an incomplete grid (tests only)")
    return p


def main(argv: Sequence[str] | None = None, deps: B.Deps | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("t", "select"):
        B.prepare_device(torch.device("cpu"))  # these commands never need a GPU
    elif B.resolve_device(args.device).type == "cpu":
        B.prepare_device(torch.device("cpu"))  # before the module of B and the libraries are imported; a GPU run settles its card inside train
    if args.command == "select":
        sel = run_select(args.model, args.runs, args.val_out, args.tau_file, args.t_file, args.out, args.allow_partial,
                         expect_rows=None if args.allow_partial else (B.EXPECTED_VAL_ROWS, B.EXPECTED_VAL_ITEMS))
        print(format_selection(sel))
        return {STATUS_SELECTED: 0, STATUS_NONE: EXIT_NO_SELECTABLE, STATUS_PENDING: EXIT_PENDING}[sel["status"]]
    if deps is None:
        if args.dry_run_synthetic:
            deps = B.synthetic_deps(importlib.import_module("_part3_b_prompts"), n_items=args.dry_run_synthetic, per_item=4)
        else:
            deps = B.real_deps()
    if args.command == "t":
        doc = compute_t(args, deps)
        print(json.dumps({"out": args.out, "model": doc["model_id"], "T": doc["T"], "median_abs_d": doc["median_abs_d"], "abs_d_quantiles": doc["abs_d_quantiles"], "share_d_positive": doc["share_d_positive"],
                          "expected": doc["expected"]}, indent=2))
        return 0
    if args.command == "train":
        manifest = train(args, deps)
        print(json.dumps({"status": manifest["status"], "out": args.out, "adapters": manifest["adapters"], "elapsed_sec": manifest["elapsed_sec"]}))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
