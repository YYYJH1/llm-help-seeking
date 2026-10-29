"""Calibration of the verbalised confidence and its three gates.

The raw confidence ``c`` of an end-side model (``nmi.scenario.readout``) is turned into a probability ``z`` that its P-answer is correct by a TWO-PARAMETER logistic map

    z = sigmoid(a + b * logit(c~)),      c~ = c clipped to [1e-4, 1 - 1e-4],      b >= 0,

fitted by minimum log loss on the correctness of the answers. A single temperature cannot remove an overall bias (a model with accuracy 0.195 and mean confidence 0.958 keeps a mean ``z`` of at least
0.458 for every positive temperature); the intercept ``a`` can, and ``b >= 0`` keeps a more confident answer from being given a lower probability.

* CROSS-FITTING: 5 folds, the fold of an item is ``int(sha256(item_key), 16) mod 5`` (sha256 of the key string, read as a hexadecimal number), which gives the out-of-fold ``z`` of every development item.
  A second fit on the whole development pool is frozen and used for the sealed pool.
* FAILURE: a fit that does not converge, whose parameters are not finite or run away (|a| or b above 25 on the logit scale: the likelihood has no maximum because the data are separated), or a training
  fold with one class only, is a calibration failure. Any failed fold or a failed final fit makes the calibration gate fail; ``z`` is then invalid for the model (the analyses that need it are not testable).
* THE CALIBRATION GATE (the three gates, on the DEVELOPMENT pool only; any gate that fails makes ``z`` invalid):
  1. digit quality (``digit_quality``): the share of ALL REGISTERED items whose digit mass (the total probability of the ten digit tokens) is at least 0.9 must be at least 99 %. An item without a usable digit
     distribution (its answer-phase result is missing, or its digit mass is 0 or not a number) does not pass; it stays in the denominator;
  2. discrimination: the AUROC of the raw ``c`` for the correctness of the answer, with a bootstrap over items (2,000 resamples, seed 0, percentile interval), must have a 95 % lower bound above 0.5;
  3. calibration: the out-of-fold log loss must be below that of the constant prediction (the development accuracy), and the out-of-fold ECE (10 equal-width bins) at most 0.10.
Items with no answer-phase result (missing) take no part in the fits, the AUROC or the calibration gate, and count against gate 1 through ``n_registered``; the rule that a missing share above 1 % makes ``z``
invalid is applied by the caller (``scripts/run_scenario.py``). An item whose digit mass is 0 has no ``c``: it counts against gate 1 and takes no part in the fits.

The fits and gates 2 and 3 exist for the development pool only. A confirmatory (sealed) batch gets no refit and no gate 2 or 3: its ``z`` is the frozen final fit applied to its confidences, and only
the QUALITY GATE of ``nmi.scenario.analysis`` (digit quality, with the same ``digit_quality`` and the same denominator rule, and answer-phase missing share at most 1 %) is checked there.

Nothing here sees a question, an answer or a generated text.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from nmi.measure.calibration import auroc, expected_calibration_error

N_FOLDS = 5
CLIP = 1e-4
DIGIT_MASS_MIN = 0.9
DIGIT_PASS_RATE_MIN = 0.99
AUROC_RESAMPLES = 2000
AUROC_SEED = 0
AUROC_LEVEL = 0.95
ECE_BINS = 10
ECE_MAX = 0.10
MAX_ABS_PARAM = 100.0  # the box of the optimiser
SEPARATION_LIMIT = 25.0  # |a| or b beyond this on the logit scale marks (quasi-)separation: the likelihood has no maximum (the limit of nmi.measure.calibration.fit_logistic)
KKT_TOLERANCE = 1e-6


def fold_of(item_key: str, n_folds: int = N_FOLDS) -> int:
    """``int(sha256(item_key), 16) mod n_folds``: the digest of the key string, read as a hexadecimal number."""
    return int(hashlib.sha256(item_key.encode("utf-8")).hexdigest(), 16) % n_folds


def clipped_logit(c: Any) -> np.ndarray:
    """logit of c clipped to [CLIP, 1 - CLIP]."""
    return logit(np.clip(np.asarray(c, dtype=np.float64), CLIP, 1.0 - CLIP))


@dataclass(frozen=True)
class Fit:
    """One fit of the two-parameter map. ``ok`` is False when it failed (``reason`` says how, without data); a and b are then NaN."""

    a: float
    b: float
    ok: bool
    n: int
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "ok": self.ok, "n": self.n, "reason": self.reason}

    def eta(self, c: Any) -> np.ndarray:
        return self.a + self.b * clipped_logit(c)

    def z(self, c: Any) -> np.ndarray:
        """The calibrated probability of the raw confidence(s) ``c``."""
        return expit(self.eta(c))


def _failed(n: int, reason: str) -> Fit:
    return Fit(math.nan, math.nan, False, n, reason)


def _loss_and_grad(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    eta = theta[0] + theta[1] * x
    loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    residual = expit(eta) - y
    return loss, np.array([residual.mean(), (residual * x).mean()])


def fit_two_parameter(c: Any, correct: Any) -> Fit:
    """Minimum log loss of ``correct`` (0/1) on ``z = sigmoid(a + b logit(c~))`` with ``b >= 0``. The problem is convex; the solution is accepted only if it satisfies the optimality conditions."""
    cc, y = np.asarray(c, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    n = int(cc.size)
    if cc.ndim != 1 or cc.shape != y.shape:
        raise ValueError("c and correct must be vectors of the same length")
    if n == 0:
        return _failed(0, "no items")
    if not np.all(np.isfinite(cc)) or np.any(cc < 0.0) or np.any(cc > 1.0):
        return _failed(n, "a confidence is not in [0, 1]")
    if y.min() == y.max():
        return _failed(n, "one class only")
    x = clipped_logit(cc)
    start = np.array([float(logit(np.clip(y.mean(), CLIP, 1.0 - CLIP))), 0.0])
    try:
        result = minimize(
            _loss_and_grad, start, args=(x, y), jac=True, method="L-BFGS-B", bounds=[(-MAX_ABS_PARAM, MAX_ABS_PARAM), (0.0, MAX_ABS_PARAM)],
            options={"maxiter": 20000, "maxfun": 100000, "ftol": 1e-15, "gtol": 1e-11},
        )
    except (FloatingPointError, ValueError, ArithmeticError):
        return _failed(n, "the optimiser raised")
    a, b = float(result.x[0]), float(result.x[1])
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(float(result.fun))):
        return _failed(n, "parameters are not finite")
    if abs(a) > SEPARATION_LIMIT or b > SEPARATION_LIMIT:
        return _failed(n, "no maximum: the parameters run away (separated data)")
    _, grad = _loss_and_grad(np.array([a, b]), x, y)
    at_lower = b <= 1e-12
    if abs(grad[0]) > KKT_TOLERANCE or (grad[1] < -KKT_TOLERANCE if at_lower else abs(grad[1]) > KKT_TOLERANCE):
        return _failed(n, "did not converge (optimality conditions not met)")
    return Fit(a, b, True, n)


# ---------------------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------------------

def logloss_from_eta(eta: np.ndarray, y: np.ndarray) -> float:
    """Mean log loss of the predictions sigmoid(eta) for the 0/1 labels y (stable for any eta)."""
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta))


def constant_logloss(y: np.ndarray) -> float:
    """Mean log loss of predicting the accuracy ``y.mean()`` for every item (0 when all items are alike)."""
    p = float(np.mean(y))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def bootstrap_auroc(c: Any, correct: Any, *, resamples: int = AUROC_RESAMPLES, seed: int = AUROC_SEED, level: float = AUROC_LEVEL) -> dict[str, Any]:
    """AUROC of ``c`` for ``correct`` with a percentile bootstrap interval over items (a resample with one class only is skipped and counted)."""
    cc, y = np.asarray(c, dtype=np.float64), np.asarray(correct).astype(bool)
    n = int(cc.size)
    point = auroc(cc, y) if n else None
    out: dict[str, Any] = {"value": point, "lower": None, "upper": None, "n": n, "resamples": resamples, "seed": seed, "level": level, "skipped": 0}
    if point is None:
        return out
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(resamples, n))
    values = []
    for idx in draws:
        v = auroc(cc[idx], y[idx])
        if v is not None:
            values.append(v)
    out["skipped"] = resamples - len(values)
    if values:
        tail = (1.0 - level) / 2.0 * 100.0
        out["lower"], out["upper"] = float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))
    return out


def digit_quality(masses: Any, n_registered: int | None = None) -> dict[str, Any]:
    """The digit-quality gate of a batch: the share of ALL REGISTERED items whose digit mass is at least ``DIGIT_MASS_MIN`` (0.9) must be at least ``DIGIT_PASS_RATE_MIN`` (99 %).

    ``masses`` are the digit masses of the items that have an answer-phase result (None or NaN: no digit distribution was read); ``n_registered`` is the number of all registered items of the batch (default:
    ``len(masses)``, nobody missing), so the registered items without a result, those with a mass of 0 and those with no mass at all count as NOT passing and stay in the denominator. The same function is the
    gate 1 of the development pool (``calibrate``) and the digit-quality check of the quality gate of any batch (``nmi.scenario.analysis.quality_gate``). Plain data comes back."""
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    n_have = int(values.size)
    n_reg = n_have if n_registered is None else int(n_registered)
    if n_reg < n_have:
        raise ValueError(f"n_registered ({n_reg}) is smaller than the number of masses ({n_have})")
    usable = np.isfinite(values) & (values > 0.0)
    passing = usable & (values >= DIGIT_MASS_MIN)
    n_pass = int(passing.sum())
    n_missing = n_reg - n_have
    rate = n_pass / n_reg if n_reg else 0.0
    return {
        "value": rate, "threshold": DIGIT_PASS_RATE_MIN, "mass_min": DIGIT_MASS_MIN, "n": n_reg, "n_registered": n_reg, "n_pass": n_pass, "n_missing": n_missing,
        "n_no_mass": int((~usable).sum()), "n_low_mass": int((usable & ~passing).sum()), "n_no_distribution": n_missing + int((~usable).sum()),
        "denominator": "all registered items", "pass": bool(n_reg > 0 and rate >= DIGIT_PASS_RATE_MIN),
    }


@dataclass
class CalibrationResult:
    """The cross-fitted calibration of one model on the development pool, with the three gates (the calibration gate)."""

    n_items: int  # answer-phase items that are not missing
    n_fitted: int  # ... with a finite c
    accuracy: float | None
    folds: list[Fit]
    final: Fit
    gates: dict[str, dict[str, Any]]
    failures: list[str]
    z_valid: bool
    fold_index: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.int64))  # per input item, -1 for an item without a finite c
    z_oof: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))  # per input item, NaN where there is none
    n_registered: int = 0  # all registered items of the batch (the denominator of gate 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "calibration_gate", "scope": "development pool: the three gates, the cross-fitted and the frozen final fit (a confirmatory batch gets neither a refit nor gates 2 and 3)",
            "n_registered": self.n_registered, "n_items": self.n_items, "n_fitted": self.n_fitted, "accuracy": self.accuracy, "n_folds": N_FOLDS, "clip": CLIP, "folds": [f.as_dict() for f in self.folds],
            "final": self.final.as_dict(), "gates": self.gates, "failures": self.failures, "z_valid": self.z_valid,
        }


def calibrate(item_keys: Sequence[str], c: Any, correct: Any, digit_mass: Any, *, n_registered: int | None = None) -> CalibrationResult:
    """Cross-fit the map and judge the three gates (the calibration gate of the development pool). The four inputs are per ANSWER-PHASE item that is not missing, in the same order; ``c`` is NaN where the digit
    mass is 0. ``n_registered`` is the number of ALL registered items of the batch, the denominator of gate 1 (default: the number of items given, nobody missing)."""
    keys = list(item_keys)
    cc = np.asarray(c, dtype=np.float64)
    y_all = np.asarray(correct).astype(np.float64)
    mass = np.asarray(digit_mass, dtype=np.float64)
    n = len(keys)
    if not (cc.shape == y_all.shape == mass.shape == (n,)):
        raise ValueError("item_keys, c, correct and digit_mass must have the same length")
    registered = n if n_registered is None else int(n_registered)
    if registered < n:
        raise ValueError(f"n_registered ({registered}) is smaller than the number of items given ({n})")
    failures: list[str] = []
    usable = np.isfinite(cc)
    folds_all = np.array([fold_of(k) for k in keys], dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
    fold_index = np.where(usable, folds_all, -1)
    z_oof = np.full(n, np.nan)
    fits: list[Fit] = []
    for k in range(N_FOLDS):
        train = usable & (folds_all != k)
        fit = fit_two_parameter(cc[train], y_all[train]) if train.any() else _failed(0, "no items")
        fits.append(fit)
        if fit.ok:
            test = usable & (folds_all == k)
            z_oof[test] = fit.z(cc[test])
        else:
            failures.append(f"fold {k}: {fit.reason}")
    final = fit_two_parameter(cc[usable], y_all[usable]) if usable.any() else _failed(0, "no items")
    if not final.ok:
        failures.append(f"final fit: {final.reason}")

    # gate 1: digit quality, over all registered items (an item without a result or without mass does not pass)
    gate1 = digit_quality(mass, registered)
    # gate 2: discrimination of the raw confidence
    boot = bootstrap_auroc(cc[usable], y_all[usable]) if usable.any() else bootstrap_auroc([], [])
    gate2 = {**boot, "threshold": 0.5, "pass": bool(boot["lower"] is not None and boot["lower"] > 0.5)}
    # gate 3: out-of-fold calibration
    have_z = np.isfinite(z_oof)
    all_folds_ok = all(f.ok for f in fits) and final.ok and bool(usable.any()) and bool(have_z.sum() == usable.sum())
    gate3: dict[str, Any] = {"logloss_oof": None, "logloss_constant": None, "ece_oof": None, "ece_bins": ECE_BINS, "ece_max": ECE_MAX, "fits_ok": all_folds_ok, "pass": False}
    if all_folds_ok:
        y = y_all[usable]
        eta_oof = np.empty(int(usable.sum()))
        usable_idx = np.flatnonzero(usable)
        for pos, i in enumerate(usable_idx):
            eta_oof[pos] = fits[int(folds_all[i])].eta(cc[i])
        ll, ll0 = logloss_from_eta(eta_oof, y), constant_logloss(y)
        ece = expected_calibration_error(z_oof[usable], y, ECE_BINS).ece
        gate3.update(logloss_oof=ll, logloss_constant=ll0, ece_oof=float(ece), pass_logloss=bool(ll < ll0), pass_ece=bool(ece <= ECE_MAX))
        gate3["pass"] = bool(ll < ll0 and ece <= ECE_MAX)
    gates = {"digit_quality": gate1, "auroc": gate2, "calibration": gate3}
    z_valid = bool(all(g["pass"] for g in gates.values()) and not failures)
    return CalibrationResult(
        n_items=n, n_fitted=int(usable.sum()), accuracy=float(y_all[usable].mean()) if usable.any() else None, folds=fits, final=final, gates=gates, failures=failures, z_valid=z_valid,
        fold_index=fold_index, z_oof=z_oof, n_registered=registered,
    )
