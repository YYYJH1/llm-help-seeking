"""Confidence calibration and the rational benchmark r_hat(z, w).

Contents

* temperature scaling on the P-answer option log-probabilities, the calibrated confidence z,
  ECE (15 equal-width bins) and AUROC;
* a logistic-regression core (``fit_logistic``), shared with the psychometric model;
* the correctness map r_hat(z, w) in two versions, both fitted on the calibration split from the
  P-ask self-answer correctness at a_H = 0.8, c0 = 0 and w in the fitted set:

  - **main (``calibration.correctness_map.method`` in the configuration): ``LogisticRHat``**,
    ``logit r = alpha + beta * logit(z~) + gamma * w`` with
    ``z~ = clip(z, z_clip)`` (``logistic.z_clip`` of the configuration, [0.25, 1 - 1e-4]), fitted by maximum
    likelihood on the stacked rows. Canonical threshold ``logit(T*) = (logit t - alpha - gamma w) / beta`` for the target
    ``t = a_H - c0 - w``; ``r_z = beta t (1 - t) / (T* (1 - T*))``, ``r_w = gamma t (1 - t)``,
    ``theta* = (1 + r_w) / r_z``. NA (with a reason code) when ``beta <= 0``, when ``t`` is not in (0, 1),
    when ``T*`` lies outside the observed range of the calibration z, when ``r_z <= 0.01`` or when
    ``theta* <= 0.01`` (and, as a safeguard, when the fit did not converge). There is no r_w == 0 test.
  - **sensitivity: ``RHatFamily``**, the isotonic version (one isotonic fit per fitted w,
    the r_w == 0 test, interpolation between fitted w, ``T*`` as a lower inverse, ``r_z`` by +-0.05
    differences). Its result fields are marked ``method="isotonic"``
    (``sensitivity_method`` of the configuration).
  - **shape check** (``logistic_shape_check``): the misfit rule (design-effect-adjusted residuals, the
    same thresholds) applied to the logistic fit on z deciles x fitted w, with the rows that fitted it, and a
    chi-square omnibus test beside it. Descriptive only (with 1,000 items neither has
    power): a severe misfit marks the result ``shape_sensitive`` and changes no value. R is always reported with both
    versions.

Conventions of the isotonic version:

* z is the maximum of the temperature-scaled softmax over the four options.
* ``IsotonicCurve``: isotonic (non-decreasing) regression of the 0/1 correctness on z. Equal z
  values are pooled first. The curve is linear between the knots, where the knots are the first
  and the last z of every constant block (this is what sklearn's ``IsotonicRegression``
  predicts with ``out_of_bounds="clip"``). Outside the data range the curve is constant.
* The "common support" of two curves is the intersection of their z ranges; the maximum
  absolute difference of two piecewise-linear curves is taken exactly (at the union of knots and
  at the ends of the common support).
* r_w == 0 (all fitted curves against the w = 0.2 curve): when it holds, the curve at a
  target w is the fitted curve with the nearest w (the smaller w on an exact tie). Otherwise the
  curve is the linear interpolation in w between the two neighbouring fitted curves, with w
  clamped to [min fitted w, max fitted w]. The target value a_H - c0 - w always uses the real w.
* r_w from the fitted curves: between two nodes, the difference quotient of those two curves;
  exactly on an interior node, the central quotient of its two neighbours; on an end node, the
  one-sided quotient.
* T* = inf {z : r_hat(z, w) >= a_H - c0 - w}; NA when the target lies outside the value range of
  r_hat(., w) (a tolerance of 1e-12 absorbs floating point noise).
* r_z: central difference with step 0.05 at T*; one-sided at a support edge; NA when r_z <= 0.01
  (and when the support is too narrow for either difference).
* theta* = (1 + r_w) / r_z; NA when theta* <= 0.01.

Why the logistic version is the main one: with 1,000 calibration items the
isotonic r_hat is very noisy and the r_w == 0 rule almost never fires. A validation against known truths
(median and 5-95% range of theta* / truth, both versions; logistic, linear and non-logistic truths) shows that
the logistic version is unbiased for a logistic truth (r = z is one) and has a much narrower range, but that it is
biased where the truth is not logistic in logit(z) (linear: about +17% at w = 0.2, with a sign that changes with
w). The shape check is what exposes such a case. Which version is main is read from the configuration
(``RHatSpec.main_method``); both are always computed and reported.

NA results carry a ``reason`` string and a ``reason_code``; numbers that are undefined are ``None``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import brentq, isotonic_regression
from scipy.special import expit, logsumexp
from scipy.stats import chi2, rankdata

# ---------------------------------------------------------------------------------------
# temperature scaling, ECE, AUROC
# ---------------------------------------------------------------------------------------

TEMPERATURE_BOUNDS = (1e-2, 1e2)  # search range of the temperature
R_HAT_METHODS = ("logistic", "isotonic")
DEFAULT_MAIN_R_HAT = "logistic"  # used only when the configuration names no main method
Z_CLIP = (0.25, 1.0 - 1e-4)  # default of logistic.z_clip: four options give z >= 0.25; the cap keeps logit(z~) finite
# The reason codes ``LogisticRHat.normative`` can give; the configuration lists the same six (logistic.na).
LOGISTIC_NA_CODES = frozenset(
    {
        "beta_not_positive", "target_outside_unit_interval", "threshold_outside_observed_range", "plateau",
        "theta_star_too_small", "fit_not_converged",
    }
)
_LOGIT_FLOOR = -1e4  # log-probabilities of -inf are floored here (a zero-probability option)


class CalibrationError(ValueError):
    """Input of a calibration function is malformed."""


def _as_log_p(log_p: Any) -> np.ndarray:
    arr = np.asarray(log_p, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise CalibrationError(f"option log-probabilities must have shape (n, k >= 2), got {arr.shape}")
    if np.any(np.isnan(arr)) or np.any(arr == np.inf):
        raise CalibrationError("option log-probabilities must not contain NaN or +inf")
    return np.maximum(arr, _LOGIT_FLOOR)


def scaled_option_probs(log_p: Any, temperature: float) -> np.ndarray:
    """Softmax of ``log_p / temperature`` over the options (rows sum to one)."""
    if not (temperature > 0 and math.isfinite(temperature)):
        raise CalibrationError(f"temperature must be positive and finite, got {temperature!r}")
    scaled = _as_log_p(log_p) / temperature
    return np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))


def confidence(log_p: Any, temperature: float) -> np.ndarray:
    """Calibrated confidence z: the largest scaled option probability."""
    return scaled_option_probs(log_p, temperature).max(axis=1)


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll: float  # mean negative log-likelihood of the correct option at ``temperature``
    nll_unscaled: float  # the same at temperature 1
    at_bound: str | None  # "lower" or "upper" when the minimiser hit the search range
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature, "nll": self.nll, "nll_unscaled": self.nll_unscaled,
            "at_bound": self.at_bound, "n": self.n,
        }


def _mean_nll(log_p: np.ndarray, label: np.ndarray, beta: float) -> float:
    scaled = beta * log_p
    lse = logsumexp(scaled, axis=1)
    return float(np.mean(lse - scaled[np.arange(len(label)), label]))


def fit_temperature(
    log_p: Any, correct_index: Any, *, bounds: tuple[float, float] = TEMPERATURE_BOUNDS
) -> TemperatureFit:
    """Temperature that minimises the negative log-likelihood of the correct option.

    ``log_p`` has shape (n, 4): the log-probabilities of the four options under P-answer;
    ``correct_index`` holds the 0-based index of the correct option. With beta = 1 / T the loss
    is convex in beta, so the minimiser is the root of its derivative
    ``mean(E_beta[l] - l_correct)``, found by Brent's method inside ``bounds``.
    """
    lp = _as_log_p(log_p)
    label = np.asarray(correct_index)
    if label.shape != (lp.shape[0],) or not np.issubdtype(label.dtype, np.integer):
        raise CalibrationError("correct_index must be an integer vector with one entry per item")
    if lp.shape[0] == 0:
        raise CalibrationError("temperature scaling needs at least one item")
    if label.min() < 0 or label.max() >= lp.shape[1]:
        raise CalibrationError("correct_index out of range")
    t_lo, t_hi = bounds
    if not (0 < t_lo < t_hi):
        raise CalibrationError(f"bad temperature bounds {bounds}")
    rows = np.arange(lp.shape[0])
    chosen = lp[rows, label]

    def slope(beta: float) -> float:
        scaled = beta * lp
        p = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
        return float(np.mean(np.sum(p * lp, axis=1) - chosen))

    beta_lo, beta_hi = 1.0 / t_hi, 1.0 / t_lo
    s_lo, s_hi = slope(beta_lo), slope(beta_hi)
    at_bound: str | None = None
    if s_lo >= 0:  # loss already rising at the flattest allowed distribution
        beta, at_bound = beta_lo, "upper"  # largest temperature
    elif s_hi <= 0:  # still falling at the sharpest allowed distribution
        beta, at_bound = beta_hi, "lower"
    else:
        beta = float(brentq(slope, beta_lo, beta_hi, xtol=1e-14, rtol=1e-13, maxiter=200))
    return TemperatureFit(
        temperature=1.0 / beta,
        nll=_mean_nll(lp, label, beta),
        nll_unscaled=_mean_nll(lp, label, 1.0),
        at_bound=at_bound,
        n=int(lp.shape[0]),
    )


@dataclass(frozen=True)
class ECEResult:
    ece: float
    bin_edges: tuple[float, ...]
    count: tuple[int, ...]
    mean_confidence: tuple[float | None, ...]
    accuracy: tuple[float | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": self.ece, "bin_edges": list(self.bin_edges), "count": list(self.count),
            "mean_confidence": list(self.mean_confidence), "accuracy": list(self.accuracy),
        }


def expected_calibration_error(conf: Any, correct: Any, n_bins: int = 15) -> ECEResult:
    """ECE with ``n_bins`` equal-width bins on [0, 1]; a bin is (lo, hi], the first one [0, hi]
    (a confidence within 1e-12 of an edge counts in the lower bin, so 0.2 = 3/15 is not pushed up by rounding).

    ECE = sum_b (n_b / n) |accuracy_b - mean confidence_b|.
    """
    c = np.asarray(conf, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if c.ndim != 1 or c.shape != y.shape or c.size == 0:
        raise CalibrationError("conf and correct must be non-empty vectors of the same length")
    if np.any(~np.isfinite(c)) or c.min() < 0 or c.max() > 1:
        raise CalibrationError("confidences must lie in [0, 1]")
    if n_bins < 1:
        raise CalibrationError("n_bins must be positive")
    idx = np.clip(np.ceil(c * n_bins - 1e-12).astype(np.int64) - 1, 0, n_bins - 1)  # 1e-12: an edge value belongs to the lower bin
    count = np.bincount(idx, minlength=n_bins)
    sum_conf = np.bincount(idx, weights=c, minlength=n_bins)
    sum_acc = np.bincount(idx, weights=y, minlength=n_bins)
    nonempty = count > 0
    mean_conf = np.where(nonempty, sum_conf / np.maximum(count, 1), np.nan)
    acc = np.where(nonempty, sum_acc / np.maximum(count, 1), np.nan)
    ece = float(np.sum(count[nonempty] / c.size * np.abs(acc[nonempty] - mean_conf[nonempty])))
    return ECEResult(
        ece=ece,
        bin_edges=tuple(float(e) for e in np.linspace(0.0, 1.0, n_bins + 1)),
        count=tuple(int(k) for k in count),
        mean_confidence=tuple(None if math.isnan(v) else float(v) for v in mean_conf),
        accuracy=tuple(None if math.isnan(v) else float(v) for v in acc),
    )


def auroc(score: Any, positive: Any) -> float | None:
    """Area under the ROC curve of ``score`` for the 0/1 label ``positive`` (Mann-Whitney U with
    mid-ranks for ties). ``None`` when one of the classes is empty."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(positive).astype(bool)
    if s.ndim != 1 or s.shape != y.shape:
        raise CalibrationError("score and positive must be vectors of the same length")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s, method="average")
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def calibration_columns(log_p: Any, correct_index: Any, temperature: float) -> dict[str, np.ndarray]:
    """Per-item columns of the calibration table: the calibrated confidence ``z``, the P-answer
    prediction ``pred`` (1-based option number, lowest number on a tie) and whether it is ``correct``."""
    lp = _as_log_p(log_p)
    label = np.asarray(correct_index)
    if label.shape != (lp.shape[0],):
        raise CalibrationError("correct_index must have one entry per item")
    pred = np.argmax(lp, axis=1)
    return {"z": confidence(lp, temperature), "pred": pred + 1, "correct": pred == label}


def confidence_report(log_p: Any, correct_index: Any, temperature: float, n_bins: int = 15) -> dict[str, Any]:
    """Accuracy, ECE (``n_bins`` equal-width bins) and AUROC of the calibrated confidence on one split."""
    cols = calibration_columns(log_p, correct_index, temperature)
    ece = expected_calibration_error(cols["z"], cols["correct"], n_bins)
    return {
        "n": int(len(cols["z"])), "temperature": float(temperature), "accuracy": float(np.mean(cols["correct"])),
        "mean_confidence": float(np.mean(cols["z"])), "ece": ece.ece, "ece_bins": ece.to_dict(),
        "auroc": auroc(cols["z"], cols["correct"]),
    }


# ---------------------------------------------------------------------------------------
# logistic regression core
# ---------------------------------------------------------------------------------------

_SEPARATION_BETA = 25.0  # |coefficient| beyond this on the logit scale marks (quasi-)separation


@dataclass(frozen=True)
class LogitResult:
    """Result of ``fit_logistic``."""

    beta: np.ndarray
    converged: bool
    n_iter: int
    loglik: float
    separation_suspected: bool


def _loglik(eta: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    beta0: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> LogitResult:
    """Maximum-likelihood logistic regression with frequency weights (Newton, step halving).

    The first column of ``X`` must be the intercept. ``y`` may be fractional (an expected correctness):
    the log-likelihood ``sum w (y eta - log(1 + e^eta))`` is then a quasi-likelihood with the same maximiser.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if y.shape != (n,) or w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
        raise CalibrationError("fit_logistic: inconsistent shapes or weights")
    if beta0 is None:
        beta = np.zeros(p)
        m = min(max(float(np.sum(w * y) / w.sum()), 1e-6), 1 - 1e-6)
        beta[0] = math.log(m / (1 - m))  # the first column is the intercept
    else:
        beta = np.array(beta0, dtype=np.float64)
    ll = _loglik(X @ beta, y, w)
    converged = False
    n_iter = 0
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        eta = X @ beta
        mu = expit(eta)
        grad = X.T @ (w * (y - mu))
        info = (X * (w * mu * (1.0 - mu))[:, None]).T @ X
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError as error:
            raise CalibrationError("singular information matrix (collinear design or separation)") from error
        t = 1.0
        while True:
            candidate = beta + t * step
            ll_new = _loglik(X @ candidate, y, w)
            if ll_new >= ll - 1e-12 or t < 1e-10:
                break
            t *= 0.5
        move = float(np.max(np.abs(candidate - beta)))
        beta, ll_old, ll = candidate, ll, ll_new
        if move < 1e-8 or abs(ll - ll_old) < tol * (1.0 + abs(ll)):
            converged = True
            break
    return LogitResult(beta, converged, n_iter, ll, bool(np.max(np.abs(beta)) > _SEPARATION_BETA))


# ---------------------------------------------------------------------------------------
# isotonic curves
# ---------------------------------------------------------------------------------------

_TOL = 1e-12


@dataclass(frozen=True)
class IsotonicCurve:
    """Non-decreasing piecewise-linear curve r(z) between block-boundary knots."""

    z_knots: np.ndarray
    r_knots: np.ndarray
    n: int  # number of observations (or total weight) behind the fit

    def __post_init__(self) -> None:
        zk, rk = np.asarray(self.z_knots, float), np.asarray(self.r_knots, float)
        if zk.ndim != 1 or zk.shape != rk.shape or zk.size == 0:
            raise CalibrationError("knots must be equal-length non-empty vectors")
        if np.any(np.diff(zk) <= 0):
            raise CalibrationError("z knots must be strictly increasing")
        if np.any(np.diff(rk) < -_TOL):
            raise CalibrationError("r knots must be non-decreasing")

    def __call__(self, z: Any) -> np.ndarray:
        """Evaluate with constant extension outside the data range."""
        return np.interp(z, self.z_knots, self.r_knots)

    @property
    def support(self) -> tuple[float, float]:
        return float(self.z_knots[0]), float(self.z_knots[-1])

    @property
    def value_range(self) -> tuple[float, float]:
        return float(self.r_knots[0]), float(self.r_knots[-1])


def fit_isotonic(z: Any, y: Any, weights: Any = None) -> IsotonicCurve:
    """Weighted isotonic regression of ``y`` on ``z`` (non-decreasing), tied z values pooled.

    ``weights`` are case weights (bootstrap multiplicities); rows with weight 0 are dropped.
    """
    zz = np.asarray(z, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if zz.ndim != 1 or zz.shape != yy.shape:
        raise CalibrationError("z and y must be vectors of the same length")
    ww = np.ones_like(zz) if weights is None else np.asarray(weights, dtype=np.float64)
    if ww.shape != zz.shape or np.any(ww < 0) or np.any(~np.isfinite(ww)):
        raise CalibrationError("weights must be finite, non-negative and match z")
    if np.any(~np.isfinite(zz)) or np.any(~np.isfinite(yy)):
        raise CalibrationError("z and y must be finite")
    keep = ww > 0
    zz, yy, ww = zz[keep], yy[keep], ww[keep]
    if zz.size == 0:
        raise CalibrationError("isotonic fit needs at least one observation")
    order = np.argsort(zz, kind="stable")
    zz, yy, ww = zz[order], yy[order], ww[order]
    uz, start = np.unique(zz, return_index=True)
    sum_w = np.add.reduceat(ww, start)
    mean_y = np.add.reduceat(ww * yy, start) / sum_w
    res = isotonic_regression(mean_y, weights=sum_w, increasing=True)
    fitted, blocks = np.asarray(res.x), np.asarray(res.blocks)
    zk: list[float] = []
    rk: list[float] = []
    for j in range(len(blocks) - 1):
        first, last = int(blocks[j]), int(blocks[j + 1]) - 1
        value = float(fitted[first])
        zk.append(float(uz[first]))
        rk.append(value)
        if last != first:
            zk.append(float(uz[last]))
            rk.append(value)
    return IsotonicCurve(np.asarray(zk), np.asarray(rk), n=int(round(float(ww.sum()))))


def common_support(a: IsotonicCurve, b: IsotonicCurve) -> tuple[float, float] | None:
    lo, hi = max(a.support[0], b.support[0]), min(a.support[1], b.support[1])
    return (lo, hi) if lo <= hi else None


def max_abs_difference(a: IsotonicCurve, b: IsotonicCurve) -> float | None:
    """Exact maximum of |a(z) - b(z)| over the common support; ``None`` without a common support."""
    support = common_support(a, b)
    if support is None:
        return None
    lo, hi = support
    grid = np.concatenate([a.z_knots, b.z_knots, [lo, hi]])
    grid = np.unique(grid[(grid >= lo) & (grid <= hi)])
    return float(np.max(np.abs(a(grid) - b(grid))))


def inverse_lower(z_knots: np.ndarray, r_knots: np.ndarray, target: float) -> float | None:
    """inf {z : r(z) >= target} for a non-decreasing piecewise-linear r; ``None`` outside the range.

    The range test uses a tolerance of 1e-12 on both ends.
    """
    r_min, r_max = float(r_knots[0]), float(r_knots[-1])
    if target < r_min - _TOL or target > r_max + _TOL:
        return None
    k = int(np.searchsorted(r_knots, target - _TOL, side="left"))
    if k >= len(r_knots):  # target within tolerance above the last value
        return float(z_knots[-1])
    if k == 0:
        return float(z_knots[0])
    r0, r1 = float(r_knots[k - 1]), float(r_knots[k])
    t = min(max((target - r0) / (r1 - r0), 0.0), 1.0)
    return float(z_knots[k - 1] + t * (z_knots[k] - z_knots[k - 1]))


# ---------------------------------------------------------------------------------------
# binned residual diagnostic (misfit rule and the shape check of the logistic r_hat)
# ---------------------------------------------------------------------------------------

_SEVERE_RE = re.compile(r">\s*([0-9.]+)\s+exceeds\s+([0-9.]+)")


def misfit_rule_from_config(cfg: Mapping[str, Any]) -> tuple[int, float, float]:
    """``(z bins, |standardised residual| threshold, maximum share of flagged bins)`` of ``psychometrics.misfit``."""
    rule = cfg["psychometrics"]["misfit"]
    match = _SEVERE_RE.search(str(rule["severe_if"]))
    if match is None or "deciles" not in str(rule["bins"]):
        raise CalibrationError(f"cannot read the misfit rule from {dict(rule)!r}")
    return 10, float(match.group(1)), float(match.group(2))


@dataclass(frozen=True)
class MisfitResult:
    """Binned standardised residuals (misfit rule, and the shape check of the logistic r_hat).

    The ``adjusted_`` fields decide: the residuals are divided by ``sqrt(phi * variance)`` where ``phi``
    (``dispersion_factor``, at least 1) is the design effect of the item clusters, pooled over all bins: the sum
    over bins of the item-clustered residual variance divided by the sum of the binomial variances. The fields
    without a prefix are the literal binomial diagnostic (variance ``sum p (1 - p)`` of a bin, rows treated as
    independent), kept as a descriptive quantity: rows of one item share the item's idiosyncrasies, so it is
    over-dispersed when the model is right on average.
    """

    n_bins_defined: int
    n_flagged: int
    share_flagged: float | None
    severe: bool
    max_abs_residual: float | None
    abs_threshold: float
    max_share: float
    dispersion_factor: float = 1.0
    adjusted_n_flagged: int = 0
    adjusted_share_flagged: float | None = None
    adjusted_severe: bool = False
    adjusted_max_abs_residual: float | None = None
    # descriptive omnibus (not part of any rule): sum of the squared adjusted residuals of the defined bins; the
    # p-value (given only when the number of fitted parameters is known) uses chi2(bins - parameters) and is liberal
    # when the bins share items (labels shared across w); it has far more power than the 3-sigma rule
    chi_square: float | None = None
    chi_square_df: int | None = None
    chi_square_p_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _bin_summary(resid: np.ndarray, threshold: float, max_share: float) -> tuple[int, int, float | None, bool, float | None]:
    if resid.size == 0:
        return 0, 0, None, False, None
    flagged = int(np.sum(np.abs(resid) > threshold))
    share = flagged / resid.size
    return int(resid.size), flagged, share, share > max_share, float(np.max(np.abs(resid)))


def binned_misfit(
    z: Any,
    w: Any,
    y: Any,
    p: Any,
    item: Any,
    n_items: int,
    *,
    z_bins: int,
    abs_threshold: float,
    max_share: float,
    item_weight: Any = None,
    n_fitted_params: int | None = None,
) -> MisfitResult:
    """Standardised residuals over ``z_bins`` quantile bins of z times the distinct ``w`` levels.

    ``y`` are the 0/1 outcomes, ``p`` the fitted probabilities, ``item`` the cluster (item position) of every row.
    A bin's residual is ``sum(y - p) / sqrt(phi * sum p (1 - p))``; the fit is severely misfitting when more than
    ``max_share`` of the bins with a defined residual (data and variance) have ``|residual| > abs_threshold``.
    ``item_weight`` are integer item multiplicities (a bootstrap resample): the copies of an item count as separate
    items, the z quantiles are those of the resampled rows, and the cluster variance is ``sum_items k_i s_i^2``
    with ``s_i`` the item's residual sum in the bin. ``n_fitted_params`` only serves the descriptive chi-square p-value.
    """
    z, w, y, p = (np.asarray(v, dtype=np.float64) for v in (z, w, y, p))
    item = np.asarray(item, dtype=np.int64)
    if not (z.ndim == 1 and z.shape == w.shape == y.shape == p.shape == item.shape) or z.size == 0:
        raise CalibrationError("z, w, y, p and item must be non-empty vectors of the same length")
    if item.min() < 0 or item.max() >= n_items:
        raise CalibrationError("item must be positions in [0, n_items)")
    if item_weight is None:
        k_item = np.ones(n_items)
        z_edges_source = z
    else:
        k_item = np.asarray(item_weight, dtype=np.float64)
        if k_item.shape != (n_items,) or np.any(k_item < 0) or np.any(k_item != np.round(k_item)):
            raise CalibrationError("item_weight must be non-negative integer multiplicities, one per item")
        z_edges_source = np.repeat(z, k_item[item].astype(np.int64))
        if z_edges_source.size == 0:
            raise CalibrationError("no row has positive weight")
    row_k = k_item[item]
    edges = np.quantile(z_edges_source, np.linspace(0.0, 1.0, z_bins + 1))
    z_bin = np.searchsorted(edges[1:-1], z, side="right")
    levels = np.unique(np.round(w, 9))
    w_bin = np.searchsorted(levels, np.round(w, 9))
    cell = z_bin * len(levels) + w_bin
    n_cells = z_bins * len(levels)
    count = np.bincount(cell, weights=row_k, minlength=n_cells)
    residual = y - p
    observed_minus_expected = np.bincount(cell, weights=row_k * residual, minlength=n_cells)
    variance = np.bincount(cell, weights=row_k * p * (1.0 - p), minlength=n_cells)
    defined = (count > 0) & (variance > 1e-12)
    literal = _bin_summary(observed_minus_expected[defined] / np.sqrt(variance[defined]), abs_threshold, max_share)
    # design effect of the item clusters, pooled over the bins: squared residual sum of every item inside a bin
    item_sums = np.bincount(cell * n_items + item, weights=residual, minlength=n_cells * n_items)
    cluster_variance = np.sum(k_item[None, :] * item_sums.reshape(n_cells, n_items) ** 2, axis=1)
    phi = 1.0
    adjusted: tuple[int, int, float | None, bool, float | None] = (0, 0, None, False, None)
    chi_square = chi_df = chi_p = None
    if defined.any():
        phi = max(1.0, float(cluster_variance[defined].sum() / variance[defined].sum()))
        adjusted_residual = observed_minus_expected[defined] / np.sqrt(phi * variance[defined])
        adjusted = _bin_summary(adjusted_residual, abs_threshold, max_share)
        chi_square = float(np.sum(adjusted_residual**2))
        if n_fitted_params is not None and int(defined.sum()) > n_fitted_params:
            chi_df = int(defined.sum()) - n_fitted_params
            chi_p = float(chi2.sf(chi_square, chi_df))
    return MisfitResult(
        n_bins_defined=literal[0], n_flagged=literal[1], share_flagged=literal[2], severe=literal[3],
        max_abs_residual=literal[4], abs_threshold=abs_threshold, max_share=max_share,
        dispersion_factor=phi, adjusted_n_flagged=adjusted[1], adjusted_share_flagged=adjusted[2],
        adjusted_severe=adjusted[3], adjusted_max_abs_residual=adjusted[4],
        chi_square=chi_square, chi_square_df=chi_df, chi_square_p_value=chi_p,
    )


# ---------------------------------------------------------------------------------------
# r_hat(z, w) family
# ---------------------------------------------------------------------------------------

_THETA_NA_RE = re.compile(r"NA if\s*<=\s*([0-9.]+)")


@dataclass(frozen=True)
class RHatSpec:
    """Numbers of ``calibration.correctness_map`` in the configuration."""

    fitted_w: tuple[float, ...]
    reference_w: float  # the curve every other curve is compared with in the r_w test (w = 0.2)
    r_w_zero_max_diff: float  # r_w == 0 when every max |difference| is <= this (0.03)
    r_z_step: float  # 0.05
    r_z_na_if_le: float  # 0.01
    theta_star_na_if_le: float  # 0.01
    fixed_a_H: float
    fixed_c0: float
    z_clip: tuple[float, float] = Z_CLIP  # logistic.z_clip
    main_method: str = DEFAULT_MAIN_R_HAT  # correctness_map.method
    sensitivity_method: str = "isotonic"  # correctness_map.sensitivity_method
    shape_z_bins: int = 10  # the shape check follows the misfit rule: z deciles ...
    shape_abs_residual: float = 3.0  # ... |standardised residual| > 3 ...
    shape_max_share: float = 0.10  # ... in more than 10% of the bins

    def __post_init__(self) -> None:
        if len(self.fitted_w) < 2 or list(self.fitted_w) != sorted(set(self.fitted_w)):
            raise CalibrationError("fitted_w must be at least two strictly increasing values")
        if not any(abs(self.reference_w - w) < 1e-9 for w in self.fitted_w):
            raise CalibrationError("reference_w must be one of fitted_w")
        lower, upper = self.z_clip
        if not 0.25 <= lower < upper < 1.0:  # z is a four-option confidence: at least 0.25 and below 1
            raise CalibrationError(f"z_clip must satisfy 0.25 <= lower < upper < 1, got {self.z_clip}")
        if self.main_method not in R_HAT_METHODS or self.sensitivity_method not in R_HAT_METHODS:
            raise CalibrationError(f"r_hat methods must be among {R_HAT_METHODS}")
        if self.main_method == self.sensitivity_method:
            raise CalibrationError("the sensitivity method must differ from the main method")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RHatSpec":
        c = cfg["calibration"]["correctness_map"]
        match = _THETA_NA_RE.search(str(c["theta_star"]))
        if match is None:
            raise CalibrationError(f"cannot read the theta* threshold from {c['theta_star']!r}")
        main = str(c.get("method", DEFAULT_MAIN_R_HAT))
        if main not in R_HAT_METHODS:
            raise CalibrationError(f"calibration.correctness_map.method must be one of {R_HAT_METHODS}, got {main!r}")
        sensitivity = c.get("sensitivity_method")
        sensitivity = str(sensitivity) if sensitivity is not None else next(m for m in R_HAT_METHODS if m != main)
        z_clip = Z_CLIP
        logistic = c.get("logistic")
        if logistic is not None:  # z_clip and the reason codes are read from the configuration and checked against the code
            z_clip = (float(logistic["z_clip"][0]), float(logistic["z_clip"][1]))
            codes = {str(code) for code in logistic["na"]}
            if codes != LOGISTIC_NA_CODES:
                raise CalibrationError(
                    f"logistic.na must be exactly the codes of the code ({sorted(LOGISTIC_NA_CODES)}); "
                    f"missing {sorted(LOGISTIC_NA_CODES - codes)}, unexpected {sorted(codes - LOGISTIC_NA_CODES)}"
                )
        z_bins, abs_residual, max_share = misfit_rule_from_config(cfg)
        return cls(
            fitted_w=tuple(float(w) for w in c["fitted_w"]),
            reference_w=0.2,  # the configuration fixes the reference curve at w = 0.2 (name of the key)
            r_w_zero_max_diff=float(c["r_w_zero_if_max_abs_diff_vs_w02_le"]),
            r_z_step=float(c["r_z"]["step"]),
            r_z_na_if_le=float(c["r_z"]["na_if_le"]),
            theta_star_na_if_le=float(match.group(1)),
            fixed_a_H=float(c["fixed"]["a_H"]),
            fixed_c0=float(c["fixed"]["c0"]),
            z_clip=z_clip,
            main_method=main,
            sensitivity_method=sensitivity,
            shape_z_bins=z_bins,
            shape_abs_residual=abs_residual,
            shape_max_share=max_share,
        )


@dataclass(frozen=True)
class RWZeroTest:
    max_abs_diff_by_w: dict[float, float | None]
    is_zero: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_diff_by_w": {str(w): v for w, v in self.max_abs_diff_by_w.items()},
            "is_zero": self.is_zero, "threshold": self.threshold,
        }


@dataclass(frozen=True)
class ResolvedCurve:
    """r_hat(., w) for one target w: the curve, its support and the pieces that give r_w."""

    w: float
    mode: str  # "nearest" (r_w == 0), "node" (w is a fitted node) or "interpolated"
    z_knots: np.ndarray
    r_knots: np.ndarray
    support: tuple[float, float]
    used_w: tuple[float, ...]  # fitted w values that make up the curve
    _r_w_terms: tuple[tuple[int, int], ...] = field(repr=False, default=())  # (lower, upper) curve indices
    _curves: tuple[IsotonicCurve, ...] = field(repr=False, default=())
    _fitted_w: tuple[float, ...] = field(repr=False, default=())

    def __call__(self, z: Any) -> np.ndarray:
        return np.interp(z, self.z_knots, self.r_knots)

    @property
    def value_range(self) -> tuple[float, float]:
        return float(self.r_knots[0]), float(self.r_knots[-1])

    def r_w(self, z: float) -> float:
        """Derivative of r_hat with respect to w at ``z`` (0 when r_w is taken as zero)."""
        if not self._r_w_terms:
            return 0.0
        lo, hi = self._r_w_terms[0]
        return float(
            (self._curves[hi](z) - self._curves[lo](z)) / (self._fitted_w[hi] - self._fitted_w[lo])
        )


@dataclass(frozen=True)
class ThresholdResult:
    """T*, r_z, r_w and theta* at one condition (a_H, c0, w), or an NA with its reason."""

    a_H: float
    c0: float
    w: float
    target: float  # a_H - c0 - w
    status: str  # "ok" or "NA"
    reason: str | None
    reason_code: str | None  # isotonic: "target_outside_range", "support_too_narrow", "plateau", "theta_star_too_small";
    # logistic: "fit_not_converged", "beta_not_positive", "target_outside_unit_interval",
    # "threshold_outside_observed_range", "plateau", "theta_star_too_small"
    mode: str | None
    T_star: float | None
    r_z: float | None
    r_w: float | None
    r_w_is_zero: bool
    theta_star: float | None
    used_w: tuple[float, ...] = ()
    method: str = "isotonic"  # "logistic" (main) or "isotonic" (sensitivity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method, "a_H": self.a_H, "c0": self.c0, "w": self.w, "target": self.target,
            "status": self.status, "reason": self.reason, "reason_code": self.reason_code, "mode": self.mode,
            "T_star": self.T_star, "r_z": self.r_z, "r_w": self.r_w, "r_w_is_zero": self.r_w_is_zero,
            "theta_star": self.theta_star, "used_w": list(self.used_w),
        }


class RHatFamily:
    """The four isotonic curves (one per fitted w) and everything derived from them (sensitivity version)."""

    def __init__(self, curves: Mapping[float, IsotonicCurve], spec: RHatSpec) -> None:
        missing = [w for w in spec.fitted_w if not any(abs(w - k) < 1e-9 for k in curves)]
        extra = [k for k in curves if not any(abs(w - k) < 1e-9 for w in spec.fitted_w)]
        if missing or extra:
            raise CalibrationError(f"curves must be given exactly for fitted_w {spec.fitted_w}; missing {missing}, extra {extra}")
        self.spec = spec
        self.fitted_w = spec.fitted_w
        lookup = {round(k, 9): v for k, v in curves.items()}
        self.curves: tuple[IsotonicCurve, ...] = tuple(lookup[round(w, 9)] for w in spec.fitted_w)
        self._ref = next(i for i, w in enumerate(spec.fitted_w) if abs(w - spec.reference_w) < 1e-9)
        self.r_w_test = self._test_r_w_zero()

    # ---- the r_w == 0 decision ------------------------------------------------------------
    def _test_r_w_zero(self) -> RWZeroTest:
        diffs: dict[float, float | None] = {}
        worst = 0.0
        undefined = False
        for i, w in enumerate(self.fitted_w):
            if i == self._ref:
                continue
            d = max_abs_difference(self.curves[i], self.curves[self._ref])
            diffs[w] = d
            if d is None:
                undefined = True  # no common support: r_w cannot be declared zero
            else:
                worst = max(worst, d)
        is_zero = (not undefined) and worst <= self.spec.r_w_zero_max_diff + _TOL  # _TOL absorbs rounding of a difference of exactly 0.03
        return RWZeroTest(diffs, is_zero, self.spec.r_w_zero_max_diff)

    @property
    def r_w_is_zero(self) -> bool:
        return self.r_w_test.is_zero

    # ---- curve at a target w ----------------------------------------------------------------
    def _nearest_index(self, w: float) -> int:
        distance = [abs(w - fw) for fw in self.fitted_w]
        best = min(distance)
        for i, d in enumerate(distance):  # fitted_w increases, so the first hit is the smaller w
            if d <= best + 1e-12:
                return i
        raise AssertionError("unreachable")

    def resolve(self, w: float) -> ResolvedCurve:
        """r_hat(., w) following the r_w rule: nearest fitted curve, or interpolation in w."""
        if not math.isfinite(w):
            raise CalibrationError("w must be finite")
        fw = self.fitted_w
        if self.r_w_is_zero:
            i = self._nearest_index(w)
            curve = self.curves[i]
            return ResolvedCurve(
                w=w, mode="nearest", z_knots=curve.z_knots, r_knots=curve.r_knots,
                support=curve.support, used_w=(fw[i],), _curves=self.curves, _fitted_w=fw,
            )
        wc = min(max(w, fw[0]), fw[-1])
        node = next((i for i, v in enumerate(fw) if abs(wc - v) < 1e-12), None)
        if node is not None:
            curve = self.curves[node]
            if node == 0:
                terms = ((0, 1),)
            elif node == len(fw) - 1:
                terms = ((len(fw) - 2, len(fw) - 1),)
            else:
                terms = ((node - 1, node + 1),)
            return ResolvedCurve(
                w=w, mode="node", z_knots=curve.z_knots, r_knots=curve.r_knots, support=curve.support,
                used_w=(fw[node],), _r_w_terms=terms, _curves=self.curves, _fitted_w=fw,
            )
        hi = int(np.searchsorted(fw, wc, side="right"))
        lo = hi - 1
        a, b = self.curves[lo], self.curves[hi]
        support = common_support(a, b)
        if support is None:
            raise CalibrationError(f"the curves at w={fw[lo]} and w={fw[hi]} have no common support")
        t = (wc - fw[lo]) / (fw[hi] - fw[lo])
        grid = np.concatenate([a.z_knots, b.z_knots, [support[0], support[1]]])
        grid = np.unique(grid[(grid >= support[0]) & (grid <= support[1])])
        values = (1.0 - t) * a(grid) + t * b(grid)
        return ResolvedCurve(
            w=w, mode="interpolated", z_knots=grid, r_knots=values, support=support,
            used_w=(fw[lo], fw[hi]), _r_w_terms=((lo, hi),), _curves=self.curves, _fitted_w=fw,
        )

    # ---- threshold, r_z, theta* -----------------------------------------------------------------
    def r_z_at(self, curve: ResolvedCurve, z: float) -> tuple[float | None, str | None]:
        """r_z by a central difference of step h; one-sided at a support edge."""
        h = self.spec.r_z_step
        lo_s, hi_s = curve.support
        has_lo = z - h >= lo_s - _TOL
        has_hi = z + h <= hi_s + _TOL
        if has_lo and has_hi:
            return float((curve(z + h) - curve(z - h)) / (2 * h)), None
        if has_lo:
            return float((curve(z) - curve(z - h)) / h), None
        if has_hi:
            return float((curve(z + h) - curve(z)) / h), None
        return None, f"support [{lo_s:.4g}, {hi_s:.4g}] is narrower than the difference step {h}"

    def normative(self, a_H: float, c0: float, w: float) -> ThresholdResult:
        """Canonical threshold and responsiveness at the stated condition (a_H, c0, w)."""
        target = float(a_H) - float(c0) - float(w)
        curve = self.resolve(w)
        zero = self.r_w_is_zero

        def na(code: str, reason: str, **kw: Any) -> ThresholdResult:
            return ThresholdResult(
                a_H=float(a_H), c0=float(c0), w=float(w), target=target, status="NA", reason=reason,
                reason_code=code, mode=curve.mode, T_star=kw.get("T_star"), r_z=kw.get("r_z"), r_w=kw.get("r_w"),
                r_w_is_zero=zero, theta_star=kw.get("theta_star"), used_w=curve.used_w,
            )

        t_star = inverse_lower(curve.z_knots, curve.r_knots, target)
        if t_star is None:
            lo, hi = curve.value_range
            return na("target_outside_range", f"target {target:.6g} outside the range [{lo:.6g}, {hi:.6g}] of r_hat")
        r_z, why = self.r_z_at(curve, t_star)
        if r_z is None:
            return na("support_too_narrow", why or "r_z undefined", T_star=t_star)
        if r_z <= self.spec.r_z_na_if_le:
            return na("plateau", f"r_z = {r_z:.6g} <= {self.spec.r_z_na_if_le} (plateau)", T_star=t_star, r_z=r_z)
        r_w = curve.r_w(t_star)
        theta = (1.0 + r_w) / r_z
        if theta <= self.spec.theta_star_na_if_le:
            return na(
                "theta_star_too_small", f"theta* = {theta:.6g} <= {self.spec.theta_star_na_if_le}",
                T_star=t_star, r_z=r_z, r_w=r_w, theta_star=theta,
            )
        return ThresholdResult(
            a_H=float(a_H), c0=float(c0), w=float(w), target=target, status="ok", reason=None, reason_code=None,
            mode=curve.mode, T_star=t_star, r_z=r_z, r_w=r_w, r_w_is_zero=zero, theta_star=theta,
            used_w=curve.used_w,
        )


def fit_r_hat_family(
    samples: Mapping[float, tuple[Any, Any]],
    spec: RHatSpec,
    *,
    weights: Mapping[float, Any] | None = None,
) -> RHatFamily:
    """One isotonic fit per fitted w from ``{w: (z, self_correct)}`` (calibration split).

    ``self_correct`` is the 0/1 correctness of the P-ask self-answer (the most likely of the
    digits 1-4) at the fixed condition (a_H, c0) and this w. ``weights`` carries bootstrap
    multiplicities with the same layout.
    """
    curves: dict[float, IsotonicCurve] = {}
    for w in spec.fitted_w:
        key = next((k for k in samples if abs(k - w) < 1e-9), None)
        if key is None:
            raise CalibrationError(f"no calibration samples for fitted w={w}")
        z, y = samples[key]
        wt = None
        if weights is not None:
            wkey = next((k for k in weights if abs(k - w) < 1e-9), None)
            wt = weights[wkey] if wkey is not None else None
        curves[w] = fit_isotonic(z, y, wt)
    return RHatFamily(curves, spec)


# ---------------------------------------------------------------------------------------
# logistic r_hat (main method)
# ---------------------------------------------------------------------------------------

def logit_of_confidence(z: Any, clip: tuple[float, float] = Z_CLIP) -> np.ndarray:
    """``logit(z~)`` with ``z~ = clip(z, *clip)`` (default [0.25, 1 - 1e-4])."""
    zc = np.clip(np.asarray(z, dtype=np.float64), clip[0], clip[1])
    return np.log(zc / (1.0 - zc))


@dataclass(frozen=True)
class LogisticRHat:
    """``logit r_hat(z, w) = alpha + beta * logit(z~) + gamma * w`` and the canonical benchmark derived from it."""

    alpha: float
    beta: float
    gamma: float
    z_min: float  # observed range of the calibration z (rows with positive weight)
    z_max: float
    n_rows: int
    converged: bool
    separation_suspected: bool
    loglik: float
    spec: RHatSpec

    def r(self, z: Any, w: Any) -> np.ndarray:
        """Fitted probability that the self-answer is correct."""
        return expit(
            self.alpha + self.beta * logit_of_confidence(z, self.spec.z_clip) + self.gamma * np.asarray(w, dtype=np.float64)
        )

    def normative(self, a_H: float, c0: float, w: float) -> ThresholdResult:
        """Canonical threshold and responsiveness at the stated condition (a_H, c0, w)."""
        target = float(a_H) - float(c0) - float(w)
        lo_w, hi_w = self.spec.fitted_w[0], self.spec.fitted_w[-1]
        mode = "logistic" if lo_w - 1e-12 <= w <= hi_w + 1e-12 else "logistic_extrapolated"  # w beyond the fitted range

        def result(status: str, code: str | None, reason: str | None, **kw: Any) -> ThresholdResult:
            return ThresholdResult(
                a_H=float(a_H), c0=float(c0), w=float(w), target=target, status=status, reason=reason,
                reason_code=code, mode=mode, T_star=kw.get("T_star"), r_z=kw.get("r_z"), r_w=kw.get("r_w"),
                r_w_is_zero=False, theta_star=kw.get("theta_star"), used_w=(), method="logistic",
            )

        if not self.converged:
            return result("NA", "fit_not_converged", "the logistic fit of r_hat did not converge")
        if not self.beta > 0:
            return result("NA", "beta_not_positive", f"beta = {self.beta:.6g} <= 0: r_hat does not increase with z")
        if not 0.0 < target < 1.0:
            return result("NA", "target_outside_unit_interval", f"target {target:.6g} is not in (0, 1)")
        logit_t = math.log(target / (1.0 - target))
        t_star = float(expit((logit_t - self.alpha - self.gamma * float(w)) / self.beta))
        if t_star < self.z_min - 1e-9 or t_star > self.z_max + 1e-9:
            return result(
                "NA", "threshold_outside_observed_range",
                f"T* = {t_star:.6g} outside the observed z range [{self.z_min:.6g}, {self.z_max:.6g}]", T_star=t_star,
            )
        slope = target * (1.0 - target)
        r_z = self.beta * slope / (t_star * (1.0 - t_star))
        r_w = self.gamma * slope
        if r_z <= self.spec.r_z_na_if_le:
            return result(
                "NA", "plateau", f"r_z = {r_z:.6g} <= {self.spec.r_z_na_if_le} (plateau)", T_star=t_star, r_z=r_z, r_w=r_w
            )
        theta = (1.0 + r_w) / r_z
        if theta <= self.spec.theta_star_na_if_le:
            return result(
                "NA", "theta_star_too_small", f"theta* = {theta:.6g} <= {self.spec.theta_star_na_if_le}",
                T_star=t_star, r_z=r_z, r_w=r_w, theta_star=theta,
            )
        return result("ok", None, None, T_star=t_star, r_z=r_z, r_w=r_w, theta_star=theta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "logistic", "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "z_observed_range": [self.z_min, self.z_max], "z_clip": list(self.spec.z_clip), "n_rows": self.n_rows,
            "converged": self.converged, "separation_suspected": self.separation_suspected, "loglik": self.loglik,
        }


def fit_logistic_r_hat(
    samples: Mapping[float, tuple[Any, Any]],
    spec: RHatSpec,
    *,
    weights: Mapping[float, Any] | None = None,
) -> LogisticRHat:
    """Maximum-likelihood fit of the logistic r_hat from ``{w: (z, self_correct)}`` (calibration split).

    The four (or more) fitted w are stacked into one regression on ``[1, logit(z~), w]``. ``self_correct`` is
    the 0/1 correctness of the P-ask self-answer; fractional values (expected correctness) are accepted
    for tests. ``weights`` carries bootstrap multiplicities with the same layout; rows without weight drop out
    and do not count for the observed z range.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    z_all: list[np.ndarray] = []
    for w in spec.fitted_w:
        key = next((k for k in samples if abs(k - w) < 1e-9), None)
        if key is None:
            raise CalibrationError(f"no calibration samples for fitted w={w}")
        z = np.asarray(samples[key][0], dtype=np.float64)
        y = np.asarray(samples[key][1], dtype=np.float64)
        if z.ndim != 1 or z.shape != y.shape or z.size == 0:
            raise CalibrationError("z and the correctness labels must be non-empty vectors of the same length")
        if np.any(~np.isfinite(z)) or np.any(~np.isfinite(y)) or y.min() < 0 or y.max() > 1:
            raise CalibrationError("z must be finite and the correctness labels must lie in [0, 1]")
        wt = np.ones_like(z)
        if weights is not None:
            wkey = next((k for k in weights if abs(k - w) < 1e-9), None)
            if wkey is not None:
                wt = np.asarray(weights[wkey], dtype=np.float64)
                if wt.shape != z.shape or np.any(wt < 0) or np.any(~np.isfinite(wt)):
                    raise CalibrationError("weights must be finite, non-negative and match z")
        keep = wt > 0
        if not keep.any():
            raise CalibrationError(f"no observation with positive weight at w={w}")
        xs.append(np.column_stack([
            np.ones(int(keep.sum())), logit_of_confidence(z[keep], spec.z_clip), np.full(int(keep.sum()), float(w)),
        ]))
        ys.append(y[keep])
        ws.append(wt[keep])
        z_all.append(z[keep])
    X, y_all, w_all = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    if np.linalg.matrix_rank(X) < 3:
        raise CalibrationError("the logistic r_hat needs z and w to vary (design is rank deficient)")
    res = fit_logistic(X, y_all, w_all)
    z_cat = np.concatenate(z_all)
    return LogisticRHat(
        alpha=float(res.beta[0]), beta=float(res.beta[1]), gamma=float(res.beta[2]),
        z_min=float(z_cat.min()), z_max=float(z_cat.max()), n_rows=int(len(y_all)), converged=res.converged,
        separation_suspected=res.separation_suspected, loglik=res.loglik, spec=spec,
    )



def logistic_shape_check(
    fit: LogisticRHat,
    samples: Mapping[float, tuple[Any, Any]],
    *,
    weights: Mapping[float, Any] | None = None,
) -> MisfitResult:
    """Shape check of the logistic r_hat: the misfit rule on z deciles x fitted w, on the rows that fitted it.

    ``samples`` is ``{w: (z, self_correct)}`` as given to ``fit_logistic_r_hat`` (the same items at every fitted w);
    ``weights`` are item multiplicities (a bootstrap resample, the same at every w). The thresholds (3, 10%) and the
    design-effect adjustment are those of ``misfit_diagnostic``. ``adjusted_severe`` marks the result shape-sensitive (descriptive only).
    """
    spec = fit.spec
    z_rows, w_rows, y_rows, item_rows, p_rows = [], [], [], [], []
    n_items = None
    item_weight = None
    for w in spec.fitted_w:
        key = next((k for k in samples if abs(k - w) < 1e-9), None)
        if key is None:
            raise CalibrationError(f"no calibration samples for fitted w={w}")
        z = np.asarray(samples[key][0], dtype=np.float64)
        y = np.asarray(samples[key][1], dtype=np.float64)
        if z.ndim != 1 or z.shape != y.shape or z.size == 0:
            raise CalibrationError("z and the correctness labels must be non-empty vectors of the same length")
        if n_items is None:
            n_items = z.size
        elif z.size != n_items:
            raise CalibrationError("the shape check needs the same items at every fitted w")
        if weights is not None:
            wkey = next((k for k in weights if abs(k - w) < 1e-9), None)
            if wkey is not None:
                wt = np.asarray(weights[wkey], dtype=np.float64)
                if item_weight is not None and not np.array_equal(wt, item_weight):
                    raise CalibrationError("the shape check needs the same item multiplicities at every fitted w")
                item_weight = wt
        z_rows.append(z)
        y_rows.append(y)
        w_rows.append(np.full(z.size, float(w)))
        item_rows.append(np.arange(z.size))
        p_rows.append(fit.r(z, w))
    return binned_misfit(
        np.concatenate(z_rows), np.concatenate(w_rows), np.concatenate(y_rows), np.concatenate(p_rows),
        np.concatenate(item_rows), int(n_items), z_bins=spec.shape_z_bins, abs_threshold=spec.shape_abs_residual,
        max_share=spec.shape_max_share, item_weight=item_weight, n_fitted_params=3,  # alpha, beta, gamma
    )


@dataclass(frozen=True)
class CalibrationSamples:
    """Calibration-split data of the r_hat fits: one row per item; the self-answer correctness
    (0/1) of the P-ask trial at each fitted w, aligned with ``z``."""

    z: np.ndarray
    self_correct: Mapping[float, np.ndarray]

    def __post_init__(self) -> None:
        for w, y in self.self_correct.items():
            if len(y) != len(self.z):
                raise CalibrationError(f"calibration samples at w={w}: length differs from z")

    @property
    def n_items(self) -> int:
        return len(self.z)

    def family(self, spec: RHatSpec, weights: np.ndarray | None = None) -> RHatFamily:
        """The isotonic r_hat (sensitivity version)."""
        samples = {w: (self.z, y) for w, y in self.self_correct.items()}
        wts = None if weights is None else {w: weights for w in self.self_correct}
        return fit_r_hat_family(samples, spec, weights=wts)

    def logistic(self, spec: RHatSpec, weights: np.ndarray | None = None) -> LogisticRHat:
        """The logistic r_hat (main version); ``weights`` are item multiplicities (bootstrap)."""
        samples = {w: (self.z, y) for w, y in self.self_correct.items()}
        wts = None if weights is None else {w: weights for w in self.self_correct}
        return fit_logistic_r_hat(samples, spec, weights=wts)

    def r_hats(self, spec: RHatSpec, weights: np.ndarray | None = None) -> dict[str, Any]:
        """Both versions, keyed by method name."""
        return {"logistic": self.logistic(spec, weights), "isotonic": self.family(spec, weights)}

    def shape_check(self, fit: LogisticRHat, weights: np.ndarray | None = None) -> MisfitResult:
        """The K1(iii) rule applied to ``fit`` on the rows that fitted it (``calibration.logistic_shape_check``)."""
        samples = {w: (self.z, y) for w, y in self.self_correct.items()}
        wts = None if weights is None else {w: weights for w in self.self_correct}
        return logistic_shape_check(fit, samples, weights=wts)
