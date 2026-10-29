"""Recompute the TriviaQA expert-model numbers of the paper (Fig. 6 and the Methods) from the files in this directory.

Reads the parquet and csv files next to this script and prints each recomputed value beside the value quoted in the paper.
Requires numpy, pandas, pyarrow and scipy (about two minutes).

Conventions: questions in ascending order of item_key; intervals are the 2.5 and 97.5 percentiles of 2,000 resamples of
questions drawn with numpy.random.default_rng(0).integers(0, 2000, (2000, 2000)); the external rule consults when
confidence < 0.64 - w (waiting costs) or < 0.64 + x (battery costs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

PKG = Path(__file__).resolve().parent
MODELS = ["qwen3-1.7b", "qwen3-4b", "qwen3-14b", "qwen2.5-7b-instruct", "olmo-3-7b-instruct", "smollm3-3b",
          "gemma-4-e4b-it", "phi-4-mini-instruct"]
PRES = ["explicit_cost", "server_state", "battery_cost"]
LEVELS = [0, 10, 20, 45, 80, 150]
CONDS = [(p, l) for p in PRES for l in LEVELS]
STATED = 0.64
N_BOOT, SEED = 2000, 0
OK = []


def report(name, got, paper, tol):
    # Rates over 2,000 questions are multiples of 0.05 points, so some values lie exactly on a rounding boundary
    # (for example -6.55 is displayed as -6.6). The bound is inclusive and padded against floating-point error.
    good = abs(got - paper) <= tol + 1e-9
    OK.append(good)
    print(f"{'OK  ' if good else 'DIFF'} {name}: recomputed {got:.4f} | paper {paper} (tol {tol})")


dec = pd.read_parquet(PKG / "consultation_choices_heldout.parquet")
ans = pd.read_parquet(PKG / "assistant_answers_heldout.parquet")
exp = pd.read_parquet(PKG / "expert_answers_heldout.parquet").set_index("item_key")
cal = pd.read_csv(PKG / "confidence_calibration.csv").set_index("model")

# ----------------------------------------------------------------------------- arrays per model (items sorted by key)
data = {}
w_eff = np.array([l / 100 if p != "battery_cost" else 0.0 for p, l in CONDS])
x_eff = np.array([l / 100 if p == "battery_cost" else 0.0 for p, l in CONDS])
for m in MODELS:
    a = ans[ans.model == m].sort_values("item_key").reset_index(drop=True)
    d = dec[dec.model == m].copy()
    d["cond"] = list(zip(d.presentation.astype(str), d.level))
    keys = a.item_key.to_numpy()
    n = len(keys)
    assert n == 2000 and len(d) == 36000
    pos = pd.Series(np.arange(n), index=keys)
    ci = {c: j for j, c in enumerate(CONDS)}
    ii = pos.loc[d.item_key].to_numpy()
    jj = d.cond.map(ci).to_numpy()
    out = np.full((n, 18), "", dtype=object)
    out[ii, jj] = d.outcome.astype(str).to_numpy()
    assert (out != "").all()
    data[m] = dict(keys=keys, e=exp.loc[keys, "expert_correct"].to_numpy(float), s=a.independent_correct.to_numpy(float),
                   z=a.confidence_calibrated.to_numpy(float), off=(out == "offload"), fail=(out == "format_failure"))


def payoffs(M):
    e, s, z, off, fail = M["e"][:, None], M["s"][:, None], M["z"], M["off"], M["fail"]
    own = np.where(off, e - w_eff[None, :], np.where(fail, 0.0, s - x_eff[None, :]))
    thr = np.round(STATED - w_eff + x_eff, 9)
    ask = z[:, None] < thr[None, :]
    rule = np.where(ask, e - w_eff[None, :], s - x_eff[None, :])
    return own, rule, ask


def counts_matrix(n, R=N_BOOT, seed=SEED):
    draws = np.random.default_rng(seed).integers(0, n, size=(R, n))
    flat = (draws + (np.arange(R, dtype=np.int64) * n)[:, None]).ravel()
    c = np.bincount(flat, minlength=R * n).reshape(R, n).astype(np.float64)
    return np.vstack([np.ones((1, n)), c])


# ----------------------------------------------------------------------------- Fig. 6a: routing gain, 18-condition mean
print("== Fig. 6a  routing gain (rule minus own routing, points per question, mean of 18 conditions)")
C = counts_matrix(2000)
gain = {}
for m in MODELS:
    own, rule, _ = payoffs(data[m])
    diff = (C @ (rule - own)) / C.sum(1, keepdims=True)           # (R+1, 18)
    avg = diff.mean(1)
    lo, hi = np.percentile(avg[1:], [2.5, 97.5])
    gain[m] = (avg[0], lo, hi)
    print(f"   {m:22s} {avg[0]:.4f} [{lo:.4f}, {hi:.4f}]")
report("gain, OLMo-3-7B-Instruct", gain["olmo-3-7b-instruct"][0], 0.273, 0.0005)
report("gain, Gemma-4-E4B-it", gain["gemma-4-e4b-it"][0], 0.271, 0.0005)
report("gain, Qwen3-14B (smallest)", gain["qwen3-14b"][0], 0.180, 0.0005)
report("gain range min", min(v[0] for v in gain.values()), 0.180, 0.0005)
report("gain range max", max(v[0] for v in gain.values()), 0.273, 0.0005)
print("   all lower 95% limits above zero:", all(v[1] > 0 for v in gain.values()))

# ----------------------------------------------------------------------------- Fig. 6b: 0.1 -> 1.5 contrasts
print("== Fig. 6b  change in offload rate, cost 0.1 -> 1.5 (percentage points)")
rows = []
for m in MODELS:
    off = data[m]["off"]
    _, _, ask = payoffs(data[m])
    for p in ("explicit_cost", "battery_cost"):
        j10, j150 = CONDS.index((p, 10)), CONDS.index((p, 150))
        rows.append((m, p, 100 * (off[:, j150].mean() - off[:, j10].mean()), 100 * (ask[:, j150].mean() - ask[:, j10].mean())))
b = pd.DataFrame(rows, columns=["model", "presentation", "observed_pp", "baseline_pp"]).set_index(["model", "presentation"])
print(b.round(2).to_string())
report("observed min over 16 contrasts", b.observed_pp.min(), -6.6, 0.05)
report("observed max over 16 contrasts", b.observed_pp.max(), 2.4, 0.05)
ex = b.xs("explicit_cost", level="presentation").baseline_pp
report("baseline explicit-cost decrease, smallest magnitude", ex.max(), -15.8, 0.05)
report("baseline explicit-cost decrease, largest magnitude", ex.min(), -100.0, 0.05)
bat = b.xs("battery_cost", level="presentation")
report("battery Qwen3-4B observed", bat.loc["qwen3-4b", "observed_pp"], 2.4, 0.05)
report("battery Gemma-4-E4B-it observed", bat.loc["gemma-4-e4b-it", "observed_pp"], 1.0, 0.05)
report("battery Qwen2.5-7B-Instruct observed", bat.loc["qwen2.5-7b-instruct", "observed_pp"], -1.7, 0.05)
report("battery Qwen3-14B baseline", bat.loc["qwen3-14b", "baseline_pp"], 71.4, 0.05)
report("battery Qwen3-14B observed", bat.loc["qwen3-14b", "observed_pp"], -0.2, 0.05)

# ----------------------------------------------------------------------------- Fig. 6d: offload rates
print("== Fig. 6d  offload rate (%), explicit cost / server state")
sub = {}
draws = np.random.default_rng(SEED).integers(0, 2000, size=(N_BOOT, 2000))
for m in MODELS:
    off = data[m]["off"]
    r = {}
    for p in ("explicit_cost", "server_state"):
        for l in LEVELS:
            v = off[:, CONDS.index((p, l))].astype(float)
            bt = v[draws].mean(1)
            r[(p, l)] = (100 * v.mean(), *(100 * np.percentile(bt, [2.5, 97.5])))
    sub[m] = r
    print(f"   {m:22s} w=1.5 explicit {r[('explicit_cost',150)][0]:.2f} [{r[('explicit_cost',150)][1]:.2f},{r[('explicit_cost',150)][2]:.2f}]"
          f"  server-state D=150 {r[('server_state',150)][0]:.2f}")
report("Gemma explicit cost w=1.5 (%)", sub["gemma-4-e4b-it"][("explicit_cost", 150)][0], 90.3, 0.05)
report("Phi-4-mini explicit cost w=1.5 (%)", sub["phi-4-mini-instruct"][("explicit_cost", 150)][0], 72.0, 0.05)
n_models = sum(sub[m][("explicit_cost", 150)][1] > 5.0 for m in MODELS)
report("models with lower 95% limit above 5% at w=1.5", n_models, 5, 0)
report("Phi server state D=0 (%)", sub["phi-4-mini-instruct"][("server_state", 0)][0], 68.6, 0.05)
report("Phi server state D=20 (%)", sub["phi-4-mini-instruct"][("server_state", 20)][0], 29.7, 0.05)
report("Phi server state D=150 (%)", sub["phi-4-mini-instruct"][("server_state", 150)][0], 70.8, 0.05)


# ----------------------------------------------------------------------------- Fig. 6c: confidence coefficient
def batched_slope(y, x, W, iters=40):
    """Weighted logistic regression logit P(y) = alpha_c + beta x, one intercept per condition column of y, for a batch of
    resampling-weight vectors W (R, n). Profile Newton on beta; conditions without outcome variation are left out."""
    R = W.shape[0]
    Wc = W[:, :, None]
    n1 = (Wc * y[None]).sum(1)
    n0 = (Wc * (1 - y)[None]).sum(1)
    active = (n1 > 0) & (n0 > 0)
    alpha = np.where(active, logit(np.clip(n1 / np.maximum(n1 + n0, 1), 1e-9, 1 - 1e-9)), 0.0)
    beta = np.zeros(R)
    xc = x[None, :, None]
    for _ in range(iters):
        p = expit(alpha[:, None, :] + beta[:, None, None] * xc)
        res = Wc * (y[None] - p) * active[:, None, :]
        wgt = Wc * p * (1 - p) * active[:, None, :]
        ga, ac = res.sum(1), np.maximum(wgt.sum(1), 1e-300)
        gb = (res * xc).sum((1, 2))
        bc = (wgt * xc).sum(1)
        sbb = (wgt * xc ** 2).sum((1, 2))
        den = sbb - (np.where(active, bc ** 2 / ac, 0)).sum(1)
        num = gb - (np.where(active, bc * ga / ac, 0)).sum(1)
        d_beta = num / den
        d_alpha = np.where(active, (ga - bc * d_beta[:, None]) / ac, 0.0)
        step = np.minimum(1.0, 4.0 / np.maximum(np.maximum(np.abs(d_beta), np.abs(d_alpha).max(1)), 1e-12))
        alpha = alpha + step[:, None] * d_alpha
        beta = beta + step * d_beta
        if np.abs(step * np.maximum(np.abs(d_beta), np.abs(d_alpha).max(1))).max() < 1e-9:
            break
    return beta


print("== Fig. 6c  coefficient of logit(calibrated confidence), 18 conditions, intercept per condition")
slopes = {}
for m in MODELS:
    z = data[m]["z"]
    if np.unique(z).size == 1:
        print(f"   {m:22s} constant confidence {z[0]:.3f}: slope not identified")
        continue
    x = logit(np.clip(z, 1e-4, 1 - 1e-4))
    y = data[m]["off"].astype(float)
    full = batched_slope(y, x, np.ones((1, 2000)))[0]
    boots = np.concatenate([batched_slope(y, x, C[1 + k: 1 + k + 100]) for k in range(0, N_BOOT, 100)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    slopes[m] = (full, lo, hi)
    print(f"   {m:22s} {full:.4f} [{lo:.4f}, {hi:.4f}]")
report("slope min (Qwen3-14B)", min(v[0] for v in slopes.values()), -0.558, 0.0005)
report("slope max (SmolLM3-3B)", max(v[0] for v in slopes.values()), -0.171, 0.0005)
report("models with upper 95% limit below zero", sum(v[2] < 0 for v in slopes.values()), 7, 0)
const_z = [m for m in MODELS if np.unique(data[m]["z"]).size == 1]
report("constant confidence value of the model without slope", float(data[const_z[0]]["z"][0]), 0.516, 0.0005)

# ----------------------------------------------------------------------------- calibration map
print("== calibration map: refit on the development pool and apply to held-out confidence")
dev = pd.read_parquet(PKG / "assistant_answers_development.parquet")
worst_a = worst_z = 0.0
for m in MODELS:
    d = dev[dev.model == m]
    lp = np.stack(d.digit_logprobs.to_numpy()).astype(np.float64)
    c = d.confidence_raw.to_numpy()
    xl = logit(np.clip(c, 1e-4, 1 - 1e-4))
    y = d.independent_correct.to_numpy(float)

    def nll(t):
        eta = t[0] + t[1] * xl
        return np.sum(np.logaddexp(0, eta) - y * eta)

    r = minimize(nll, x0=[0.0, 1.0], method="L-BFGS-B", bounds=[(-100, 100), (0, 100)], options={"ftol": 1e-15, "gtol": 1e-10})
    a_fit, b_fit = r.x
    a_ref, b_ref = cal.loc[m, "intercept_a"], cal.loc[m, "slope_b"]
    worst_a = max(worst_a, abs(a_fit - a_ref), abs(b_fit - b_ref))
    h = ans[ans.model == m]
    z_new = expit(a_ref + b_ref * logit(np.clip(h.confidence_raw.to_numpy(), 1e-4, 1 - 1e-4)))
    worst_z = max(worst_z, np.abs(z_new - h.confidence_calibrated.to_numpy()).max())
print(f"   max |refit - stored| over (a, b): {worst_a:.2e};  max |z recomputed - z stored| held-out: {worst_z:.2e}")
OK.append(worst_z < 1e-6)
print(f"\nSUMMARY: {sum(OK)} of {len(OK)} checks reproduced")
