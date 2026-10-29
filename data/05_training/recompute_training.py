#!/usr/bin/env python3
"""Recompute the training-experiment statistics (Fig. 5) from the released tables.

Usage: python recompute_training.py [directory with the released tables, default: this directory]
Requires numpy, pandas and pyarrow. Runs in a few seconds on one CPU core.

Definitions follow the Methods: regret r = max_a V^(a) - V^(Y); E1 = penalized threshold error and E2 = threshold-rule
disagreement (sweep set); E3 = decision inconsistency (equal_net set); 95% percentile intervals from 2,000 resamples of
questions (default_rng(0)) after averaging seeds within questions. The script prints each manuscript value next to the
value recomputed from the tables. Rows marked release diagnostic are not quoted in the manuscript.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
MODELS = ["qwen3-4b", "gemma-4-e4b-it", "smollm3-3b"]
RESAMPLES, SEED = 2000, 0
NEVER_FLIP_ERROR = 0.20
runs = pd.read_csv(D / "training_runs.csv")
_draws = {}


def draws(n):
    if n not in _draws:
        _draws[n] = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))
    return _draws[n]


def ci(v):
    boot = np.asarray(v, dtype=np.float64)[draws(len(v))].mean(axis=1)
    return tuple(np.percentile(boot, [2.5, 97.5]))


# ---------------------------------------------------------------- statistics (definitions of the Methods)
def item_index(df):
    keys, inv = np.unique(df["item_key"].to_numpy(), return_inverse=True)
    return keys, inv


def regret_per_item(sub, ask):
    """r = max(V_help, V_self) - (V_help if asked else V_self), averaged over the trials of an item."""
    keys, inv = item_index(sub)
    ph, ps = sub["payoff_help"].to_numpy(), sub["payoff_self"].to_numpy()
    r = np.maximum(ph, ps) - np.where(ask, ph, ps)
    return np.bincount(inv, weights=r, minlength=len(keys)) / np.bincount(inv, minlength=len(keys))


def fit_step(asked, net, z0, valid):
    m = len(asked)
    cands = np.concatenate([[net[0] - 0.025], (net[:-1] + net[1:]) / 2.0, [net[-1] + 0.025]])
    pos = np.arange(m)
    mism = np.array([int(((pos >= j) != asked)[valid].sum()) for j in range(m + 1)])
    best = mism.min()
    tied = np.flatnonzero(mism == best)
    j = int(min(tied, key=lambda c: (abs(cands[c] - z0), cands[c])))
    t = float(cands[j])
    at_end = j == 0 or j == m
    err = NEVER_FLIP_ERROR if at_end else abs(t - z0)
    return err, best / max(int(valid.sum()), 1)


def sweep_per_item(sub, ask):
    """E1 (penalized threshold error) and E2 (disagreement with the fitted step rule) per item."""
    keys, inv = item_index(sub)
    net, z0, tie = sub["net"].to_numpy(), sub["z0"].to_numpy(), sub["tie_point"].to_numpy()
    e1, e2 = np.zeros(len(keys)), np.zeros(len(keys))
    order_all = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order_all], np.arange(len(keys) + 1))
    for i in range(len(keys)):
        sel = order_all[bounds[i] : bounds[i + 1]]
        order = sel[np.argsort(net[sel], kind="stable")]
        e1[i], e2[i] = fit_step(ask[order], net[order], float(z0[order][0]), ~tie[order])
    return e1, e2


def equal_net_per_item(sub, ask):
    """E3: share of the matched-score groups of an item whose three expressions do not give the same decision."""
    keys, inv = item_index(sub)
    g = sub["group"].astype(str).to_numpy()
    gi = np.unique(g, return_inverse=True)[1]
    cell = inv * (gi.max() + 1) + gi
    ucell, cinv = np.unique(cell, return_inverse=True)
    s = np.bincount(cinv, weights=ask.astype(float))
    c = np.bincount(cinv)
    flag = ~((s == 0) | (s == c))
    item_of_cell = ucell // (gi.max() + 1)
    return np.bincount(item_of_cell, weights=flag.astype(float), minlength=len(keys)) / np.bincount(
        item_of_cell, minlength=len(keys)
    )


def ece(z, correct, bins=10):
    b = np.minimum((z * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += abs(correct[m].mean() - z[m].mean()) * m.sum()
    return tot / len(z)


# ---------------------------------------------------------------- compute everything
res = {}  # (model, objective_tag, stat) -> dict
objs = {}
for m in MODELS:
    dec = pd.read_parquet(D / f"trials_{m}.parquet")
    ao = pd.read_parquet(D / f"answer_only_{m}.parquet")
    r = runs[runs.model == m]
    base_cols = {"base": "base", "base_sft": "base_sft"}
    cache = {}

    def per_item(set_name, tag, kind):
        key = (set_name, tag, kind)
        if key in cache:
            return cache[key]
        sub = dec[dec["set"] == set_name]
        ask = sub[f"ask__{tag}"].to_numpy()
        if kind == "regret":
            v = regret_per_item(sub, ask)
        elif kind == "sweep":
            v = sweep_per_item(sub, ask)
        elif kind == "equal":
            v = equal_net_per_item(sub, ask)
        cache[key] = v
        return v

    for objective, grp in r.groupby("objective", sort=False):
        tags = list(grp.column_tag)
        pb = grp.paired_base.iloc[0]
        objs[(m, objective)] = (tags, pb)
        for sname in ["in_dist", "unseen_combo", "deduct_position", "waiting", "price_extrap"]:
            b = per_item(sname, pb, "regret")
            t = np.mean([per_item(sname, tg, "regret") for tg in tags], axis=0)
            res[(m, objective, sname)] = dict(base=b.mean(), trained=t.mean(), diff=(t - b).mean(), rel=1 - t.mean() / b.mean(), b_i=b, t_i=t)
        # sweep
        bs = per_item("sweep", pb, "sweep")
        ts = [per_item("sweep", tg, "sweep") for tg in tags]
        res[(m, objective, "E1")] = dict(base=bs[0].mean(), trained=np.mean([x[0] for x in ts], axis=0).mean(), t_i=np.mean([x[0] for x in ts], axis=0))
        res[(m, objective, "E2")] = dict(base=bs[1].mean(), trained=np.mean([x[1] for x in ts], axis=0).mean())
        be = per_item("equal_net", pb, "equal")
        te = np.mean([per_item("equal_net", tg, "equal") for tg in tags], axis=0)
        res[(m, objective, "E3")] = dict(base=be.mean(), trained=te.mean())
        # answer-only accuracy and calibration
        acc_b = ao[f"correct__{pb}"].to_numpy().astype(float)
        acc_t = np.mean([ao[f"correct__{tg}"].to_numpy().astype(float) for tg in tags], axis=0)
        ece_b = ece(ao[f"z__{pb}"].to_numpy(), acc_b)
        ece_t = np.mean([ece(ao[f"z__{tg}"].to_numpy(), ao[f"correct__{tg}"].to_numpy().astype(float)) for tg in tags])
        res[(m, objective, "acc")] = dict(base=acc_b.mean(), trained=acc_t.mean())
        res[(m, objective, "ece")] = dict(base=ece_b, trained=ece_t)

main_objs = ["CAPT", "SFT", "soft-target"]
rows = []
checks = []


def rng(vals, fmt):
    return f"{fmt % min(vals)}-{fmt % max(vals)}"


def report(claim, paper, got, ok):
    checks.append((claim, paper, got, ok))


def displays_as(x, shown, unit):
    """True when x is displayed as `shown` at the given unit of the last decimal; halves round away from zero."""
    return abs(x - shown) <= unit / 2 + 1e-9


# 1. regret reduction 59%-87% across nine combinations
rel = [res[(m, o, "in_dist")]["rel"] for m in MODELS for o in main_objs if (m, o) in objs]
assert len(rel) == 9, len(rel)
lo, hi = min(rel), max(rel)
report("regret reduction, nine model-objective combinations", "59%-87%", f"{lo*100:.1f}%-{hi*100:.1f}%", round(lo * 100) == 59 and round(hi * 100) == 87)
# 2. base 0.100-0.129 -> trained 0.016-0.041
bs_ = [res[(m, o, "in_dist")]["base"] for m in MODELS for o in main_objs if (m, o) in objs]
tr_ = [res[(m, o, "in_dist")]["trained"] for m in MODELS for o in main_objs if (m, o) in objs]
report("base regret (points/question)", "0.100-0.129", f"{min(bs_):.4f}-{max(bs_):.4f}", round(min(bs_), 3) == 0.100 and round(max(bs_), 3) == 0.129)
report("trained regret (points/question)", "0.016-0.041", f"{min(tr_):.4f}-{max(tr_):.4f}", round(min(tr_), 3) == 0.016 and round(max(tr_), 3) == 0.041)
# 3. controls and CAPT in Qwen3-4B
fc = res[("qwen3-4b", "fixed-cost control", "in_dist")]["rel"]
rt = res[("qwen3-4b", "rotated-terms control", "in_dist")]["rel"]
cp = res[("qwen3-4b", "CAPT", "in_dist")]["rel"]
report("Qwen3-4B regret reduction: fixed-cost / rotated-terms / CAPT", "51% / 56% / 80%", f"{fc*100:.1f}% / {rt*100:.1f}% / {cp*100:.1f}%", round(fc * 100) == 51 and round(rt * 100) == 56 and round(cp * 100) == 80)
# 4. E3
e3b = [res[(m, o, "E3")]["base"] for m in MODELS for o in main_objs[:1] if (m, o) in objs]
e3t = [res[(m, o, "E3")]["trained"] for m in MODELS for o in main_objs if (m, o) in objs]
report("E3 base models", "0.05-0.10", f"{min(e3b):.3f}-{max(e3b):.3f}", round(min(e3b), 2) == 0.05 and round(max(e3b), 2) == 0.10)
report("E3 trained, nine combinations", "0.11-0.31", f"{min(e3t):.3f}-{max(e3t):.3f}", round(min(e3t), 2) == 0.11 and round(max(e3t), 2) == 0.31)
e3s = [res[(m, "soft-target", "E3")]["trained"] for m in MODELS]
lowest_e3 = all(res[(m, "soft-target", "E3")]["trained"] <= min(res[(m, o, "E3")]["trained"] for o in main_objs) for m in MODELS)
report("E3 soft-target, three models (lowest of the main objectives in each model)", "0.107-0.129 (all > 0.10)", f"{min(e3s):.3f}-{max(e3s):.3f}", round(min(e3s), 3) == 0.107 and round(max(e3s), 3) == 0.129 and min(e3s) > 0.10 and lowest_e3)
ctl3 = [res[("qwen3-4b", o, "E3")]["trained"] for o in ["fixed-cost control", "rotated-terms control"]]
report("E3 controls (Qwen3-4B)", "< 0.01", " / ".join(f"{x:.4f}" for x in ctl3), max(ctl3) < 0.01)
# 5. E1
e1b = [res[(m, "CAPT", "E1")]["base"] for m in MODELS]
e1t = [res[(m, o, "E1")]["trained"] for m in MODELS for o in main_objs if (m, o) in objs]
report("E1 base models", "0.19-0.20", f"{min(e1b):.3f}-{max(e1b):.3f}", round(min(e1b), 2) == 0.19 and round(max(e1b), 2) == 0.20)
report("E1 trained, nine combinations", "0.11-0.15 (> 0.05)", f"{min(e1t):.3f}-{max(e1t):.3f}", round(min(e1t), 2) == 0.11 and round(max(e1t), 2) == 0.15 and min(e1t) > 0.05)
c1 = [res[("qwen3-4b", o, "E1")]["trained"] for o in ["fixed-cost control", "rotated-terms control"]]
report("E1 controls (fixed-cost / rotated-terms)", "0.200 / 0.199", " / ".join(f"{x:.3f}" for x in c1), [round(x, 3) for x in c1] == [0.200, 0.199])
# 6. transfer
d_unseen = [res[(m, o, "unseen_combo")]["diff"] for m in MODELS for o in main_objs if (m, o) in objs]
report("regret change, unseen term combinations (nine)", "-0.027 to -0.155", f"{max(d_unseen):+.3f} to {min(d_unseen):+.3f}", round(max(d_unseen), 3) == -0.027 and round(min(d_unseen), 3) == -0.155)
d_wait = [res[(m, o, "waiting")]["diff"] for m in MODELS for o in main_objs if (m, o) in objs]
report("regret change, waiting-cost wording (nine)", "-0.059 to -0.103", f"{max(d_wait):+.3f} to {min(d_wait):+.3f}", round(max(d_wait), 3) == -0.059 and round(min(d_wait), 3) == -0.103)
d_smol = [res[("smollm3-3b", o, "deduct_position")]["diff"] for o in main_objs]
report("regret change, moved deduction, SmolLM3-3B (three)", "-0.177 to -0.205", f"{max(d_smol):+.3f} to {min(d_smol):+.3f}", round(max(d_smol), 3) == -0.177 and round(min(d_smol), 3) == -0.205)
d_ctl = [res[("qwen3-4b", o, "deduct_position")]["diff"] for o in ["fixed-cost control", "rotated-terms control"]]
report("regret change, moved deduction, Qwen3-4B fixed-cost / rotated-terms", "+0.152 / +0.096", " / ".join(f"{x:+.3f}" for x in d_ctl), [round(x, 3) for x in d_ctl] == [0.152, 0.096])
d_pq = [res[(m, o, "price_extrap")]["diff"] for m in ("qwen3-4b", "gemma-4-e4b-it") for o in main_objs]
report("regret change, waiting costs 0.4-0.7, main objectives, Qwen3-4B and Gemma-4-E4B-it (six)", "-0.067 to -0.198", f"{max(d_pq):+.3f} to {min(d_pq):+.3f}", len(d_pq) == 6 and round(max(d_pq), 3) == -0.067 and round(min(d_pq), 3) == -0.198)
d_pc = [res[("qwen3-4b", o, "price_extrap")]["diff"] for o in ["fixed-cost control", "rotated-terms control"]]
report("regret change, waiting costs 0.4-0.7, Qwen3-4B fixed-cost / rotated-terms controls", "-0.217 / -0.187", " / ".join(f"{x:+.3f}" for x in d_pc), [round(x, 3) for x in d_pc] == [-0.217, -0.187])
d_ps = [res[("smollm3-3b", o, "price_extrap")]["diff"] for o in main_objs]
report("regret change, waiting costs 0.4-0.7, SmolLM3-3B (three)", "+0.031 to +0.095", f"{min(d_ps):+.3f} to {max(d_ps):+.3f}", round(min(d_ps), 3) == 0.031 and round(max(d_ps), 3) == 0.095)
# 6b. disagreement with the fitted threshold rule (E2), base models and the nine trained combinations
e2b = [res[(m, "CAPT", "E2")]["base"] for m in MODELS]
e2t = [res[(m, o, "E2")]["trained"] for m in MODELS for o in main_objs if (m, o) in objs]
report("E2 base models", "0.004-0.019", f"{min(e2b):.4f}-{max(e2b):.4f}", displays_as(min(e2b), 0.004, 0.001) and displays_as(max(e2b), 0.019, 0.001))
report("E2 trained, nine combinations", "at most 0.003", f"maximum {max(e2t):.4f}", len(e2t) == 9 and displays_as(max(e2t), 0.003, 0.001) and max(e2t) <= 0.003 + 1e-9)
# 6c. answer-only accuracy and expected calibration error on the evaluation questions, main objectives
d_acc = [100 * (res[(m, o, "acc")]["trained"] - res[(m, o, "acc")]["base"]) for m in MODELS for o in main_objs if (m, o) in objs]
d_ece = [res[(m, o, "ece")]["trained"] - res[(m, o, "ece")]["base"] for m in MODELS for o in main_objs if (m, o) in objs]
report("change in answer-only accuracy, nine combinations (pp)", "-0.9 to +1.0", f"{min(d_acc):+.2f} to {max(d_acc):+.2f}", len(d_acc) == 9 and displays_as(min(d_acc), -0.9, 0.1) and displays_as(max(d_acc), 1.0, 0.1))
report("change in expected calibration error, nine combinations", "-0.020 to +0.034", f"{min(d_ece):+.4f} to {max(d_ece):+.4f}", len(d_ece) == 9 and displays_as(min(d_ece), -0.020, 0.001) and displays_as(max(d_ece), 0.034, 0.001))

# 7. bootstrap intervals of the figure panel (Qwen3-4B, in-distribution regret)
b = res[("qwen3-4b", "CAPT", "in_dist")]
lo_b, hi_b = ci(b["b_i"])
lo_t, hi_t = ci(b["t_i"])
checks.append(("Qwen3-4B in-distribution regret, base [95% CI] (release diagnostic)", "0.1291 [0.1237, 0.1342]", f"{b['base']:.4f} [{lo_b:.4f}, {hi_b:.4f}]", abs(b['base'] - 0.1290552) < 1e-6 and abs(lo_b - 0.1236819) < 1e-6 and abs(hi_b - 0.1342253) < 1e-6))
checks.append(("Qwen3-4B in-distribution regret, CAPT [95% CI] (release diagnostic)", "0.0264 [0.0245, 0.0283]", f"{b['trained']:.4f} [{lo_t:.4f}, {hi_t:.4f}]", abs(b['trained'] - 0.0263949) < 1e-6 and abs(lo_t - 0.0245340) < 1e-6 and abs(hi_t - 0.0283195) < 1e-6))

print("%-100s %-28s %-30s %s" % ("quantity", "reference", "recomputed from released tables", "match"))
for c, p, g, ok in checks:
    print("%-100s %-28s %-30s %s" % (c, p, g, "yes" if ok else "NO"))
print()
print("matches: %d / %d" % (sum(ok for *_, ok in checks), len(checks)))
print()
print("Per model-objective values (base -> trained):")
print("%-16s %-24s %-9s %-9s %-7s %-7s %-7s %-9s %-9s" % ("model", "objective", "regret_b", "regret_t", "drop", "E1", "E3", "d_acc", "d_ece"))
for (m, o), (tags, pb) in objs.items():
    g = lambda s: res[(m, o, s)]
    print(
        "%-16s %-24s %-9.4f %-9.4f %-7.3f %-7.3f %-7.3f %+9.4f %+9.4f"
        % (m, o, g("in_dist")["base"], g("in_dist")["trained"], g("in_dist")["rel"], g("E1")["trained"], g("E3")["trained"], g("acc")["trained"] - g("acc")["base"], g("ece")["trained"] - g("ece")["base"])
    )
