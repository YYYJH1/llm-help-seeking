#!/usr/bin/env python
"""Recompute the numerical-task results (Fig. 4a and the quoted statements) from the files in this directory.

Usage:  python recompute_numeric_tasks.py [--write-summary]

Reads trials.parquet, conditions.csv and repeat_readings.parquet from the directory of this script, recomputes the accuracy of every
model and task with its stratified bootstrap interval, and prints each number quoted in the paper next to the recomputed value.
With --write-summary the table is also written to accuracy_summary.csv; otherwise the recomputed table is compared with that file.
Requires numpy, pandas and pyarrow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = ["qwen3-1.7b", "qwen3-4b", "qwen3-14b", "qwen3-32b", "qwen2.5-7b-instruct", "gemma-4-e4b-it", "smollm3-3b"]
LABEL = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-14b": "Qwen3-14B", "qwen3-32b": "Qwen3-32B",
         "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct", "gemma-4-e4b-it": "Gemma-4-E4B-it", "smollm3-3b": "SmolLM3-3B"}
# quantity -> (task, field, threshold)
QUANTITIES = {
    "identification_own_accuracy": ("identification", "own_accuracy", 0.95),
    "identification_expert_accuracy": ("identification", "expert_accuracy", 0.95),
    "identification_cost": ("identification", "cost", 0.95),
    "comparison": ("comparison", None, 0.95),
    "action_selection": ("action_selection", None, 0.90),
    "action_selection_with_scores": ("action_selection_with_scores", None, 0.90),
}
WEIGHTS = {1: 0.5, 2: 0.25, 3: 0.25}   # strata: consulting scores higher / answering higher with cost <= 1 / cost > 1
RESAMPLES, SEED = 2000, 0

results: list[tuple[bool, str]] = []


def check(ok: bool, text: str) -> None:
    results.append((bool(ok), text))
    print(("OK   " if ok else "FAIL ") + text)


def pct(x: float) -> float:
    return round(100 * x + 1e-9, 1)


def bootstrap_design(order_stratum: np.ndarray):
    """One block of resampled condition counts per stratum, in the order of ascending stratum and condition number."""
    rng = np.random.default_rng(SEED)
    blocks = {}
    for s in (1, 2, 3):
        n_s = int((order_stratum == s).sum())
        draws = rng.integers(0, n_s, size=(RESAMPLES, n_s))
        flat = (draws + (np.arange(RESAMPLES) * n_s)[:, None]).ravel()
        blocks[s] = np.bincount(flat, minlength=RESAMPLES * n_s).reshape(RESAMPLES, n_s).astype(float)
    return blocks


def stratified(values: np.ndarray, order_stratum: np.ndarray, blocks) -> tuple[float, float, float]:
    """Weighted mean over strata of the condition means, with the 2.5% and 97.5% quantiles over the resamples."""
    point, boot = 0.0, np.zeros(RESAMPLES)
    for s, w in WEIGHTS.items():
        v = values[order_stratum == s]
        point += w * v.mean()
        boot += w * (blocks[s] @ v) / blocks[s].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def main() -> int:
    write = "--write-summary" in sys.argv[1:]
    trials = pd.read_parquet(HERE / "trials.parquet")
    cond = pd.read_csv(HERE / "conditions.csv")
    repeats = pd.read_parquet(HERE / "repeat_readings.parquet")

    print("== Design and integrity")
    check(trials.groupby("model", observed=True).size().reindex(MODELS).eq(16800).all(),
          "16,800 trials for each of the 7 models (400 combinations x 42 trials)")
    per_cond = trials.groupby(["model", "condition"], observed=True).size()
    check((per_cond == 42).all() and trials["condition"].nunique() == 400, "42 trials in each of the 400 combinations, identical for every model")
    same = all(trials[trials.model == m].groupby("condition")[["own_accuracy", "expert_accuracy", "cost"]].first().round(2)
               .equals(cond.set_index("condition")[["own_accuracy", "expert_accuracy", "cost"]].round(2)) for m in MODELS)
    check(same, "stated values of every combination agree with conditions.csv in all 7 models")
    strata = cond["stratum"].value_counts().sort_index().to_dict()
    check(strata == {1: 200, 2: 100, 3: 100}, f"strata of the 400 combinations {strata} (paper: 200 consulting higher, 100 answering higher with w<=1, 100 with w>1)")
    check(cond["score_difference"].abs().min() >= 0.05 - 1e-9, f"smallest difference of the two expected scores {cond['score_difference'].abs().min():.2f} (paper: near-ties below 0.05 removed)")
    consult_higher = cond["better_action"].eq("consult")
    always_answer = sum(WEIGHTS[s] * (~consult_higher[cond.stratum == s]).mean() for s in (1, 2, 3))
    always_consult = sum(WEIGHTS[s] * consult_higher[cond.stratum == s].mean() for s in (1, 2, 3))
    check(abs(always_answer - 0.5) < 1e-12 and abs(always_consult - 0.5) < 1e-12,
          f"weights 1/2, 1/4, 1/4: always-answer accuracy {always_answer:.3f}, always-consult accuracy {always_consult:.3f} (paper: 50% each)")
    act = trials[trials.task.isin(["action_selection", "action_selection_with_scores"])]
    check(abs(act.help_label.eq(1).mean() - 0.5) < 1e-12, "in action selection, label 1 denotes consulting on half of the trials")
    tie_ok = (trials["tie"] == (trials["margin"].abs() < 1e-6)).all()
    sign = np.where(trials["margin"].abs() < 1e-6, 0, np.where(trials["margin"] > 0, 1, 2))
    check(tie_ok and (trials["choice"].fillna(0).astype(int).to_numpy() == sign).all(),
          "stored tie and choice agree with the margin (tie: |margin| < 1e-6; choice 1 if margin > 0, else 2)")
    check(bool(trials["gate_pass"].all()), f"digits 1 and 2 carry at least half of the probability in all {len(trials):,} trials (smallest mass {trials['mass'].min():.3f})")
    # label rules: identification -> the first option is the stated value when label_order is 0; comparison and action selection -> the label of the larger
    # expected score (label 1 shows the expected score of answering when label_order is 0, and label 2 is the label of consulting in the action tasks)
    consult_higher_trial = trials["condition"].map(cond.set_index("condition")["score_difference"]).to_numpy() > 0
    first = trials["label_order"].to_numpy() == 0
    expected_correct = np.where(trials["task"].eq("identification").to_numpy(), np.where(first, 1, 2), np.where(consult_higher_trial == first, 2, 1))
    is_action = trials["task"].isin(["action_selection", "action_selection_with_scores"]).to_numpy()
    expected_help = np.where(is_action, np.where(first, 2, 1), 0)
    check((trials["correct_label"].to_numpy() == expected_correct).all() and (trials["help_label"].fillna(0).astype(int).to_numpy() == expected_help).all(),
          "stored correct_label and help_label agree with the label rules in all trials")

    # ---- accuracy per model and quantity
    order = cond.sort_values(["stratum", "condition"]).reset_index(drop=True)
    pos = pd.Series(np.arange(len(order)), index=order["condition"])
    blocks = bootstrap_design(order["stratum"].to_numpy())
    rows = []
    for m in MODELS:
        tm = trials[trials.model == m]
        correct = (~tm["tie"].to_numpy()) & (tm["choice"].to_numpy() == tm["correct_label"].to_numpy())
        for q, (task, field, thr) in QUANTITIES.items():
            mask = (tm["task"] == task).to_numpy()
            if field is not None:
                mask = mask & (tm["field"] == field).to_numpy()
            idx = pos.loc[tm["condition"].to_numpy()[mask]].to_numpy()
            num = np.bincount(idx, weights=correct[mask].astype(float), minlength=len(order))
            den = np.bincount(idx, minlength=len(order))
            acc, lo, hi = stratified(num / den, order["stratum"].to_numpy(), blocks)
            rows.append({"model": m, "quantity": q, "n_trials": int(mask.sum()), "accuracy": acc, "ci_low": lo, "ci_high": hi,
                         "threshold": thr, "lower_bound_reaches_threshold": bool(lo >= thr - 1e-12)})
    summary = pd.DataFrame(rows)
    path = HERE / "accuracy_summary.csv"
    if write:
        summary.to_csv(path, index=False, lineterminator="\n")
        print(f"wrote {path.name} ({len(summary)} rows)")
    elif path.exists():
        stored = pd.read_csv(path)
        diff = max(float((stored[c] - summary[c]).abs().max()) for c in ("accuracy", "ci_low", "ci_high"))
        check(diff < 1e-12 and stored["lower_bound_reaches_threshold"].eq(summary["lower_bound_reaches_threshold"]).all(),
              f"accuracy_summary.csv equals the recomputed table (largest absolute difference {diff:.1e}, 42 values with intervals)")

    print("\n== Fig. 4a: share of correct choices (stratum-weighted), 95% bootstrap interval over combinations")
    wide = summary.assign(cell=lambda d: d.apply(lambda r: f"{100 * r.accuracy:5.1f} [{100 * r.ci_low:5.1f},{100 * r.ci_high:5.1f}]", axis=1)) \
        .pivot(index="model", columns="quantity", values="cell").reindex(MODELS)[list(QUANTITIES)]
    wide.index = [LABEL[m] for m in wide.index]
    print(wide.to_string())

    acc = summary.pivot(index="model", columns="quantity", values="accuracy").reindex(MODELS)
    lo = summary.pivot(index="model", columns="quantity", values="ci_low").reindex(MODELS)

    print("\n== Statements in the Results (Fig. 4a)")
    exact = [m for m in MODELS if acc.loc[m, "identification_expert_accuracy"] == 1.0]
    other = [LABEL[m] for m in MODELS if m not in exact]
    check(len(exact) == 6, f"models identifying the expert's stated accuracy correctly on every trial: recomputed {len(exact)} | paper six (not {', '.join(other)}: "
          f"{pct(acc.loc['qwen3-1.7b', 'identification_expert_accuracy'])}%)")
    for text, q, paper in (("identification of the cost", "identification_cost", (97.4, 100.0)),
                           ("identification of the own probability of answering correctly", "identification_own_accuracy", (53.7, 100.0)),
                           ("comparison of two numbers", "comparison", (98.5, 100.0)),
                           ("action selection from the stated terms", "action_selection", (58.5, 69.0)),
                           ("action selection with both expected scores", "action_selection_with_scores", (83.4, 99.7))):
        got = (pct(acc[q].min()), pct(acc[q].max()))
        check(got == paper, f"{text}: recomputed {got[0]}% to {got[1]}% | paper {paper[0]}% to {paper[1]}%")
    worst = acc["action_selection_with_scores"].idxmin(), acc["action_selection_with_scores"].idxmax()
    check((worst[0], worst[1]) == ("smollm3-3b", "qwen3-14b"), f"lowest and highest accuracy with both expected scores: {LABEL[worst[0]]} {pct(acc.loc[worst[0], 'action_selection_with_scores'])}%, "
          f"{LABEL[worst[1]]} {pct(acc.loc[worst[1], 'action_selection_with_scores'])}% | release diagnostic: SmolLM3-3B 83.4%, Qwen3-14B 99.7% (the text quotes the range without model names)")
    above = sorted(LABEL[m] for m in MODELS if lo.loc[m, "action_selection_with_scores"] > 0.90)
    paper_above = sorted(["Qwen3-4B", "Qwen3-14B", "Qwen3-32B", "Gemma-4-E4B-it"])
    check(above == paper_above, f"models with the lower 95% bound above 90% with both expected scores: recomputed {', '.join(above)} | paper {', '.join(paper_above)}")
    check(pct(acc.loc["qwen3-4b", "identification_own_accuracy"]) == 53.7, f"Qwen3-4B identifies its own probability on {pct(acc.loc['qwen3-4b', 'identification_own_accuracy'])}% | release diagnostic: lowest value of the range quoted in the text, 53.7%")

    print("\n== Repeated readings of Qwen3-32B")
    for m in ("qwen3-32b", "qwen3-14b"):
        r = repeats[repeats.model == m]
        gates = r["gate_pass_first"] & r["gate_pass_second"]
        tied = gates & (r["choice_first"].isna() | r["choice_second"].isna())
        flip = gates & r["choice_first"].notna() & r["choice_second"].notna() & (r["choice_first"] != r["choice_second"])
        equal = gates & ~tied & ~flip
        print(f"   {LABEL[m]}: {len(r)} prompts, {int(equal.sum())} identical choices without ties, {int(tied.sum())} involving a tie, {int(flip.sum())} reversals")
        if m == "qwen3-32b":
            check((len(r), int(equal.sum()), int(tied.sum()), int(flip.sum())) == (300, 297, 3, 0),
                  f"Qwen3-32B: identical choices without tied probabilities in {int(equal.sum())} of {len(r)}, {int(tied.sum())} with ties, {int(flip.sum())} reversals | release diagnostic: 297 of 300, three with ties, no reversals (not stated in the text)")
        else:
            check((len(r), int(equal.sum())) == (300, 300), f"Qwen3-14B: identical choices without ties in {int(equal.sum())} of {len(r)}")

    n_fail = sum(not ok for ok, _ in results)
    print(f"\nSUMMARY: {len(results) - n_fail} of {len(results)} checks reproduced")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
