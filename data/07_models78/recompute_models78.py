"""Recompute the 46-model paper results (Extended Data Fig. 1 and Results) and diagnostics for the full 78-model archive.

Reads the parquet and csv files next to this script and prints each recomputed value beside the reference value (the
paper text or the plotted value of Extended Data Fig. 1). Lines marked release diagnostic are not quoted in the paper.
Requires numpy, pandas and pyarrow (a few seconds).

Conventions: the ask rate of a point is the number of ask outcomes divided by all planned trials at that cost; questions
follow bootstrap_item_order.csv for 100-question series and the stored order for other series; intervals are the
2.5 and 97.5 percentiles of 2,000 resamples of questions drawn
with numpy.random.default_rng(0).integers(0, n, (2000, n)), the same resamples at all six costs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
COSTS = [0.0, 0.1, 0.2, 0.45, 0.8, 1.5]
OK = []


def report(name, got, paper, tol, kind="reference"):
    good = abs(got - paper) <= tol
    OK.append(good)
    print(f"{'OK  ' if good else 'DIFF'} {name}: recomputed {got:.4f} | {kind} {paper} (tol {tol})")


models = pd.read_csv(PKG / "models.csv").fillna("")
main = pd.read_parquet(PKG / "choices_all_models.parquet")
alt = pd.read_parquet(PKG / "choices_alternative_request_formats.parquet")
bootstrap_order = pd.read_csv(PKG / "bootstrap_item_order.csv").set_index("item_key").bootstrap_position.to_dict()
print("models in roster:", len(models), "| local:", (models.source_kind == "local").sum(), "| api:", (models.source_kind == "api").sum())
print("models with trial records:", main.model.nunique(), "| rows:", len(main), "| alternative-format rows:", len(alt))
shown = models[models.shown_in_figure]
print("displayed models:", len(shown), "| developers:", shown.developer.nunique())


def series(model, a_h=0.8):
    """Items (in stored order) x six costs, True where the decision is ask, plus the item count."""
    kind = models.set_index("model").loc[model, "series_shown"]
    df = alt if (kind == "alternative_request_format" and a_h == 0.8) else main
    d = df[(df.model == model) & np.isclose(df.expert_accuracy, a_h)]
    n = d.item_key.nunique()
    out = np.zeros((n, 6), dtype=bool)
    first = d[np.isclose(d.waiting_cost, 0.0)].item_key.tolist()
    if n == 100:
        assert set(first) == set(bootstrap_order), "100-question series differs from the recorded bootstrap item set"
        first.sort(key=bootstrap_order.__getitem__)
    pos = {k: i for i, k in enumerate(first)}
    for j, w in enumerate(COSTS):
        dj = d[np.isclose(d.waiting_cost, w)]
        out[[pos[k] for k in dj.item_key], j] = (dj.decision.astype(str) == "ask").to_numpy()
    return out


def curve(model, a_h=0.8, R=2000):
    ask = series(model, a_h).astype(float)
    n = ask.shape[0]
    draws = np.random.default_rng(0).integers(0, n, size=(R, n))
    means = np.stack([ask[draws[k]].mean(0) for k in range(R)])
    lo, hi = np.quantile(means, [0.025, 0.975], axis=0)
    return ask.mean(0), lo, hi


# The paper counts displayed models by the lower 95% bound of the ask rate at the highest waiting cost.
low15 = {m: curve(m)[1][-1] for m in shown.model}
report("displayed models whose lower 95% bound of the ask rate at w=1.5 exceeds 5%", sum(v > 0.05 for v in low15.values()), 28, 0)
report("displayed models whose lower 95% bound of the ask rate at w=1.5 exceeds 50%", sum(v > 0.5 for v in low15.values()), 16, 0)
# Recent frontier models that do not ask at the highest waiting cost (expert accuracy 0.80), printed as a release diagnostic.
for display in ("GPT-6.1 Sol", "Claude Opus 5.5", "Gemini 4 Argon"):
    name = models.loc[models.display_name == display, "model"].item()
    report(f"{display}, ask outcomes at w=1.5", int(series(name)[:, -1].sum()), 0, 0, kind="release diagnostic")
rate15 ={m: series(m)[:, -1].mean() for m in shown.model}
n5 = sum(v > 0.05 for v in rate15.values())
nh = sum(v > 0.5 for v in rate15.values())
report("displayed models with a point estimate above 5% at w=1.5", n5, 31, 0, kind="release diagnostic")
report("displayed models with a point estimate above 50% at w=1.5", nh, 18, 0, kind="release diagnostic")
report("displayed models", len(shown), 46, 0)
report("developers among displayed models", shown.developer.nunique(), 9, 0)
report("models in the full archive", len(models), 78, 0, kind="release diagnostic")

print("== curves with paired item-bootstrap intervals (2,000 resamples, seed 0)")
r, lo, hi = curve("mistral-7b-instruct-v0.3")
report("Mistral-7B-v0.3, w=0 ask rate", r[0], 0.615, 5e-4)
report("Mistral-7B-v0.3, w=0 lower limit", lo[0], 0.584, 5e-4)
report("Mistral-7B-v0.3, w=1.5 ask rate", r[-1], 0.687, 5e-4)
report("Mistral-7B-v0.3, w=1.5 lower limit", lo[-1], 0.658, 5e-4)
report("Mistral-7B-v0.3, w=1.5 upper limit", hi[-1], 0.717, 5e-4)
r, lo, hi = curve("qwen3-14b")
report("Qwen3-14B, w=0 ask rate", r[0], 0.403, 5e-4)
report("Qwen3-14B, w=0 lower limit", lo[0], 0.373, 5e-4)
report("Qwen3-14B, w=0 upper limit", hi[0], 0.432, 5e-4)
report("Qwen3-14B, w=1.5 ask rate", r[-1], 0.625, 5e-4)
r, lo, hi = curve("api:openai/gpt-6-sol", 0.95)
report("gpt-6-sol at stated expert accuracy 0.95, w=0 ask rate", r[0], 0.323, 5e-4)
report("gpt-6-sol at stated expert accuracy 0.95, w=0 lower limit", lo[0], 0.295, 5e-4)
report("gpt-6-sol at stated expert accuracy 0.95, w=0 upper limit", hi[0], 0.352, 5e-4)
report("gpt-6-sol at stated expert accuracy 0.95, w=1.5 ask rate", r[-1], 0.0, 5e-4)
r, lo, hi = curve("api:openai/gpt-3.5-turbo")
report("100-question API series, w=0.45 lower limit", lo[3], 0.56, 5e-4)
report("100-question API series, w=0.8 upper limit", hi[4], 0.74, 5e-4)
report("100-question API series, w=1.5 upper limit", hi[-1], 0.82, 5e-4)
print(f"\nSUMMARY: {sum(OK)} of {len(OK)} checks reproduced")
