"""Recompute the main-experiment numbers of the paper from the files in this directory.

Reads confidence.parquet, choices.parquet, conditions.csv and models.csv next to this script and prints each
recomputed value beside the value quoted in the paper. Requires numpy, pandas, pyarrow and scipy.

    python recompute_main_experiment.py                 # all quantities with 2,000 bootstrap resamples (about 4 min)
    python recompute_main_experiment.py --no-coefficient-bootstrap   # skip the slow interval of the confidence coefficient

Conventions (Methods of the paper): questions are put in ascending order of item_key; the bootstrap resamples whole
questions (all 54 conditions of a question stay together) with numpy.random.default_rng(0).integers(0, n, (2000, n));
intervals are the 2.5 and 97.5 percentiles; the expected-score baseline asks when confidence < a_H - c0 - w.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
MODELS = ['qwen3-1.7b', 'qwen3-4b', 'qwen3-14b', 'qwen3-32b', 'qwen2.5-7b-instruct', 'gemma-4-e4b-it', 'smollm3-3b']
A_H, C0, W = (0.60, 0.80, 0.95), (0.0, 0.1, 0.2), (0.0, 0.1, 0.2, 0.45, 0.8, 1.5)
CELLS = list(itertools.product(A_H, C0, W))            # 54 conditions in ascending order
NINE = list(itertools.product(A_H, C0))                # expert accuracy x fixed fee pooled at each waiting cost
RESAMPLES, SEED = 2000, 0


def cell(a, c, w):
    return CELLS.index((a, c, w))


def cols_at(w):
    return [cell(a, c, w) for a, c in NINE]


def load_model(conf, choices, model):
    q = conf[conf.model == model].sort_values('item_key')
    keys = q.item_key.to_numpy()
    d = choices[choices.model == model].sort_values(['expert_accuracy', 'fixed_fee', 'waiting_cost', 'item_key'])
    assert len(d) == 54 * len(keys)
    ask = d.ask.to_numpy(float).reshape(54, len(keys)).T          # (questions, 54)
    valid = d.valid.to_numpy(bool).reshape(54, len(keys)).T
    assert (d.item_key.to_numpy().reshape(54, len(keys)) == keys[None, :]).all()
    z = q.confidence.to_numpy(float)
    thr = np.array([round(a - c - w, 9) for a, c, w in CELLS])
    norm = (z[:, None] < thr[None, :]).astype(float)               # expected-score baseline
    return keys, z, ask, norm, valid


def boot_means(matrix, draws, chunk=200):
    m = np.asarray(matrix, float)
    out = np.empty((draws.shape[0],) + m.shape[1:])
    for i in range(0, draws.shape[0], chunk):
        out[i:i + chunk] = m[draws[i:i + chunk]].mean(axis=1)
    return out


def interval(vec, draws):
    means = boot_means(vec, draws)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.mean(vec)), float(lo), float(hi)


def deciles(z):
    ranks = rankdata(z, method='ordinal')
    edges = np.quantile(ranks, np.linspace(0, 1, 11))
    return np.searchsorted(edges[1:-1], ranks, side='left')


def decile_curve(z, v):
    b = deciles(z)
    return np.array([v[b == k].mean() for k in range(10)])


def fit_confidence_coefficient(z, ask, draws=None, chunk=50):
    """Slope of logit(z) in a logistic model of asking with one intercept per condition (Newton iterations).
    Conditions in which every outcome is the same do not enter the fit. Returns the full-sample slope and, if draws
    are given, the 2.5 and 97.5 percentiles of the slopes of the bootstrap resamples."""
    x = logit(np.clip(z, 1e-4, 1 - 1e-4))
    y = ask                                                              # (n, K)
    n, K = y.shape

    def newton(wt, start=None, iters=50):
        wt = np.atleast_2d(wt)                                           # (B, n) question weights
        B = wt.shape[0]
        n1 = wt @ y                                                      # (B, K)
        active = ((n1 > 0) & (n1 < wt.sum(axis=1, keepdims=True))).astype(float)
        if start is None:
            alpha = logit(np.clip(n1 / wt.sum(axis=1, keepdims=True), 1e-6, 1 - 1e-6))
            beta = np.zeros(B)
        else:
            alpha, beta = np.tile(start[0], (B, 1)), np.full(B, start[1])
        wx = wt * x[None, :]
        wxx = wx * x[None, :]
        for _ in range(iters):
            p = expit(alpha[:, None, :] + beta[:, None, None] * x[None, :, None])      # (B, n, K)
            r = (y[None] - p) * active[:, None, :]
            h = p * (1 - p) * active[:, None, :]
            g_a = (wt[:, None, :] @ r)[:, 0, :]
            g_b = (wx[:, None, :] @ r)[:, 0, :].sum(axis=1)
            D = (wt[:, None, :] @ h)[:, 0, :] + 1e-12
            c = (wx[:, None, :] @ h)[:, 0, :]
            E = (wxx[:, None, :] @ h)[:, 0, :].sum(axis=1)
            d_b = (g_b - (c / D * g_a).sum(axis=1)) / (E - (c * c / D).sum(axis=1))
            d_a = (g_a - c * d_b[:, None]) / D
            alpha, beta = alpha + d_a * active, beta + d_b
            if max(np.abs(d_a).max(), np.abs(d_b).max()) < 1e-8:
                break
        return beta, alpha

    beta0, alpha0 = newton(np.ones(n))
    out = {'beta': float(beta0[0])}
    if draws is not None:
        betas = np.empty(draws.shape[0])
        for i in range(0, draws.shape[0], chunk):
            idx = draws[i:i + chunk]
            wt = np.zeros((idx.shape[0], n))
            np.add.at(wt, (np.arange(idx.shape[0])[:, None], idx), 1.0)
            betas[i:i + chunk] = newton(wt, start=(alpha0[0], beta0[0]))[0]
        out['lo'], out['hi'] = (float(v) for v in np.quantile(betas, [0.025, 0.975]))
    return out


RESULTS = []


def check(label, value, paper, tol):
    ok = abs(value - paper) <= tol
    RESULTS.append(ok)
    print(f"{'OK  ' if ok else 'DIFF'} {label}: recomputed {value:.4f} | paper {paper:g} (tol {tol:g})")


def consistent(label, value, tol):
    ok = value <= tol
    RESULTS.append(ok)
    print(f"{'OK  ' if ok else 'DIFF'} {label}: {value:.3g} (tolerance {tol:g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-coefficient-bootstrap', action='store_true')
    args = ap.parse_args()
    conf = pd.read_parquet(HERE / 'confidence.parquet')
    choices = pd.read_parquet(HERE / 'choices.parquet')
    n = conf.item_key.nunique()
    draws = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))
    data = {m: load_model(conf, choices, m) for m in MODELS}
    print(f'{len(MODELS)} models, {n} questions, 54 conditions, {RESAMPLES} resamples')

    # consistency of the stored fields
    temps = pd.read_csv(HERE / 'models.csv').set_index('model').calibration_temperature
    s = conf[[f'logprob_{j}' for j in range(1, 5)]].to_numpy(float) / conf.model.astype(str).map(temps).to_numpy(float)[:, None]
    p = np.exp(s - s.max(axis=1, keepdims=True))
    z_re = (p / p.sum(axis=1, keepdims=True)).max(axis=1)
    print('\n== Stored fields')
    consistent('confidence recomputed from the option log-probabilities and the temperature, maximum difference', np.abs(z_re - conf.confidence.to_numpy(float)).max(), 2e-6)
    lp = choices[[f'logprob_{j}' for j in range(1, 6)]].to_numpy(float)
    consistent('margin recomputed from the log-probabilities, maximum difference', np.abs(lp[:, 4] - lp[:, :4].max(axis=1) - choices.margin.to_numpy(float)).max(), 5e-6)
    consistent('readouts whose stored ask value differs from margin > 0', int((choices.ask.to_numpy() != (choices.margin.to_numpy() > 0)).sum()), 0)
    consistent('readouts whose validity flag differs from digit_mass >= 0.5', int(((choices.digit_mass.to_numpy() >= 0.5) != choices.valid.to_numpy()).sum()), 0)
    print(f"readouts without a valid answer-only measurement: {int((~conf.valid).sum())} of {len(conf)}")

    # release diagnostic (the manuscript states no threshold): share of valid readouts in every condition, at least 99%
    worst = min(v.mean(axis=0).min() for *_, v in data.values())
    print(f'lowest share of valid readouts in any model and condition: {worst:.4f} (release diagnostic, expected at least 0.99)')
    RESULTS.append(worst >= 0.99)

    rate9 = {}
    for m in MODELS:
        _, z, ask, norm, _ = data[m]
        rate9[m] = {w: (ask[:, cols_at(w)].mean(axis=1), norm[:, cols_at(w)].mean(axis=1)) for w in W}

    print('\n== Fig. 1b: ask rate pooled over the nine expert-accuracy and fee combinations (observed | baseline), %')
    iv = {}
    for m in MODELS:
        iv[m] = {w: interval(rate9[m][w][0], draws) for w in W}
        row = ' '.join(f'{100 * iv[m][w][0]:5.1f}' for w in W)
        base = ' '.join(f'{100 * rate9[m][w][1].mean():5.1f}' for w in W)
        print(f'  {m:20s} w={W}: {row} | {base}')
    hi15 = {m: iv[m][1.5] for m in MODELS}
    lo_m, hi_m = min(hi15, key=lambda k: hi15[k][0]), max(hi15, key=lambda k: hi15[k][0])
    check(f'ask rate at w=1.5, lowest ({lo_m})', 100 * hi15[lo_m][0], 20.8, 0.05)
    check(f'  lower limit', 100 * hi15[lo_m][1], 19.3, 0.05)
    check(f'  upper limit', 100 * hi15[lo_m][2], 22.3, 0.05)
    check(f'ask rate at w=1.5, highest ({hi_m})', 100 * hi15[hi_m][0], 94.1, 0.05)
    check(f'  lower limit', 100 * hi15[hi_m][1], 93.3, 0.05)
    check(f'  upper limit', 100 * hi15[hi_m][2], 95.0, 0.05)
    check('baseline ask rate at w=1.5, maximum over models (%)', 100 * max(rate9[m][1.5][1].mean() for m in MODELS), 0.0, 1e-9)
    check('baseline ask rate at w=0.8, maximum over models (%)', 100 * max(rate9[m][0.8][1].mean() for m in MODELS), 0.0, 1e-9)
    r08 = [100 * iv[m][0.8][0] for m in MODELS]
    check('ask rate at w=0.8, lowest (%)', min(r08), 20.3, 0.05)
    check('ask rate at w=0.8, highest (%)', max(r08), 96.2, 0.05)

    print('\n== Results text: cost contrasts pooled over the nine combinations (percentage points)')
    first_obs, first_base = {}, {}
    for m in MODELS:
        first_obs[m] = 100 * (rate9[m][0.1][0] - rate9[m][0.0][0]).mean()
        first_base[m] = 100 * (rate9[m][0.1][1] - rate9[m][0.0][1]).mean()
        print(f'  {m:20s} 0 -> 0.1: observed {first_obs[m]:+6.2f}, baseline {first_base[m]:+6.2f}; '
              f'0.1 -> 1.5: observed {100 * (rate9[m][1.5][0] - rate9[m][0.1][0]).mean():+6.2f}; '
              f'0 -> 1.5: observed {100 * (rate9[m][1.5][0] - rate9[m][0.0][0]).mean():+6.2f}')
    check('first cost, observed change, minimum', min(first_obs.values()), -0.6, 0.05)
    check('first cost, observed change, maximum', max(first_obs.values()), 14.4, 0.05)
    check('first cost, baseline reduction, smallest', -max(first_base.values()), 10.1, 0.05)
    check('first cost, baseline reduction, largest', -min(first_base.values()), 15.6, 0.05)
    raised = [first_obs[m] for m in ('qwen3-1.7b', 'qwen3-14b', 'qwen3-32b', 'gemma-4-e4b-it')]
    check('first cost, increase in Qwen3-1.7B/14B/32B and Gemma, minimum', min(raised), 12.0, 0.05)
    check('first cost, increase in Qwen3-1.7B/14B/32B and Gemma, maximum', max(raised), 14.4, 0.05)
    g = data['gemma-4-e4b-it']
    check('Gemma-4-E4B-it ask rate at w=0.1 (%)', 100 * iv['gemma-4-e4b-it'][0.1][0], 62.0, 0.05)
    check('Gemma-4-E4B-it ask rate at w=1.5 (%)', 100 * iv['gemma-4-e4b-it'][1.5][0], 56.0, 0.05)
    top_minus_zero = {m: 100 * (rate9[m][1.5][0] - rate9[m][0.0][0]).mean() for m in MODELS}
    others = [v for m, v in top_minus_zero.items() if m != 'qwen2.5-7b-instruct']
    check('w=1.5 minus w=0, six models, minimum', min(others), 3.7, 0.05)
    check('w=1.5 minus w=0, six models, maximum', max(others), 15.1, 0.05)
    check('w=1.5 minus w=0, Qwen2.5-7B-Instruct', top_minus_zero['qwen2.5-7b-instruct'], -0.6, 0.05)

    print('\n== Fig. 2a: ask rate in confidence deciles, nine combinations pooled (%)')
    dec = {}
    for m in MODELS:
        _, z, ask, norm, _ = data[m]
        dec[m] = {(w, s): 100 * decile_curve(z, rate9[m][w][i]) for w in (0.0, 0.1) for i, s in enumerate(('obs', 'norm'))}
        print(f'  {m:20s} w=0 decile 1: {dec[m][(0.0, "obs")][0]:5.1f}, decile 10: {dec[m][(0.0, "obs")][9]:5.1f};  '
              f'w=0.1 decile 10: observed {dec[m][(0.1, "obs")][9]:5.1f}, baseline {dec[m][(0.1, "norm")][9]:5.1f}')
    check('least confident decile at w=0, minimum (%)', min(dec[m][(0.0, 'obs')][0] for m in MODELS), 38.3, 0.05)
    check('least confident decile at w=0, maximum (%)', max(dec[m][(0.0, 'obs')][0] for m in MODELS), 97.9, 0.05)
    check('most confident decile at w=0, minimum (%)', min(dec[m][(0.0, 'obs')][9] for m in MODELS), 0.0, 0.05)
    check('most confident decile at w=0, maximum (%)', max(dec[m][(0.0, 'obs')][9] for m in MODELS), 52.0, 0.05)
    check('Qwen3-32B, most confident decile at w=0.1 (%)', dec['qwen3-32b'][(0.1, 'obs')][9], 83.9, 0.05)
    check('baseline, most confident decile at w=0.1, maximum over models (%)', max(dec[m][(0.1, 'norm')][9] for m in MODELS), 4.9, 0.05)
    every = sum(bool((dec[m][(0.1, 'obs')] > dec[m][(0.0, 'obs')]).all()) for m in MODELS)
    check('models whose ask rate rises in every decile from w=0 to w=0.1', every, 5, 0)

    print('\n== Fig. 2b: coefficient of logit(confidence), 54 conditions, one intercept per condition')
    coef = {}
    for m in MODELS:
        _, z, ask, *_ = data[m]
        coef[m] = fit_confidence_coefficient(z, ask, None if args.no_coefficient_bootstrap else draws)
        extra = '' if args.no_coefficient_bootstrap else f"  [{coef[m]['lo']:.3f}, {coef[m]['hi']:.3f}]"
        print(f"  {m:20s} {coef[m]['beta']:+.3f}{extra}")
    check('coefficient, most negative', min(c['beta'] for c in coef.values()), -1.367, 0.0005)
    check('coefficient, closest to zero', max(c['beta'] for c in coef.values()), -0.516, 0.0005)
    if not args.no_coefficient_bootstrap:
        check('models whose 95% interval lies below zero', sum(c['hi'] < 0 for c in coef.values()), 7, 0)

    print('\n== Fig. 2c: change in ask rate when the stated expert accuracy rises from 60% to 95% (fee 0, waiting cost 0), pp')
    hi_c, lo_c = cell(0.95, 0.0, 0.0), cell(0.60, 0.0, 0.0)
    ch, bs, ivs = {}, {}, {}
    for m in MODELS:
        _, z, ask, norm, _ = data[m]
        ivs[m] = interval(ask[:, hi_c] - ask[:, lo_c], draws)
        ch[m], bs[m] = 100 * ivs[m][0], 100 * (norm[:, hi_c] - norm[:, lo_c]).mean()
        print(f'  {m:20s} observed {ch[m]:+5.2f} [{100 * ivs[m][1]:+5.2f}, {100 * ivs[m][2]:+5.2f}], baseline {bs[m]:+5.2f}')
    lo_m, hi_m = min(ch, key=ch.get), max(ch, key=ch.get)
    check(f'observed change, lowest ({lo_m})', ch[lo_m], -4.3, 0.05)
    check('  lower limit', 100 * ivs[lo_m][1], -5.6, 0.05)
    check('  upper limit', 100 * ivs[lo_m][2], -3.1, 0.05)
    check(f'observed change, highest ({hi_m})', ch[hi_m], 5.6, 0.05)
    check('  lower limit', 100 * ivs[hi_m][1], 4.4, 0.05)
    check('  upper limit', 100 * ivs[hi_m][2], 6.8, 0.05)
    check('baseline change, smallest', min(bs.values()), 42.0, 0.05)
    check('baseline change, largest', max(bs.values()), 72.4, 0.05)
    decreasing = [m for m in ('qwen3-1.7b', 'qwen3-4b', 'qwen3-32b', 'smollm3-3b')]
    check('Qwen3-1.7B/4B/32B and SmolLM3: smallest reduction (pp)', min(-ch[m] for m in decreasing), 0.8, 0.05)
    check('Qwen3-1.7B/4B/32B and SmolLM3: largest reduction (pp)', max(-ch[m] for m in decreasing), 4.3, 0.05)
    ratios = [ch[m] / bs[m] for m in MODELS if ch[m] > 0]
    check('largest observed change as a share of its baseline change', max(ratios), 0.11, 0.005)

    print(f'\nSUMMARY: {sum(RESULTS)} of {len(RESULTS)} checks reproduced')


if __name__ == '__main__':
    main()
