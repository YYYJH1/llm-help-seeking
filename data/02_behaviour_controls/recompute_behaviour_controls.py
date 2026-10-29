"""Recompute the control-experiment numbers of the paper (Fig. 3a to 3e and the Results text) from the package files.

Reads choices.parquet in this directory and, for the original prompt and the confidence estimates, confidence.parquet
and choices.parquet of ../01_main_experiment. Prints each recomputed value beside the value quoted in the paper.
Requires numpy, pandas and pyarrow.

Conventions: questions in ascending order of item_key; the bootstrap resamples whole questions with
numpy.random.default_rng(0).integers(0, n, (2000, n)); intervals are the 2.5 and 97.5 percentiles; contrasts are
computed per question before averaging, so the intervals are paired. The expected-score baseline asks when
confidence < 0.8 - x (deduction on consultation, wait wording, stated expected score, goal reminder),
< 0.8 + x (deduction on independent answers) or < (0.8 + x) / (1 + x) (deduction on incorrect independent answers),
with x the cost level.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / '01_main_experiment'
MODELS = ['qwen3-1.7b', 'qwen3-4b', 'qwen3-14b', 'qwen3-32b', 'qwen2.5-7b-instruct', 'gemma-4-e4b-it', 'smollm3-3b']
LEVELS = (0.0, 0.1, 0.2, 0.45, 0.8, 1.5)
RESAMPLES, SEED = 2000, 0
THRESHOLD = {                                    # expected-score baseline: ask when confidence < threshold(x)
    'original': lambda x: 0.8 - x,
    'deduct_consultation': lambda x: 0.8 - x,
    'deduct_independent_answer': lambda x: 0.8 + x,
    'deduct_incorrect_independent_answer': lambda x: (0.8 + x) / (1 + x),
    'expected_score_stated': lambda x: 0.8 - x,
    'goal_reminder': lambda x: 0.8 - x,
}
RESULTS = []


def check(label, value, paper, tol):
    ok = abs(value - paper) <= tol
    RESULTS.append(ok)
    print(f"{'OK  ' if ok else 'DIFF'} {label}: recomputed {value:.4f} | paper {paper:g} (tol {tol:g})")


def flag(label, ok):
    RESULTS.append(bool(ok))
    print(f"{'OK  ' if ok else 'DIFF'} {label}")


def main():
    conf = pd.read_parquet(MAIN / 'confidence.parquet', columns=['model', 'item_key', 'confidence'])
    orig = pd.read_parquet(MAIN / 'choices.parquet', columns=['model', 'item_key', 'expert_accuracy', 'fixed_fee', 'waiting_cost', 'ask'])
    orig = orig[(orig.expert_accuracy == 0.8) & (orig.fixed_fee == 0.0)]
    ctl = pd.read_parquet(HERE / 'choices.parquet', columns=['model', 'item_key', 'variant', 'cost_level', 'ask', 'valid'])
    n = conf.item_key.nunique()
    draws = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))
    print(f'{len(MODELS)} models, {n} questions, {ctl.groupby(["variant", "cost_level"], observed=True, dropna=False).ngroups} control conditions, {RESAMPLES} resamples')
    share = ctl.groupby(['variant', 'model', 'cost_level'], observed=True, dropna=False).valid.mean()
    print('\n== Readout validity: models whose readouts are valid on at least 99% of questions at every cost level')
    for variant, g in share.groupby(level='variant', observed=True):
        low = sorted({m for (v, m, x), s in g.items() if s < 0.99})
        print(f'  {variant:38s} {len(MODELS) - len(low)} of {len(MODELS)} models' + (f' (below 99%: {", ".join(low)})' if low else ''))

    def interval(vec):
        means = np.concatenate([vec[draws[i:i + 250]].mean(axis=1) for i in range(0, RESAMPLES, 250)])
        lo, hi = np.quantile(means, [0.025, 0.975])
        return 100 * vec.mean(), 100 * lo, 100 * hi

    res = {}
    for m in MODELS:
        q = conf[conf.model == m].sort_values('item_key')
        keys, z = q.item_key.to_numpy(), q.confidence.to_numpy(float)
        obs = {'original': {}}
        o = orig[orig.model == m]
        for x in LEVELS:
            g = o[o.waiting_cost == x].sort_values('item_key')
            assert (g.item_key.to_numpy() == keys).all()
            obs['original'][x] = g.ask.to_numpy(float)
        for (variant, x), g in ctl[ctl.model == m].groupby(['variant', 'cost_level'], observed=True, dropna=False):
            g = g.sort_values('item_key')
            assert (g.item_key.to_numpy() == keys).all()
            obs.setdefault(variant, {})[None if pd.isna(x) else float(x)] = g.ask.to_numpy(float)
        norm = {v: {x: (z < round(f(x), 9)).astype(float) for x in LEVELS} for v, f in THRESHOLD.items()}
        d = lambda s, v, hi, lo: s[v][hi] - s[v][lo]
        r = {}
        r['mention'] = interval(obs['wait_no_deduction'][0.0] - obs['no_deduction_no_wait'][None])
        r['duration'] = interval(d(obs, 'wait_no_deduction', 0.1, 0.0))
        for name, v in (('answer', 'deduct_independent_answer'), ('wrong', 'deduct_incorrect_independent_answer')):
            r[name] = interval(d(obs, v, 0.1, 0.0) - d(obs, 'deduct_consultation', 0.1, 0.0))
            r[name + '_baseline'] = 100 * (d(norm, v, 0.1, 0.0) - d(norm, 'deduct_consultation', 0.1, 0.0)).mean()
        for v in ('original', 'expected_score_stated', 'goal_reminder'):
            r[v + '_first'] = interval(d(obs, v, 0.1, 0.0))
            r[v + '_rise'] = interval(d(obs, v, 1.5, 0.1))
        res[m] = r

    def fmt(t):
        return f'{t[0]:+6.2f} [{t[1]:+6.2f}, {t[2]:+6.2f}]'

    print('\n== Fig. 3a: costless waiting (percentage points)')
    for m in MODELS:
        print(f"  {m:20s} wait mentioned (none -> 0 s) {fmt(res[m]['mention'])}   wait length (0 -> 10 s) {fmt(res[m]['duration'])}")
    up = [res[m]['mention'] for m in MODELS if res[m]['mention'][1] > 0]
    flag(f'wait mention: {len(up)} models with a positive interval (paper: five of seven)', len(up) == 5)
    check('wait mention, smallest increase (pp)', min(t[0] for t in up), 0.7, 0.05)
    check('wait mention, largest increase (pp)', max(t[0] for t in up), 23.0, 0.05)
    check('wait length, Qwen3-32B (pp)', res['qwen3-32b']['duration'][0], -8.1, 0.05)
    check('wait length, Qwen2.5-7B-Instruct (pp)', res['qwen2.5-7b-instruct']['duration'][0], -1.6, 0.05)
    check('wait length, Gemma-4-E4B-it (pp)', res['gemma-4-e4b-it']['duration'][0], -1.1, 0.05)
    check('wait length, SmolLM3-3B (pp)', res['smollm3-3b']['duration'][0], 0.7, 0.05)

    print('\n== Fig. 3b and 3c: moving the deduction (change of the first-cost response relative to deduction on consultation, pp)')
    for m in MODELS:
        r = res[m]
        print(f"  {m:20s} answering minus asking {fmt(r['answer'])} baseline {r['answer_baseline']:+6.2f};   "
              f"wrong answer minus asking {fmt(r['wrong'])} baseline {r['wrong_baseline']:+6.2f}")
    below = [m for m in MODELS if res[m]['answer'][2] < res[m]['answer_baseline']]
    flag(f'deduction on independent answers: {len(below)} models with an interval below the baseline (paper: six; Qwen3-32B close to it)', len(below) == 6)
    check('Qwen3-32B, observed minus baseline (pp)', res['qwen3-32b']['answer'][0] - res['qwen3-32b']['answer_baseline'], 0.0, 1.0)
    below = [m for m in MODELS if res[m]['wrong'][2] < res[m]['wrong_baseline']]
    flag(f'deduction on incorrect answers: {len(below)} models with an interval below the baseline (paper: smaller than the baseline in all)', len(below) == 7)
    opposite = sorted(m for m in MODELS if res[m]['wrong'][2] < 0)
    flag(f'deduction on incorrect answers, opposite direction: {opposite} (paper: Qwen3-32B and SmolLM3-3B)', opposite == ['qwen3-32b', 'smollm3-3b'])

    print('\n== Fig. 3d and 3e: response to the first cost (0 -> 0.1) and to the higher cost (0.1 -> 1.5), three prompts (pp)')
    for m in MODELS:
        r = res[m]
        print(f"  {m:20s} original {fmt(r['original_first'])} / {fmt(r['original_rise'])};  expected score stated {fmt(r['expected_score_stated_first'])} / "
              f"{fmt(r['expected_score_stated_rise'])};  goal reminder {fmt(r['goal_reminder_first'])} / {fmt(r['goal_reminder_rise'])}")
    f3 = {m: res[m]['expected_score_stated_first'] for m in MODELS}
    flag('stated expected score: asking rises at the first cost in all seven models', all(t[1] > 0 for t in f3.values()))
    lo, hi = min(f3, key=lambda k: f3[k][0]), max(f3, key=lambda k: f3[k][0])
    check(f'first cost, smallest rise ({lo})', f3[lo][0], 0.7, 0.05)
    check('  lower limit', f3[lo][1], 0.3, 0.05)
    check('  upper limit', f3[lo][2], 1.1, 0.05)
    check(f'first cost, largest rise ({hi})', f3[hi][0], 12.0, 0.05)
    check('  lower limit', f3[hi][1], 10.8, 0.05)
    check('  upper limit', f3[hi][2], 13.4, 0.05)
    falls = sorted(m for m in MODELS if res[m]['expected_score_stated_rise'][2] < 0)
    flag(f'stated expected score, higher cost: asking falls only in {falls} (paper: Qwen3-14B)', falls == ['qwen3-14b'])
    check('goal reminder, Gemma-4-E4B-it, first cost (pp)', res['gemma-4-e4b-it']['goal_reminder_first'][0], 12.0, 0.05)
    check('goal reminder, Gemma-4-E4B-it, higher cost (pp)', res['gemma-4-e4b-it']['goal_reminder_rise'][0], -9.8, 0.05)

    print(f'\nSUMMARY: {sum(RESULTS)} of {len(RESULTS)} checks reproduced')


if __name__ == '__main__':
    main()
