#!/usr/bin/env python3
"""Recompute ARC-Challenge numbers of the paper from the packaged data files only.

Usage: python recompute_arc.py [DATA_DIR]      (default: the directory that contains this file)
Requires numpy, pandas, pyarrow.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
MODELS = ['qwen3-4b', 'gemma-4-e4b-it', 'smollm3-3b']
LAB = {'qwen3-4b': 'Qwen3-4B', 'gemma-4-e4b-it': 'Gemma-4-E4B-it', 'smollm3-3b': 'SmolLM3-3B'}


# ------------------------------------------------------------------ statistics (2,000 resamples of questions, seed 0)
def question_means(ids, values):
    """Average the finite values inside each question; questions in sorted-key order."""
    df = pd.DataFrame({'q': ids, 'v': values}).dropna()
    return df.groupby('q', sort=True)['v'].mean()


def boot(ids, values, n_boot=2000, seed=0):
    x = question_means(ids, values).to_numpy(float)
    rng = np.random.default_rng(seed)
    s = np.empty(n_boot)
    for a in range(0, n_boot, 64):
        k = min(64, n_boot - a)
        s[a:a + k] = x[rng.integers(len(x), size=(k, len(x)))].mean(axis=1)
    lo, hi = np.quantile(s, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def signflip(ids, values, n=2000, seed=0):
    x = question_means(ids, values).to_numpy(float)
    obs = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    ext = 0
    for a in range(0, n, 64):
        k = min(64, n - a)
        signs = rng.choice([-1., 1.], size=(k, len(x)))
        ext += int(np.sum(np.abs((signs * x).mean(axis=1)) >= obs - 1e-14))
    return (ext + 1) / (n + 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    run = 0.
    for k, i in enumerate(order):
        run = max(run, min(1., (len(p) - k) * p[i]))
        out[i] = run
    return out


def fmt(est, lo, hi, scale=100.):
    return f'{est * scale:+.2f} [{lo * scale:+.2f}, {hi * scale:+.2f}]'


rows = []


def report(name, got, paper, ok):
    rows.append((name, got, paper, 'OK' if ok else 'DIFF'))
    print(f'{"OK  " if ok else "DIFF"} | {name} | package: {got} | paper: {paper}')


# ------------------------------------------------------------------ load
beh = pd.read_parquet(D / 'behaviour_test.parquet')
ans = pd.read_parquet(D / 'confidence_answers.parquet')
con = pd.read_csv(D / 'contrasts.csv')
cnd = pd.read_csv(D / 'conditions.csv')
net = {(r.setting, r.condition_id): r.expert_net_score for r in cnd.itertuples()}
ask = beh.pivot_table(index=['model', 'setting', 'item_key'], columns='condition_id', values='ask', aggfunc='first')
ask = beh.set_index(['model', 'setting', 'item_key', 'condition_id'])['ask'].astype('float')  # NaN when the readout is invalid
z = ans.set_index(['model', 'item_key'])['confidence_z']


def contrast_values(model, setting, contrast, which='observed'):
    """Per-row (question, label order) paired differences source minus base for one contrast."""
    c = con[(con.setting == setting) & (con.contrast == contrast)]
    qs, vals = [], []
    sub = ask.loc[(model, setting)]
    zz = z.loc[model]
    for r in c.itertuples():
        if which == 'observed':
            a = sub.xs(r.base_condition, level='condition_id')
            b = sub.xs(r.source_condition, level='condition_id')
            d = (b - a)  # NaN propagates: only pairs with both readouts valid are kept
        else:
            zs = zz.reindex(sub.xs(r.base_condition, level='condition_id').index)
            d = (zs < r.source_net_score).astype(float) - (zs < r.base_net_score).astype(float)
        qs.append(d.index.to_numpy())
        vals.append(d.to_numpy(float))
    return np.concatenate(qs), np.concatenate(vals)


# ------------------------------------------------------------------ 1. Figure 4b / Figure 3f contrasts (512 questions)
paper_first = {'qwen3-4b': (5.0, 3.6, 6.4), 'smollm3-3b': (1.2, 0.6, 1.9), 'gemma-4-e4b-it': (-1.1, -2.2, 0.1)}
for m in MODELS:
    q, v = contrast_values(m, 'original', 'first_cost')
    e, lo, hi = boot(q, v)
    p = paper_first[m]
    ok = all(abs(a - b) < 0.06 for a, b in zip((e * 100, lo * 100, hi * 100), p))
    report(f'ask-rate change, cost 0 -> 0.1, original setting, {LAB[m]} (pp)', fmt(e, lo, hi), f'{p[0]:+.1f} [{p[1]:+.1f}, {p[2]:+.1f}]', ok)

paper_base = {'qwen3-4b': -12.9, 'smollm3-3b': -11.7, 'gemma-4-e4b-it': -6.8}
for m in MODELS:
    q, v = contrast_values(m, 'original', 'first_cost', which='baseline')
    e = question_means(q, v).mean() * 100
    report(f'expected-score baseline change, cost 0 -> 0.1, {LAB[m]} (pp)', f'{e:+.1f}', f'{paper_base[m]:+.1f}', abs(e - paper_base[m]) < 0.06)

paper_wait = {'qwen3-4b': (-2.6, -4.3, -1.1), 'gemma-4-e4b-it': (4.9, 3.2, 6.4)}
for m, p in paper_wait.items():
    q, v = contrast_values(m, 'original', 'semantic')
    e, lo, hi = boot(q, v)
    ok = all(abs(a - b) < 0.06 for a, b in zip((e * 100, lo * 100, hi * 100), p))
    report(f'ask-rate change, cost stated as a wait, {LAB[m]} (pp)', fmt(e, lo, hi), f'{p[0]:+.1f} [{p[1]:+.1f}, {p[2]:+.1f}]', ok)

q, v = contrast_values('qwen3-4b', 'original', 'equal_value_zero')
e, lo, hi = boot(q, v)
report('Figure 3f, Qwen3-4B, accuracy and cost raised together from free help, original (pp)', fmt(e, lo, hi),
       '+7.6 [+6.1, +9.4]', all(abs(a - b) < 0.06 for a, b in zip((e * 100, lo * 100, hi * 100), (7.6, 6.1, 9.4))))

# higher cost: largest reduction in asking across the three models (text: at most 2.1 pp)
dec = []
for m in MODELS:
    q, v = contrast_values(m, 'original', 'positive_cost')
    dec.append(question_means(q, v).mean() * 100)
report('ask-rate change, cost 0.1 -> 1.5: largest decrease over the three models (pp)', f'{min(dec):+.2f}', 'at most 2.1 decrease',
       abs(min(dec)) <= 2.15)

base_dec = []
for m in MODELS:
    q, v = contrast_values(m, 'original', 'positive_cost', which='baseline')
    base_dec.append(question_means(q, v).mean() * 100)
report('expected-score baseline change, cost 0.1 -> 1.5 (pp, smallest and largest decrease over models)',
       f'{-max(base_dec):.1f} to {-min(base_dec):.1f} decrease', '14.5 to 41.8 decrease',
       abs(-max(base_dec) - 14.5) < 0.06 and abs(-min(base_dec) - 41.8) < 0.06)

# ------------------------------------------------------------------ 2. asking at the highest cost (w = 1.5, stated expert accuracy 0.80, neutral wording, no fee)
paper_hi = {'qwen3-4b': 58.3, 'gemma-4-e4b-it': 12.3, 'smollm3-3b': 6.9}
hi_cond = cnd[(cnd.setting == 'original') & (cnd.waiting_cost == 1.5) & (cnd.fixed_fee == 0) & (cnd.wording == 'neutral')
              & (cnd.expert_accuracy == 0.80)].condition_id
for m in MODELS:
    a = beh[(beh.model == m) & (beh.setting == 'original') & beh.condition_id.isin(hi_cond)]
    rate = question_means(a.item_key, a.ask.astype('float')).mean() * 100
    report(f'ask rate at cost 1.5 with stated expert accuracy 0.80, original setting, {LAB[m]} (%)', f'{rate:.1f}', f'{paper_hi[m]:.1f}', abs(rate - paper_hi[m]) < 0.06)

# ------------------------------------------------------------------ 3. activation patching, held-out tests
pt = pd.read_parquet(D / 'patching_test.parquet')
sites = pd.read_csv(D / 'patching_sites.csv')
pt['eff'] = (pt.margin_swap - pt.margin_swap_ref) - (pt.margin_random - pt.margin_random_ref)
tests = []
for (m, s, layer, pos, c, d), g in pt.groupby(['model', 'setting', 'layer', 'position', 'contrast', 'direction'], sort=True):
    e, lo, hi = boot(g.item_key, g.eff)
    tests.append(dict(model=m, setting=s, layer=layer, position=pos, contrast=c, direction=d, est=e, lo=lo, hi=hi,
                      p=signflip(g.item_key, g.eff)))
tests = pd.DataFrame(tests)
assert len(tests) == 72
tests['holm'] = holm(tests.p)
sg = {(r.model, r.layer, r.position, r.contrast): r.development_sign for r in sites.itertuples()}
tests['dev_sign'] = [sg[(r.model, r.layer, r.position, r.contrast)] for r in tests.itertuples()]
tests['expected'] = np.where(tests.direction == 'forward', tests.dev_sign, -tests.dev_sign)
tests['replicated'] = (tests.est * tests.expected > 0) & (tests.holm <= .05) & ((tests.lo > 0) | (tests.hi < 0))
both = tests.groupby(['model', 'setting', 'layer', 'position', 'contrast']).replicated.all()
report('comparisons with margin changes in both predicted directions after Holm correction (of 36)', int(both.sum()), '18 of 36', int(both.sum()) == 18)

flag = both.reset_index(name='both_directions')
fc = flag[flag.contrast == 'first_cost']
ok_fc = all(fc[(fc.model == m) & (fc.position == pos)].both_directions.all() for m in ('qwen3-4b', 'smollm3-3b') for pos in ('cost_number', 'decision'))
report('first cost: both predicted directions at both positions in Qwen3-4B and SmolLM3-3B (both settings)', bool(ok_fc), 'reproduced', bool(ok_fc))
sw = flag[flag.contrast == 'semantic']
ok_sw = bool(sw[sw.position == 'decision'].both_directions.all() and not sw[sw.position == 'cost_number'].both_directions.any())
report('waiting-cost wording: both directions at the response position in all three models, at no cost-number position',
       ok_sw, 'response position only', ok_sw)

# ------------------------------------------------------------------ 4. prompt interventions
pi = pd.read_parquet(D / 'prompt_intervention_test.parquet')
pi = pi.merge(cnd[['setting', 'condition_id', 'expert_net_score']], on=['setting', 'condition_id'])
pi = pi.merge(ans[['model', 'item_key', 'confidence_z']], on=['model', 'item_key'])


def score(ask_col, correct_col, v):
    a = pi[ask_col].astype('float')
    return np.where(a.isna(), np.nan, np.where(a == 1, v, pi[correct_col].astype('float')))


pi['score_before'] = score('ask_before', 'answer_correct_before', pi.expert_net_score)
pi['score_after'] = score('ask_after', 'answer_correct_after', pi.expert_net_score)
pi['d_score'] = pi.score_after - pi.score_before
pi['d_acc'] = pi.answer_correct_after.astype('float') - pi.answer_correct_before.astype('float')
grp = []
for (m, s, arm), g in pi.groupby(['model', 'setting', 'arm'], sort=True):
    e, lo, hi = boot(g.item_key, g.d_score)
    complete = g.d_score.notna().all()
    p = signflip(g.item_key, g.d_score) if complete else 1.0
    grp.append(dict(model=m, setting=s, arm=arm, d_score=e, lo=lo, hi=hi, d_acc=boot(g.item_key, g.d_acc)[0], p=p))
grp = pd.DataFrame(grp)
assert len(grp) == 18
grp['holm'] = holm(grp.p)
o = grp[(grp.setting == 'original') & (grp.arm == 'score_table')]
report('score table, original setting: change in expected score per question (min, max over models)',
       f'{o.d_score.min():+.3f} to {o.d_score.max():+.3f}', '-0.265 to +0.026', abs(o.d_score.min() + 0.265) < 0.0006 and abs(o.d_score.max() - 0.026) < 0.0006)
report('score table, original setting: change in independent-answer accuracy (pp, min, max)',
       f'{o.d_acc.max() * 100:+.1f} to {o.d_acc.min() * 100:+.1f}', '-5.6 to -35.6',
       abs(o.d_acc.max() * 100 + 5.6) < 0.06 and abs(o.d_acc.min() * 100 + 35.6) < 0.06)
q4 = grp[(grp.model == 'qwen3-4b') & (grp.setting == 'original') & (grp.arm == 'score_table')].iloc[0]
report('Qwen3-4B score-table gain, original setting (points per question; Holm P over 18 comparisons)',
       f'{q4.d_score:+.3f} (Holm P = {q4.holm:.3f})', '+0.026, not significant after correction', abs(q4.d_score - 0.026) < 0.0006 and q4.holm > 0.05)

# external expected-score rule versus the models' own choices (original prompt, original setting)
ext = []
for m in MODELS:
    g = pi[(pi.model == m) & (pi.setting == 'original') & (pi.arm == 'score_table')]
    ask_ext = (g.expert_net_score - g.confidence_z) > 0
    s_ext = np.where(ask_ext, g.expert_net_score, g.answer_correct_before.astype('float'))
    ext.append((m, question_means(g.item_key, s_ext).mean(), question_means(g.item_key, g.score_before).mean()))
report('external rule scores higher than the models own choices in all three models (original setting)',
       '; '.join(f'{LAB[m]} {a:.3f} vs {b:.3f}' for m, a, b in ext), 'higher in all three', all(a > b for _, a, b in ext))

# ------------------------------------------------------------------ 5. development scan (Extended Data Fig. 2 heat maps)
dv = pd.read_parquet(D / 'patching_scan_development.parquet')
dv['eff'] = (dv.margin_swap - dv.margin_swap_ref) - (dv.margin_random - dv.margin_random_ref)
qd = dv.groupby(['model', 'contrast', 'position', 'layer', 'direction', 'item_key']).eff.mean().unstack('direction')
T = ((qd['forward'] - qd['reverse']) / 2).groupby(['model', 'contrast', 'position', 'layer']).mean()
print(f'development scan: {len(T)} cells (model x contrast x position x layer), {dv.item_key.nunique()} questions; '
      f'example Qwen3-4B first_cost cost_number layer 0 T = {T.loc[("qwen3-4b", "first_cost", "cost_number", 0)]:.6f}')

bad = [r for r in rows if r[3] != 'OK']
print(f'\n{len(rows) - len(bad)} of {len(rows)} checks agree with the paper')
