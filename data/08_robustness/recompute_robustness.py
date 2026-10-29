"""Additional analyses of saved model outputs. No generation or training.

The primary data and definitions remain unchanged. Cost directions and the five-point
band are reported separately. Alternative TriviaQA scores use the original development
fits; all eight Figure 6 models, candidates and choices are retained. The condition-level
control uses development accuracy and never selects its policy from held-out labels.
"""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,itertools
import numpy as np
import pandas as pd
from scipy.special import expit,logit

import argparse
parser=argparse.ArgumentParser(description="Additional cost-response, calibration and fixed-answer routing checks from stored outputs.")
parser.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--out',type=Path,default=Path.cwd()/'build/robustness_check')
args=parser.parse_args()
INPUT=Path(__file__).resolve().parent
DATA=args.data_root.resolve();WORK=args.out.resolve();WORK.mkdir(parents=True,exist_ok=True)
R=2000;SEED=0
def interval(v):
    x=np.asarray(v,float)
    assert np.isfinite(x).all()
    draws=np.random.default_rng(SEED).integers(0,len(x),(R,len(x)))
    means=np.concatenate([x[draws[i:i+100]].mean(axis=1)for i in range(0,R,100)])
    lo,hi=np.quantile(means,[.025,.975])
    return {'value':float(x.mean()),'lo':float(lo),'hi':float(hi),'n':len(x)}

# Supplemental direction/magnitude labels use exactly the original paired draws.
conf=pd.read_parquet(DATA/'01_main_experiment/confidence.parquet')
choices=pd.read_parquet(DATA/'01_main_experiment/choices.parquet')
cost=[];calib=[];bins=[]
for model,q in conf.groupby('model',sort=True):
    q=q.sort_values('item_key');keys=q.item_key.to_numpy();n=len(q)
    d=choices[choices.model==model].sort_values(['expert_accuracy','fixed_fee','waiting_cost','item_key'])
    assert len(d)==n*54
    assert np.all(d.item_key.to_numpy().reshape(54,n)==keys[None,:])
    y=d.ask.to_numpy(float).reshape(3,3,6,n).mean(axis=(0,1)).T
    for name,x in [('first_cost',y[:,1]-y[:,0]),('higher_cost',y[:,5]-y[:,1])]:
        v=interval(x);lo,hi=v['lo'],v['hi']
        original='reversal'if lo>0 else'decrease'if hi<-.05 else'little_change'if lo>-.05 and hi<.05 else'undecided'
        direction='increase'if lo>0 else'decrease'if hi<0 else'unresolved'
        cost.append(dict(model=model,contrast=name,original_category=original,direction=direction,
            within_five_pp=lo>-.05 and hi<.05,**v))
    p=q.confidence.to_numpy(float);correct=q.answer_correct.to_numpy(float)
    ids=np.minimum((p*10).astype(int),9)
    for j in range(10):
        ix=ids==j
        if ix.any():bins.append(dict(model=model,bin=j,n=int(ix.sum()),confidence=float(p[ix].mean()),accuracy=float(correct[ix].mean())))
    ece=sum(np.mean(ids==j)*abs(correct[ids==j].mean()-p[ids==j].mean())for j in range(10)if np.any(ids==j))
    residuals=(correct-p)[:,None]*np.eye(10)[ids]
    ece_draws=np.random.default_rng(SEED).integers(0,n,(R,n))
    ece_boot=np.concatenate([np.abs(residuals[ece_draws[k:k+50]].mean(1)).sum(1)for k in range(0,R,50)])
    ece_lo,ece_hi=np.quantile(ece_boot,[.025,.975])
    raw=q[[f'logprob_{k}'for k in range(1,5)]].to_numpy(float);raw=np.exp(raw-raw.max(1,keepdims=True));praw=(raw/raw.sum(1,keepdims=True)).max(1)
    rawids=np.minimum((praw*10).astype(int),9)
    raw_ece=sum(np.mean(rawids==j)*abs(correct[rawids==j].mean()-praw[rawids==j].mean())for j in range(10)if np.any(rawids==j))
    calib.append(dict(model=model,n=n,ece=float(ece),ece_lo=float(ece_lo),ece_hi=float(ece_hi),raw_ece=float(raw_ece),mean_confidence=float(p.mean()),answer_accuracy=float(correct.mean())))
pd.DataFrame(cost).to_csv(WORK/'cost_direction_magnitude.csv',index=False,lineterminator='\n')
pd.DataFrame(calib).to_csv(WORK/'calibration_diagnostics.csv',index=False,lineterminator='\n')
pd.DataFrame(bins).to_csv(WORK/'calibration_bins.csv',index=False,lineterminator='\n')

# Original scoring flags, without text, are verified against the frozen remote files.
flags=pd.read_parquet(INPUT/'scoring_flags.parquet')
tables={alias:g.drop(columns='alias').sort_values('item_key').reset_index(drop=True)for alias,g in flags.groupby('alias',sort=True)}
frozen_fits=pd.read_csv(INPUT/'sensitivity_calibration.csv',float_precision='round_trip').set_index(['model','scoring'])
published=pd.read_csv(INPUT/'published_primary_gains.csv',float_precision='round_trip').set_index('model')
ans=pd.read_parquet(DATA/'06_triviaqa_expert/assistant_answers_heldout.parquet')
expert=pd.read_parquet(DATA/'06_triviaqa_expert/expert_answers_heldout.parquet').sort_values('item_key')
dec=pd.read_parquet(DATA/'06_triviaqa_expert/consultation_choices_heldout.parquet')
parameters=pd.read_csv(DATA/'06_triviaqa_expert/confidence_calibration.csv').set_index('model')
levels=[0,10,20,45,80,150];conditions=list(itertools.product(['explicit_cost','server_state','battery_cost'],levels))
wait=np.array([0 if k=='battery_cost'else l/100 for k,l in conditions])
battery=np.array([l/100 if k=='battery_cost'else 0 for k,l in conditions])
threshold=np.round(.64-wait+battery,9)
scorecols={'primary':'correct','full_alias':'correct_full_alias','negation':'correct_negation','short_boundary':'correct_short_boundary'}
records=[];constant=[];fits=[];invariants=[]
rawexpert=tables['expert__qwen3-32b-awq']
assert np.array_equal(rawexpert.item_key.to_numpy(),expert.item_key.to_numpy())
assert np.array_equal(rawexpert.correct.to_numpy(),expert.expert_correct.to_numpy())
for model,aq in ans.groupby('model',sort=True):
    aq=aq.sort_values('item_key');keys=aq.item_key.to_numpy();n=len(keys)
    raw=tables['answer__'+model.replace('.', 'p')]
    assert np.array_equal(raw.item_key.to_numpy(),keys)
    assert np.array_equal(raw.correct.to_numpy(),aq.independent_correct.to_numpy())
    assert np.allclose(raw.c.to_numpy(float),aq.confidence_raw.to_numpy(float),atol=1e-12,rtol=0)
    dd=dec[dec.model==model].copy();dd['cond']=list(zip(dd.presentation.astype(str),dd.level))
    dd=dd.set_index(['item_key','cond']).outcome.unstack('cond').reindex(index=keys,columns=conditions)
    assert dd.notna().all().all()
    off=dd.to_numpy()=='offload';fail=dd.to_numpy()=='format_failure'
    primary_z=aq.confidence_calibrated.to_numpy(float)
    for scoring,col in scorecols.items():
        variant='primary'if scoring=='primary'else'score_'+scoring
        fit=frozen_fits.loc[(model,scoring)]
        a,b=float(fit['intercept']),float(fit['slope'])
        zinfo={'valid':bool(fit['original_gate_valid'])}
        z=expit(a+b*logit(np.clip(raw.c.to_numpy(float),1e-4,1-1e-4)))
        if scoring=='primary': assert np.max(np.abs(z-primary_z))<1e-10
        s=raw[col].to_numpy(float);e=rawexpert[col].to_numpy(float)
        assert np.isfinite(s).all()and np.isfinite(e).all()
        own=np.where(off,e[:,None]-wait,np.where(fail,0,s[:,None]-battery))
        ask=z[:,None]<threshold
        route=np.where(ask,e[:,None]-wait,s[:,None]-battery)
        gain=interval((route-own).mean(1))
        rec=dict(model=model,scoring=scoring,original_calibration_gate_valid=zinfo['valid'],**gain)
        records.append(rec);fits.append(dict(model=model,scoring=scoring,intercept=a,slope=b,original_gate_valid=zinfo['valid']))
        if scoring=='primary':
            # Reproduce the Figure 6 estimates before the additional comparisons.
            point=float(published.loc[model,'value'])
            assert abs(float(published.loc[model,'lo'])-gain['lo'])<1e-12
            assert abs(float(published.loc[model,'hi'])-gain['hi'])<1e-12
            assert abs(point-gain['value'])<1e-12
            c=float(parameters.loc[model,'development_accuracy'])
            askconstant=np.broadcast_to(c<threshold,ask.shape)
            costonly=np.where(askconstant,e[:,None]-wait,s[:,None]-battery)
            improvement=interval((costonly-own).mean(1));increment=interval((route-costonly).mean(1))
            same=np.array_equal(ask,askconstant)
            constant.append(dict(model=model,development_accuracy=c,same_decisions=same,
                cost_only_gain=improvement['value'],cost_only_lo=improvement['lo'],cost_only_hi=improvement['hi'],
                item_confidence_increment=increment['value'],increment_lo=increment['lo'],increment_hi=increment['hi'],n=n))
            invariants.append(dict(model=model,primary_point_matches_published=True,max_z_difference=float(np.max(np.abs(z-primary_z)))))
pd.DataFrame(records).to_csv(WORK/'triviaqa_scoring_sensitivity.csv',index=False,lineterminator='\n')
pd.DataFrame(constant).to_csv(WORK/'triviaqa_cost_only.csv',index=False,lineterminator='\n')
pd.DataFrame(fits).to_csv(WORK/'triviaqa_sensitivity_fits.csv',index=False,lineterminator='\n')
assert next(x for x in constant if x['model']=='gemma-4-e4b-it')['same_decisions']
summary={'cost':{'n_contrasts':len(cost),'directions':dict(Counter(x['direction']for x in cost)),
    'within_five_pp':sum(x['within_five_pp']for x in cost),'positive_and_within_five_pp':sum(x['direction']=='increase'and x['within_five_pp']for x in cost),
    'models_increase_or_small_both':sum(all(x['direction']=='increase'or x['within_five_pp']for x in cost if x['model']==m)for m in sorted(conf.model.unique()))},
    'calibration':calib,'triviaqa':records,'cost_only':constant,'primary_invariants':invariants,
    'conventions':{'bootstrap_resamples':R,'seed':SEED,'interval':'95% percentile, paired questions; 18 conditions averaged within question',
        'cost_only':'Question-level confidence replaced by development-set answer accuracy. Same action within condition; no held-out labels used for policy selection.',
        'sensitivity':'Use the archived development fit for each scoring rule. Keep original stated accuracy 0.64, candidates, costs, choices and all eight displayed models.'}}
(WORK/'analysis_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'cost':summary['cost'],'calibration_ece_range':[min(x['ece']for x in calib),max(x['ece']for x in calib)],
    'sensitivity_by_scoring':{name:{'gain_min':min(x['value']for x in records if x['scoring']==name),'gain_max':max(x['value']for x in records if x['scoring']==name),'positive_lower_bounds':sum(x['lo']>0 for x in records if x['scoring']==name)}for name in scorecols},
    'cost_only':constant},ensure_ascii=False,indent=2))
