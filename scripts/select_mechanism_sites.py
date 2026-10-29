#!/usr/bin/env python3
"""Computed, question-paired C3 site ranking/validation; never opens test results.

Development fills two fixed slots from 128 questions: one pre-question site
(cost_number/rule_block), one decision site at relative depth <=0.90.
Last-decile decision sites are direct-output diagnostics, never C4 candidates.
Validation confirms those sites on 256 separate questions. File-bound evidence is recomputed by the C4 caller;
hand-written passed flags cannot activate the branch. Missing A3 stays unresolved.

Two-stage usage (all outputs must be new paths under NMI_OUTPUT_ROOT):
  python -B scripts/select_mechanism_sites.py --dev-runs DEV_RUN --dev-only --out DEV_SELECTION
  # Run validation only at the two sites in DEV_SELECTION/sites.json.
  python -B scripts/select_mechanism_sites.py --dev-runs DEV_RUN --val-runs VAL_RUN --out VALIDATED_SELECTION
The second command recomputes the identical development-only winners and never
uses validation to replace a failed slot. Last-decile decision diagnostics may
be retained in the source runs but cannot fill either selection slot. Missing
architecture metadata or a missing slot fails closed. Production layer count
comes from model_sources/path/config.json whose hash must match the manifest;
toy manifests require an explicit internal synthetic_test_only argument and cannot produce a production C4 gate.
"""
from __future__ import annotations
import os
import argparse
import hashlib
import json
import tarfile
import torch
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nmi.mechanism.analysis import (normalize_row,paired_bootstrap,paired_signflip,holm_adjust,freeze_sites)
from nmi.mechanism.prompts import pair_conditions,conditions
from nmi.mechanism.data import DATA_ROOT,load_split
from nmi.mechanism.runtime import score_logits

FAMILIES=('first_cost','positive_cost','semantic')
ARMS=('swap','identity','random')
DIRECTIONS=('forward','reverse')
MODELS=('qwen3-4b','smollm3-3b','gemma-4-e4b-it')
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def architecture_layers(manifest):
    """Verify decoder depth from the recorded model config, never sampled max(layer)."""
    sources=manifest.get('model_sources',{})
    if sources.get('model_type')=='synthetic' and not sources.get('path'):
        n=sources.get('num_hidden_layers')
        if not isinstance(n,int) or isinstance(n,bool) or n<2:raise ValueError('toy manifest requires explicit num_hidden_layers >=2')
        return n,{'source':'explicit synthetic manifest','num_hidden_layers':n},{}
    base=Path(sources.get('path',''));config=base/'config.json'
    expected=sources.get('files',{}).get('config.json',{}).get('sha256')
    if not expected or not config.is_file() or sha(config)!=expected:
        raise ValueError('num_hidden_layers requires hash-verified model_sources config.json')
    cfg=json.loads(config.read_text())
    if cfg.get('model_type')!=sources.get('model_type'):
        raise ValueError('recorded/config model architecture mismatch')
    text=cfg.get('text_config',cfg);n=text.get('num_hidden_layers')
    if not isinstance(n,int) or isinstance(n,bool) or n<2:raise ValueError('verified text decoder requires num_hidden_layers >=2')
    return n,{'source':str(config.resolve()),'config_sha256':expected,'model_type':cfg['model_type'],'num_hidden_layers':n},{str(config.resolve()):expected}


def verified_provenance(directory, manifest):
    identity=manifest['identity']
    if identity.get('schema_version')!=2 or not identity.get('sources'):
        raise ValueError('production C3 requires schema_version=2 and bound sources')
    archive=Path(str(directory)+'.sources.tar.gz');bindings={}
    tar=tarfile.open(archive,'r:gz') if archive.is_file() else None
    try:
        members={m.name:m for m in tar.getmembers() if m.isfile()} if tar else {}
        if tar:bindings[str(archive)]=sha(archive)
        for name,expected in identity['sources'].items():
            path=Path(name)
            rel=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else None
            if rel in members:
                actual=hashlib.sha256(tar.extractfile(members[rel]).read()).hexdigest()
            elif path.is_file():actual=sha(path);bindings[str(path)]=actual
            else:raise ValueError('C3 source unavailable: '+name)
            if actual!=expected:raise ValueError('C3 execution source hash mismatch: '+name)
    finally:
        if tar:tar.close()
    return bindings


def compare_score(record, actual, *, context):
    for key,value in actual.items():
        if key not in record:raise ValueError(context+' missing score '+key)
        other=record[key]
        if isinstance(value,(float,int)) and not isinstance(value,bool):
            same=bool(np.isclose(value,other,rtol=1e-7,atol=1e-8))
        elif isinstance(value,list):same=bool(np.allclose(value,other,rtol=1e-7,atol=1e-8))
        else:same=value==other
        if not same:raise ValueError(context+' raw logits score mismatch: '+key)


class ArtifactVerifier:
    """Retain verified small baseline metadata, never repeated full activation tensors."""
    def __init__(self,directory,bindings):
        self.directory=directory;self.bindings=bindings;self.baselines={}
    def load(self,row):
        path=(self.directory/row['artifact']).resolve()
        if not path.is_relative_to(self.directory) or not path.is_file() or sha(path)!=row['artifact_sha256']:
            raise ValueError('C3 raw artifact hash/path mismatch')
        self.bindings[str(path)]=row['artifact_sha256']
        return torch.load(path,map_location='cpu',weights_only=True)
    def baseline(self,row):
        cachekey=(row['row_id'],row.get('artifact_sha256'))
        if cachekey not in self.baselines:
            art=self.load(row)
            for key in ('row_id','item_key','fold','model','condition_id','correct_answer','mapping_id'):
                if key=='fold':
                    actual='val' if art['metadata'].get(key)=='validation' else art['metadata'].get(key)
                else:actual=art['metadata'].get(key)
                if actual!=row.get(key):raise ValueError('C3 baseline artifact metadata mismatch')
            if hashlib.sha256(art['prompt'].encode()).hexdigest()!=row['prompt_sha256'] or art['prompt_sha256']!=row['prompt_sha256']:
                raise ValueError('C3 baseline prompt hash mismatch')
            result=score_logits(art['logits'],art['label_ids'],row['correct_answer'])
            original=row.get('_raw_score_record')
            if original is not None:
                compare_score(original,result,context='baseline')
            elif not isinstance(row.get('action'),str):
                # Public normalized-row entry: compare the same documented
                # representation, including unknown action and FP32 mass clamp.
                normalized=normalize_row({**result,'correct_answer':row['correct_answer']})
                compare_score(row,{key:normalized.get(key,value) for key,value in result.items()},context='normalized baseline')
                compare_score(row,{key:normalized[key] for key in ('raw_mass','mass_adjusted')},context='normalized baseline provenance')
            else:compare_score(row,result,context='baseline')
            self.baselines[cachekey]={'label_ids':art['label_ids'],'positions':art['positions'],'input_ids':art['input_ids']}
        return self.baselines[cachekey]
    def intervention(self,patch,base,donor):
        b=self.baseline(base);d=self.baseline(donor);art=self.load(patch)
        from nmi.mechanism.runtime import digest
        if patch.get('intervention_id')!=digest(art['intervention'])[:24]:raise ValueError('intervention identity digest mismatch')
        for key,value in art['intervention'].items():
            if patch.get(key)!=value:raise ValueError('C3 intervention artifact metadata mismatch')
        for key in ('label_ids','positions','input_ids'):
            if art[key]!=b[key]:raise ValueError('C3 recipient input/label alignment mismatch')
        for key in ('matched_batch_sha256','matched_batch_size','matched_recipient_row'):
            if art[key]!=patch[key]:raise ValueError('C3 matched batch metadata mismatch')
        if art['matched_batch_size']%2 or not 0<=art['matched_recipient_row']<art['matched_batch_size']//2:
            raise ValueError('invalid matched 2B recipient row')
        result=score_logits(art['logits'],b['label_ids'],base['correct_answer'])
        matched=score_logits(art['matched_baseline_logits'],b['label_ids'])
        donor_score=score_logits(art['matched_donor_logits'],d['label_ids'])
        compare_score(patch,result,context='intervention')
        compare_score(patch['matched_baseline'],matched,context='matched baseline')
        compare_score(patch['matched_donor'],donor_score,context='matched donor')
        compare_score(art['matched_baseline'],matched,context='artifact matched baseline')
        compare_score(art['matched_donor'],donor_score,context='artifact matched donor')
        if patch['arm']=='identity' and not torch.equal(art['logits'],art['matched_baseline_logits']):
            raise ValueError('identity must preserve complete raw logits exactly')
        # Statistics consume verified same-2B raw readouts, never archival logits.
        return {**patch,**result,'matched_baseline':matched,'matched_donor':donor_score}


def read_run(directory,*,role,expected_n,synthetic_test_only=False):
    p=Path(directory).resolve();manifest=json.loads((p/'manifest.json').read_text())
    settings=manifest['identity']['settings'];fold=settings['fold']
    permitted=('train','dev') if role=='development' else ('validation','val')
    if fold not in permitted:raise PermissionError('Selection accepts development/validation only; pilot/test outcomes forbidden')
    if manifest.get('status')!='complete':raise ValueError('Selection requires completed run')
    sources={str(p/'manifest.json'):sha(p/'manifest.json')}
    synthetic=manifest.get('model_sources',{}).get('model_type')=='synthetic'
    if synthetic and not synthetic_test_only:raise PermissionError('synthetic fixtures require explicit synthetic_test_only isolation')
    if synthetic_test_only and not synthetic:raise PermissionError('synthetic test bypass cannot load production runs')
    verifier=None;expected_items={}
    if not synthetic:
        if settings['model'] not in MODELS:raise ValueError('unknown production model')
        sources.update(verified_provenance(p,manifest))
        split_root=Path(settings.get('split_root') or DATA_ROOT).resolve()
        canonical=load_split(fold,root=split_root)
        selected=canonical[:settings['max_items']]
        from nmi.mechanism.runtime import digest
        if manifest['identity'].get('items_sha256')!=digest([str(i.item_key) for i in selected]):raise ValueError('frozen item keys mismatch')
        expected_items={str(i.item_key):i.gt_position-1 for i in selected}
        expected_patched=[str(i.item_key) for i in selected[:settings['probe_items']]]
        if len(expected_patched)!=expected_n:raise ValueError('planned patched question count mismatch')
        sources[str(split_root/'manifest.json')]=sha(split_root/'manifest.json')
        sources[str(split_root/(fold+'.jsonl'))]=sha(split_root/(fold+'.jsonl'))
        verifier=ArtifactVerifier(p,sources)
    num_layers,architecture,architecture_sources=architecture_layers(manifest);sources.update(architecture_sources)
    rows={}
    for line in (p/'rows.jsonl').read_text().splitlines():
        original=json.loads(line)
        row=normalize_row(original)
        row['_raw_score_record']=original
        if row['fold']!=('val' if fold=='validation' else fold):raise PermissionError('Wrong fold in source rows')
        if not synthetic and (row['model']!=settings['model'] or row['domain']!=settings['domain'] or row['question_id'] not in expected_items or row['correct_answer']!=expected_items[row['question_id']]):raise ValueError('baseline frozen identity/ground truth mismatch')
        if row['row_id'] in rows:raise ValueError('duplicate baseline row_id')
        rows[row['row_id']]=row
    if not synthetic:
        planned_conditions={c.condition_id:c for c in conditions(settings['domain'])}
        expected_cells={(q,c) for q in expected_items for c in planned_conditions}
        actual_cells=[(r['question_id'],r['condition_id']) for r in rows.values()]
        if len(actual_cells)!=len(expected_cells) or set(actual_cells)!=expected_cells:raise ValueError('incomplete frozen 36-condition baseline grid')
        if any(r['mapping_id']!=planned_conditions[r['condition_id']].mapping_id for r in rows.values()):raise ValueError('frozen condition mapping mismatch')
    patches=[];keys=set();sites=set();index={}
    expected_pairs={(r['family'],r['mapping_id']):r for r in pair_conditions(settings['domain'])}
    for line in (p/'swaps.jsonl').read_text().splitlines():
        patch=json.loads(line);base=rows[patch['recipient_row_id']];donor=rows[patch['donor_row_id']]
        if base['question_id']!=donor['question_id'] or base['mapping_id']!=donor['mapping_id']:raise ValueError('donor pairing mismatch')
        if patch['contrast_family'] not in FAMILIES or patch['arm'] not in ARMS or patch['direction'] not in DIRECTIONS:raise ValueError('unexpected intervention design')
        planned=expected_pairs[patch['contrast_family'],int(base['mapping_id'])]
        pair=(planned['base_condition_id'],planned['source_condition_id'])
        if patch['direction']=='reverse':pair=pair[::-1]
        if (base['condition_id'],donor['condition_id'])!=pair:raise ValueError('recorded donor/recipient conditions do not match frozen contrast')
        if not patch.get('matched_batch_sha256') or not isinstance(patch.get('matched_baseline'),dict):raise ValueError('matched reference required')
        if verifier:patch=verifier.intervention(patch,base,donor)
        site=(int(patch['layer']),patch['position'])
        if not 0<=site[0]<num_layers:raise ValueError('intervention layer outside verified architecture')
        if site[1] not in ('cost_number','rule_block','decision'):raise ValueError('unknown semantic position')
        sites.add(site);keys.add(base['question_id'])
        key=(site,patch['contrast_family'],patch['direction'],base['question_id'],int(base['mapping_id']),patch['arm'])
        if key in index:raise ValueError('duplicate site/condition/arm')
        index[key]=(base,patch);patches.append((base,patch))
    if not synthetic and keys!=set(expected_patched):raise ValueError('patched keys differ from frozen first-N questions')
    if len(keys)!=expected_n:raise ValueError(f'{role} requires exactly {expected_n} distinct patched questions, found {len(keys)}')
    for site in sites:
        for family in FAMILIES:
            for direction in DIRECTIONS:
                for q in keys:
                    for mapping in (0,1):
                        for arm in ARMS:
                            if (site,family,direction,q,mapping,arm) not in index:raise ValueError('Incomplete C3 factorial cells; no selective intersection allowed')
    for name in ('rows.jsonl','swaps.jsonl','equivalence.json'):
        file=p/name
        if file.is_file():sources[str(file)]=sha(file)
    eq=json.loads((p/'equivalence.json').read_text()) if (p/'equivalence.json').exists() else {}
    checks=eq.get('checks',[])
    eq_pass=bool(checks) and all(c.get('max_abs_logits_difference')==0 and c.get('same_action') is True for c in checks)
    if verifier:
        eq_cache={}
        for check in checks:
            key=(check['artifact'],check['artifact_sha256'])
            if key not in eq_cache:eq_cache[key]=verifier.load(check)
            art=eq_cache[key];name={'same_batch_identity':'batch_identity','same_batch_collect':'batch_collect'}.get(check['comparison'])
            if name is None:raise ValueError('unexpected equivalence comparison')
            before=art['batch_plain'];after=art[name]
            if not torch.equal(before['logits'],after['logits']):raise ValueError('raw backend equivalence logits differ')
            if before['input_ids']!=after['input_ids'] or before['label_ids']!=after['label_ids']:raise ValueError('backend equivalence inputs differ')
        if not eq_pass:raise ValueError('production C3 requires verified backend equivalence')
    return dict(path=str(p),model=settings['model'],domain=settings['domain'],fold=fold,index=index,sites=sites,questions=sorted(keys),patches=patches,sources=sources,equivalence_pass=eq_pass,num_hidden_layers=num_layers,architecture=architecture)


def direction_vector(run,site,family,direction):
    values=[]
    for q in run['questions']:
        permap=[]
        for mapping in (0,1):
            _,sw=run['index'][site,family,direction,q,mapping,'swap']
            _,ra=run['index'][site,family,direction,q,mapping,'random']
            permap.append((sw['m']-sw['matched_baseline']['m'])-(ra['m']-ra['matched_baseline']['m']))
        values.append(float(np.mean(permap)))
    return np.asarray(values)


def ranking(run):
    candidates=[]
    for site in sorted(run['sites']):
        families={}
        for family in FAMILIES:
            t=(direction_vector(run,site,family,'forward')-direction_vector(run,site,family,'reverse'))/2
            families[family]=dict(mean_T=float(t.mean()),n_questions=len(t))
        score=float(np.mean([abs(v['mean_T']) for v in families.values()]))
        depth=site[0]/(run['num_hidden_layers']-1)
        slot='pre_question' if site[1] in ('cost_number','rule_block') else ('decision_not_last_decile' if depth<=.90 else 'direct_output_diagnostic')
        candidates.append(dict(model=run['model'],layer=site[0],position=site[1],relative_depth=depth,selection_slot=slot,c4_candidate=slot!='direct_output_diagnostic',contrast='mean_abs_T_three_families',score=score,fold='train' if run['fold']=='train' else 'dev',families=families))
    return sorted(candidates,key=lambda r:(-r['score'],r['layer'],r['position']))


def select_development(dev_runs,*,n_dev=128,synthetic_test_only=False,_loaded_runs=None):
    """Executable phase 1: freeze one winner in each slot before validation runs."""
    dev=_loaded_runs if _loaded_runs is not None else [read_run(p,role='development',expected_n=n_dev,synthetic_test_only=synthetic_test_only) for p in dev_runs]
    if not dev or len({x['model'] for x in dev})!=len(dev):raise ValueError('one nonempty development run per model required')
    sources={};ranks=[];winners=[]
    for d in dev:
        candidates=ranking(d);chosen=[]
        for slot in ('pre_question','decision_not_last_decile'):
            eligible=[r for r in candidates if r['selection_slot']==slot and np.isfinite(r['score'])]
            if not eligible:raise ValueError('Fail closed: no available candidate for required slot '+slot)
            chosen.append(eligible[0])
        winners.extend(chosen);ranks.append(dict(model=d['model'],architecture=d['architecture'],candidates=candidates,slot_winners=chosen));sources.update(d['sources'])
    sites=freeze_sites(winners,max_sites=2,provenance={
        'input_files':sources,
        'rule':'one pre-question cost_number/rule_block winner plus one decision winner at layer/(num_hidden_layers-1)<=0.90; each slot ranks mean_abs_T_three_families; ties layer ascending then position lexicographic',
        'revision_scope':'Revised after engineering pilots, before any development results; not a preregistration claim.',
        'direct_output_diagnostic':'Last 10% decision layers excluded from C4 candidate selection; retained as diagnostics in development ranking'})
    return dict(schema='nmi.mechanism.slot_selection/2',development_ranking=ranks,sites=sites,computed_from=sources,
                recipe={'dev_runs':[r['path'] for r in dev],'n_dev':n_dev})


def site_validation(run,site,dev_record,*,n_boot=2000):
    q=run['questions'];families={};pvalues=[]
    for family in FAMILIES:
        directions={}
        for direction in DIRECTIONS:
            vector=direction_vector(run,site,family,direction)
            stat=paired_bootstrap(q,vector,n_boot=n_boot);p=paired_signflip(q,vector,n_resamples=n_boot)
            directions[direction]=dict(**stat,p_value=p['p_value']);pvalues.append(p['p_value'])
        t=(direction_vector(run,site,family,'forward')-direction_vector(run,site,family,'reverse'))/2
        families[family]=dict(directions=directions,T=paired_bootstrap(q,t,n_boot=n_boot),dev_mean_T=dev_record['families'][family]['mean_T'])
    selected=[(base,patch) for base,patch in run['patches'] if (patch['layer'],patch['position'])==site]
    identity=[p for _,p in selected if p['arm']=='identity']
    identity_ok=bool(identity) and all(p['m']==p['matched_baseline']['m'] and p['legal_mass']==p['matched_baseline']['legal_mass'] for p in identity)
    swap=[(base,patch) for base,patch in selected if patch['arm']=='swap']
    qids=[];accuracy=[];quality_delta=[];before_quality=[];after_quality=[]
    for base,patch in swap:
        baseline=patch['matched_baseline'];correct=int(base['correct_answer'])
        qids.append(base['question_id'])
        accuracy.append(int(np.argmax(patch['semantic_logits'][:4])==correct)-int(np.argmax(baseline['semantic_logits'][:4])==correct))
        before_quality.append(int(baseline['legal_mass']>=.5));after_quality.append(int(patch['legal_mass']>=.5))
        quality_delta.append(after_quality[-1]-before_quality[-1])
    ac=paired_bootstrap(qids,accuracy,n_boot=n_boot);qd=paired_bootstrap(qids,quality_delta,n_boot=n_boot)
    return dict(model=run['model'],layer=site[0],position=site[1],families=families,
                identity_zero=identity_ok,n_identity=len(identity),
                baseline_quality=float(np.mean(before_quality)),intervention_quality=float(np.mean(after_quality)),
                quality_delta=qd,forced_accuracy_delta=ac,
                quality_pass=bool(min(np.mean(before_quality),np.mean(after_quality))>=.95 and qd['ci'][0]>=-.05),
                accuracy_noninferiority=ac['ci'][0]>=-.05),pvalues


def build_report(dev_runs,val_runs,*,n_dev=128,n_val=256,n_boot=2000,rule_control=None,synthetic_test_only=False):
    if len(dev_runs)!=len(val_runs) or not dev_runs:raise ValueError('matching nonempty dev/validation run lists required')
    dev=[read_run(p,role='development',expected_n=n_dev,synthetic_test_only=synthetic_test_only) for p in dev_runs]
    val=[read_run(p,role='validation',expected_n=n_val,synthetic_test_only=synthetic_test_only) for p in val_runs]
    if not synthetic_test_only and (len(dev)!=3 or {r['model'] for r in dev}!=set(MODELS)):
        raise ValueError('formal C3 requires all three fixed models and 36 Holm tests')
    if not synthetic_test_only and (n_dev,n_val,n_boot)!=(128,256,2000):raise ValueError('formal C3 freezes train128/validation256/bootstrap2000')
    if len({x['model'] for x in dev})!=len(dev):raise ValueError('one run per model required')
    val_index={x['model']:x for x in val}
    if set(val_index)!={x['model'] for x in dev}:raise ValueError('development/validation model identity mismatch')
    selection=select_development(dev_runs,n_dev=n_dev,synthetic_test_only=synthetic_test_only,_loaded_runs=dev)
    all_sources=dict(selection['computed_from']);dev_ranks=selection['development_ranking'];sites=selection['sites']
    for d in dev:
        v=val_index[d['model']]
        if set(d['questions'])&set(v['questions']):raise ValueError('question leakage between selection and validation')
        if d['domain']!=v['domain'] or d['num_hidden_layers']!=v['num_hidden_layers']:raise ValueError('development/validation domain or architecture mismatch')
        all_sources.update(v['sources'])
    if not synthetic_test_only:
        for model,v in val_index.items():
            frozen={(r['layer'],r['position']) for r in sites['selected_sites'] if r['model']==model}
            if len(frozen)!=2 or v['sites']!=frozen:raise ValueError('validation must use exactly the two development-frozen slots')
    validations=[];pvalues=[]
    for selected in sites['selected_sites']:
        v=val_index[selected['model']];site=(selected['layer'],selected['position'])
        if site not in v['sites']:raise ValueError('selected development site absent from validation')
        metrics,p=site_validation(v,site,selected,n_boot=n_boot);validations.append(metrics);pvalues.extend(p)
    if not synthetic_test_only and len(pvalues)!=36:raise ValueError('formal Holm family must contain exactly 36 tests')
    adjusted=iter(holm_adjust(pvalues))
    for site in validations:
        supported=[]
        for family in FAMILIES:
            f=site['families'][family];sign=np.sign(f['dev_mean_T']);ok=sign!=0
            for direction in DIRECTIONS:
                metric=f['directions'][direction];metric['holm_p']=next(adjusted)
                target_sign=sign if direction=='forward' else -sign
                metric['directional_replication']=metric['estimate']*target_sign>0 and (metric['ci'][0]>0 or metric['ci'][1]<0) and metric['holm_p']<=.05
                ok=ok and metric['directional_replication']
            f['bidirectional_replicated']=bool(ok)
            if ok:supported.append(family)
        site['replicated_families']=supported;site['c3_effect_replicated']=bool(supported)
    # A3 optional measured paired cases, never a caller-supplied passed flag.
    rule=dict(status='unresolved',reason='No independent measured A3 intervention output supplied')
    if rule_control:
        path=Path(rule_control).resolve();rc=json.loads(path.read_text());all_sources[str(path)]=sha(path)
        cases=rc.get('paired_cases',[])
        if rc.get('schema')!='nmi.mechanism.rule_control/1' or not cases:raise ValueError('A3 requires measured paired_cases schema')
        if any(c.get('fold') not in ('train','dev','validation','val') for c in cases):raise PermissionError('A3 rule control cannot use pilot/test for confirmatory gate')
        # Fixed factorial A3 cases are descriptive; independent generated scenario
        # IDs permit question-cluster uncertainty, never cell-count pseudo-n.
        if rc.get('independent_scenarios') is not True:
            rule=dict(status='unresolved',reason='Fixed 96-cell A3 grid lacks independent scenario units for the requested CI')
        else:
            q=[c['scenario_key'] for c in cases];diff=[]
            for c in cases:
                if c.get('baseline_optimal') not in (0,1) or c.get('intervention_optimal') not in (0,1):raise ValueError('A3 measured optimality must be binary')
                if not c.get('raw_artifact_sha256'):raise ValueError('A3 requires bound raw artifact hashes')
                diff.append(c['intervention_optimal']-c['baseline_optimal'])
            result=paired_bootstrap(q,diff,n_boot=n_boot)
            rule=dict(status='pass' if result['ci'][0]>=-.05 else 'fail',delta=result)
    for p in (Path(__file__),Path(__file__).parents[1]/'nmi/mechanism/analysis.py'):
        all_sources[str(p.resolve())]=sha(p)
    # Gate by model; never let strong results on one model pass a different one.
    gates={}
    for d in dev:
        model=d['model'];vv=[r for r in validations if r['model']==model];v=val_index[model]
        gate=dict(backend_equivalence=d['equivalence_pass'] and v['equivalence_pass'],
                  c3_localized=all(r['c3_effect_replicated'] for r in vv),
                  bidirectional_effect=all(r['c3_effect_replicated'] for r in vv),
                  sham_control_pass=all(r['identity_zero'] for r in vv),
                  readout_quality=all(r['quality_pass'] for r in vv),
                  answer_noninferiority=all(r['accuracy_noninferiority'] for r in vv),
                  validation_generalization=all(r['c3_effect_replicated'] for r in vv),sites_frozen=True)
        if rule['status']!='unresolved':gate['rule_control_noninferiority']=rule['status']=='pass'
        gates[model]=gate
    return dict(schema='nmi.mechanism.computed_c3_metrics/1',recipe=dict(dev_runs=[r['path'] for r in dev],val_runs=[r['path'] for r in val],n_dev=n_dev,n_val=n_val,n_boot=n_boot,rule_control=str(rule_control) if rule_control else None,synthetic_test_only=synthetic_test_only),
                computed_from=all_sources,development_ranking=dev_ranks,sites=sites,validation=validations,
                holm_family_size=len(pvalues),holm_scope='All supplied models x two selected sites x three families x two directions',
                thresholds={'minimum_readout_fraction':.95,'max_accuracy_drop':.05,'max_readout_drop':.05,'validation_alpha':.05},
                rule_control=rule,computed_gate_checks=gates)


def write_report(report,out):
    out=Path(out).resolve()
    if not out.is_relative_to(os.environ.get('NMI_OUTPUT_ROOT', '/')):raise ValueError('outputs must be under NMI_OUTPUT_ROOT')
    out.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        p=out/name
        with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
        return p
    sitepath=write('sites.json',report['sites']);metricpath=write('metrics.json',report)
    gates={}
    for model,checks in report['computed_gate_checks'].items():
        evidence={key:dict(passed=value,evidence_ref=str(metricpath),evidence_sha256=sha(metricpath),metric_key=f'computed_gate_checks.{model}.{key}') for key,value in checks.items()}
        if 'rule_control_noninferiority' not in checks:evidence['rule_control_noninferiority']=dict(status='unresolved',evidence_ref=str(metricpath),evidence_sha256=sha(metricpath),reason=report['rule_control']['reason'])
        payload=dict(schema='nmi.mechanism.computed_c3_gate/1',model=model,metrics_path=str(metricpath),metrics_sha256=sha(metricpath),sites_path=str(sitepath),sites_sha256=sha(sitepath),computed_from=report['computed_from'],**evidence)
        gates[model]=str(write('gate_'+model+'.json',payload))
    return dict(metrics=str(metricpath),sites=str(sitepath),gates=gates)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dev-runs',nargs='+',required=True);p.add_argument('--val-runs',nargs='+');p.add_argument('--out',required=True);p.add_argument('--dev-only',action='store_true')
    p.add_argument('--n-dev',type=int,default=128);p.add_argument('--n-val',type=int,default=256);p.add_argument('--n-boot',type=int,default=2000);p.add_argument('--rule-control')
    a=p.parse_args(argv)
    if a.dev_only:
        if a.val_runs:p.error('--dev-only must not read validation outcomes')
        report=select_development(a.dev_runs,n_dev=a.n_dev);out=Path(a.out).resolve()
        if not out.is_relative_to(os.environ.get('NMI_OUTPUT_ROOT', '/')):raise ValueError('outputs must be under NMI_OUTPUT_ROOT')
        out.mkdir(parents=True,exist_ok=True)
        for name,value in [('sites.json',report['sites']),('development_selection.json',report)]:
            with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({'sites':str(out/'sites.json'),'selection':str(out/'development_selection.json')}))
    else:
        if not a.val_runs:p.error('validation phase requires --val-runs, or use --dev-only first')
        report=build_report(a.dev_runs,a.val_runs,n_dev=a.n_dev,n_val=a.n_val,n_boot=a.n_boot,rule_control=a.rule_control)
        print(json.dumps(write_report(report,a.out)))
    return 0


if __name__=='__main__':raise SystemExit(main())
