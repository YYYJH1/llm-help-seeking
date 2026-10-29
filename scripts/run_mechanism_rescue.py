#!/usr/bin/env python3
"""Separate transparent D rescues, with matched native/variant prompt batches.

No hidden-state training or model load at import. Pilot defaults to eight
questions; validation is 256 and independent test is the frozen 512, both with
all 36 conditions. Existing F3 *condition-block* wording is sourced explicitly;
the full current labelled prompt is not claimed to reproduce historical F3.
"""
from __future__ import annotations
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nmi.mechanism.analysis import normalize_row,intervention_decomposition,paired_bootstrap,quality_bounds
from nmi.mechanism.runtime import (load_runtime,inspect_local_model,sha256_file,digest,data_path,
    append_json,atomic_json,save_artifact,score_logits,validate_frozen_execution)

ROOT=Path(__file__).resolve().parents[1]
ARMS=('external_net_score','length_matched_irrelevant','f3_score_block')


def insert_before_decision(record,addition):
    start,end=record.spans['decision']
    if end!=len(record.text) or record.text[start:end]!='Answer:':
        raise ValueError('Original final decision span must identify trailing Answer:')
    text=record.text[:start]+addition+record.text[start:]
    spans=dict(record.spans)
    spans['decision']=(start+len(addition),end+len(addition))
    spans['rescue_addition']=(start,start+len(addition))
    return dict(text=text,spans=spans,option_labels=list(record.option_labels),help_label=record.help_label)


def net_score_addition(row):
    if row.get('z') is None or not np.isfinite(row['z']):raise ValueError('Independent externally calibrated z required')
    return (f'\nSeparate diagnostic score table using an external estimate:\n'
        f'- Answer yourself: expected score {row["z"]:.8f}.\n'
        f'- Ask the helper: expected net score {row["a_H"]:.2f} - {row["c0"]:.2f} - '
        f'{row["w"]:.2f} = {row["V_H"]:.2f}.\n'
        'Choose the existing action label with the larger expected score. If equal, either action is optimal.\n')


def f3_addition(row):
    from nmi.mech_prompts import MechPromptBook
    from nmi.prompts import display_a_H_pct
    book=MechPromptBook.load(ROOT/'configs/prompts/mech_v1.yaml')
    template=book.blocks['F3']
    if '0.80' not in template or '80%' not in template or '0.00' not in template:
        raise ValueError('Original F3 numerical literals changed; audit adaptation before execution')
    generalized=template.replace('0.80',f'{row["a_H"]:.2f}').replace('80%',f'{display_a_H_pct(row["a_H"])}%').replace('0.00',f'{row["c0"]:.2f}')
    value=Decimal(str(row['a_H']))-Decimal(str(row['c0']))-Decimal(str(row['w']))
    block=generalized.format(x=row['w'],ev=value)
    registered=float(row['a_H'])==.8 and float(row['c0'])==0 and float(row['w']) in (0,.1,.2,.45,.8,1.5)
    if registered and block!=book.conditions_block('F3',float(row['w'])):
        raise ValueError('Registered F3 condition block not reproduced byte-for-byte')
    return '\n'+block+'\n',dict(source='configs/prompts/mech_v1.yaml condition_blocks.F3',
        exact_registered_condition_block=registered,
        limitation='F3 block appended to current prompt; full historical F3 layout/rules/labels not reproduced; other numerical values are explicitly adapted')


def token_count(tokenizer,runtime,text):
    from nmi.prompts import render_chat
    rendered=render_chat(tokenizer,text,runtime.chat_kwargs,assistant_prefix=runtime.assistant_answer_prefix)
    return len(tokenizer.encode(rendered,add_special_tokens=False))


def _irrelevant_control_uncached(tokenizer,runtime,record,target_tokens,*,max_iterations=2048):
    """Find exact full-rendered-token equality, otherwise fail rather than relax."""
    prefix='\nUnrelated archive annotation: '
    suffix='. This archive annotation is unrelated to the question and decision.\n'
    body=''
    units=(' archive',' record',' ordinary',' neutral',' label',' entry',' reference',' note',' blue',' red',' text')
    for _ in range(max_iterations):
        candidate=insert_before_decision(record,prefix+body+suffix)
        count=token_count(tokenizer,runtime,candidate['text'])
        if count==target_tokens:return candidate
        if count>target_tokens:raise ValueError('Irrelevant control minimum length exceeds structured arm')
        choices=[]
        for unit in units:
            trial=insert_before_decision(record,prefix+body+unit+suffix)
            size=token_count(tokenizer,runtime,trial['text'])
            if count<size<=target_tokens:choices.append((size,unit))
        if not choices:raise ValueError('Cannot construct exactly token-length-matched irrelevant control')
        _,unit=max(choices,key=lambda x:(x[0],-units.index(x[1])))
        body+=unit
    raise ValueError('Token-length control search exceeded deterministic bound')


def irrelevant_control(tokenizer,runtime,record,target_tokens,*,max_iterations=2048):
    """Reuse exact note bytes within an unchanged question/labels/template tail.

    Cache entries contain only the appended annotation, never a previous full
    prompt. Different rules retain their own bytes/spans. Every hit still checks
    the current *entire rendered input*. A failed check falls back to the
    unchanged deterministic search rather than trimming or decoding tokens.
    """
    if not getattr(runtime,'rescue_length_cache_enabled',True):
        return _irrelevant_control_uncached(tokenizer,runtime,record,target_tokens,max_iterations=max_iterations)
    if not hasattr(runtime,'_rescue_length_cache'):
        runtime._rescue_length_cache={}
        runtime.rescue_length_cache_stats=dict(requests=0,hits=0,searches=0,failed_checks=0)
    stats=runtime.rescue_length_cache_stats;stats['requests']+=1
    minimum=insert_before_decision(record,'\nUnrelated archive annotation: . This archive annotation is unrelated to the question and decision.\n')
    gap=target_tokens-token_count(tokenizer,runtime,minimum['text'])
    rule_start,rule_end=record.spans['rule_block']
    # Preserve the entire question/options/decision tail in the key. Only the
    # separately delimited rule block may differ; no cross-question reuse.
    context=record.text[:rule_start]+record.text[rule_end:]
    key=(id(tokenizer),runtime.assistant_answer_prefix,
         json.dumps(runtime.chat_kwargs,sort_keys=True),
         hashlib.sha256(context.encode()).hexdigest(),gap)
    cached=runtime._rescue_length_cache.get(key)
    if cached is not None:
        candidate=insert_before_decision(record,cached)
        if token_count(tokenizer,runtime,candidate['text'])==target_tokens:
            stats['hits']+=1
            return candidate
        stats['failed_checks']+=1
    stats['searches']+=1
    result=_irrelevant_control_uncached(tokenizer,runtime,record,target_tokens,max_iterations=max_iterations)
    begin,end=result['spans']['rescue_addition']
    runtime._rescue_length_cache[key]=result['text'][begin:end]
    return result


def variants(runtime,record,row):
    structured=insert_before_decision(record,net_score_addition(row))
    n=token_count(runtime.tokenizer,runtime,structured['text'])
    control=irrelevant_control(runtime.tokenizer,runtime,record,n)
    f3,metadata=f3_addition(row)
    return {'external_net_score':structured,'length_matched_irrelevant':control,
        'f3_score_block':insert_before_decision(record,f3)},dict(structured_tokens=n,
        irrelevant_tokens=token_count(runtime.tokenizer,runtime,control['text']),f3=metadata)


def expected_score(row,readout):
    if readout.get('action') is None:return None
    return float(row['V_H'] if readout['action'] else readout['answer_correct'])


def normative_regret(row,readout):
    if readout.get('action') is None:return None
    selected=row['V_H'] if readout['action'] else row['z']
    return float(max(row['V_H'],row['z'])-selected)


def comparisons(row,before,after):
    score_before,score_after=expected_score(row,before),expected_score(row,after)
    regret_before,regret_after=normative_regret(row,before),normative_regret(row,after)
    lo,hi=min(0.,row['V_H']),max(1.,row['V_H'])
    blo,bhi=(lo,hi) if score_before is None else (score_before,score_before)
    alo,ahi=(lo,hi) if score_after is None else (score_after,score_after)
    policies={}
    for name,action in [('always_self',0),('always_help',1),('external_margin',int(row['M']>0))]:
        policies[name]=dict(action=action,expected_score=float(row['V_H'] if action else before['answer_correct']),
            normative_regret=float(max(row['V_H'],row['z'])-(row['V_H'] if action else row['z'])))
    return dict(expected_score_before=score_before,expected_score_after=score_after,
        expected_score_difference=None if score_before is None or score_after is None else score_after-score_before,
        expected_score_difference_bounds=[alo-bhi,ahi-blo],
        normative_regret_before=regret_before,normative_regret_after=regret_after,
        normative_regret_difference=None if regret_before is None or regret_after is None else regret_after-regret_before,
        offline_policies=policies,
        score_note='Self branch uses the matching prompt arm factual-answer correctness, including intervention-induced answer changes')


def summarize(rows):
    grouped={}
    for r in rows:grouped.setdefault((r['model'],r['domain'],r['arm']),[]).append(r)
    summaries=[]
    for key,group in sorted(grouped.items()):
        q=[r['question_id'] for r in group]
        before=[];after=[]
        for r in group:
            shared={k:r[k] for k in ('question_id','condition_id','model','domain','label_mapping','correct_answer')}
            before.append(normalize_row({**shared,**r['matched_baseline']}))
            after.append(normalize_row({**shared,**r['variant_readout']}))
        scores=[r['comparison']['expected_score_difference'] for r in group]
        regrets=[r['comparison']['normative_regret_difference'] for r in group]
        positive=[r for r in group if r['M']>0]
        summaries.append(dict(model=key[0],domain=key[1],arm=key[2],n_records=len(group),
            effects=intervention_decomposition(before,after),
            expected_score_difference=paired_bootstrap(q,[np.nan if v is None else v for v in scores]),
            expected_score_difference_lower_bound=paired_bootstrap(q,[r['comparison']['expected_score_difference_bounds'][0] for r in group]),
            expected_score_difference_upper_bound=paired_bootstrap(q,[r['comparison']['expected_score_difference_bounds'][1] for r in group]),
            normative_regret_difference=paired_bootstrap(q,[np.nan if v is None else v for v in regrets]),
            beneficial_help_subset=dict(n=len(positive),
                ask_after=quality_bounds([normalize_row(r['variant_readout'])['action'] for r in positive],question_ids=[r['question_id'] for r in positive]))))
    return summaries


def read_baselines(runs,phase,max_items):
    rows=[];manifests=[]
    for run in runs:
        p=Path(run);manifest=json.loads((p/'manifest.json').read_text());manifests.append(manifest)
        if manifest.get('status')!='complete':raise ValueError('Rescue needs completed baseline runs')
        for line in (p/'rows.jsonl').read_text().splitlines():
            if line.strip():
                r=normalize_row(json.loads(line));r['_run_dir']=str(p);rows.append(r)
    expected='val' if phase=='validation' else phase
    if not rows or any(r['fold']!=expected for r in rows):raise ValueError('Baseline fold does not match rescue phase')
    if len({r['model'] for r in rows})!=1:raise ValueError('One model per rescue invocation')
    keys=sorted({r['question_id'] for r in rows})
    if phase=='pilot':keys=keys[:max_items or 8]
    elif len(keys)!=(256 if phase=='validation' else 512):raise ValueError('Fixed validation256/test512 required')
    rows=[r for r in rows if r['question_id'] in set(keys)]
    from nmi.mechanism.prompts import conditions
    lookup={}
    for r in rows:
        key=(r['question_id'],r['domain'],r['condition_id'])
        if key in lookup:raise ValueError('Duplicate baseline condition')
        lookup[key]=r
        if r.get('z') is None:raise ValueError('Missing independently calibrated baseline z')
    domains=sorted({r['domain'] for r in rows})
    if phase=='test' and domains!=['id','ood']:raise ValueError('Final rescue test requires both frozen domains')
    for q in keys:
        for domain in domains:
            if {r['condition_id'] for r in rows if r['question_id']==q and r['domain']==domain}!={c.condition_id for c in conditions(domain)}:
                raise ValueError('Rescue requires the entire frozen36 condition grid per question/domain')
    return sorted(rows,key=lambda r:(r['question_id'],r['domain'],r['condition_id'])),manifests


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['pilot','validation','test'],default='pilot')
    p.add_argument('--conditions',choices=['all'],default='all')
    p.add_argument('--max-items',type=int,default=None,help='Pilot engineering only; default8')
    p.add_argument('--model-path',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--attention-backend',default='sdpa')
    p.add_argument('--batch-size',type=int,default=3,help='Logical pairs; physical batch is2B')
    p.add_argument('--inspect-only',action='store_true');p.add_argument('--data-root')
    p.add_argument('--max-wall-seconds',type=float,default=0)
    p.add_argument('--disable-length-cache',action='store_true',help='Reproduce the unchanged deterministic annotation search without reuse')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.batch_size<1 or (args.max_items is not None and args.max_items<1):raise ValueError('Positive batch/item counts required')
    if args.phase!='pilot' and args.max_items is not None:raise ValueError('Only pilot may set --max-items')
    from nmi.mechanism.data import DATA_ROOT,load_split
    data_root=Path(args.data_root) if args.data_root else DATA_ROOT
    out=data_path(args.output);out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'nmi/mechanism/analysis.py',ROOT/'nmi/mechanism/runtime.py',ROOT/'nmi/mechanism/prompts.py',ROOT/'nmi/mechanism/data.py',ROOT/'nmi/prompts.py',ROOT/'nmi/mech_prompts.py',ROOT/'configs/prompts/mech_v1.yaml',ROOT/'configs/prompts/ask_v1_1.yaml',Path(args.protocol),data_root/'manifest.json']
    if args.phase!='test':files.append(data_root/(args.phase+'.jsonl'))
    sources={str(p.resolve()):sha256_file(p) for p in files}
    # Test settings must freeze before opening baseline test outputs/artifacts.
    reservation=None
    if args.phase=='test':
        selection_path=data_root/'frozen_selection.json';selection=json.loads(selection_path.read_text())
        declared_model=json.loads((Path(args.runs[0])/'manifest.json').read_text())['identity']['settings']['model']
        frozen_settings=dict(task='mechanism_rescue',phase='test',model=declared_model,runs=[str(Path(p).resolve()) for p in args.runs],
            model_path=str(Path(args.model_path).resolve()),conditions='all',arms=list(ARMS),batch_size=args.batch_size,
            exact_rendered_token_length_control=True,domains=['id','ood'],n_questions=512,
            length_cache_enabled=not args.disable_length_cache,
            source_sha256=sources)
        validate_frozen_execution(selection,frozen_settings)
        # Separate append-only reservation from native C3 baseline consumption.
        import fcntl
        ledger=data_path(data_root/'rescue_test_consumption.jsonl')
        reservation=dict(model=declared_model,selection_sha256=sha256_file(selection_path),settings_sha256=digest(frozen_settings),output=str(out.resolve()))
        with ledger.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);previous=[json.loads(l) for l in f if l.strip()]
            same=[r for r in previous if r.get('model')==declared_model]
            if same and any(r!=reservation for r in same):raise PermissionError('Inconsistent repeated rescue test consumption')
            if not same:f.seek(0,2);f.write(json.dumps(reservation)+'\n');f.flush()
            fcntl.flock(f,fcntl.LOCK_UN)
    items=load_split(args.phase,root=data_root,allow_test=args.phase=='test')
    by_key={str(i.item_key):i for i in items}
    rows,manifests=read_baselines(args.runs,args.phase,args.max_items)
    if not {r['question_id'] for r in rows}<=set(by_key):raise ValueError('Baseline keys do not belong to frozen phase split')
    settings=[m['identity']['settings'] for m in manifests]
    prefixes={s.get('answer_prefix','') for s in settings}
    if len(prefixes)!=1:raise ValueError('Baseline runs use inconsistent assistant prefixes')
    prefix=next(iter(prefixes))
    if any(s['attention_backend']!=args.attention_backend for s in settings):raise ValueError('Attention backend differs from baseline')
    for run in args.runs:
        for name in ('rows.jsonl','manifest.json'):sources[str((Path(run)/name).resolve())]=sha256_file(Path(run)/name)
    model_sources=inspect_local_model(args.model_path)
    if any(m.get('model_sources')!=model_sources for m in manifests):raise ValueError('Model/tokenizer differs from baseline hashes')
    identity=dict(schema_version=1,phase=args.phase,sources=sources,model_sources=model_sources,
        assistant_answer_prefix=prefix,arms=list(ARMS),batch_size=args.batch_size,conditions='all',
        length_cache_enabled=not args.disable_length_cache,
        question_ids_sha256=digest(sorted({r['question_id'] for r in rows})),n_conditions=len(rows),
        baseline_artifacts_sha256=digest([r['artifact_sha256'] for r in rows]),test_reservation=reservation)
    if args.inspect_only:
        atomic_json(out/'inspection.json',dict(identity=identity,n_variant_prompts=len(rows)*3,n_forward_rows=len(rows)*6,model_loaded=False));return 0
    manifest_path=out/'manifest.json'
    previous_resources={}
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['identity']!=identity:raise ValueError('Rescue resume identity changed')
        previous_resources=previous.get('resources',{})
    else:atomic_json(manifest_path,dict(identity=identity,status='prepared'))
    runtime=load_runtime(args.model_path,args.device,args.attention_backend,assistant_answer_prefix=prefix)
    runtime.rescue_length_cache_enabled=not args.disable_length_cache
    from nmi.mechanism.prompts import conditions,render_prompt
    condition_maps={domain:{c.condition_id:c for c in conditions(domain)} for domain in {r['domain'] for r in rows}}
    results_path=out/'rows.jsonl'
    results=[json.loads(l) for l in results_path.read_text().splitlines()] if results_path.exists() else []
    done={(r['recipient_row_id'],r['arm']) for r in results};began=time.perf_counter();preparation_seconds=0.
    def checkpoint(status):
        current=runtime.resources();total={k:current[k]+previous_resources.get(k,0) if k in ('forward_count','forward_seconds','gpu_hours') else max(current[k],previous_resources.get(k,0)) for k in current}
        timing=dict(cpu_preparation_seconds_this_invocation=preparation_seconds,
            gpu_resident_wall_seconds_this_invocation=time.perf_counter()-began,
            length_cache=getattr(runtime,'rescue_length_cache_stats',None))
        atomic_json(manifest_path,dict(identity=identity,status=status,n_completed=len(results),resources=total,timing=timing))
        if status!='running':
            atomic_json(out/'summary.json',dict(status=status,groups=summarize(results),resources=total,timing=timing,
                limitation='Transparent external-z rescue and F3-style prompt controls; not identification of a native value mechanism'))
    for start in range(0,len(rows),args.batch_size):
        chunk=rows[start:start+args.batch_size];baselines=[];variant_sets=[];meta=[]
        preparation_began=time.perf_counter()
        for r in chunk:
            item=by_key[r['question_id']]
            if item.gt_position-1!=r['correct_answer'] or item.choices[item.gt_position-1]!=item.correct_text:
                raise ValueError('Baseline correct answer/order mismatch')
            path=Path(r['_run_dir'])/r['artifact']
            if sha256_file(path)!=r['artifact_sha256']:raise ValueError('Baseline artifact checksum mismatch')
            a=torch.load(path,map_location='cpu',weights_only=True)
            record=render_prompt(item,condition_maps[r['domain']][r['condition_id']])
            if record.text!=a['user_text']:raise ValueError('Current original prompt differs from baseline user_text')
            prepared=runtime.prepare(record)
            if prepared['input_ids']!=a['input_ids'] or prepared['label_ids']!=a['label_ids']:
                raise ValueError('Original prompt/tokenizer/prefix/label identity changed')
            plans,metadata=variants(runtime,record,r)
            prepared_variants={arm:runtime.prepare(rec) for arm,rec in plans.items()}
            if len(prepared_variants['external_net_score']['input_ids'])!=len(prepared_variants['length_matched_irrelevant']['input_ids']):
                raise ValueError('Full input token-length control failed')
            if any(p['label_ids']!=prepared['label_ids'] for p in prepared_variants.values()):raise ValueError('Rescue changed action labels')
            baselines.append(prepared);variant_sets.append(prepared_variants);meta.append(metadata)
        preparation_seconds+=time.perf_counter()-preparation_began
        for arm in ARMS:
            if all((r['row_id'],arm) in done for r in chunk):continue
            if args.max_wall_seconds>0 and time.perf_counter()-began>=args.max_wall_seconds:
                checkpoint('budget_stop');return 0
            prepared=baselines+[v[arm] for v in variant_sets]
            output=runtime.forward_batch(prepared,collect=False)
            batch_hash=digest([{k:p[k] for k in ('input_ids','positions','label_ids')} for p in prepared])
            for i,r in enumerate(chunk):
                if (r['row_id'],arm) in done:continue
                base_output=output[i];changed_output=output[len(chunk)+i]
                bscore=score_logits(base_output['logits'],base_output['label_ids'],r['correct_answer'])
                ascore=score_logits(changed_output['logits'],changed_output['label_ids'],r['correct_answer'])
                b=normalize_row({**r,**bscore});a=normalize_row({**r,**ascore})
                tag=digest([r['row_id'],arm])[:24];artifact_path='artifacts/'+tag+'.pt'
                h=save_artifact(out/artifact_path,dict(matched_native=base_output,variant=changed_output,
                    arm=arm,matched_batch_sha256=batch_hash,length_control=meta[i]))
                result=dict(question_id=r['question_id'],recipient_row_id=r['row_id'],condition_id=r['condition_id'],
                    model=r['model'],domain=r['domain'],fold=r['fold'],label_mapping=r['label_mapping'],
                    correct_answer=r['correct_answer'],a_H=r['a_H'],c0=r['c0'],w=r['w'],z=r['z'],V_H=r['V_H'],M=r['M'],
                    arm=arm,matched_batch_sha256=batch_hash,matched_baseline=bscore,variant_readout=ascore,
                    comparison=comparisons(r,b,a),length_control=meta[i],artifact=artifact_path,artifact_sha256=h)
                append_json(results_path,result);results.append(result);done.add((r['row_id'],arm))
            checkpoint('running')
    checkpoint('complete');return 0

if __name__=='__main__':raise SystemExit(main())
