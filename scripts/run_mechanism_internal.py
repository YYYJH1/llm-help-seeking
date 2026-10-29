#!/usr/bin/env python3
"""Local native-HF mechanism pilot; baseline batches and resumable exact swaps.

No GPU work happens for --inspect-only. Executing requires a hashed protocol.
Use CUDA_VISIBLE_DEVICES externally; --device cuda:0 addresses that allocation.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmi.mechanism.runtime import (POSITIONS, append_json, atomic_json, data_path, digest,
    inspect_local_model, load_runtime, relative_layers, save_artifact, score_logits, sha256_file,
    validate_frozen_execution, register_test_consumption, test_consumption_event)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['qwen3-4b','gemma-4-e4b-it','smollm3-3b'])
    p.add_argument('--model-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--protocol')
    p.add_argument('--inspect-only', action='store_true')
    p.add_argument('--fold', choices=['pilot','train','validation','test'], default='pilot')
    p.add_argument('--split-root')
    p.add_argument('--allow-test', action='store_true')
    p.add_argument('--domain', choices=['id','ood'], default='id')
    p.add_argument('--max-items', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--attention-backend', default='sdpa')
    p.add_argument('--equivalence-items', type=int, default=8)
    p.add_argument('--equivalence-atol', type=float, default=0.02)
    p.add_argument('--swaps', action='store_true')
    p.add_argument('--with-calibrated-z', action='store_true', help='independent original P-answer pass with frozen model temperature')
    p.add_argument('--probe-items', type=int, default=128)
    p.add_argument('--sites-json', help='exact [[layer,position],...] pairs; overrides layer/position cross product')
    p.add_argument('--analysis-policy', default='matched_2b_v1')
    p.add_argument('--answer-prefix', default='', help='exact assistant prefill for main decisions only; empty reproduces v1')
    p.add_argument('--layers', help='comma-separated layer indices; default 8 relative depths')
    p.add_argument('--positions', default=','.join(POSITIONS))
    p.add_argument('--arms', default='swap,identity,random')
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--max-wall-seconds', type=float, default=0, help='stop between batches/forwards; 0 means launcher owns budget')
    return p


def records(path):
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text().splitlines()):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f'incomplete/corrupt JSONL {path}:{i+1}; inspect before resuming') from e
    return rows


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_items < 1 or args.batch_size < 1 or args.probe_items < 1:
        raise ValueError('item and batch counts must be positive')
    out = data_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.inspect_only:
        atomic_json(out/'model_inspection.json', inspect_local_model(args.model_path))
        return 0
    if not args.protocol:
        raise ValueError('--protocol required before execution')
    if args.fold == 'test' and not args.allow_test:
        raise ValueError('test access requires explicit --allow-test')
    from nmi.mechanism.data import load_split, DATA_ROOT
    from nmi.mechanism.prompts import conditions, render_prompt, pair_conditions, calibrated_answer_chat_text, CALIBRATION_TEMPERATURES
    import torch
    positions = args.positions.split(',')
    arms = args.arms.split(',')
    if not set(positions) <= set(POSITIONS) or not set(arms) <= {'swap','identity','random'}:
        raise ValueError('invalid positions or arms')
    source_files = [Path(__file__), Path(__file__).parents[1]/'nmi/mechanism/runtime.py', Path(__file__).parents[1]/'nmi/mechanism/prompts.py', Path(__file__).parents[1]/'nmi/mechanism/data.py', Path(args.protocol)]
    kwargs = {'allow_test': args.allow_test}
    if args.split_root:
        kwargs['root'] = args.split_root
    split_root = Path(args.split_root) if args.split_root else DATA_ROOT
    source_files += [split_root/'manifest.json']
    if args.fold != 'test':
        source_files += [split_root/(args.fold+'.jsonl')]
    conds = conditions(domain=args.domain)
    explicit_sites = json.loads(args.sites_json) if args.sites_json else None
    if explicit_sites is not None:
        if not args.swaps or not explicit_sites:
            raise ValueError('explicit nonempty sites require --swaps')
        if any(not isinstance(x,list) or len(x)!=2 or not isinstance(x[0],int) or x[0]<0 or x[1] not in POSITIONS for x in explicit_sites):
            raise ValueError('invalid layer-position pairs')
        if len({tuple(x) for x in explicit_sites})!=len(explicit_sites):
            raise ValueError('duplicate sites')
    reservation = None
    selection_sha = None
    if args.fold == 'test':
        if args.swaps and explicit_sites is None:
            raise PermissionError('test C3 requires exact --sites-json, never a layer-position cross product')
        selection_path = split_root/'frozen_selection.json'
        selection = json.loads(selection_path.read_text())
        selection_sha = sha256_file(selection_path)
        if selection.get('data_manifest_sha256') != sha256_file(split_root/'manifest.json'):
            raise PermissionError('frozen selection binds a different data manifest')
        frozen_sites = explicit_sites or []
        settings = {'model':args.model,'domain':args.domain,'swaps':args.swaps,'sites':frozen_sites,
                    'layers':sorted({x[0] for x in frozen_sites}),
                    'positions':sorted({x[1] for x in frozen_sites}),
                    'arms':arms if args.swaps else [],'batch_size':args.batch_size,'seed':args.seed,
                    'condition_grid_sha256':digest([dataclasses.asdict(c) for c in conds]),
                    'analysis_policy':args.analysis_policy,'assistant_answer_prefix':args.answer_prefix,'max_items':args.max_items,'probe_items':args.probe_items,
                    'with_calibrated_z':args.with_calibrated_z,'attention_backend':args.attention_backend,
                    'model_path':str(Path(args.model_path).resolve()),'protocol_sha256':sha256_file(args.protocol),
                    'equivalence_items':args.equivalence_items,'equivalence_atol':args.equivalence_atol}
        validate_frozen_execution(selection,settings)
        reservation = register_test_consumption(split_root,selection_sha,settings,out)
        source_files += [selection_path]
    loaded_items = load_split(args.fold, **kwargs)
    items = loaded_items[:args.max_items]
    if not items:
        raise ValueError('empty split')
    if reservation is not None:
        test_consumption_event(split_root,reservation,'materialized',opened_items=len(loaded_items),selected_items=len(items),selected_item_keys_sha256=digest([str(i.item_key) for i in items]))
    plans = []
    for item in items:
        for condition in conds:
            rec = render_prompt(item, condition)
            key = str(item.item_key)
            cid = str(condition.condition_id)
            rid = digest([args.model,args.fold,key,cid])[:24]
            answer = int(item.gt_position) - 1
            if isinstance(answer, str):
                answer = 'ABCD'.find(answer) if answer in 'ABCD' else int(answer)
            metadata = {'row_id':rid, 'item_key':key, 'fold':args.fold, 'domain':args.domain, 'model':args.model, 'condition_id':cid, 'a_H':condition.a_h, 'c0':condition.c0, 'w':condition.w, 'semantic':condition.frame, 'mapping_id':condition.mapping_id, 'z':None, 'correct_answer':answer}
            plans.append((rec,metadata))
    identity = {'schema_version':2, 'frozen_selection_sha256':selection_sha, 'settings': {k:v for k,v in vars(args).items() if k not in ('max_wall_seconds',)}, 'sources': {str(p.resolve()):sha256_file(p) for p in source_files}, 'items_sha256':digest([str(i.item_key) for i in items]), 'prompt_plan_sha256':digest([{'text':r.text,'spans':r.spans,**m} for r,m in plans]), 'conditions':[dataclasses.asdict(c) for c in conds]}
    model_sources = inspect_local_model(args.model_path)
    manifest = out/'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['identity'] != identity or previous['model_sources'] != model_sources:
            raise ValueError('resume identity mismatch: protocol, prompts, data, source or options changed')
    else:
        atomic_json(manifest, {'identity':identity, 'model_sources':model_sources, 'status':'prepared'})
    old_stats = json.loads((out/'resources.json').read_text()) if (out/'resources.json').exists() else {}
    started = time.perf_counter()
    runtime = load_runtime(args.model_path,args.device,args.attention_backend,assistant_answer_prefix=args.answer_prefix)
    if runtime.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(runtime.device)
    # Actual engineering evidence, including raw tensors, is persisted independently
    # of whether the frozen tolerance passes. A failed bridge stops all mechanisms.
    eqpath = out/'equivalence.json'
    if not eqpath.exists() and args.equivalence_items > 0:
        # Evenly spaced across items/condition blocks, covering different lengths.
        n_eq = min(args.equivalence_items,len(plans))
        indices = sorted({round(i*(len(plans)-1)/max(n_eq-1,1)) for i in range(n_eq)})
        selected = [plans[i] for i in indices]
        prepared = [runtime.prepare(r) for r,m in selected]
        batch_plain = runtime.forward_batch(prepared,collect=False)
        batch_identity = runtime.forward_batch(prepared,collect=False,identity_hooks=True)
        batch_collect = runtime.forward_batch(prepared,collect=True)
        checks, bridges = [], []
        for i,((record,meta),p) in enumerate(zip(selected,prepared)):
            single = runtime.forward(p,collect=False)
            artifact = {'batch_plain':batch_plain[i],'batch_identity':batch_identity[i],'batch_collect':batch_collect[i],'single_plain':single}
            artifact_path = 'equivalence/'+meta['row_id']+'.pt'
            h = save_artifact(out/artifact_path,artifact)
            ref = score_logits(batch_plain[i]['logits'],p['label_ids'])
            for name,result in [('same_batch_identity',batch_identity[i]),('same_batch_collect',batch_collect[i])]:
                scored = score_logits(result['logits'],p['label_ids'])
                diff = float((batch_plain[i]['logits']-result['logits']).abs().max())
                checks.append({'row_id':meta['row_id'],'comparison':name,'max_abs_logits_difference':diff,'m_difference':scored['m']-ref['m'],'same_action':scored['action']==ref['action'],'artifact':artifact_path,'artifact_sha256':h,'passed':diff == 0.0})
            scored = score_logits(single['logits'],p['label_ids'])
            diff = float((single['logits']-batch_plain[i]['logits']).abs().max())
            bridges.append({'row_id':meta['row_id'],'comparison':'single_vs_batch_bridge_only','max_abs_logits_difference':diff,'m_difference':scored['m']-ref['m'],'same_action':scored['action']==ref['action'],'original_atol':args.equivalence_atol,'passed':diff<=args.equivalence_atol and scored['action']==ref['action'],'required_for_mechanism':False})
        atomic_json(eqpath,{'same_batch_required_atol':0.0,'checks':checks,'single_batch_bridge':bridges,'single_batch_bridge_passed':all(c['passed'] for c in bridges),'passed':all(c['passed'] for c in checks),'measurement_policy':'causal deltas use matched identical 2B reference only; bridge differences are not causal effects'})
    if eqpath.exists() and not json.loads(eqpath.read_text())['passed']:
        raise ValueError('engineering equivalence failed; inspect saved raw outputs before proceeding')
    rows = records(out/'rows.jsonl')
    done = {r['row_id']:r for r in rows}
    for r in rows:
        if sha256_file(out/r['artifact']) != r['artifact_sha256']:
            raise ValueError('resume artifact hash mismatch: '+r['artifact'])
    def checkpoint(status):
        stats = runtime.resources()
        elapsed = time.perf_counter()-started
        stats.update({'session_wall_seconds':elapsed, 'cumulative_wall_seconds':old_stats.get('cumulative_wall_seconds',0)+elapsed, 'cumulative_gpu_hours':old_stats.get('cumulative_gpu_hours',0)+(elapsed/3600 if runtime.device.type=='cuda' else 0), 'cumulative_forward_seconds':old_stats.get('cumulative_forward_seconds',0)+stats['forward_seconds'], 'cumulative_forward_count':old_stats.get('cumulative_forward_count',0)+stats['forward_count'], 'status':status, 'baseline_rows':len(done)})
        atomic_json(out/'resources.json',stats)
    def expired():
        return args.max_wall_seconds > 0 and time.perf_counter()-started >= args.max_wall_seconds
    if args.with_calibrated_z:
        answer_rows = records(out/'answers.jsonl')
        for row in answer_rows:
            if sha256_file(out/row['artifact']) != row['artifact_sha256']:
                raise ValueError('resume answer artifact hash mismatch')
        z_values = {r['item_key']:r['z'] for r in answer_rows}
        missing = [item for item in items if str(item.item_key) not in z_values]
        temperature = CALIBRATION_TEMPERATURES[args.model]
        for j in range(0,len(missing),args.batch_size):
            if expired():
                checkpoint('budget_stop');return 0
            batch_items = missing[j:j+args.batch_size]
            prepared = []
            for item in batch_items:
                text = calibrated_answer_chat_text(item,runtime.tokenizer,{'enable_thinking':False})
                ids = runtime.tokenizer.encode(text,add_special_tokens=False)
                labels = []
                for label in ('1','2','3','4'):
                    full = runtime.tokenizer.encode(text+label,add_special_tokens=False)
                    if full[:-1] != ids:
                        raise ValueError('calibrated answer prefix is not a one-token continuation')
                    labels.append(full[-1])
                prepared.append({'prompt':text,'input_ids':ids,'positions':{k:len(ids)-1 for k in POSITIONS},'label_ids':labels})
            for item,artifact in zip(batch_items,runtime.forward_batch(prepared,collect=False)):
                values = artifact['logits'][artifact['label_ids']]
                z = float(torch.softmax(values.double()/temperature,dim=-1).max())
                path = 'answers/'+digest(str(item.item_key))[:24]+'.pt'
                h = save_artifact(out/path,artifact)
                z_values[str(item.item_key)] = z
                append_json(out/'answers.jsonl',{'item_key':str(item.item_key),'z':z,'temperature':temperature,'raw_answer_logits':values.tolist(),'forced_answer_index':int(values.argmax()),'correct_answer':item.gt_position-1,'artifact':path,'artifact_sha256':h})
            checkpoint('answers_running')
        for record,meta in plans:
            meta['z'] = z_values[meta['item_key']]
    pending = [(r,m) for r,m in plans if m['row_id'] not in done]
    for j in range(0,len(pending),args.batch_size):
        if expired():
            checkpoint('budget_stop'); return 0
        batch = pending[j:j+args.batch_size]
        outputs = runtime.forward_batch([runtime.prepare(r) for r,m in batch])
        for (record,meta), artifact in zip(batch,outputs):
            artifact['metadata'] = meta
            path = 'activations/'+meta['row_id']+'.pt'
            h = save_artifact(out/path,artifact)
            row = {**meta, **score_logits(artifact['logits'],artifact['label_ids'],meta['correct_answer']), 'label_mapping':artifact['label_mapping'], 'positions':artifact['positions'], 'prompt_sha256':artifact['prompt_sha256'], 'n_tokens':len(artifact['input_ids']), 'artifact':path, 'artifact_sha256':h}
            append_json(out/'rows.jsonl',row)
            done[row['row_id']] = row
        if reservation is not None:
            test_consumption_event(split_root,reservation,'baseline_written',row_ids=[m['row_id'] for r,m in batch])
        checkpoint('baseline_running')
    if args.swaps:
        layer_ids = list(map(int,args.layers.split(','))) if args.layers else relative_layers(len(runtime.layers))
        sites = [tuple(x) for x in explicit_sites] if explicit_sites is not None else [(layer,position) for layer in layer_ids for position in positions]
        layer_ids = sorted({layer for layer,position in sites})
        if not set(layer_ids) <= set(range(len(runtime.layers))):
            raise ValueError('layer outside model')
        previous_swaps = records(out/'swaps.jsonl')
        for row in previous_swaps:
            if sha256_file(out/row['artifact']) != row['artifact_sha256']:
                raise ValueError('resume intervention artifact hash mismatch')
        swap_done = {r['intervention_id'] for r in previous_swaps}
        lookup = {(r['item_key'],r['condition_id']):r for r in done.values()}
        jobs = []
        def flush_jobs():
            if not jobs:
                return
            outputs = runtime.interventions_batch([j[0] for j in jobs])
            for (request,spec,iid,base_row),artifact in zip(jobs,outputs):
                path = 'interventions/'+iid+'.pt'
                artifact['intervention'] = spec
                h = save_artifact(out/path,artifact)
                row = {**spec,'intervention_id':iid, **score_logits(artifact['logits'],artifact['label_ids'],base_row['correct_answer']), 'baseline_m':artifact['matched_baseline']['m'], 'unmatched_archival_baseline_m':base_row['m'], 'matched_baseline':artifact['matched_baseline'], 'matched_donor':artifact['matched_donor'], 'matched_batch_sha256':artifact['matched_batch_sha256'], 'matched_batch_size':artifact['matched_batch_size'], 'matched_recipient_row':artifact['matched_recipient_row'], 'donor_delta_l2':artifact['donor_delta_l2'], 'requested_delta_l2':artifact['requested_delta_l2'], 'actual_delta_l2':artifact['actual_delta_l2'], 'artifact':path, 'artifact_sha256':h}
                append_json(out/'swaps.jsonl',row)
                swap_done.add(iid)
            if reservation is not None:
                test_consumption_event(split_root,reservation,'interventions_written',intervention_ids=[j[2] for j in jobs])
            jobs.clear()
            checkpoint('swaps_running')
        for item in items[:args.probe_items]:
            for pair in pair_conditions(args.domain):
                a = lookup[(str(item.item_key),pair['base_condition_id'])]
                b = lookup[(str(item.item_key),pair['source_condition_id'])]
                for base_row,donor_row,direction in ((a,b,'forward'),(b,a,'reverse')):
                    base = torch.load(out/base_row['artifact'],weights_only=True,map_location='cpu')
                    donor = torch.load(out/donor_row['artifact'],weights_only=True,map_location='cpu')
                    for layer,position in sites:
                        for arm in arms:
                            spec = {'recipient_row_id':base_row['row_id'],'donor_row_id':donor_row['row_id'],'contrast_family':pair['family'],'direction':direction,'layer':layer,'module_name':runtime.layer_names[layer],'position':position,'arm':arm,'seed':args.seed}
                            iid = digest(spec)[:24]
                            if iid in swap_done:
                                continue
                            if expired():
                                checkpoint('budget_stop');return 0
                            jobs.append(((base,donor,layer,position,arm,args.seed),spec,iid,base_row))
                            if len(jobs) >= args.batch_size:
                                    flush_jobs()
        flush_jobs()
    if reservation is not None:
        test_consumption_event(split_root,reservation,'complete',baseline_rows=len(done))
    checkpoint('complete')
    info = json.loads(manifest.read_text()); info['status']='complete';atomic_json(manifest,info)
    print(json.dumps({'status':'complete','output':str(out),'baseline_rows':len(done),'resources':str(out/'resources.json')}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
