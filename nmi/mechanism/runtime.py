"""Local, single-example HF mechanism forwards; no model loads at import time.

Cache decoder-layer *outputs* at three semantically aligned positions. All
interventions are independent forwards and hooks are removed even on failure.
This is an intervention runner, not evidence that an activation has a particular
psychological meaning. bf16 weights are preserved; logits are scored in fp32.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

POSITIONS = ('cost_number', 'rule_block', 'decision')


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def data_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if not p.is_relative_to(os.environ.get('NMI_OUTPUT_ROOT', '/')):
        raise ValueError('All output paths must resolve under NMI_OUTPUT_ROOT')
    return p


def atomic_json(path: str | Path, value: Any) -> None:
    p = data_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.partial')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(p)


def save_artifact(path: str | Path, value: Any) -> str:
    p = data_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.partial')
    torch.save(value, tmp)
    tmp.replace(p)
    return sha256_file(p)


def append_json(path: str | Path, value: Any) -> None:
    with data_path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def decoder_layers(model: Any):
    for path in (('model', 'layers'), ('model', 'language_model', 'layers'), ('language_model', 'layers'), ('language_model', 'model', 'layers')):
        obj = model
        for part in path:
            obj = getattr(obj, part, None)
        if isinstance(obj, torch.nn.ModuleList) and len(obj):
            return obj
    raise ValueError(f'Unsupported decoder structure: {type(model).__name__}')


def relative_layers(n: int, count: int = 8) -> list[int]:
    if n < 1 or count < 1:
        raise ValueError('positive layer counts required')
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)}) if count > 1 else [n // 2]


def align_positions(tokenizer, user_text: str, rendered: str, spans: Mapping[str, Any]):
    """Use fast-tokenizer character offsets, never guessed absolute indices."""
    start = rendered.find(user_text)
    if start < 0 or rendered.find(user_text, start + 1) >= 0:
        raise ValueError('User text must occur exactly once in rendered chat')
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if not ids:
        raise ValueError('empty input')
    positions = {'decision': len(ids) - 1}
    for name in POSITIONS[:-1]:
        if name not in spans:
            raise ValueError(f'missing semantic span: {name}')
        a, b = map(int, spans[name])
        if not 0 <= a < b <= len(user_text):
            raise ValueError(f'invalid character span {name}: {(a,b)}')
        hits = [i for i, (lo, hi) in enumerate(offsets) if hi > start + a and lo < start + b and hi > lo]
        if not hits:
            raise ValueError(f'no token overlaps semantic span {name}')
        positions[name] = hits[-1]
    return list(map(int, ids)), positions, [list(map(int, x)) for x in offsets]


def label_token_ids(tokenizer, rendered: str, option_labels, help_label):
    labels = [str(x) for x in option_labels] + [str(help_label)]
    if len(labels) != 5 or len(set(labels)) != 5:
        raise ValueError('five distinct semantic labels required')
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    ids = []
    for label in labels:
        full = tokenizer.encode(rendered + label, add_special_tokens=False)
        if full[:len(prefix)] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(f'label {label!r} not exactly one continuation token; require separate readout bridge')
        ids.append(int(full[-1]))
    if len(set(ids)) != 5:
        raise ValueError('labels alias the same token')
    return ids


def score_logits(logits: torch.Tensor, label_ids, correct_answer=None):
    x = logits.detach().float().cpu().reshape(-1)
    if not torch.isfinite(x).all() or len(label_ids) != 5:
        raise ValueError('non-finite logits or invalid labels')
    ids = torch.tensor(label_ids, dtype=torch.long)
    legal = x[ids]
    logp = torch.log_softmax(x, dim=-1)
    p = logp[ids].exp()
    mass = float(p.sum())
    semantic = int(legal.argmax())
    free = int(x.argmax())
    m = float(legal[4] - legal[:4].max())
    # Strict m > 0 rule for asking for help; ties choose the first answer.
    chosen = 4 if m > 0 else int(legal[:4].argmax())
    result = {'semantic_logits': legal.tolist(), 'semantic_logprobs': logp[ids].tolist(), 'semantic_probs': p.tolist(), 'legal_mass': mass, 'digitmass': mass, 'valid': mass >= 0.5, 'adequate': mass >= 0.5, 'free_token_valid': free in label_ids, 'free_token_id': free, 'free_token_logprob': float(logp[free]), 'm': m, 'action': 'help' if chosen == 4 else f'answer{chosen}', 'chosen_semantic_index': chosen, 'help_probability_legal': float(torch.softmax(legal, dim=-1)[4]), 'max_answer_semantic_index': int(legal[:4].argmax())}
    if correct_answer is not None:
        result['forced_answer_correct'] = int(legal[:4].argmax()) == int(correct_answer)
    return result


def random_delta(reference: torch.Tensor, seed: int):
    """CPU RNG independent of global and CUDA state; match the actual swap norm."""
    ref = reference.detach().float().cpu()
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    vec = torch.randn(ref.shape, generator=g)
    if float(ref.norm()) == 0:
        return torch.zeros_like(ref)
    return vec * (ref.norm() / vec.norm())


@contextlib.contextmanager
def installed_hooks(layers, positions, cache, patch=None):
    handles = []
    try:
        for index, layer in enumerate(layers):
            def hook(module, args, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError('Expected single-example decoder tensor [1,T,D]')
                if patch is not None and index == int(patch['layer']):
                    pos = positions[patch['position']]
                    new = hidden.clone()
                    delta = patch['delta'].to(device=hidden.device, dtype=hidden.dtype)
                    if delta.shape != hidden[0, pos].shape:
                        raise ValueError('intervention dimension mismatch')
                    new[0, pos] = patch['value'].to(device=hidden.device, dtype=hidden.dtype) if 'value' in patch else hidden[0, pos] + delta
                    hidden = new
                    output = (new, *output[1:]) if isinstance(output, tuple) else new
                if cache is not None:
                    cache[str(index)] = hidden[0, [positions[k] for k in POSITIONS]].detach().float().cpu().clone()
                return output
            handles.append(layer.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


class Runtime:
    def __init__(self, model, tokenizer, *, device='cuda:0', chat_kwargs=None, assistant_answer_prefix=''):
        if not isinstance(assistant_answer_prefix,str):
            raise TypeError('assistant_answer_prefix must be an exact string')
        self.assistant_answer_prefix = assistant_answer_prefix
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.chat_kwargs = dict(chat_kwargs or {'enable_thinking': False})
        self.layers = decoder_layers(model)
        self.layer_names = {i: name for name, module in model.named_modules() for i, layer in enumerate(self.layers) if module is layer}
        self.forward_seconds = 0.0
        self.forward_count = 0

    def prepare(self, record):
        from nmi.prompts import render_chat
        text = record.text if hasattr(record, 'text') else record['text']
        spans = record.spans if hasattr(record, 'spans') else record['spans']
        options = record.option_labels if hasattr(record, 'option_labels') else record['option_labels']
        help_label = record.help_label if hasattr(record, 'help_label') else record['help_label']
        rendered = render_chat(self.tokenizer, text, self.chat_kwargs, assistant_prefix=self.assistant_answer_prefix)
        ids, positions, offsets = align_positions(self.tokenizer, text, rendered, spans)
        labels = label_token_ids(self.tokenizer, rendered, options, help_label)
        return {'user_text': text, 'prompt': rendered, 'prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(), 'input_ids': ids, 'positions': positions, 'offset_mapping': offsets, 'label_ids': labels, 'label_mapping': dict(zip([str(x) for x in options] + [str(help_label)], ['answer0','answer1','answer2','answer3','help'])), 'chat_kwargs': self.chat_kwargs, 'assistant_answer_prefix':self.assistant_answer_prefix}

    def forward(self, prepared, *, patch=None, collect=True, identity_hooks=False):
        if patch is not None and (int(patch['layer']) not in range(len(self.layers)) or patch['position'] not in POSITIONS):
            raise ValueError('invalid intervention site')
        cache = {} if collect else None
        ids = torch.tensor([prepared['input_ids']], dtype=torch.long, device=self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        began = time.perf_counter()
        hooks = installed_hooks(self.layers, prepared['positions'], cache, patch) if collect or patch is not None or identity_hooks else contextlib.nullcontext()
        with torch.inference_mode(), hooks:
            output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.forward_seconds += time.perf_counter() - began
        self.forward_count += 1
        return {**prepared, 'logits': output.logits[0, -1].detach().float().cpu().clone(), 'activations': cache or {}}

    def forward_batch(self, prepared, *, patches=None, collect=True, identity_hooks=False):
        """Left-padded baseline batch, explicit positions; CPU fp32 site cache."""
        if not prepared:
            return []
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError('tokenizer has neither pad nor EOS id')
        longest = max(len(p['input_ids']) for p in prepared)
        ids = torch.full((len(prepared), longest), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        sites = []
        for row, p in enumerate(prepared):
            n = len(p['input_ids'])
            ids[row, -n:] = torch.tensor(p['input_ids'], device=self.device)
            mask[row, -n:] = 1
            sites.append([longest-n+p['positions'][k] for k in POSITIONS])
        caches = {}
        handles = []
        if patches is not None and len(patches) != len(prepared):
            raise ValueError('one patch per batch row required')
        try:
            for layer_id, layer in (enumerate(self.layers) if collect or patches is not None or identity_hooks else []):
                def collect_hook(module, args, output, layer_id=layer_id):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.shape[:2] != ids.shape:
                        raise ValueError('unexpected decoder shape')
                    changed = None
                    if patches is not None:
                        for i, patch in enumerate(patches):
                            if patch is None or int(patch['layer']) != layer_id:
                                continue
                            if changed is None:
                                changed = h.clone()
                            site = sites[i][POSITIONS.index(patch['position'])]
                            if 'value' in patch:
                                changed[i,site] = patch['value'].to(device=h.device,dtype=h.dtype)
                            else:
                                changed[i,site] = h[i,site] + patch['delta'].to(device=h.device,dtype=h.dtype)
                    if changed is not None:
                        h = changed
                    if collect:
                        caches[str(layer_id)] = torch.stack([h[i, sites[i]] for i in range(len(prepared))]).detach().float().cpu()
                    if changed is not None:
                        return (changed,*output[1:]) if isinstance(output,tuple) else changed
                handles.append(layer.register_forward_hook(collect_hook))
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            began = time.perf_counter()
            with torch.inference_mode():
                out = self.model(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1)-1).clamp_min(0), use_cache=False, logits_to_keep=1)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.forward_seconds += time.perf_counter() - began
            self.forward_count += len(prepared)
            return [{**p, 'logits': out.logits[i,-1].detach().float().cpu().clone(), 'activations': {k:v[i].clone() for k,v in caches.items()}} for i,p in enumerate(prepared)]
        finally:
            for h in handles:
                h.remove()

    def intervention(self, base, donor, layer, position, arm, seed=0):
        idx = POSITIONS.index(position)
        target = donor['activations'][str(layer)][idx]
        original = base['activations'][str(layer)][idx]
        delta = target.float() - original.float()
        if arm == 'identity':
            delta = torch.zeros_like(delta)
        elif arm == 'random':
            delta = random_delta(delta, seed)
        elif arm != 'swap':
            raise ValueError(f'unknown intervention arm {arm}')
        patch = {'layer': layer, 'position': position, 'delta': delta}
        if arm == 'swap':
            patch['value'] = target
        elif arm == 'identity':
            patch['value'] = original
        return self.forward(base, patch=patch, collect=False)

    def interventions_batch(self, requests):
        """Matched 2B design: the reference and intervention passes have an identical input matrix.

        Donors occupy the last B rows and are never patched. The cached reference is
        valid only for exactly this ordered matrix/position set; single-example
        forwards provide inputs only, never intervention values or baselines.
        """
        if not requests:
            return []
        bases = [r[0] for r in requests]
        donors = [r[1] for r in requests]
        plans = bases + donors
        key = digest([{'input_ids':p['input_ids'],'positions':p['positions'],'label_ids':p['label_ids']} for p in plans])
        if getattr(self,'_matched_reference_key',None) != key:
            self._matched_reference = self.forward_batch(plans,collect=True)
            self._matched_reference_key = key
        reference = self._matched_reference
        patches, norms = [], []
        dtype = next(self.model.parameters()).dtype
        for i,(_,_,layer,position,arm,seed) in enumerate(requests):
            if layer not in range(len(self.layers)) or position not in POSITIONS:
                raise ValueError('invalid intervention site')
            idx = POSITIONS.index(position)
            original = reference[i]['activations'][str(layer)][idx]
            target = reference[i+len(requests)]['activations'][str(layer)][idx]
            patch = {'layer':layer,'position':position}
            if arm == 'swap':
                patch['value'] = target
                actual = target-original
                requested = actual
            elif arm == 'identity':
                patch['value'] = original
                actual = requested = torch.zeros_like(original)
            elif arm == 'random':
                requested = random_delta(target-original,seed)
                patch['delta'] = requested
                actual = (original.to(dtype)+requested.to(dtype)).float()-original
            else:
                raise ValueError('unknown intervention arm')
            patches.append(patch)
            norms.append({'donor_delta_l2':float((target-original).norm()),'requested_delta_l2':float(requested.norm()),'actual_delta_l2':float(actual.norm()),'recipient_activation':original,'donor_activation':target})
        outputs = self.forward_batch(plans,patches=patches+[None]*len(requests),collect=False)
        result = []
        for i,(request,output) in enumerate(zip(requests,outputs)):
            arm = request[4]
            if arm == 'identity' and not torch.equal(reference[i]['logits'],output['logits']):
                raise ValueError('same-batch identity changed logits; reject mechanism measurement')
            output.update(matched_batch_sha256=key,matched_batch_size=len(plans),matched_recipient_row=i,
                matched_baseline_logits=reference[i]['logits'].clone(),
                matched_donor_logits=reference[i+len(requests)]['logits'].clone(),
                matched_baseline=score_logits(reference[i]['logits'],bases[i]['label_ids']),
                matched_donor=score_logits(reference[i+len(requests)]['logits'],donors[i]['label_ids']),
                **norms[i])
            result.append(output)
        return result

    def resources(self):
        return {'forward_count': self.forward_count, 'forward_seconds': self.forward_seconds, 'gpu_hours': self.forward_seconds / 3600 if self.device.type == 'cuda' else 0.0, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else 0}


def inspect_local_model(path):
    """No weights loaded. Report installed support and immutable file hashes."""
    import transformers
    from transformers import AutoConfig
    p = Path(path).resolve()
    cfg = AutoConfig.from_pretrained(p, local_files_only=True)
    files = [x for x in p.iterdir() if x.is_file() and (x.name.endswith(('.json', '.jinja', '.model')) or x.name.endswith('.safetensors'))]
    return {'path': str(p), 'transformers_version': transformers.__version__, 'torch_version': torch.__version__, 'model_type': cfg.model_type, 'architectures': cfg.architectures, 'files': {f.name: {'sha256': sha256_file(f), 'bytes': f.stat().st_size} for f in sorted(files)}, 'cuda_executed': False}


def load_runtime(path, device='cuda:0', attn_implementation='sdpa', *, assistant_answer_prefix=''):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    cls = AutoModelForImageTextToText if cfg.model_type == 'gemma4' else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True, padding_side='left')
    model = cls.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16, attn_implementation=attn_implementation).to(device).eval()
    if {p.dtype for p in model.parameters() if p.is_floating_point()} - {torch.bfloat16}:
        raise ValueError('loaded weights not uniformly bf16')
    return Runtime(model, tokenizer, device=device, chat_kwargs={'enable_thinking': False}, assistant_answer_prefix=assistant_answer_prefix)


def validate_frozen_execution(selection: Mapping[str, Any], settings: Mapping[str, Any]):
    """Exact execution entry match; a field that is absent never acts as a wildcard."""
    entries = selection.get('execution_settings')
    if not isinstance(entries,list) or not entries:
        raise PermissionError('test requires nonempty frozen execution_settings')
    if not selection.get('analysis_choices') or not selection.get('development_evidence_sha256'):
        raise PermissionError('test requires frozen analysis and development evidence')
    if sum(entry == dict(settings) for entry in entries) != 1:
        raise PermissionError('test execution settings do not exactly match one frozen entry')
    return digest(settings)


def register_test_consumption(root, selection_sha256, settings, output):
    """Append-only, interprocess-locked reservation. Only identical-run resume allowed.

    Reservation occurs before opening held-out text. Separate progress events
    record actual materialization and outputs; a failed attempt is not a result.
    """
    import fcntl
    ledger = data_path(Path(root)/'test_consumption.jsonl')
    record = {'event':'reserved','selection_sha256':selection_sha256,'settings_sha256':digest(settings),
              'model':settings['model'],'domain':settings['domain'],'swaps':settings['swaps'],
              'output':str(data_path(output)),'settings':dict(settings)}
    ledger.parent.mkdir(parents=True,exist_ok=True)
    with ledger.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.seek(0)
        previous = [json.loads(line) for line in f if line.strip()]
        same = [r for r in previous if r.get('event')=='reserved' and
                (r['model'],r['domain'],r['swaps'])==(record['model'],record['domain'],record['swaps'])]
        if same and any(r != record for r in same):
            raise PermissionError('inconsistent repeat test consumption; same output/settings/selection required')
        if not same:
            f.seek(0,2);f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        fcntl.flock(f,fcntl.LOCK_UN)
    return record


def test_consumption_event(root, reservation, event, **details):
    import fcntl
    ledger = data_path(Path(root)/'test_consumption.jsonl')
    row = {k:reservation[k] for k in ('selection_sha256','settings_sha256','model','domain','swaps','output')}
    row.update(event=event,**details)
    with ledger.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        fcntl.flock(f,fcntl.LOCK_UN)
