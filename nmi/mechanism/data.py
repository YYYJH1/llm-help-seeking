"""Loader of the mechanism-only frozen splits (pilot, train, validation and test); never materializes old sealed holdout rows.

The splits are read from ``DATA_ROOT`` and verified against its ``manifest.json``. They were frozen by a result-blind key-generation step that is not part of this release, and
the frozen files are not released (see data/04_arc_patching/README.md).
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from nmi.data.keys import item_key, keylist_digests
from nmi.paths import HF_HUB
from nmi.data.registry import McItem

PROJECT=Path(__file__).resolve().parents[2]
DATA_ROOT=PROJECT/'results/mechanism_20261016/data'
MMLU_ROOT=Path(os.environ.get('NMI_MMLU_ROOT', HF_HUB/'datasets--edinburgh-dawg--mmlu-redux-2.0'/'snapshots'/'372ea425445d51e1ba1188c56e5e893f8138621f'))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(name: str, *, root=DATA_ROOT, allow_test: bool=False) -> list[McItem]:
    """Load verified frozen records; test is rejected unless explicitly requested."""
    if name not in ('pilot','train','validation','test'):raise ValueError('unknown split')
    if name=='test' and not allow_test:raise PermissionError('test requires allow_test=True after analysis choices freeze')
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if name=='test':
        selection=root/'frozen_selection.json'
        if not selection.is_file():raise PermissionError('test requires frozen_selection.json from dev/validation')
        frozen=json.loads(selection.read_text())
        if frozen.get('data_manifest_sha256')!=sha256_file(root/'manifest.json'):raise PermissionError('selection must bind data manifest hash')
        if not frozen.get('analysis_choices') or not frozen.get('development_evidence_sha256'):raise PermissionError('selection must bind analysis choices and dev/validation evidence hash')
        evidence=Path(frozen.get('development_evidence_path',''))
        if not evidence.is_file() or sha256_file(evidence)!=frozen['development_evidence_sha256']:raise PermissionError('development evidence file/hash must verify before test access')
        import pyarrow.parquet as pq
        locpath=root/'test_locators.json'
        if sha256_file(locpath)!=manifest['files'][locpath.name]:raise ValueError('locator hash mismatch')
        locators=json.loads(locpath.read_text());items=[]
        path=Path(next(k for k in manifest['source_files'] if '/test-' in k))
        if sha256_file(path)!=manifest['source_files'][str(path)]:raise ValueError('ARC source hash mismatch')
        table=pq.read_table(path)
        for loc in locators:
            r=table.slice(loc['row'],1).to_pylist()[0]; texts=r['choices']['text'];gt=r['choices']['label'].index(r['answerKey'])
            if item_key(r['question'],texts[gt])!=loc['item_key']:raise ValueError('test key mismatch')
            items.append(McItem(loc['item_key'],'mechanism_fresh_arc_test','arc_challenge',f'ARC-Challenge/{path.name}#{loc["row"]}',None,r['question'],tuple(texts),gt+1,texts[gt],None,loc['row']))
        if keylist_digests([x.item_key for x in items])[1]!=manifest['key_digests']['test']['sha256']:raise ValueError('test keys mismatch')
        return items
    p=root/f'{name}.jsonl'
    if sha256_file(p)!=manifest['files'][p.name]:raise ValueError('split content hash mismatch')
    items=[]
    for line in p.read_text().splitlines():
        r=json.loads(line);r['choices']=tuple(r['choices']);r['arrangement']=tuple(r['arrangement']) if r['arrangement'] else None;items.append(McItem(**r))
    keys=[x.item_key for x in items]
    if len(keys)!=manifest['counts'][name] or keylist_digests(keys)[1]!=manifest['key_digests'][name]['sha256']:raise ValueError('split keys/count mismatch')
    return items
