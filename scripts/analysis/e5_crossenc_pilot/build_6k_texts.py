"""Build EXP-050 E5 text pairs for the fixed-6k top-12 set (72,000 pairs).

Pair order and ids come from outputs/experiments/CL-003/eval6k-top12-v2.npz (verified
identical to EXP-044 pairs.jsonl). Texts are rebuilt from the raw train TSVs with the
exact reader and f-strings used to create the EXP-050 training pairs
(prepare_e5_large.py / prepare_scale.read_records: csv.DictReader, default quoting):
    query  = f'{name.strip()} | {address.strip()}'
    target = f'{tid[:2].lower()}: {name.strip()} | {address.strip()}'
No labels are written to the staged file. Fold4 is not touched (6k set is folds 2/3).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/Users/earther/Desktop/Amazon ML Challange')
OUT = ROOT / 'outputs/analysis/e5_crossenc_pilot'
EXP044 = Path('/Users/earther/.codex/worktrees/aml-neural-warroom/Amazon ML Challange/outputs/experiments/warroom_neural/EXP-044/pairs.jsonl')
csv.field_size_limit(sys.maxsize)


def read_records(path: Path, keep: set[str]) -> dict[str, tuple[str, str]]:
    found = {}
    with path.open(newline='') as src:
        for row in csv.DictReader(src, delimiter='\t'):
            if row['entity_id'] in keep:
                found[row['entity_id']] = (row['business_name'], row['business_address'])
    if found.keys() != keep:
        raise ValueError(f'Missing {len(keep - found.keys())} records from {path}')
    return found


def main() -> None:
    z = np.load(ROOT / 'outputs/experiments/CL-003/eval6k-top12-v2.npz', allow_pickle=False)
    q = z['q'].astype(str); t = z['t'].astype(str)
    if len(q) != 72_000 or len(set(q)) != 6_000:
        raise ValueError('Unexpected 6k inventory')
    folds = set()
    with EXP044.open() as src:
        for i, line in enumerate(src):
            r = json.loads(line)
            if r['q'] != q[i] or r['t'] != t[i]:
                raise ValueError('EXP-044 / eval6k order mismatch')
            folds.add(r['fold'])
    if folds - {2, 3}:
        raise ValueError('Non fold-2/3 row in fixed 6k')
    base = ROOT / 'student_resource/dataset/train'
    src1 = read_records(base / 'train_source1.tsv', set(q))
    tgt = read_records(base / 'train_source2.tsv', {x for x in t if x.startswith('S2-')})
    tgt.update(read_records(base / 'train_source3.tsv', {x for x in t if x.startswith('S3-')}))
    out = OUT / 'stage/data/pairs6k_e5.jsonl'
    with out.open('w') as f:
        for i, (qi, ti) in enumerate(zip(q, t)):
            qn, qa = src1[qi]; tn, ta = tgt[ti]
            f.write(json.dumps({'i': i, 'q': qi, 't': ti,
                                'query_text': f'{qn.strip()} | {qa.strip()}',
                                'target_text': f'{ti[:2].lower()}: {tn.strip()} | {ta.strip()}'},
                               ensure_ascii=False) + '\n')
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps({'pairs': len(q), 'queries': 6000, 'sha256': digest, 'path': str(out)}))


if __name__ == '__main__':
    main()
