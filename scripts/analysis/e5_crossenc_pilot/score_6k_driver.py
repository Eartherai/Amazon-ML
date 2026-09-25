"""Score the fixed-6k top-12 pairs (72,000) with the frozen EXP-050 E5 pair classifier.

Runs inside one finite SageMaker Training job used purely as a batch-inference harness.
Reads no labels. Writes logits and a throughput report into /opt/ml/model so they are
packaged into the job's output model.tar.gz. Fold4 CLOSED (6k set is folds 2/3 only).
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path('/opt/ml/input/data')
OUT = Path('/opt/ml/model')
WORK = Path('/tmp/e5ckpt')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class PairModel(torch.nn.Module):
    def __init__(self, path: Path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(path, local_files_only=True)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(hidden.float()).squeeze(-1)


def score(model, tokenizer, rows, order, batch_size, amp=True):
    logits = np.full(len(rows), np.nan, dtype=np.float32)
    torch.cuda.synchronize()
    started = time.perf_counter()
    gpu_seconds = 0.0
    tokens = 0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([rows[i]['query_text'] for i in idx], [rows[i]['target_text'] for i in idx],
                        padding=True, truncation='longest_first', max_length=128, return_tensors='pt')
        tokens += int(enc['attention_mask'].sum())
        g0 = time.perf_counter()
        ids = enc['input_ids'].to('cuda', non_blocking=True)
        mask = enc['attention_mask'].to('cuda', non_blocking=True)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            out = model(ids, mask).float().cpu().numpy()
        gpu_seconds += time.perf_counter() - g0
        logits[np.asarray(idx)] = out
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    return logits, {'pairs': len(order), 'batch_size': batch_size, 'wall_seconds': wall,
                    'pairs_per_second_wall': len(order) / wall,
                    'gpu_forward_seconds': gpu_seconds,
                    'pairs_per_second_gpu_only': len(order) / gpu_seconds,
                    'nonpad_tokens': tokens}


def main() -> None:
    t0 = time.perf_counter()
    config = json.loads((ROOT / 'code/config.json').read_text())
    if config['fold4'] != 'CLOSED' or config['run_id'] != 'P5-E5-SCORE6K-001':
        raise ValueError('Wrong policy/run id')
    pairs_path = ROOT / 'data/pairs6k_e5.jsonl'
    if sha(pairs_path) != config['pairs_sha256']:
        raise ValueError('Pair file checksum mismatch')
    tars = sorted((ROOT / 'model').rglob('model.tar.gz'))
    if len(tars) != 1:
        raise ValueError(f'Expected one checkpoint tar, found {tars}')
    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tars[0]) as tf:
        tf.extractall(WORK, filter='data')
    found = sorted(WORK.rglob('COMPLETE.json'))
    if len(found) != 1:
        raise ValueError(f'Expected one COMPLETE.json, found {found}')
    ckpt = found[0].parent
    receipt = json.loads(found[0].read_text())
    if receipt['run_id'] != 'P5-E5-18K-001' or receipt['fold4'] != 'CLOSED' or receipt['train_pairs'] != 420_347:
        raise ValueError('Checkpoint provenance mismatch')
    enc_dir = ckpt / 'e5_encoder'
    for name, digest in receipt['checkpoint_files_sha256'].items():
        if sha(enc_dir / name) != digest:
            raise ValueError(f'Encoder file hash mismatch: {name}')
    if sha(ckpt / 'head.pt') != receipt['head_sha256']:
        raise ValueError('Head hash mismatch')
    rows = [json.loads(line) for line in pairs_path.open()]
    if len(rows) != 72_000 or [r['i'] for r in rows] != list(range(72_000)):
        raise ValueError('Pair inventory mismatch')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(enc_dir, local_files_only=True, use_fast=True)
    model = PairModel(enc_dir)
    model.head.load_state_dict(torch.load(ckpt / 'head.pt', map_location='cpu', weights_only=True))
    model.to('cuda').eval()
    setup_seconds = time.perf_counter() - t0
    lengths = np.array([len(r['query_text']) + len(r['target_text']) for r in rows])
    sorted_order = np.argsort(lengths, kind='stable').tolist()
    natural = list(range(len(rows)))
    # warm-up
    score(model, tokenizer, rows, natural[:2048], 512)
    logits, rep_sorted = score(model, tokenizer, rows, sorted_order, 512)
    if not np.isfinite(logits).all():
        raise ValueError('Non-finite logits')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'e5_logits_6k.tsv').open('w') as f:
        f.write('i\tq\tt\te5_logit\n')
        for r, v in zip(rows, logits):
            f.write(f"{r['i']}\t{r['q']}\t{r['t']}\t{float(v):.6f}\n")
    print(json.dumps({'written': True, 'sorted_b512': rep_sorted}), flush=True)
    extra = {}
    try:
        logits_nat, extra['natural_b512'] = score(model, tokenizer, rows, natural, 512)
        extra['sorted_vs_natural_max_abs_diff'] = float(np.abs(logits - logits_nat).max())
        extra['sorted_b1024'] = score(model, tokenizer, rows, sorted_order, 1024)[1]
        logits32, rep32 = score(model, tokenizer, rows, natural[:4096], 256, amp=False)
        extra['fp32_check'] = {**rep32, 'max_abs_diff_vs_fp16': float(np.abs(logits32[:4096] - logits[:4096]).max())}
    except Exception as exc:  # timing extras must never lose the main logits
        extra['extra_error'] = repr(exc)
    report = {'run_id': config['run_id'], 'fold4': 'CLOSED', 'checkpoint': receipt['run_id'],
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'pairs': len(rows), 'setup_seconds': setup_seconds,
              'sorted_b512': rep_sorted, **extra,
              'logit_mean': float(logits.mean()), 'logit_std': float(logits.std()),
              'logits_sha256': sha(OUT / 'e5_logits_6k.tsv')}
    (OUT / 'SCORE_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
