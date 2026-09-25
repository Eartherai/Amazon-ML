"""Offline license/size audit of the EXP-045 cross-encoder and EXP-050 E5 checkpoints.

Reads only safetensors headers (no weight tensors are loaded), HuggingFace local
download metadata and local config/manifest files. No network access. Writes one
JSON evidence file under outputs/analysis/model-license-audit/.

License values below are the publisher metadata recorded from huggingface.co
model API/README at the pinned revisions (fetched 2026-09-25); they are copied
into the evidence file as constants, not re-fetched here.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/Users/earther/Desktop/Amazon ML Challange')
WARROOM = Path('/Users/earther/.codex/worktrees/aml-neural-warroom/Amazon ML Challange/outputs/experiments/warroom_neural')
GPU = Path('/Users/earther/.codex/worktrees/aml-gpu-rerank/Amazon ML Challange')
E5_SNAPSHOT = Path('/Users/earther/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots/'
                   '614241f622f53c4eeff9890bdc4f31cfecc418b3')
OUT = ROOT / 'outputs/analysis/model-license-audit/checkpoint_audit.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def header_params(path: Path) -> dict:
    with path.open('rb') as fp:
        size = struct.unpack('<Q', fp.read(8))[0]
        header = json.loads(fp.read(size))
    header.pop('__metadata__', None)
    by_dtype: Counter = Counter()
    for spec in header.values():
        by_dtype[spec['dtype']] += math.prod(spec['shape'])
    return {'tensors': len(header), 'by_dtype': dict(by_dtype), 'total_values': sum(by_dtype.values()),
            'float_params': sum(v for k, v in by_dtype.items() if k.startswith(('F', 'BF'))),
            'sha256': sha256(path), 'bytes': path.stat().st_size}


def hf_download_metadata(model_dir: Path) -> dict:
    out = {}
    for meta in sorted((model_dir / '.cache/huggingface/download').glob('*.metadata')):
        lines = meta.read_text().splitlines()
        out[meta.name.removesuffix('.metadata')] = {'commit': lines[0], 'etag': lines[1]}
    return out


def main() -> None:
    ce_base = WARROOM / 'EXP-042/model'
    ce_fold1 = WARROOM / 'EXP-042/finetuned_fold1'
    ce_045 = WARROOM / 'EXP-045/finetuned_fold1_scale'
    staged = json.loads((GPU / 'artifacts/cloud/phase5/exp050-e5-large-v001/code/config.json').read_text())
    e5_hashes = {name: sha256(E5_SNAPSHOT / name) for name in staged['model_sha256']}
    e5_params = header_params(E5_SNAPSHOT / 'model.safetensors')
    hidden = json.loads((E5_SNAPSHOT / 'config.json').read_text())['hidden_size']
    e5_head = hidden + 1  # torch.nn.Linear(hidden_size, 1) in train_e5_driver.PairModel
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'fold4': 'CLOSED',
        'network': 'none (local headers/metadata only)',
        'exp045_cross_encoder': {
            'model_id': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
            'revision': '1427fd652930e4ba29e8149678df786c240d8825',
            'weights_license_publisher': 'apache-2.0',
            'hf_base_model_tag': 'nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large',
            'hf_datasets_tag': 'unicamp-dl/mmarco',
            'local_download_metadata': hf_download_metadata(ce_base),
            'base_config_name_or_path': json.loads((ce_base / 'config.json').read_text()).get('_name_or_path'),
            'base_weights': header_params(ce_base / 'model.safetensors'),
            'exp042_finetuned_fold1': header_params(ce_fold1 / 'model.safetensors'),
            'exp045_finetuned_fold1_scale': header_params(ce_045 / 'model.safetensors'),
            'exp045_train_manifest': json.loads((WARROOM / 'EXP-045/train_manifest.json').read_text()),
            'exp042_finetune_manifest': json.loads((WARROOM / 'EXP-042/finetune_manifest.json').read_text()),
        },
        'exp050_e5': {
            'model_id': 'intfloat/multilingual-e5-small',
            'revision': '614241f622f53c4eeff9890bdc4f31cfecc418b3',
            'note': 'launch_e5_large.py / exp050-e5-large-v001 load the multilingual-e5-small snapshot; '
                    '"large" names the 18k-S1 training scale, not intfloat/multilingual-e5-large',
            'weights_license_publisher': 'mit',
            'snapshot_dir': str(E5_SNAPSHOT),
            'encoder_weights': e5_params,
            'pair_head_params': e5_head,
            'pair_model_float_params': e5_params['float_params'] + e5_head,
            'staged_config_model_revision': staged['model_revision'],
            'staged_sha256_match': {k: e5_hashes[k] == v for k, v in staged['model_sha256'].items()},
            'staged_train_queries': staged['train_queries'],
            'staged_train_pairs': staged['train_pairs'],
            'staged_data_sha256': staged['data_sha256'],
        },
        'shared_tokenizer_sentencepiece_sha256': {
            'cross_encoder': sha256(ce_base / 'sentencepiece.bpe.model'),
            'e5_small': sha256(E5_SNAPSHOT / 'sentencepiece.bpe.model'),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=False) + '\n')
    print(json.dumps({'written': str(OUT),
                      'ce_float_params': report['exp045_cross_encoder']['exp045_finetuned_fold1_scale']['float_params'],
                      'e5_pair_float_params': report['exp050_e5']['pair_model_float_params'],
                      'e5_staged_match': all(report['exp050_e5']['staged_sha256_match'].values())}))


if __name__ == '__main__':
    main()
