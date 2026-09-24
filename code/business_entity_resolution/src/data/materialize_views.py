"""Bounded, resumable Parquet foundation with lossless raw text and derived views."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
import polars as pl
from src.preprocessing import VARIANTS, normalize, config_hash

VERSION = '1.0.0'
RAW_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def derive(frame: pl.DataFrame) -> pl.DataFrame:
    """Preserve raw strings; Python Unicode NFC/casefold plus native LMN filtering."""
    expressions = []
    for raw, prefix in [('business_name', 'name'), ('business_address', 'address')]:
        text = pl.col(raw)
        folded = text.fill_null('').map_elements(
            lambda s: unicodedata.normalize('NFC', s).casefold(), return_dtype=pl.String)
        light = folded.str.replace_all(r'[^\p{L}\p{M}\p{N}]+', ' ').str.strip_chars(' ')
        expressions.extend([
            text.is_null().alias(prefix + '_is_null'),
            (text.fill_null('') == '').alias(prefix + '_is_empty'),
            text.fill_null('').map_elements(lambda s: not s or s.isspace(), return_dtype=pl.Boolean).alias(prefix + '_is_blank'),
            light.alias(prefix + '_light'),
            text.fill_null('').str.extract_all(r'\p{Nd}+').alias(prefix + '_numeric_tokens'),
        ])
    return frame.with_columns(expressions)


def regenerate(value: str | None, view: str, field: str = 'name') -> str:
    """Regenerate an optional view without learned dictionaries or external data."""
    if view in {'legal_map', 'address_map'}:
        raise ValueError('Learned mapping views require a separate validated versioned map')
    return normalize(value, VARIANTS[view], field)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def run(config_path: Path, data_root: Path, output: Path, sources: list[str] | None = None) -> dict:
    config = json.loads(config_path.read_text())
    identity = config_hash(config)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / 'identity.json'
    identity_doc = {'config_hash': identity, 'module_sha256': sha256(Path(__file__)),
                    'preprocessing_sha256': sha256(Path(__file__).parents[1] / 'preprocessing.py')}
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity_doc:
        raise ValueError('Output directory belongs to different code/config; choose a new version')
    if not identity_path.exists():
        atomic_json(identity_path, identity_doc)
    chosen = sources or [f'{split}_source{i}' for split in ('train', 'test') for i in (1, 2, 3)]
    all_manifests = []
    for source in chosen:
        if source not in [f'{s}_source{i}' for s in ('train', 'test') for i in (1, 2, 3)]:
            raise ValueError(source)
        input_path = data_root / source.split('_')[0] / (source + '.tsv')
        dest = output / source
        dest.mkdir(exist_ok=True)
        manifest_path = dest / 'manifest.json'
        input_hash = sha256(input_path)
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if previous and (previous['input_sha256'] != input_hash or previous['config_hash'] != identity):
            raise ValueError('Changed input/config cannot resume existing shards')
        if previous and previous['complete']:
            for shard in previous['shards']:
                if sha256(dest / shard['file']) != shard['sha256']:
                    raise ValueError('Corrupted completed shard')
            all_manifests.append(previous)
            print(json.dumps({'source': source, 'resumed_complete': True}), flush=True)
            continue
        started = time.perf_counter()
        manifest = {'source': source, 'input_path': str(input_path.resolve()), 'input_sha256': input_hash,
            'input_bytes': input_path.stat().st_size, 'config_hash': identity, 'complete': False,
            'shards': [], 'created_utc': datetime.now(timezone.utc).isoformat(), 'rows': 0}
        reader = pl.scan_csv(input_path, separator='\t', schema_overrides={c:pl.String for c in RAW_COLUMNS},
            empty_string_is_null=False, infer_schema_length=0, low_memory=True).collect_batches(
                chunk_size=config['rows_per_shard'], maintain_order=True, engine='streaming')
        pending = []
        pending_rows = 0
        index = 0
        def write_chunk(raw: pl.DataFrame) -> None:
            nonlocal index
            if shutil.disk_usage(output).free < config['minimum_free_gib'] * 1024**3:
                raise RuntimeError('Disk reserve reached; source remains resumable')
            path = dest / f'part-{index:05d}.parquet'
            old = (previous['shards'][index] if previous and index < len(previous['shards']) else None)
            if old and path.exists():
                if sha256(path) != old['sha256'] or old['rows'] != raw.height:
                    raise ValueError('Resume shard mismatch')
                shard = old
            else:
                if path.exists():
                    raise ValueError('Unmanifested existing shard; inspect before continuing')
                begin = time.perf_counter()
                transformed = derive(raw)
                temp = path.with_suffix('.parquet.tmp')
                transformed.write_parquet(temp, compression='zstd', compression_level=3, row_group_size=100000)
                # Verify round-trip every raw record, including blank strings and Unicode.
                roundtrip = pl.read_parquet(temp)
                if not raw.equals(roundtrip.select(RAW_COLUMNS), null_equal=True):
                    raise AssertionError('Raw round-trip failed')
                if roundtrip.height != raw.height:
                    raise AssertionError('Row count changed')
                temp.replace(path)
                shard = {'file':path.name, 'rows':raw.height, 'bytes':path.stat().st_size,
                    'logical_bytes':transformed.estimated_size(), 'sha256':sha256(path),
                    'elapsed_seconds':time.perf_counter()-begin,
                    'schema':{k:str(v) for k,v in transformed.schema.items()}}
            manifest['shards'].append(shard)
            manifest['rows'] += raw.height
            atomic_json(manifest_path, manifest)
            print(json.dumps({'source':source, 'shard':index, 'rows':manifest['rows'], 'bytes':shard['bytes']}), flush=True)
            index += 1
        for batch in reader:
            if batch.height:
                pending.append(batch)
                pending_rows += batch.height
                if pending_rows >= config['rows_per_shard']:
                    combined = pl.concat(pending)
                    while combined.height >= config['rows_per_shard']:
                        write_chunk(combined.head(config['rows_per_shard']))
                        combined = combined.slice(config['rows_per_shard'])
                    pending = [combined] if combined.height else []
                    pending_rows = combined.height
        if pending_rows:
            write_chunk(pl.concat(pending))
        manifest.update(complete=True, elapsed_seconds=time.perf_counter()-started,
            total_bytes=sum(s['bytes'] for s in manifest['shards']),
            peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3 if platform.system()=='Darwin' else 1024**2))
        atomic_json(manifest_path, manifest)
        all_manifests.append(manifest)
    result = {'version': VERSION, 'config':config, **identity_doc, 'sources':all_manifests,
        'rows':sum(s['rows'] for s in all_manifests), 'bytes':sum(s['total_bytes'] for s in all_manifests),
        'python':platform.python_version(), 'unicode':unicodedata.unidata_version, 'polars':pl.__version__,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'raw_roundtrip_verified':True, 'learned_parameters':False,
        'created_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(output / 'manifest.json', result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/preprocessing/FOUNDATION-001.json'))
    p.add_argument('--data-root',type=Path,default=Path('student_resource/dataset'))
    p.add_argument('--output',type=Path,default=Path('artifacts/processed/prep-v001'))
    p.add_argument('--sources',nargs='+')
    args=p.parse_args()
    run(args.config,args.data_root,args.output,args.sources)

if __name__ == '__main__':
    main()
