"""Strict TSV ID-list parsing and deterministic submission writing."""
from pathlib import Path
from collections.abc import Iterable, Mapping
import hashlib

def parse_id_list(text:str) -> set[str]:
    if text=='': return set()
    parts=text.split(',')
    if any(p!=p.strip() or not p or not p.startswith(('S2-','S3-')) for p in parts):
        raise ValueError('Invalid target ID list')
    if len(parts)!=len(set(parts)): raise ValueError('Duplicate target ID')
    return set(parts)

def read_lists(path:Path,column:str='matched_entity_ids') -> dict[str,set[str]]:
    out={}
    with path.open(encoding='utf-8',newline='') as f:
        if f.readline().rstrip('\r\n')!=f'source1_entity_id\t{column}': raise ValueError('Invalid TSV header')
        for line in f:
            cells=line.rstrip('\r\n').split('\t')
            if len(cells)!=2: raise ValueError('Expected exactly two TSV columns')
            key,values=cells
            if not key.startswith('S1-') or key!=key.strip(): raise ValueError('Invalid source ID')
            if key in out: raise ValueError('Duplicate S1 row')
            out[key]=parse_id_list(values)
    return out

def validate_maps(required:set[str],valid_targets:set[str],matches:Mapping[str,set[str]],candidates:Mapping[str,set[str]]) -> None:
    """Unlike the official helper, membership and ID checks are hard failures."""
    if required!=matches.keys() or required!=candidates.keys(): raise ValueError('S1 coverage mismatch')
    for key in required:
        if not matches[key]<=candidates[key]: raise ValueError('Match missing from scored candidates')
        if not candidates[key]<=valid_targets: raise ValueError('Unknown candidate target')

def write_lists(path:Path,rows:Mapping[str,set[str]],column:str) -> None:
    """Exclusive create prevents accidental overwrite of an experiment result."""
    with path.open('x',encoding='utf-8',newline='') as f:
        f.write(f'source1_entity_id\t{column}\n')
        for key in sorted(rows): f.write(key+'\t'+','.join(sorted(rows[key]))+'\n')

def shard_for(entity_id:str,n_shards:int) -> int:
    if n_shards<1: raise ValueError('n_shards must be positive')
    return int.from_bytes(hashlib.sha256(entity_id.encode()).digest()[:8],'big')%n_shards

def merge_shards(parts:Iterable[Mapping[str,set[str]]],required:set[str]) -> dict[str,set[str]]:
    result={}
    for part in parts:
        if result.keys() & part.keys(): raise ValueError('Duplicate S1 across shards')
        result.update(part)
    if result.keys()!=required: raise ValueError('Merged coverage mismatch')
    return dict(sorted(result.items()))
