"""CL-005: stage-2 features for all 192 SUB-003 top-12 test score shards (score>=0.4 or top-12).

Inputs: SHA256-receipted P5-SUB002-NGPU-85c999e8 score shards (source1, target,
base_score, neural_logit), organizer test TSVs, and stage-2 models from
train_stage2.py. Test-side IDF uses only test Source 1 text (unsupervised
statistics on test are permitted by the organizer Q&A). Writes per-pair
probabilities for every variant so decisions can be re-derived without
recomputation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import os, importlib  # noqa: E402
FEATV = os.environ.get("FEATV", "v1")
F = importlib.import_module("stage2_features_v2" if FEATV == "v2" else "stage2_features")

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "student_resource/dataset/test"
EXP = ROOT / "outputs/experiments/CL-005"
S1: dict = {}
TG: dict = {}
IDF: tuple = ()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def test_idf(path: Path):
    if path.exists():
        d = json.loads(path.read_text())
        return d["name"], d["name_default"], d["addr"], d["addr_default"]
    ndf, adf = defaultdict(int), defaultdict(int)
    for name, addr in S1.values():
        for tok in set(F.norm(name).split()):
            ndf[tok] += 1
        for tok in set(F.address_tokens(F.norm(addr))):
            adf[tok] += 1
    n = len(S1)
    d = {"n": n, "name": {k: math.log(n / (v + 1)) for k, v in ndf.items() if v >= 2}, "name_default": math.log(n / 2),
         "addr": {k: math.log(n / (v + 1)) for k, v in adf.items() if v >= 2}, "addr_default": math.log(n / 2)}
    path.write_text(json.dumps(d))
    return d["name"], d["name_default"], d["addr"], d["addr_default"]


def work(shard: Path):
    out = EXP / f"features-{FEATV}" / (shard.name.replace("-scores.tsv.gz", ".npz"))
    if out.exists():
        return shard.name, "cached"
    import polars as pl
    frame = pl.read_parquet(EXP / "joined" / shard.name.replace("-scores.tsv.gz", ".parquet"))
    groups = defaultdict(list)
    neural, s1r, tr = {}, {}, {}
    for q, t, b, nlg, qn, qa, tn, ta in frame.iter_rows():
        groups[q].append((t, float(b)))
        neural[(q, t)] = float("nan")
        if q not in s1r:
            s1r[q] = F.Record(qn or "", qa or "")
        if t not in tr:
            tr[t] = F.Record(tn or "", ta or "")
    keys, X = F.build_matrix(groups, s1r, tr, *IDF)
    nl = np.array([neural[k] for k in keys], dtype=np.float32)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, q=np.array([k[0] for k in keys]), t=np.array([k[1] for k in keys]), X=X, neural=nl)
    tmp.rename(out)
    return shard.name, len(keys)


def main():
    global S1, TG, IDF
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=9)
    args = ap.parse_args()
    t0 = time.time()
    (EXP / f"features-{FEATV}").mkdir(parents=True, exist_ok=True)
    shards = sorted((EXP / "test_scores").glob("*-scores.tsv.gz"))
    if len(shards) != 192:
        raise ValueError(f"expected 192 shards, found {len(shards)}")
    receipts = json.loads((EXP / "score_receipts.json").read_text())
    for s in shards:
        if receipts[s.name] != sha(s):
            raise ValueError(f"receipt mismatch {s.name}")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    opts = "delim='\t',header=true,quote='',escape='',all_varchar=true"
    joined = EXP / "joined"
    joined.mkdir(exist_ok=True)
    if len(list(joined.glob("*.parquet"))) != 192:
        con.execute(f"CREATE TABLE s1 AS SELECT entity_id, business_name qn, business_address qa FROM read_csv('{TEST}/test_source1.tsv',{opts})")
        con.execute(f"""CREATE TABLE tg AS SELECT entity_id, business_name tn, business_address ta FROM read_csv('{TEST}/test_source2.tsv',{opts})
                        UNION ALL SELECT entity_id, business_name, business_address FROM read_csv('{TEST}/test_source3.tsv',{opts})""")
        for shard in shards:
            dest = joined / shard.name.replace("-scores.tsv.gz", ".parquet")
            if dest.exists():
                continue
            con.execute(f"""COPY (SELECT p.source1_entity_id q, p.target_id t, p.score::DOUBLE b, NULL::DOUBLE nl, s1.qn, s1.qa, tg.tn, tg.ta
                FROM read_csv('{shard}',delim='\t',header=true,quote='',escape='',all_varchar=true) p
                JOIN s1 ON s1.entity_id=p.source1_entity_id JOIN tg ON tg.entity_id=p.target_id) TO '{dest}' (FORMAT PARQUET)""")
        print(json.dumps({"joined": len(list(joined.glob('*.parquet'))), "join_s": round(time.time() - t0, 1)}), flush=True)
    if not (EXP / "test_s1_idf.json").exists():
        S1 = {r[0]: (r[1] or "", r[2] or "") for r in con.execute(
            f"SELECT entity_id,business_name,business_address FROM read_csv('{TEST}/test_source1.tsv',{opts})").fetchall()}
    IDF = test_idf(EXP / "test_s1_idf.json")
    S1 = {}
    con.close()
    print(json.dumps({"idf_s": round(time.time() - t0, 1)}), flush=True)
    ctx = mp.get_context("fork")
    done = 0
    with ctx.Pool(args.procs) as pool:
        for name, n in pool.imap_unordered(work, shards):
            done += 1
            if done % 12 == 0 or done == 192:
                print(json.dumps({"done": done, "last": name, "pairs": n, "elapsed_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
