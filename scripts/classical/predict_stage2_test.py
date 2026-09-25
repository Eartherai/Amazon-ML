"""CL-002: score every saved test sidecar pair with stage-2 models and build matching TSVs.

All pair probabilities are saved (pairs.parquet) so decision variants never need
feature recomputation. Variants differ only in the model (classical / +EXP-045
logit), France policy (stage-2 vs frozen SUB-002 base rows) and optional
postprocessing. Output rows follow official test_source1 order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs/experiments/CL-002"
TEST = ROOT / "student_resource/dataset/test"
SUB002_V001 = ROOT / "outputs/submissions/SUB-002/validated-v001/matching_results.tsv"
HEADER = "source1_entity_id\tmatched_entity_ids\n"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def read_matching(path: Path) -> dict[str, list[str]]:
    out = {}
    with path.open() as f:
        assert f.readline() == HEADER
        for line in f:
            q, _, raw = line.rstrip("\n").partition("\t")
            out[q] = raw.split(",") if raw else []
    return out


def score_pairs(models: dict, out: Path) -> pl.DataFrame:
    if out.exists():
        return pl.read_parquet(out)
    frames = []
    for path in sorted((EXP / "features").glob("*.npz")):
        z = np.load(path)
        cols = {"q": z["q"], "t": z["t"], "base": z["X"][:, 0], "neural": z["neural"]}
        for name, (booster, meta) in models.items():
            X = z["X"] if meta["variant"] == "classical" else np.hstack([z["X"], z["neural"][:, None]])
            if X.shape[1] != len(meta["features"]):
                raise ValueError(f"feature width mismatch for {name}")
            cols[f"p_{name}"] = booster.predict(X, num_threads=8)
        frames.append(pl.DataFrame(cols).with_columns(pl.lit(path.stem).alias("shard")))
    df = pl.concat(frames)
    df.write_parquet(out, compression="zstd")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=EXP / "models")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    models = {}
    for variant in ("classical", "neural"):
        meta = json.loads((args.models / f"stage2-{variant}.json").read_text())
        booster = lgb.Booster(model_file=str(args.models / f"stage2-{variant}.txt"))
        if sha(args.models / f"stage2-{variant}.txt") != meta["model_sha256"]:
            raise ValueError("model hash mismatch")
        models[variant] = (booster, meta)
    df = score_pairs(models, EXP / f"pairs-{args.tag}.parquet")
    s1 = pl.read_csv(TEST / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    order, country = s1["entity_id"].to_list(), dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
    base = read_matching(SUB002_V001)
    if set(base) != set(order):
        raise ValueError("SUB-002 base universe differs from test S1")
    receipts = {}
    for variant, (_, meta) in models.items():
        thr = meta["threshold"]
        sel = df.filter(pl.col(f"p_{variant}") >= thr).select("q", "t")
        chosen = defaultdict(list)
        for q, t in sel.iter_rows():
            chosen[q].append(t)
        for france in ("stage2", "base"):
            name = f"CL-002-{variant}-fr{france}-{args.tag}"
            dest = ROOT / "outputs/submissions" / name
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / "matching_results.tsv"
            stats = defaultdict(lambda: {"s1": 0, "links": 0, "empty": 0, "changed_vs_sub002": 0})
            with path.open("w") as f:
                f.write(HEADER)
                for q in order:
                    c = country[q]
                    ids = base[q] if (france == "base" and c not in ("India", "US")) else sorted(set(chosen.get(q, [])))
                    f.write(q + "\t" + ",".join(ids) + "\n")
                    st = stats[c]
                    st["s1"] += 1
                    st["links"] += len(ids)
                    st["empty"] += int(not ids)
                    st["changed_vs_sub002"] += int(set(ids) != set(base[q]))
            receipts[name] = {"sha256": sha(path), "threshold": thr, "model_sha256": meta["model_sha256"],
                              "france_policy": france, "stats": {k: dict(v) for k, v in stats.items()}}
            (dest / "provenance.json").write_text(json.dumps({**receipts[name], "pairs_scored": len(df),
                "source_sidecar": "P5-SUB002-NGPU-85c999e8 192 SHA-receipted shards", "fold4": "CLOSED"}, indent=2))
            print(json.dumps({name: receipts[name]}), flush=True)
    (EXP / f"variants-{args.tag}.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
