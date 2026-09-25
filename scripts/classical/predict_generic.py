"""Score saved test stage-2 feature shards with one LightGBM stage-2 model and write matching TSVs.

--model: LightGBM text model; --meta: its JSON (features, threshold, model_sha256).
--with-neural appends the saved EXP-045 logit column (NaN where absent).
Writes France-stage2 and France-base variants plus all pair probabilities.
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
TEST = ROOT / "student_resource/dataset/test"
BASE = ROOT / "outputs/submissions/SUB-002/validated-v001/matching_results.tsv"
HEADER = "source1_entity_id\tmatched_entity_ids\n"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "outputs/experiments/CL-002/features")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--with-neural", action="store_true")
    ap.add_argument("--threshold", type=float)
    args = ap.parse_args()
    meta = json.loads(args.meta.read_text())
    if sha(args.model) != meta["model_sha256"]:
        raise ValueError("model hash mismatch")
    thr = args.threshold if args.threshold is not None else meta["threshold"]
    booster = lgb.Booster(model_file=str(args.model))
    frames = []
    for path in sorted(args.features.glob("*.npz")):
        z = np.load(path)
        X = z["X"] if not args.with_neural else np.hstack([z["X"], z["neural"][:, None]])
        frames.append(pl.DataFrame({"q": z["q"], "t": z["t"], "base": z["X"][:, 0], "p": booster.predict(X, num_threads=8)}))
    df = pl.concat(frames)
    out_dir = ROOT / "outputs/experiments/CL-003/test_probs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / f"{args.name}.parquet", compression="zstd")
    s1 = pl.read_csv(TEST / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    order, country = s1["entity_id"].to_list(), dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
    base = {}
    with BASE.open() as f:
        assert f.readline() == HEADER
        for line in f:
            q, _, raw = line.rstrip("\n").partition("\t")
            base[q] = raw.split(",") if raw else []
    chosen = defaultdict(list)
    for q, t in df.filter(pl.col("p") >= thr).select("q", "t").iter_rows():
        chosen[q].append(t)
    for france in ("stage2", "base"):
        name = f"{args.name}-fr{france}"
        dest = ROOT / "outputs/submissions" / name
        dest.mkdir(parents=True, exist_ok=False)
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
        prov = {"matching_sha256": sha(path), "threshold": thr, "model": str(args.model), "model_sha256": meta["model_sha256"],
                "france_policy": france, "with_neural": args.with_neural, "pairs_scored": len(df),
                "stats": {k: dict(v) for k, v in stats.items()}, "eval_6k": meta.get("eval_6k"), "fold4": "CLOSED",
                "source_sidecar": "P5-SUB002-NGPU-85c999e8 192 SHA-receipted shards (subset of SUB-002 candidates)"}
        (dest / "provenance.json").write_text(json.dumps(prov, indent=2))
        print(json.dumps({name: {"sha": prov["matching_sha256"][:12], "thr": thr, "stats": prov["stats"]}}), flush=True)


if __name__ == "__main__":
    main()
