"""CL-003: top-12 held-fold OOF first-stage scores for every 200k feature-store S1.

Each fold-k Source 1 is scored only by EXP-033's `model-size100000-heldk`, which
never saw fold k, so every score is out of fold (the same construction as the
EXP-044 6k set, extended to all folds 1-3). Output feeds classical stage-2
training; the EXP-044 6k S1 are removed locally and kept as the evaluation set.
Fold4 CLOSED (absent from the 200k store).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

from src.models.sub001_features import NAMES


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    complete = json.loads((args.features / "COMPLETE.json").read_text())
    metrics = json.loads((args.features / "metrics.json").read_text())
    if complete["queries"] != 200_000 or complete["fold4"] != "CLOSED" or metrics["features"] != NAMES:
        raise ValueError("Wrong source feature store")
    files = sorted(args.features.glob("features-*.parquet"))
    if len(files) != complete["feature_parts"]:
        raise ValueError("Incomplete feature parts")
    scan = pl.scan_parquet([str(x) for x in files])
    args.output.mkdir()
    report = {"folds": {}}
    for held in (1, 2, 3):
        frame = scan.filter(pl.col("fold") == held).select(["source1_entity_id", "target_id", "label", *NAMES]).collect()
        model = lgb.Booster(model_file=str(args.models / f"model-size100000-held{held}.txt"))
        if model.feature_name() != NAMES:
            raise ValueError("Model feature mismatch")
        scores = model.predict(frame.select(NAMES).to_numpy(), num_threads=args.threads)
        frame = frame.select("source1_entity_id", "target_id", "label").with_columns(pl.Series("base_score", scores))
        del scores
        top = (frame.sort(["source1_entity_id", "base_score", "target_id"], descending=[False, True, False])
               .with_columns(pl.int_range(1, pl.len() + 1).over("source1_entity_id").alias("rank"))
               .filter(pl.col("rank") <= args.top)
               .with_columns(pl.lit(held).alias("fold")))
        path = args.output / f"top{args.top}-fold{held}.parquet"
        top.write_parquet(path, compression="zstd")
        report["folds"][held] = {"scored_pairs": len(frame), "top_pairs": len(top),
                                 "queries": top["source1_entity_id"].n_unique(), "sha256": sha(path),
                                 "seconds": time.perf_counter() - started}
        print(json.dumps({"held": held, **report["folds"][held]}), flush=True)
        del frame, top
    report.update({"scope": "EXP-033 100k held-fold OOF top-12 for all 200k feature-store S1; Fold4 CLOSED",
                   "seconds": time.perf_counter() - started})
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
