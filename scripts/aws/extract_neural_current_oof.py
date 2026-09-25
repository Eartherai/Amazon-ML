"""Extract current LightGBM top-12 scores for a disjoint neural confirmation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.models.sub001_features import NAMES


THRESHOLDS = {2: 0.79, 3: 0.83}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    selected = pl.read_parquet(args.selection / "selected_queries.parquet")
    if len(selected) != 6_000 or selected["entity_id"].n_unique() != 6_000 or set(selected["fold"]) != {2, 3}:
        raise ValueError("Wrong selected query inventory")
    selection_manifest = json.loads((args.selection / "manifest.json").read_text())
    item = selection_manifest["files"][0]
    if item["bytes"] != (args.selection / item["name"]).stat().st_size or item["sha256"] != sha(args.selection / item["name"]):
        raise ValueError("Selection checksum mismatch")
    complete = json.loads((args.features / "COMPLETE.json").read_text())
    metrics = json.loads((args.features / "metrics.json").read_text())
    if complete["queries"] != 200_000 or complete["fold4"] != "CLOSED" or metrics["features"] != NAMES:
        raise ValueError("Wrong source feature store")
    files = sorted(args.features.glob("features-*.parquet"))
    if len(files) != complete["feature_parts"]:
        raise ValueError("Incomplete feature parts")
    scan = pl.scan_parquet([str(x) for x in files])
    qids = set(selected["entity_id"])
    truth = {qid: set() for qid in qids}
    for qid, tid in pl.read_parquet(args.labels / "truth.parquet").iter_rows():
        if qid in truth:
            truth[qid].add(tid)
    baseline = {qid: set() for qid in qids}
    frames = []
    for held in (2, 3):
        frame = scan.filter((pl.col("fold") == held) & pl.col("source1_entity_id").is_in(qids))\
            .select(["source1_entity_id", "target_id", "label", *NAMES]).collect()
        model = lgb.Booster(model_file=str(args.models / f"model-size100000-held{held}.txt"))
        if model.feature_name() != NAMES:
            raise ValueError("Model feature mismatch")
        scores = model.predict(frame.select(NAMES).to_numpy(), num_threads=args.threads)
        frame = frame.select("source1_entity_id", "target_id", "label").with_columns(pl.Series("base_score", scores))
        for qid, tid, _, score in frame.iter_rows():
            if score >= THRESHOLDS[held]:
                baseline[qid].add(tid)
        top = frame.sort(["source1_entity_id", "base_score", "target_id"], descending=[False, True, False])\
            .group_by("source1_entity_id", maintain_order=True).head(12)
        top = top.join(selected.rename({"entity_id": "source1_entity_id"}),
                       on="source1_entity_id", how="inner", validate="m:1")
        frames.append(top)
        print(json.dumps({"held": held, "scored_pairs": len(frame), "top_pairs": len(top),
                          "seconds": time.perf_counter() - started}), flush=True)
    result = pl.concat(frames).sort("source1_entity_id", "base_score", "target_id",
                                     descending=[False, True, False])
    if result["source1_entity_id"].n_unique() != 6_000:
        raise ValueError("Missing selected query in feature store")
    args.output.mkdir()
    result.write_parquet(args.output / "top12.parquet", compression="zstd")
    (args.output / "baseline.json").write_text(json.dumps({q: sorted(v) for q, v in baseline.items()}, sort_keys=True))
    (args.output / "truth.json").write_text(json.dumps({q: sorted(v) for q, v in truth.items()}, sort_keys=True))
    def f05(t: set[str], pred: set[str]) -> float:
        if not t:
            return float(not pred)
        tp = len(t & pred)
        return 1.25 * tp / (1.25 * tp + len(pred - t) + .25 * len(t - pred)) if tp else 0.
    macro = sum(f05(truth[q], baseline[q]) for q in qids) / len(qids)
    report = {"scope": "200k current 51-feature LightGBM held-fold OOF; 6000 disjoint S1; Fold4 CLOSED",
              "queries": len(qids), "top_pairs": len(result), "base_macro_f0_5": macro,
              "baseline_links": sum(map(len, baseline.values())), "seconds": time.perf_counter() - started,
              "files": {x.name: {"sha256": sha(x), "bytes": x.stat().st_size}
                        for x in args.output.iterdir() if x.is_file()}}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
