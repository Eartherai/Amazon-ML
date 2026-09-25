"""Measure 20k/50k/100k classical LightGBM fits on a new fixed 15k OOF set.

Validation entities were never in the earlier 20k sample. For each outer fold,
fit entities come only from the other two folds and owner-safe target negatives.
Thresholds 0.83/0.79/0.83 were frozen on the old 20k before selecting this new
evaluation set. These are fixed-threshold learning-curve diagnostics; later
calibration may reselect thresholds on additional inner data, never these labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.evaluation import evaluate
from src.models.sub001_features import NAMES


THRESHOLDS = {1: 0.83, 2: 0.79, 3: 0.83}
SIZES = (20_000, 50_000, 100_000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def select_fit_ids(queries: pl.DataFrame, old_ids: set[str], held_fold: int, size: int) -> set[str]:
    """Nested deterministic samples balanced over the two training folds."""
    if size % 2 or held_fold not in THRESHOLDS:
        raise ValueError("Invalid sample size or fold")
    selected: set[str] = set()
    for fold in sorted(set(THRESHOLDS) - {held_fold}):
        ids = queries.filter(pl.col("fold") == fold)["entity_id"].to_list()
        ordered = sorted(ids, key=lambda value: (value not in old_ids, hashlib.sha256(value.encode()).digest(), value))
        if len(ordered) < size // 2:
            raise ValueError("Insufficient training entities")
        selected.update(ordered[:size // 2])
    if len(selected) != size:
        raise ValueError("Unexpected fit-entity count")
    return selected


def target_sets(eval_frame: pl.DataFrame, truth_frame: pl.DataFrame) -> dict[str, set[str]]:
    truth = {qid: set() for qid in eval_frame["entity_id"]}
    for qid, tid in truth_frame.iter_rows():
        if qid in truth:
            truth[qid].add(tid)
    return truth


def predict_sets(frame: pl.DataFrame, probabilities: np.ndarray, threshold: float,
                 query_ids: list[str]) -> dict[str, set[str]]:
    predicted = {qid: set() for qid in query_ids}
    for qid, tid, probability in zip(frame["source1_entity_id"], frame["target_id"], probabilities, strict=True):
        if probability >= threshold:
            predicted[qid].add(tid)
    return predicted


def upload_file(path: Path, bucket: str | None, prefix: str | None) -> dict | None:
    if bucket:
        from upload_verified import upload
        return upload(path, bucket, prefix.rstrip("/") + "/" + path.name)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--upload-bucket")
    parser.add_argument("--upload-prefix")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.threads < 1 or bool(args.upload_bucket) != bool(args.upload_prefix):
        raise ValueError("Invalid thread count or upload settings")
    args.output.mkdir(parents=True)
    checkpoint_receipts = []
    started = time.perf_counter()
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    for item in manifest["files"]:
        path = args.inputs / item["name"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Training input checksum mismatch: {path}")
    queries = pl.read_parquet(args.inputs / "labeled_queries.parquet")
    old = pl.read_parquet(args.inputs / "old_20k_ids.parquet")
    eval_frame = pl.read_parquet(args.inputs / "eval_queries.parquet")
    truth_frame = pl.read_parquet(args.inputs / "truth.parquet")
    if len(queries) != 200_000 or len(old) != 20_000 or len(eval_frame) != 15_000:
        raise ValueError("Unexpected training/evaluation population")
    if set(old["entity_id"]) & set(eval_frame["entity_id"]):
        raise ValueError("Old training IDs overlap new evaluation IDs")
    if set(eval_frame["fold"].unique().to_list()) != {1, 2, 3}:
        raise ValueError("Unexpected evaluation folds")
    feature_files = sorted(args.features.glob("features-*.parquet"))
    if not feature_files:
        raise FileNotFoundError("No feature parts")
    feature_report = json.loads((args.features / "metrics.json").read_text())
    if feature_report["queries"] != 200_000 or feature_report["features"] != NAMES:
        raise ValueError("Feature store manifest mismatch")
    base_params = json.loads(Path("configs/baselines/BASELINE-P4-001.yaml").read_text())["hyperparameters"]
    base_params["n_jobs"] = args.threads
    old_ids = set(old["entity_id"])
    results = {size: {"truth": {}, "predicted": {}, "folds": []} for size in SIZES}
    feature_scan = pl.scan_parquet([str(path) for path in feature_files])
    if feature_scan.collect_schema().names() != ["source1_entity_id", "target_id", "country", "fold", "owner_fold", "label", *NAMES]:
        raise ValueError("Unexpected feature columns or order")
    for held_fold in (1, 2, 3):
        outer_eval = eval_frame.filter(pl.col("fold") == held_fold)
        eval_ids = outer_eval["entity_id"].to_list()
        if len(eval_ids) != 5_000:
            raise ValueError("Unexpected outer evaluation count")
        truth = target_sets(outer_eval, truth_frame)
        evaluated = feature_scan.filter(pl.col("source1_entity_id").is_in(eval_ids)).collect()
        if set(evaluated["source1_entity_id"]) != set(eval_ids):
            raise ValueError("Missing candidates for evaluation query")
        valid_folds = sorted(set(THRESHOLDS) - {held_fold})
        validation_x = evaluated.select(NAMES).to_numpy()
        for size in SIZES:
            fit_ids = select_fit_ids(queries, old_ids, held_fold, size)
            fit = feature_scan.filter(pl.col("source1_entity_id").is_in(fit_ids) &
                                      pl.col("owner_fold").is_in([-1, *valid_folds])).collect()
            if fit["source1_entity_id"].n_unique() != size:
                raise ValueError("Fit rows do not cover every selected entity")
            if fit.filter((pl.col("label") == 1) & (~pl.col("owner_fold").is_in(valid_folds))).height:
                raise ValueError("Positive pair excluded by owner mask")
            fit_x = fit.select(NAMES).to_numpy()
            fit_y = fit["label"].to_numpy()
            model = lgb.LGBMClassifier(**base_params)
            fit_start = time.perf_counter()
            model.fit(fit_x, fit_y, feature_name=NAMES)
            fit_seconds = time.perf_counter() - fit_start
            model_path = args.output / f"model-size{size}-held{held_fold}.txt"
            model.booster_.save_model(str(model_path))
            receipt = upload_file(model_path, args.upload_bucket, args.upload_prefix)
            if receipt:
                checkpoint_receipts.append(receipt)
            probabilities = model.predict_proba(validation_x)[:, 1]
            predicted = predict_sets(evaluated, probabilities, THRESHOLDS[held_fold], eval_ids)
            fold_result = {"size": size, "held_fold": held_fold, "fit_entities": size,
                           "fit_pairs": len(fit), "fit_positives": int(fit_y.sum()),
                           "threshold": THRESHOLDS[held_fold], "fit_seconds": fit_seconds,
                           "validation_entities": len(eval_ids), "validation_pairs": len(evaluated),
                           "overall": evaluate(truth, predicted),
                           "by_country": {country: evaluate({qid: truth[qid] for qid in outer_eval.filter(pl.col("country") == country)["entity_id"]},
                                                           {qid: predicted[qid] for qid in outer_eval.filter(pl.col("country") == country)["entity_id"]})
                                          for country in sorted(outer_eval["country"].unique().to_list())},
                           "model_sha256": sha256(model_path)}
            report_path = args.output / f"size{size}-held{held_fold}.json"
            report_path.write_text(json.dumps(fold_result, indent=2) + "\n")
            receipt = upload_file(report_path, args.upload_bucket, args.upload_prefix)
            if receipt:
                checkpoint_receipts.append(receipt)
                (args.output / "checkpoint_receipts.json").write_text(json.dumps(checkpoint_receipts, indent=2) + "\n")
            results[size]["truth"].update(truth)
            results[size]["predicted"].update(predicted)
            results[size]["folds"].append(fold_result)
            print(json.dumps({"size": size, "held_fold": held_fold, "macro_f0_5": fold_result["overall"]["macro_f0_5"],
                              "fit_seconds": fit_seconds, "elapsed": time.perf_counter() - started}), flush=True)
            del fit_x, fit_y, fit, model, probabilities, predicted
        del validation_x, evaluated
    summary = {"experiment": "EXP-033", "scope": "Fixed 15k new entity OOF; Fold4 CLOSED",
               "threshold_policy": "0.83/0.79/0.83 frozen from old20k before new evaluation selection",
               "feature_store_sha256": sha256(args.features / "metrics.json"),
               "sizes": {}, "runtime_seconds": time.perf_counter() - started,
               "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    for size in SIZES:
        truth = results[size]["truth"]
        predicted = results[size]["predicted"]
        if len(truth) != 15_000:
            raise ValueError("Incomplete pooled evaluation")
        summary["sizes"][str(size)] = {"overall": evaluate(truth, predicted),
                                       "by_country": {country: evaluate({qid: truth[qid] for qid in eval_frame.filter(pl.col("country") == country)["entity_id"]},
                                                                       {qid: predicted[qid] for qid in eval_frame.filter(pl.col("country") == country)["entity_id"]})
                                                      for country in sorted(eval_frame["country"].unique().to_list())},
                                       "folds": results[size]["folds"]}
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "COMPLETE.json").write_text(json.dumps({"experiment": "EXP-033", "sizes": list(SIZES),
                                                  "validation_queries": 15_000, "models": 9,
                                                  "fold4": "CLOSED"}, indent=2) + "\n")
    print(json.dumps({"complete": True, "scores": {size: summary["sizes"][str(size)]["overall"]["macro_f0_5"]
                                                for size in SIZES}}), flush=True)


if __name__ == "__main__":
    main()
