"""Evaluate target ownership resolution on 200k strictly out-of-fold entities.

Each held fold uses its EXP-033 100k model trained only on the other folds.
Every query's full EXP-032 candidate set is scored. Fold 4 stays closed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.models.sub001_features import NAMES


THRESHOLDS = {1: 0.83, 2: 0.79, 3: 0.83}


def entity_score(truth: set[str], pred: set[str]) -> float:
    if not truth:
        return float(not pred)
    tp = len(truth & pred)
    if not tp:
        return 0.0
    fp = len(pred - truth)
    fn = len(truth - pred)
    return 1.25 * tp / (1.25 * tp + fp + 0.25 * fn)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    started = time.perf_counter()
    feature_report = json.loads((a.features / "metrics.json").read_text())
    if feature_report["queries"] != 200_000 or feature_report["features"] != NAMES:
        raise ValueError("Wrong feature store")
    queries = pl.read_parquet(a.inputs / "labeled_queries.parquet", columns=["entity_id", "country", "fold"])
    if len(queries) != 200_000 or set(queries["fold"].to_list()) != {1, 2, 3}:
        raise ValueError("Unexpected query population")
    country = dict(queries.select("entity_id", "country").iter_rows())
    truth = {qid: set() for qid in queries["entity_id"]}
    for qid, tid in pl.read_parquet(a.inputs / "truth.parquet").iter_rows():
        truth[qid].add(tid)
    files = sorted(a.features.glob("features-*.parquet"))
    if len(files) != json.loads((a.features / "COMPLETE.json").read_text())["feature_parts"]:
        raise ValueError("Incomplete feature parts")
    scan = pl.scan_parquet([str(path) for path in files])
    predictions: dict[str, set[str]] = {qid: set() for qid in truth}
    probabilities: dict[tuple[str, str], float] = {}
    fold_reports = []
    for held in (1, 2, 3):
        eval_frame = scan.filter(pl.col("fold") == held).select(["source1_entity_id", "target_id", *NAMES]).collect()
        model = lgb.Booster(model_file=str(a.models / f"model-size100000-held{held}.txt"))
        if model.feature_name() != NAMES:
            raise ValueError("Model feature names differ")
        scores = model.predict(eval_frame.select(NAMES).to_numpy(), num_threads=a.threads)
        selected = np.flatnonzero(scores >= THRESHOLDS[held])
        qids = eval_frame["source1_entity_id"].to_list()
        tids = eval_frame["target_id"].to_list()
        for index in selected:
            qid, tid = qids[int(index)], tids[int(index)]
            predictions[qid].add(tid)
            probabilities[qid, tid] = float(scores[int(index)])
        fold_reports.append({"held_fold": held, "candidate_pairs": len(eval_frame),
                             "predicted_links": len(selected), "threshold": THRESHOLDS[held]})
        print(json.dumps({"fold": held, "pairs": len(eval_frame), "predicted": len(selected),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
        del eval_frame, scores, qids, tids, model
    owners: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (qid, tid), prob in probabilities.items():
        owners[tid].append((qid, prob))
    resolved = {qid: set(ids) for qid, ids in predictions.items()}
    conflict_targets = 0
    removed = 0
    removed_true = 0
    for tid, options in owners.items():
        if len(options) <= 1:
            continue
        conflict_targets += 1
        winner = max(options, key=lambda pair: (pair[1], pair[0]))[0]
        for qid, _ in options:
            if qid != winner:
                resolved[qid].remove(tid)
                removed += 1
                removed_true += int(tid in truth[qid])
    base_scores = np.array([entity_score(truth[qid], predictions[qid]) for qid in truth])
    new_scores = np.array([entity_score(truth[qid], resolved[qid]) for qid in truth])
    delta = new_scores - base_scores
    report = {"scope": "EXP-033 100k per-held-fold models; all 200k EXP-032 queries strictly OOF; original thresholds; Fold4 CLOSED",
              "entities": len(truth), "folds": fold_reports,
              "baseline_macro_f0_5": float(base_scores.mean()),
              "resolved_macro_f0_5": float(new_scores.mean()),
              "delta_macro_f0_5": float(delta.mean()),
              "delta_normal_95_ci": [float(delta.mean() - 1.96 * delta.std(ddof=1) / math.sqrt(len(delta))),
                                     float(delta.mean() + 1.96 * delta.std(ddof=1) / math.sqrt(len(delta)))],
              "conflict_targets": conflict_targets, "removed_links": removed,
              "removed_true_links": removed_true,
              "by_country": {}, "elapsed_seconds": time.perf_counter() - started}
    labels = np.array([country[qid] for qid in truth])
    for value in sorted(set(country.values())):
        mask = labels == value
        report["by_country"][value] = {"queries": int(mask.sum()),
            "baseline_macro_f0_5": float(base_scores[mask].mean()),
            "resolved_macro_f0_5": float(new_scores[mask].mean()),
            "delta_macro_f0_5": float(delta[mask].mean())}
    a.output.mkdir(parents=True)
    (a.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
