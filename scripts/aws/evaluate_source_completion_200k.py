"""Nested OOF test of adding one missing-source match per Source 1 entity.

The three saved EXP-033 100k models score their held-out folds. For each held
fold, the addition threshold is chosen only on the other two OOF folds. This
reuses existing EXP-032 candidate features; Fold 4 stays closed.
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


BASE_THRESHOLDS = {1: 0.83, 2: 0.79, 3: 0.83}
ADD_THRESHOLDS = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)


def f05(truth: set[str], pred: set[str]) -> float:
    if not truth:
        return float(not pred)
    tp = len(truth & pred)
    if not tp:
        return 0.0
    fp, fn = len(pred - truth), len(truth - pred)
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
    meta = {qid: (country, fold) for qid, country, fold in queries.iter_rows()}
    truth = {qid: set() for qid in meta}
    for qid, tid in pl.read_parquet(a.inputs / "truth.parquet").iter_rows():
        truth[qid].add(tid)
    files = sorted(a.features.glob("features-*.parquet"))
    if len(files) != json.loads((a.features / "COMPLETE.json").read_text())["feature_parts"]:
        raise ValueError("Incomplete feature parts")
    scan = pl.scan_parquet([str(path) for path in files])
    predicted = {qid: set() for qid in meta}
    pair_score: dict[tuple[str, str], float] = {}
    best: dict[tuple[str, str], tuple[float, str]] = {}
    fold_reports = []
    for held in (1, 2, 3):
        frame = scan.filter(pl.col("fold") == held).select(["source1_entity_id", "target_id", *NAMES]).collect()
        model = lgb.Booster(model_file=str(a.models / f"model-size100000-held{held}.txt"))
        if model.feature_name() != NAMES:
            raise ValueError("Model feature mismatch")
        scores = model.predict(frame.select(NAMES).to_numpy(), num_threads=a.threads)
        qids, tids = frame["source1_entity_id"].to_list(), frame["target_id"].to_list()
        selected = np.flatnonzero(scores >= 0.3)
        for index in selected:
            i = int(index)
            qid, tid, prob = qids[i], tids[i], float(scores[i])
            source = tid[:2]
            key = (qid, source)
            if key not in best or (prob, tid) > best[key]:
                best[key] = (prob, tid)
            if prob >= BASE_THRESHOLDS[held]:
                predicted[qid].add(tid)
                pair_score[qid, tid] = prob
        fold_reports.append({"held_fold": held, "candidate_pairs": len(frame),
                             "scored_above_0_3": len(selected),
                             "base_predicted_links": sum(len(predicted[qid]) for qid in meta if meta[qid][1] == held)})
        print(json.dumps({"fold": held, "pairs": len(frame), "above_0_3": len(selected),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
        del frame, scores, qids, tids, model
    qid_order = list(meta)
    base_f = np.array([f05(truth[qid], predicted[qid]) for qid in qid_order])
    candidate = {}
    delta = np.zeros(len(qid_order))
    candidate_prob = np.zeros(len(qid_order))
    for index, qid in enumerate(qid_order):
        present = predicted[qid]
        if not present:
            continue
        has_s2 = any(tid.startswith("S2-") for tid in present)
        has_s3 = any(tid.startswith("S3-") for tid in present)
        if has_s2 == has_s3:
            continue
        missing = "S3" if has_s2 else "S2"
        value = best.get((qid, missing))
        if value is None or value[1] in present:
            continue
        candidate[qid] = value
        candidate_prob[index] = value[0]
        delta[index] = f05(truth[qid], present | {value[1]}) - base_f[index]
    folds = np.array([meta[qid][1] for qid in qid_order])
    countries = np.array([meta[qid][0] for qid in qid_order])
    curves = {}
    for threshold in ADD_THRESHOLDS:
        active = candidate_prob >= threshold
        curves[str(threshold)] = {"macro_f0_5": float((base_f + delta * active).mean()),
                                  "delta": float((delta * active).mean()),
                                  "added_tp": sum(candidate[qid][1] in truth[qid]
                                                  for index, qid in enumerate(qid_order) if active[index] and qid in candidate),
                                  "added_fp": sum(candidate[qid][1] not in truth[qid]
                                                  for index, qid in enumerate(qid_order) if active[index] and qid in candidate)}
    selected_thresholds = {}
    nested_deltas = np.zeros(len(qid_order))
    for held in (1, 2, 3):
        fit = folds != held
        threshold = max(ADD_THRESHOLDS,
                        key=lambda t: (float((delta[fit] * (candidate_prob[fit] >= t)).mean()), t))
        selected_thresholds[held] = threshold
        mask = folds == held
        nested_deltas[mask] = delta[mask] * (candidate_prob[mask] >= threshold)
    chosen = {qid: candidate[qid] for index, qid in enumerate(qid_order)
              if qid in candidate and candidate_prob[index] >= selected_thresholds[meta[qid][1]]}
    added = {qid: set(predicted[qid]) for qid in qid_order}
    for qid, (prob, tid) in chosen.items():
        added[qid].add(tid)
        pair_score[qid, tid] = prob
    owners: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for qid, links in added.items():
        for tid in links:
            owners[tid].append((qid, pair_score[qid, tid]))
    removed = removed_true = 0
    conflict_targets = 0
    for tid, options in owners.items():
        if len(options) <= 1:
            continue
        conflict_targets += 1
        winner = max(options, key=lambda item: (item[1], item[0]))[0]
        for qid, _ in options:
            if qid != winner:
                added[qid].remove(tid)
                removed += 1
                removed_true += int(tid in truth[qid])
    combined_f = np.array([f05(truth[qid], added[qid]) for qid in qid_order])
    ci = lambda x: [float(x.mean() - 1.96 * x.std(ddof=1) / math.sqrt(len(x))),
                    float(x.mean() + 1.96 * x.std(ddof=1) / math.sqrt(len(x)))]
    report = {"scope": "All 200k EXP-032 queries strictly OOF under EXP-033 100k per-fold models; addition threshold cross-fit on other OOF folds; Fold4 CLOSED",
              "entities": len(qid_order), "folds": fold_reports, "baseline_macro_f0_5": float(base_f.mean()),
              "candidate_single_source_queries": len(candidate), "curves_exploratory": curves,
              "nested_thresholds": selected_thresholds,
              "nested_source_completion_macro_f0_5": float((base_f + nested_deltas).mean()),
              "nested_source_completion_delta": float(nested_deltas.mean()),
              "nested_delta_normal_95_ci": ci(nested_deltas),
              "nested_added_tp": sum(tid in truth[qid] for qid, (_, tid) in chosen.items()),
              "nested_added_fp": sum(tid not in truth[qid] for qid, (_, tid) in chosen.items()),
              "combined_with_owner_resolution_macro_f0_5": float(combined_f.mean()),
              "combined_delta": float((combined_f - base_f).mean()),
              "combined_delta_normal_95_ci": ci(combined_f - base_f),
              "conflict_targets_after_additions": conflict_targets,
              "owner_removed_links": removed, "owner_removed_true_links": removed_true,
              "by_country": {}, "elapsed_seconds": time.perf_counter() - started}
    for country in sorted(set(countries)):
        mask = countries == country
        report["by_country"][country] = {"queries": int(mask.sum()),
            "source_completion_delta": float(nested_deltas[mask].mean()),
            "combined_delta": float((combined_f[mask] - base_f[mask]).mean())}
    a.output.mkdir(parents=True)
    (a.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
