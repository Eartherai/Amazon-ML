"""Exploratory OOF specialist for targets lacking an address.

The outer models never train on their held-out S1 fold or negatives owned by
that fold. Reported threshold sweeps are exploratory on the same OOF sample;
they are not a promotion gate. Fold 4 stays closed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.models.sub001_features import NAMES


ROOT = Path(__file__).resolve().parents[2]


def lists(path: Path, selected: set[str] | None = None) -> dict[str, set[str]]:
    with path.open(newline="") as stream:
        return {row["source1_entity_id"]: set(row["matched_entity_ids"].split(","))
                if row["matched_entity_ids"] else set()
                for row in csv.DictReader(stream, delimiter="\t")
                if selected is None or row["source1_entity_id"] in selected}


def score(truth: set[str], prediction: set[str]) -> float:
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    if not tp:
        return 0.0
    fp, fn = len(prediction - truth), len(truth - prediction)
    return 1.25 * tp / (1.25 * tp + fp + 0.25 * fn)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    base = pl.scan_parquet(str(ROOT / "outputs/oof/P4-B-001/features/part-*.parquet"))
    base = base.filter(pl.col("target_address_missing") == 1)
    numeric = pl.scan_parquet(str(ROOT / "artifacts/features/P4-NUMERIC-B-001/part-*.parquet"))
    frame = base.join(numeric, on=["source1_entity_id", "target_id"], how="left", validate="1:1").collect()
    if frame.select(pl.col(NAMES).null_count()).row(0) != tuple(0 for _ in NAMES):
        raise ValueError("Missing feature values")
    truth = lists(ROOT / "student_resource/dataset/train/train_ground_truth.tsv",
                  set(pl.read_parquet(ROOT / "artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet")["entity_id"]))
    baseline = lists(ROOT / "outputs/experiments/P4-NUMERIC-B-001/with_canonical_numeric/entity_predictions.tsv")
    if set(truth) != set(baseline):
        raise ValueError("Truth and baseline entity sets differ")
    qids = frame["source1_entity_id"].to_list()
    tids = frame["target_id"].to_list()
    labels = frame["label"].to_numpy()
    fold = frame["fold"].to_numpy()
    owner = frame["owner_fold"].to_numpy()
    X = frame.select(NAMES).to_numpy()
    predicted = np.zeros(len(frame), np.float32)
    fit_reports = []
    params = json.loads((ROOT / "configs/baselines/BASELINE-P4-001.yaml").read_text())["hyperparameters"]
    params.update(n_jobs=4, n_estimators=250, deterministic=True, force_col_wise=True)
    for held in (1, 2, 3):
        fit_mask = (fold != held) & ((owner == -1) | ((owner != held) & (owner != 4)))
        eval_mask = fold == held
        if np.any((labels == 1) & fit_mask & (owner != fold)):
            raise ValueError("Positive owner mismatch")
        model = lgb.LGBMClassifier(**params)
        model.fit(X[fit_mask], labels[fit_mask], feature_name=NAMES)
        predicted[eval_mask] = model.predict_proba(X[eval_mask])[:, 1]
        fit_reports.append({"held_fold": held, "fit_pairs": int(fit_mask.sum()),
                            "fit_positives": int(labels[fit_mask].sum()),
                            "eval_pairs": int(eval_mask.sum())})
    base_score = sum(score(truth[q], baseline[q]) for q in truth) / len(truth)
    result = {"scope": "Exploratory 20k outer OOF specialist; thresholds assessed on same OOF and require independent confirmation; Fold4 CLOSED", "base": base_score, "pairs": len(frame), "positives": int(labels.sum()), "fit_reports": fit_reports, "thresholds": []}
    for threshold in (0.5, 0.65, 0.75, 0.83, 0.9, 0.95):
        additions: dict[str, set[str]] = defaultdict(set)
        tp = fp = 0
        for q, t, y, prob in zip(qids, tids, labels, predicted, strict=True):
            if prob >= threshold and t not in baseline[q]:
                additions[q].add(t)
                tp += int(y)
                fp += int(not y)
        current = sum(score(truth[q], baseline[q] | additions[q]) for q in truth) / len(truth)
        result["thresholds"].append({"threshold": threshold, "macro_f0_5": current,
                                      "delta": current - base_score, "added_tp": tp, "added_fp": fp})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
