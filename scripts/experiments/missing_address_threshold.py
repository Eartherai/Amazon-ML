"""Nested missing-address threshold trial on frozen 20k NUMERIC-V2 candidates.

Only inner OOF from an outer training partition chooses its missing-address
threshold. The outer fold is used once to compare the selected rule with the
frozen global-threshold baseline. Fold4 is never read.
"""
import argparse
import hashlib
import json
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.models.phase4_oof import NAMES as BASE, fitting_mask
from src.models.phase4_numeric import NAMES as EXTRA


def score(frame: pl.DataFrame, query: pl.DataFrame, prob: np.ndarray,
          global_threshold: float, missing_threshold: float) -> dict:
    ids = query["entity_id"].to_list()
    lookup = {entity_id: i for i, entity_id in enumerate(ids)}
    idx = np.fromiter((lookup[x] for x in frame["source1_entity_id"]),
                      dtype=np.int32, count=frame.height)
    missing = ((frame["query_address_missing"].to_numpy() > 0) |
               (frame["target_address_missing"].to_numpy() > 0))
    keep = prob >= np.where(missing, missing_threshold, global_threshold)
    pred = np.bincount(idx[keep], minlength=len(ids))
    tp = np.bincount(idx[keep], weights=frame["label"].to_numpy()[keep], minlength=len(ids))
    true_count = query["n_matches"].to_numpy()
    entity = np.divide(1.25 * tp, pred + 0.25 * true_count,
                       out=np.ones(len(ids)), where=(pred + 0.25 * true_count) > 0)
    countries = query["country"].to_numpy()
    return {"entities": len(ids), "macro_f0_5": float(entity.mean()),
            "precision": float(tp.sum() / pred.sum()) if pred.sum() else 1.0,
            "recall": float(tp.sum() / true_count.sum()) if true_count.sum() else 1.0,
            "tp": int(tp.sum()), "predicted_pairs": int(pred.sum()),
            "by_country": {country: float(entity[countries == country].mean())
                           for country in sorted(set(countries))},
            "per_entity_scores": entity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path("configs/experiments/EXP-029.json")
    config = json.loads(config_path.read_text())
    started = time.perf_counter()
    q = pl.read_parquet("artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet")
    if q.height != 20000 or set(q["fold"].unique()) != {1, 2, 3}:
        raise ValueError("Unexpected validation population; Fold4 must remain closed")
    base = pl.read_parquet("outputs/oof/P4-B-001/features/part-*.parquet")
    numeric = pl.read_parquet("artifacts/features/P4-NUMERIC-B-001/part-*.parquet")
    frame = base.join(numeric, on=["source1_entity_id", "target_id"], how="left", validate="1:1")
    if frame.height != 3943627 or frame.select(pl.any_horizontal(pl.col(EXTRA).is_null()).any()).item():
        raise ValueError("Incomplete frozen features")
    names = BASE + EXTRA
    x = frame.select(names).to_numpy()
    y = frame["label"].to_numpy()
    fold = frame["fold"].to_numpy()
    owner = frame["owner_fold"].to_numpy()
    params = json.loads(Path("configs/baselines/BASELINE-P4-001.yaml").read_text())["hyperparameters"]
    params["n_jobs"] = 1
    frozen_metrics = json.loads(Path("outputs/experiments/P4-NUMERIC-B-001/with_canonical_numeric/metrics.json").read_text())
    report = {"experiment": config["experiment_id"], "timestamp": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "fold4": "CLOSED", "folds": []}
    args.output.mkdir(parents=True, exist_ok=False)
    for held in [1, 2, 3]:
        allowed = [v for v in [1, 2, 3] if v != held]
        inner = np.full(len(frame), np.nan)
        fit_rows = []
        for inner_held in allowed:
            fit_fold = [v for v in allowed if v != inner_held]
            mask = fitting_mask(fold, owner, fit_fold)
            if np.any((y == 1) & (fold == fit_fold[0]) & ~mask):
                raise AssertionError("Ownership filter excluded a positive")
            model = lgb.LGBMClassifier(**params)
            model.fit(x[mask], y[mask], feature_name=names)
            validation_mask = fold == inner_held
            inner[validation_mask] = model.booster_.predict(x[validation_mask], num_threads=1)
            fit_rows.append(int(mask.sum()))
        training_mask = np.isin(fold, allowed)
        if np.isnan(inner[training_mask]).any():
            raise ValueError("Incomplete inner OOF scores")
        training_frame = frame.filter(pl.col("fold").is_in(allowed))
        training_query = q.filter(pl.col("fold").is_in(allowed))
        global_threshold = config["outer_baseline_thresholds"][str(held)]
        grid = sorted(set([float(v) for v in config["missing_threshold_grid"]] + [global_threshold]))
        trials = []
        for threshold in grid:
            metrics = score(training_frame, training_query, inner[training_mask], global_threshold, threshold)
            metrics.pop("per_entity_scores")
            trials.append({"missing_threshold": threshold, **metrics})
        # Deterministic complexity tie-break: prefer the baseline threshold.
        selected = max(trials, key=lambda row: (row["macro_f0_5"],
                                                -abs(row["missing_threshold"] - global_threshold),
                                                row["missing_threshold"]))["missing_threshold"]
        outer_mask = fold == held
        outer_model = lgb.Booster(model_file=str(Path(
            f"outputs/experiments/P4-NUMERIC-B-001/with_canonical_numeric/fold-{held}.txt")))
        if outer_model.feature_name() != names:
            raise ValueError("Frozen outer model feature order mismatch")
        outer_prob = outer_model.predict(x[outer_mask], num_threads=1)
        outer_frame = frame.filter(pl.col("fold") == held)
        outer_query = q.filter(pl.col("fold") == held)
        baseline = score(outer_frame, outer_query, outer_prob, global_threshold, global_threshold)
        proposed = score(outer_frame, outer_query, outer_prob, global_threshold, selected)
        expected = next(row for row in frozen_metrics["folds"] if row["fold"] == held)
        if abs(baseline["macro_f0_5"] - expected["scores"]["overall"]["macro_f0_5"]) > 1e-7:
            raise ValueError("Frozen baseline reproduction failed")
        baseline_entity = baseline.pop("per_entity_scores")
        proposed_entity = proposed.pop("per_entity_scores")
        np.save(args.output / f"fold-{held}-paired-delta.npy", proposed_entity - baseline_entity)
        fold_result = {"held_fold": held, "inner_fit_pairs": fit_rows,
                       "global_threshold": global_threshold, "selected_missing_threshold": selected,
                       "inner_trials": trials, "baseline": baseline, "proposed": proposed,
                       "delta_macro": float((proposed_entity - baseline_entity).mean())}
        report["folds"].append(fold_result)
        (args.output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"held_fold": held, "selected": selected,
                          "baseline": baseline["macro_f0_5"], "proposed": proposed["macro_f0_5"]}), flush=True)
    deltas = np.concatenate([np.load(args.output / f"fold-{held}-paired-delta.npy") for held in [1, 2, 3]])
    rng = np.random.default_rng(20260925)
    bootstrap = [rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(2000)]
    report["pooled_delta_macro"] = float(deltas.mean())
    report["paired_bootstrap_95_ci"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    report["runtime_seconds"] = time.perf_counter() - started
    report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"delta": report["pooled_delta_macro"], "ci": report["paired_bootstrap_95_ci"]}), flush=True)


if __name__ == "__main__":
    main()
