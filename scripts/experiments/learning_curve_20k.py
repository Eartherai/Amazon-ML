"""Paired, entity-level NUMERIC-V2 learning curve on the existing 20k sample.

All sizes share each outer held-out fold. The threshold for each outer fold was
selected within its other two folds by the frozen 20k nested procedure; this
diagnostic does not reselect thresholds for smaller sizes. Fold4 is never read.
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


def entity_scores(frame: pl.DataFrame, probs: np.ndarray, query: pl.DataFrame, threshold: float) -> dict:
    ids = query["entity_id"].to_list()
    loc = {entity_id: i for i, entity_id in enumerate(ids)}
    idx = np.array([loc[x] for x in frame["source1_entity_id"]], dtype=np.int32)
    keep = probs >= threshold
    counts = query["n_matches"].to_numpy()
    pred = np.bincount(idx[keep], minlength=len(ids))
    tp = np.bincount(idx[keep], weights=frame["label"].to_numpy()[keep], minlength=len(ids))
    score = np.divide(1.25 * tp, pred + 0.25 * counts, out=np.ones(len(ids)), where=(pred + 0.25 * counts) > 0)
    countries = query["country"].to_numpy()
    singleton = counts == 0
    result = {
        "entities": len(ids),
        "macro_f0_5": float(score.mean()),
        "precision": float(tp.sum() / pred.sum()) if pred.sum() else 1.0,
        "recall": float(tp.sum() / counts.sum()) if counts.sum() else 1.0,
        "singleton_f0_5": float(score[singleton].mean()),
        "non_singleton_f0_5": float(score[~singleton].mean()),
        "by_country": {country: float(score[countries == country].mean()) for country in sorted(set(countries))},
        "per_entity_scores": score,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    q = pl.read_parquet("artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet")
    if set(q["fold"].unique()) != {1, 2, 3} or q.height != 20000:
        raise ValueError("Unexpected sample; Fold4 must stay closed")
    base = pl.read_parquet("outputs/oof/P4-B-001/features/part-*.parquet")
    numeric = pl.read_parquet("artifacts/features/P4-NUMERIC-B-001/part-*.parquet")
    frame = base.join(numeric, on=["source1_entity_id", "target_id"], how="left", validate="1:1")
    names = BASE + EXTRA
    if frame.select(pl.any_horizontal(pl.col(EXTRA).is_null()).any()).item():
        raise ValueError("Missing numeric features")
    x = frame.select(names).to_numpy()
    y = frame["label"].to_numpy()
    fold = frame["fold"].to_numpy()
    owner = frame["owner_fold"].to_numpy()
    ids = frame["source1_entity_id"].to_numpy()
    params = json.loads(Path("configs/baselines/BASELINE-P4-001.yaml").read_text())["hyperparameters"]
    params["n_jobs"] = 1
    thresholds = {1: 0.83, 2: 0.79, 3: 0.83}
    report = {
        "experiment": "EXP-027",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "design": "Fixed outer-fold validation entities; deterministic sample_hash subsets of available fit entities; frozen per-outer 20k nested thresholds; no fold4",
        "limitations": "Sizes describe training entities per outer fit, not the full 20k population. Thresholds were selected by the 20k procedure, not nested independently for each smaller fit size. This is a diagnostic learning curve, not an unbiased final model comparison.",
        "folds": [],
    }
    for held in [1, 2, 3]:
        allowed = [f for f in [1, 2, 3] if f != held]
        available = q.filter(pl.col("fold").is_in(allowed)).sort("sample_hash")
        heldq = q.filter(pl.col("fold") == held).sort("entity_id")
        heldframe = frame.filter(pl.col("fold") == held)
        testmask = fold == held
        fold_report = {"held_fold": held, "threshold": thresholds[held], "sizes": []}
        for count in [2000, 5000, 10000, len(available)]:
            begin = time.perf_counter()
            selected = set(available.head(count)["entity_id"])
            mask = fitting_mask(fold, owner, allowed) & np.isin(ids, list(selected))
            if np.any((y == 1) & np.isin(fold, allowed) & np.isin(ids, list(selected)) & ~mask):
                raise AssertionError("Positive excluded by ownership filter")
            model = lgb.LGBMClassifier(**params)
            model.fit(x[mask], y[mask], feature_name=names)
            probs = model.booster_.predict(x[testmask], num_threads=1)
            metrics = entity_scores(heldframe, probs, heldq, thresholds[held])
            entity = metrics.pop("per_entity_scores")
            np.save(args.output / f"held{held}-size{count}-scores.npy", entity)
            fold_report["sizes"].append({"fit_entities": count, "fit_pairs": int(mask.sum()), "fit_positives": int(y[mask].sum()), "seconds": time.perf_counter() - begin, **metrics})
            print(json.dumps({"held": held, **fold_report["sizes"][-1]}), flush=True)
        report["folds"].append(fold_report)
        (args.output / "progress.json").write_text(json.dumps(report, indent=2))
    report["runtime_seconds"] = time.perf_counter() - started
    report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
