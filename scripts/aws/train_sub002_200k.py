"""Fit SUB-002 on all 200k unlocked, owner-safe EXP-032 feature pairs.

The 51 features, frozen retrieval and LightGBM parameters match SUB-001. This
is a final fit on development folds 1-3; Fold 4 labels stay closed. EXP-033
provides the separate fixed 15k held-out learning-curve evidence for scale.
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    if a.output.exists() or a.threads < 1:
        raise ValueError("Fresh output directory and positive thread count required")
    start = time.perf_counter()
    report = json.loads((a.features / "metrics.json").read_text())
    if report["queries"] != 200_000 or report["features"] != NAMES:
        raise ValueError("EXP-032 feature store mismatch")
    files = sorted(a.features.glob("features-*.parquet"))
    if len(files) != json.loads((a.features / "COMPLETE.json").read_text())["feature_parts"]:
        raise ValueError("Feature part inventory mismatch")
    scan = pl.scan_parquet([str(path) for path in files])
    if scan.collect_schema().names() != ["source1_entity_id", "target_id", "country", "fold", "owner_fold", "label", *NAMES]:
        raise ValueError("Feature schema or order mismatch")
    fit = scan.filter(pl.col("fold").is_in([1, 2, 3]) & pl.col("owner_fold").is_in([-1, 1, 2, 3])).collect()
    if fit["source1_entity_id"].n_unique() != 200_000:
        raise ValueError("Training entities incomplete")
    if fit.filter((pl.col("label") == 1) & (pl.col("owner_fold") != pl.col("fold"))).height:
        raise ValueError("Positive owner mismatch")
    params = json.loads(Path("configs/baselines/BASELINE-P4-001.yaml").read_text())["hyperparameters"]
    params.update(n_jobs=a.threads, deterministic=True, force_col_wise=True)
    X, y = fit.select(NAMES).to_numpy(), fit["label"].to_numpy()
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, feature_name=NAMES)
    a.output.mkdir(parents=True)
    model_path = a.output / "model.txt"
    model.booster_.save_model(str(model_path))
    manifest = {"experiment": "SUB-002-200K", "source": "EXP-032", "validation": "EXP-033 fixed 15k heldout: 20k .93492235, 50k .93630527, 100k .93715814; 200k final fit has no independent heldout score", "fold4": "CLOSED", "fit_queries": 200_000, "fit_pairs": len(fit), "fit_positives": int(y.sum()), "features": NAMES, "params": params, "feature_store_metrics_sha256": sha256(a.features / "metrics.json"), "model_sha256": sha256(model_path), "lightgbm_version": lgb.__version__, "elapsed_seconds": time.perf_counter() - start}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
