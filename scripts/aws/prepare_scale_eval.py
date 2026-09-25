"""Reuse the immutable 15k holdout with full unlocked training inputs.

EXP-037 uses identical held-out S1 and truth as EXP-033. Its fit pool adds
unlocked fold 0; absolute scores therefore belong to a new protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import polars as pl


ROOT = Path("artifacts/cloud/phase5")
FULL = ROOT / "feature-unlocked-v001"
OLD = ROOT / "learning-eval-200k-v001"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "learning-eval-unlocked-v001")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original_eval = pl.read_parquet(OLD / "eval_queries.parquet")
    old_ids = pl.read_parquet(OLD / "old_20k_ids.parquet")
    full_queries = pl.scan_parquet(FULL / "labeled_queries.parquet")
    selected = full_queries.filter(pl.col("entity_id").is_in(original_eval["entity_id"]))
    current_eval = selected.select(original_eval.columns).collect().sort("entity_id")
    if (len(current_eval) != 15_000 or not current_eval.equals(original_eval.sort("entity_id"))
            or set(current_eval["fold"].unique().to_list()) != {1, 2, 3}
            or set(current_eval["entity_id"]) & set(old_ids["entity_id"])):
        raise ValueError("Full feature input changes the frozen evaluation population")
    ids = set(current_eval["entity_id"])
    prior_truth = pl.scan_parquet(OLD / "truth.parquet").filter(
        pl.col("source1_entity_id").is_in(ids)).collect().sort("source1_entity_id", "target_id")
    full_truth = pl.scan_parquet(FULL / "truth.parquet").filter(
        pl.col("source1_entity_id").is_in(ids)).collect().sort("source1_entity_id", "target_id")
    if not prior_truth.equals(full_truth):
        raise ValueError("Full input changes frozen evaluation truth")
    args.output.mkdir(parents=True)
    for source, name in ((FULL / "labeled_queries.parquet", "labeled_queries.parquet"),
                         (FULL / "truth.parquet", "truth.parquet"),
                         (OLD / "eval_queries.parquet", "eval_queries.parquet"),
                         (OLD / "old_20k_ids.parquet", "old_20k_ids.parquet")):
        os.link(source, args.output / name)
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(args.output.iterdir()) if path.is_file()]
    report = {"experiment": "EXP-037", "scope": "Full unlocked fit pool folds0-3; exact EXP-033 15k heldout; Fold4 CLOSED",
              "training_pool": 1_765_649, "evaluation_queries": 15_000,
              "evaluation_truth_pairs": len(full_truth),
              "evaluation_queries_sha256": sha256(OLD / "eval_queries.parquet"),
              "files": files}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"training_pool": report["training_pool"],
                      "evaluation_queries": report["evaluation_queries"],
                      "evaluation_truth_pairs": len(full_truth)}), flush=True)


if __name__ == "__main__":
    main()
