"""Freeze new, previously unseen fold1–3 entities for the 20k/50k/100k curve."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import polars as pl


ROOT = Path("artifacts/cloud/phase5")
OUTPUT = ROOT / "learning-eval-200k-v001"
SOURCE = ROOT / "feature-200k-input-v001"
OLD = Path("artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    old = pl.read_parquet(OLD, columns=["entity_id", "fold"])
    queries = pl.read_parquet(SOURCE / "labeled_queries.parquet", columns=["entity_id", "country", "fold", "n_matches"])
    if len(old) != 20_000 or len(queries) != 200_000:
        raise ValueError("Unexpected source sizes")
    old_ids = set(old["entity_id"])
    new = queries.filter(~pl.col("entity_id").is_in(old_ids))
    selected = []
    for fold in (1, 2, 3):
        frame = new.filter(pl.col("fold") == fold).with_columns(
            pl.col("entity_id").map_elements(lambda value: hashlib.sha256(value.encode()).hexdigest(),
                                               return_dtype=pl.String).alias("selection_hash")
        ).sort("selection_hash", "entity_id").head(5_000)
        if len(frame) != 5_000:
            raise ValueError(f"Insufficient new validation entities in fold {fold}")
        selected.append(frame.drop("selection_hash"))
    eval_queries = pl.concat(selected).sort("entity_id")
    if eval_queries["entity_id"].n_unique() != 15_000 or set(eval_queries["entity_id"]) & old_ids:
        raise ValueError("New evaluation set overlaps old development sample")
    eval_queries.write_parquet(OUTPUT / "eval_queries.parquet", compression="zstd")
    old.select("entity_id", "fold").sort("entity_id").write_parquet(OUTPUT / "old_20k_ids.parquet", compression="zstd")
    shutil.copy2(SOURCE / "labeled_queries.parquet", OUTPUT / "labeled_queries.parquet")
    shutil.copy2(SOURCE / "truth.parquet", OUTPUT / "truth.parquet")
    con = duckdb.connect(config={"threads": 2})
    truth_count = con.execute("SELECT count(*) FROM read_parquet(?)", [str(OUTPUT / "truth.parquet")]).fetchone()[0]
    con.close()
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(OUTPUT.iterdir()) if path.is_file()]
    report = {"experiment": "EXP-033", "scope": "Fixed 15k new fold1–3 validation S1; Fold4 CLOSED",
              "evaluation_per_fold": 5_000, "evaluation_total": 15_000, "old_20k_size": 20_000,
              "training_pool": 200_000, "truth_pairs": truth_count,
              "selection": "SHA256(entity_id) lowest 5k per fold after excluding old 20k",
              "thresholds_from_old20k": {"1": 0.83, "2": 0.79, "3": 0.83},
              "source_queries_sha256": sha256(SOURCE / "labeled_queries.parquet"),
              "source_truth_sha256": sha256(SOURCE / "truth.parquet"), "files": files}
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"evaluation_queries": len(eval_queries), "by_fold_country":
                      eval_queries.group_by("fold", "country").len().sort("fold", "country").to_dicts()}), flush=True)


if __name__ == "__main__":
    main()
