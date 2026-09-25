"""ERR-MINE-001 step 1: link table for the 194k training S1 top-12 stage-2 v2 OOF.

Inputs (read-only):
  outputs/experiments/CL-003/trainmat-top12-v2.npz   keys (N x 2), X (N x 57, stage2_features_v2.NAMES)
  outputs/experiments/CL-003/train-oof-top12-v2.npy  grouped 3-fold stage-2 OOF probabilities aligned to keys
  outputs/experiments/CL-003/s3/results/top12-fold{1,2,3}.parquet  pair labels / first-stage fold
  student_resource/dataset/train/train_ground_truth.tsv  (filtered to non-fold4 S1 at scan time)
  artifacts/validation/v1/validation_folds.parquet  (entity folds; fold4 rows dropped before any truth join)

Outputs (outputs/analysis/error_mining/):
  links.parquet   one row per (S1, target) in keys: p, label, selected X columns, country, S1 truth size
  s1.parquet      one row per S1: n_true, n_true_top12, n_pred, tp, country
  truth_links.parquet  every true (S1, target) link of the 194k S1 with an in_keys / in_top12_raw flag
  owned_targets.parquet  target ids owned by any non-fold4 training S1 (ownership diagnostics), with owner count

Fold4 truth is never read into memory beyond the lazy-scan filter; no test data is touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/classical"))
import stage2_features_v2 as F2  # noqa: E402

CL3 = ROOT / "outputs/experiments/CL-003"
OUT = ROOT / "outputs/analysis/error_mining"
TRAIN = ROOT / "student_resource/dataset/train"
TSV = dict(separator="\t", quote_char=None, infer_schema_length=0)

KEEP = [
    "base", "rank", "gap_top", "n_strong", "n_mid", "sib_name", "sib_addr", "sib_dup",
    "q_nonlatin", "t_nonlatin", "t_s2", "q_addr_missing", "t_addr_missing",
    "core_eq", "core_jw", "core_tset", "core_contain", "core_q_only", "core_t_only", "first_core_eq",
    "name_idf_cos", "name_idf_qmiss_max", "name_idf_tmiss_max", "sk_eq", "sk_jw", "compact_jw", "initials_match", "t_domain",
    "addr_tset", "addr_idf_cos", "addr_idf_qmiss_max", "addr_idf_tmiss_max", "state_eq", "state_conflict",
    "num_exact", "num_fuzzy", "num_q_unmatched", "num_t_unmatched", "first_num_rel", "digits_jw",
    "num_del", "num_sub", "num_contain", "addr_sk_cover_q",
]


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(CL3 / "trainmat-top12-v2.npz", allow_pickle=False)
    keys = z["keys"]
    X = z["X"]
    assert X.shape[1] == len(F2.NAMES) == 57, X.shape
    oof = np.load(CL3 / "train-oof-top12-v2.npy")
    assert len(oof) == len(keys) == X.shape[0]
    idx = [F2.NAMES.index(c) for c in KEEP]
    cols = {c: X[:, i].copy() for c, i in zip(KEEP, idx)}
    del X
    links = pl.DataFrame({"q": keys[:, 0].astype(str), "t": keys[:, 1].astype(str), "p": oof.astype(np.float64), **cols})
    del keys, cols
    print(json.dumps({"stage": "loaded", "rows": links.height, "s": round(time.time() - t0, 1)}), flush=True)

    top = pl.concat([pl.read_parquet(CL3 / f"s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    top = top.rename({"source1_entity_id": "q", "target_id": "t", "rank": "rank_raw", "base_score": "base_raw"})
    s1s = links.select("q").unique()
    top = top.join(s1s, on="q", how="inner")
    links = links.join(top.select("q", "t", "label", "fold", "rank_raw"), on=["q", "t"], how="left")
    assert links["label"].null_count() == 0, "every key must have a top-12 label"

    folds = pl.read_parquet(ROOT / "artifacts/validation/v1/validation_folds.parquet").select("source1_entity_id", "fold", "country")
    open_ids = folds.filter(pl.col("fold") != 4)
    # Lazy scan + semi join drops fold4 rows at read time.
    gt = (pl.scan_csv(TRAIN / "train_ground_truth.tsv", **TSV)
          .join(open_ids.lazy().select("source1_entity_id"), on="source1_entity_id", how="semi")
          .collect())
    gt_links = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("t"))
                .explode("t").filter(pl.col("t") != "").select(pl.col("source1_entity_id").alias("q"), "t"))
    owned = gt_links.group_by("t").agg(pl.len().alias("n_owners"), pl.col("q").first().alias("owner_any"))
    owned.write_parquet(OUT / "owned_targets.parquet")

    truth = gt_links.join(s1s, on="q", how="inner")
    in_keys = links.select("q", "t").with_columns(pl.lit(True).alias("in_keys"))
    in_top_raw = top.select("q", "t").with_columns(pl.lit(True).alias("in_top12_raw"))
    truth = (truth.join(in_keys, on=["q", "t"], how="left").join(in_top_raw, on=["q", "t"], how="left")
             .with_columns(pl.col("in_keys").fill_null(False), pl.col("in_top12_raw").fill_null(False)))
    truth.write_parquet(OUT / "truth_links.parquet")

    # consistency: parquet label == truth membership for keys
    chk = links.select("q", "t", "label").join(truth.select("q", "t").with_columns(pl.lit(1).alias("tl")), on=["q", "t"], how="left")
    mism = int((chk["label"].cast(pl.Int32) != chk["tl"].fill_null(0)).sum())

    country = folds.select(pl.col("source1_entity_id").alias("q"), "country", pl.col("fold").alias("efold"))
    s1 = (s1s.join(truth.group_by("q").agg(pl.len().alias("n_true"), pl.col("in_keys").sum().alias("n_true_keys"),
                                           pl.col("in_top12_raw").sum().alias("n_true_top12raw")), on="q", how="left")
          .with_columns(pl.col("n_true").fill_null(0), pl.col("n_true_keys").fill_null(0), pl.col("n_true_top12raw").fill_null(0))
          .join(country, on="q", how="left"))
    assert s1["efold"].null_count() == 0 and int((s1["efold"] == 4).sum()) == 0
    links = links.join(s1.select("q", "n_true", "country"), on="q", how="left")
    links.write_parquet(OUT / "links.parquet")
    s1.write_parquet(OUT / "s1.parquet")
    print(json.dumps({"stage": "done", "rows": links.height, "s1": s1.height, "true_links": truth.height,
                      "label_truth_mismatch": mism, "owned_targets_non_fold4": owned.height,
                      "s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
