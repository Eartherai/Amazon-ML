"""ERR-MINE-001 step 3: per-link error table with exact loss attribution and raw texts (training data only).

FP link: p >= thr and label 0. Rejected link: p < thr and label 1 (target inside top-12 keys).
Loss attribution per S1: fp loss (f_nofp - f_cur; singleton: 1 - f_cur) split equally over its FP links;
rejected loss (f_top - f_nofp) split equally over its rejected links. Sum over links == fp + rejected mass.
Writes outputs/analysis/error_mining/error_links.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/analysis/error_mining"
TRAIN = ROOT / "student_resource/dataset/train"
TSV = dict(separator="\t", quote_char=None, infer_schema_length=0)
THR = json.loads((ROOT / "outputs/experiments/CL-003/stage2-200k-top12-v2.json").read_text())["threshold"]


def main() -> None:
    L = pl.read_parquet(OUT / "links.parquet")
    S = pl.read_parquet(OUT / "s1_losses.parquet")
    n = S.height
    L = L.with_columns((pl.col("p") >= THR).alias("pred"))
    E = L.filter((pl.col("pred") & (pl.col("label") == 0)) | (~pl.col("pred") & (pl.col("label") == 1)))
    E = E.with_columns(pl.when(pl.col("pred")).then(pl.lit("FP")).otherwise(pl.lit("REJ")).alias("etype"))
    cnt = E.group_by("q", "etype").len().rename({"len": "n_err_type"})
    E = E.join(cnt, on=["q", "etype"]).join(S.select("q", "f_cur", "f_nofp", "f_top", "single"), on="q")
    E = E.with_columns(
        pl.when(pl.col("etype") == "FP").then((pl.col("f_nofp") - pl.col("f_cur")) / pl.col("n_err_type"))
        .otherwise((pl.col("f_top") - pl.col("f_nofp")) / pl.col("n_err_type")).alias("loss"),
        pl.when(pl.col("etype") == "REJ").then(pl.lit("rejected"))
        .when(pl.col("single")).then(pl.lit("singleton_fp")).otherwise(pl.lit("nonsingleton_fp")).alias("bucket"))
    # context for each error link: does this S1 have any accepted TP? max p of the S1's other links
    ctx = L.group_by("q").agg((pl.col("pred") & (pl.col("label") == 1)).sum().alias("s1_tp"), pl.col("pred").sum().alias("s1_npred"),
                              pl.col("p").max().alias("s1_pmax"))
    E = E.join(ctx, on="q")
    # target owned by another non-fold4 training S1
    owned = pl.read_parquet(OUT / "owned_targets.parquet")
    E = E.join(owned.select("t", "n_owners"), on="t", how="left").with_columns(pl.col("n_owners").fill_null(0))
    # within-sample claims at threshold
    claims = L.filter(pl.col("pred")).group_by("t").agg(pl.len().alias("n_claim"), pl.col("p").max().alias("claim_pmax"))
    E = E.join(claims, on="t", how="left").with_columns(pl.col("n_claim").fill_null(0), pl.col("claim_pmax").fill_null(0.0))

    s1_ids = E.select(pl.col("q").alias("entity_id")).unique()
    t_ids = E.select(pl.col("t").alias("entity_id")).unique()
    s1t = pl.scan_csv(TRAIN / "train_source1.tsv", **TSV).join(s1_ids.lazy(), on="entity_id", how="semi").select(
        pl.col("entity_id").alias("q"), pl.col("business_name").alias("q_name"), pl.col("business_address").alias("q_addr")).collect()
    tt = pl.concat([pl.scan_csv(TRAIN / f"train_source{k}.tsv", **TSV).join(t_ids.lazy(), on="entity_id", how="semi")
                    .select(pl.col("entity_id").alias("t"), pl.col("business_name").alias("t_name"), pl.col("business_address").alias("t_addr")).collect()
                    for k in (2, 3)])
    E = E.join(s1t, on="q", how="left").join(tt, on="t", how="left")
    E.write_parquet(OUT / "error_links.parquet")
    tot = E.group_by("bucket").agg(pl.len(), (pl.col("loss").sum() / n).alias("mass")).sort("bucket")
    print(tot)
    print(json.dumps({"errors": E.height, "mass_total": float(E["loss"].sum() / n)}))


if __name__ == "__main__":
    main()
