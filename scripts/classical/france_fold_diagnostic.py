"""CL-040: label-free diagnostic for accent folding on France (no France labels exist).

Reference labels = the accent-invariant stage-2 v2 decision (test prob p2 >= 0.67 with ownership) on test band pairs.
Metric = pair AUC of the CE logit against that reference, per country, for raw CE (4EP, unfolded) and for the France
accent-folded CE. If folding moves France agreement toward the India/US level, raw accents were confusing the CE.
This is agreement, not accuracy. Fold4 CLOSED.
"""
import glob, json
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(__file__).resolve().parents[2]


def auc(p, y):
    o = np.argsort(p, kind="stable"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1); n = y.sum()
    return float((r[y == 1].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def main():
    s1 = pl.read_csv(ROOT / "student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select(pl.col("entity_id").alias("q"), "country")
    pr = pl.read_parquet(ROOT / "outputs/experiments/CL-003/test_probs/CL-005-top12v2-v1.parquet").select("q", "t", "base", "p").filter(pl.col("base") >= 0.02)
    pr = pr.with_columns(((pl.col("p") >= 0.67) & (pl.col("p") == pl.col("p").max().over("t"))).cast(pl.Int8).alias("ref")).join(s1, on="q")
    raw = pl.read_parquet(ROOT / "outputs/experiments/CL-025/bundles/ce4ep_test.parquet").select("q", "t", pl.col("logit").alias("ce_raw"))
    fold = pl.concat([pl.read_parquet(f) for f in glob.glob(str(ROOT / "outputs/experiments/CL-040/frfold/frband-*.parquet"))]).select("q", "t", pl.col("logit").alias("ce_fold"))
    d = pr.join(raw, on=["q", "t"], how="inner").join(fold, on=["q", "t"], how="left")
    acc = pl.read_csv(ROOT / "student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("").select(
        pl.col("entity_id").alias("q"), (pl.col("business_name") + pl.col("business_address")).str.contains(r"[^\x00-\x7F]").alias("accented"))
    d = d.join(acc, on="q")
    rows = []
    for c in ("US", "India", "France"):
        x = d.filter(pl.col("country") == c)
        for sub, xs in (("all", x), ("accented_S1", x.filter(pl.col("accented"))), ("ascii_S1", x.filter(~pl.col("accented")))):
            if xs.height == 0 or xs["ref"].sum() == 0: continue
            r = {"country": c, "subset": sub, "pairs": xs.height, "ref_pos_rate": round(float(xs["ref"].mean()), 4), "auc_raw": round(auc(xs["ce_raw"].to_numpy(), xs["ref"].to_numpy()), 5)}
            if c == "France" and xs["ce_fold"].null_count() < xs.height:
                xf = xs.drop_nulls("ce_fold"); r["auc_fold"] = round(auc(xf["ce_fold"].to_numpy(), xf["ref"].to_numpy()), 5); r["auc_raw_same_pairs"] = round(auc(xf["ce_raw"].to_numpy(), xf["ref"].to_numpy()), 5)
            rows.append(r); print(json.dumps(r), flush=True)
    pl.DataFrame(rows).write_csv(ROOT / "outputs/experiments/CL-040/france_fold_diagnostic.csv")


if __name__ == "__main__":
    main()
