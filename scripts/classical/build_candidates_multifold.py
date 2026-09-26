"""CL-054: per-fold candidate tables (same columns as aml-shared/candidates_f3.parquet) built from the fully
fold-held multi-fold pipeline (CL-042 stack dump, CL-050 dense inputs, dense stacker trained on the other folds,
tau 0.8, sparse band p>=0.02, ownership), so worker correctors can be multi-fold confirmed. Fold4 CLOSED.
Writes /Users/earther/Desktop/aml-shared/candidates_mf_f{1,2,3}.parquet and truth_mf.parquet."""
import sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "outputs/experiments/CL-050"; SH = Path("/Users/earther/Desktop/aml-shared")
sys.path.insert(0, str(ROOT / "scripts/classical"))
from dense_multifold import PS
st = pl.read_parquet(ROOT / "outputs/experiments/CL-042/multifold-stack-v2q-ce1.parquet").filter(pl.col("base") >= 0.02)
gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
truth = {q: set(r.split(",")) - {""} for q, r in gt.iter_rows()}
FEAT = [pl.col("p_text_oof").log(), "ce", "cos", "rank"]
dens = {}
for h in (1, 2, 3):
    d = pl.read_parquet(D / f"f{h}-dense.parquet").join(pl.read_parquet(D / f"ce2-f{h}-dense.parquet").select("q", "t", pl.col("logit").alias("ce")), on=["q", "t"], how="left").filter(pl.col("p_text_oof") >= 0.05)
    dens[h] = d.with_columns(pl.Series("y", [int(t in truth.get(q, ())) for q, t in d.select("q", "t").iter_rows()]))
txt = {}
for f in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
    x = pl.read_csv(ROOT / "student_resource/dataset/train" / f, separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
    txt.update({i: (n, a) for i, n, a in x.select("entity_id", "business_name", "business_address").iter_rows()})
s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
for h in (1, 2, 3):
    tr = pl.concat([dens[k] for k in (1, 2, 3) if k != h])
    m = lgb.LGBMClassifier(**PS).fit(tr.select(FEAT).to_numpy().astype(np.float32), tr["y"].to_numpy())
    d = dens[h].with_columns(pl.Series("prob", m.predict_proba(dens[h].select(FEAT).to_numpy().astype(np.float32))[:, 1]))
    s = st.filter(pl.col("fold") == h)
    best = {}
    for q, t, p, th in s.select("q", "t", "prob", "thr").iter_rows():
        if p >= th and (t not in best or p > best[t][1]): best[t] = (q, p)
    chosen = {(q, t) for t, (q, _) in best.items()}; add = {}
    for q, t, p in d.sort("prob", descending=True).select("q", "t", "prob").iter_rows():
        if p < 0.8: break
        if t not in best and t not in add: add[t] = q
    chosen |= {(q, t) for t, q in add.items()}
    c = pl.concat([s.select("q", "t", pl.lit("sparse").alias("route"), pl.col("label").cast(pl.Int64), "prob", "thr"),
                   d.select("q", "t", pl.lit("dense").alias("route"), pl.col("y").cast(pl.Int64).alias("label"), "prob", pl.lit(0.8).alias("thr"))]).unique(["q", "t"], keep="first")
    c = c.with_columns(pl.Series("decision", [int((q, t) in chosen) for q, t in c.select("q", "t").iter_rows()]),
                       pl.col("q").replace_strict(s1c).alias("country"), pl.lit(h).alias("fold"),
                       pl.Series("q_name", [txt[q][0] for q in c["q"]]), pl.Series("q_addr", [txt[q][1] for q in c["q"]]),
                       pl.Series("t_name", [txt[t][0] for t in c["t"]]), pl.Series("t_addr", [txt[t][1] for t in c["t"]]))
    c.write_parquet(SH / f"candidates_mf_f{h}.parquet"); print(h, c.height, c["q"].n_unique(), int(c["decision"].sum()), flush=True)
qs = sorted(set(st["q"].to_list()))
pl.DataFrame({"q": qs, "targets": [",".join(sorted(truth.get(q, set()))) for q in qs], "fold": [int(x) for x in st.unique("q").sort("q")["fold"]]}).write_parquet(SH / "truth_mf.parquet")
