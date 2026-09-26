"""CL-048: shared hard-pair set + candidate table for Codex workers (fold-3 OOF, train labels only; Fold4 CLOSED).

Writes to /Users/earther/Desktop/aml-shared/:
  candidates_f3.parquet  every fold-3 India/US candidate the primary system sees: sparse band (stage-1 p>=0.02 in top-12)
                         and dense top-10 (p_text>=0.05); columns q,t,route,country,label,prob (primary stack or dense
                         stacker, OOF),thr,decision,q_name,q_addr,t_name,t_addr,split (dev/test by S1 hash)
  hardpairs_f3.parquet   subset: errors (FP/FN at the current decision) and uncertain pairs (0.2<=prob<=0.95)
  truth_f3.parquet       q, targets (full truth incl. retrieval misses)
"""
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]; SH = Path("/Users/earther/Desktop/aml-shared")
PS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=8, random_state=0, deterministic=True)
st = pl.read_parquet(ROOT / "outputs/experiments/CL-027/fold3-stack-v2q.parquet")
dn = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored.parquet") for c in ("India", "US")]).select("q", "t", "cos", "rank", "p_text")
dn = dn.join(pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-034/cedense-f3-{c}.parquet") for c in ("India", "US")]).rename({"logit": "ce"}), on=["q", "t"], how="left")
qs = sorted(set(st["q"].to_list()) & set(dn["q"].to_list()))
st = st.filter(pl.col("q").is_in(qs) & (pl.col("base") >= 0.02)); dn = dn.filter(pl.col("q").is_in(qs) & (pl.col("p_text") >= 0.05))
gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("").filter(pl.col("source1_entity_id").is_in(qs))
truth = {q: set(r.split(",")) - {""} for q, r in gt.iter_rows()}
yd = np.array([int(t in truth[q]) for q, t in dn.select("q", "t").iter_rows()])
Fd = dn.select(pl.col("p_text").log(), "ce", "cos", "rank").to_numpy().astype(np.float32)
half = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in dn["q"].to_list()]); pd_ = np.zeros(len(dn))
for h in (0, 1): pd_[half == h] = lgb.LGBMClassifier(**PS).fit(Fd[half != h], yd[half != h]).predict_proba(Fd[half == h])[:, 1]
sp = st.select("q", "t", pl.lit("sparse").alias("route"), "label", "prob", "thr")
de = dn.select("q", "t", pl.lit("dense").alias("route"), pl.Series("label", yd).cast(pl.Int64), pl.Series("prob", pd_), pl.lit(0.72).alias("thr"))
c = pl.concat([sp.with_columns(pl.col("label").cast(pl.Int64)), de]).unique(["q", "t"], keep="first")
# current decision = the primary rule: sparse ownership among sparse pairs at thr, then dense additions on unclaimed targets
best = {}
for q, t, p, th in sp.select("q", "t", "prob", "thr").iter_rows():
    if p >= th and (t not in best or p > best[t][1]): best[t] = (q, p)
chosen = {(q, t) for t, (q, _) in best.items()}; claimed = set(best); add = {}
for q, t, p in de.sort("prob", descending=True).select("q", "t", "prob").iter_rows():
    if p < 0.72: break
    if t not in claimed and t not in add: add[t] = q
chosen |= {(q, t) for t, q in add.items()}
c = c.with_columns(pl.Series("decision", [int((q, t) in chosen) for q, t in c.select("q", "t").iter_rows()]))
txt = {}
for f in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
    d = pl.read_csv(ROOT / "student_resource/dataset/train" / f, separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
    txt.update({i: (n, a) for i, n, a in d.select("entity_id", "business_name", "business_address").iter_rows()})
s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
c = c.with_columns(pl.col("q").replace_strict(s1c).alias("country"),
                   pl.Series("q_name", [txt[q][0] for q in c["q"]]), pl.Series("q_addr", [txt[q][1] for q in c["q"]]),
                   pl.Series("t_name", [txt[t][0] for t in c["t"]]), pl.Series("t_addr", [txt[t][1] for t in c["t"]]),
                   pl.Series("split", ["dev" if int(hashlib.sha256(("split" + q).encode()).hexdigest(), 16) % 2 == 0 else "test" for q in c["q"]]))
c.write_parquet(SH / "candidates_f3.parquet")
hard = c.filter(((pl.col("decision") == 1) & (pl.col("label") == 0)) | ((pl.col("decision") == 0) & (pl.col("label") == 1)) | pl.col("prob").is_between(0.2, 0.95))
hard.write_parquet(SH / "hardpairs_f3.parquet")
pl.DataFrame({"q": list(truth), "targets": [",".join(sorted(v)) for v in truth.values()]}).write_parquet(SH / "truth_f3.parquet")
print(c.group_by("route").len(), hard.group_by(["route", "label", "decision"]).len().sort(["route", "label", "decision"]))
