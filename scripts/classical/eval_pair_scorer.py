"""CL-035: standalone evaluation of a pair scorer on fold-3 band pairs (q,t,logit): pair AUC per country and exact
macro F0.5 with ownership; the threshold is chosen on US and applied to India (and vice versa). Fold4 CLOSED.
Usage: eval_pair_scorer.py SCORES.parquet [grid values...]
"""
import json, sys
from collections import defaultdict
import numpy as np, polars as pl

f = sys.argv[1]; grid = [float(x) for x in sys.argv[2:]] or [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6]
b = pl.read_parquet(f).select("q", "t", "logit")
top = pl.read_parquet("outputs/experiments/CL-003/s3/results/top12-fold3.parquet").rename({"source1_entity_id": "q", "target_id": "t"})
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select(pl.col("entity_id").alias("q"), pl.col("country").alias("c"))
d = b.join(top.select("q", "t", "label"), on=["q", "t"]).join(s1, on="q")
q3 = top.select("q").unique().join(s1, on="q")
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(q3["q"].to_list()))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1); n = y.sum(); return float((r[y == 1].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


res = {}
for c in ("US", "India"):
    x = d.filter(pl.col("c") == c); rows = x.select("q", "t", "logit").rows(); qs = q3.filter(pl.col("c") == c)["q"].to_list(); m = {}
    for th in grid:
        best = {}
        for q, t, v in rows:
            if v >= th and (t not in best or v > best[t][1]): best[t] = (q, v)
        pr = defaultdict(set)
        for t, (q, _) in best.items(): pr[q].add(t)
        m[th] = float(np.mean([f05(pr.get(q, set()), truth[q]) for q in qs]))
    res[c] = {"auc": round(auc(x["logit"].to_numpy(), x["label"].to_numpy()), 5), "grid": {k: round(v, 5) for k, v in m.items()}}
for s, t in (("US", "India"), ("India", "US")):
    th = max(res[s]["grid"], key=res[s]["grid"].get)
    res[f"{s}->{t}"] = {"thr": th, "src": res[s]["grid"][th], "tgt": res[t]["grid"][th], "tgt_best": max(res[t]["grid"].values())}
print(json.dumps({"file": f, **res}, indent=1))
