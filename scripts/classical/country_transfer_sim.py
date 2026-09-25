"""CL-017: unseen-country simulation with the cached v2 stage-2 matrix (194k OOF S1, top-12 pairs).
Variants: full stage-2 (with first-stage base) and text-only (no base/context). Train on one country,
evaluate on the other vs in-country (hash half split). Exact macro F0.5 with full truth, threshold
chosen on the training country's inner OOF (no target-country labels used for selection)."""
import hashlib, json, sys
from collections import defaultdict
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "scripts/classical"); import stage2_features_v2 as F
z = np.load("outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"]
K = list(map(tuple, keys.tolist()))
top = pl.concat([pl.read_parquet(f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
y = np.array([lab[k] for k in K])
tc = dict(pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
ctry = np.array([tc[q] for q, _ in K]); half = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q, _ in K])
s1s = sorted(set(keys[:, 0].tolist()))
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(s1s))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)
def macro(idx, prob, th):
    pred = defaultdict(set); qs = {K[i][0] for i in idx}
    for i, p in zip(idx, prob):
        if p >= th: pred[K[i][0]].add(K[i][1])
    return float(np.mean([f05(pred[q], truth[q]) for q in qs]))
P = dict(objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=9)
text_cols = list(range(9, X.shape[1])); full_cols = list(range(X.shape[1]))
for name, cols in (("full", full_cols), ("textonly", text_cols)):
    for src in ("US", "India"):
        tr = np.where((ctry == src) & (half == 0))[0]
        # threshold from training-country inner split (half 0 -> quarter split)
        q4 = np.array([int(hashlib.sha256(K[i][0].encode()).hexdigest(), 16) % 4 for i in tr])
        m = lgb.LGBMClassifier(**P).fit(X[tr[q4 != 0]][:, cols], y[tr[q4 != 0]])
        vi = tr[q4 == 0]; pv = m.predict_proba(X[vi][:, cols])[:, 1]
        th = max(np.arange(0.4, 0.95, 0.03), key=lambda t: macro(vi, pv, t))
        m = lgb.LGBMClassifier(**P).fit(X[tr][:, cols], y[tr])
        for dst in ("US", "India"):
            te = np.where((ctry == dst) & (half == 1))[0]
            print(json.dumps({"features": name, "train": src, "eval": dst, "threshold": round(float(th), 2), "macro": round(macro(te, m.predict_proba(X[te][:, cols])[:, 1], th), 5)}), flush=True)
