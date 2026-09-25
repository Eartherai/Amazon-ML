"""CL-018: self-training / pseudo-label domain adaptation simulation (unseen country = India or US).
Text-only stage-2 (no first-stage features). Source country fully labeled; target country: half 0 is the
UNLABELED pool used for pseudo-labels, half 1 is evaluation only. Threshold always from source inner split."""
import hashlib, json, sys
from collections import defaultdict
import numpy as np, polars as pl, lightgbm as lgb
z = np.load("outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"]
K = list(map(tuple, keys.tolist()))
top = pl.concat([pl.read_parquet(f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
y = np.array([lab[k] for k in K])
tc = dict(pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
ctry = np.array([tc[q] for q, _ in K]); h = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 4 for q, _ in K])
s1s = sorted(set(keys[:, 0].tolist()))
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(s1s))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
cols = list(range(9, X.shape[1]))
P = dict(objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=9)
def f05(Pp, T):
    if not Pp and not T: return 1.0
    tp = len(Pp & T)
    if not tp: return 0.0
    a, r = tp / len(Pp), tp / len(T); return 1.25 * a * r / (0.25 * a + r)
def macro(idx, prob, th, own=False):
    pred = defaultdict(set); qs = {K[i][0] for i in idx}
    if own:
        best = {}
        for i, p in zip(idx, prob):
            if p >= th and (K[i][1] not in best or p > best[K[i][1]][1]): best[K[i][1]] = (K[i][0], p)
        for t, (q, _) in best.items(): pred[q].add(t)
    else:
        for i, p in zip(idx, prob):
            if p >= th: pred[K[i][0]].add(K[i][1])
    return float(np.mean([f05(pred[q], truth[q]) for q in qs]))
src, dst = sys.argv[1], sys.argv[2]
S = np.where((ctry == src) & (h < 2))[0]; SV = np.where((ctry == src) & (h == 2))[0]
U = np.where((ctry == dst) & (h < 2))[0]; E = np.where((ctry == dst) & (h >= 2))[0]
m = lgb.LGBMClassifier(**P).fit(X[S][:, cols], y[S])
th = max(np.arange(0.4, 0.95, 0.03), key=lambda t: macro(SV, m.predict_proba(X[SV][:, cols])[:, 1], t))
pe = m.predict_proba(X[E][:, cols])[:, 1]
print(json.dumps({"round": 0, "src": src, "dst": dst, "th": round(float(th), 2), "eval": round(macro(E, pe, th), 5), "eval_own": round(macro(E, pe, th, True), 5)}), flush=True)
for rnd, (hi, lo) in enumerate([(0.95, 0.05), (0.9, 0.1), (0.9, 0.1)], start=1):
    pu = m.predict_proba(X[U][:, cols])[:, 1]
    # positive pseudo-label only if also the ownership winner for its target within the pool
    best = {}
    for j, i in enumerate(U):
        t = K[i][1]
        if t not in best or pu[j] > pu[best[t]]: best[t] = j
    win = np.zeros(len(U), bool); win[list(best.values())] = True
    posm = (pu >= hi) & win; negm = pu <= lo
    Xs = np.vstack([X[S][:, cols], X[U[posm]][:, cols], X[U[negm]][:, cols]])
    ys = np.concatenate([y[S], np.ones(posm.sum()), np.zeros(negm.sum())])
    pl_prec = float(y[U[posm]].mean()) if posm.sum() else None; neg_err = float(y[U[negm]].mean()) if negm.sum() else None
    m = lgb.LGBMClassifier(**P).fit(Xs, ys)
    th = max(np.arange(0.4, 0.95, 0.03), key=lambda t: macro(SV, m.predict_proba(X[SV][:, cols])[:, 1], t))
    pe = m.predict_proba(X[E][:, cols])[:, 1]
    print(json.dumps({"round": rnd, "pos_pl": int(posm.sum()), "pos_pl_precision": pl_prec, "neg_pl": int(negm.sum()), "neg_pl_error": neg_err, "th": round(float(th), 2), "eval": round(macro(E, pe, th), 5), "eval_own": round(macro(E, pe, th, True), 5)}), flush=True)
