"""CL-043: is the QNORM (v2q) stage-2 gain inside the CE stack robust to the stacker cross-fit protocol?
Fold-3 S1, stack features as in production; stacker cross-fit with K in {2,3,5} S1 partitions x 3 salts; threshold curve
reported (0.5..0.9) plus the max. Paired comparison v2 vs v2q on identical partitions. Fold4 CLOSED."""
import hashlib, json
from collections import defaultdict
import numpy as np, polars as pl, lightgbm as lgb
PS = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.9, verbose=-1, n_jobs=8, random_state=0, deterministic=True, force_col_wise=True)
lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)
z = np.load("outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"]
top = pl.concat([pl.read_parquet(f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list())); lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
m3 = np.array([fold[q] == 3 for q in keys[:, 0].tolist()]); K = list(map(tuple, keys[m3].tolist())); X3 = X[m3]; y = np.array([lab[k] for k in K])
qa = np.array([k[0] for k in K]); ta = np.array([k[1] for k in K]); qs = sorted(set(qa.tolist()))
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(qs))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
ce = {(q, t): v for q, t, v in pl.read_parquet("outputs/experiments/CL-014/e5b4-ep3.parquet").select("q", "t", "logit").iter_rows()}
lv = np.array([ce.get(k, np.nan) for k in K], np.float32)
c = pl.DataFrame({"q": qa, "v": lv}).with_columns(pl.col("v").fill_nan(None))
rk = c.select(pl.col("v").rank("ordinal", descending=True).over("q")).to_series().fill_null(np.nan).to_numpy(); gp = c.select(pl.col("v").max().over("q") - pl.col("v")).to_series().fill_null(np.nan).to_numpy()
def macro(p, th):
    best = {}
    for q, t, v in zip(qa, ta, p):
        if v >= th and (t not in best or v > best[t][1]): best[t] = (q, v)
    d = defaultdict(set)
    for t, (q, _) in best.items(): d[q].add(t)
    return float(np.mean([f05(d.get(q, set()), truth[q]) for q in qs]))
res = defaultdict(dict)
for tag, f in (("v2", "train-oof-top12-v2.npy"), ("v2q", "train-oof-top12-v2q.npy")):
    p2 = np.load("outputs/experiments/CL-003/" + f)[m3]
    Fm = np.column_stack([lg(p2), X3[:, 0], X3[:, 2], X3[:, 3], X3[:, 4], lv, rk, gp]).astype(np.float32)
    for kf in (2, 3, 5):
        for salt in ("", "a", "b"):
            part = np.array([int(hashlib.sha256((salt + q).encode()).hexdigest(), 16) % kf for q in qa])
            oof = np.zeros(len(y))
            for k in range(kf): oof[part == k] = lgb.LGBMClassifier(**PS).fit(Fm[part != k], y[part != k]).predict_proba(Fm[part == k])[:, 1]
            curve = {th: macro(oof, th) for th in (0.5, 0.6, 0.66, 0.7, 0.74, 0.8, 0.9)}
            res[tag][f"k{kf}{salt}"] = curve
            print(json.dumps({"stage2": tag, "k": kf, "salt": salt, "curve": {k: round(v, 5) for k, v in curve.items()}, "max": round(max(curve.values()), 5)}), flush=True)
