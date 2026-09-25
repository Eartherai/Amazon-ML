"""CL-013: stack EXP-050 E5 CE logit with stage-2 top-12 v2 prob on fixed 6k (cross-fit folds 2/3, inner threshold)."""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, lightgbm as lgb, polars as pl
E = "/Users/earther/.codex/worktrees/aml-neural-warroom/Amazon ML Challange/outputs/experiments/warroom_neural/EXP-044/"
truth = json.load(open(E + "truth.json")); pairs = [json.loads(l) for l in open(E + "pairs.jsonl")]
fold = {p["q"]: p["fold"] for p in pairs}; country = {p["q"]: p["country"] for p in pairs}
z = np.load("outputs/experiments/CL-003/eval6k-top12-v2.npz")
keys = list(zip(z["q"].tolist(), z["t"].tolist())); p2 = z["p"]; X = z["X"]
e5 = pl.read_csv("outputs/experiments/CL-013/e5_logits_6k.tsv", separator="\t")
qc, tc, lc = "q", "t", "e5_logit"
el = {(q, t): v for q, t, v in e5.select(qc, tc, lc).iter_rows()}
miss = sum(k not in el for k in keys); print("e5 coverage missing", miss, "of", len(keys))
lg = lambda v: math.log(min(max(v, 1e-6), 1 - 1e-6) / (1 - min(max(v, 1e-6), 1 - 1e-6)))
Fm = np.array([[lg(pp), el.get(k, np.nan), x[1], x[2], x[3]] for k, pp, x in zip(keys, p2, X)], dtype=np.float32)
y = np.array([1 if t in set(truth[q]) else 0 for q, t in keys]); kf = np.array([fold[q] for q, _ in keys])
def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)
def macro(pred, s1s): return float(np.mean([f05(pred.get(s, set()), set(truth[s])) for s in s1s]))
def choose(idx, prob):
    by = defaultdict(list); s1s = sorted({keys[i][0] for i in idx})
    for i, pr in zip(idx, prob): by[keys[i][0]].append((keys[i][1], pr))
    best = max((macro({s: {t for t, pr in v if pr >= th} for s, v in by.items()}, s1s), th) for th in np.arange(0.3, 0.96, 0.02))
    return best[1]
params = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, verbose=-1, n_jobs=8)
res = {}
for name, cols_ in (("stage2_only", [0]), ("stage2+e5", [0, 1]), ("e5_only", [1])):
    pred = defaultdict(set)
    for trf, tef in ((2, 3), (3, 2)):
        tr, te = np.where(kf == trf)[0], np.where(kf == tef)[0]
        s1tr = sorted({keys[i][0] for i in tr}); inner = {s: i % 3 for i, s in enumerate(s1tr)}
        ip = np.zeros(len(tr))
        for k in range(3):
            m = np.array([inner[keys[i][0]] != k for i in tr])
            ip[~m] = lgb.LGBMClassifier(**params).fit(Fm[tr[m]][:, cols_], y[tr[m]]).predict_proba(Fm[tr[~m]][:, cols_])[:, 1]
        th = choose(tr, ip)
        pr = lgb.LGBMClassifier(**params).fit(Fm[tr][:, cols_], y[tr]).predict_proba(Fm[te][:, cols_])[:, 1]
        for i, v in zip(te, pr):
            if v >= th: pred[keys[i][0]].add(keys[i][1])
    s1s = sorted(truth); per = {s: f05(pred.get(s, set()), set(truth[s])) for s in s1s}
    res[name] = (float(np.mean(list(per.values()))), per)
base = res["stage2_only"][1]
for n, (m, per) in res.items():
    d = np.array([per[s] - base[s] for s in sorted(truth)]); rng = np.random.default_rng(0)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    print(n, round(m, 6), "delta", round(float(d.mean()), 6), "ci", [round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)],
          "India", round(np.mean([per[s] for s in truth if country[s] == "India"]), 5), "US", round(np.mean([per[s] for s in truth if country[s] == "US"]), 5))
