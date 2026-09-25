"""CL-004: stack 200k stage-2 OOF probability with EXP-045 logit on the fixed 6k (cross-fit folds 2/3)."""
import json, sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_stage2_exp044 as P
ROOT = Path(__file__).resolve().parents[2]
scope = sys.argv[1] if len(sys.argv) > 1 else "sidecar"
z = np.load(ROOT / f"outputs/experiments/CL-003/eval6k-{scope}.npz")
keys = list(zip(z["q"].tolist(), z["t"].tolist())); p = z["p"]; X = z["X"]
truth = json.loads((P.EXP044 / "truth.json").read_text())
pairs = [json.loads(l) for l in (P.EXP044 / "pairs.jsonl").open()]
fold = {x["q"]: x["fold"] for x in pairs}; country = {x["q"]: x["country"] for x in pairs}
nl = {}
for line in (P.EXP044.parent / "EXP-045/neural_scores.jsonl").open():
    r = json.loads(line); nl[(r["q"], r["t"])] = r["score"]
lg = lambda v: math.log(min(max(v,1e-6),1-1e-6)/(1-min(max(v,1e-6),1-1e-6)))
F = np.array([[lg(pp), nl.get(k, np.nan), x[1], x[2], x[3]] for k, pp, x in zip(keys, p, X)], dtype=np.float32)
y = np.array([1 if t in set(truth[s]) else 0 for s, t in keys]); kf = np.array([fold[s] for s, _ in keys])
params = dict(objective="binary", n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=40, verbose=-1, n_jobs=8)
allpred, basepred = defaultdict(set), defaultdict(set)
thr0 = json.loads((ROOT / f"outputs/experiments/CL-003/stage2-200k-{scope}.json").read_text())["threshold"]
for (s, t), pp in zip(keys, p):
    if pp >= thr0: basepred[s].add(t)
for tr_f, te_f in ((2, 3), (3, 2)):
    tr, te = np.where(kf == tr_f)[0], np.where(kf == te_f)[0]
    tr_s1 = sorted({keys[i][0] for i in tr}); inner = {s: i % 3 for i, s in enumerate(tr_s1)}
    ip = np.zeros(len(tr))
    for k in range(3):
        m = np.array([inner[keys[i][0]] != k for i in tr])
        ip[~m] = lgb.LGBMClassifier(**params).fit(F[tr[m]], y[tr[m]]).predict_proba(F[tr[~m]])[:, 1]
    thr, _ = P.choose_threshold([keys[i] for i in tr], ip, truth, tr_s1)
    pr = lgb.LGBMClassifier(**params).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
    for i, v in zip(te, pr):
        if v >= thr: allpred[keys[i][0]].add(keys[i][1])
    print("dir", tr_f, "->", te_f, "thr", round(thr, 2))
s1s = sorted(truth)
e1, per1 = P.evaluate(s1s, basepred, truth, country); e2, per2 = P.evaluate(s1s, allpred, truth, country)
d = np.array([per2[s] - per1[s] for s in s1s]); rng = np.random.default_rng(0)
b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
print(json.dumps({"scope": scope, "stage2_200k": e1, "hybrid_stack": e2, "delta": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}, indent=1))
