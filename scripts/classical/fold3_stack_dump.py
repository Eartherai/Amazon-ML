"""CL-027: dump fold-3 cross-fit stack probabilities (stage-2 + CE, same recipe as CL-024) with their
per-half OOF thresholds so candidate-restricted systems can be re-scored without refitting.
Output: outputs/experiments/CL-027/fold3-stack.parquet (q, t, base, rank, label, prob, half, thr). Fold4 CLOSED.
"""
import sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ce = sys.argv[1] if len(sys.argv) > 1 else "outputs/experiments/CL-014/e5b4-ep3.parquet"
__file__ = str(Path(__file__).resolve().parent / "stack_ce_fold3.py")
sys.argv = ["x", ce]
exec(open(__file__).read().split("params = dict(")[0])
params = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=8, random_state=0, deterministic=True, force_col_wise=True)
prob = np.zeros(len(K)); thr = {}
for h in (0, 1):
    tr, te = np.where(half != h)[0], np.where(half == h)[0]
    trq = sorted({K[i][0] for i in tr}); inner = {q: j % 3 for j, q in enumerate(trq)}; ip = np.zeros(len(tr))
    for k in range(3):
        msk = np.array([inner[K[i][0]] != k for i in tr])
        ip[~msk] = lgb.LGBMClassifier(**params).fit(F[tr[msk]], y[tr[msk]]).predict_proba(F[tr[~msk]])[:, 1]
    thr[h] = max((np.mean(list(macro(decide(tr, ip, th, True), trq).values())), th) for th in np.arange(0.4, 0.95, 0.02))[1]
    prob[te] = lgb.LGBMClassifier(**params).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
out = Path("outputs/experiments/CL-027"); out.mkdir(parents=True, exist_ok=True)
pl.DataFrame({"q": [k[0] for k in K], "t": [k[1] for k in K], "base": X[:, 0], "rank": X[:, 2], "label": y, "prob": prob, "half": half,
              "thr": np.array([thr[h] for h in half])}).write_parquet(out / "fold3-stack.parquet")
print({"thr": {int(k): float(v) for k, v in thr.items()}, "pairs": len(K)})
