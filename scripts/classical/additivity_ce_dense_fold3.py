import hashlib, numpy as np, polars as pl, lightgbm as lgb, sys
from collections import defaultdict
__file__="/Users/earther/Desktop/Amazon ML Challange/scripts/classical/stack_ce_fold3.py"
sys.argv=["x","outputs/experiments/CL-014/e5base-ep1.parquet"]
src=open(__file__).read().split("params = dict(")[0]
exec(src)
params = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=8)
cols=list(range(F.shape[1])); prob=np.zeros(len(K)); thr={}
for h in (0,1):
    tr,te=np.where(half!=h)[0],np.where(half==h)[0]
    trq=sorted({K[i][0] for i in tr}); inner={q:j%3 for j,q in enumerate(trq)}; ip=np.zeros(len(tr))
    for k in range(3):
        msk=np.array([inner[K[i][0]]!=k for i in tr])
        ip[~msk]=lgb.LGBMClassifier(**params).fit(F[tr[msk]][:,cols],y[tr[msk]]).predict_proba(F[tr[~msk]][:,cols])[:,1]
    thr[h]=max((np.mean(list(macro(decide(tr,ip,th,True),trq).values())),th) for th in np.arange(0.4,0.95,0.02))[1]
    prob[te]=lgb.LGBMClassifier(**params).fit(F[tr][:,cols],y[tr]).predict_proba(F[te][:,cols])[:,1]
np.save("outputs/experiments/CL-014/fold3-stack-prob.npy",prob)
pred=defaultdict(set); best={}
for i,(q,t) in enumerate(K):
    if prob[i]>=thr[half[i]] and (t not in best or prob[i]>best[t][1]): best[t]=(q,prob[i])
for t,(q,_) in best.items(): pred[q].add(t)
qs=sorted(truth); m0=np.mean([f05(pred.get(q,set()),truth[q]) for q in qs])
claimed={t for v in pred.values() for t in v}
d=pl.concat([pl.read_parquet(f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored.parquet") for c in ("India","US")]).filter(pl.col("q").is_in(qs))
for th in (0.7,0.8,0.9):
    add=defaultdict(set)
    for q,t,pp in d.select("q","t","p_text").iter_rows():
        if pp>=th and t not in claimed: add[q].add(t)
    per={q:f05(pred.get(q,set())|add[q],truth[q]) for q in qs}
    m=np.mean(list(per.values()))
    ind=np.mean([per[q] for q in qs if tc[q]=="India"]); us=np.mean([per[q] for q in qs if tc[q]=="US"])
    print(f"stack+own {m0:.5f} + dense p_text>={th}: {m:.5f} delta {m-m0:+.5f} India {ind:.5f} US {us:.5f}", flush=True)
