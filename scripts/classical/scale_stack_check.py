"""CL-065: does stage-2 training-size gain survive the CE stacker? Stage-2 v2q fit on a hash share of train folds 1-2,
predicts fold-3; the production stacker (8 features, e5b4-ep3 held CE) is cross-fit on fold-3 (salted 3-fold S1 partition)
and scored with ownership at its best threshold. Exact macro F0.5 on fold-3 S1 (top-12 universe). Fold4 CLOSED."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts/classical"))
from stage2_qnorm import qnorm, PARAMS as P2
from assemble_final_v2 import PARAMS as PS, logit, ce_ctx, own_decide, f05
z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"].astype(np.float32)
top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list())); lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
qa, ta = keys[:, 0], keys[:, 1]; y = np.array([lab[(q, t)] for q, t in zip(qa.tolist(), ta.tolist())], np.int8)
fo = np.array([fold[q] for q in qa.tolist()]); X = np.hstack([X, qnorm(qa, X)]); f3 = fo == 3
frac = np.array([int(hashlib.sha256(("lc" + q).encode()).hexdigest(), 16) % 100 for q in qa.tolist()])
K3 = list(zip(qa[f3].tolist(), ta[f3].tolist())); X3, y3 = X[f3], y[f3]
ce = {(q, t): v for q, t, v in pl.read_parquet(ROOT / "outputs/experiments/CL-014/e5b4-ep3.parquet").select("q", "t", "logit").iter_rows()}
lv = np.array([ce.get(k, np.nan) for k in K3], np.float32); rk, gp = ce_ctx(K3, lv)
part = np.array([int(hashlib.sha256(("a" + q).encode()).hexdigest(), 16) % 3 for q, _ in K3])
qs = sorted({k[0] for k in K3})
gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(qs))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
for pct in [int(a) for a in sys.argv[1:]]:
    tr = np.isin(fo, [1, 2]) & (frac < pct)
    p2 = lgb.LGBMClassifier(**P2).fit(X[tr], y[tr]).predict_proba(X3)[:, 1]
    Fm = np.column_stack([logit(p2), X3[:, 0], X3[:, 2], X3[:, 3], X3[:, 4], lv, rk, gp]).astype(np.float32); oof = np.zeros(len(K3))
    for k in range(3):
        oof[part == k] = lgb.LGBMClassifier(**PS).fit(Fm[part != k], y3[part != k]).predict_proba(Fm[part == k])[:, 1]
    def mac(p, th):
        d = own_decide(K3, p, th); return float(np.mean([f05(d.get(q, set()), truth[q]) for q in qs]))
    s2 = max(mac(p2, th) for th in (0.62, 0.66, 0.7, 0.74)); st = max(mac(oof, th) for th in np.arange(0.6, 0.84, 0.02))
    print(json.dumps({"pct": pct, "train_rows": int(tr.sum()), "stage2_macro": round(s2, 6), "stack_macro": round(st, 6)}), flush=True)
