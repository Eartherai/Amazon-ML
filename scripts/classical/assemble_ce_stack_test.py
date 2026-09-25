"""CL-020: apply the fold-3-trained stacker (stage-2 + CE) to test top-12 pairs and write stacked probabilities.

Stacker features (same order as stack_ce_fold3.py 'stack_all' with one CE model):
stage2_logit, base, rank, gap_top, n_strong, ce_logit, ce_rank, ce_gap (CE NaN outside the band).
Trains the stacker on ALL fold-3 S1 of the 194k OOF set (threshold passed in, chosen by cross-fit),
scores test pairs, writes q,t,p parquet for finalize_postprocess.py.
"""
import argparse, glob, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]


def ce_context(K, lv):
    by = defaultdict(list)
    for i, (q, _) in enumerate(K):
        if not np.isnan(lv[i]): by[q].append(i)
    rk = np.full(len(K), np.nan, np.float32); gap = np.full(len(K), np.nan, np.float32)
    for q, idx in by.items():
        vals = lv[idx]; order = np.argsort(-vals)
        rk[np.array(idx)[order]] = np.arange(1, len(idx) + 1); gap[idx] = vals.max() - vals
    return rk, gap


def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce-held", required=True)
    ap.add_argument("--ce-test-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True)
    keys, X = z["keys"], z["X"]; p2 = np.load(ROOT / "outputs/experiments/CL-003/train-oof-top12-v2.npy")
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list()))
    m3 = np.array([fold[q] == 3 for q in keys[:, 0].tolist()]); keys, X, p2 = keys[m3], X[m3], p2[m3]
    K = list(map(tuple, keys.tolist()))
    lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    y = np.array([lab[k] for k in K])
    ce = {(q, t): v for q, t, v in pl.read_parquet(a.ce_held).select("q", "t", "logit").iter_rows()}
    lv = np.array([ce.get(k, np.nan) for k in K], np.float32); rk, gp = ce_context(K, lv)
    F = np.column_stack([logit(p2), X[:, 0], X[:, 2], X[:, 3], X[:, 4], lv, rk, gp]).astype(np.float32)
    params = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=8)
    model = lgb.LGBMClassifier(**params).fit(F, y)
    probs = pl.read_parquet(ROOT / "outputs/experiments/CL-003/test_probs/CL-005-top12v2-v1.parquet")
    feats = ROOT / "outputs/experiments/CL-005/features-v2"
    ce_parts = {Path(f).stem: f for f in glob.glob(str(Path(a.ce_test_dir) / "*.parquet"))}
    out, missing = [], []
    for f in sorted(feats.glob("*.npz")):
        zz = np.load(f); q, t, Xt = zz["q"], zz["t"], zz["X"]
        Kt = list(zip(q.tolist(), t.tolist()))
        pt = probs.filter(pl.col("q").is_in(list(set(q.tolist()))))
        pmap = {(qq, tt): pp for qq, tt, pp in pt.select("q", "t", "p").iter_rows()}
        p2t = np.array([pmap[k] for k in Kt])
        if f.stem in ce_parts:
            c = {(qq, tt): v for qq, tt, v in pl.read_parquet(ce_parts[f.stem]).select("q", "t", "logit").iter_rows()}
        else:
            c = {}; missing.append(f.stem)
        lvt = np.array([c.get(k, np.nan) for k in Kt], np.float32); rkt, gpt = ce_context(Kt, lvt)
        Ft = np.column_stack([logit(p2t), Xt[:, 0], Xt[:, 2], Xt[:, 3], Xt[:, 4], lvt, rkt, gpt]).astype(np.float32)
        out.append(pl.DataFrame({"q": q, "t": t, "p": model.predict_proba(Ft)[:, 1]}))
    pl.concat(out).write_parquet(a.out)
    print(json.dumps({"rows": sum(len(o) for o in out), "shards_without_ce": missing}))


if __name__ == "__main__":
    main()
