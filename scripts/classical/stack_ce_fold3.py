"""CL-016: stack cross-encoder held-fold-3 logits with stage-2 OOF probabilities on ~65k fold-3 S1.

Population: fold-3 S1 of the 194k stage-2 training set (keys from trainmat-top12-v2.npz, OOF stage-2
probabilities from train-oof-top12-v2.npy). CE logits exist for band pairs (first-stage >= 0.02);
elsewhere NaN. Cross-fit over two hash halves of S1 with an inner grouped 3-fold threshold choice.
Reports exact macro F0.5 (full truth incl. retrieval misses), paired bootstrap CI vs stage-2 alone,
and the same after max-probability target ownership. Fold4 CLOSED.
"""
import hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
ce_files = sys.argv[1:]
z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True)
keys, X = z["keys"], z["X"]
p2 = np.load(ROOT / "outputs/experiments/CL-003/train-oof-top12-v2.npy")
top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list()))
m3 = np.array([fold[q] == 3 for q in keys[:, 0].tolist()])
keys, X, p2 = keys[m3], X[m3], p2[m3]
K = list(map(tuple, keys.tolist()))
lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
y = np.array([lab[k] for k in K])
s1s = sorted({k[0] for k in K})
gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(s1s))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
tc = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
feats, names = [], []
lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
feats.append(lg(p2)); names.append("stage2_logit")
for i, n in ((0, "base"), (2, "rank"), (3, "gap_top"), (4, "n_strong")):
    feats.append(X[:, i]); names.append(n)
for f in ce_files:
    ce = pl.read_parquet(f).select("q", "t", "logit")
    d = {(q, t): v for q, t, v in ce.iter_rows()}
    lv = np.array([d.get(k, np.nan) for k in K], dtype=np.float32)
    # CE context within S1: rank and gap to the S1's best CE logit
    by = defaultdict(list)
    for i, (q, _) in enumerate(K):
        if not np.isnan(lv[i]): by[q].append(i)
    rk = np.full(len(K), np.nan, np.float32); gap = np.full(len(K), np.nan, np.float32)
    for q, idx in by.items():
        vals = lv[idx]; order = np.argsort(-vals)
        rk[np.array(idx)[order]] = np.arange(1, len(idx) + 1); gap[idx] = vals.max() - vals
    tag = Path(f).stem
    feats += [lv, rk, gap]; names += [f"{tag}_logit", f"{tag}_rank", f"{tag}_gap"]
    print(json.dumps({"ce": tag, "coverage": float(np.mean(~np.isnan(lv))), "pos_cov": float(np.mean(~np.isnan(lv[y == 1])))}))
F = np.vstack(feats).T.astype(np.float32)
half = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q, _ in K])


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def decide(idx, prob, thr, own):
    if own:
        best = {}
        for i, pr in zip(idx, prob):
            if pr >= thr and (K[i][1] not in best or pr > best[K[i][1]][1]): best[K[i][1]] = (K[i][0], pr)
        out = defaultdict(set)
        for t, (q, _) in best.items(): out[q].add(t)
        return out
    out = defaultdict(set)
    for i, pr in zip(idx, prob):
        if pr >= thr: out[K[i][0]].add(K[i][1])
    return out


def macro(pred, qs): return {q: f05(pred.get(q, set()), truth[q]) for q in qs}


params = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=8)
results = {}
for name, cols in (("stage2_only", [0]), ("stage2+ctx", [0, 1, 2, 3, 4]), ("stack_all", list(range(F.shape[1])))):
    for own in (False, True):
        per = {}
        for h in (0, 1):
            tr, te = np.where(half != h)[0], np.where(half == h)[0]
            trq = sorted({K[i][0] for i in tr}); inner = {q: j % 3 for j, q in enumerate(trq)}
            ip = np.zeros(len(tr))
            for k in range(3):
                msk = np.array([inner[K[i][0]] != k for i in tr])
                ip[~msk] = lgb.LGBMClassifier(**params).fit(F[tr[msk]][:, cols], y[tr[msk]]).predict_proba(F[tr[~msk]][:, cols])[:, 1]
            best = max((np.mean(list(macro(decide(tr, ip, th, own), trq).values())), th) for th in np.arange(0.4, 0.95, 0.02))
            pr = lgb.LGBMClassifier(**params).fit(F[tr][:, cols], y[tr]).predict_proba(F[te][:, cols])[:, 1]
            teq = sorted({K[i][0] for i in te})
            per.update(macro(decide(te, pr, best[1], own), teq))
        results[(name, own)] = per
        print(json.dumps({"model": name, "ownership": own, "macro": float(np.mean(list(per.values())))}), flush=True)
base = results[("stage2_only", False)]
qs = sorted(base)
for (name, own), per in results.items():
    d = np.array([per[q] - base[q] for q in qs]); rng = np.random.default_rng(0)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    ind = np.mean([per[q] for q in qs if tc[q] == "India"]); us = np.mean([per[q] for q in qs if tc[q] == "US"])
    print(f"{name:12s} own={own!s:5s} macro={np.mean([per[q] for q in qs]):.6f} delta={d.mean():+.6f} ci=[{np.percentile(b, 2.5):+.5f},{np.percentile(b, 97.5):+.5f}] India={ind:.5f} US={us:.5f}")
