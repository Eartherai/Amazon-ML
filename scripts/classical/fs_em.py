"""CL-031: Fellegi-Sunter pilot (Track D). Unsupervised EM on each country's fold-3 candidate pairs (no labels);
comparison vectors are binned text-only pair features (quantile bins fit on the same unlabeled pairs).
Decision: posterior >= 0.5 (fully unsupervised) plus target ownership; labels are used only to score.
Also reports the macro at the best threshold on the target (oracle, reference only) and pair AUC. Fold4 CLOSED.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/classical")); import stage2_features_v2 as F
COMP = ["core_jw", "core_tset", "first_core_eq", "name_idf_cos", "sk_jw", "compact_jw", "addr_idf_cos", "addr_tset", "addr_sk_cover_q",
        "num_exact", "num_sub", "num_q_unmatched", "state_conflict", "t_addr_missing"]


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def macro(qa, ta, p, th, qs, truth):
    best = {}
    for q, t, v in zip(qa, ta, p):
        if v >= th and (t not in best or v > best[t][1]): best[t] = (q, v)
    d = defaultdict(set)
    for t, (q, _) in best.items(): d[q].add(t)
    return float(np.mean([f05(d.get(q, set()), truth[q]) for q in qs]))


def binize(X):
    B = np.zeros(X.shape, np.int64); K = []
    for j in range(X.shape[1]):
        v = X[:, j]; u = np.unique(v)
        if len(u) <= 6:
            B[:, j] = np.searchsorted(u, v); K.append(len(u))
        else:
            e = np.unique(np.quantile(v, [0.2, 0.4, 0.6, 0.8])); B[:, j] = np.searchsorted(e, v, side="right"); K.append(len(e) + 1)
    return B, K


def em(B, K, iters=60):
    n, d = B.shape; lam = 0.15
    m = [np.linspace(1, 3, k) / np.linspace(1, 3, k).sum() for k in K]; u = [np.linspace(3, 1, k) / np.linspace(3, 1, k).sum() for k in K]
    for _ in range(iters):
        lm = np.log(lam) + sum(np.log(m[j][B[:, j]]) for j in range(d)); lu = np.log(1 - lam) + sum(np.log(u[j][B[:, j]]) for j in range(d))
        g = 1 / (1 + np.exp(lu - lm)); lam = float(g.mean())
        m = [np.bincount(B[:, j], weights=g, minlength=K[j]) + 1e-3 for j in range(d)]; m = [x / x.sum() for x in m]
        u = [np.bincount(B[:, j], weights=1 - g, minlength=K[j]) + 1e-3 for j in range(d)]; u = [x / x.sum() for x in u]
    return g, lam


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1); npos = y.sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def main():
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True)
    keys, X = z["keys"], z["X"]
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list()))
    lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    qa, ta = keys[:, 0], keys[:, 1]; fo = np.array([fold[q] for q in qa.tolist()]); co = np.array([s1c[q] for q in qa.tolist()])
    cols = [F.NAMES.index(c) for c in COMP]; rows = []
    for c in ("India", "US"):
        m = (co == c) & (fo == 3); B, K = binize(X[m][:, cols]); g, lam = em(B, K)
        y = np.array([lab[(q, t)] for q, t in zip(qa[m].tolist(), ta[m].tolist())]); qs = sorted(set(qa[m].tolist()))
        un = macro(qa[m], ta[m], g, 0.5, qs, truth)
        orc = max((macro(qa[m], ta[m], g, th, qs, truth), th) for th in (0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99))
        row = {"country": c, "pairs": int(m.sum()), "lambda": round(lam, 4), "true_rate": round(float(y.mean()), 4), "auc": round(auc(g, y), 5),
               "macro_unsup_0.5": round(un, 6), "macro_oracle_thr": round(orc[0], 6), "oracle_thr": orc[1]}
        rows.append(row); print(json.dumps(row), flush=True)
        np.save(ROOT / f"outputs/experiments/CL-030/fs-posterior-fold3-{c}.npy", g)
    pl.DataFrame(rows).write_csv(ROOT / "outputs/experiments/CL-030/fs_em.csv")


if __name__ == "__main__":
    main()
