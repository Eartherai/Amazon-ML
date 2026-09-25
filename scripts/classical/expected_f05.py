"""CL-019: expected-F0.5 per-S1 set decision on 194k OOF (calibrate on hash half A, evaluate on half B and vice versa).
For each S1 with calibrated p sorted desc: E[F](k) ~= 1.25*S_k / (k + 0.25*(sum_p + m)), E[F](0) ~= prod(1-p)*exp(-m).
m = expected true links outside candidates per S1 (fit on the calibration half). Compare vs global threshold."""
import hashlib, json, math
from collections import defaultdict
import numpy as np, polars as pl
from sklearn.isotonic import IsotonicRegression
z = np.load("outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys = z["keys"]
p = np.load("outputs/experiments/CL-003/train-oof-top12-v2.npy")
K = list(map(tuple, keys.tolist()))
top = pl.concat([pl.read_parquet(f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
y = np.array([lab[k] for k in K])
s1s = sorted(set(keys[:, 0].tolist()))
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(s1s))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
half = {q: int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in s1s}
hk = np.array([half[q] for q, _ in K])
def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)
res = defaultdict(dict)
for h in (0, 1):
    cal = hk != h; ev = hk == h
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p[cal], y[cal])
    pc = iso.predict(p)
    cq = [q for q in s1s if half[q] != h]
    m_hat = np.mean([len(truth[q]) for q in cq]) - (y[cal].sum() / len(cq))  # expected true links outside top-12 per S1
    by = defaultdict(list)
    for i in np.where(ev)[0]: by[K[i][0]].append((K[i][1], pc[i], p[i]))
    eq = [q for q in s1s if half[q] == h]
    for name, fn in [("thr0.67", None), ("expF", "exp"), ("expF_m0", "exp0")]:
        scores = []
        for q in eq:
            c = sorted(by.get(q, []), key=lambda x: -x[1])
            if fn is None:
                P = {t for t, _, raw in c if raw >= 0.67}
            else:
                m = m_hat if fn == "exp" else 0.0
                ps = np.array([x[1] for x in c]); N = ps.sum() + m
                best_k, best = 0, float(np.prod(1 - ps) * math.exp(-m))
                S = 0.0
                for k in range(1, len(ps) + 1):
                    S += ps[k - 1]; v = 1.25 * S / (k + 0.25 * N)
                    if v > best: best, best_k = v, k
                P = {t for t, _, _ in c[:best_k]}
            scores.append(f05(P, truth[q]))
        res[name][h] = float(np.mean(scores))
    print(json.dumps({"half": h, "m_hat": round(float(m_hat), 4), **{k: round(v[h], 5) for k, v in res.items()}}), flush=True)
print({k: round(np.mean(list(v.values())), 5) for k, v in res.items()})
