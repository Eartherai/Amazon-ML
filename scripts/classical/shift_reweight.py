"""CL-010: covariate-shift check. Per S1 count of near-text candidates (core_jw>=0.85 & addr_tset>=0.7) in top-12;
reweight 194k train OOF S1 to each test country's count distribution; weighted macro F0.5 vs threshold."""
import glob, sys, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "scripts/classical"); import stage2_features_v2 as F
ix = {n: i for i, n in enumerate(F.NAMES)}
def near(X): return (X[:, ix["core_jw"]] >= 0.85) & (X[:, ix["addr_tset"]] >= 0.7)
def cnt(qs, m):
    c = Counter(np.asarray(qs)[m].tolist()); return c
CAP = 6
test_dist = {}
for c in ("France", "India", "US"):
    h = Counter()
    for f in sorted(glob.glob(f"outputs/experiments/CL-005/features-v2/{c}-*.npz")):
        z = np.load(f); m = near(z["X"]); per = cnt(z["q"], m)
        qs = set(z["q"].tolist())
        for q in qs: h[min(per.get(q, 0), CAP)] += 1
    tot = sum(h.values()); test_dist[c] = {k: h[k] / tot for k in range(CAP + 1)}
z = np.load("outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys = z["keys"]; X = z["X"]
p = np.load(sys.argv[1] if len(sys.argv) > 1 else "outputs/experiments/CL-003/train-oof-top12-v2.npy")
m = near(X); per = cnt(keys[:, 0], m)
s1s = sorted(set(keys[:, 0].tolist()))
tc = dict(pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(s1s))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
bucket = {q: min(per.get(q, 0), CAP) for q in s1s}
train_dist = Counter(bucket.values()); n = len(s1s)
print("near-text count dist  train:", {k: round(train_dist[k] / n, 3) for k in range(CAP + 1)})
for c, d in test_dist.items(): print(f"  test {c}:", {k: round(v, 3) for k, v in d.items()})
by = defaultdict(list)
for (q, t), pp in zip(keys.tolist(), p): by[q].append((t, pp))
def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)
thrs = [0.5, 0.55, 0.6, 0.67, 0.72, 0.78, 0.84, 0.9]
per_b = {b: {th: [] for th in thrs} for b in range(CAP + 1)}
for q in s1s:
    for th in thrs: per_b[bucket[q]][th].append(f05({t for t, pp in by[q] if pp >= th}, truth[q]))
mean_b = {b: {th: (np.mean(v) if v else np.nan) for th, v in d.items()} for b, d in per_b.items()}
print("per-bucket macro @0.67:", {b: round(mean_b[b][0.67], 4) for b in mean_b})
for c, d in list(test_dist.items()) + [("train", {k: train_dist[k] / n for k in range(CAP + 1)})]:
    row = {th: round(sum(d[b] * mean_b[b][th] for b in d if not np.isnan(mean_b[b][th])), 5) for th in thrs}
    print(c, "reweighted macro by threshold:", row)
