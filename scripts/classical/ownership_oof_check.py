"""CL-008: does max-probability target ownership help on top of stage-2? 194k training S1, grouped 3-fold OOF."""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_stage2_exp044 as P
import train_stage2_200k as T
ROOT = Path(__file__).resolve().parents[2]
z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True)
keys = [tuple(k) for k in z["keys"]]; X = z["X"]
top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
y = np.array([lab[k] for k in keys])
part = np.array([int(hashlib.sha256(k[0].encode()).hexdigest(), 16) % 3 for k in keys])
oof = np.zeros(len(keys))
for k in range(3):
    oof[part == k] = lgb.LGBMClassifier(**T.PARAMS).fit(X[part != k], y[part != k]).predict_proba(X[part == k])[:, 1]
thr = json.loads((ROOT / "outputs/experiments/CL-003/stage2-200k-top12-v2.json").read_text())["threshold"]
s1s = sorted({k[0] for k in keys})
gt = pl.read_csv(P.TRAIN / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(s1s))
truth = {q: set(raw.split(",")) if raw else set() for q, raw in gt.iter_rows()}
country = {s: "?" for s in s1s}
plain, best = defaultdict(set), {}
for (q, t), p in zip(keys, oof):
    if p >= thr:
        plain[q].add(t)
        if t not in best or p > best[t][1]: best[t] = (q, p)
own = defaultdict(set)
for t, (q, p) in best.items(): own[q].add(t)
e0, per0 = P.evaluate(s1s, plain, truth, country); e1, per1 = P.evaluate(s1s, own, truth, country)
d = np.array([per1[s] - per0[s] for s in s1s]); rng = np.random.default_rng(0)
b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]
conf = sum(len(v) for v in plain.values()) - sum(len(v) for v in own.values())
print(json.dumps({"plain": e0["macro_f0_5"], "ownership": e1["macro_f0_5"], "delta": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "links_removed": conf, "s1": len(s1s)}, indent=1))
np.save(ROOT / "outputs/experiments/CL-003/train-oof-top12-v2.npy", oof)
