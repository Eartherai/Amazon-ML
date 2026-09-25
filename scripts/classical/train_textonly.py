"""CL-009: stage-2 without first-stage-derived features (base, rank, gap, strong-sibling context).

Motivation: test first-stage scores are shifted (obvious matches: test India base mean 0.934 vs
train OOF 0.984). Uses cached 194k trainmat (v2); evaluated on fixed 6k; threshold from train inner OOF.
"""
import json, sys, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_features_v2 as F
ROOT = Path(__file__).resolve().parents[2]
DROP = {"base", "base_logit", "rank", "gap_top", "n_strong", "n_mid", "sib_name", "sib_addr", "sib_dup"}
keep = [i for i, n in enumerate(F.NAMES) if n not in DROP]
names = [F.NAMES[i] for i in keep]
PARAMS = dict(objective="binary", n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=9)
def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    p, r = tp / len(P), tp / len(T); return 1.25 * p * r / (0.25 * p + r)
z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True)
keys, X, ekeys, EX = z["keys"], z["X"][:, keep], z["ekeys"], z["EX"][:, keep]
top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
y = np.array([lab[(q, t)] for q, t in keys.tolist()])
part = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 3 for q in keys[:, 0].tolist()])
oof = np.zeros(len(y))
for k in range(3):
    oof[part == k] = lgb.LGBMClassifier(**PARAMS).fit(X[part != k], y[part != k]).predict_proba(X[part == k])[:, 1]
s1s = sorted(set(keys[:, 0].tolist()))
gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(s1s))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
by = defaultdict(list)
for (q, t), pp in zip(keys.tolist(), oof): by[q].append((t, pp))
curve = {round(th, 2): float(np.mean([f05({t for t, pp in by[q] if pp >= th}, T) for q, T in truth.items()])) for th in np.arange(0.4, 0.86, 0.03)}
thr = max(curve, key=curve.get)
model = lgb.LGBMClassifier(**PARAMS).fit(X, y)
path = ROOT / "outputs/experiments/CL-003/stage2-200k-top12-textonly.txt"
model.booster_.save_model(str(path))
et = json.loads((Path("/Users/earther/.codex/worktrees/aml-neural-warroom/Amazon ML Challange/outputs/experiments/warroom_neural/EXP-044/truth.json")).read_text())
ep = model.predict_proba(EX)[:, 1]
eb = defaultdict(list)
for (q, t), pp in zip(ekeys.tolist(), ep): eb[q].append((t, pp))
ev = {round(th, 2): float(np.mean([f05({t for t, pp in eb[q] if pp >= th}, set(T)) for q, T in et.items()])) for th in (thr - 0.1, thr, thr + 0.05)}
meta = {"features": names, "threshold": thr, "train_inner_curve": curve, "eval6k": ev, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dropped": sorted(DROP)}
(ROOT / "outputs/experiments/CL-003/stage2-200k-top12-textonly.json").write_text(json.dumps(meta, indent=1))
np.save(ROOT / "outputs/experiments/CL-003/train-oof-textonly.npy", oof)
print(json.dumps({"thr": thr, "train_inner_best": curve[thr], "eval6k": ev}))
