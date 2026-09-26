"""CL-047: fold-held text-only pair models (48 v2 pair features, train_textonly.py PARAMS) for honest dense-rescue
evaluation: model 'notfold{h}' is trained on top-12 pairs of S1 in folds != h and is the only model allowed to score
fold-h dense pairs. Also rescores the fold-3 dense top-10 pairs (CL-021 X caches) -> CL-047/f3-{c}-dense-top10-oofp.parquet.
Fold4 CLOSED."""
import json, sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/experiments/CL-047"; OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(objective="binary", n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=9, random_state=0, deterministic=True, force_col_wise=True)
z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"][:, 9:57].astype(np.float32)
top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list())); lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
y = np.array([lab[(q, t)] for q, t in keys.tolist()], np.int8); fo = np.array([fold[q] for q in keys[:, 0].tolist()])
for h in (1, 2, 3):
    m = lgb.LGBMClassifier(**PARAMS).fit(X[fo != h], y[fo != h]); m.booster_.save_model(str(OUT / f"textonly-notfold{h}.txt"))
    print(json.dumps({"model": f"notfold{h}", "train_pairs": int((fo != h).sum())}), flush=True)
    if h == 3:
        for c in ("India", "US"):
            d = pl.read_parquet(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored.parquet")
            Xd = np.load(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored-X.npy")
            d.with_columns(pl.Series("p_text_oof", m.predict_proba(Xd)[:, 1])).write_parquet(OUT / f"f3-{c}-dense-top10-oofp.parquet")
