"""CL-041: stage-2 v2q = the 57 v2 features + 48 within-S1 z-scores of the pair features (QNORM, CL-039).

Reproduces train_stage2_200k.py exactly (same 194k top-12 matrix, same hash 3-fold S1 partition, same PARAMS) with the
extra features, writes train-oof-top12-v2q.npy, the final model stage2-200k-top12-v2q.txt, and test probabilities
CL-005-top12v2q.parquet (q, t, base, p) from the saved CL-005 test feature shards. With --bag, test p is the mean of the
three fold models (same distribution as the OOF p the stacker trains on; CL-043) -> CL-005-top12v2q-bag.parquet. Fold4 CLOSED.
"""
import hashlib, json, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/CL-003"
PARAMS = dict(objective="binary", n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=9, random_state=0, deterministic=True, force_col_wise=True)


def qnorm(q, X):
    df = pl.DataFrame({"q": q, **{f"c{j}": X[:, j] for j in range(9, 57)}})
    return df.select([((pl.col(f"c{j}") - pl.col(f"c{j}").mean().over("q")) / (pl.col(f"c{j}").std().over("q").fill_null(0.0) + 1e-3)).alias(f"z{j}")
                      for j in range(9, 57)]).to_numpy().astype(np.float32)


def main():
    import sys; bag = "--bag" in sys.argv
    t0 = time.time()
    z = np.load(OUT / "trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"].astype(np.float32)
    top = pl.concat([pl.read_parquet(OUT / f"s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    y = np.array([lab[(q, t)] for q, t in keys.tolist()], np.int8)
    XX = np.hstack([X, qnorm(keys[:, 0], X)])
    part = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 3 for q in keys[:, 0].tolist()])
    oof = np.zeros(len(y)); fms = []
    for k in range(3):
        fm = lgb.LGBMClassifier(**PARAMS).fit(XX[part != k], y[part != k]); fms.append(fm)
        oof[part == k] = fm.predict_proba(XX[part == k])[:, 1]
        fm.booster_.save_model(str(OUT / f"stage2-200k-top12-v2q-fold{k}.txt"))
        print(json.dumps({"fold": k, "s": round(time.time() - t0)}), flush=True)
    if not bag:
        np.save(OUT / "train-oof-top12-v2q.npy", oof)
    else:
        assert np.allclose(oof, np.load(OUT / "train-oof-top12-v2q.npy"), atol=1e-6), "OOF not reproduced"
    if bag:
        class Bag:
            def predict_proba(self, M): return np.column_stack([np.zeros(len(M)), np.mean([m.predict_proba(M)[:, 1] for m in fms], axis=0)])
        model = Bag(); path = OUT / "stage2-200k-top12-v2q-fold0.txt"
    else:
        model = lgb.LGBMClassifier(**PARAMS).fit(XX, y); path = OUT / "stage2-200k-top12-v2q.txt"; model.booster_.save_model(str(path))
    base = pl.read_parquet(OUT / "test_probs/CL-005-top12v2-v1.parquet").select("q", "t", "base").unique(["q", "t"])
    parts = []
    for f in sorted((ROOT / "outputs/experiments/CL-005/features-v2").glob("*.npz")):
        zz = np.load(f); Xt = zz["X"].astype(np.float32)
        parts.append(pl.DataFrame({"q": zz["q"], "t": zz["t"], "p": model.predict_proba(np.hstack([Xt, qnorm(zz["q"], Xt)]))[:, 1]}))
    tp = pl.concat(parts).join(base, on=["q", "t"], how="left").select("q", "t", "base", "p")
    tp.write_parquet(OUT / ("test_probs/CL-005-top12v2q-bag.parquet" if bag else "test_probs/CL-005-top12v2q.parquet"))
    print(json.dumps({"oof_pairs": len(y), "test_pairs": tp.height, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
