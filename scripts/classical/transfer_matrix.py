"""CL-030: cross-country transfer stress tests for the stage-2 pair matcher, with rank/context ablations.

Protocol per (source -> target, variant): train on SOURCE-country S1 of folds 1-2; choose the ownership decision
threshold by 2-fold cross-fit inside the source (fold1<->fold2), never on target labels; evaluate exact macro F0.5
(full truth, retrieval misses count) on TARGET-country fold-3 S1. In-domain reference: source == target.
Also reports the target-oracle threshold (calibration shift = oracle - transferred). Fold4 CLOSED.

Caveat: the first-stage base score (features 0-8) came from a stage-1 model trained on both countries, so variants
that use it are optimistic for transfer; the TEXT variant (pair features only) is fully source-trained.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/classical")); import stage2_features_v2 as F
PARAMS = dict(objective="binary", n_estimators=500, learning_rate=0.06, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1, n_jobs=8, random_state=0, deterministic=True, force_col_wise=True)
N = F.NAMES
VARIANTS = {
    "FULL": list(range(57)),
    "NO_RANK": [i for i in range(57) if N[i] != "rank"],
    "NO_ABS_BASE": [i for i in range(57) if N[i] not in ("base", "base_logit", "rank")],  # relative context only (gap_top, n_strong, n_mid, sib_*)
    "TEXT": list(range(9, 57)),
    "TEXT_NO_SCRIPT": [i for i in range(9, 57) if N[i] not in ("q_nonlatin", "t_nonlatin")],
}


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def decide(qa, ta, p, th):
    best = {}
    for q, t, v in zip(qa, ta, p):
        if v >= th and (t not in best or v > best[t][1]): best[t] = (q, v)
    out = defaultdict(set)
    for t, (q, _) in best.items(): out[q].add(t)
    return out


def macro(qa, ta, p, th, qs, truth):
    d = decide(qa, ta, p, th); return float(np.mean([f05(d.get(q, set()), truth[q]) for q in qs]))


def main():
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True)
    keys, X = z["keys"], z["X"].astype(np.float32)
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list()))
    lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
    qa, ta = keys[:, 0], keys[:, 1]
    y = np.array([lab[(q, t)] for q, t in zip(qa.tolist(), ta.tolist())], np.int8)
    fo = np.array([fold[q] for q in qa.tolist()]); co = np.array([s1c[q] for q in qa.tolist()])
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    grid = np.arange(0.3, 0.96, 0.02); rows = []
    for vname, cols in VARIANTS.items():
        for src in ("US", "India"):
            m1, m2 = (co == src) & (fo == 1), (co == src) & (fo == 2)
            # source-internal cross-fit for the threshold
            p1 = lgb.LGBMClassifier(**PARAMS).fit(X[m2][:, cols], y[m2]).predict_proba(X[m1][:, cols])[:, 1]
            p2 = lgb.LGBMClassifier(**PARAMS).fit(X[m1][:, cols], y[m1]).predict_proba(X[m2][:, cols])[:, 1]
            sq = sorted(set(qa[m1 | m2].tolist())); sqa = np.concatenate([qa[m1], qa[m2]]); sta = np.concatenate([ta[m1], ta[m2]]); sp = np.concatenate([p1, p2])
            thr = max(grid, key=lambda th: macro(sqa, sta, sp, th, sq, truth))
            model = lgb.LGBMClassifier(**PARAMS).fit(X[m1 | m2][:, cols], y[m1 | m2])
            for tgt in ("US", "India"):
                mt = (co == tgt) & (fo == 3); tq = sorted(set(qa[mt].tolist())); pt = model.predict_proba(X[mt][:, cols])[:, 1]
                m_tr = macro(qa[mt], ta[mt], pt, thr, tq, truth)
                orc = max((macro(qa[mt], ta[mt], pt, th, tq, truth), th) for th in grid[::2])
                row = {"variant": vname, "source": src, "target": tgt, "src_thr": round(float(thr), 2), "macro": round(m_tr, 6),
                       "target_oracle_thr": round(float(orc[1]), 2), "macro_at_oracle_thr": round(orc[0], 6), "s1": len(tq)}
                rows.append(row); print(json.dumps(row), flush=True)
    out = ROOT / "outputs/experiments/CL-030"; out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out / "transfer_matrix.csv")


if __name__ == "__main__":
    main()
