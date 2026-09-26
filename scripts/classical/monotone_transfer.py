"""CL-045: monotone constraints and stronger regularization for stage-2 transfer robustness.

Variants on TEXT+QNORM (the honest unseen-country analogue): BASE (CL-039 params), MONO (+1 on similarity features,
-1 on conflict / unmatched / missing-evidence features, 0 otherwise, raw features only), REG (fewer leaves, larger
leaves, stronger L2, 50% feature sampling), MONO+REG. Protocol as CL-030/039 (source folds 1-2 train, source cross-fit
threshold, target fold-3 exact macro F0.5). Fold4 CLOSED.
"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/classical")); import stage2_features_v2 as F
from transfer_matrix import PARAMS, macro
from stage2_qnorm import qnorm

UP = {"core_eq", "core_jw", "core_tset", "core_tsort", "core_contain", "core_jacc", "core_fcover_q", "core_fcover_t", "first_core_eq", "name_idf_cos",
      "name_idf_shared_max", "sk_eq", "sk_jw", "sk_tset", "compact_jw", "initials_match", "addr_tset", "addr_jacc", "addr_idf_cos", "addr_idf_shared_max",
      "addr_fcover_q", "addr_fcover_t", "state_eq", "num_exact", "num_fuzzy", "digits_jw", "addr_sk_cover_q", "addr_sk_cover_t", "addr_sk_tset"}
DOWN = {"name_idf_qmiss_max", "name_idf_tmiss_max", "addr_idf_qmiss_max", "addr_idf_tmiss_max", "state_conflict", "num_q_unmatched", "num_t_unmatched", "num_sub"}
REG = dict(num_leaves=31, min_child_samples=200, reg_lambda=10.0, colsample_bytree=0.5)


def main():
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"].astype(np.float32)
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list()))
    lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    qa, ta = keys[:, 0], keys[:, 1]
    y = np.array([lab[(q, t)] for q, t in zip(qa.tolist(), ta.tolist())], np.int8)
    fo = np.array([fold[q] for q in qa.tolist()]); co = np.array([s1c[q] for q in qa.tolist()])
    M = np.hstack([X[:, 9:57], qnorm(qa, X)])
    mono = [1 if n in UP else -1 if n in DOWN else 0 for n in F.NAMES[9:57]] + [0] * 48
    V = {"BASE": dict(PARAMS), "MONO": dict(PARAMS, monotone_constraints=mono, monotone_constraints_method="advanced"),
         "REG": dict(PARAMS, **REG), "MONO+REG": dict(PARAMS, **REG, monotone_constraints=mono, monotone_constraints_method="advanced")}
    grid = np.arange(0.3, 0.96, 0.02); rows = []
    for vname, prm in V.items():
        for src, tgt in (("US", "India"), ("India", "US")):
            m1, m2 = (co == src) & (fo == 1), (co == src) & (fo == 2)
            p1 = lgb.LGBMClassifier(**prm).fit(M[m2], y[m2]).predict_proba(M[m1])[:, 1]
            p2 = lgb.LGBMClassifier(**prm).fit(M[m1], y[m1]).predict_proba(M[m2])[:, 1]
            sq = sorted(set(qa[m1 | m2].tolist()))
            thr = max(grid, key=lambda th: macro(np.concatenate([qa[m1], qa[m2]]), np.concatenate([ta[m1], ta[m2]]), np.concatenate([p1, p2]), th, sq, truth))
            model = lgb.LGBMClassifier(**prm).fit(M[m1 | m2], y[m1 | m2]); out = {}
            for c in (src, tgt):
                mt = (co == c) & (fo == 3); out[c] = macro(qa[mt], ta[mt], model.predict_proba(M[mt])[:, 1], thr, sorted(set(qa[mt].tolist())), truth)
            rows.append({"variant": vname, "source": src, "in_domain": round(out[src], 6), "transfer": round(out[tgt], 6), "thr": round(float(thr), 2)})
            print(json.dumps(rows[-1]), flush=True)
    out = ROOT / "outputs/experiments/CL-045"; out.mkdir(parents=True, exist_ok=True); pl.DataFrame(rows).write_csv(out / "monotone_transfer.csv")


if __name__ == "__main__":
    main()
