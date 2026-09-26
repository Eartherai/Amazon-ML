"""CL-039: do query-relative pair features transfer better? (Track: unseen-domain robustness)

TEXT = 48 pair features. REL adds, for each of the 20 strongest similarity features, (x - max over the S1's top-12)
and the within-S1 dense rank; QNORM replaces raw features by within-S1 z-scores. Same protocol as CL-030:
train source folds 1-2, threshold by source cross-fit, evaluate target fold-3 (exact macro F0.5, full truth). Fold4 CLOSED.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/classical")); import stage2_features_v2 as F
from transfer_matrix import PARAMS, macro

SIM = ["core_jw", "core_tset", "core_jacc", "core_fcover_q", "core_fcover_t", "name_idf_cos", "name_idf_shared_max", "sk_jw", "sk_tset", "compact_jw",
       "addr_tset", "addr_jacc", "addr_idf_cos", "addr_idf_shared_max", "addr_fcover_q", "addr_fcover_t", "num_exact", "num_fuzzy", "addr_sk_cover_q", "digits_jw"]


def main():
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True)
    keys, X = z["keys"], z["X"].astype(np.float32)
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list()))
    lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    qa, ta = keys[:, 0], keys[:, 1]
    y = np.array([lab[(q, t)] for q, t in zip(qa.tolist(), ta.tolist())], np.int8)
    fo = np.array([fold[q] for q in qa.tolist()]); co = np.array([s1c[q] for q in qa.tolist()])
    idx = [F.NAMES.index(n) for n in SIM]
    df = pl.DataFrame({"q": qa, **{f"f{j}": X[:, j] for j in idx}})
    rel = df.select([(pl.col(f"f{j}") - pl.col(f"f{j}").max().over("q")).alias(f"r{j}") for j in idx] +
                    [pl.col(f"f{j}").rank("dense", descending=True).over("q").alias(f"k{j}") for j in idx]).to_numpy().astype(np.float32)
    TEXT = X[:, 9:57]
    zq = pl.DataFrame({"q": qa, **{f"c{j}": X[:, j] for j in range(9, 57)}}).select(
        [((pl.col(f"c{j}") - pl.col(f"c{j}").mean().over("q")) / (pl.col(f"c{j}").std().over("q") + 1e-3)).alias(f"z{j}") for j in range(9, 57)]).to_numpy().astype(np.float32)
    import os
    V = {"TEXT": TEXT, "TEXT+REL": np.hstack([TEXT, rel]), "REL_ONLY": rel, "TEXT+QNORM": np.hstack([TEXT, zq])}
    if os.environ.get("FULLMODE"):
        V = {"FULL": X, "FULL+QNORM": np.hstack([X, zq])}
    grid = np.arange(0.3, 0.96, 0.02); rows = []
    if os.environ.get("FULLMODE"):  # in-domain both countries: train folds 1-2 (all), threshold by fold1<->fold2 cross-fit, evaluate fold 3 per country
        for vname, M in V.items():
            m1, m2 = fo == 1, fo == 2
            p1 = lgb.LGBMClassifier(**PARAMS).fit(M[m2], y[m2]).predict_proba(M[m1])[:, 1]; p2 = lgb.LGBMClassifier(**PARAMS).fit(M[m1], y[m1]).predict_proba(M[m2])[:, 1]
            sq = sorted(set(qa[m1 | m2].tolist())); thr = max(grid, key=lambda th: macro(np.concatenate([qa[m1], qa[m2]]), np.concatenate([ta[m1], ta[m2]]), np.concatenate([p1, p2]), th, sq, truth))
            model = lgb.LGBMClassifier(**PARAMS).fit(M[m1 | m2], y[m1 | m2]); m3 = fo == 3; p3 = model.predict_proba(M[m3])[:, 1]
            res = {c: round(macro(qa[m3 & (co == c)], ta[m3 & (co == c)], p3[co[m3] == c], thr, sorted(set(qa[m3 & (co == c)].tolist())), truth), 6) for c in ("US", "India")}
            res["all"] = round(macro(qa[m3], ta[m3], p3, thr, sorted(set(qa[m3].tolist())), truth), 6)
            print(json.dumps({"variant": vname, "setting": "both-country in-domain fold3", **res, "thr": round(float(thr), 2)}), flush=True)
    for vname, M in V.items():
        for src, tgt in (("US", "India"), ("India", "US")):
            m1, m2 = (co == src) & (fo == 1), (co == src) & (fo == 2)
            p1 = lgb.LGBMClassifier(**PARAMS).fit(M[m2], y[m2]).predict_proba(M[m1])[:, 1]
            p2 = lgb.LGBMClassifier(**PARAMS).fit(M[m1], y[m1]).predict_proba(M[m2])[:, 1]
            sq = sorted(set(qa[m1 | m2].tolist())); sqa = np.concatenate([qa[m1], qa[m2]]); sta = np.concatenate([ta[m1], ta[m2]]); sp = np.concatenate([p1, p2])
            thr = max(grid, key=lambda th: macro(sqa, sta, sp, th, sq, truth))
            model = lgb.LGBMClassifier(**PARAMS).fit(M[m1 | m2], y[m1 | m2])
            out = {}
            for c in (src, tgt):
                mt = (co == c) & (fo == 3); tq = sorted(set(qa[mt].tolist()))
                out[c] = macro(qa[mt], ta[mt], model.predict_proba(M[mt])[:, 1], thr, tq, truth)
            row = {"variant": vname, "source": src, "in_domain": round(out[src], 6), "transfer": round(out[tgt], 6), "thr": round(float(thr), 2)}
            rows.append(row); print(json.dumps(row), flush=True)
    out = ROOT / "outputs/experiments/CL-039"; out.mkdir(parents=True, exist_ok=True); pl.DataFrame(rows).write_csv(out / ("relative_features_transfer_full.csv" if os.environ.get("FULLMODE") else "relative_features_transfer.csv"))


if __name__ == "__main__":
    main()
