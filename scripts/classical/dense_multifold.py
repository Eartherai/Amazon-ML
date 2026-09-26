"""CL-050: multi-fold confirmation of the CE-decided dense rescue (the P0 validation after SUB011 = 0.97205).

For each fold h in 1..3 (India/US S1 of the 194k set):
  core  = multi-fold v2q + CE stack (CL-042 dump: stacker trained on the other folds), candidates restricted to the
          stage-1 band p>=0.02, max-probability ownership at the fold's inner threshold
  dense = held-fold-h bi-encoder top-10 new pairs, pruned at text-only p>=0.05 where p comes from the model that never
          saw fold h (CL-047 notfold{h}); features [log p_text_oof, CE logit from the fold-h-held 2-epoch CE, cos, rank]
          decided by a dense stacker trained ONLY on the other folds' dense pairs, threshold chosen on those folds
Reports exact macro F0.5 per fold and pooled for core and core + dense. Fold4 CLOSED.
Inputs in outputs/experiments/CL-050: f{h}-dense.parquet (q,t,cos,rank,p_text_oof) and ce2-f{h}-dense.parquet (q,t,logit).
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "outputs/experiments/CL-050"
PS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=8, random_state=0, deterministic=True)


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def main():
    st = pl.read_parquet(ROOT / "outputs/experiments/CL-042/multifold-stack-v2q-ce1.parquet").filter(pl.col("base") >= 0.02)
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
    truth = {q: set(r.split(",")) - {""} for q, r in gt.iter_rows()}
    s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
    dens = {}
    for h in (1, 2, 3):
        d = pl.read_parquet(D / f"f{h}-dense.parquet").join(pl.read_parquet(D / f"ce2-f{h}-dense.parquet").select("q", "t", pl.col("logit").alias("ce")), on=["q", "t"], how="left")
        d = d.filter(pl.col("p_text_oof") >= 0.05)
        dens[h] = d.with_columns(pl.Series("y", [int(t in truth.get(q, ())) for q, t in d.select("q", "t").iter_rows()]))
    FEAT = [pl.col("p_text_oof").log(), "ce", "cos", "rank"]
    all_q = {h: sorted(set(st.filter(pl.col("fold") == h)["q"].to_list())) for h in (1, 2, 3)}

    def run_fold(h, dense_model=None, tau=None):
        s = st.filter(pl.col("fold") == h); best = {}
        for q, t, p, th in s.select("q", "t", "prob", "thr").iter_rows():
            if p >= th and (t not in best or p > best[t][1]): best[t] = (q, p)
        pred = defaultdict(set)
        for t, (q, _) in best.items(): pred[q].add(t)
        if dense_model is not None:
            d = dens[h]; pr = dense_model.predict_proba(d.select(FEAT).to_numpy().astype(np.float32))[:, 1]; add = {}
            for i in np.argsort(-pr, kind="stable"):
                if pr[i] < tau: break
                q, t = d["q"][int(i)], d["t"][int(i)]
                if t not in best and t not in add: add[t] = q
            for t, q in add.items(): pred[q].add(t)
        per = {q: f05(pred.get(q, set()), truth[q]) for q in all_q[h]}
        return per

    rows, pooled_core, pooled_dense = [], {}, {}
    for h in (1, 2, 3):
        others = [k for k in (1, 2, 3) if k != h]
        tr = pl.concat([dens[k] for k in others])
        model = lgb.LGBMClassifier(**PS).fit(tr.select(FEAT).to_numpy().astype(np.float32), tr["y"].to_numpy())
        # threshold chosen on the other folds (their own dense pairs scored by a model trained on the remaining fold)
        grid = (0.5, 0.6, 0.7, 0.8, 0.9); score = {g: 0.0 for g in grid}
        for k in others:
            kk = [x for x in others if x != k][0]
            mk = lgb.LGBMClassifier(**PS).fit(dens[kk].select(FEAT).to_numpy().astype(np.float32), dens[kk]["y"].to_numpy())
            for g in grid: score[g] += float(np.mean(list(run_fold(k, mk, g).values())))
        tau = max(score, key=score.get)
        core, full = run_fold(h), run_fold(h, model, tau)
        pooled_core.update(core); pooled_dense.update(full)
        r = {"fold": h, "s1": len(core), "core": round(float(np.mean(list(core.values()))), 6), "core_plus_dense": round(float(np.mean(list(full.values()))), 6), "tau": tau,
             "dense_pairs": dens[h].height}
        r["delta"] = round(r["core_plus_dense"] - r["core"], 6); rows.append(r); print(json.dumps(r), flush=True)
    q = sorted(pooled_core)
    pc, pdn = float(np.mean([pooled_core[x] for x in q])), float(np.mean([pooled_dense[x] for x in q]))
    r = {"fold": "pooled", "s1": len(q), "core": round(pc, 6), "core_plus_dense": round(pdn, 6), "delta": round(pdn - pc, 6),
         "India_core": round(float(np.mean([pooled_core[x] for x in q if s1c[x] == "India"])), 6), "India_dense": round(float(np.mean([pooled_dense[x] for x in q if s1c[x] == "India"])), 6),
         "US_core": round(float(np.mean([pooled_core[x] for x in q if s1c[x] == "US"])), 6), "US_dense": round(float(np.mean([pooled_dense[x] for x in q if s1c[x] == "US"])), 6)}
    rows.append(r); print(json.dumps(r), flush=True)
    pl.DataFrame(rows, strict=False).write_csv(D / "dense_multifold.csv")


if __name__ == "__main__":
    main()
