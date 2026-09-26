"""CL-062: retrieval-expansion rescue on top of the post-corrector base (candidates_mfu_f*, pooled 0.987277).

Extra candidate buckets per fold (all fold-held signals):
  sparse_low : stage-1 top-12 pairs below the band (base < 0.02) -> multi-fold stack prob (CE absent), base, rank
  dense_prun : dense top-10 pairs pruned by text-only p_text_oof < 0.05 -> p_text_oof, fold-held 2-epoch CE logit, cos, rank
  dense_1150 : dense ranks 11-50 -> p_text_oof (fold-held text-only model), cos, rank
Plus S1 context from the post-corrector decisions (accepted count, best accepted prob, candidate counts) and target context
(owned by any S1 -> never added). A LightGBM add-model is trained on the other two folds' extra pairs, its threshold chosen
on those folds by 2-fold cross-fit of the realized macro gain; adds only, one S1 per target. Exact macro F0.5 on truth_mf
S1 of the held fold. Fold4 CLOSED.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]; SH = Path("/Users/earther/Desktop/aml-shared"); R = ROOT / "outputs/experiments"
PS = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.9, reg_lambda=5.0, verbose=-1, n_jobs=6, random_state=0, deterministic=True, force_col_wise=True)
FEATS = ["b_sparse", "b_prun", "b_1150", "score", "base", "srank", "ce", "cos", "drank", "q_nacc", "q_maxacc", "q_ncand", "t_ncomp"]


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def build(h, truth):
    base = pl.read_parquet(SH / f"candidates_mfu_f{h}.parquet").select("q", "t", "prob", "decision")
    have = base.select("q", "t")
    st = pl.read_parquet(R / "CL-042/multifold-stack-v2q-ce1.parquet").filter((pl.col("fold") == h) & (pl.col("base") < 0.02))
    top = pl.read_parquet(R / f"CL-003/s3/results/top12-fold{h}.parquet").rename({"source1_entity_id": "q", "target_id": "t"}).select("q", "t", "rank")
    sp = st.join(top, on=["q", "t"], how="left").select("q", "t", pl.col("prob").alias("score"), "base", pl.col("rank").cast(pl.Float64).alias("srank")).with_columns(
        pl.lit(1).alias("b_sparse"), pl.lit(0).alias("b_prun"), pl.lit(0).alias("b_1150"), pl.lit(None, pl.Float64).alias("ce"), pl.lit(None, pl.Float64).alias("cos"), pl.lit(None, pl.Float64).alias("drank"))
    fd = pl.read_parquet(R / f"CL-050/f{h}-dense.parquet").filter(pl.col("p_text_oof") < 0.05)
    fd = fd.join(pl.read_parquet(R / f"CL-050/ce2-f{h}-dense.parquet").select("q", "t", pl.col("logit").alias("ce")), on=["q", "t"], how="left")
    dp = fd.select("q", "t", pl.col("p_text_oof").alias("score"), pl.lit(None, pl.Float64).alias("base"), pl.lit(None, pl.Float64).alias("srank"),
                   pl.lit(0).alias("b_sparse"), pl.lit(1).alias("b_prun"), pl.lit(0).alias("b_1150"), "ce", "cos", pl.col("rank").cast(pl.Float64).alias("drank"))
    d5 = pl.read_parquet(R / f"CL-062/f{h}-dense-r11-50.oof.parquet")
    d5 = d5.select("q", "t", pl.col("p_text_oof").alias("score"), pl.lit(None, pl.Float64).alias("base"), pl.lit(None, pl.Float64).alias("srank"),
                   pl.lit(0).alias("b_sparse"), pl.lit(0).alias("b_prun"), pl.lit(1).alias("b_1150"), pl.lit(None, pl.Float64).alias("ce"), "cos", pl.col("rank").cast(pl.Float64).alias("drank"))
    x = pl.concat([sp, dp, d5], how="vertical_relaxed").unique(["q", "t"], keep="first").join(have, on=["q", "t"], how="anti")
    acc = base.filter(pl.col("decision") == 1)
    qctx = base.group_by("q").agg(pl.len().alias("q_ncand"), (pl.col("decision") == 1).sum().alias("q_nacc"),
                                  pl.when(pl.col("decision") == 1).then(pl.col("prob")).otherwise(None).max().alias("q_maxacc"))
    owned = set(acc["t"].to_list())
    x = x.join(qctx, on="q", how="left").with_columns(pl.col("q_ncand").fill_null(0), pl.col("q_nacc").fill_null(0))
    x = x.join(x.group_by("t").len().rename({"len": "t_ncomp"}), on="t", how="left")
    x = x.filter(~pl.col("t").is_in(list(owned)))
    x = x.with_columns(pl.Series("y", [int(t in truth.get(q, ())) for q, t in x.select("q", "t").iter_rows()]))
    pred = defaultdict(set)
    for q, t in acc.select("q", "t").iter_rows(): pred[q].add(t)
    return x, pred


def realize(x, p, thr, pred0, qs, truth):
    add = {}
    for i in np.argsort(-p, kind="stable"):
        if p[i] < thr: break
        q, t = x["q"][int(i)], x["t"][int(i)]
        if t not in add: add[t] = q
    pred = {q: set(v) for q, v in pred0.items()}
    for t, q in add.items(): pred.setdefault(q, set()).add(t)
    return float(np.mean([f05(pred.get(q, set()), truth[q]) for q in qs])), len(add), sum(int(t in truth.get(q, ())) for t, q in add.items())


def main():
    tr_ = pl.read_parquet(SH / "truth_mf.parquet"); truth = {q: set(s.split(",")) - {""} for q, s in tr_.select("q", "targets").iter_rows()}
    qfold = dict(tr_.select("q", "fold").iter_rows())
    data = {h: build(h, truth) for h in (1, 2, 3)}
    for h in (1, 2, 3):
        x = data[h][0]; print(json.dumps({"fold": h, "extra_pairs": x.height, "positives": int(x["y"].sum()),
                                          "by_bucket": {b: [int(x.filter(pl.col(b) == 1).height), int(x.filter(pl.col(b) == 1)["y"].sum())] for b in ("b_sparse", "b_prun", "b_1150")}}), flush=True)
    grid = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]; rows = []; per_all_b, per_all_n = {}, {}
    for h in (1, 2, 3):
        others = [k for k in (1, 2, 3) if k != h]
        tr = pl.concat([data[k][0] for k in others]); m = lgb.LGBMClassifier(**PS).fit(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy())
        score = {g: 0.0 for g in grid}
        for k in others:  # threshold: model trained on the remaining fold, realized gain on fold k
            kk = [z for z in others if z != k][0]; mk = lgb.LGBMClassifier(**PS).fit(data[kk][0].select(FEATS).to_numpy().astype(np.float32), data[kk][0]["y"].to_numpy())
            xk = data[k][0]; pk = mk.predict_proba(xk.select(FEATS).to_numpy().astype(np.float32))[:, 1]; qk = [q for q in truth if qfold[q] == k]
            for g in grid: score[g] += realize(xk, pk, g, data[k][1], qk, truth)[0]
        thr = max(score, key=score.get)
        x, pred0 = data[h]; ph = m.predict_proba(x.select(FEATS).to_numpy().astype(np.float32))[:, 1]; qh = [q for q in truth if qfold[q] == h]
        b = float(np.mean([f05(pred0.get(q, set()), truth[q]) for q in qh])); n, nadd, ntp = realize(x, ph, thr, pred0, qh, truth)
        rows.append({"fold": h, "thr": thr, "base": round(b, 6), "new": round(n, 6), "delta": round(n - b, 6), "adds": nadd, "adds_true": ntp, "precision": round(ntp / max(nadd, 1), 4)})
        print(json.dumps(rows[-1]), flush=True)
    d = float(np.mean([r["delta"] for r in rows])); print(json.dumps({"pooled_delta_approx": round(d, 6)}))
    out = R / "CL-062"; pl.DataFrame(rows).write_csv(out / "rx_rescue.csv")


if __name__ == "__main__":
    main()
