"""CL-069 final: CE-scored sparse below-band rescue, test-consistent features, MF re-check, frozen all-fold model, test adds.

Features (identical definitions on folds and test): ce (e5-base CE logit; fold-held on folds, fold-3-held model on test),
score (stack probability of the pair; CE features absent in both), base (stage-1 score), srank (rank by stage-1 base within
the S1's top-12), q_nacc / q_maxacc / q_ncand (S1 context from the post-corrector decisions), t_ncomp (number of sparse
below-band pairs listing the target). Pairs whose target is already accepted are excluded; adds only, highest model score
first, one S1 per target. MF protocol for the check; frozen model on all three folds with threshold THR for test.
Usage: splow_final.py mf | splow_final.py test  (test needs outputs/experiments/CL-069/test/ce/*.parquet). Fold4 CLOSED.
"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]; SH = Path("/Users/earther/Desktop/aml-shared"); R = ROOT / "outputs/experiments"
sys.path.insert(0, str(ROOT / "scripts/classical"))
from rx_rescue import PS, f05, realize
FE = ["ce", "score", "base", "srank", "q_nacc", "q_maxacc", "q_ncand", "t_ncomp"]
THR = 0.8


def assemble(pairs, table, owned_extra=None):
    """pairs: q, t, ce, score, base, srank. table: q, t, prob, decision (current decisions). Returns feature frame."""
    acc = table.filter(pl.col("decision") == 1)
    qctx = table.group_by("q").agg(pl.len().alias("q_ncand"), (pl.col("decision") == 1).sum().alias("q_nacc"),
                                   pl.when(pl.col("decision") == 1).then(pl.col("prob")).otherwise(None).max().alias("q_maxacc"))
    owned = set(acc["t"].to_list()) | (owned_extra or set())
    x = pairs.join(table.select("q", "t"), on=["q", "t"], how="anti").filter(~pl.col("t").is_in(list(owned)))
    x = x.join(qctx, on="q", how="left").with_columns(pl.col("q_ncand").fill_null(0), pl.col("q_nacc").fill_null(0))
    x = x.join(x.group_by("t").len().rename({"len": "t_ncomp"}), on="t", how="left")
    pred = defaultdict(set)
    for q, t in acc.select("q", "t").iter_rows(): pred[q].add(t)
    return x, pred


def fold_data(h, truth):
    table = pl.read_parquet(SH / f"candidates_mfu_f{h}.parquet").select("q", "t", "prob", "decision")
    st = pl.read_parquet(R / "CL-042/multifold-stack-v2q-ce1.parquet").filter((pl.col("fold") == h) & (pl.col("base") < 0.02))
    top = pl.read_parquet(R / f"CL-003/s3/results/top12-fold{h}.parquet").rename({"source1_entity_id": "q", "target_id": "t"}).select("q", "t", "rank")
    ce = pl.read_parquet(R / f"CL-069/ce-f{h}.parquet").select("q", "t", pl.col("logit").alias("ce"))
    pairs = st.join(top, on=["q", "t"], how="left").join(ce, on=["q", "t"], how="inner").select(
        "q", "t", "ce", pl.col("prob").alias("score"), "base", pl.col("rank").cast(pl.Float64).alias("srank"))
    x, pred = assemble(pairs, table)
    return x.with_columns(pl.Series("y", [int(t in truth.get(q, ())) for q, t in x.select("q", "t").iter_rows()])), pred


def M(df): return df.select(FE).to_numpy().astype(np.float32)


def main():
    mode = sys.argv[1]
    tr_ = pl.read_parquet(SH / "truth_mf.parquet"); truth = {q: set(s.split(",")) - {""} for q, s in tr_.select("q", "targets").iter_rows()}
    qfold = dict(tr_.select("q", "fold").iter_rows()); data = {h: fold_data(h, truth) for h in (1, 2, 3)}
    if mode == "mf":
        rows = []
        for h in (1, 2, 3):
            tr = pl.concat([data[k][0] for k in (1, 2, 3) if k != h]); m = lgb.LGBMClassifier(**PS).fit(M(tr), tr["y"].to_numpy())
            x, pred0 = data[h]; ph = m.predict_proba(M(x))[:, 1]; qh = [q for q in truth if qfold[q] == h]
            b = float(np.mean([f05(pred0.get(q, set()), truth[q]) for q in qh]))
            for thr in (0.7, 0.8, 0.9):
                n, nadd, ntp = realize(x, ph, thr, pred0, qh, truth)
                rows.append({"fold": h, "thr": thr, "delta": round(n - b, 6), "adds": nadd, "true": ntp, "prec": round(ntp / max(nadd, 1), 4), "adds_per_100k": round(1e5 * nadd / len(qh), 1), "n": len(qh)})
                print(json.dumps(rows[-1]), flush=True)
        for thr in (0.7, 0.8, 0.9):
            rr = [r for r in rows if r["thr"] == thr]; print(json.dumps({"thr": thr, "pooled": round(sum(r["delta"] * r["n"] for r in rr) / sum(r["n"] for r in rr), 6)}))
        return
    # ---- test
    allx = pl.concat([data[h][0] for h in (1, 2, 3)]); m = lgb.LGBMClassifier(**PS).fit(M(allx), allx["y"].to_numpy())
    m.booster_.save_model(str(R / "CL-069/splow_final_model.txt"))
    tp = pl.read_parquet(R / "CL-069/test/test-sparselow-indus.parquet")
    ce = pl.concat([pl.read_parquet(f).select("q", "t", pl.col("logit").alias("ce")) for f in sorted(glob.glob(str(R / "CL-069/test/ce/*.parquet")))]).unique(["q", "t"], keep="first")
    bag = pl.read_parquet(R / "CL-003/test_probs/CL-005-top12v2q-bag.parquet").select("q", "t", "base").unique(["q", "t"], keep="first")
    rk = bag.with_columns(pl.col("base").rank("ordinal", descending=True).over("q").cast(pl.Float64).alias("srank")).select("q", "t", "srank")
    stk = pl.read_parquet(ROOT / "outputs/experiments/CL-014/test-stack-CL-044-v2qbag-stack.parquet").select("q", "t", pl.col("p").alias("score")).unique(["q", "t"], keep="first")
    pairs = tp.join(ce, on=["q", "t"], how="inner").join(stk, on=["q", "t"], how="left").join(bag, on=["q", "t"], how="left").join(rk, on=["q", "t"], how="left")
    print(json.dumps({"test_pairs": tp.height, "with_ce": pairs.height, "score_null": int(pairs["score"].null_count()), "base_null": int(pairs["base"].null_count())}), flush=True)
    # current decisions: post-corrector table (same state as candidates_mfu for the folds) for context; ownership from the final SUB017 decisions
    table = pl.read_parquet(R / "CL-066/test_cands_frblend050_postuniv.parquet").select("q", "t", "prob", "decision")
    final = pl.read_csv(ROOT / "outputs/submissions/CL-070-v9-frblend050-univ-recall/matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
    owned_final = set(final.filter(pl.col("matched_entity_ids") != "").select(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")["matched_entity_ids"].to_list())
    x, _ = assemble(pairs, table, owned_final)
    p = m.predict_proba(M(x))[:, 1]; add = {}
    for i in np.argsort(-p, kind="stable"):
        if p[i] < THR: break
        q, t = x["q"][int(i)], x["t"][int(i)]
        if t not in add: add[t] = q
    out = pl.DataFrame({"q": list(add.values()), "t": list(add.keys()), "new_decision": [1] * len(add)}, schema={"q": pl.String, "t": pl.String, "new_decision": pl.Int64})
    out.write_parquet(R / "CL-069/test/splow_test_adds.parquet")
    nq = 1473092
    print(json.dumps({"test_adds": out.height, "adds_per_100k_s1": round(1e5 * out.height / nq, 1), "ce_mean_test": round(float(x["ce"].mean()), 4), "ce_mean_folds": round(float(allx["ce"].mean()), 4),
                      "p_ge_thr_frac_test": round(float((p >= THR).mean()), 6), "score_mean_test": round(float(x["score"].mean()), 5), "score_mean_folds": round(float(allx["score"].mean()), 5)}))


if __name__ == "__main__":
    main()
