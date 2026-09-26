"""CL-090 evaluation: value of the supervised hard-pair judge logit on top of the current best India/US system (fold 3).

Base decisions for fold-3 S1: post-corrector table (candidates_mfu_f3) + META-002 U2EU overrides + CE sparse below-band adds.
Corrector re-decides the judge population pairs (pairs scored by the judge) with LightGBM on [table prob, META-002 score, current
decision] (A) versus the same plus the judge logit (B). 2-fold S1 cross-fit inside fold 3 (model AND threshold for each half fit on the other half; clean inner cross-fit),
adds and removes, ownership (base owners keep unless removed). Exact per-S1 macro F0.5 over all fold-3 truth_mf S1.
Usage: judge_eval.py JUDGE_PARQUET (q, t, logit) [META_SCORES]
"""
import hashlib, sys
from collections import defaultdict
import numpy as np, polars as pl, lightgbm as lgb

SH = "/Users/earther/Desktop/aml-shared"; R = "/Users/earther/Desktop/Amazon ML Challange/outputs/experiments"
P = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, n_jobs=6, random_state=0, deterministic=True)


def f05(Pr, T):
    if not Pr and not T: return 1.0
    tp = len(Pr & T)
    if not tp: return 0.0
    a, r = tp / len(Pr), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def main():
    tr = pl.read_parquet(f"{SH}/truth_mf.parquet").filter(pl.col("fold") == 3); truth = {q: set(s.split(",")) - {""} for q, s in tr.select("q", "targets").iter_rows()}
    qs = list(truth)
    c = pl.read_parquet(f"{SH}/candidates_mfu_f3.parquet").select("q", "t", "prob", "decision", "label", "country")
    dec = {(q, t): d for q, t, d in c.select("q", "t", "decision").iter_rows()}
    for q, t, v in pl.read_parquet("/Users/earther/Desktop/aml-agent-claude-separate/results/meta2/C_U2/eu/mf_overrides/fold3.parquet").select("q", "t", "new_decision").iter_rows():
        dec[(q, t)] = int(v)
    base_pred = defaultdict(set); owner = {}
    for (q, t), d in dec.items():
        if d and t not in owner: base_pred[q].add(t); owner[t] = q
    for q, t in pl.read_parquet(f"{R}/CL-069/adds/fold3.parquet").select("q", "t").iter_rows():
        if t not in owner: base_pred[q].add(t); owner[t] = q
    b0 = float(np.mean([f05(base_pred.get(q, set()), truth[q]) for q in qs]))
    j = pl.read_parquet(sys.argv[1]).select("q", "t", pl.col("logit").alias("judge"))
    ms = sys.argv[2] if len(sys.argv) > 2 else "/Users/earther/Desktop/aml-agent-claude-separate/results/meta2/C_U2/scores_f3.parquet"
    m = pl.read_parquet(ms); mcol = [x for x in m.columns if x not in ("q", "t")][0]
    x = c.join(j, on=["q", "t"], how="inner").join(m.select("q", "t", pl.col(mcol).alias("meta")), on=["q", "t"], how="left")
    x = x.with_columns(pl.Series("cur", [int(t in base_pred.get(q, ())) for q, t in x.select("q", "t").iter_rows()]))
    x = x.filter(pl.col("q").is_in(qs))
    half = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in x["q"].to_list()])
    y = x["label"].to_numpy()
    def auc(v):
        r = v.argsort().argsort() + 1; n1 = y.sum(); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    print({"pairs": x.height, "pos": float(y.mean()), "auc_prob": round(auc(x["prob"].to_numpy()), 4), "auc_judge": round(auc(x["judge"].to_numpy()), 4),
           "auc_meta": round(auc(x["meta"].fill_null(-1).to_numpy()), 4), "base_macro": round(b0, 6)}, flush=True)
    for name, fe in (("A_no_judge", ["prob", "meta", "cur"]), ("B_with_judge", ["prob", "meta", "cur", "judge"]), ("C_judge_only", ["judge", "cur"])):
        X = x.select(fe).to_numpy().astype(np.float32); pr = np.zeros(len(y))
        for h in (0, 1):
            pr[half == h] = lgb.LGBMClassifier(**P).fit(X[half != h], y[half != h]).predict_proba(X[half == h])[:, 1]
        qq, tt = x["q"].to_list(), x["t"].to_list(); hq = {q: int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in qs}
        def realize(sel_mask, thr_of):
            pred = {q: set(v) for q, v in base_pred.items()}; own = dict(owner)
            for i in np.argsort(pr):  # removes (low scores) then adds (high), each pair with its own half's threshold
                if sel_mask[i] and pr[i] < thr_of[half[i]] and tt[i] in pred.get(qq[i], ()): pred[qq[i]].discard(tt[i]); own.pop(tt[i], None)
            for i in np.argsort(-pr):
                if sel_mask[i] and pr[i] >= thr_of[half[i]] and own.get(tt[i], qq[i]) == qq[i]: pred.setdefault(qq[i], set()).add(tt[i]); own[tt[i]] = qq[i]
            return pred
        grid = (0.4, 0.5, 0.6, 0.7, 0.8); thr_of = {}
        for h in (0, 1):  # threshold for half h chosen on the OTHER half's S1 only
            o = 1 - h; sel = half == o; qo = [q for q in qs if hq[q] == o]
            thr_of[h] = max(grid, key=lambda g: np.mean([f05(pr_.get(q, set()), truth[q]) for pr_ in [realize(sel, {0: g, 1: g})] for q in qo]))
        pred = realize(np.ones(len(pr), bool), thr_of)
        best = (thr_of, float(np.mean([f05(pred.get(q, set()), truth[q]) for q in qs])))
        print({"variant": name, "best_thr": best[0], "macro": round(best[1], 6), "delta_vs_base": round(best[1] - b0, 6)}, flush=True)


if __name__ == "__main__":
    main()
