"""CL-042: multi-fold confirmation of the primary sparse stack (stage-2 OOF + e5-base CE held-fold logits) on all
~194k S1 of folds 1-3. CE logits per fold come from CEs that never trained on that fold (owner-safe). The stacker is
cross-fit by fold (train on two folds, predict the third); its ownership threshold is chosen by an inner fold split of
the training folds. Compares stage-2 v2 vs v2q (QNORM) OOF, with and without CE. Exact macro F0.5, full truth. Fold4 CLOSED.
Usage: multifold_stack.py CE_F1 CE_F2 CE_F3
"""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
PS = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.9, verbose=-1, n_jobs=8, random_state=0, deterministic=True, force_col_wise=True)
lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def macro_of(qa, ta, p, th, qs, truth):
    best = {}
    for q, t, v in zip(qa, ta, p):
        if v >= th and (t not in best or v > best[t][1]): best[t] = (q, v)
    d = defaultdict(set)
    for t, (q, _) in best.items(): d[q].add(t)
    per = {q: f05(d.get(q, set()), truth[q]) for q in qs}
    return per


def main():
    ce_files = sys.argv[1:4]
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"].astype(np.float32)
    qa, ta = keys[:, 0], keys[:, 1]
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list()))
    lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    y = np.array([lab[(q, t)] for q, t in zip(qa.tolist(), ta.tolist())], np.int8); fo = np.array([fold[q] for q in qa.tolist()])
    s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    ce = {}
    for f in ce_files:
        ce.update({(q, t): v for q, t, v in pl.read_parquet(f).select("q", "t", "logit").iter_rows()})
    lv = np.array([ce.get((q, t), np.nan) for q, t in zip(qa.tolist(), ta.tolist())], np.float32)
    ctx = pl.DataFrame({"q": qa, "v": lv}).with_columns(pl.col("v").fill_nan(None))
    rk = ctx.select(pl.col("v").rank("ordinal", descending=True).over("q")).to_series().fill_null(np.nan).to_numpy().astype(np.float32)
    gp = ctx.select(pl.col("v").max().over("q") - pl.col("v")).to_series().fill_null(np.nan).to_numpy().astype(np.float32)
    rows = []
    for tag, oof_file in (("v2", "train-oof-top12-v2.npy"), ("v2q", "train-oof-top12-v2q.npy")):
        p2 = np.load(ROOT / "outputs/experiments/CL-003" / oof_file)
        for use_ce in (False, True):
            cols = [lg(p2), X[:, 0], X[:, 2], X[:, 3], X[:, 4]] + ([lv, rk, gp] if use_ce else [])
            Fm = np.column_stack(cols).astype(np.float32); per_all = {}
            for h in (1, 2, 3):
                tr, te = np.where(fo != h)[0], np.where(fo == h)[0]
                trf = fo[tr]; a, b = sorted(set(trf.tolist()))
                ip = np.zeros(len(tr))
                for k in (a, b):
                    m = trf != k; ip[~m] = lgb.LGBMClassifier(**PS).fit(Fm[tr[m]], y[tr[m]]).predict_proba(Fm[tr[~m]])[:, 1]
                # threshold on a fixed 30k-S1 sample of the training folds (speed); held-fold evaluation stays exact and complete
                trq_all = sorted(set(qa[tr].tolist())); samp = set(sorted(trq_all, key=lambda q: hashlib.sha256(("thr" + q).encode()).hexdigest())[:30000])
                sm = np.array([q in samp for q in qa[tr].tolist()]); trq = sorted(samp)
                thr = max(np.arange(0.5, 0.9, 0.04), key=lambda th: np.mean(list(macro_of(qa[tr][sm], ta[tr][sm], ip[sm], th, trq, truth).values())))
                pt = lgb.LGBMClassifier(**PS).fit(Fm[tr], y[tr]).predict_proba(Fm[te])[:, 1]
                teq = sorted(set(qa[te].tolist())); per = macro_of(qa[te], ta[te], pt, thr, teq, truth); per_all.update(per)
                rows.append({"stage2": tag, "ce": use_ce, "fold": h, "thr": round(float(thr), 2), "macro": round(float(np.mean(list(per.values()))), 6), "s1": len(teq)})
                print(json.dumps(rows[-1]), flush=True)
            allq = sorted(per_all)
            rows.append({"stage2": tag, "ce": use_ce, "fold": "all", "macro": round(float(np.mean([per_all[q] for q in allq])), 6), "s1": len(allq),
                         "India": round(float(np.mean([per_all[q] for q in allq if s1c[q] == "India"])), 6), "US": round(float(np.mean([per_all[q] for q in allq if s1c[q] == "US"])), 6)})
            print(json.dumps(rows[-1]), flush=True)
    out = ROOT / "outputs/experiments/CL-042"; out.mkdir(parents=True, exist_ok=True); pl.DataFrame(rows, strict=False).write_csv(out / "multifold_stack.csv")


if __name__ == "__main__":
    main()
