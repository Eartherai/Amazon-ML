"""CL-034: unseen-domain simulation of the PRIMARY system (stage-2 + CE stack + ownership + dense rescue).

Every learned component is trained on SOURCE-country data only:
  stage-2 GBDT on source S1 of folds 1-2 (FULL = with stage-1 context, optimistic because stage-1 saw both countries;
  TEXT = pair features only), CE = source-only e5-base CE (held fold-3 logits), stacker + threshold on source fold-3
  (threshold by 2-fold S1 cross-fit inside the source), dense rescue decided by the source-only TEXT model (p >= 0.8).
Evaluated with exact macro F0.5 on target-country fold-3 S1 (full truth). Source==target rows are in-domain
references computed by cross-fitting the stacker over two S1 halves of the source fold-3. Fold4 CLOSED.
Usage: transfer_stack.py SRC CE_PARQUET
"""
import hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
P = dict(objective="binary", n_estimators=500, learning_rate=0.06, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, verbose=-1, n_jobs=8, random_state=0, deterministic=True, force_col_wise=True)
PS = dict(P, n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, colsample_bytree=0.9)
lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))


def f05(Pr, T):
    if not Pr and not T: return 1.0
    tp = len(Pr & T)
    if not tp: return 0.0
    a, r = tp / len(Pr), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def decide(qa, ta, p, th):
    best = {}
    for q, t, v in zip(qa, ta, p):
        if v >= th and (t not in best or v > best[t][1]): best[t] = (q, v)
    out = defaultdict(set)
    for t, (q, _) in best.items(): out[q].add(t)
    return out


def ce_ctx(qa, lv):
    by = defaultdict(list)
    for i, q in enumerate(qa):
        if not np.isnan(lv[i]): by[q].append(i)
    rk = np.full(len(qa), np.nan, np.float32); gp = np.full(len(qa), np.nan, np.float32)
    for q, idx in by.items():
        v = lv[idx]; o = np.argsort(-v); rk[np.array(idx)[o]] = np.arange(1, len(idx) + 1); gp[idx] = v.max() - v
    return rk, gp


def main():
    src, ce_file = sys.argv[1], sys.argv[2]
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
    trm = (co == src) & np.isin(fo, [1, 2]); f3 = fo == 3
    TEXT = list(range(9, 57))
    m_full = lgb.LGBMClassifier(**P).fit(X[trm], y[trm]); m_text = lgb.LGBMClassifier(**P).fit(X[trm][:, TEXT], y[trm])
    q3, t3, X3, y3, c3 = qa[f3], ta[f3], X[f3], y[f3], co[f3]
    p_full, p_text = m_full.predict_proba(X3)[:, 1], m_text.predict_proba(X3[:, TEXT])[:, 1]
    ce = {(q, t): v for q, t, v in pl.read_parquet(ce_file).select("q", "t", "logit").iter_rows()}
    lv = np.array([ce.get((q, t), np.nan) for q, t in zip(q3.tolist(), t3.tolist())], np.float32); rk, gp = ce_ctx(q3.tolist(), lv)
    if os.environ.get("CE_REL"):  # calibration-invariant CE features: within-S1 z-score replaces the raw logit
        zz = pl.DataFrame({"q": q3, "v": lv}).with_columns(pl.col("v").fill_nan(None)).select(
            ((pl.col("v") - pl.col("v").mean().over("q")) / (pl.col("v").std().over("q").fill_null(1.0) + 1.0))).to_series().fill_null(np.nan).to_numpy().astype(np.float32)
        lv = zz
    FS = {"FULL": np.column_stack([lg(p_full), X3[:, 0], X3[:, 2], X3[:, 3], X3[:, 4], lv, rk, gp]),
          "TEXT": np.column_stack([lg(p_text), lv, rk, gp]), "TEXT_NOCE": np.column_stack([lg(p_text)]), "FULL_NOCE": np.column_stack([lg(p_full), X3[:, 0], X3[:, 2], X3[:, 3], X3[:, 4]])}
    if len(sys.argv) > 3:  # extra pair scorer (e.g. zero-shot reranker) appended as logit + within-S1 rank + gap
        ex = {(q, t): v for q, t, v in pl.read_parquet(sys.argv[3]).select("q", "t", "logit").iter_rows()}
        lx = np.array([ex.get((q, t), np.nan) for q, t in zip(q3.tolist(), t3.tolist())], np.float32); rx, gx = ce_ctx(q3.tolist(), lx)
        FS = {"FULL+X": np.column_stack([FS["FULL"], lx, rx, gx]), "TEXT+X": np.column_stack([FS["TEXT"], lx, rx, gx]), "TEXT_NOCE+X": np.column_stack([FS["TEXT_NOCE"], lx, rx, gx])}
    # dense rescue pairs scored by the source-only TEXT model
    dparts = []
    for c in ("India", "US"):
        d = pl.read_parquet(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored.parquet").select("q", "t")
        Xd = np.load(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored-X.npy")
        dparts.append(d.with_columns(pl.Series("p", m_text.predict_proba(Xd)[:, 1])))
    dn = pl.concat(dparts); dq, dt, dp = dn["q"].to_list(), dn["t"].to_list(), dn["p"].to_numpy()
    half = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in q3.tolist()])
    grid = np.arange(0.3, 0.96, 0.02); rows = []

    def evaluate(pred_idx, prob, th, qs):
        d = decide(q3[pred_idx], t3[pred_idx], prob, th); claimed = {t for v in d.values() for t in v}
        qset = set(qs); add = {}
        for q, t, p in sorted(zip(dq, dt, dp), key=lambda r: -r[2]):
            if p < 0.8: break
            if q in qset and t not in claimed and t not in add: add[t] = q
        dd = {q: set(v) for q, v in d.items()}
        for t, q in add.items(): dd.setdefault(q, set()).add(t)
        return float(np.mean([f05(dd.get(q, set()), truth[q]) for q in qs])), float(np.mean([f05(d.get(q, set()), truth[q]) for q in qs]))

    for name, Fm in FS.items():
        s = c3 == src
        # source cross-fit (in-domain reference + threshold)
        oof = np.zeros(len(q3))
        for h in (0, 1):
            tr, te = s & (half != h), s & (half == h)
            oof[te] = lgb.LGBMClassifier(**PS).fit(Fm[tr], y3[tr]).predict_proba(Fm[te])[:, 1]
        sq = sorted(set(q3[s].tolist())); idx_s = np.where(s)[0]
        def src_macro(th):
            dd = decide(q3[idx_s], t3[idx_s], oof[idx_s], th); return float(np.mean([f05(dd.get(q, set()), truth[q]) for q in sq]))
        thr = max(grid, key=src_macro)
        ind, ind_nod = evaluate(idx_s, oof[idx_s], thr, sq)
        final = lgb.LGBMClassifier(**PS).fit(Fm[s], y3[s])
        for tgt in sorted(set(c3.tolist()) - {src}):
            it = np.where(c3 == tgt)[0]; tq = sorted(set(q3[it].tolist()))
            pt = final.predict_proba(Fm[it])[:, 1]; tr_m, tr_nod = evaluate(it, pt, thr, tq)
            if os.environ.get("DUMP_SYSTEM") == name:  # target predictions for pseudo-labelling (no target labels used)
                pl.DataFrame({"q": q3[it], "t": t3[it], "prob": pt, "thr": np.full(len(it), thr)}).write_parquet(
                    ROOT / f"outputs/experiments/CL-034/pred_{name}_{src}_to_{tgt}.parquet")
            row = {"system": name, "source": src, "ce": Path(ce_file).stem, "thr": round(float(thr), 2), "in_domain_" + src: round(ind, 6), "in_domain_no_dense": round(ind_nod, 6),
                   "transfer_" + tgt: round(tr_m, 6), "transfer_no_dense": round(tr_nod, 6)}
            rows.append(row); print(json.dumps(row), flush=True)
    out = ROOT / "outputs/experiments/CL-034"; out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out / f"transfer_stack_{src}_{Path(ce_file).stem}{'_cerel' if os.environ.get('CE_REL') else ''}{'_' + Path(sys.argv[3]).stem if len(sys.argv) > 3 else ''}.csv")


if __name__ == "__main__":
    main()
