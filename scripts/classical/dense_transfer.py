"""CL-071: transfer behaviour of the dense-rescue decision rule (US<->India simulator of France).

Sparse decisions: the production-like FULL stack trained on the SOURCE country (stage-2 v2q on source folds 1-2, stacker
with source-only CE features on source fold-3, threshold by source S1 cross-fit) applied to the target fold-3 (as CL-034/060
round 0). Dense rescue candidates: fold-3 dense top-10 pairs (CL-021) of unclaimed targets. Rules compared, each fit on
the SOURCE only and applied to the target (and cross-fit in-domain for the source reference):
  text   : source-only text model p >= 0.8 (the simulator default)
  ce     : dense stacker [logit p_text, source-only CE logit, cos, rank] (the production France rule, CE-decided)
  noce   : dense stacker [logit p_text, cos, rank]
  blend  : 0.5 * ce + 0.5 * noce (thresholds averaged)
Dense stacker thresholds are chosen on the source by S1 half cross-fit of the full realized macro. Exact per-S1 macro F0.5
on target fold-3 S1 with full truth. Target labels only score. Fold4 CLOSED.
Usage: dense_transfer.py SRC SPARSE_CE_PARQUET DENSE_CE_PARQUET
"""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts/classical"))
from stage2_qnorm import qnorm
from transfer_stack import P, PS, lg, f05, decide, ce_ctx

DS = dict(PS, n_estimators=300, num_leaves=15)


def main():
    src, ce_file, dce_file = sys.argv[1], sys.argv[2], sys.argv[3]; tgt = "India" if src == "US" else "US"
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"].astype(np.float32)
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list())); lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    qa, ta = keys[:, 0], keys[:, 1]; y = np.array([lab[(q, t)] for q, t in zip(qa.tolist(), ta.tolist())], np.int8)
    fo = np.array([fold[q] for q in qa.tolist()]); co = np.array([s1c[q] for q in qa.tolist()]); X = np.hstack([X, qnorm(qa, X)])
    trm = (co == src) & np.isin(fo, [1, 2]); f3 = fo == 3; it = np.where(f3 & (co == tgt))[0]; ist = np.where(f3 & (co == src))[0]
    ce = {(q, t): v for q, t, v in pl.read_parquet(ce_file).select("q", "t", "logit").iter_rows()}
    lv = np.array([ce.get((q, t), np.nan) for q, t in zip(qa.tolist(), ta.tolist())], np.float32)
    rk, gp = np.full(len(qa), np.nan, np.float32), np.full(len(qa), np.nan, np.float32)
    for idx in (it, ist):
        r_, g_ = ce_ctx(qa[idx].tolist(), lv[idx]); rk[idx], gp[idx] = r_, g_
    # ---- sparse FULL stack (source-trained), source OOF by S1 halves, target prediction
    m2 = lgb.LGBMClassifier(**P).fit(X[trm], y[trm]); p2 = np.zeros(len(qa)); idx = np.concatenate([it, ist]); p2[idx] = m2.predict_proba(X[idx])[:, 1]
    Fm = np.column_stack([lg(p2), X[:, 0], X[:, 2], X[:, 3], X[:, 4], lv, rk, gp]).astype(np.float32)
    half_s = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in qa[ist].tolist()]); oof = np.zeros(len(ist))
    for h in (0, 1):
        oof[half_s == h] = lgb.LGBMClassifier(**PS).fit(Fm[ist][half_s != h], y[ist][half_s != h]).predict_proba(Fm[ist][half_s == h])[:, 1]
    sq = sorted(set(qa[ist].tolist())); tq = sorted(set(qa[it].tolist()))
    thr = max(np.arange(0.4, 0.95, 0.02), key=lambda th: np.mean([f05(d.get(q, set()), truth[q]) for d in [decide(qa[ist], ta[ist], oof, th)] for q in sq]))
    pt = lgb.LGBMClassifier(**PS).fit(Fm[ist], y[ist]).predict_proba(Fm[it])[:, 1]
    dec_s = decide(qa[ist], ta[ist], oof, thr); dec_t = decide(qa[it], ta[it], pt, thr)
    # ---- dense pairs with source-only text model p, CE logit (single-country CE), cos, rank
    d = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored.parquet").select("q", "t", "cos", "rank") for c in ("India", "US")])
    dX = np.vstack([np.load(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored-X.npy") for c in ("India", "US")])
    m_text = lgb.LGBMClassifier(**P).fit(X[trm][:, 9:57], y[trm]); d = d.with_columns(pl.Series("pt", m_text.predict_proba(dX)[:, 1]))
    dce = pl.read_parquet(dce_file).select("q", "t", "logit").unique(["q", "t"], keep="first")
    d = d.join(dce, on=["q", "t"], how="left").with_columns(pl.col("q").replace_strict(s1c, default="").alias("c"))
    d = d.with_columns(pl.Series("y", [int(t in truth.get(q, ())) for q, t in d.select("q", "t").iter_rows()]))
    log(event="dense", pairs=d.height, ce_null=int(d["logit"].null_count()))
    def feats(df, kind):
        cols = [lg(df["pt"].to_numpy()), df["cos"].to_numpy(), df["rank"].to_numpy().astype(np.float32)]
        if kind == "ce": cols.insert(1, df["logit"].fill_null(np.nan).to_numpy())
        return np.column_stack(cols).astype(np.float32)
    def realize(dec, df, prob, th, qs):
        claimed = {t for v in dec.values() for t in v}; qset = set(qs); add = {}
        qq, tt = df["q"].to_list(), df["t"].to_list()
        for i in np.argsort(-prob, kind="stable"):
            if prob[i] < th: break
            if qq[i] in qset and tt[i] not in claimed and tt[i] not in add: add[tt[i]] = qq[i]
        full = {q: set(v) for q, v in dec.items()}
        for t_, q_ in add.items(): full.setdefault(q_, set()).add(t_)
        return float(np.mean([f05(full.get(q, set()), truth[q]) for q in qs]))
    ds, dt = d.filter(pl.col("c") == src), d.filter(pl.col("c") == tgt)
    hs = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in ds["q"].to_list()])
    grid = np.arange(0.3, 0.96, 0.05); rows = []
    base_t = float(np.mean([f05(dec_t.get(q, set()), truth[q]) for q in tq]))
    rows.append({"rule": "none", "in_domain": round(float(np.mean([f05(dec_s.get(q, set()), truth[q]) for q in sq])), 6), "transfer": round(base_t, 6)})
    rows.append({"rule": "text>=0.8", "in_domain": round(realize(dec_s, ds, ds["pt"].to_numpy(), 0.8, sq), 6), "transfer": round(realize(dec_t, dt, dt["pt"].to_numpy(), 0.8, tq), 6)})
    probs_s, probs_t, ths = {}, {}, {}
    for kind in ("ce", "noce"):
        Fs, Ft, ys = feats(ds, kind), feats(dt, kind), ds["y"].to_numpy()
        oo = np.zeros(ds.height)
        for h in (0, 1):
            oo[hs == h] = lgb.LGBMClassifier(**DS).fit(Fs[hs != h], ys[hs != h]).predict_proba(Fs[hs == h])[:, 1]
        th = max(grid, key=lambda g: realize(dec_s, ds, oo, g, sq))
        pt_ = lgb.LGBMClassifier(**DS).fit(Fs, ys).predict_proba(Ft)[:, 1]
        probs_s[kind], probs_t[kind], ths[kind] = oo, pt_, th
        rows.append({"rule": f"{kind} stacker", "thr": round(float(th), 2), "in_domain": round(realize(dec_s, ds, oo, th, sq), 6), "transfer": round(realize(dec_t, dt, pt_, th, tq), 6)})
    for w in (0.25, 0.5, 0.75):
        th = w * ths["ce"] + (1 - w) * ths["noce"]
        ps_, pt_ = w * probs_s["ce"] + (1 - w) * probs_s["noce"], w * probs_t["ce"] + (1 - w) * probs_t["noce"]
        rows.append({"rule": f"blend w_ce={w}", "thr": round(float(th), 3), "in_domain": round(realize(dec_s, ds, ps_, th, sq), 6), "transfer": round(realize(dec_t, dt, pt_, th, tq), 6)})
    for r in rows: r.update(source=src, target=tgt); print(json.dumps(r), flush=True)
    out = ROOT / "outputs/experiments/CL-071"; out.mkdir(parents=True, exist_ok=True); pl.DataFrame(rows, strict=False).write_csv(out / f"dense_transfer_{src}.csv")


def log(**k): print(json.dumps(k), flush=True)


if __name__ == "__main__":
    main()
