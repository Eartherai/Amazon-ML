"""CL-060 / CL-063b: label-free self-training of the full primary stack for an unseen country (US<->India simulator of France).

Round 0 = the CL-034 source-only system with QNORM (stage-2 on source folds 1-2, source-only CE, stacker on source fold-3,
dense rescue by the source-only text model). Each round: pseudo-label the target fold-3 pairs from the current target
predictions (positives: prob >= POS and the pair owns its target; negatives: prob <= NEG), retrain stage-2 on source
labels + target pseudo rows (weight W) and the stacker on source fold-3 + target pseudo rows, re-predict the target.
Target labels are used only to score. Transductive, as it would be on France test. Fold4 CLOSED.
PRIOR=1 adds the prior-matched target threshold (target links per S1 = source in-domain links per S1, CL-063).
Usage: selftrain_stack.py SRC CE_PARQUET [rounds]
"""
import hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts/classical"))
from stage2_qnorm import qnorm
from transfer_stack import P, PS, lg, f05, decide, ce_ctx

POS, NEG, W = float(os.environ.get("POS", 0.9)), float(os.environ.get("NEG", 0.02)), float(os.environ.get("W", 1.0))


def main():
    src, ce_file = sys.argv[1], sys.argv[2]; rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"].astype(np.float32)
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list())); lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    s1c = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    qa, ta = keys[:, 0], keys[:, 1]; y = np.array([lab[(q, t)] for q, t in zip(qa.tolist(), ta.tolist())], np.int8)
    fo = np.array([fold[q] for q in qa.tolist()]); co = np.array([s1c[q] for q in qa.tolist()])
    X = np.hstack([X, qnorm(qa, X)]); TEXT = list(range(9, X.shape[1]))
    tgt = "India" if src == "US" else "US"
    trm = (co == src) & np.isin(fo, [1, 2]); f3 = fo == 3; it = np.where(f3 & (co == tgt))[0]; ist = np.where(f3 & (co == src))[0]
    if os.environ.get("INDOMAIN"):  # control: target = held S1 half of the source country's fold-3 (no domain shift)
        hh = np.array([int(hashlib.sha256(("ind" + q).encode()).hexdigest(), 16) % 2 for q in qa.tolist()])
        it = np.where(f3 & (co == src) & (hh == 1))[0]; ist = np.where(f3 & (co == src) & (hh == 0))[0]
    ce = {(q, t): v for q, t, v in pl.read_parquet(ce_file).select("q", "t", "logit").iter_rows()}
    lv = np.array([ce.get((q, t), np.nan) for q, t in zip(qa.tolist(), ta.tolist())], np.float32)
    rk, gp = np.full(len(qa), np.nan, np.float32), np.full(len(qa), np.nan, np.float32)
    for idx in (it, ist):
        r_, g_ = ce_ctx(qa[idx].tolist(), lv[idx]); rk[idx], gp[idx] = r_, g_
    tq = sorted(set(qa[it].tolist()))
    # dense rescue pairs (plain 48-feature text model; retrained with pseudo-labels too)
    dd = []
    for c in ("India", "US"):
        d = pl.read_parquet(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored.parquet").select("q", "t")
        dd.append((d, np.load(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored-X.npy")))
    dq = np.concatenate([d["q"].to_numpy() for d, _ in dd]); dt = np.concatenate([d["t"].to_numpy() for d, _ in dd]); dX = np.vstack([x for _, x in dd])

    def run(system, pseudo=None):
        cols = TEXT if system == "TEXT" else list(range(X.shape[1]))
        Xs, ys, ws = X[trm][:, cols], y[trm], np.ones(trm.sum())
        if pseudo is not None:
            pi, pyv = pseudo; Xs = np.vstack([Xs, X[it[pi]][:, cols]]); ys = np.concatenate([ys, pyv]); ws = np.concatenate([ws, np.full(len(pi), W)])
        m2 = lgb.LGBMClassifier(**P).fit(Xs, ys, sample_weight=ws)
        p2 = np.zeros(len(qa)); idx = np.concatenate([it, ist]); p2[idx] = m2.predict_proba(X[idx][:, cols])[:, 1]
        base = [X[:, 0], X[:, 2], X[:, 3], X[:, 4]] if system == "FULL" else []
        Fm = np.column_stack([lg(p2)] + base + [lv, rk, gp]).astype(np.float32)
        Fs, ysrc, wsrc = Fm[ist], y[ist], np.ones(len(ist))
        if pseudo is not None:
            pi, pyv = pseudo; Fs = np.vstack([Fs, Fm[it[pi]]]); ysrc = np.concatenate([ysrc, pyv]); wsrc = np.concatenate([wsrc, np.full(len(pi), W)])
        # threshold from a 2-fold S1 cross-fit on the source fold-3 only
        half = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in qa[ist].tolist()]); oof = np.zeros(len(ist))
        for h in (0, 1):
            oof[half == h] = lgb.LGBMClassifier(**PS).fit(Fm[ist][half != h], y[ist][half != h]).predict_proba(Fm[ist][half == h])[:, 1]
        sq = sorted(set(qa[ist].tolist()))
        thr = max(np.arange(0.4, 0.95, 0.02), key=lambda th: np.mean([f05(dd_.get(q, set()), truth[q]) for dd_ in [decide(qa[ist], ta[ist], oof, th)] for q in sq]))
        st = lgb.LGBMClassifier(**PS).fit(Fs, ysrc, sample_weight=wsrc); pt = st.predict_proba(Fm[it])[:, 1]
        # dense rescue: plain text model retrained with the same pseudo rows (48 raw pair features)
        Xd_s, yd_s, wd_s = X[trm][:, 9:57], y[trm], np.ones(trm.sum())
        if pseudo is not None:
            pi, pyv = pseudo; Xd_s = np.vstack([Xd_s, X[it[pi]][:, 9:57]]); yd_s = np.concatenate([yd_s, pyv]); wd_s = np.concatenate([wd_s, np.full(len(pi), W)])
        mdn = lgb.LGBMClassifier(**P).fit(Xd_s, yd_s, sample_weight=wd_s); dp = mdn.predict_proba(dX)[:, 1]
        def realize(idx, prob, th, qs):
            dec = decide(qa[idx], ta[idx], prob, th); claimed = {t for v in dec.values() for t in v}; qset = set(qs); add = {}
            for i in np.argsort(-dp):
                if dp[i] < 0.8: break
                if dq[i] in qset and dt[i] not in claimed and dt[i] not in add: add[dt[i]] = dq[i]
            full = {q: set(v) for q, v in dec.items()}
            for t_, q_ in add.items(): full.setdefault(q_, set()).add(t_)
            return float(np.mean([f05(full.get(q, set()), truth[q]) for q in qs])), float(np.mean([len(full.get(q, ())) for q in qs]))
        m_fixed, links_fixed = realize(it, pt, thr, tq)
        extra = {"links": round(links_fixed, 4)}
        if os.environ.get("PRIOR"):
            src_links = realize(ist, oof, thr, sq)[1]
            curve = {round(float(th), 3): realize(it, pt, th, tq) for th in np.arange(0.3, 0.97, 0.02)}
            thp = min(curve, key=lambda th: abs(curve[th][1] - src_links))
            extra.update({"src_links": round(src_links, 4), "prior_thr": thp, "prior_macro": round(curve[thp][0], 6), "prior_links": round(curve[thp][1], 4)})
        return m_fixed, pt, thr, extra

    rows = []
    for system in ("TEXT", "FULL"):
        m0, pt, thr, ex = run(system); rows.append({"system": system, "source": src, "round": 0, "transfer_macro": round(m0, 6), "thr": round(float(thr), 2), **ex}); print(json.dumps(rows[-1]), flush=True)
        for r in range(1, rounds + 1):
            owner = defaultdict(lambda: (-1, None))
            for i, (q, t) in enumerate(zip(qa[it].tolist(), ta[it].tolist())):
                if pt[i] > owner[t][0]: owner[t] = (pt[i], i)
            own = np.zeros(len(it), bool); own[[v[1] for v in owner.values() if v[1] is not None]] = True
            pos = np.where((pt >= POS) & own)[0]; neg = np.where(pt <= NEG)[0]
            pi = np.concatenate([pos, neg]); pyv = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
            yt = y[it[pi]]; prec_pos = float(yt[:len(pos)].mean()) if len(pos) else None; npv = float(1 - yt[len(pos):].mean()) if len(neg) else None
            m, pt, thr, ex = run(system, (pi, pyv))
            rows.append({"system": system, "source": src, "round": r, "transfer_macro": round(m, 6), "thr": round(float(thr), 2), "pseudo_pos": len(pos), "pos_precision": round(prec_pos, 4),
                         "pseudo_neg": len(neg), "neg_npv": round(npv, 4), **ex}); print(json.dumps(rows[-1]), flush=True)
    out = ROOT / "outputs/experiments/CL-060"; out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, strict=False).write_csv(out / f"selftrain_{src}_{POS}_{NEG}_{W}{'_prior' if os.environ.get('PRIOR') else ''}{'_indomain' if os.environ.get('INDOMAIN') else ''}.csv")


if __name__ == "__main__":
    main()
