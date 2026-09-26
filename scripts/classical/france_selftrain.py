"""CL-064: label-free self-training of stage-2 + CE stacker on France test pairs (the CL-060 FULL round-1 procedure).

Simulator evidence (CL-060/063, US -> India, the under-linking regime France is in by the label-free link-count
diagnostic): +0.0108 transfer macro. The reverse direction (India -> US, over-linking) loses, so this is applied only
because France under-links (3.17 links per S1 vs 3.27-3.29 India/US on test; truth prior 3.46 in both train countries).

Steps (no labels of France exist; train labels only):
  1. Pseudo labels from the current France stack probabilities (CL-044 v2q bag stack): positive = p >= POS and the pair
     owns its target among France pairs; negative = p <= NEG.
  2. Stage-2 v2q (105 features) refit on train folds 1-2 + France pseudo rows (weight so pseudo mass / source rows = the
     simulator ratio 0.29); it predicts train fold-3 (stacker training rows) and every France pair.
  3. Stacker (8 features) refit on train fold-3 + France pseudo rows (mass ratio 0.58); threshold by salted 3-fold S1
     cross-fit on train fold-3 only (as assemble_final_v2).
  4. France p is logit-shifted so the new threshold maps to 0.72 (the downstream compact assembly threshold) and written
     into a copy of the stack parquet; India/US rows are untouched.
Fold4 CLOSED.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts/classical"))
from stage2_qnorm import qnorm, PARAMS as P2
from assemble_final_v2 import PARAMS as PS, logit, ce_ctx, own_decide, f05

POS, NEG = float(os.environ.get("POS", 0.9)), float(os.environ.get("NEG", 0.02))
R2, RS = float(os.environ.get("R2", 0.29)), float(os.environ.get("RS", 0.58))
OUT = ROOT / "outputs/experiments/CL-064"; OUT.mkdir(parents=True, exist_ok=True)
STACK = ROOT / "outputs/experiments/CL-014/test-stack-CL-044-v2qbag-stack.parquet"


def log(**k): print(json.dumps({"t": round(time.time() - T0, 1), **k}), flush=True)


def main():
    # ---- train matrix
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=True); keys, X = z["keys"], z["X"].astype(np.float32)
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    fold = dict(zip(top["source1_entity_id"].to_list(), top["fold"].to_list())); lab = {(q, t): l for q, t, l in top.select("source1_entity_id", "target_id", "label").iter_rows()}
    qa, ta = keys[:, 0], keys[:, 1]; y = np.array([lab[(q, t)] for q, t in zip(qa.tolist(), ta.tolist())], np.int8)
    fo = np.array([fold[q] for q in qa.tolist()]); X = np.hstack([X, qnorm(qa, X)])
    tr12, f3 = np.isin(fo, [1, 2]), fo == 3
    log(event="train", rows=len(qa), tr12=int(tr12.sum()), f3=int(f3.sum()))
    # ---- France test features, current stack p, CE logits
    fq, ft, fx = [], [], []
    for f in sorted((ROOT / "outputs/experiments/CL-005/features-v2").glob("France-*.npz")):
        zz = np.load(f); fq.append(zz["q"]); ft.append(zz["t"]); fx.append(zz["X"].astype(np.float32))
    fq, ft = np.concatenate(fq), np.concatenate(ft); FX = np.vstack(fx); FX = np.hstack([FX, qnorm(fq, FX)])
    fr = pl.DataFrame({"q": fq, "t": ft}).with_row_index("i")
    st = pl.read_parquet(STACK).unique(["q", "t"], keep="first")
    pcur = fr.join(st, on=["q", "t"], how="left").sort("i")["p"].to_numpy(); assert not np.isnan(pcur).any()
    ce_t = pl.read_parquet(ROOT / "outputs/experiments/CL-025/bundles/ce4ep_test.parquet", columns=["q", "t", "logit"]).filter(pl.col("q").is_in(np.unique(fq).tolist())).unique(["q", "t"], keep="first")
    lvt = fr.join(ce_t, on=["q", "t"], how="left").sort("i")["logit"].fill_null(np.nan).to_numpy().astype(np.float32)
    FK = list(zip(fq.tolist(), ft.tolist())); rkt, gpt = ce_ctx(FK, lvt)
    log(event="france", pairs=len(fq), s1=int(len(np.unique(fq))), ce_hits=int((~np.isnan(lvt)).sum()))
    # ---- pseudo labels (ownership among France pairs by current p)
    order = np.lexsort((ft, -pcur)); seen, own = set(), np.zeros(len(fq), bool)
    for i in order:
        if ft[i] not in seen: seen.add(ft[i]); own[i] = True
    pos = np.where((pcur >= POS) & own)[0]; neg = np.where(pcur <= NEG)[0]
    pi = np.concatenate([pos, neg]); pyv = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
    w2 = R2 * tr12.sum() / len(pi); ws = RS * f3.sum() / len(pi)
    log(event="pseudo", pos=len(pos), neg=len(neg), w_stage2=round(w2, 4), w_stacker=round(ws, 4))
    # ---- stage-2 refit
    Xs = np.vstack([X[tr12], FX[pi]]); ys = np.concatenate([y[tr12], pyv]); wv = np.concatenate([np.ones(tr12.sum()), np.full(len(pi), w2)])
    m2 = lgb.LGBMClassifier(**P2).fit(Xs, ys, sample_weight=wv); del Xs
    p2f3 = m2.predict_proba(X[f3])[:, 1]; p2fr = m2.predict_proba(FX)[:, 1]
    m2.booster_.save_model(str(OUT / "stage2-v2q-f12-frpseudo.txt"))
    log(event="stage2", france_p2_mean=round(float(p2fr.mean()), 5))
    # ---- stacker (fold-3 train + France pseudo)
    K3 = list(zip(qa[f3].tolist(), ta[f3].tolist())); X3 = X[f3]; y3 = y[f3]
    ce = {(q, t): v for q, t, v in pl.read_parquet(ROOT / "outputs/experiments/CL-014/e5b4-ep3.parquet").select("q", "t", "logit").iter_rows()}
    lv = np.array([ce.get(k, np.nan) for k in K3], np.float32); rk, gp = ce_ctx(K3, lv)
    Fm = np.column_stack([logit(p2f3), X3[:, 0], X3[:, 2], X3[:, 3], X3[:, 4], lv, rk, gp]).astype(np.float32)
    Ff = np.column_stack([logit(p2fr), FX[:, 0], FX[:, 2], FX[:, 3], FX[:, 4], lvt, rkt, gpt]).astype(np.float32)
    part = np.array([int(hashlib.sha256(("a" + q).encode()).hexdigest(), 16) % 3 for q, _ in K3]); oof = np.zeros(len(K3))
    for k in range(3):
        oof[part == k] = lgb.LGBMClassifier(**PS).fit(Fm[part != k], y3[part != k]).predict_proba(Fm[part == k])[:, 1]
    qs = sorted({k[0] for k in K3})
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(qs))
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    def macro(th):
        d = own_decide(K3, oof, th); return float(np.mean([f05(d.get(q, set()), truth[q]) for q in qs]))
    grid = {round(float(th), 2): macro(th) for th in np.arange(0.5, 0.9, 0.02)}; thr = max(grid, key=grid.get)
    log(event="stacker_threshold", thr=thr, fold3_oof_macro=round(grid[thr], 6))
    Fs = np.vstack([Fm, Ff[pi]]); ysv = np.concatenate([y3, pyv]); wsv = np.concatenate([np.ones(len(y3)), np.full(len(pi), ws)])
    pf = lgb.LGBMClassifier(**PS).fit(Fs, ysv, sample_weight=wsv).predict_proba(Ff)[:, 1]
    padj = 1 / (1 + np.exp(-(logit(pf) - logit(thr) + logit(0.72))))
    # ---- diagnostics: links per S1 at 0.72 with ownership among France sparse pairs only
    def links(p):
        d = own_decide(FK, p, 0.72); return float(sum(len(v) for v in d.values()) / len(np.unique(fq)))
    log(event="france_links_sparse", before=round(links(pcur), 4), after=round(links(padj), 4), changed_decisions=int(((pcur >= 0.72) != (padj >= 0.72)).sum()))
    new = pl.DataFrame({"q": fq, "t": ft, "p_new": padj})
    out = st.join(new, on=["q", "t"], how="left").with_columns(pl.coalesce("p_new", "p").alias("p")).drop("p_new")
    assert out.height == st.height
    name = f"test-stack-CL-064-frst-{POS}-{NEG}.parquet"; out.write_parquet(ROOT / "outputs/experiments/CL-014" / name)
    pl.DataFrame({"q": fq, "t": ft, "p_old": pcur, "p_new": padj}).write_parquet(OUT / f"france_p_{POS}_{NEG}.parquet")
    log(event="done", stack=name)


if __name__ == "__main__":
    T0 = time.time(); main()
