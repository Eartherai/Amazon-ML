"""CL-034b: decide dense-rescue pairs with the expensive matcher (e5-base CE 4EP logit + text-only prob + dense cos/rank)
instead of the text-only model alone. Fold-3 India/US; the dense stacker is cross-fit over two S1 halves; sparse decisions
come from the CL-027 fold-3 stack dump (per-half thresholds). Candidate set: sparse band p>=0.02 (top-12) plus dense
top-10 pruned at p_text >= PRUNE; rescue decisions use the dense stacker at a threshold chosen on the other half. Fold4 CLOSED.
"""
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
PS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1,
          verbose=-1, n_jobs=8, random_state=0, deterministic=True)


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def main():
    st = pl.read_parquet(ROOT / "outputs/experiments/CL-027/fold3-stack.parquet")
    dn = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored.parquet") for c in ("India", "US")]).select("q", "t", "cos", "rank", "p_text")
    ce = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-034/cedense-f3-{c}.parquet") for c in ("India", "US")]).rename({"logit": "ce"})
    dn = dn.join(ce, on=["q", "t"], how="left")
    qs = sorted(set(st["q"].to_list()) & set(dn["q"].to_list()))
    st = st.filter(pl.col("q").is_in(qs) & (pl.col("base") >= 0.02)); dn = dn.filter(pl.col("q").is_in(qs))
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(qs))
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    tc = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())
    best = {}
    for q, t, p, th in st.select("q", "t", "prob", "thr").iter_rows():
        if p >= th and (t not in best or p > best[t][1]): best[t] = (q, p)
    base_pred = defaultdict(set)
    for t, (q, _) in best.items(): base_pred[q].add(t)
    claimed = set(best)
    y = np.array([int(t in truth[q]) for q, t in dn.select("q", "t").iter_rows()])
    Fd = dn.select(pl.col("p_text").log(), "ce", "cos", "rank").to_numpy().astype(np.float32)
    half = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in dn["q"].to_list()])
    prob = np.zeros(len(dn))
    for h in (0, 1):
        prob[half == h] = lgb.LGBMClassifier(**PS).fit(Fd[half != h], y[half != h]).predict_proba(Fd[half == h])[:, 1]
    dq, dt, pt = dn["q"].to_list(), dn["t"].to_list(), dn["p_text"].to_numpy()

    def final(score, th, prune):
        add = {}
        for i in np.argsort(-score, kind="stable"):
            if score[i] < th: break
            if pt[i] < prune: continue
            if dt[i] not in claimed and dt[i] not in add: add[dt[i]] = dq[i]
        pred = {q: set(v) for q, v in base_pred.items()}
        for t, q in add.items(): pred.setdefault(q, set()).add(t)
        per = {q: f05(pred.get(q, set()), truth[q]) for q in qs}
        return float(np.mean(list(per.values()))), float(np.mean([v for q, v in per.items() if tc[q] == "India"])), float(np.mean([v for q, v in per.items() if tc[q] == "US"])), len(add)

    rows = []
    for prune in (0.0, 0.05, 0.2):
        n_c = st.height + int((pt >= prune).sum())
        m = final(pt, 0.8, prune); rows.append({"rule": "text-only p>=0.8", "prune": prune, "macro": m[0], "India": m[1], "US": m[2], "added": m[3], "cand_per_s1": n_c / len(qs)})
        for th in (0.3, 0.4, 0.5, 0.6, 0.7):
            m = final(prob, th, prune); rows.append({"rule": f"CE dense stacker >= {th}", "prune": prune, "macro": m[0], "India": m[1], "US": m[2], "added": m[3], "cand_per_s1": n_c / len(qs)})
    for r in rows: print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    pl.DataFrame(rows).write_csv(ROOT / "outputs/experiments/CL-034/dense_ce_rescue.csv")


if __name__ == "__main__":
    main()
