"""CL-037: stress tests 6/7 — per-slice exact macro F0.5 of the fold-3 PRIMARY compact system
(band p>=0.02 sparse stack + ownership + CE-decided dense rescue at 0.72, prune p_text>=0.05) versus the
previous text-only dense rule. Slices are defined from the Source 1 record only. Fold4 CLOSED.
"""
import hashlib, json, re, unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
PS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=8, random_state=0, deterministic=True)


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
    st = st.filter(pl.col("q").is_in(qs) & (pl.col("base") >= 0.02)); dn = dn.filter(pl.col("q").is_in(qs) & (pl.col("p_text") >= 0.05))
    gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(qs))
    truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
    s1 = pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("").filter(pl.col("entity_id").is_in(qs))
    best = {}
    for q, t, p, th in st.select("q", "t", "prob", "thr").iter_rows():
        if p >= th and (t not in best or p > best[t][1]): best[t] = (q, p)
    base = defaultdict(set)
    for t, (q, _) in best.items(): base[q].add(t)
    claimed = set(best)
    y = np.array([int(t in truth[q]) for q, t in dn.select("q", "t").iter_rows()])
    Fd = dn.select(pl.col("p_text").log(), "ce", "cos", "rank").to_numpy().astype(np.float32)
    half = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 for q in dn["q"].to_list()])
    prob = np.zeros(len(dn))
    for h in (0, 1): prob[half == h] = lgb.LGBMClassifier(**PS).fit(Fd[half != h], y[half != h]).predict_proba(Fd[half == h])[:, 1]
    dq, dt, pt = dn["q"].to_list(), dn["t"].to_list(), dn["p_text"].to_numpy()

    def preds(score, th):
        add = {}
        for i in np.argsort(-score, kind="stable"):
            if score[i] < th: break
            if dt[i] not in claimed and dt[i] not in add: add[dt[i]] = dq[i]
        p = {q: set(v) for q, v in base.items()}
        for t, q in add.items(): p.setdefault(q, set()).add(t)
        return {q: f05(p.get(q, set()), truth[q]) for q in qs}

    new, old = preds(prob, 0.72), preds(pt, 0.8)
    info = {}
    for q, n, a, c in s1.select("entity_id", "business_name", "business_address", "country").iter_rows():
        s = n + " " + a
        info[q] = {"country": c, "non_ascii": any(ord(ch) > 127 for ch in s), "non_latin": any(ord(ch) > 0x24F and unicodedata.category(ch).startswith("L") for ch in s),
                   "missing_addr": not a.strip(), "short_addr": 0 < len(a.strip()) < 15, "short_name": len(n.strip()) <= 6, "has_digits_name": bool(re.search(r"\d", n)),
                   "no_digits_addr": not re.search(r"\d", a), "singleton": not truth[q], "multi_5plus": len(truth[q]) >= 5}
    slices = ["country", "non_ascii", "non_latin", "missing_addr", "short_addr", "short_name", "has_digits_name", "no_digits_addr", "singleton", "multi_5plus"]
    rows = []
    for sname in slices:
        groups = defaultdict(list)
        for q in qs: groups[info[q][sname]].append(q)
        for val, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
            rows.append({"slice": sname, "value": str(val), "s1": len(members), "share": round(len(members) / len(qs), 4),
                         "primary_new": round(float(np.mean([new[q] for q in members])), 5), "text_only_dense": round(float(np.mean([old[q] for q in members])), 5)})
    df = pl.DataFrame(rows); df.write_csv(ROOT / "outputs/experiments/CL-034/slice_report.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=160): print(df)
    print(json.dumps({"overall_new": round(float(np.mean(list(new.values()))), 6), "overall_old": round(float(np.mean(list(old.values()))), 6)}))


if __name__ == "__main__":
    main()
