"""CL-027: exact fold-3 macro F0.5 of the stage-2+CE stack + dense rescue when the final matcher only sees a
restricted candidate set. Inputs: CL-027/fold3-stack.parquet (cross-fit stack prob + per-half OOF threshold),
CL-021 fold-3 dense top-10 new pairs scored by the text-only model. Final decisions are restricted to candidates,
then max-probability ownership; dense additions (p_text >= DENSE_THR, unclaimed, ownership among additions).
Caveat: stack features were computed on the full top-12 (stage-2 sibling context uses first-stage scores only). Fold4 CLOSED.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(__file__).resolve().parents[2]
st = pl.read_parquet(ROOT / "outputs/experiments/CL-027/fold3-stack.parquet")
dn = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored.parquet") for c in ("India", "US")]).select("q", "t", "rank", "p_text")
qs = sorted(set(st["q"].to_list()) & set(dn["q"].to_list()))
st = st.filter(pl.col("q").is_in(qs)); dn = dn.filter(pl.col("q").is_in(qs))
gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).filter(pl.col("source1_entity_id").is_in(qs))
truth = {q: set(r.split(",")) if r else set() for q, r in gt.iter_rows()}
tc = dict(pl.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select("entity_id", "country").iter_rows())


def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)


def run(name, base_thr, max_rank, dense_filter, dense_thr=0.8):
    s = st.filter((pl.col("base") >= base_thr) & (pl.col("rank") <= max_rank))
    d = dn.filter(dense_filter)
    n_cand = s.height + d.height
    best = {}
    for q, t, p, th in s.select("q", "t", "prob", "thr").iter_rows():
        if p >= th and (t not in best or p > best[t][1]): best[t] = (q, p)
    pred = defaultdict(set)
    for t, (q, _) in best.items(): pred[q].add(t)
    claimed = set(best); add = {}
    for q, t, p in d.filter(pl.col("p_text") >= dense_thr).sort(["p_text", "q", "t"], descending=[True, False, False]).select("q", "t", "p_text").iter_rows():
        if t not in claimed and t not in add: add[t] = q
    for t, q in add.items(): pred[q].add(t)
    per = {q: f05(pred.get(q, set()), truth[q]) for q in qs}
    m = float(np.mean(list(per.values())))
    row = {"system": name, "macro_f05": round(m, 6), "India": round(float(np.mean([v for q, v in per.items() if tc[q] == "India"])), 5),
           "US": round(float(np.mean([v for q, v in per.items() if tc[q] == "US"])), 5), "cand_per_s1": round(n_cand / len(qs), 3), "pairs": n_cand, "s1": len(qs)}
    print(json.dumps(row), flush=True); return row


rows = []
ALL = pl.lit(True)
rows.append(run("P0 sparse top-12, stack+own (no dense)", 0.0, 12, pl.lit(False)))
rows.append(run("P1 sparse top-12 + dense top-10 (current SUB005 recipe)", 0.0, 12, ALL))
for bt in (0.005, 0.01, 0.02):
    for pt in (0.02, 0.05, 0.1, 0.2):
        rows.append(run(f"band p>={bt} (top-12) + dense top-10 p_text>={pt}", bt, 12, pl.col("p_text") >= pt))
for mr in (8, 10):
    rows.append(run(f"band p>=0.01 & rank<={mr} + dense top-10 p_text>=0.05", 0.01, mr, pl.col("p_text") >= 0.05))
rows.append(run("band p>=0.01 + dense top-5 p_text>=0.05", 0.01, 12, (pl.col("rank") <= 5) & (pl.col("p_text") >= 0.05)))
pl.DataFrame(rows).write_csv(ROOT / "outputs/experiments/CL-027/restricted_macro.csv")
