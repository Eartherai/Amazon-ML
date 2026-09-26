"""CL-061: where is the unretrieved truth? Oracle decomposition on the multi-fold candidate tables (India/US, 3 folds).
Buckets for true pairs missing from the current candidate set: sparse top-12 below the stage-1 band, dense top-10 pruned
by text-only p < 0.05, dense ranks 11-50, and nowhere. Oracle F0.5 gain if each bucket were added. Fold4 CLOSED."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path(__file__).resolve().parents[2]; SH = Path("/Users/earther/Desktop/aml-shared")
def f05(P, T):
    if not P and not T: return 1.0
    tp = len(P & T)
    if not tp: return 0.0
    a, r = tp / len(P), tp / len(T); return 1.25 * a * r / (0.25 * a + r)
gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
truth = {q: set(r.split(",")) - {""} for q, r in gt.iter_rows()}
top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)]).rename({"source1_entity_id": "q", "target_id": "t"})
dense_all = {1: [ROOT / f"outputs/experiments/CL-050/held-fold1-{c}-dense.parquet" for c in ("India", "US")],
             2: [ROOT / f"outputs/experiments/CL-050/held-fold2-{c}-dense.parquet" for c in ("India", "US")],
             3: [ROOT / f"outputs/experiments/CL-021/held-fold3-{c}-dense.parquet" for c in ("India", "US")]}
rows = []; pooled = defaultdict(list)
for h in (1, 2, 3):
    c = pl.read_parquet(SH / f"candidates_mf_f{h}.parquet").select("q", "t")
    qs = sorted(set(pl.read_parquet(SH / "truth_mf.parquet").filter(pl.col("fold") == h)["q"].to_list()))  # evaluation population
    cand = defaultdict(set)
    for q, t in c.iter_rows(): cand[q].add(t)
    sp_low = defaultdict(set)
    for q, t in top.filter((pl.col("fold") == h) & (pl.col("base_score") < 0.02)).select("q", "t").iter_rows(): sp_low[q].add(t)
    d = pl.concat([pl.read_parquet(f) for f in dense_all[h]])
    fd = pl.read_parquet(ROOT / f"outputs/experiments/CL-050/f{h}-dense.parquet") if h != 3 else pl.read_parquet(ROOT / "outputs/experiments/CL-050/f3-dense.parquet")
    pruned = defaultdict(set)
    for q, t in fd.filter(pl.col("p_text_oof") < 0.05).select("q", "t").iter_rows(): pruned[q].add(t)
    d11 = defaultdict(set)
    for q, t in d.filter(pl.col("rank") > 10).select("q", "t").iter_rows(): d11[q].add(t)
    miss = {q: truth.get(q, set()) - cand[q] for q in qs}
    nt = sum(len(truth.get(q, ())) for q in qs); nm = sum(len(v) for v in miss.values())
    def orc(extra):
        return float(np.mean([f05(((cand[q] | extra.get(q, set())) & truth.get(q, set())), truth.get(q, set())) for q in qs]))
    base = orc({})
    b = {"sparse_below_band": sp_low, "dense_pruned": pruned, "dense_rank11_50": d11}
    r = {"fold": h, "S1": len(qs), "true_links": nt, "missing_links": nm, "oracle": round(base, 6)}
    for k, v in b.items():
        r[f"{k}_links"] = sum(len(miss[q] & v.get(q, set())) for q in qs); r[f"{k}_oracle_gain"] = round(orc(v) - base, 6)
    allx = {q: sp_low.get(q, set()) | pruned.get(q, set()) | d11.get(q, set()) for q in qs}
    r["all_three_oracle_gain"] = round(orc(allx) - base, 6); r["nowhere_links"] = nm - sum(len(miss[q] & allx.get(q, set())) for q in qs)
    rows.append(r); print(json.dumps(r), flush=True)
pl.DataFrame(rows).write_csv(ROOT / "outputs/experiments/CL-061/oracle_buckets.csv")
