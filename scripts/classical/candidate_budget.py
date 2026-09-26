"""CL-027: candidate-budget curves and system table (organizer update: candidate_pairs.tsv is ranked, smaller is better).

All numbers are out-of-fold on labeled train S1 with exact per-S1 truth (retrieval misses count). Fold4 CLOSED.
  RAW   P4-B-001: 20k natural S1, full char3 name top-100 + address top-100 union (~197/S1), nested stage-1 OOF score.
  TOP12 CL-003: 200k S1 (folds 1-3), stage-1 top-12 with OOF base score (SUB-002 lineage model).
  DENSE CL-021: fold-3 India/US, e5-small bi-encoder (trained folds 1-2) top-50 new pairs outside sparse top-12,
        top-10 of those scored by the text-only pair model (p_text).
Per system: link recall, complete-entity recall (S1 with >=1 true link, all links present), oracle macro F0.5
(predict exactly the true candidates; empty truth -> empty -> 1), candidates/S1 avg/p50/p90/p95/p99/max, total pairs.
"""
import json
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/CL-027"; OUT.mkdir(parents=True, exist_ok=True)
gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
gt = gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias("T"))
n_true = dict(zip(gt["source1_entity_id"].to_list(), gt["T"].list.len().to_list()))
truth_pairs = gt.select(pl.col("source1_entity_id").alias("q"), pl.col("T").alias("t")).explode("t").drop_nulls().with_columns(pl.lit(1).alias("y"))


def summarize(name, cand, qs, extra=None):
    """cand: DataFrame q,t (unique) restricted to population qs (list). Label by exact truth."""
    c = cand.join(truth_pairs, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    per = c.group_by("q").agg(pl.len().alias("n"), pl.col("y").sum().alias("hit"))
    base = pl.DataFrame({"q": qs, "nt": [n_true.get(q, 0) for q in qs]})
    per = base.join(per, on="q", how="left").with_columns(pl.col("n").fill_null(0), pl.col("hit").fill_null(0))
    nt, hit, n = per["nt"].to_numpy(), per["hit"].to_numpy(), per["n"].to_numpy()
    r = np.where(nt > 0, hit / np.maximum(nt, 1), 1.0)
    orc = np.where(nt == 0, 1.0, np.where(hit == 0, 0.0, 1.25 * r / (0.25 + r)))
    pos = nt > 0
    row = {"system": name, "s1": len(qs), "link_recall": float(hit.sum() / nt.sum()), "complete_entity_recall": float(np.mean(hit[pos] == nt[pos])),
           "oracle_f05": float(orc.mean()), "avg": float(n.mean()), "p50": float(np.percentile(n, 50)), "p90": float(np.percentile(n, 90)),
           "p95": float(np.percentile(n, 95)), "p99": float(np.percentile(n, 99)), "max": int(n.max()), "pairs": int(n.sum())}
    if extra: row.update(extra)
    return row


def main():
    rows = []
    # ---------------- RAW pool (20k) ----------------
    raw = pl.read_parquet(ROOT / "outputs/oof/P4-B-001/pair_scores.parquet").rename({"source1_entity_id": "q", "target_id": "t"}).unique(["q", "t"])
    raw = raw.with_columns(pl.col("score").rank("ordinal", descending=True).over("q").alias("r"))
    qs = raw["q"].unique().sort().to_list()
    rows.append(summarize("RAW char3 name+addr top100 union", raw.select("q", "t"), qs, {"source": "RAW"}))
    for k in (1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100):
        rows.append(summarize(f"RAW stage1 top-{k}", raw.filter(pl.col("r") <= k).select("q", "t"), qs, {"source": "RAW", "K": k}))
    for th in (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
        rows.append(summarize(f"RAW stage1 p>={th}", raw.filter(pl.col("score") >= th).select("q", "t"), qs, {"source": "RAW", "thr": th}))
        for k in (8, 12, 20):
            rows.append(summarize(f"RAW stage1 top-{k} & p>={th}", raw.filter((pl.col("score") >= th) & (pl.col("r") <= k)).select("q", "t"), qs, {"source": "RAW", "thr": th, "K": k}))
    # pruner operating points: fraction of in-pool positives kept by a global score threshold
    ps = np.sort(raw.filter(pl.col("label") == 1)["score"].to_numpy())
    for keep in (0.99, 0.995, 0.997, 0.999):
        th = float(ps[int(np.floor((1 - keep) * len(ps)))])
        rows.append(summarize(f"RAW pruner keep {keep:.1%} of pool positives", raw.filter(pl.col("score") >= th).select("q", "t"), qs, {"source": "RAW", "thr": th, "pool_pos_keep": keep}))
    # ---------------- TOP12 (200k) ----------------
    top = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-003/s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)]).rename({"source1_entity_id": "q", "target_id": "t"})
    q2 = top["q"].unique().sort().to_list()
    for k in (1, 2, 3, 5, 8, 10, 12):
        rows.append(summarize(f"TOP12 stage1 top-{k}", top.filter(pl.col("rank") <= k).select("q", "t"), q2, {"source": "TOP12", "K": k}))
    for th in (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
        rows.append(summarize(f"TOP12 top-12 & p>={th}", top.filter(pl.col("base_score") >= th).select("q", "t"), q2, {"source": "TOP12", "thr": th, "K": 12}))
    # ---------------- fold-3 sparse + dense ----------------
    t3 = top.filter(pl.col("fold") == 3)
    dn = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-021/held-fold3-{c}-dense.parquet") for c in ("India", "US")]).select("q", "t", pl.col("rank").alias("drank"))
    sc = pl.concat([pl.read_parquet(ROOT / f"outputs/experiments/CL-021/f3-{c}-dense-top10-scored.parquet") for c in ("India", "US")]).select("q", "t", "p_text")
    q3 = sorted(set(t3["q"].to_list()) & set(dn["q"].to_list()))
    t3 = t3.filter(pl.col("q").is_in(q3)); dn = dn.filter(pl.col("q").is_in(q3)).join(sc, on=["q", "t"], how="left")
    S = lambda df: df.select("q", "t")
    rows.append(summarize("F3 sparse top-12", S(t3), q3, {"source": "F3"}))
    rows.append(summarize("F3 CE band: top-12 & p>=0.02", S(t3.filter(pl.col("base_score") >= 0.02)), q3, {"source": "F3"}))
    for kd in (5, 10, 20, 50):
        rows.append(summarize(f"F3 sparse top-12 + dense top-{kd} new", pl.concat([S(t3), S(dn.filter(pl.col("drank") <= kd))]).unique(), q3, {"source": "F3"}))
    for th in (0.01, 0.02, 0.05):
        for pt in (0.05, 0.2, 0.5):
            c = pl.concat([S(t3.filter(pl.col("base_score") >= th)), S(dn.filter((pl.col("drank") <= 10) & (pl.col("p_text") >= pt)))]).unique()
            rows.append(summarize(f"F3 band p>={th} + dense top-10 p_text>={pt}", c, q3, {"source": "F3", "thr": th, "dense_thr": pt}))
    df = pl.DataFrame(rows)
    df.write_csv(OUT / "candidate_budget.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=250):
        print(df.select("system", "s1", "link_recall", "complete_entity_recall", "oracle_f05", "avg", "p50", "p95", "p99", "max", "pairs"))


if __name__ == "__main__":
    main()
