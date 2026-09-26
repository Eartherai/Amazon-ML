"""CL-028: compact cascade submission — matching_results.tsv plus the honest candidate_pairs.tsv.

Cascade (per country partition):
  retrieval  char3 name/address top-100 union (+ France test-fitted char3 route) and e5-small dense top-10
  pruning    stage-1 LightGBM top-12 with base >= --cand-base; dense / France-route pairs with text-only p >= --cand-text
  candidates exactly the pruned pairs (plus the exact-duplicate route below); written to candidate_pairs.tsv
  matching   stage-2 + e5-base CE stack (saved test probabilities from assemble_final_v2.py) on sparse candidates,
             max-probability ownership at --thr; dense/France-route additions at p >= --add-thr on unclaimed targets,
             ownership among additions; exact content-duplicate expansion (a deterministic exact-key blocking route,
             its pairs are added to the candidate set)
  France     'new' uses the pipeline; 'vsafe' keeps VSAFE France rows restricted to the France candidates
Every matched pair is a candidate pair by construction (asserted). Fold4 CLOSED.
"""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import duckdb, numpy as np, polars as pl

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "student_resource/dataset/test"
OPTS = "delim='\t',header=true,quote='',escape='',all_varchar=true"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--stack", default=str(ROOT / "outputs/experiments/CL-014/test-stack-CL-025-B-ce4ep-dense-frvsafe.parquet"))
    ap.add_argument("--dense", default=str(ROOT / "outputs/experiments/CL-025/dense-test-all-scored.parquet"))
    ap.add_argument("--fr-route", default=str(ROOT / "outputs/experiments/CL-012/France-rescue.parquet"))
    ap.add_argument("--fr-add", default=str(ROOT / "outputs/experiments/CL-012/France-rescue-VSAFE.parquet"))
    ap.add_argument("--vsafe", default=str(ROOT / "outputs/submissions/UPLOAD_FINAL_VSAFE/UPLOAD_FINAL_VSAFE_matching_results.tsv"))
    ap.add_argument("--france", choices=["new", "vsafe"], default="vsafe")
    ap.add_argument("--cand-base", type=float, default=0.02)
    ap.add_argument("--cand-text", type=float, default=0.2)
    ap.add_argument("--thr", type=float, default=0.70)
    ap.add_argument("--add-thr", type=float, default=0.8)
    a = ap.parse_args()
    s1 = pl.read_csv(TEST / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0).select(pl.col("entity_id").alias("q"), "country")
    order, country = s1["q"].to_list(), dict(s1.iter_rows())
    # ---- candidates
    base = pl.read_parquet(ROOT / "outputs/experiments/CL-003/test_probs/CL-005-top12v2-v1.parquet").select("q", "t", "base").unique(["q", "t"])
    sp = pl.read_parquet(a.stack).join(base, on=["q", "t"], how="inner").filter(pl.col("base") >= a.cand_base)
    dn = pl.read_parquet(a.dense).select("q", "t", "p_text").filter(pl.col("p_text") >= a.cand_text)
    fr = pl.read_parquet(a.fr_route).select("q", "t", pl.col("p").alias("p_text")).filter(pl.col("p_text") >= a.cand_text)
    extra = pl.concat([dn, fr]).group_by("q", "t").agg(pl.col("p_text").max())
    cand = defaultdict(set)
    for q, t in sp.select("q", "t").iter_rows(): cand[q].add(t)
    for q, t in extra.select("q", "t").iter_rows(): cand[q].add(t)
    # ---- sparse decisions (ownership among candidates)
    best = {}
    for q, t, p in sp.select("q", "t", "p").iter_rows():
        if p >= a.thr and (t not in best or p > best[t][1]): best[t] = (q, p)
    chosen = defaultdict(set)
    for t, (q, _) in best.items(): chosen[q].add(t)
    claimed = set(best)
    # ---- additions: dense / France-route candidates at add-thr, plus VSAFE-filtered France rescue pairs that are candidates
    fa = pl.read_parquet(a.fr_add).select("q", "t", pl.col("p").alias("p_text"))
    pool = pl.concat([extra.filter(pl.col("p_text") >= a.add_thr), fa]).group_by("q", "t").agg(pl.col("p_text").max()).sort(["p_text", "q", "t"], descending=[True, False, False])
    add = {}
    for q, t, p in pool.iter_rows():
        if t in claimed or t in add or t not in cand.get(q, ()): continue
        add[t] = q
    for t, q in add.items(): chosen[q].add(t)
    # ---- France policy
    if a.france == "vsafe":
        with open(a.vsafe) as f:
            f.readline()
            for line in f:
                q, _, raw = line.rstrip("\n").partition("\t")
                if country[q] == "France":
                    chosen[q] = {t for t in raw.split(",") if t and t in cand.get(q, ())}
    # ---- exact content-duplicate route
    con = duckdb.connect()
    groups = con.execute(f"""WITH t AS (SELECT entity_id, business_name n, business_address a, country c FROM read_csv('{TEST}/test_source2.tsv',{OPTS})
          UNION ALL SELECT entity_id, business_name, business_address, country FROM read_csv('{TEST}/test_source3.tsv',{OPTS})),
        g AS (SELECT n, a, c FROM t GROUP BY n, a, c HAVING count(*) > 1)
        SELECT t.entity_id, t.n, t.a, t.c FROM t JOIN g ON t.n IS NOT DISTINCT FROM g.n AND t.a IS NOT DISTINCT FROM g.a AND t.c=g.c ORDER BY t.entity_id""").fetchall()
    members, key_of = defaultdict(list), {}
    for i, n, ad, c in groups: members[(n, ad, c)].append(i); key_of[i] = (n, ad, c)
    allc = {t for v in chosen.values() for t in v}; dup = 0
    for q in order:
        for t in sorted(chosen.get(q, ())):
            if t in key_of:
                for o in members[key_of[t]]:
                    if o not in allc: allc.add(o); chosen[q].add(o); cand[q].add(o); dup += 1
    # ---- write
    dest = ROOT / "outputs/submissions" / a.name; dest.mkdir(parents=True, exist_ok=False)
    mp, cp = dest / "matching_results.tsv", dest / "candidate_pairs.tsv"
    per = defaultdict(lambda: {"s1": 0, "links": 0, "empty": 0, "cand": 0}); cn = []
    with open(mp, "w") as fm, open(cp, "w") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n"); fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for q in order:
            m, c = chosen.get(q, set()), cand.get(q, set())
            assert m <= c, q
            fm.write(q + "\t" + ",".join(sorted(m)) + "\n"); fc.write(q + "\t" + ",".join(sorted(c)) + "\n")
            k = country[q]; per[k]["s1"] += 1; per[k]["links"] += len(m); per[k]["empty"] += int(not m); per[k]["cand"] += len(c); cn.append(len(c))
    cn = np.array(cn)
    prov = {"matching_sha256": sha(mp), "candidate_sha256": sha(cp), "args": vars(a), "dup_added": dup, "dense_or_route_added": len(add),
            "cand_per_s1": {"avg": float(cn.mean()), "p50": float(np.percentile(cn, 50)), "p90": float(np.percentile(cn, 90)), "p95": float(np.percentile(cn, 95)),
                            "p99": float(np.percentile(cn, 99)), "max": int(cn.max()), "pairs": int(cn.sum())},
            "per_country": {k: {**v, "cand_per_s1": v["cand"] / v["s1"]} for k, v in per.items()}, "fold4": "CLOSED"}
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2)); print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
