"""Final set construction from saved test pair probabilities.

Steps (each switchable, counts recorded):
 1. threshold stage-2 probabilities (+ optional accepted rescue pairs),
 2. ownership: a target predicted for several S1 is kept only for the highest-probability S1
    (training truth: every target has at most one owner; EXP-034 OOF +0.00047),
 3. duplicate expansion: raw content-identical copies (same name/address/country) of a kept
    target are added to the same S1 when unclaimed (training: 43,910/43,910 groups single-owner),
 4. France policy: stage-2 or frozen SUB-002 V001 rows.
Writes matching TSV (test order) and an augmented candidate-additions list for the package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "student_resource/dataset/test"
BASE = ROOT / "outputs/submissions/SUB-002/validated-v001/matching_results.tsv"
HEADER = "source1_entity_id\tmatched_entity_ids\n"
OPTS = "delim='\t',header=true,quote='',escape='',all_varchar=true"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True, help="parquet q,t,p from predict_generic")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--rescue", type=Path, help="parquet q,t,p rescue pairs")
    ap.add_argument("--rescue-threshold", type=float, default=0.66)
    ap.add_argument("--ownership", action="store_true")
    ap.add_argument("--dup-expand", action="store_true")
    ap.add_argument("--france", choices=["stage2", "base"], default="stage2")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    df = pl.read_parquet(a.probs).select("q", "t", "p").filter(pl.col("p") >= a.threshold).with_columns(pl.lit("stage2").alias("src"))
    stats = {"stage2_links": len(df)}
    if a.rescue:
        r = pl.read_parquet(a.rescue).select("q", "t", "p").filter(pl.col("p") >= a.rescue_threshold).with_columns(pl.lit("rescue").alias("src"))
        stats["rescue_links"] = len(r)
        df = pl.concat([df, r]).unique(subset=["q", "t"], keep="first")
    if a.ownership:
        before = len(df)
        df = df.sort(["t", "p", "q"], descending=[False, True, False]).unique(subset=["t"], keep="first", maintain_order=True)
        stats["ownership_removed"] = before - len(df)
    added = []
    if a.dup_expand:
        con = duckdb.connect()
        groups = con.execute(f"""WITH t AS (SELECT entity_id, business_name n, business_address a, country c FROM read_csv('{TEST}/test_source2.tsv',{OPTS})
              UNION ALL SELECT entity_id, business_name, business_address, country FROM read_csv('{TEST}/test_source3.tsv',{OPTS})),
            g AS (SELECT n, a, c FROM t GROUP BY n, a, c HAVING count(*) > 1)
            SELECT t.entity_id, t.n, t.a, t.c FROM t JOIN g ON t.n IS NOT DISTINCT FROM g.n AND t.a IS NOT DISTINCT FROM g.a AND t.c=g.c""").fetchall()
        members = defaultdict(list)
        key_of = {}
        for i, n, ad, c in groups:
            members[(n, ad, c)].append(i)
            key_of[i] = (n, ad, c)
        claimed = set(df["t"].to_list())
        extra = []
        for q, t, p in df.filter(pl.col("t").is_in(list(key_of))).select("q", "t", "p").iter_rows():
            for other in members[key_of[t]]:
                if other not in claimed:
                    claimed.add(other)
                    extra.append((q, other, p, "dup"))
        added = extra
        stats["dup_expanded_links"] = len(extra)
        if extra:
            df = pl.concat([df, pl.DataFrame(extra, schema=["q", "t", "p", "src"], orient="row")])
    chosen = defaultdict(list)
    for q, t in df.select("q", "t").iter_rows():
        chosen[q].append(t)
    s1 = pl.read_csv(TEST / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    order, country = s1["entity_id"].to_list(), dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
    base = {}
    with BASE.open() as f:
        assert f.readline() == HEADER
        for line in f:
            q, _, raw = line.rstrip("\n").partition("\t")
            base[q] = raw.split(",") if raw else []
    dest = ROOT / "outputs/submissions" / a.name
    dest.mkdir(parents=True, exist_ok=False)
    path = dest / "matching_results.tsv"
    per = defaultdict(lambda: {"s1": 0, "links": 0, "empty": 0})
    with path.open("w") as f:
        f.write(HEADER)
        for q in order:
            c = country[q]
            ids = base[q] if (a.france == "base" and c not in ("India", "US")) else sorted(set(chosen.get(q, [])))
            f.write(q + "\t" + ",".join(ids) + "\n")
            per[c]["s1"] += 1
            per[c]["links"] += len(ids)
            per[c]["empty"] += int(not ids)
    extra_cands = df.filter(pl.col("src").is_in(["rescue", "dup"])).select("q", "t")
    extra_cands.write_parquet(dest / "candidate_additions.parquet")
    prov = {"matching_sha256": sha(path), "args": {k: str(v) for k, v in vars(a).items()}, "stats": stats,
            "per_country": {k: dict(v) for k, v in per.items()}, "candidate_additions": len(extra_cands), "fold4": "CLOSED"}
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
