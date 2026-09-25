"""Test-side ownership diagnostics (no labels): per-country contest structure and variant link removals.

Input: CL-005 top-12 v2 stage-2 test probabilities (all test S1), threshold 0.67 (as SUB-003).
Outputs: outputs/analysis/ownership_variants/test_diagnostics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ov_lib as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TEST = ROOT / "student_resource/dataset/test"
PROBS = ROOT / "outputs/experiments/CL-003/test_probs/CL-005-top12v2-v1.parquet"
OUT = ROOT / "outputs/analysis/ownership_variants"
THR = 0.67
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def q3(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    return {k: round(float(np.percentile(x, v)), 6) for k, v in (("p10", 10), ("p50", 50), ("p90", 90))}


def main():
    claims = pl.scan_parquet(PROBS).filter(pl.col("p") >= THR).select("q", "t", "p").collect()
    log("claims", claims.height)
    s1 = (pl.scan_csv(TEST / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
          .select(pl.col("entity_id").alias("q"), "country",
                  pl.concat_str([pl.col(c).fill_null("\x00") for c in ("business_name", "business_address", "country")],
                                separator="\x1f").hash(7).alias("h")).collect())
    cq = s1.select("q", "country")
    groups = L.dup_groups(str(TEST / "test_source2.tsv"), str(TEST / "test_source3.tsv"))
    log("test dup-group rows", groups.height, "groups", groups["gid"].n_unique())
    ann_t = L.annotate(claims).join(cq, on="q", how="left")
    ann_g = L.annotate(claims, groups).join(cq, on="q", how="left")
    assert ann_t["country"].null_count() == 0

    countries = sorted(s1["country"].unique().to_list())
    out = {"threshold": THR, "claims": claims.height, "per_country": {}}

    # contested targets, keyed by the best claimant's country
    con = ann_t.filter(pl.col("n_s1") >= 2)
    hh = s1.select("q", "h")
    los = (con.filter(pl.col("q") != pl.col("best_q")).join(hh, on="q", how="left")
           .join(hh.rename({"q": "best_q", "h": "hb"}), on="best_q", how="left")
           .with_columns((pl.col("h") == pl.col("hb")).fill_null(False).alias("ident")))
    ident_t = los.group_by("t").agg(pl.col("ident").any())
    per_t = (con.group_by("t").agg(pl.col("n_s1").first(), pl.col("best_p").first(), pl.col("second_p").first(),
                                   pl.col("best_q").first(), pl.col("country").n_unique().alias("n_countries"))
             .join(cq.rename({"q": "best_q", "country": "bc"}), on="best_q", how="left")
             .join(ident_t, on="t", how="left")
             .with_columns((pl.col("best_p") - pl.col("second_p")).alias("margin")))
    # variant removals (claims keyed by the claimant's own country)
    kept = {}
    for name, rule, m, _ in L.VARIANTS[1:]:
        src = ann_g if rule == "E" else ann_t
        kept[name] = src.with_columns(L.keep_expr(rule, m).alias("keep"))
    exp = {}
    for base in ("B", "E_dupunit"):
        kb = kept[base].filter(pl.col("keep")).select("q", "t", "p")
        ke = L.dup_expand(kb, groups)
        exp[base + "+dupexpand"] = ke.join(kb.select("q", "t"), on=["q", "t"], how="anti").join(cq, on="q", how="left")
    s1_links_A = claims.group_by("q").agg(pl.len().alias("nA"))
    for c in countries + ["ALL"]:
        fc = (pl.lit(True) if c == "ALL" else pl.col("country") == c)
        fb = (pl.lit(True) if c == "ALL" else pl.col("bc") == c)
        pc = per_t.filter(fb)
        n_claims = ann_t.filter(fc).height
        dist = pc["n_s1"].to_numpy()
        d = {"s1": int(s1.filter(fc).height), "claims_A": n_claims,
             "contested_targets": pc.height,
             "claims_on_contested_targets": int(dist.sum()),
             "claims_per_contested_target": {"2": int((dist == 2).sum()), "3": int((dist == 3).sum()), "4": int((dist == 4).sum()),
                                             "5-9": int(((dist >= 5) & (dist <= 9)).sum()), "10+": int((dist >= 10).sum()),
                                             "mean": float(dist.mean()) if len(dist) else None, "max": int(dist.max()) if len(dist) else None},
             "margin_best_minus_second": q3(pc["margin"].to_numpy()),
             "margin_eq0": int((pc["margin"] == 0).sum()), "margin_lt0.02": int((pc["margin"] < 0.02).sum()),
             "margin_lt0.05": int((pc["margin"] < 0.05).sum()), "margin_lt0.10": int((pc["margin"] < 0.10).sum()),
             "margin_lt0.20": int((pc["margin"] < 0.20).sum()),
             "best_p_quantiles": q3(pc["best_p"].to_numpy()),
             "cross_country_contested": int((pc["n_countries"] > 1).sum()),
             "contested_with_raw_identical_S1_claimants": int(pc["ident"].fill_null(False).sum()),
             "margin0_and_raw_identical": int((pc["ident"].fill_null(False) & (pc["margin"] == 0)).sum()),
             "removed": {}}
        for name, kv in kept.items():
            r = kv.filter(fc & ~pl.col("keep"))
            d["removed"][name] = {"links": r.height, "frac_of_claims": r.height / max(n_claims, 1),
                                  "p_ge_0.9": int((r["p"] >= 0.9).sum()), "p_ge_0.95": int((r["p"] >= 0.95).sum())}
        rb = kept["B"].filter(fc & ~pl.col("keep"))
        d["B_removed_p_quantiles"] = q3(rb["p"].to_numpy())
        lost = rb.group_by("q").agg(pl.len().alias("lost")).join(s1_links_A, on="q")
        d["B_s1_losing_links"] = lost.height
        d["B_s1_emptied"] = int((lost["lost"] == lost["nA"]).sum())
        for name, e in exp.items():
            d["added_" + name] = e.filter(fc).height
        out["per_country"][c] = d
        log(c, json.dumps({k: d[k] for k in ("claims_A", "contested_targets", "margin_best_minus_second", "margin_eq0")}),
            "B removed", d["removed"]["B"])
    out["seconds"] = time.time() - T0
    out["fold4"] = "CLOSED"
    (OUT / "test_diagnostics.json").write_text(json.dumps(out, indent=1))
    log("wrote", OUT / "test_diagnostics.json")


if __name__ == "__main__":
    main()
