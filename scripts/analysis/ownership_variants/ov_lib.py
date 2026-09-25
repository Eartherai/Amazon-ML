"""Ownership-variant rules over above-threshold claims (q, t, p) and exact per-S1 F0.5.

Variants (claims are stage-2 links with p >= threshold; a unit is a target, or a raw
content-identical target group for E):
  A  no ownership
  B  highest-probability S1 keeps the unit (ties: smallest S1 id, as finalize_postprocess.py)
  C  margin-aware: a non-best claim is dropped only when best_p - p >= m
  D  ownership (B) only when best_p >= 0.9, otherwise keep all claims
  E  B over duplicate units (raw name+address+country identical targets = one unit)
  F  contested unit with best_p - second_p < 0.02 loses ALL claims; otherwise B
Duplicate expansion (suffix +X) mirrors scripts/classical/finalize_postprocess.py.
"""
from __future__ import annotations

from collections import defaultdict

import duckdb
import numpy as np
import polars as pl

CSV_OPTS = "delim='\t',header=true,quote='',escape='',all_varchar=true"


def dup_groups(source2: str, source3: str, mem_gb: int = 3) -> pl.DataFrame:
    """entity_id -> gid for raw content-identical target groups (size > 1), same SQL semantics as
    finalize_postprocess.py (NULL-safe name/address equality, exact country)."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem_gb}GB'")
    con.execute("SET threads=2")
    rows = con.execute(f"""WITH t AS (SELECT entity_id, business_name n, business_address a, country c FROM read_csv('{source2}',{CSV_OPTS})
          UNION ALL SELECT entity_id, business_name, business_address, country FROM read_csv('{source3}',{CSV_OPTS})),
        g AS (SELECT n, a, c, row_number() OVER () AS gid FROM (SELECT n, a, c FROM t GROUP BY n, a, c HAVING count(*) > 1))
        SELECT t.entity_id, g.gid FROM t JOIN g ON t.n IS NOT DISTINCT FROM g.n AND t.a IS NOT DISTINCT FROM g.a AND t.c=g.c""").fetchall()
    con.close()
    return pl.DataFrame(rows, schema={"t": pl.String, "gid": pl.Int64}, orient="row")


def annotate(claims: pl.DataFrame, groups: pl.DataFrame | None = None) -> pl.DataFrame:
    """Add unit u, claimant unit score pq, best_q, best_p, second_p, n_s1 to each claim."""
    c = claims.select("q", "t", "p")
    if groups is None:
        c = c.with_columns(pl.col("t").alias("u"))
    else:
        c = c.join(groups, on="t", how="left").with_columns(
            pl.when(pl.col("gid").is_null()).then(pl.col("t")).otherwise(pl.lit("G") + pl.col("gid").cast(pl.String)).alias("u")).drop("gid")
    qp = c.group_by(["u", "q"]).agg(pl.col("p").max().alias("pq")).sort(["u", "pq", "q"], descending=[False, True, False])
    qp = qp.with_columns(pl.int_range(pl.len()).over("u").alias("r"))
    best = qp.filter(pl.col("r") == 0).select("u", pl.col("q").alias("best_q"), pl.col("pq").alias("best_p"))
    second = qp.filter(pl.col("r") == 1).select("u", pl.col("pq").alias("second_p"))
    n = qp.group_by("u").agg(pl.len().alias("n_s1"))
    stats = best.join(second, on="u", how="left").join(n, on="u")
    return c.join(qp.select("u", "q", "pq"), on=["u", "q"]).join(stats, on="u")


def keep_expr(rule: str, m: float | None = None) -> pl.Expr:
    isbest = pl.col("q") == pl.col("best_q")
    single = pl.col("n_s1") == 1
    if rule == "A":
        return pl.lit(True)
    if rule in ("B", "E"):
        return isbest
    if rule == "C":
        return isbest | ((pl.col("best_p") - pl.col("pq")) < m)
    if rule == "D":
        return single | isbest | (pl.col("best_p") < m)
    if rule == "F":
        return single | (isbest & ((pl.col("best_p") - pl.col("second_p")) >= m))
    raise ValueError(rule)


VARIANTS = [
    ("A", "A", None, False),
    ("B", "B", None, False),
    ("C_m0.02", "C", 0.02, False),
    ("C_m0.05", "C", 0.05, False),
    ("C_m0.10", "C", 0.10, False),
    ("C_m0.20", "C", 0.20, False),
    ("D_best0.9", "D", 0.9, False),
    ("E_dupunit", "E", None, False),
    ("F_dropall0.02", "F", 0.02, False),
]
# exploratory (not pre-specified): other drop-all margins and dropping every contested claim
VARIANTS_EXTRA = [
    ("Fx_dropall0.01", "F", 0.01, False),
    ("Fx_dropall0.05", "F", 0.05, False),
    ("Fx_dropall0.10", "F", 0.10, False),
    ("Fx_dropall0.20", "F", 0.20, False),
    ("Fx_dropall_contested", "F", 2.0, False),
]
VARIANTS_ALL = VARIANTS + VARIANTS_EXTRA


def apply_variant(ann_t: pl.DataFrame, ann_g: pl.DataFrame, rule: str, m: float | None) -> pl.DataFrame:
    src = ann_g if rule == "E" else ann_t
    return src.filter(keep_expr(rule, m)).select("q", "t", "p")


def dup_expand(kept: pl.DataFrame, groups: pl.DataFrame) -> pl.DataFrame:
    """Add unclaimed raw-identical copies of kept targets to the same S1 (finalize_postprocess.py order)."""
    members = defaultdict(list)
    gid_of = {}
    for t, g in groups.iter_rows():
        members[g].append(t)
        gid_of[t] = g
    kept = kept.sort(["t", "p", "q"], descending=[False, True, False])
    claimed = set(kept["t"].to_list())
    extra = []
    for q, t, p in kept.filter(pl.col("t").is_in(list(gid_of))).select("q", "t", "p").iter_rows():
        for other in members[gid_of[t]]:
            if other not in claimed:
                claimed.add(other)
                extra.append((q, other, p))
    if not extra:
        return kept
    return pl.concat([kept, pl.DataFrame(extra, schema={"q": pl.String, "t": pl.String, "p": pl.Float64}, orient="row")])


def per_s1_scores(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame):
    """Exact per-S1 F0.5 aligned to s1['q'] order, plus tp/npred/ntrue arrays. empty/empty = 1."""
    pred = pred.select("q", "t").unique()
    tp = pred.join(truth, on=["q", "t"], how="inner").group_by("q").agg(pl.len().alias("tp"))
    npred = pred.group_by("q").agg(pl.len().alias("np"))
    ntrue = truth.group_by("q").agg(pl.len().alias("nt"))
    d = (s1.select("q").with_row_index("i").join(npred, on="q", how="left").join(ntrue, on="q", how="left")
         .join(tp, on="q", how="left").fill_null(0).sort("i"))
    tpa, npa, nta = (d[c].to_numpy().astype(np.float64) for c in ("tp", "np", "nt"))
    f = np.zeros(len(d))
    both_empty = (npa == 0) & (nta == 0)
    f[both_empty] = 1.0
    ok = tpa > 0
    prec = np.where(ok, tpa / np.maximum(npa, 1), 0)
    rec = np.where(ok, tpa / np.maximum(nta, 1), 0)
    f[ok] = 1.25 * prec[ok] * rec[ok] / (0.25 * prec[ok] + rec[ok])
    return f, tpa, npa, nta


def summarize(f, tpa, npa, nta, country: np.ndarray) -> dict:
    out = {"macro_f0_5": float(f.mean()), "precision": float(tpa.sum() / max(npa.sum(), 1)),
           "recall": float(tpa.sum() / max(nta.sum(), 1)), "singleton_f0_5": float(f[nta == 0].mean()),
           "nonsingleton_f0_5": float(f[nta > 0].mean()), "links": int(npa.sum()), "n": int(len(f))}
    for c in np.unique(country):
        out[str(c)] = float(f[country == c].mean())
    return out


def paired_bootstrap(D: np.ndarray, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """D: n x V per-entity deltas. Returns n_boot x V bootstrap means (same resamples for all columns)."""
    n = D.shape[0]
    nz = np.flatnonzero(np.any(D != 0, axis=1))
    Dn = D[nz]
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, D.shape[1]))
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, n, n), minlength=n)
        out[b] = cnt[nz] @ Dn / n
    return out
