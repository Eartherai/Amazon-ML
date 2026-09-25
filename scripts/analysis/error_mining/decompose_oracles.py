"""ERR-MINE-001 step 2: exact macro F0.5 loss decomposition and oracles on the 194k training OOF.

Per S1 with truth T, current prediction P = {t in top-12 keys: p >= thr}:
  F0.5(tp, npred, ntrue) = 1.25 tp / (npred + 0.25 ntrue), empty/empty = 1.
  f_cur  = F(P, T); f_nofp = F(P & T, T); f_top = F((P & T) | (T & keys), T)
  singleton S1 (T empty): loss = 1 - f_cur (any prediction -> 0)
  non-singleton: fp = f_nofp - f_cur, rejected = f_top - f_nofp, outside = 1 - f_top
Oracles: top-12 (pred = T & keys), entity-decision (best per-S1 threshold on its own p, ties kept together),
global-threshold sweep, within-sample ownership oracle, cross-population ownership diagnostics.
Writes outputs/analysis/error_mining/decomposition.json and s1_losses.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/analysis/error_mining"
THR = json.loads((ROOT / "outputs/experiments/CL-003/stage2-200k-top12-v2.json").read_text())["threshold"]


def f05(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(a, dtype=np.float64) for a in (tp, npred, ntrue))
    den = npred + 0.25 * ntrue
    out = np.where(den > 0, 1.25 * tp / np.where(den > 0, den, 1.0), 1.0)
    return np.where((npred == 0) & (ntrue == 0), 1.0, out)


def boot_ci(v, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def macro_from(links_pred: pl.DataFrame, s1: pl.DataFrame) -> tuple[float, np.ndarray]:
    """links_pred: rows (q, label) of predicted links. Returns macro and per-S1 vector aligned to s1."""
    agg = links_pred.group_by("q").agg(pl.len().alias("np"), pl.col("label").sum().alias("tp"))
    m = s1.select("q", "n_true").join(agg, on="q", how="left").fill_null(0)
    per = f05(m["tp"].to_numpy(), m["np"].to_numpy(), m["n_true"].to_numpy())
    return float(per.mean()), per


def main() -> None:
    L = pl.read_parquet(OUT / "links.parquet", columns=["q", "t", "p", "label", "rank", "t_s2", "country", "n_true"])
    s1 = pl.read_parquet(OUT / "s1.parquet").sort("q")
    truth = pl.read_parquet(OUT / "truth_links.parquet")
    n = s1.height
    L = L.with_columns((pl.col("p") >= THR).alias("pred"))

    agg = L.group_by("q").agg(pl.col("pred").sum().alias("n_pred"), (pl.col("pred") & (pl.col("label") == 1)).sum().alias("tp"),
                              (pl.col("label") == 1).sum().alias("n_true_keys_chk"))
    S = s1.join(agg, on="q", how="left").fill_null(0)
    assert (S["n_true_keys"] == S["n_true_keys_chk"]).all()
    tp, npd, nt, ntk = (S[c].to_numpy() for c in ("tp", "n_pred", "n_true", "n_true_keys"))
    f_cur = f05(tp, npd, nt)
    f_nofp = f05(tp, tp, nt)
    f_top = f05(ntk, ntk, nt)
    single = nt == 0
    loss = 1 - f_cur
    comp = {
        "singleton_false_merge": float(loss[single].sum() / n),
        "fp_links_nonsingleton": float((f_nofp - f_cur)[~single].sum() / n),
        "rejected_true_in_top12": float((f_top - f_nofp)[~single].sum() / n),
        "true_outside_top12": float((1 - f_top)[~single].sum() / n),
    }
    # split outside-top12 into protocol filter (true link in raw top-12 but removed because eval-owned) vs real miss
    ntr = S["n_true_top12raw"].to_numpy()
    f_topraw = f05(ntr, ntr, nt)
    comp_outside_split = {
        "outside_raw_top12_rank_gt12_or_unretrieved": float((1 - f_topraw)[~single].sum() / n),
        "removed_by_eval_owned_target_filter": float((f_topraw - f_top)[~single].sum() / n),
    }
    macro = float(f_cur.mean())
    total_loss = 1 - macro
    assert abs(sum(comp.values()) - total_loss) < 1e-9

    # per country / singleton slices
    country = S["country"].to_numpy()
    slices = {}
    for c in sorted(set(country.tolist())):
        m = country == c
        slices[c] = {"n": int(m.sum()), "macro": float(f_cur[m].mean()), "singleton_share": float(single[m].mean()),
                     "loss_mass_of_total": float(loss[m].sum() / n),
                     "singleton_false_merge": float(loss[m & single].sum() / n),
                     "fp_links_nonsingleton": float((f_nofp - f_cur)[m & ~single].sum() / n),
                     "rejected_true_in_top12": float((f_top - f_nofp)[m & ~single].sum() / n),
                     "true_outside_top12": float((1 - f_top)[m & ~single].sum() / n)}

    # oracles
    oracle_top12 = float(f_top.mean())
    oracle_nofp = float(f_nofp.mean())
    oracle_no_reject = float(f05(ntk, npd + (ntk - tp), nt).mean())  # keep FPs, add every rejected true link
    # entity-decision oracle: best prefix over distinct p levels (ties kept together), empty allowed
    G = (L.group_by("q", "p").agg(pl.len().alias("k"), pl.col("label").sum().alias("y"))
         .sort(["q", "p"], descending=[False, True])
         .with_columns(pl.col("k").cum_sum().over("q").alias("ck"), pl.col("y").cum_sum().over("q").alias("cy"))
         .join(S.select("q", "n_true"), on="q", how="left"))
    G = G.with_columns(pl.Series("f", f05(G["cy"].to_numpy(), G["ck"].to_numpy(), G["n_true"].to_numpy())))
    best = G.group_by("q").agg(pl.col("f").max().alias("fbest"))
    E = S.select("q", "n_true").join(best, on="q", how="left").fill_null(0.0)
    f_empty = f05(0, 0, E["n_true"].to_numpy())
    f_ent = np.maximum(E["fbest"].to_numpy(), f_empty)
    oracle_entity = float(f_ent.mean())
    # entity-decision oracle restricted to "keep/drop the whole current decision or its top-k prefix": same object;
    # also report how many S1 the oracle changes
    ent_changed = int((f_ent > f_cur + 1e-12).sum())

    # global threshold sweep (in-sample; diagnostic only)
    sweep = {}
    for thr in np.round(np.arange(0.40, 0.91, 0.01), 2):
        mm, _ = macro_from(L.filter(pl.col("p") >= thr).select("q", "label"), S)
        sweep[float(thr)] = mm
    best_thr = max(sweep, key=sweep.get)

    # ownership: within-sample conflicts at threshold
    PR = L.filter(pl.col("pred")).select("q", "t", "p", "label")
    claims = PR.group_by("t").agg(pl.len().alias("n_claim"))
    PR = PR.join(claims, on="t", how="left")
    conflict_fp = PR.filter((pl.col("n_claim") > 1) & (pl.col("label") == 0))
    m_own_oracle, per_own = macro_from(PR.filter(~((pl.col("n_claim") > 1) & (pl.col("label") == 0))).select("q", "label"), S)
    # realized max-probability ownership (reproduces CL-008)
    best_owner = PR.sort("p", descending=True).group_by("t").agg(pl.col("q").first().alias("owner"))
    own_pred = PR.join(best_owner, on="t").filter(pl.col("q") == pl.col("owner"))
    m_own_real, per_own_real = macro_from(own_pred.select("q", "label"), S)

    # cross-population ownership: FP links whose target is truly owned by some non-fold4 training S1
    owned = pl.read_parquet(OUT / "owned_targets.parquet")
    fp_all = PR.filter(pl.col("label") == 0).join(owned.select("t", "n_owners"), on="t", how="left")
    fp_owned = fp_all.filter(pl.col("n_owners").is_not_null())
    m_xown_oracle, per_xown = macro_from(PR.join(fp_owned.select("q", "t").with_columns(pl.lit(True).alias("drop")), on=["q", "t"], how="left")
                                         .filter(pl.col("drop").is_null()).select("q", "label"), S)
    # owner inside the 194k sample vs outside
    s1_set = S.select("q")
    tl = truth.select(pl.col("q").alias("owner_q"), "t")
    fp_owner_in_sample = fp_owned.join(tl, on="t", how="inner").select("q", "t").unique().height

    # combined oracles
    both = float(np.maximum(f_ent, f_cur).mean())

    # outside-top12 counts by country and source
    outside = truth.filter(~pl.col("in_keys")).join(S.select("q", "country"), on="q", how="left").with_columns(
        pl.col("t").str.slice(0, 2).alias("src"))
    out_tab = outside.group_by("country", "src", "in_top12_raw").len().sort("len", descending=True)
    tl_tab = truth.join(S.select("q", "country"), on="q", how="left").with_columns(pl.col("t").str.slice(0, 2).alias("src")).group_by(
        "country", "src").len().rename({"len": "n_true_links"})
    out_rate = (outside.filter(~pl.col("in_top12_raw")).group_by("country", "src").len().join(tl_tab, on=["country", "src"])
                .with_columns((pl.col("len") / pl.col("n_true_links")).alias("miss_rate")).sort("len", descending=True))

    # per-S1 table for step 3
    S2 = S.with_columns(pl.Series("f_cur", f_cur), pl.Series("f_nofp", f_nofp), pl.Series("f_top", f_top),
                        pl.Series("f_entity", f_ent), pl.Series("single", single))
    S2.write_parquet(OUT / "s1_losses.parquet")

    per_cur = f_cur
    res = {
        "experiment": "ERR-MINE-001",
        "threshold": THR,
        "n_s1": n,
        "n_singleton_s1": int(single.sum()),
        "n_links_top12": L.height,
        "n_true_links": int(nt.sum()),
        "n_true_links_in_top12_keys": int(ntk.sum()),
        "n_pred_links": int(npd.sum()),
        "n_tp": int(tp.sum()),
        "n_fp": int((npd - tp).sum()),
        "n_fp_on_singletons": int((npd - tp)[single].sum()),
        "n_rejected_true_in_top12": int((ntk - tp).sum()),
        "n_true_outside_top12": int((nt - ntk).sum()),
        "n_true_outside_but_in_raw_top12_eval_filter": int((ntr - ntk).sum()),
        "macro_f05_current": macro,
        "macro_ci95": boot_ci(per_cur),
        "singleton_f05": float(f_cur[single].mean()),
        "nonsingleton_f05": float(f_cur[~single].mean()),
        "total_loss": total_loss,
        "loss_components": comp,
        "loss_components_share": {k: v / total_loss for k, v in comp.items()},
        "outside_split": comp_outside_split,
        "oracle_no_fp": oracle_nofp,
        "oracle_no_rejected": oracle_no_reject,
        "oracle_top12": oracle_top12,
        "oracle_top12_raw_before_eval_filter": float(f_topraw.mean()),
        "oracle_entity_decision": oracle_entity,
        "oracle_entity_decision_s1_improved": ent_changed,
        "global_threshold_sweep_best": [best_thr, sweep[best_thr]],
        "global_threshold_sweep_at_0.67": sweep.get(0.67),
        "ownership_within_sample": {
            "realized_maxprob": m_own_real, "realized_delta": m_own_real - macro,
            "oracle_perfect_conflict_resolution": m_own_oracle, "oracle_delta": m_own_oracle - macro,
            "targets_claimed_by_multiple_s1": int((claims["n_claim"] > 1).sum()),
            "fp_links_in_conflicts": conflict_fp.height,
        },
        "ownership_cross_population_non_fold4": {
            "fp_links_total": fp_all.height,
            "fp_links_target_owned_by_other_train_s1": fp_owned.height,
            "fp_links_owner_inside_194k_sample": fp_owner_in_sample,
            "oracle_drop_all_owned_fp": m_xown_oracle, "oracle_delta": m_xown_oracle - macro,
            "note": "owners restricted to folds 0-3 (fold4 labels not read); in test all S1 compete for the same pool",
        },
        "country_slices": slices,
        "outside_top12_by_country_source": out_tab.to_dicts(),
        "outside_top12_miss_rate": out_rate.to_dicts(),
        "cited_not_recomputed": {"EXP-032_full_candidate_link_recall": 0.9660719915, "full_candidate_oracle_20k": "~0.9885"},
    }
    (OUT / "decomposition.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("country_slices", "outside_top12_by_country_source", "outside_top12_miss_rate")}, indent=1))
    print(json.dumps(slices, indent=1))
    print(out_rate)


if __name__ == "__main__":
    main()
