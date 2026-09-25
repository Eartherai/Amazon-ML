"""Ownership variants on the 194k training OOF (CL-003 top-12 v2 stage-2 grouped 3-fold OOF probabilities).

Exact macro F0.5 per S1 (empty/empty = 1) against train_ground_truth.tsv restricted to the 194k S1.
Paired entity bootstrap (1000 resamples, same resamples for every variant).
Stress test: 200k store (194k OOF + 6k eval stage-2 probabilities), cross-owner analysis with the
fold0-3 owner map only (fold4 labels are filtered out before any use), and a density-scaling curve.
Outputs: outputs/analysis/ownership_variants/train_results.json
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
CL3 = ROOT / "outputs/experiments/CL-003"
TRAIN = ROOT / "student_resource/dataset/train"
OUT = ROOT / "outputs/analysis/ownership_variants"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def q3(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    return {k: float(np.percentile(x, v)) for k, v in (("p10", 10), ("p50", 50), ("p90", 90))}


def main():
    thr = json.loads((CL3 / "stage2-200k-top12-v2.json").read_text())["threshold"]
    z = np.load(CL3 / "trainmat-top12-v2.npz", allow_pickle=True)
    keys = z["keys"]
    oof = np.load(CL3 / "train-oof-top12-v2.npy")
    assert len(keys) == len(oof)
    pairs = pl.DataFrame({"q": keys[:, 0].astype(str), "t": keys[:, 1].astype(str), "p": oof})
    del keys, z
    log("pairs", pairs.shape)

    folds = pl.read_parquet(ROOT / "artifacts/validation/v1/validation_folds.parquet").select(
        pl.col("source1_entity_id").alias("q"), "country", "fold", "n_matches")
    s1 = pairs.select("q").unique().sort("q").join(folds, on="q", how="left")
    assert s1["fold"].null_count() == 0 and s1.filter(pl.col("fold") == 4).height == 0
    log("s1", s1.height, s1["fold"].value_counts().sort("fold").rows())
    open_q = folds.filter(pl.col("fold") != 4).select("q")  # fold4 CLOSED: never read its truth

    gt = (pl.scan_csv(TRAIN / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
          .rename({"source1_entity_id": "q"}).join(open_q.lazy(), on="q", how="inner")
          .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("t"))
          .select("q", "t").explode("t").filter(pl.col("t") != "").collect())
    assert gt["t"].n_unique() == gt.height, "a fold0-3 target has more than one owner"
    log("fold0-3 owner links", gt.height)
    truth = gt.join(s1.select("q"), on="q", how="inner")
    owner = gt.rename({"q": "owner"})
    del gt
    country = s1["country"].to_numpy()

    claims = pairs.filter(pl.col("p") >= thr)
    del pairs
    log("claims A", claims.height)
    groups = L.dup_groups(str(TRAIN / "train_source2.tsv"), str(TRAIN / "train_source3.tsv"), mem_gb=2)
    log("train dup-group rows", groups.height, "groups", groups["gid"].n_unique())
    ann_t = L.annotate(claims)
    ann_g = L.annotate(claims, groups)

    # ---------- variants ----------
    res, F = {}, {}
    kept_sets = {}
    for name, rule, m, _ in L.VARIANTS_ALL:
        kept = L.apply_variant(ann_t, ann_g, rule, m)
        kept_sets[name] = kept
        f, tpa, npa, nta = L.per_s1_scores(kept, truth, s1)
        F[name] = f
        res[name] = L.summarize(f, tpa, npa, nta, country) | {"links_removed": claims.height - kept.height, "links_added": 0}
    for base in ("B", "E_dupunit", "F_dropall0.02"):
        kept = L.dup_expand(kept_sets[base], groups)
        name = base + "+dupexpand"
        f, tpa, npa, nta = L.per_s1_scores(kept, truth, s1)
        F[name] = f
        res[name] = L.summarize(f, tpa, npa, nta, country) | {
            "links_removed": claims.height - kept_sets[base].height, "links_added": kept.height - kept_sets[base].height}
    # honest cross-fitted drop-all margin: contested targets split by hash(t) % 2; the margin chosen on one
    # half (others under B) is applied to the other half.
    grid = [0.01, 0.02, 0.05, 0.10, 0.20, 2.0]
    half = (pl.col("t").hash(11) % 2)
    b_keep = L.keep_expr("B")

    def split_rule(m0, m1):
        e = pl.when(half == 0).then(L.keep_expr("F", m0) if m0 is not None else b_keep).otherwise(
            L.keep_expr("F", m1) if m1 is not None else b_keep)
        return ann_t.filter(e).select("q", "t", "p")

    sel = {}
    for h in (0, 1):
        scores = {}
        for m in grid:
            kept = split_rule(m, None) if h == 0 else split_rule(None, m)
            scores[m] = float(L.per_s1_scores(kept, truth, s1)[0].mean())
        sel[h] = max(grid, key=lambda m: (scores[m], -m))
        log("crossfit half", h, "scores", scores, "chosen", sel[h])
    kept = split_rule(sel[1], sel[0])
    f, tpa, npa, nta = L.per_s1_scores(kept, truth, s1)
    F["Fcv_crossfit_margin"] = f
    res["Fcv_crossfit_margin"] = L.summarize(f, tpa, npa, nta, country) | {
        "links_removed": claims.height - kept.height, "links_added": 0,
        "chosen_margin_half0_used_on_half1": sel[0], "chosen_margin_half1_used_on_half0": sel[1], "grid": grid}
    names = list(F)
    D = np.stack([F[n] - F["A"] for n in names], axis=1)
    DB = np.stack([F[n] - F["B"] for n in names], axis=1)
    bo = L.paired_bootstrap(D, 1000, 0)
    bb = L.paired_bootstrap(DB, 1000, 1)
    for j, n in enumerate(names):
        res[n]["delta_vs_A"] = float(D[:, j].mean())
        res[n]["ci95_vs_A"] = [float(np.percentile(bo[:, j], 2.5)), float(np.percentile(bo[:, j], 97.5))]
        res[n]["delta_vs_B"] = float(DB[:, j].mean())
        res[n]["ci95_vs_B"] = [float(np.percentile(bb[:, j], 2.5)), float(np.percentile(bb[:, j], 97.5))]
        res[n]["s1_changed_vs_A"] = int((D[:, j] != 0).sum())
        res[n]["s1_improved_vs_A"] = int((D[:, j] > 0).sum())
        res[n]["s1_worsened_vs_A"] = int((D[:, j] < 0).sum())
    log("variants done")
    for n in names:
        log(f"{n:22s} macro={res[n]['macro_f0_5']:.9f} d={res[n]['delta_vs_A']:+.7f} "
            f"CI=[{res[n]['ci95_vs_A'][0]:+.7f},{res[n]['ci95_vs_A'][1]:+.7f}] rm={res[n]['links_removed']} add={res[n]['links_added']}")

    # ---------- contest diagnostics (194k claims, unit = target) ----------
    lab = ann_t.join(truth.with_columns(pl.lit(1).alias("y")), on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    con = lab.filter(pl.col("n_s1") >= 2)
    per_t = con.group_by("t").agg(
        pl.col("n_s1").first(), pl.col("best_p").first(), pl.col("second_p").first(),
        (pl.col("y") * (pl.col("q") == pl.col("best_q"))).max().alias("best_true"),
        (pl.col("y") * (pl.col("q") != pl.col("best_q"))).max().alias("other_true"),
    ).with_columns((pl.col("best_p") - pl.col("second_p")).alias("margin"))
    bins = [0.0, 0.02, 0.05, 0.10, 0.20, 1.01]
    by_bin = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = per_t.filter((pl.col("margin") >= lo) & (pl.col("margin") < hi))
        by_bin.append({"margin": [lo, min(hi, 1.0)], "targets": b.height, "best_true": int(b["best_true"].sum()),
                       "other_true": int(b["other_true"].sum()),
                       "none_true": int(((b["best_true"] == 0) & (b["other_true"] == 0)).sum())})
    removed_B = con.filter(pl.col("q") != pl.col("best_q"))
    contest = {
        "contested_targets": per_t.height, "claims_on_contested": con.height,
        "claims_per_contested_target": {str(k): v for k, v in per_t["n_s1"].value_counts().sort("n_s1").rows()},
        "margin_quantiles": q3(per_t["margin"].to_numpy()), "margin_eq0": int((per_t["margin"] == 0).sum()),
        "margin_lt0.02": int((per_t["margin"] < 0.02).sum()),
        "best_true": int(per_t["best_true"].sum()), "other_true": int(per_t["other_true"].sum()),
        "none_true": int(((per_t["best_true"] == 0) & (per_t["other_true"] == 0)).sum()),
        "by_margin": by_bin,
        "B_removed_claims": removed_B.height, "B_removed_true": int(removed_B["y"].sum()),
        "B_removed_p_ge_0.9": int((removed_B["p"] >= 0.9).sum()),
        "B_removed_p_ge_0.9_true": int(removed_B.filter(pl.col("p") >= 0.9)["y"].sum()),
    }
    log("contest", json.dumps({k: v for k, v in contest.items() if k != "by_margin"}))

    # ---------- S1 raw duplicates ----------
    s1raw = (pl.scan_csv(TRAIN / "train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
             .select(pl.col("entity_id").alias("q"),
                     pl.concat_str([pl.col(c).fill_null("\x00") for c in ("business_name", "business_address", "country")],
                                   separator="\x1f").hash(7).alias("h")).collect())
    s1raw = s1raw.join(folds.filter(pl.col("fold") != 4).select("q", "n_matches"), on="q", how="inner")
    g = s1raw.group_by("h").agg(pl.len().alias("size"), (pl.col("n_matches") > 0).sum().alias("nonempty"))
    g = g.filter(pl.col("size") > 1)
    s1dup = {"fold0_3_s1": s1raw.height, "dup_groups": g.height, "s1_in_dup_groups": int(g["size"].sum()),
             "groups_by_nonempty_members": {str(k): v for k, v in g["nonempty"].value_counts().sort("nonempty").rows()},
             "groups_by_size": {str(k): v for k, v in g["size"].value_counts().sort("size").rows()[:10]}}
    hh = s1raw.select("q", "h")
    losers = (con.filter(pl.col("q") != pl.col("best_q")).select("t", "q", "best_q", "p", "best_p")
              .join(hh, on="q", how="left").join(hh.rename({"q": "best_q", "h": "hb"}), on="best_q", how="left")
              .with_columns((pl.col("h") == pl.col("hb")).fill_null(False).alias("ident")))
    pt = losers.group_by("t").agg(pl.col("ident").any(), (pl.col("best_p") - pl.col("p")).min().alias("mg"))
    s1dup["contested_targets_with_raw_identical_S1_claimants"] = int(pt["ident"].sum())
    s1dup["contested_targets_margin0"] = int((pt["mg"] == 0).sum())
    s1dup["contested_targets_margin0_raw_identical"] = int((pt["ident"] & (pt["mg"] == 0)).sum())
    del s1raw, hh, losers, pt
    log("s1dup", json.dumps(s1dup))

    # ---------- stress test ----------
    e = np.load(CL3 / "eval6k-top12-v2.npz", allow_pickle=True)
    ev = pl.DataFrame({"q": e["q"].astype(str), "t": e["t"].astype(str), "p": e["p"].astype(np.float64)})
    ev_s1 = ev.select("q").unique()
    evf = ev_s1.join(folds, on="q", how="left")
    assert evf.filter(pl.col("fold") == 4).height == 0 and ev_s1.join(s1.select("q"), on="q", how="inner").height == 0
    store_claims = pl.concat([claims, ev.filter(pl.col("p") >= thr)])
    store_q = pl.concat([s1.select("q"), ev_s1])
    sc = store_claims.group_by("t").agg(pl.col("q").n_unique().alias("n"))
    st = {"store_s1": store_q.height, "store_claims": store_claims.height,
          "contested_targets_194k": per_t.height,
          "contested_targets_200k_store": int((sc["n"] >= 2).sum()),
          "claims_on_contested_200k_store": int(sc.filter(pl.col("n") >= 2)["n"].sum()),
          "train_s1_total": int(folds.height), "density_store": store_q.height / folds.height}
    # cross-owner analysis on the 194k A claims
    x = claims.join(owner, on="t", how="left")
    x = x.with_columns(
        pl.when(pl.col("owner") == pl.col("q")).then(pl.lit("true"))
        .when(pl.col("owner").is_null()).then(pl.lit("fp_no_fold0_3_owner"))
        .when(pl.col("owner").is_in(store_q["q"].to_list())).then(pl.lit("fp_owner_in_store"))
        .otherwise(pl.lit("fp_owner_fold0_3_outside_store")).alias("cat"))
    st["claim_categories"] = dict(x["cat"].value_counts().sort("cat").rows())
    ins = x.filter(pl.col("cat") == "fp_owner_in_store").join(
        store_claims.rename({"q": "owner", "p": "owner_p"}), on=["owner", "t"], how="left")
    st["fp_owner_in_store_owner_claims"] = int(ins["owner_p"].is_not_null().sum())
    st["fp_owner_in_store_owner_higher_p"] = int((ins["owner_p"] > ins["p"]).sum())
    rate = st["fp_owner_in_store_owner_higher_p"] / max(ins.height, 1)
    st["fp_fixable_rate_given_owner_present"] = rate
    st["fp_owner_outside_store_est_fixable_at_full_density"] = rate * st["claim_categories"].get("fp_owner_fold0_3_outside_store", 0)
    st["fp_total_A"] = int((x["cat"] != "true").sum())
    # true claims that meet a higher-p competitor in the store (would be stolen under B)
    tr = x.filter(pl.col("cat") == "true").join(store_claims.group_by("t").agg(pl.col("p").max().alias("mx"), pl.len().alias("k")), on="t")
    st["true_claims_with_store_competitor"] = int((tr["k"] >= 2).sum())
    st["true_claims_outscored_in_store"] = int((tr["mx"] > tr["p"]).sum())
    del x, ins, tr
    log("stress", json.dumps(st))

    # density-scaling curve: sub-sample S1 of the 194k, measure per-S1 delta vs A
    rng = np.random.default_rng(20260925)
    s1_list = s1["q"].to_numpy()
    fa = F["A"]
    curve = []
    for frac in (0.125, 0.25, 0.5, 1.0):
        for seed in range(5 if frac < 1 else 1):
            if frac < 1:
                sel = rng.random(len(s1_list)) < frac
            else:
                sel = np.ones(len(s1_list), bool)
            ssub = s1.filter(pl.Series(sel))
            csub = claims.join(ssub.select("q"), on="q", how="inner")
            at, ag = L.annotate(csub), L.annotate(csub, groups)
            row = {"frac": frac, "seed": seed, "s1": ssub.height, "density": ssub.height / folds.height,
                   "claims": csub.height, "contested_targets": int(at.filter(pl.col("n_s1") >= 2)["t"].n_unique())}
            tsub = truth.join(ssub.select("q"), on="q", how="inner")
            base = L.per_s1_scores(csub, tsub, ssub)[0]
            for name, rule, m, _ in L.VARIANTS_ALL[1:]:
                kept = L.apply_variant(at, ag, rule, m)
                row[name] = {"delta": float((L.per_s1_scores(kept, tsub, ssub)[0] - base).mean()),
                             "removed_frac": (csub.height - kept.height) / csub.height}
            curve.append(row)
            log("curve", frac, seed, row["contested_targets"], row["B"])
    fit = {}
    dens = np.array([r["density"] for r in curve])
    for name, *_ in L.VARIANTS_ALL[1:]:
        dl = np.array([r[name]["delta"] for r in curve])
        rf = np.array([r[name]["removed_frac"] for r in curve])
        slope = float((dens * dl).sum() / (dens * dens).sum())
        rslope = float((dens * rf).sum() / (dens * dens).sum())
        fit[name] = {"delta_per_unit_density": slope, "extrapolated_delta_full_density": slope,
                     "removed_frac_per_unit_density": rslope, "extrapolated_removed_frac_full_density": rslope}
    st["density_curve"] = curve
    st["density_linear_fit_through_origin"] = fit

    out = {"threshold": thr, "s1": s1.height, "claims_A": claims.height, "variants": res, "contest": contest,
           "s1_duplicates": s1dup, "stress": st, "seconds": time.time() - T0, "fold4": "CLOSED",
           "note": "OOF probs: CL-003 train-oof-top12-v2.npy (grouped 3-fold within 194k training S1); truth fold0-3 only"}
    (OUT / "train_results.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT / "train_per_s1_f05.npz", q=s1["q"].to_numpy().astype("U12"), **{n.replace("+", "_plus_").replace(".", "p"): F[n] for n in names})
    log("wrote", OUT / "train_results.json")


if __name__ == "__main__":
    main()
