"""Shared population summary for the public-gap analysis (no labels needed for the core metrics).

summarize() takes one country's top-12 stage-2 rows: integer S1 codes into the full country S1 list
(so S1 with no candidate rows count as predicted singletons), stage-2 probabilities, selected
stage2_features_v2 columns, and integer target codes. Optional per-row labels and per-S1 true counts
add measured F0.5 figures (train only).
"""
from __future__ import annotations

import numpy as np

THR = 0.67
FEATS = ["base", "rank", "n_strong", "n_mid", "sib_dup", "q_nonlatin", "t_nonlatin", "t_s2",
         "q_addr_missing", "t_addr_missing", "core_jw", "addr_tset", "num_exact", "num_q_unmatched"]
ROOT_REL = "outputs/analysis/public_gap"


def _f05(tp, npred, ntrue):
    den = npred + 0.25 * ntrue
    out = np.where(den == 0, 1.0, 1.25 * tp / np.where(den == 0, 1.0, den))
    return out


def summarize(s1, n_s1, p, F, tcode, label=None, ntrue_full=None, mc_draws=16, seed=0):
    s1 = np.asarray(s1, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    acc = p >= THR
    rows = np.bincount(s1, minlength=n_s1)
    links = np.bincount(s1[acc], minlength=n_s1)
    has = rows > 0
    out = {"s1": int(n_s1), "rows": int(len(p)), "s1_with_rows_pct": 100 * has.mean(),
           "rows_per_s1_all": rows.mean(), "rows_per_s1_with_rows": rows[has].mean() if has.any() else 0.0}
    out["pred_singleton_pct"] = 100 * (links == 0).mean()
    out["links_mean"] = links.mean()
    for q in (50, 90, 95, 99):
        out[f"links_p{q}"] = float(np.percentile(links, q))
    out["links_max"] = int(links.max())
    for k in range(6):
        out[f"links_eq{k}_pct"] = 100 * (links == k).mean()
    out["links_ge6_pct"] = 100 * (links >= 6).mean()
    # top1/top2 per S1 (over S1 with rows)
    order = np.lexsort((-p, s1))
    ss, ps = s1[order], p[order]
    first = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    top1 = ps[first]
    nxt = first + 1
    ok = nxt < len(ss)
    ok[ok] &= ss[nxt[ok]] == ss[first[ok]]
    top2 = np.zeros_like(top1)
    top2[ok] = ps[nxt[ok]]
    out["top1_mean"], out["top2_mean"], out["margin_mean"] = top1.mean(), top2.mean(), (top1 - top2).mean()
    out["top1_ge_thr_pct_of_s1_with_rows"] = 100 * (top1 >= THR).mean()
    out["sum_p_per_s1_all"] = np.bincount(s1, weights=p, minlength=n_s1).mean()
    # ownership conflicts among accepted links
    tc = tcode[acc]
    _, inv, cnt = np.unique(tc, return_inverse=True, return_counts=True)
    out["accepted_links"] = int(acc.sum())
    out["conflict_link_share_pct"] = 100 * (cnt[inv] > 1).mean() if len(tc) else 0.0
    out["conflict_targets"] = int((cnt > 1).sum())
    out["conflict_extra_links"] = int((cnt[cnt > 1] - 1).sum())
    pa = p[acc]
    out["acc_hi_conf_p95_pct"] = 100 * (pa >= 0.95).mean()
    out["acc_low_conf_067_080_pct"] = 100 * (pa < 0.8).mean()
    out["acc_p_mean"] = pa.mean()
    fa = {k: v[acc] for k, v in F.items()}
    out["acc_core_jw_mean"] = fa["core_jw"].mean()
    out["acc_addr_tset_mean"] = fa["addr_tset"].mean()
    both_addr = (fa["q_addr_missing"] == 0) & (fa["t_addr_missing"] == 0)
    out["acc_addr_tset_mean_both_addr"] = fa["addr_tset"][both_addr].mean()
    both_num = fa["num_q_unmatched"] >= 0
    out["acc_both_num_pct"] = 100 * both_num.mean()
    conf = (fa["num_q_unmatched"] > 0) & (fa["num_exact"] == 0)
    out["acc_num_conflict_pct_of_both_num"] = 100 * conf[both_num].mean()
    out["acc_t_addr_missing_pct"] = 100 * fa["t_addr_missing"].mean()
    out["acc_q_addr_missing_pct"] = 100 * fa["q_addr_missing"].mean()
    out["acc_t_nonlatin_pct"] = 100 * fa["t_nonlatin"].mean()
    out["acc_q_nonlatin_pct"] = 100 * fa["q_nonlatin"].mean()
    out["acc_sib_dup_pct"] = 100 * fa["sib_dup"].mean()
    out["acc_base_mean"] = fa["base"].mean()
    out["acc_base_lt05_pct"] = 100 * (fa["base"] < 0.5).mean()
    out["acc_rank_mean"] = fa["rank"].mean()
    out["acc_t_s2_pct"] = 100 * fa["t_s2"].mean()
    # uncertain band
    band = (p >= 0.4) & (p < 0.9)
    out["band_04_09_pair_pct"] = 100 * band.mean()
    out["band_04_09_s1_pct"] = 100 * (np.bincount(s1[band], minlength=n_s1) > 0).mean()
    # all-row context
    out["all_base_mean"] = F["base"].mean()
    out["all_core_jw_mean"] = F["core_jw"].mean()
    out["all_addr_tset_mean"] = F["addr_tset"].mean()
    out["all_t_addr_missing_pct"] = 100 * F["t_addr_missing"].mean()
    out["all_t_nonlatin_pct"] = 100 * F["t_nonlatin"].mean()
    out["all_sib_dup_pct"] = 100 * F["sib_dup"].mean()
    r1 = F["rank"] == 1
    out["rank1_n_strong_mean"] = F["n_strong"][r1].mean()
    out["rank1_n_mid_mean"] = F["n_mid"][r1].mean()
    out["rank1_base_mean"] = F["base"][r1].mean()
    # self-estimated macro F0.5 on top-12 truth, assuming calibrated independent probabilities
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(mc_draws):
        y = rng.random(len(p)) < p
        tp = np.bincount(s1[acc & y], minlength=n_s1)
        ny = np.bincount(s1[y], minlength=n_s1)
        est.append(_f05(tp, links, ny).mean())
    out["self_est_macro_top12"] = float(np.mean(est))
    out["self_est_macro_top12_mc_sd"] = float(np.std(est))
    if label is not None:
        lab = np.asarray(label, dtype=bool)
        tp = np.bincount(s1[acc & lab], minlength=n_s1)
        n12 = np.bincount(s1[lab], minlength=n_s1)
        out["actual_macro_top12_truth"] = _f05(tp, links, n12).mean()
        out["actual_precision"] = tp.sum() / max(links.sum(), 1)
        out["calib_mean_p"] = p.mean()
        out["calib_pos_rate"] = lab.mean()
        out["acc_true_pos_pct"] = 100 * lab[acc].mean()
        if ntrue_full is not None:
            nt = np.asarray(ntrue_full)
            per = _f05(tp, links, nt)
            out["actual_macro_full_truth"] = per.mean()
            out["actual_recall_full"] = tp.sum() / max(nt.sum(), 1)
            out["true_links_mean"] = nt.mean()
            out["true_singleton_pct"] = 100 * (nt == 0).mean()
            out["true_link_top12_coverage_pct"] = 100 * n12.sum() / max(nt.sum(), 1)
            out["actual_singleton_f05"] = per[nt == 0].mean() if (nt == 0).any() else None
            for q in (50, 90, 95, 99):
                out[f"true_links_p{q}"] = float(np.percentile(nt, q))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}
