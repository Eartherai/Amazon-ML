"""Train-side (194k held S1) population summary of CL-003 top-12 v2 stage-2 OOF probabilities, per country.

Inputs (read-only): outputs/experiments/CL-003/trainmat-top12-v2.npz (keys, X), train-oof-top12-v2.npy,
student_resource/dataset/train/{train_source1.tsv, train_ground_truth.tsv}.
Output: outputs/analysis/public_gap/train_oof_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/classical"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_features_v2 import NAMES  # noqa: E402
from common import FEATS, summarize  # noqa: E402

TRAIN = ROOT / "student_resource/dataset/train"
OUT = ROOT / "outputs/analysis/public_gap"


def main():
    t0 = time.time()
    z = np.load(ROOT / "outputs/experiments/CL-003/trainmat-top12-v2.npz", allow_pickle=False)
    keys = z["keys"]
    X = z["X"]
    idx = [NAMES.index(f) for f in FEATS]
    Xs = X[:, idx].copy()
    del X
    oof = np.load(ROOT / "outputs/experiments/CL-003/train-oof-top12-v2.npy")
    assert len(oof) == len(keys) == len(Xs)
    df = pl.DataFrame({"q": keys[:, 0], "t": keys[:, 1]}).with_row_index("row")
    del keys
    s1 = pl.read_csv(TRAIN / "train_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0,
                     columns=["entity_id", "country"])
    pop_country = dict(s1.group_by("country").len().iter_rows())
    sample = df.select("q").unique()
    s1 = s1.join(sample, left_on="entity_id", right_on="q", how="inner")
    gt = pl.read_csv(TRAIN / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    gt = gt.join(s1.select("entity_id"), left_on="source1_entity_id", right_on="entity_id", how="inner")
    gt = gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(
        pl.element().filter(pl.element() != "")).alias("ids"))
    ntrue = dict(zip(gt["source1_entity_id"].to_list(), gt["ids"].list.len().to_list()))
    pos = gt.select(pl.col("source1_entity_id").alias("q"), pl.col("ids").alias("t")).explode("t").drop_nulls()
    pos = pos.with_columns(pl.lit(True).alias("label"))
    df = df.join(pos, on=["q", "t"], how="left").sort("row").with_columns(pl.col("label").fill_null(False))
    assert len(df) == len(oof) and (df["row"].to_numpy() == np.arange(len(oof))).all()
    res = {"source": {"trainmat": "outputs/experiments/CL-003/trainmat-top12-v2.npz",
                      "oof": "outputs/experiments/CL-003/train-oof-top12-v2.npy", "threshold": 0.67,
                      "note": "stage-2 OOF grouped 3-fold within 194k S1; base scores are first-stage held-fold OOF"},
           "population_train_s1_by_country": pop_country, "countries": {}}
    countries = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
    for c in sorted(set(countries.values())) + ["All"]:
        ids = sorted(q for q, cc in countries.items() if c == "All" or cc == c)
        m = pl.DataFrame({"q": ids, "s1i": np.arange(len(ids), dtype=np.int64)})
        sub = df.join(m, on="q", how="inner").sort("row")
        rows = sub["row"].to_numpy()
        tcode = np.unique(sub["t"].to_numpy(), return_inverse=True)[1]
        F = {f: Xs[rows, j] for j, f in enumerate(FEATS)}
        nt = np.array([ntrue.get(q, 0) for q in ids])
        r = summarize(sub["s1i"].to_numpy(), len(ids), oof[rows], F, tcode,
                      label=sub["label"].to_numpy(), ntrue_full=nt)
        if c != "All":
            r["sample_fraction_of_train_country_s1"] = len(ids) / pop_country[c]
        res["countries"][c] = r
        print(c, json.dumps({k: r[k] for k in ("s1", "pred_singleton_pct", "links_mean", "actual_macro_full_truth",
                                             "self_est_macro_top12", "actual_macro_top12_truth")}), flush=True)
    res["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_oof_summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
