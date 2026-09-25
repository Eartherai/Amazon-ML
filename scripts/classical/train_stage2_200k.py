"""CL-003: stage-2 trained on ~194k held-fold OOF S1 (P5-STAGE2-OOF-001), evaluated on the untouched EXP-044 6k.

Owner safety: the 6k evaluation S1 are removed from training, and every training
pair whose target is owned (ground truth) by an evaluation S1 is dropped.
Threshold: grouped 3-fold OOF within the training S1 only. Saves the model and
the 6k evaluation-set probabilities (OOF by construction) for stacking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import os  # noqa: E402
import importlib  # noqa: E402
FEATV = os.environ.get("FEATV", "v1")
F = importlib.import_module("stage2_features_v2" if FEATV == "v2" else "stage2_features")
import pilot_stage2_exp044 as P  # noqa: E402
P.F = F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/CL-003"
PARAMS = dict(objective="binary", n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=9)
G: dict = {}


def chunk_features(items):
    groups = dict(items)
    s1r = {q: F.Record(*G["s1"][q]) for q in groups}
    tr = {t: F.Record(*G["t"][t]) for g in groups.values() for t, _ in g}
    return F.build_matrix(groups, s1r, tr, *G["idf"])


def featurize(groups: dict, procs: int = 9):
    items = list(groups.items())
    chunks = [items[i::procs * 8] for i in range(procs * 8)]
    keys, mats = [], []
    with mp.get_context("fork").Pool(procs, maxtasksperchild=4) as pool:
        for k, X in pool.imap(chunk_features, chunks):
            keys += k
            mats.append(X)
    return keys, np.vstack(mats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["sidecar", "top12"], default="sidecar")
    args = ap.parse_args()
    t0 = time.time()
    top = pl.concat([pl.read_parquet(OUT / f"s3/results/top12-fold{k}.parquet") for k in (1, 2, 3)])
    ev_truth = json.loads((P.EXP044 / "truth.json").read_text())
    ev_pairs = [json.loads(l) for l in (P.EXP044 / "pairs.jsonl").open()]
    ev_ids = set(ev_truth)
    owned_by_eval = {t for ts in ev_truth.values() for t in ts}
    tr = top.filter(~pl.col("source1_entity_id").is_in(list(ev_ids)) & ~pl.col("target_id").is_in(list(owned_by_eval)))
    if args.scope == "sidecar":
        tr = tr.filter((pl.col("base_score") >= 0.4) | (pl.col("rank") <= 2))
    groups = defaultdict(list)
    labels = {}
    for q, t, lab, b in tr.select("source1_entity_id", "target_id", "label", "base_score").iter_rows():
        groups[q].append((t, float(b)))
        labels[(q, t)] = int(lab)
    evg = defaultdict(list)
    for p in ev_pairs:
        evg[p["q"]].append((p["t"], p["base_score"]))
    if args.scope == "sidecar":
        for s, g in evg.items():
            g.sort(key=lambda x: -x[1])
            evg[s] = [x for i, x in enumerate(g) if x[1] >= 0.4 or i < 2]
    s1_ids = sorted(set(groups) | set(evg))
    t_ids = sorted({t for g in groups.values() for t, _ in g} | {t for g in evg.values() for t, _ in g})
    G["s1"], G["t"] = P.load_texts(s1_ids, t_ids, raw=True)
    G["idf"] = P.s1_idf(ROOT / "outputs/experiments/CL-001/train_s1_idf.json", P.TRAIN / "train_source1.tsv")
    print(json.dumps({"train_s1": len(groups), "train_pairs": len(labels), "load_s": round(time.time() - t0, 1)}), flush=True)
    cache = OUT / f"trainmat-{args.scope}-{FEATV}.npz"
    if cache.exists():
        zc = np.load(cache, allow_pickle=True)
        keys, X, ekeys, EX = [tuple(k) for k in zc["keys"]], zc["X"], [tuple(k) for k in zc["ekeys"]], zc["EX"]
    else:
        keys, X = featurize(groups)
        ekeys, EX = chunk_features(list(evg.items()))
        np.savez(cache, keys=np.array(keys), X=X, ekeys=np.array(ekeys), EX=EX)
    y = np.array([labels[k] for k in keys])
    print(json.dumps({"features_s": round(time.time() - t0, 1), "pos_rate": float(y.mean())}), flush=True)
    # threshold via grouped 3-fold OOF inside training S1 (sample 60k S1 for speed of the F0.5 sweep)
    part = {s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 3 for s in groups}
    kp = np.array([part[k[0]] for k in keys])
    oof = np.zeros(len(keys))
    for k in range(3):
        m = lgb.LGBMClassifier(**PARAMS).fit(X[kp != k], y[kp != k])
        oof[kp == k] = m.predict_proba(X[kp == k])[:, 1]
    # training truth restricted to retained pairs is incomplete (retrieval misses); threshold uses pair labels in scope
    tr_truth = defaultdict(set)
    full_truth = {}
    gt = pl.read_csv(P.TRAIN / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    want = set(groups)
    for q, raw in gt.filter(pl.col("source1_entity_id").is_in(list(want))).iter_rows():
        full_truth[q] = raw.split(",") if raw else []
    sample = sorted(want, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:60000]
    samp = set(sample)
    sk = [(k, p) for k, p in zip(keys, oof) if k[0] in samp]
    thr, tr_macro = P.choose_threshold([k for k, _ in sk], np.array([p for _, p in sk]), full_truth, sample)
    model = lgb.LGBMClassifier(**PARAMS).fit(X, y)
    path = OUT / f"stage2-200k-{args.scope}{'' if FEATV == 'v1' else '-' + FEATV}.txt"
    model.booster_.save_model(str(path))
    eprob = model.predict_proba(EX)[:, 1]
    country = {p["q"]: p["country"] for p in ev_pairs}
    preds = defaultdict(set)
    for (s, t), pr in zip(ekeys, eprob):
        if pr >= thr:
            preds[s].add(t)
    evres, per = P.evaluate(sorted(ev_truth), preds, ev_truth, country)
    # best-threshold diagnostic on eval (not used for selection)
    best_thr, best_macro = P.choose_threshold(ekeys, eprob, ev_truth, sorted(ev_truth))
    np.savez_compressed(OUT / f"eval6k-{args.scope}{'' if FEATV == 'v1' else '-' + FEATV}.npz", q=np.array([k[0] for k in ekeys]), t=np.array([k[1] for k in ekeys]), p=eprob, X=EX)
    meta = {"scope": args.scope, "threshold": thr, "train_inner_macro_60k": tr_macro, "eval_6k": evres,
            "eval_oracle_threshold_diag": [best_thr, best_macro], "train_s1": len(groups), "train_pairs": len(keys),
            "features": F.NAMES, "feature_version": FEATV, "params": PARAMS, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seconds": time.time() - t0, "fold4": "CLOSED",
            "leakage": "6k eval S1 removed; training pairs with eval-owned targets removed; base scores held-fold OOF"}
    (OUT / f"stage2-200k-{args.scope}{'' if FEATV == 'v1' else '-' + FEATV}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("scope", "threshold", "train_inner_macro_60k", "eval_6k", "eval_oracle_threshold_diag", "train_pairs", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
