"""CL-001: classical stage-2 pilot on the fixed EXP-044 6k held set (folds 2/3; Fold4 CLOSED).

Inputs are the saved EXP-044 top-12 OOF pairs (first-stage 51-feature LightGBM
held-fold probabilities) and full truth. Stage-2 is cross-fitted: train on one
fold, predict the other; the entity threshold is chosen by grouped inner CV on
the training fold only. Reports exact per-S1 macro F0.5 (empty/empty = 1).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_features as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP044 = Path("/Users/earther/.codex/worktrees/aml-neural-warroom/Amazon ML Challange/outputs/experiments/warroom_neural/EXP-044")
TRAIN = ROOT / "student_resource/dataset/train"
PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=8)


def f05(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def evaluate(s1s, preds, truth, country):
    per = {s: f05(preds.get(s, set()), set(truth[s])) for s in s1s}
    tp = sum(len(preds.get(s, set()) & set(truth[s])) for s in s1s)
    npred = sum(len(preds.get(s, set())) for s in s1s)
    ntrue = sum(len(truth[s]) for s in s1s)
    single = [per[s] for s in s1s if not truth[s]]
    out = {"macro_f0_5": float(np.mean(list(per.values()))), "precision": tp / max(npred, 1), "recall": tp / max(ntrue, 1),
           "singleton_f0_5": float(np.mean(single)) if single else None, "n": len(s1s)}
    for c in ("India", "US"):
        vals = [per[s] for s in s1s if country[s] == c]
        out[c] = float(np.mean(vals)) if vals else None
    return out, per


def load_texts(ids_s1, ids_t, raw=False):
    con = duckdb.connect()
    con.execute("CREATE TABLE want_s1(id VARCHAR)")
    con.executemany("INSERT INTO want_s1 VALUES (?)", [(x,) for x in ids_s1])
    con.execute("CREATE TABLE want_t(id VARCHAR)")
    con.executemany("INSERT INTO want_t VALUES (?)", [(x,) for x in ids_t])
    opts = "delim='\t',header=true,quote='',escape='',all_varchar=true"
    s1 = con.execute(f"SELECT entity_id, business_name, business_address FROM read_csv('{TRAIN}/train_source1.tsv',{opts}) JOIN want_s1 ON id=entity_id").fetchall()
    t = con.execute(f"""SELECT entity_id, business_name, business_address FROM (
        SELECT * FROM read_csv('{TRAIN}/train_source2.tsv',{opts}) UNION ALL SELECT * FROM read_csv('{TRAIN}/train_source3.tsv',{opts}))
        JOIN want_t ON id=entity_id""").fetchall()
    if raw:
        return {r[0]: (r[1] or "", r[2] or "") for r in s1}, {r[0]: (r[1] or "", r[2] or "") for r in t}
    return {r[0]: F.Record(r[1], r[2]) for r in s1}, {r[0]: F.Record(r[1], r[2]) for r in t}


def s1_idf(path: Path, source1_tsv: Path):
    """Token document frequencies over one split's Source 1 names/addresses (no labels)."""
    if path.exists():
        d = json.loads(path.read_text())
        return d["name"], d["name_default"], d["addr"], d["addr_default"]
    con = duckdb.connect()
    rows = con.execute(f"SELECT business_name, business_address FROM read_csv('{source1_tsv}',delim='\t',header=true,quote='',escape='',all_varchar=true)").fetchall()
    ndf, adf = defaultdict(int), defaultdict(int)
    for name, addr in rows:
        for tok in set(F.norm(name or "").split()):
            ndf[tok] += 1
        for tok in set(F.address_tokens(F.norm(addr or ""))):
            adf[tok] += 1
    n = len(rows)
    name = {k: math.log(n / (v + 1)) for k, v in ndf.items() if v >= 2}
    addr = {k: math.log(n / (v + 1)) for k, v in adf.items() if v >= 2}
    d = {"n": n, "name": name, "name_default": math.log(n / 2), "addr": addr, "addr_default": math.log(n / 2)}
    path.write_text(json.dumps(d))
    return name, d["name_default"], addr, d["addr_default"]


def choose_threshold(keys, prob, truth, s1s):
    grid = np.arange(0.30, 0.96, 0.01)
    best = (-1, 0.5)
    by_s1 = defaultdict(list)
    for (s, t), p in zip(keys, prob):
        by_s1[s].append((t, p))
    for thr in grid:
        score = np.mean([f05({t for t, p in by_s1.get(s, []) if p >= thr}, set(truth[s])) for s in s1s])
        if score > best[0]:
            best = (score, float(thr))
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/experiments/CL-001")
    ap.add_argument("--scope", choices=["sidecar", "top12"], default="sidecar")
    ap.add_argument("--drop", default="", help="comma-separated feature names to ablate")
    ap.add_argument("--neural", action="store_true", help="diagnostic: add saved EXP-045 logit as a feature (sidecar only)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs = [json.loads(line) for line in (EXP044 / "pairs.jsonl").open()]
    truth = json.loads((EXP044 / "truth.json").read_text())
    country = {p["q"]: p["country"] for p in pairs}
    fold = {p["q"]: p["fold"] for p in pairs}
    s1_records, t_records = load_texts(sorted({p["q"] for p in pairs}), sorted({p["t"] for p in pairs}))
    name_idf, nd, addr_idf, ad = s1_idf(args.out / "train_s1_idf.json", TRAIN / "train_source1.tsv")
    groups = defaultdict(list)
    for p in pairs:
        groups[p["q"]].append((p["t"], p["base_score"]))
    if args.scope == "sidecar":
        for s, g in groups.items():
            g.sort(key=lambda x: -x[1])
            groups[s] = [x for i, x in enumerate(g) if x[1] >= 0.4 or i < 2]
    keys, X = F.build_matrix(groups, s1_records, t_records, name_idf, nd, addr_idf, ad)
    names = list(F.NAMES)
    if args.neural:
        nl = {}
        for line in (EXP044.parent / "EXP-045/neural_scores.jsonl").open():
            r = json.loads(line)
            nl[(r["q"], r["t"])] = r["score"]
        missing = sum(1 for k in keys if k not in nl)
        if missing and args.scope == "sidecar":
            raise ValueError(f"{missing} sidecar pairs lack EXP-045 logits")
        # Outside the sidecar (top12 scope) the logit is structurally missing -> NaN, same rule as test.
        X = np.hstack([X, np.array([[nl.get(k, np.nan)] for k in keys], dtype=np.float32)])
        names.append("neural_logit_exp045")
    if args.drop:
        drop = set(args.drop.split(","))
        keep = [i for i, n in enumerate(names) if n not in drop]
        X, names = X[:, keep], [names[i] for i in keep]
    y = np.array([1 if t in set(truth[s]) else 0 for s, t in keys])
    kfold = np.array([fold[s] for s, _ in keys])
    feat_seconds = time.time() - t0
    base_thr = 0.8083481555445471
    all_s1 = sorted(truth)
    base_pred = defaultdict(set)
    for p in pairs:
        if p["base_score"] >= base_thr:
            base_pred[p["q"]].add(p["t"])
    base_eval, base_per = evaluate(all_s1, base_pred, truth, country)
    final_pred, report = defaultdict(set), {"scope": args.scope, "pairs": len(keys), "features": names, "directions": []}
    oof_prob = np.zeros(len(keys))
    for train_fold, test_fold in ((2, 3), (3, 2)):
        tr, te = np.where(kfold == train_fold)[0], np.where(kfold == test_fold)[0]
        tr_s1 = sorted({keys[i][0] for i in tr})
        # grouped inner 3-fold CV for threshold choice
        inner = {s: i % 3 for i, s in enumerate(tr_s1)}
        inner_prob = np.zeros(len(tr))
        for k in range(3):
            fit = np.array([inner[keys[i][0]] != k for i in tr])
            m = lgb.LGBMClassifier(**PARAMS).fit(X[tr[fit]], y[tr[fit]])
            inner_prob[~fit] = m.predict_proba(X[tr[~fit]])[:, 1]
        thr, inner_score = choose_threshold([keys[i] for i in tr], inner_prob, truth, tr_s1)
        model = lgb.LGBMClassifier(**PARAMS).fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        oof_prob[te] = prob
        te_s1 = sorted(s for s in truth if fold.get(s) == test_fold)
        preds = defaultdict(set)
        for i, pr in zip(te, prob):
            if pr >= thr:
                preds[keys[i][0]].add(keys[i][1])
        final_pred.update(preds)
        ev, _ = evaluate(te_s1, preds, truth, country)
        bev, _ = evaluate(te_s1, base_pred, truth, country)
        imp = sorted(zip(names, model.booster_.feature_importance("gain")), key=lambda x: -x[1])[:15]
        report["directions"].append({"train_fold": train_fold, "test_fold": test_fold, "threshold": thr,
                                     "inner_macro": inner_score, "stage2": ev, "base": bev,
                                     "delta": ev["macro_f0_5"] - bev["macro_f0_5"], "top_gain": [(n, round(float(g), 1)) for n, g in imp]})
    ev, per = evaluate(all_s1, final_pred, truth, country)
    deltas = np.array([per[s] - base_per[s] for s in all_s1])
    rng = np.random.default_rng(0)
    boots = [deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(2000)]
    report.update({"base": base_eval, "stage2": ev, "delta": ev["macro_f0_5"] - base_eval["macro_f0_5"],
                   "delta_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   "feature_seconds": feat_seconds, "total_seconds": time.time() - t0, "fold4": "CLOSED",
                   "reference": {"EXP045_mmarco_blend_same_6k": 0.9474834591, "base_same_6k": 0.9325591312}})
    tag = args.scope + ("-neural" if args.neural else "") + ("-drop-" + args.drop.replace(",", "+") if args.drop else "")
    (args.out / f"result-{tag}.json").write_text(json.dumps(report, indent=2))
    np.save(args.out / f"oof-{tag}.npy", oof_prob)
    print(json.dumps({k: report[k] for k in ("scope", "pairs", "base", "stage2", "delta", "delta_ci95", "total_seconds")}, indent=1))
    for d in report["directions"]:
        print(d["train_fold"], "->", d["test_fold"], "thr", d["threshold"], "delta", round(d["delta"], 5), d["top_gain"][:8])


if __name__ == "__main__":
    main()
