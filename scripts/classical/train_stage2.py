"""CL-002: fit production stage-2 LightGBM on all 6k EXP-044 held S1 (sidecar pairs).

Threshold = argmax of pooled grouped 5-fold OOF exact macro F0.5 over the 6k S1.
Variant `classical` uses only stage2_features; variant `neural` adds the saved
EXP-045 logit (existing artifact, no neural training here). Fold4 CLOSED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_features as F  # noqa: E402
import pilot_stage2_exp044 as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["classical", "neural"], required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/experiments/CL-002/models")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(line) for line in (P.EXP044 / "pairs.jsonl").open()]
    truth = json.loads((P.EXP044 / "truth.json").read_text())
    country = {p["q"]: p["country"] for p in pairs}
    s1_records, t_records = P.load_texts(sorted({p["q"] for p in pairs}), sorted({p["t"] for p in pairs}))
    name_idf, nd, addr_idf, ad = P.s1_idf(ROOT / "outputs/experiments/CL-001/train_s1_idf.json", P.TRAIN / "train_source1.tsv")
    groups = defaultdict(list)
    for p in pairs:
        groups[p["q"]].append((p["t"], p["base_score"]))
    for s, g in groups.items():
        g.sort(key=lambda x: -x[1])
        groups[s] = [x for i, x in enumerate(g) if x[1] >= 0.4 or i < 2]
    keys, X = F.build_matrix(groups, s1_records, t_records, name_idf, nd, addr_idf, ad)
    names = list(F.NAMES)
    if args.variant == "neural":
        nl = {}
        for line in (P.EXP044.parent / "EXP-045/neural_scores.jsonl").open():
            r = json.loads(line)
            nl[(r["q"], r["t"])] = r["score"]
        X = np.hstack([X, np.array([[nl[k]] for k in keys], dtype=np.float32)])
        names.append("neural_logit_exp045")
    y = np.array([1 if t in set(truth[s]) else 0 for s, t in keys])
    s1s = sorted(truth)
    part = {s: i % 5 for i, s in enumerate(sorted(s1s, key=lambda s: hashlib.sha256(s.encode()).hexdigest()))}
    kpart = np.array([part[s] for s, _ in keys])
    oof = np.zeros(len(keys))
    for k in range(5):
        m = lgb.LGBMClassifier(**P.PARAMS).fit(X[kpart != k], y[kpart != k])
        oof[kpart == k] = m.predict_proba(X[kpart == k])[:, 1]
    thr, oof_macro = P.choose_threshold(keys, oof, truth, s1s)
    preds = defaultdict(set)
    for (s, t), pr in zip(keys, oof):
        if pr >= thr:
            preds[s].add(t)
    ev, _ = P.evaluate(s1s, preds, truth, country)
    model = lgb.LGBMClassifier(**P.PARAMS).fit(X, y)
    path = args.out / f"stage2-{args.variant}.txt"
    model.booster_.save_model(str(path))
    meta = {"variant": args.variant, "features": names, "threshold": thr, "oof_macro_thresholded_same_oof": oof_macro,
            "oof_eval": ev, "train_pairs": len(keys), "train_s1": len(s1s), "positives": int(y.sum()),
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "params": P.PARAMS,
            "sidecar_rule": "base>=0.4 or top2 per S1", "fold4": "CLOSED",
            "source": "EXP-044 fixed 6k folds2/3 held OOF pairs; train IDF from train_source1"}
    (args.out / f"stage2-{args.variant}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("variant", "threshold", "oof_macro_thresholded_same_oof", "oof_eval", "train_pairs", "model_sha256")}, indent=1))


if __name__ == "__main__":
    main()
