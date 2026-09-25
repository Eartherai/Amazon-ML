"""Explore entity-level F0.5 set decisions on strictly OOF 200k scores.

The fixed EXP-033 per-fold 100k LightGBM models score only their held fold.
Candidate scores >=0.3 suffice for all tested decision rules. Fold4 is closed.
Rule ranking on this OOF sample is exploratory; promotion needs separate checks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

BASE = {1: 0.83, 2: 0.79, 3: 0.83}
THRESHOLDS = (0.55, 0.65, 0.70, 0.75, 0.80, 0.85)
SALVAGE = (0.45, 0.55, 0.65, 0.75)
MARGINS = (0.01, 0.03, 0.05, 0.10)
BIASES = (-1.5, -0.75, 0.0, 0.75, 1.5)


def f05(truth: set[str], pred: set[str]) -> float:
    if not truth:
        return float(not pred)
    tp = len(truth & pred)
    if not tp:
        return 0.0
    return 1.25 * tp / (1.25 * tp + len(pred - truth) + .25 * len(truth - pred))


def expectation_prefix(rows: list[tuple[str, float]], bias: float) -> set[str]:
    """Approximate expected entity F0.5 with an explicit empty-set utility."""
    if not rows:
        return set()
    ordered = rows[:20]
    probs = []
    for _, raw in ordered:
        p = min(max(raw, 1e-7), 1 - 1e-7)
        logit = math.log(p / (1 - p)) + bias
        probs.append(1 / (1 + math.exp(-logit)))
    total = sum(probs)
    empty_utility = math.prod(1 - p for p in probs)
    best_utility, best_k = empty_utility, 0
    selected_sum = 0.0
    for k, p in enumerate(probs, 1):
        selected_sum += p
        fp = k - selected_sum
        fn = total - selected_sum
        utility = 1.25 * selected_sum / (1.25 * selected_sum + fp + .25 * fn)
        if (utility, -k) > (best_utility, -best_k):
            best_utility, best_k = utility, k
    return {tid for tid, _ in ordered[:best_k]}


def decide(rule: str, rows: list[tuple[str, float]], threshold: float) -> set[str]:
    base = [(tid, score) for tid, score in rows if score >= threshold]
    if rule == "base":
        return {tid for tid, _ in base}
    if rule.startswith("global:"):
        cutoff = float(rule.split(":")[1])
        return {tid for tid, score in rows if score >= cutoff}
    if rule.startswith("salvage:"):
        cutoff = float(rule.split(":")[1])
        return {tid for tid, _ in base} if base else ({rows[0][0]} if rows and rows[0][1] >= cutoff else set())
    if rule.startswith("two_source:"):
        cutoff = float(rule.split(":")[1])
        if base:
            return {tid for tid, _ in base}
        first = {}
        for tid, score in rows:
            first.setdefault(tid[:2], (tid, score))
        if len(first) == 2 and all(score >= cutoff for _, score in first.values()):
            return {tid for tid, _ in first.values()}
        return set()
    if rule.startswith("cap:"):
        cap = int(rule.split(":")[1])
        counts = defaultdict(int)
        result = set()
        for tid, _ in base:
            source = tid[:2]
            if counts[source] < cap:
                result.add(tid)
                counts[source] += 1
        return result
    if rule.startswith("margin:"):
        margin = float(rule.split(":")[1])
        maxima = {}
        for tid, score in rows:
            maxima[tid[:2]] = max(maxima.get(tid[:2], 0.), score)
        return {tid for tid, score in base if score >= maxima[tid[:2]] - margin}
    if rule.startswith("expected:"):
        return expectation_prefix(rows, float(rule.split(":")[1]))
    raise ValueError(rule)


def main() -> None:
    import lightgbm as lgb
    import numpy as np
    import polars as pl

    from src.models.sub001_features import NAMES

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    meta = pl.read_parquet(args.inputs / "labeled_queries.parquet", columns=["entity_id", "country", "fold"])
    if len(meta) != 200_000 or set(meta["fold"].to_list()) != {1, 2, 3}:
        raise ValueError("Wrong query population")
    truth = {qid: set() for qid in meta["entity_id"]}
    for qid, tid in pl.read_parquet(args.inputs / "truth.parquet").iter_rows():
        truth[qid].add(tid)
    country = dict(meta.select("entity_id", "country").iter_rows())
    fold = dict(meta.select("entity_id", "fold").iter_rows())
    feature_report = json.loads((args.features / "metrics.json").read_text())
    if feature_report["queries"] != 200_000 or feature_report["features"] != NAMES:
        raise ValueError("Wrong feature store")
    parts = sorted(args.features.glob("features-*.parquet"))
    if len(parts) != json.loads((args.features / "COMPLETE.json").read_text())["feature_parts"]:
        raise ValueError("Incomplete feature inventory")
    scan = pl.scan_parquet([str(path) for path in parts])
    scored: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for held in (1, 2, 3):
        frame = scan.filter(pl.col("fold") == held).select(["source1_entity_id", "target_id", *NAMES]).collect()
        model = lgb.Booster(model_file=str(args.models / f"model-size100000-held{held}.txt"))
        if model.feature_name() != NAMES:
            raise ValueError("Feature names differ")
        probabilities = model.predict(frame.select(NAMES).to_numpy(), num_threads=args.threads)
        qids = frame["source1_entity_id"].to_list()
        tids = frame["target_id"].to_list()
        indices = np.flatnonzero(probabilities >= 0.3)
        for index in indices:
            i = int(index)
            scored[qids[i]].append((tids[i], float(probabilities[i])))
        print(json.dumps({"held": held, "pairs": len(frame), "score_rows": len(indices),
                          "seconds": time.perf_counter() - started}), flush=True)
        del frame, model, probabilities, qids, tids, indices
    for rows in scored.values():
        rows.sort(key=lambda item: (-item[1], item[0]))

    rules = ["base"]
    rules += [f"global:{value}" for value in THRESHOLDS]
    rules += [f"salvage:{value}" for value in SALVAGE]
    rules += [f"two_source:{value}" for value in SALVAGE]
    rules += [f"cap:{value}" for value in (1, 2, 3)]
    rules += [f"margin:{value}" for value in MARGINS]
    rules += [f"expected:{value}" for value in BIASES]
    aggregates = {rule: {held: [0., 0] for held in (1, 2, 3)} for rule in rules}
    country_sum = {rule: defaultdict(lambda: [0., 0]) for rule in rules}
    for qid in truth:
        rows = scored.get(qid, [])
        held = fold[qid]
        for rule in rules:
            value = f05(truth[qid], decide(rule, rows, BASE[held]))
            aggregates[rule][held][0] += value
            aggregates[rule][held][1] += 1
            country_sum[rule][country[qid]][0] += value
            country_sum[rule][country[qid]][1] += 1
    report = {}
    for rule in rules:
        folds = aggregates[rule]
        total = sum(s for s, _ in folds.values())
        count = sum(n for _, n in folds.values())
        report[rule] = {"macro_f0_5": total / count,
                        "folds": {str(held): s / n for held, (s, n) in folds.items()},
                        "countries": {name: s / n for name, (s, n) in country_sum[rule].items()}}
    base = report["base"]["macro_f0_5"]
    for rule in rules:
        report[rule]["delta"] = report[rule]["macro_f0_5"] - base
    ranking = sorted(({"rule": rule, **result} for rule, result in report.items()),
                     key=lambda item: item["macro_f0_5"], reverse=True)
    output = {"scope": "Exploratory 200k strict outer OOF set decisions; fold4 CLOSED; not independently validated",
              "entities": len(truth), "base_thresholds": BASE, "base_macro_f0_5": base,
              "ranking": ranking, "seconds": time.perf_counter() - started}
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"base": base, "top10": ranking[:10], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
