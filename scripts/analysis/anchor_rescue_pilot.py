"""Measure target-to-target anchor rescue on disjoint, labeled OOF queries.

This exploratory script uses only organizer-provided records. It never writes a
submission and keeps Fold 4 closed. Candidate lists are the original scored
20k retrieval union; no new candidates are invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import polars as pl
from rapidfuzz.distance import JaroWinkler


ROOT = Path(__file__).resolve().parents[2]


def read_lists(path: Path) -> dict[str, set[str]]:
    with path.open(newline="") as stream:
        return {
            row["source1_entity_id"]: set(row["matched_entity_ids"].split(","))
            if row["matched_entity_ids"] else set()
            for row in csv.DictReader(stream, delimiter="\t")
        }


def f05(truth: set[str], prediction: set[str]) -> float:
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    if not tp:
        return 0.0
    fp = len(prediction - truth)
    fn = len(truth - prediction)
    return 1.25 * tp / (1.25 * tp + fp + 0.25 * fn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-country", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    q = pl.read_parquet(ROOT / "artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet", columns=["entity_id", "country"])
    by_country = defaultdict(list)
    for qid, country in q.iter_rows():
        by_country[country].append(qid)
    selected = set()
    for country in ("India", "US"):
        ordered = sorted(by_country[country], key=lambda x: hashlib.sha256(x.encode()).digest())
        selected.update(ordered[: args.per_country])
    predictions = read_lists(ROOT / "outputs/experiments/P4-NUMERIC-B-001/with_canonical_numeric/entity_predictions.tsv")
    truths = {}
    with (ROOT / "student_resource/dataset/train/train_ground_truth.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qid = row["source1_entity_id"]
            if qid in selected:
                truths[qid] = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
    candidates = pl.scan_parquet(str(ROOT / "outputs/oof/P4-B-001/features/part-*.parquet"))
    candidates = candidates.filter(pl.col("source1_entity_id").is_in(selected)).select(["source1_entity_id", "target_id", "label"]).collect()
    target_ids = set(candidates["target_id"].to_list())
    targets = pl.scan_parquet(ROOT / "artifacts/cloud/phase5/feature-200k-input-v001/targets.parquet")
    targets = targets.filter(pl.col("entity_id").is_in(target_ids)).select(["entity_id", "n", "a"]).collect()
    lookup = {target: (name, address) for target, name, address in targets.iter_rows()}
    if len(lookup) != len(target_ids):
        raise ValueError("Missing target metadata")
    by_query = defaultdict(list)
    for qid, target, label in candidates.iter_rows():
        by_query[qid].append((target, label))
    if set(by_query) != selected or set(truths) != selected:
        raise ValueError("Incomplete query sample")
    rules = {
        "same_name_addr80": lambda n, a: n == 1 and a >= 0.8,
        "same_name_addr70": lambda n, a: n == 1 and a >= 0.7,
        "name95_addr80": lambda n, a: n >= 0.95 and a >= 0.8,
        "name90_addr90": lambda n, a: n >= 0.9 and a >= 0.9,
        "same_address_name90": lambda n, a: a == 1 and n >= 0.9,
        "name95_addr70": lambda n, a: n >= 0.95 and a >= 0.7,
    }
    additions = {key: defaultdict(set) for key in rules}
    for qid, items in by_query.items():
        anchors = [lookup[target] for target in predictions[qid] if target in lookup]
        if not anchors:
            continue
        for target, _ in items:
            if target in predictions[qid]:
                continue
            name, address = lookup[target]
            for aname, aaddress in anchors:
                ns = JaroWinkler.normalized_similarity(name, aname) if name and aname else 0.0
                ads = JaroWinkler.normalized_similarity(address, aaddress) if address and aaddress else 0.0
                for key, rule in rules.items():
                    if rule(ns, ads):
                        additions[key][qid].add(target)
    base = sum(f05(truths[qid], predictions[qid]) for qid in selected) / len(selected)
    results = {}
    for key, add in additions.items():
        new = sum(f05(truths[qid], predictions[qid] | add[qid]) for qid in selected) / len(selected)
        tp = sum(len(ids & truths[qid]) for qid, ids in add.items())
        fp = sum(len(ids - truths[qid]) for qid, ids in add.items())
        results[key] = {"macro_f0_5": new, "delta": new - base, "added_tp": tp, "added_fp": fp}
    report = {"scope": "Exploratory 2k OOF subset; rules selected and measured on same sample; require disjoint confirmation before promotion", "queries": len(selected), "baseline": base, "candidate_pairs": len(candidates), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
