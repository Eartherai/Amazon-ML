"""Unlabeled SUB-001 shift diagnostics from completed deterministic test shards."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl


def shard_stats(shards: Path, prefix: str, count: int) -> dict:
    candidate_counts = []
    predicted_counts = []
    for index in range(count):
        base = f"{prefix}-s{index:03d}"
        paths = {kind: shards / f"{base}-{kind}.tsv.gz" for kind in ("candidates", "matching")}
        if not all(path.exists() for path in paths.values()):
            raise FileNotFoundError(base)
        rows = {}
        for kind, path in paths.items():
            with gzip.open(path, "rt", encoding="utf-8") as source:
                source.readline()
                rows[kind] = [(key, len(ids.split(",")) if ids else 0) for line in source
                              for key, ids in [line.rstrip("\n").split("\t", 1)]]
        if [x[0] for x in rows["candidates"]] != [x[0] for x in rows["matching"]]:
            raise ValueError(f"Unaligned shard {base}")
        candidate_counts.extend(x[1] for x in rows["candidates"])
        predicted_counts.extend(x[1] for x in rows["matching"])
    if not candidate_counts:
        return {"entities": 0}
    candidates = np.array(candidate_counts)
    predictions = np.array(predicted_counts)
    return {
        "entities": len(candidates),
        "scored_pairs": int(candidates.sum()),
        "mean_candidates": float(candidates.mean()),
        "candidate_p50_p95_p99": np.quantile(candidates, [0.5, 0.95, 0.99]).tolist(),
        "predicted_empty_rate": float((predictions == 0).mean()),
        "mean_predicted_matches": float(predictions.mean()),
        "predicted_match_count_histogram": dict(sorted(Counter(predictions.tolist()).items())),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    progress = json.loads((a.inference / "progress.json").read_text())
    done = {item["country"]: item["shards"] for item in progress["country_progress"]}
    if "active" in progress:
        done[progress["active"]["country"]] = progress["active"]["shards"]
    observed = {country: shard_stats(a.inference / "shards", country, count)
                for country, count in sorted(done.items()) if count}
    queries = pl.read_parquet("artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet", columns=["entity_id", "country"])
    predictions = pl.read_csv("outputs/experiments/P4-NUMERIC-B-001/with_canonical_numeric/entity_predictions.tsv", separator="\t")
    reference = queries.join(predictions, left_on="entity_id", right_on="source1_entity_id", validate="1:1")
    baseline = {}
    for country in sorted(reference["country"].unique()):
        subset = reference.filter(pl.col("country") == country)
        counts = np.array([len(x.split(",")) if x else 0 for x in subset["matched_entity_ids"].fill_null("")])
        baseline[country] = {"entities": len(counts), "predicted_empty_rate": float((counts == 0).mean()), "mean_predicted_matches": float(counts.mean())}
    test_queries = pl.read_parquet("artifacts/cloud/phase5/test-input-v001/queries.parquet", columns=["country", "n", "a"])
    covariates = {}
    for country in sorted(test_queries["country"].unique()):
        subset = test_queries.filter(pl.col("country") == country)
        covariates[country] = {
            "entities": len(subset),
            "name_nonascii_rate": float(subset["n"].str.contains(r"[^\x00-\x7F]").mean()),
            "name_missing_rate": float((subset["n"].str.len_chars() == 0).mean()),
            "address_missing_rate": float((subset["a"].str.len_chars() == 0).mean()),
            "address_digit_rate": float(subset["a"].str.contains(r"[0-9]").mean()),
            "name_length_p50_p95": [subset["n"].str.len_chars().quantile(q) for q in (0.5, 0.95)],
            "address_length_p50_p95": [subset["a"].str.len_chars().quantile(q) for q in (0.5, 0.95)],
        }
    result = {
        "scope": "Unlabeled test outputs only; no hidden labels or external lookups. Completed SHA256 query shards form a near-random subset. OOF reference uses its original per-fold thresholds, so comparison is directional.",
        "test_completed_shards": done,
        "test": observed,
        "known_country_oof_reference": baseline,
        "test_covariates": covariates,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
