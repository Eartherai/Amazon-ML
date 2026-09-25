"""Build owner-safe 51-feature rows from complete classical retrieval shards.

Only selected folds 1–3 are labeled. Owner folds are retained solely to prevent
held-out targets from entering model fitting as negative examples.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from src.models.sub001_features import NAMES, pair_features


SCHEMA = ["source1_entity_id", "target_id", "country", "fold", "owner_fold", "label", *NAMES]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def route_files(root: Path, field: str, country_index: int, shard: int) -> list[str]:
    found = sorted(glob.glob(str(root / f"{field}-c{country_index:03d}-s{shard:03d}-b*.parquet")))
    if not found:
        raise FileNotFoundError(f"Missing {field} country {country_index} shard {shard}")
    return found


def route_scores(paths: list[str], position: int, slots: dict[tuple[str, str], list[float]],
                 allowed_queries: set[str] | None = None) -> None:
    frame = pl.read_parquet(paths, columns=["source1_entity_id", "target_id", "route_score", "route_rank"])
    for query_id, target_id, score, rank in frame.iter_rows():
        if allowed_queries is not None and query_id not in allowed_queries:
            continue
        key = (query_id, target_id)
        slot = slots.setdefault(key, [0., 0., 0., 0., 0.])
        if slot[2 + position] or rank < 1:
            raise ValueError(f"Duplicate or invalid route rank for {key}")
        slot[position] = float(score)
        slot[2 + position] = 1.0 / rank
        slot[4] += 1.0


def write_part(rows: list[tuple], destination: Path) -> None:
    if not rows:
        return
    frame = pl.DataFrame(rows, schema=SCHEMA, orient="row")
    frame = frame.with_columns(pl.col(NAMES).cast(pl.Float32), pl.col("fold").cast(pl.Int8),
                               pl.col("owner_fold").cast(pl.Int8), pl.col("label").cast(pl.Int8))
    frame.write_parquet(destination, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    parser.add_argument("--expected-queries", type=int, default=200_000)
    parser.add_argument("--expected-targets", type=int, default=10_320_219)
    parser.add_argument("--folds", default="1,2,3", help="Unlocked labeled S1 folds, never fold 4")
    parser.add_argument("--restrict-to-queries", action="store_true",
                        help="Skip unlabeled/locked query routes from the full retrieval")
    parser.add_argument("--experiment-id", default="EXP-032")
    parser.add_argument("--upload-bucket")
    parser.add_argument("--upload-prefix")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    folds = tuple(int(value) for value in args.folds.split(","))
    if (min(args.shards, args.chunk_rows, args.expected_queries, args.expected_targets) < 1
            or not folds or len(folds) != len(set(folds)) or not set(folds) <= {0, 1, 2, 3}):
        raise ValueError("Positive shard/chunk sizes required")
    if bool(args.upload_bucket) != bool(args.upload_prefix):
        raise ValueError("Upload bucket and prefix must be specified together")
    args.output.mkdir(parents=True)
    receipts = []

    def save_part(rows: list[tuple], destination: Path) -> None:
        write_part(rows, destination)
        if args.upload_bucket and destination.exists():
            from upload_verified import upload
            receipt = upload(destination, args.upload_bucket, args.upload_prefix.rstrip("/") + "/" + destination.name)
            receipts.append(receipt)
            (args.output / "parts-receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")

    started = time.perf_counter()
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    for item in manifest["files"]:
        path = args.inputs / item["name"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Feature input checksum mismatch: {path}")
    queries = pl.read_parquet(args.inputs / "labeled_queries.parquet")
    if (len(queries) != args.expected_queries or queries["entity_id"].n_unique() != len(queries)
            or set(queries["fold"].unique().to_list()) != set(folds)):
        raise ValueError("Unexpected query/fold coverage")
    query_lookup = {qid: (name, address, country, int(fold)) for qid, country, name, address, fold in
                    queries.select("entity_id", "country", "n", "a", "fold").iter_rows()}
    allowed_queries = set(query_lookup) if args.restrict_to_queries else None
    truth_frame = pl.read_parquet(args.inputs / "truth.parquet")
    truth = set(truth_frame.iter_rows())
    if len(truth) != len(truth_frame):
        raise ValueError("Duplicate truth pair")
    truth_counts = Counter(query_id for query_id, _ in truth)
    owner_frame = pl.read_parquet(args.inputs / "ownership.parquet")
    owners = dict(owner_frame.iter_rows())
    if len(owners) != args.expected_targets:
        raise ValueError("Incomplete target ownership")
    trans = pl.read_parquet(args.inputs / "name_map.parquet")
    transmap = dict(trans.iter_rows())
    countries = sorted(queries["country"].unique().to_list())
    con = duckdb.connect(config={"threads": 2, "memory_limit": "4GB"})
    candidate_counts: Counter[str] = Counter()
    retrieved_positive_counts: Counter[str] = Counter()
    total_pairs = total_positive = 0
    for ci, country in enumerate(countries):
        cursor = con.execute("SELECT entity_id,n,a FROM read_parquet(?) WHERE country=?",
                             [str(args.inputs / "targets.parquet"), country])
        targets = {}
        while batch := cursor.fetchmany(100_000):
            targets.update((tid, (name, address)) for tid, name, address in batch)
        print(json.dumps({"country": country, "targets": len(targets), "seconds": time.perf_counter() - started}), flush=True)
        for shard in range(args.shards):
            scores: dict[tuple[str, str], list[float]] = {}
            route_scores(route_files(args.routes, "name", ci, shard), 0, scores, allowed_queries)
            route_scores(route_files(args.routes, "address", ci, shard), 1, scores, allowed_queries)
            pending: list[tuple] = []
            part = 0
            for (qid, tid), (name_score, address_score, name_rank, address_rank, route_count) in sorted(scores.items()):
                qname, qaddress, qcountry, fold = query_lookup[qid]
                if qcountry != country or tid not in targets or tid not in owners:
                    raise ValueError(f"Unknown or cross-country route pair: {qid},{tid}")
                tname, taddress = targets[tid]
                values = pair_features(qname, qaddress, qcountry, tname, taddress, country, tid,
                                       name_score, address_score, name_rank, address_rank,
                                       route_count, transmap)
                positive = int((qid, tid) in truth)
                pending.append((qid, tid, country, fold, int(owners[tid]), positive, *values.tolist()))
                candidate_counts[qid] += 1
                retrieved_positive_counts[qid] += positive
                total_pairs += 1
                total_positive += positive
                if len(pending) >= args.chunk_rows:
                    save_part(pending, args.output / f"features-c{ci:03d}-s{shard:03d}-b{part:05d}.parquet")
                    pending.clear()
                    part += 1
            if pending:
                save_part(pending, args.output / f"features-c{ci:03d}-s{shard:03d}-b{part:05d}.parquet")
            print(json.dumps({"country": country, "shard": shard, "candidate_pairs": total_pairs,
                              "seconds": time.perf_counter() - started}), flush=True)
        del targets
    con.close()
    counts = np.array([candidate_counts[qid] for qid in query_lookup], dtype=np.int32)
    positive_entities = sum(truth_counts[qid] > 0 for qid in query_lookup)
    complete = sum(retrieved_positive_counts[qid] == truth_counts[qid] for qid in query_lookup if truth_counts[qid] > 0)
    any_match = sum(retrieved_positive_counts[qid] > 0 for qid in query_lookup if truth_counts[qid] > 0)
    link_recall = total_positive / len(truth)
    report = {"experiment": args.experiment_id, "queries": len(query_lookup), "features": NAMES,
              "candidate_pairs": total_pairs, "retrieved_positives": total_positive,
              "truth_pairs": len(truth), "link_recall": link_recall,
              "complete_positive_entity_recall": complete / positive_entities,
              "any_match_recall": any_match / positive_entities,
              "candidates_per_query": {"mean": float(counts.mean()), "p50": float(np.quantile(counts, .5)),
                                       "p95": float(np.quantile(counts, .95)), "p99": float(np.quantile(counts, .99))},
              "fold4": "CLOSED", "seconds": time.perf_counter() - started}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "COMPLETE.json").write_text(json.dumps({"experiment": args.experiment_id, "queries": len(query_lookup),
                                                         "candidate_pairs": total_pairs, "feature_parts": len(list(args.output.glob('features-*.parquet'))),
                                                         "uploaded_parts": len(receipts) if args.upload_bucket else None,
                                                         "fold4": "CLOSED"}, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
