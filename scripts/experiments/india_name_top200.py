"""Diagnostic full-pool India name top200 extension on fixed unlabeled retrieval."""
import hashlib
import json
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy import sparse

from src.blocking.char_retrieval import fused_top_k_rows
from src.blocking.token_candidates import summarize_candidates


def main() -> None:
    config_path = Path("configs/experiments/EXP-030.json")
    config = json.loads(config_path.read_text())
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    index_dir = Path("artifacts/retrieval/P4-INDEX-PROBE-001")
    idf = Path("outputs/candidates/P4-B-002/name_char3_idf.npz")
    index_metrics = json.loads((index_dir / "metrics.json").read_text())
    if hashlib.sha256(idf.read_bytes()).hexdigest() != index_metrics["idf_sha256"]:
        raise ValueError("Index and frozen Phase4-B name IDF disagree")
    queries = (pl.read_parquet("artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet")
               .filter(pl.col("country") == "India").sort("sample_hash").head(1000))
    if queries.height != 1000 or set(queries["fold"].unique()) != {1, 2, 3}:
        raise ValueError("Unexpected query sample; Fold4 must stay closed")
    selected_ids = set(queries["entity_id"])
    queries.write_parquet(out / "queries.parquet")
    from scripts.aws.benchmark_indexes import vectorizer

    v = vectorizer(idf)
    target = sparse.load_npz(index_dir / "name-India.npz").T.tocsr()
    ids = np.load(index_dir / "name-India-ids.npy", allow_pickle=False)
    matrix = v.transform(queries["n"].to_list())
    retrieved = fused_top_k_rows(matrix, target, ids, 200, threads=1, transposed=True)
    del target, matrix
    if time.perf_counter() - started > config["runtime_cap_seconds"]:
        raise TimeoutError("Top200 diagnostic exceeded its runtime cap")
    rows = [(qid, str(tid), float(score), rank)
            for qid, (targets, scores) in zip(queries["entity_id"], retrieved)
            for rank, (tid, score) in enumerate(zip(targets, scores), 1)]
    pl.DataFrame(rows, schema=["source1_entity_id", "target_id", "route_score", "route_rank"],
                 orient="row").write_parquet(out / "name_top200.parquet", compression="zstd")
    baseline = {qid: set() for qid in selected_ids}
    names = {qid: set() for qid in selected_ids}
    for field in ("name", "address"):
        route = (pl.scan_parquet(f"outputs/candidates/P4-B-002/{field}_char3.parquet")
                 .filter(pl.col("source1_entity_id").is_in(list(selected_ids)))
                 .select("source1_entity_id", "target_id").collect())
        for qid, tid in route.iter_rows():
            baseline[qid].add(tid)
            if field == "name":
                names[qid].add(tid)
    parity = sum(set(targets[:100]) != names[qid]
                 for qid, (targets, _) in zip(queries["entity_id"], retrieved))
    if parity:
        raise RuntimeError(f"Frozen top100 name route parity failed for {parity} queries")
    db = duckdb.connect("artifacts/audit.duckdb", read_only=True,
                        config={"threads": 2, "memory_limit": "1GB"})
    truth = {qid: set() for qid in selected_ids}
    for qid, tid in db.execute(
        "SELECT p.source1_entity_id,p.target_id FROM positive_pairs p "
        "JOIN read_parquet('outputs/experiments/EXP-030-v001/queries.parquet') q "
        "ON q.entity_id=p.source1_entity_id"
    ).fetchall():
        truth[qid].add(tid)
    expanded = {qid: set(baseline[qid]) for qid in selected_ids}
    new_pairs = []
    for qid, (targets, _) in zip(queries["entity_id"], retrieved):
        for tid in targets[100:]:
            tid = str(tid)
            if tid not in expanded[qid]:
                expanded[qid].add(tid)
                new_pairs.append((qid, tid))
    before = summarize_candidates(queries.select("entity_id", "country", pl.col("n").alias("name"),
                                                 pl.col("a").alias("address")).to_dicts(),
                                  truth, baseline, 10320219)
    after = summarize_candidates(queries.select("entity_id", "country", pl.col("n").alias("name"),
                                                pl.col("a").alias("address")).to_dicts(),
                                 truth, expanded, 10320219)
    report = {"experiment": "EXP-030", "timestamp": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "query_count": queries.height, "index_target_rows": len(ids),
              "top100_name_parity_mismatches": parity,
              "new_candidate_pairs": len(new_pairs),
              "new_true_links": after["retrieved_links"] - before["retrieved_links"],
              "candidates_per_new_true_link": (len(new_pairs) / (after["retrieved_links"] - before["retrieved_links"])
                                                if after["retrieved_links"] > before["retrieved_links"] else None),
              "baseline": before, "expanded": after,
              "runtime_seconds": time.perf_counter() - started,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
              "scope": "Fixed hash-sample known-country development diagnostic; all full-pool targets; Fold4 CLOSED; no matcher trained; not an independent model score"}
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("query_count", "top100_name_parity_mismatches",
                                                    "new_candidate_pairs", "new_true_links",
                                                    "candidates_per_new_true_link", "runtime_seconds")}), flush=True)


if __name__ == "__main__":
    main()
