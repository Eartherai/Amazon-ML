"""Bounded full-pool lexical candidate pilot, with no learned matcher.

Training-only token frequency ranks eligible query keys. Full index frequency
is a deterministic fanout safety check, not a trained feature. Query labels are
read only after candidates have been generated. The supplied audit is read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import duckdb
import numpy as np


def _rss_gib() -> float:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum / 1024 ** (3 if sys.platform == "darwin" else 2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_queries(connection: duckdb.DuckDBPyConnection, count_per_country: int) -> list[dict[str, Any]]:
    """Choose deterministic fold-0 queries, retaining the natural singleton rate."""
    if count_per_country <= 0:
        raise ValueError("count_per_country must be positive")
    connection.execute("""CREATE TEMP TABLE pilot_queries AS
      SELECT entity_id, country, n, a FROM (
        SELECT s.*, row_number() OVER (PARTITION BY s.country ORDER BY sha256(s.entity_id)) sample_rank
        FROM s1_normalized s JOIN validation_folds f ON f.source1_entity_id=s.entity_id
        WHERE f.fold=0
      ) WHERE sample_rank <= ?""", [count_per_country])
    connection.execute("""CREATE TEMP TABLE query_tokens AS
      SELECT entity_id, country, 'name' field, token FROM pilot_queries,
        UNNEST(list_distinct(string_split(n,' '))) u(token)
      WHERE length(token)>=3 AND regexp_matches(token,'\\p{L}')
      UNION ALL
      SELECT entity_id, country, 'address' field, token FROM pilot_queries,
        UNNEST(list_distinct(string_split(a,' '))) u(token)
      WHERE length(token)>=3 AND regexp_matches(token,'\\p{L}')""")
    rows = connection.execute("SELECT entity_id,country,n,a FROM pilot_queries ORDER BY entity_id").fetchall()
    return [dict(zip(("entity_id", "country", "name", "address"), row)) for row in rows]


def generate_candidates(connection: duckdb.DuckDBPyConnection, config: dict[str, Any]) -> dict[str, Any]:
    """Create a temporary final_candidates table using the entire target index."""
    max_df = int(config["max_full_pool_token_df"])
    keys = int(config["max_query_tokens_per_field"])
    route_k = int(config["route_top_k_per_target_source"])
    if min(max_df, keys, route_k) <= 0:
        raise ValueError("Frequency, token-count and route caps must be positive")
    timings: dict[str, float] = {}
    started = time.perf_counter()
    partitions = connection.execute("SELECT country,left(entity_id,2) FROM targets_normalized GROUP BY 1,2 ORDER BY 1,2").fetchall()
    connection.execute("""CREATE TEMP TABLE partial_token_stats(
      country VARCHAR, field VARCHAR, token VARCHAR, full_df BIGINT, train_df BIGINT)""")
    for field, column in (("name", "n"), ("address", "a")):
        before = time.perf_counter()
        # Duplicate tokens within a record count once. Positive target ownership
        # keeps validation-owned strings out of fitted key-ranking frequencies.
        for country, source in partitions:
            # Scalar SELECT UNNEST streams batches. A correlated FROM UNNEST
            # materializes a large delimiter join on this dataset and spills.
            connection.execute(f"""INSERT INTO partial_token_stats
              WITH tokens AS (
                SELECT entity_id,country,unnest(list_distinct(string_split({column},' '))) token
                FROM targets_normalized WHERE country=? AND starts_with(entity_id,?)),
              hits AS (
                SELECT t.* FROM tokens t
                JOIN (SELECT DISTINCT country,token FROM query_tokens WHERE field='{field}') q
                USING(country,token)),
              owners AS (SELECT target_id,owner_fold FROM target_ownership WHERE starts_with(target_id,?))
              SELECT h.country,'{field}',h.token,count(*) full_df,
                count(*) FILTER(WHERE o.owner_fold IN (-1,1,2,3)) train_df
              FROM hits h JOIN owners o ON o.target_id=h.entity_id
              GROUP BY h.country,h.token""", [country,source+"-",source+"-"])
        timings[f"{field}_frequency_seconds"] = time.perf_counter()-before
        print(f"{field} frequencies complete in {timings[f'{field}_frequency_seconds']:.2f}s", file=sys.stderr, flush=True)
    connection.execute("""CREATE TEMP TABLE token_stats AS
      SELECT country,field,token,sum(full_df)::BIGINT full_df,sum(train_df)::BIGINT train_df
      FROM partial_token_stats GROUP BY country,field,token""")
    connection.execute("""CREATE TEMP TABLE selected_keys AS
      SELECT entity_id,country,field,token,full_df,train_df FROM (
        SELECT q.*,s.full_df,s.train_df,
          row_number() OVER(PARTITION BY q.entity_id,q.field
              ORDER BY s.train_df,s.full_df,q.token) key_rank
        FROM query_tokens q JOIN token_stats s USING(country,field,token)
        WHERE s.full_df BETWEEN 1 AND ?)
      WHERE key_rank <= ?""", [max_df, keys])
    connection.execute("""CREATE TEMP TABLE route_scores(
      source1_entity_id VARCHAR,target_id VARCHAR,route VARCHAR,route_score DOUBLE)""")
    for field, column in (("name", "n"), ("address", "a")):
        before = time.perf_counter()
        connection.execute(f"""INSERT INTO route_scores
          SELECT q.entity_id,t.entity_id,'exact_{field}',1.0
          FROM pilot_queries q JOIN targets_normalized t
            ON q.country=t.country AND q.{column}=t.{column}
          WHERE q.{column} IS NOT NULL AND q.{column}<>''""")
        timings[f"exact_{field}_seconds"] = time.perf_counter()-before
        before = time.perf_counter()
        # Restrict postings to selected keys after computing full-pool counts;
        # all target IDs remain eligible and every matching posting is considered.
        connection.execute(f"""CREATE TEMP TABLE {field}_hits(
          source1_entity_id VARCHAR,target_id VARCHAR,shared_tokens BIGINT,rarity_score DOUBLE)""")
        for country,source in partitions:
            connection.execute(f"""INSERT INTO {field}_hits
              WITH tokens AS (
                SELECT entity_id,country,unnest(list_distinct(string_split({column},' '))) token
                FROM targets_normalized WHERE country=? AND starts_with(entity_id,?)),
              postings AS (
                SELECT t.* FROM tokens t
                JOIN (SELECT DISTINCT country,token FROM selected_keys WHERE field='{field}') k
                USING(country,token))
              SELECT q.entity_id,p.entity_id,count(*) shared_tokens,sum(1.0/(q.train_df+1)) rarity_score
              FROM selected_keys q JOIN postings p USING(country,token)
              WHERE q.field='{field}' GROUP BY q.entity_id,p.entity_id""",[country,source+"-"])
        connection.execute(f"""INSERT INTO route_scores
          SELECT source1_entity_id,target_id,'rare_{field}',
            shared_tokens + least(rarity_score,0.99) FROM {field}_hits""")
        timings[f"rare_{field}_seconds"] = time.perf_counter()-before
        print(f"{field} retrieval complete in {timings[f'rare_{field}_seconds']:.2f}s", file=sys.stderr, flush=True)
    before = time.perf_counter()
    connection.execute("""INSERT INTO route_scores
      SELECT h.source1_entity_id,h.target_id,'numeric_name',
        h.shared_tokens+least(h.rarity_score,0.99)
      FROM name_hits h JOIN pilot_queries q ON q.entity_id=h.source1_entity_id
        JOIN targets_normalized t ON t.entity_id=h.target_id
      WHERE len(list_intersect(regexp_extract_all(q.a,'[0-9]+'),
                               regexp_extract_all(t.a,'[0-9]+')))>0""")
    timings["numeric_name_seconds"] = time.perf_counter()-before
    connection.execute("""CREATE TEMP TABLE ranked_routes AS
      SELECT *,row_number() OVER(PARTITION BY source1_entity_id,route,left(target_id,2)
        ORDER BY route_score DESC,target_id) route_rank FROM route_scores""")
    connection.execute("""CREATE TEMP TABLE capped_routes AS
      SELECT * FROM ranked_routes WHERE route_rank<=?""", [route_k])
    connection.execute("""CREATE TEMP TABLE final_candidates AS
      SELECT *,row_number() OVER(PARTITION BY source1_entity_id
        ORDER BY fusion_score DESC,target_id) candidate_rank FROM (
        SELECT source1_entity_id,target_id,
          sum(1.0/(?+route_rank)) fusion_score,
          string_agg(route,',' ORDER BY route) routes,
          min(route_rank) best_route_rank
        FROM capped_routes GROUP BY source1_entity_id,target_id)
    """, [int(config["rrf_constant"])])
    timings["total_generation_seconds"] = time.perf_counter()-started
    return {"timings": timings,
            "query_count": connection.execute("SELECT count(*) FROM pilot_queries").fetchone()[0],
            "target_pool_count": connection.execute("SELECT count(*) FROM targets_normalized").fetchone()[0],
            "target_pool_by_country_source": connection.execute(
                "SELECT country,left(entity_id,2),count(*) FROM targets_normalized GROUP BY 1,2 ORDER BY 1,2").fetchall(),
            "selected_key_counts": connection.execute(
                "SELECT field,count(*),count(DISTINCT entity_id) FROM selected_keys GROUP BY field ORDER BY field").fetchall(),
            "full_union_pairs": connection.execute("SELECT count(*) FROM final_candidates").fetchone()[0],
            "raw_route_pairs": connection.execute(
                "SELECT route,count(*) FROM route_scores GROUP BY route ORDER BY route").fetchall(),
            "frequency_scope": "train_df uses only target owners in -1,1,2,3; full_df is an inference-index fanout cap",
            "normalization": "Existing audited NFC + lower + preserve Unicode L/M/N + whitespace; no new fitted map",
            "numeric_route": "ASCII address digit-token agreement within rare-name hits; rank promotion only, no unique-pair rescue"}


def summarize_candidates(queries: list[dict[str, Any]], truth: dict[str, set[str]],
                         candidates: dict[str, set[str]], target_pool_count: int) -> dict[str, Any]:
    """Exact candidate oracle ceiling; no predicted matcher score is implied."""
    query_ids = [row["entity_id"] for row in queries]
    if set(truth) != set(query_ids) or set(candidates) != set(query_ids):
        raise ValueError("Truth and candidate maps must cover exactly every query")
    counts = np.array([len(candidates[key]) for key in query_ids], dtype=np.int64)
    true_count = sum(len(truth[key]) for key in query_ids)
    hits = {key: len(truth[key] & candidates[key]) for key in query_ids}
    positive = [key for key in query_ids if truth[key]]
    singletons = [key for key in query_ids if not truth[key]]
    oracle = {key: 1.0 if not truth[key] else 1.25*hits[key]/(hits[key]+0.25*len(truth[key]))
              for key in query_ids}
    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None
    by_country_source = []
    for country in sorted({row["country"] for row in queries}):
        ids = [row["entity_id"] for row in queries if row["country"] == country]
        for source in ("S2", "S3"):
            total = sum(sum(target.startswith(source+"-") for target in truth[key]) for key in ids)
            found = sum(sum(target.startswith(source+"-") for target in truth[key] & candidates[key]) for key in ids)
            by_country_source.append({"country": country, "target_source": source,
                "true_links": total, "retrieved_links": found,
                "link_recall": found/total if total else None})
    return {"query_count": len(query_ids), "true_links": true_count, "retrieved_links": sum(hits.values()),
            "candidate_pairs": int(counts.sum()), "target_pool_count": target_pool_count,
            "link_recall": sum(hits.values())/true_count if true_count else None,
            "oracle_macro_f0_5": mean(list(oracle.values())),
            "oracle_non_singleton_f0_5": mean([oracle[key] for key in positive]),
            "oracle_singleton_f0_5": 1.0 if singletons else None,
            "positive_entity_any_coverage": mean([float(hits[key]>0) for key in positive]),
            "positive_entity_all_coverage": mean([float(hits[key]==len(truth[key])) for key in positive]),
            "singleton_count": len(singletons), "singleton_rate": len(singletons)/len(query_ids) if query_ids else None,
            "singleton_candidate_free_rate": mean([float(not candidates[key]) for key in singletons]),
            "average_candidates_per_query": float(counts.mean()) if len(counts) else None,
            "candidate_quantiles": dict(zip(("p50", "p95", "p99"), np.quantile(counts,[.5,.95,.99]).tolist())) if len(counts) else {},
            "max_candidates_per_query": int(counts.max()) if len(counts) else 0,
            "candidate_reduction_ratio_vs_unrestricted_pool": 1-int(counts.sum())/(len(query_ids)*target_pool_count) if query_ids and target_pool_count else None,
            "by_country_source": by_country_source,
            "by_country": [{"country": country, "queries": len(ids),
                "link_recall": sum(hits[key] for key in ids)/sum(len(truth[key]) for key in ids)
                   if sum(len(truth[key]) for key in ids) else None,
                "oracle_macro_f0_5": mean([oracle[key] for key in ids]),
                "singleton_count": sum(not truth[key] for key in ids)}
                for country in sorted({row["country"] for row in queries})
                for ids in [[row["entity_id"] for row in queries if row["country"]==country]]]}


def run(database: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Generate, persist and evaluate a reproducible pilot without model fitting."""
    config = json.loads(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    scratch = output_dir / "scratch"
    scratch.mkdir()
    start = time.perf_counter()
    connection = duckdb.connect(str(database), read_only=True,
                               config={"threads": 2, "memory_limit": "2GB"})
    connection.execute("SET temp_directory=?", [str(scratch)])
    connection.execute("SET max_temp_directory_size='1GiB'")
    queries = prepare_queries(connection, int(config["queries_per_country"]))
    generation = generate_candidates(connection, config)
    # Ground truth is consumed only here, after the complete candidate table.
    truth: dict[str, set[str]] = {row["entity_id"]: set() for row in queries}
    for s1, target in connection.execute("""SELECT p.source1_entity_id,p.target_id FROM positive_pairs p
        JOIN pilot_queries q ON q.entity_id=p.source1_entity_id""").fetchall():
        truth[s1].add(target)
    rows = connection.execute("""SELECT source1_entity_id,target_id,candidate_rank
        FROM final_candidates ORDER BY source1_entity_id,candidate_rank""").fetchall()
    metrics: dict[str, Any] = {}
    for cap in list(config["total_top_k"]) + [None]:
        selected: dict[str, set[str]] = {row["entity_id"]: set() for row in queries}
        for s1, target, rank in rows:
            if cap is None or rank <= cap:
                selected[s1].add(target)
        key = "route_capped_union" if cap is None else f"total_top_{cap}"
        metrics[key] = summarize_candidates(queries, truth, selected, generation["target_pool_count"])
    per_route: dict[str, Any] = {}
    route_rows = connection.execute("SELECT source1_entity_id,target_id,route FROM capped_routes").fetchall()
    for route in sorted({row[2] for row in route_rows}):
        selected = {row["entity_id"]: set() for row in queries}
        for s1,target,which in route_rows:
            if which==route:
                selected[s1].add(target)
        per_route[route] = summarize_candidates(queries, truth, selected, generation["target_pool_count"])
    for table in ("pilot_queries", "selected_keys", "token_stats", "final_candidates", "capped_routes"):
        path = output_dir / (table + ".parquet")
        connection.execute(f"COPY (SELECT * FROM {table} ORDER BY 1,2) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)", [str(path)])
    connection.close()
    report = {"run_id": config["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "database": str(database), "database_mode": "read_only; temporary tables only",
        "config": config, "config_sha256": _sha256(config_path), "module_sha256": _sha256(Path(__file__)),
        "runtime_seconds": time.perf_counter()-start, "peak_process_rss_gib": _rss_gib(),
        "generation": generation, "metrics": metrics, "per_route": per_route,
        "artifacts": {path.name: _sha256(path) for path in output_dir.glob("*.parquet")},
        "limitations": ["1,000 balanced-country development queries, not population-weighted and not locked evaluation",
            "Lexical exact-token routes cannot solve absent shared text or cross-script matches",
            "Frequency cap and three-token key budget discard high-fanout/nonexact matches",
            "K applies to final RRF ranked union; route caps apply separately per target source",
            "Candidate oracle is an upper bound, not a trained-model validation score",
            "Numeric route uses ASCII digits and promotes existing name hits rather than adding unique pairs"]}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (output_dir / "config.json").write_text(config_path.read_text())
    print(json.dumps({"run_id": report["run_id"], "runtime_seconds": report["runtime_seconds"],
                     "peak_process_rss_gib": report["peak_process_rss_gib"],
                     "metrics": {k:{x:v[x] for x in ("link_recall","candidate_pairs","oracle_macro_f0_5")}
                                 for k,v in metrics.items()}}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.database,args.config,args.output_dir)


if __name__ == "__main__":
    main()
