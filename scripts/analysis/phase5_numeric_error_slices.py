"""Aggregate known-country NUMERIC-V2 misses without touching Fold4 or test labels."""
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def main() -> None:
    root = Path("outputs/analysis/P5-NUMERIC-ERRORS-004")
    db = duckdb.connect("artifacts/audit.duckdb", read_only=True,
                        config={"threads": 2, "memory_limit": "3GB"})
    query_file = "artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet"
    candidate_dir = "outputs/candidates/P4-B-002"
    prediction_file = "outputs/experiments/P4-NUMERIC-B-001/with_canonical_numeric/entity_predictions.tsv"
    db.execute(f"CREATE TEMP VIEW q AS SELECT entity_id, country, fold FROM read_parquet('{query_file}')")
    if db.execute("SELECT count(*) FROM q").fetchone()[0] != 20000 or {
        row[0] for row in db.execute("SELECT DISTINCT fold FROM q").fetchall()
    } != {1, 2, 3}:
        raise ValueError("Unexpected query population; Fold4 must remain closed")
    db.execute(f"CREATE TEMP TABLE cand AS SELECT source1_entity_id, target_id FROM read_parquet('{candidate_dir}/name_char3.parquet') "
               f"UNION SELECT source1_entity_id, target_id FROM read_parquet('{candidate_dir}/address_char3.parquet')")
    db.execute("CREATE TEMP TABLE pred AS SELECT source1_entity_id, target_id FROM "
               "(SELECT source1_entity_id, unnest(str_split(coalesce(matched_entity_ids, ''), ',')) target_id "
               f"FROM read_csv('{prediction_file}', delim='\t', header=true, auto_detect=false, "
               "columns={'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'})) "
               "WHERE target_id <> ''")
    db.execute("CREATE TEMP TABLE links AS SELECT q.country, q.fold, "
               "CASE WHEN starts_with(p.target_id,'S2-') THEN 'S2' ELSE 'S3' END target_source, "
               "(c.target_id IS NOT NULL) retrieved, (r.target_id IS NOT NULL) predicted, "
               "(t.n<>'' AND t.n=s.n) exact_name, "
               "(t.a='' OR s.a='') any_address_missing, "
               "regexp_matches(t.n, '[^\\x00-\\x7F]') nonascii_target_name "
               "FROM positive_pairs p JOIN q ON q.entity_id=p.source1_entity_id "
               "JOIN targets_normalized t ON t.entity_id=p.target_id "
               "JOIN s1_normalized s ON s.entity_id=p.source1_entity_id "
               "LEFT JOIN cand c ON c.source1_entity_id=p.source1_entity_id AND c.target_id=p.target_id "
               "LEFT JOIN pred r ON r.source1_entity_id=p.source1_entity_id AND r.target_id=p.target_id")
    dimensions = ["country", "target_source", "exact_name", "any_address_missing", "nonascii_target_name"]
    report = {"run": "P5-NUMERIC-ERRORS-004", "timestamp": datetime.now(timezone.utc).isoformat(),
              "scope": "20k natural folds1-3 known-country development labels; frozen P4-B-002 candidate set and NUMERIC-V2 predictions; Fold4 CLOSED",
              "dimensions": {}}
    for dimension in dimensions:
        rows = db.execute(f"SELECT {dimension} category, count(*) true_links, "
                          "count(*) FILTER (WHERE retrieved) retrieved_links, "
                          "count(*) FILTER (WHERE predicted) predicted_links, "
                          "count(*) FILTER (WHERE NOT retrieved) retrieval_misses, "
                          "count(*) FILTER (WHERE retrieved AND NOT predicted) matcher_misses "
                          f"FROM links GROUP BY {dimension} ORDER BY {dimension}").fetchall()
        report["dimensions"][dimension] = [dict(zip(
            ["category", "true_links", "retrieved_links", "predicted_links", "retrieval_misses", "matcher_misses"], row))
            for row in rows]
    # Cross-tab isolates the India/US and S2/S3 bottlenecks.
    rows = db.execute("SELECT country, target_source, count(*) true_links, "
                      "count(*) FILTER (WHERE NOT retrieved) retrieval_misses, "
                      "count(*) FILTER (WHERE retrieved AND NOT predicted) matcher_misses "
                      "FROM links GROUP BY country,target_source ORDER BY country,target_source").fetchall()
    report["country_source"] = [dict(zip(
        ["country", "target_source", "true_links", "retrieval_misses", "matcher_misses"], row)) for row in rows]
    rows = db.execute("SELECT q.country, (s.a='' OR t.a='') any_address_missing, "
                      "count(*) predicted_pairs, count(p.target_id) correct_pairs "
                      "FROM pred r JOIN q ON q.entity_id=r.source1_entity_id "
                      "JOIN s1_normalized s ON s.entity_id=r.source1_entity_id "
                      "JOIN targets_normalized t ON t.entity_id=r.target_id "
                      "LEFT JOIN positive_pairs p ON p.source1_entity_id=r.source1_entity_id AND p.target_id=r.target_id "
                      "GROUP BY q.country,any_address_missing ORDER BY q.country,any_address_missing").fetchall()
    report["predicted_country_address"] = [dict(zip(
        ["country", "any_address_missing", "predicted_pairs", "correct_pairs"], row)) for row in rows]
    totals = db.execute("SELECT count(*), count(*) FILTER (WHERE NOT retrieved), "
                        "count(*) FILTER (WHERE retrieved AND NOT predicted) FROM links").fetchone()
    report["total"] = dict(zip(["true_links", "retrieval_misses", "matcher_misses"], totals))
    root.mkdir(parents=True, exist_ok=False)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["total"]))


if __name__ == "__main__":
    main()
