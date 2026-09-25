"""Prepare full-retrieval feature inputs from unlocked folds 0–3 only.

Fold 4 query labels and positive links never enter this output. The all-target
ownership file retains fold IDs only to exclude held-out targets from fit
negatives. Source data and the audited database remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import duckdb


ROOT = Path("artifacts/cloud/phase5")
QUERIES = ROOT / "full-query-v001/queries.parquet"
TARGETS = ROOT / "index-input-v001/targets.parquet"
OWNERSHIP = ROOT / "feature-200k-input-v001/ownership.parquet"
NAME_MAP = ROOT / "feature-200k-input-v001/name_map.parquet"
DATABASE = Path("artifacts/audit.duckdb")
EXPECTED_UNLOCKED = 1_765_649
EXPECTED_TARGETS = 10_320_219


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "feature-unlocked-v001")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not all(path.is_file() for path in (QUERIES, TARGETS, OWNERSHIP, NAME_MAP, DATABASE)):
        raise FileNotFoundError("Full-retrieval inputs or audited database missing")
    if shutil.disk_usage(".").free < 9 * 1024**3:
        raise RuntimeError("Insufficient disk reserve for feature input export")
    args.output.mkdir(parents=True)
    for source, name in ((TARGETS, "targets.parquet"), (OWNERSHIP, "ownership.parquet"),
                         (NAME_MAP, "name_map.parquet")):
        os.link(source, args.output / name)
    con = duckdb.connect(str(DATABASE), read_only=True,
                         config={"threads": 2, "memory_limit": "3GB"})
    try:
        con.execute(f"COPY (SELECT q.entity_id,q.country,q.n,q.a,v.fold,v.n_matches "
                    f"FROM read_parquet({literal(QUERIES)}) q "
                    "JOIN validation_folds v ON v.source1_entity_id=q.entity_id "
                    "WHERE v.fold IN (0,1,2,3) ORDER BY q.entity_id) "
                    f"TO {literal(args.output / 'labeled_queries.parquet')} "
                    "(FORMAT PARQUET,COMPRESSION ZSTD)")
        con.execute(f"COPY (SELECT p.source1_entity_id,p.target_id FROM positive_pairs p "
                    f"JOIN read_parquet({literal(args.output / 'labeled_queries.parquet')}) q "
                    "ON q.entity_id=p.source1_entity_id "
                    "ORDER BY p.source1_entity_id,p.target_id) "
                    f"TO {literal(args.output / 'truth.parquet')} "
                    "(FORMAT PARQUET,COMPRESSION ZSTD)")
        fold_rows = con.execute("SELECT fold,count(*) FROM read_parquet(?) "
                                "GROUP BY fold ORDER BY fold",
                                [str(args.output / "labeled_queries.parquet")]).fetchall()
        counts = {int(fold): count for fold, count in fold_rows}
        truth_count = con.execute("SELECT count(*) FROM read_parquet(?)",
                                  [str(args.output / "truth.parquet")]).fetchone()[0]
        n_matches = con.execute("SELECT sum(n_matches) FROM read_parquet(?)",
                                [str(args.output / "labeled_queries.parquet")]).fetchone()[0]
        target_count = con.execute("SELECT count(*) FROM read_parquet(?)",
                                   [str(TARGETS)]).fetchone()[0]
        ownership_count = con.execute("SELECT count(*) FROM read_parquet(?)",
                                      [str(OWNERSHIP)]).fetchone()[0]
    finally:
        con.close()
    if (set(counts) != {0, 1, 2, 3} or sum(counts.values()) != EXPECTED_UNLOCKED
            or truth_count != n_matches or target_count != EXPECTED_TARGETS
            or ownership_count != EXPECTED_TARGETS):
        raise ValueError("Unlocked feature input coverage or truth counts disagree")
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(args.output.iterdir()) if path.is_file()]
    manifest = {"scope": "EXP-036 full retrieval feature store, query labels and links only for folds 0-3; Fold4 CLOSED",
                "query_count": sum(counts.values()), "queries_by_fold": counts,
                "target_count": target_count, "ownership_count": ownership_count,
                "truth_pairs": truth_count, "full_query_sha256": sha256(QUERIES),
                "files": files}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"queries": sum(counts.values()), "by_fold": counts,
                      "truth_pairs": truth_count, "targets": target_count,
                      "file_bytes": sum(item["bytes"] for item in files)}), flush=True)


if __name__ == "__main__":
    main()
