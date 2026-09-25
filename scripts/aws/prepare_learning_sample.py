"""Select 200k unlocked S1 queries for a classical learning-size experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import duckdb


ROOT = Path("artifacts/cloud/phase5")
OUTPUT = ROOT / "learning-200k-input-v001"
FULL = ROOT / "full-query-v001"
SAMPLE_B = Path("artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet")
LIMITS = {1: 66_667, 2: 66_667, 3: 66_666}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    con = duckdb.connect("artifacts/audit.duckdb", read_only=True, config={"threads": 4, "memory_limit": "4GB"})
    con.execute("CREATE TEMP TABLE old_ids AS SELECT entity_id FROM read_parquet(?)", [str(SAMPLE_B)])
    if con.execute("SELECT count(*)-count(DISTINCT entity_id) FROM old_ids").fetchone()[0]:
        raise ValueError("Duplicate old sample IDs")
    con.execute("CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)", [str(FULL / "queries.parquet")])
    con.execute("""CREATE TEMP TABLE chosen AS
      SELECT q.entity_id,q.country,q.n,q.a,v.fold,(o.entity_id IS NOT NULL) old_sample,
             row_number() OVER(PARTITION BY v.fold
               ORDER BY (o.entity_id IS NOT NULL) DESC, sha256(q.entity_id),q.entity_id) sample_rank
      FROM q JOIN validation_folds v ON q.entity_id=v.source1_entity_id
      LEFT JOIN old_ids o ON q.entity_id=o.entity_id
      WHERE v.fold IN (1,2,3)""")
    where = "(fold=1 AND sample_rank<=66667) OR (fold=2 AND sample_rank<=66667) OR (fold=3 AND sample_rank<=66666)"
    con.execute(f"COPY (SELECT entity_id,country,n,a FROM chosen WHERE {where} ORDER BY entity_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",
                [str(OUTPUT / "queries.parquet")])
    count, unique, old_count = con.execute(f"SELECT count(*),count(DISTINCT entity_id),count(*) FILTER (WHERE old_sample) FROM chosen WHERE {where}").fetchone()
    if count != 200_000 or unique != count or old_count != 20_000:
        raise ValueError(f"Unexpected sample coverage: {count}, {unique}, old={old_count}")
    by_fold_country = [dict(zip(("fold", "country", "rows", "old_rows"), row)) for row in con.execute(
        f"SELECT fold,country,count(*),count(*) FILTER(WHERE old_sample) FROM chosen WHERE {where} GROUP BY 1,2 ORDER BY 1,2").fetchall()]
    con.close()
    for name in ("name_char3_idf.npz", "address_char3_idf.npz"):
        shutil.copy2(FULL / name, OUTPUT / name)
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(OUTPUT.iterdir()) if path.is_file()]
    manifest = {"scope": "200k deterministic unlocked S1 queries; no labels in cloud input; Fold4 CLOSED",
                "experiment": "EXP-031", "query_count": count, "old_20k_coverage": old_count,
                "sample_rule": "per-fold old 20k first, then SHA256(entity_id); fold quotas 66667/66667/66666",
                "by_fold_country": by_fold_country, "source_full_query_sha256": sha256(FULL / "queries.parquet"),
                "source_20k_sample_sha256": sha256(SAMPLE_B), "files": files}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"query_count": count, "old_20k_coverage": old_count,
                      "by_fold_country": by_fold_country, "output": str(OUTPUT)}), flush=True)


if __name__ == "__main__":
    main()
