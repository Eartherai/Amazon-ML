"""Export bounded, label-safe inputs for the 200k classical feature store."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import duckdb


ROOT = Path("artifacts/cloud/phase5")
OUTPUT = ROOT / "feature-200k-input-v001"
SAMPLE = ROOT / "learning-200k-input-v001" / "queries.parquet"
TARGETS = ROOT / "index-input-v001" / "targets.parquet"
TRANSLITERATION = Path("artifacts/transliteration/TRANS-001/name_map.parquet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if shutil.disk_usage(".").free < 9 * 1024**3:
        raise RuntimeError("Insufficient local disk reserve")
    OUTPUT.mkdir(parents=True)
    for source, name in ((SAMPLE, "queries.parquet"), (TARGETS, "targets.parquet"),
                         (TRANSLITERATION, "name_map.parquet")):
        shutil.copy2(source, OUTPUT / name)
    con = duckdb.connect("artifacts/audit.duckdb", read_only=True,
                         config={"threads": 2, "memory_limit": "3GB"})
    con.execute("COPY (SELECT target_id,owner_fold FROM target_ownership ORDER BY target_id) "
                "TO ? (FORMAT PARQUET,COMPRESSION ZSTD)", [str(OUTPUT / "ownership.parquet")])
    con.execute(f"COPY (SELECT q.entity_id,q.country,q.n,q.a,v.fold,v.n_matches FROM read_parquet({literal(SAMPLE)}) q "
                "JOIN validation_folds v ON v.source1_entity_id=q.entity_id "
                "WHERE v.fold IN (1,2,3) ORDER BY q.entity_id) "
                f"TO {literal(OUTPUT / 'labeled_queries.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)")
    con.execute("COPY (SELECT p.source1_entity_id,p.target_id FROM positive_pairs p "
                f"JOIN read_parquet({literal(SAMPLE)}) q ON q.entity_id=p.source1_entity_id "
                "ORDER BY p.source1_entity_id,p.target_id) "
                f"TO {literal(OUTPUT / 'truth.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)")
    query_count = con.execute("SELECT count(*) FROM read_parquet(?)", [str(OUTPUT / "labeled_queries.parquet")]).fetchone()[0]
    fold4_count = con.execute("SELECT count(*) FROM read_parquet(?) WHERE fold=4", [str(OUTPUT / "labeled_queries.parquet")]).fetchone()[0]
    target_count = con.execute("SELECT count(*) FROM read_parquet(?)", [str(OUTPUT / "targets.parquet")]).fetchone()[0]
    ownership_count = con.execute("SELECT count(*) FROM read_parquet(?)", [str(OUTPUT / "ownership.parquet")]).fetchone()[0]
    truth_count = con.execute("SELECT count(*) FROM read_parquet(?)", [str(OUTPUT / "truth.parquet")]).fetchone()[0]
    if (query_count, fold4_count, target_count, ownership_count) != (200_000, 0, 10_320_219, 10_320_219):
        raise ValueError("Feature input coverage mismatch")
    con.close()
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(OUTPUT.iterdir()) if path.is_file()]
    manifest = {"scope": "EXP-032 classical feature store for 200k sampled unlocked S1; no Fold4 query labels",
                "query_count": query_count, "target_count": target_count, "ownership_count": ownership_count,
                "truth_pairs": truth_count, "source_sample_sha256": sha256(SAMPLE), "files": files}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"queries": query_count, "targets": target_count,
                      "ownership": ownership_count, "truth_pairs": truth_count,
                      "bytes": sum(item["bytes"] for item in files)}), flush=True)


if __name__ == "__main__":
    main()
