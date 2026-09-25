"""Prepare the exact frozen test representation from the official raw TSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import duckdb


EXPECTED_ROWS = {"queries": 1_732_544, "target2": 4_887_273, "target3": 5_082_316}
HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
FROZEN_FILES = (
    "name_map.parquet", "name_char3_idf.npz", "address_char3_idf.npz",
    "model.txt", "submission-config.json", "train-manifest.json",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = json.loads((args.frozen / "manifest.json").read_text())
    by_name = {item["name"]: item for item in manifest["files"]}
    if not set(FROZEN_FILES) <= by_name.keys():
        raise ValueError("Frozen artifact inventory is incomplete")
    for name in FROZEN_FILES:
        source = args.frozen / name
        if digest(source) != by_name[name]["sha256"]:
            raise ValueError(f"Frozen artifact checksum differs: {name}")
        shutil.copy2(source, args.output / name)
    con = duckdb.connect(config={"threads": 4, "memory_limit": "4GB"})
    con.execute("CREATE TEMP MACRO norm(x) AS trim(regexp_replace(lower(nfc_normalize(coalesce(x,''))), '[^\\p{L}\\p{M}\\p{N}]+', ' ', 'g'))")
    sources = {}
    for output_name, source_name in (("queries", "test_source1"), ("target2", "test_source2"), ("target3", "test_source3")):
        source = args.test_dir / f"{source_name}.tsv"
        with source.open("r", encoding="utf-8", newline="") as stream:
            if stream.readline() != HEADER:
                raise ValueError(f"Unexpected columns in {source}")
        target = args.output / f"{output_name}.parquet"
        con.execute(
            "COPY (SELECT entity_id,country,norm(business_name) AS n,norm(business_address) AS a "
            f"FROM read_csv({sql_literal(source)},delim='\\t',header=true,all_varchar=true,"
            "nullstr='__AUDIT_IMPOSSIBLE_NULL_SENTINEL__',strict_mode=true) ORDER BY entity_id) "
            f"TO {sql_literal(target)} (FORMAT PARQUET,COMPRESSION ZSTD)"
        )
        count, unique = con.execute("SELECT count(*),count(DISTINCT entity_id) FROM read_parquet(?)", [str(target)]).fetchone()
        if count != EXPECTED_ROWS[output_name] or unique != count:
            raise ValueError(f"Unexpected row or unique-ID count in {source}: {count}, {unique}")
        sources[output_name] = {"rows": count, "raw_sha256": digest(source), "parquet_sha256": digest(target)}
        print(json.dumps({"source": output_name, "rows": count}), flush=True)
    con.close()
    (args.output / "manifest.json").write_text(json.dumps({"scope": "SUB-001 reproduction inputs", "sources": sources,
                                                      "frozen_model_sha256": by_name["model.txt"]["sha256"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
