"""Losslessly reorder an already validated leaderboard TSV to test-source order."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reorder(original: Path, test_source1: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    with original.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", strict=True)
        if next(reader) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Original schema differs from official README")
        predictions: dict[str, str] = {}
        for row in reader:
            if len(row) != 2 or row[0] in predictions:
                raise ValueError("Malformed or duplicate prediction row")
            predictions[row[0]] = row[1]
    original_count = len(predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with test_source1.open("r", encoding="utf-8", newline="") as source, destination.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        test_reader = csv.reader(source, delimiter="\t", strict=True)
        if next(test_reader) != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError("Unexpected test source schema")
        output.write("source1_entity_id\tmatched_entity_ids\n")
        count = 0
        for row in test_reader:
            if len(row) != 4:
                raise ValueError("Malformed test source row")
            query_id = row[0]
            if query_id not in predictions:
                raise ValueError(f"Missing prediction for {query_id}")
            output.write(query_id + "\t" + predictions.pop(query_id) + "\n")
            count += 1
    if predictions or count != original_count:
        raise ValueError(f"Extra predictions: {len(predictions)}")
    return {"original": str(original.resolve()), "original_sha256": sha256(original),
            "destination": str(destination.resolve()), "destination_sha256": sha256(destination),
            "rows": count, "value_changes": 0, "transformation": "row order only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--test-source1", type=Path,
                        default=Path("student_resource/dataset/test/test_source1.tsv"))
    args = parser.parse_args()
    result = reorder(args.original, args.test_source1, args.destination)
    manifest = args.destination.parent / "REORDER_MANIFEST.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
