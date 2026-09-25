"""Strict, streaming validation of the challenge's leaderboard TSV.

Checks the exact README schema against the original test source files. Use
``--require-test-order`` for the defensive test-order variant; row ordering is
not stated as a requirement in the official README.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


HEADER = b"source1_entity_id\tmatched_entity_ids\n"
ID1 = re.compile(r"S1-[0-9]+\Z")
ID23 = re.compile(r"S[23]-[0-9]+\Z")


def source_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", strict=True)
        header = next(reader)
        if header != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError(f"Unexpected test source schema: {path}: {header}")
        seen: set[str] = set()
        output: list[str] = []
        pattern = ID1 if path.name == "test_source1.tsv" else ID23
        for number, row in enumerate(reader, 2):
            if len(row) != 4 or not pattern.fullmatch(row[0]):
                raise ValueError(f"Bad test row {path}:{number}")
            if row[0] in seen:
                raise ValueError(f"Duplicate test ID {row[0]}")
            seen.add(row[0])
            output.append(row[0])
    return output


def validate(path: Path, test_dir: Path, require_test_order: bool = False) -> dict:
    if path.name != "matching_results.tsv" or path.suffix != ".tsv":
        raise ValueError("Leaderboard file must be named matching_results.tsv")
    expected_order = source_ids(test_dir / "test_source1.tsv")
    expected = set(expected_order)
    targets = set(source_ids(test_dir / "test_source2.tsv"))
    targets.update(source_ids(test_dir / "test_source3.tsv"))
    if len(expected) != 1_732_544 or len(targets) != 9_969_589:
        raise ValueError("Unexpected competition test universe")
    seen: set[str] = set()
    digest = hashlib.sha256()
    first: list[list[str]] = []
    last: list[list[str]] = []
    empty = links = rows = out_of_test_order = 0
    previous = ""
    with path.open("rb") as stream:
        header = stream.readline()
        digest.update(header)
        if header != HEADER:
            raise ValueError(f"Wrong header, BOM, delimiter, or newline: {header!r}")
        for number, raw in enumerate(stream, 2):
            digest.update(raw)
            if not raw.endswith(b"\n") or raw.endswith(b"\r\n") or raw.count(b"\t") != 1 or b"\x00" in raw:
                raise ValueError(f"Malformed physical line {number}")
            left, right = raw[:-1].decode("utf-8", errors="strict").split("\t")
            if not ID1.fullmatch(left) or left not in expected or left in seen or left != left.strip():
                raise ValueError(f"Invalid, unknown or duplicate S1 ID at line {number}: {left!r}")
            seen.add(left)
            if left != expected_order[rows]:
                out_of_test_order += 1
            if previous and left <= previous:
                # The supplied original is sorted. A test-order file need not be.
                pass
            previous = left
            if right:
                parts = right.split(",")
                if (len(parts) != len(set(parts)) or any(not ID23.fullmatch(item) or item not in targets
                                                       or item != item.strip() for item in parts)):
                    raise ValueError(f"Invalid, duplicate or unknown target at line {number}")
                links += len(parts)
            else:
                empty += 1
            if len(first) < 10:
                first.append([left, right])
            last.append([left, right])
            if len(last) > 10:
                last.pop(0)
            rows += 1
    if rows != len(expected_order) or seen != expected:
        raise ValueError(f"Test S1 reconciliation failed: rows={rows}, missing={len(expected-seen)}, extra={len(seen-expected)}")
    if require_test_order and out_of_test_order:
        raise ValueError(f"{out_of_test_order} rows differ from original test order")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(), "physical_lines": rows + 1,
            "parsed_rows": rows, "columns": ["source1_entity_id", "matched_entity_ids"],
            "dtypes": ["string", "string"], "null_required_ids": 0,
            "empty_match_fields": empty, "matched_links": links,
            "duplicate_source_ids": 0, "missing_source_ids": 0,
            "unexpected_source_ids": 0, "invalid_target_ids": 0,
            "out_of_test_order_rows": out_of_test_order,
            "test_order_required_by_readme": False,
            "first_10": first, "last_10": last,
            "schema": "PASS", "ids": "PASS", "raw_tsv": "PASS", "read_back": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matching", type=Path)
    parser.add_argument("--test-dir", type=Path, default=Path("student_resource/dataset/test"))
    parser.add_argument("--require-test-order", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.matching, args.test_dir, args.require_test_order)
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
