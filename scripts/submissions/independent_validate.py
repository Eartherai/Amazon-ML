"""Independently verify raw SUB-001 TSV alignment, S1 universe and membership.

This is separate from the supplied official validator. It streams the full
matching/candidate files and compares them to the frozen test query IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import polars as pl


HEADERS = (b"source1_entity_id\tmatched_entity_ids\n",
           b"source1_entity_id\tcandidate_entity_ids\n")


def validate(matching: Path, candidates: Path, queries: Path, expected_rows: int) -> dict:
    frame = pl.read_parquet(queries, columns=["entity_id"])
    expected = set(frame["entity_id"].to_list())
    if len(frame) != expected_rows or len(expected) != expected_rows:
        raise ValueError("Frozen S1 universe is incomplete or duplicated")
    mh, ch = hashlib.sha256(), hashlib.sha256()
    counts = {"rows": 0, "empty_predictions": 0, "predicted_links": 0,
              "candidate_links": 0}
    with matching.open("rb") as m, candidates.open("rb") as c:
        mheader, cheader = m.readline(), c.readline()
        mh.update(mheader)
        ch.update(cheader)
        if (mheader, cheader) != HEADERS:
            raise ValueError("TSV headers do not match official schema")
        previous = ""
        for mline, cline in zip(m, c, strict=True):
            mh.update(mline)
            ch.update(cline)
            if mline.count(b"\t") != 1 or cline.count(b"\t") != 1 or not mline.endswith(b"\n") or not cline.endswith(b"\n"):
                raise ValueError("Malformed tab-separated row")
            mid, matched = mline[:-1].decode("utf-8").split("\t")
            cid, candidate = cline[:-1].decode("utf-8").split("\t")
            if mid != cid or (previous and mid <= previous):
                raise ValueError("Output rows are misaligned, duplicate or unsorted")
            if mid not in expected:
                raise ValueError(f"Unknown or duplicate S1 ID: {mid}")
            expected.remove(mid)
            candidate_ids = candidate.split(",") if candidate else []
            matched_ids = matched.split(",") if matched else []
            candidate_set = set(candidate_ids)
            if (len(candidate_set) != len(candidate_ids) or len(set(matched_ids)) != len(matched_ids)
                    or not set(matched_ids) <= candidate_set):
                raise ValueError(f"Duplicate or unscored match for {mid}")
            if any(not tid.startswith(("S2-", "S3-")) for tid in candidate_ids):
                raise ValueError(f"Candidate target ID has invalid source prefix for {mid}")
            counts["rows"] += 1
            counts["empty_predictions"] += not matched_ids
            counts["predicted_links"] += len(matched_ids)
            counts["candidate_links"] += len(candidate_ids)
            previous = mid
    if counts["rows"] != expected_rows or expected:
        raise ValueError(f"Missing S1 rows: observed={counts['rows']}, missing={len(expected)}")
    return {**counts, "matching_sha256": mh.hexdigest(), "candidate_sha256": ch.hexdigest(),
            "frozen_queries": str(queries.resolve()), "independent_pass": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path(
        "artifacts/cloud/phase5/sub001-input-v002/queries.parquet"))
    parser.add_argument("--expected-rows", type=int, default=1732544)
    args = parser.parse_args()
    ready = json.loads((args.validated_dir / "READY.json").read_text())
    if ready["rows"] != args.expected_rows:
        raise ValueError("Official result and independent expected count disagree")
    report = validate(args.validated_dir / "matching_results.tsv",
                      args.validated_dir / "candidate_pairs.tsv", args.queries, args.expected_rows)
    if (report["matching_sha256"] != ready["matching_sha256"]
            or report["candidate_sha256"] != ready["candidate_sha256"]):
        raise ValueError("Independent raw TSV hashes disagree with official result")
    target = args.validated_dir / "INDEPENDENT_VALIDATION.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
