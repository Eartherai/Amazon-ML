"""Apply OOF-selected source completion and target ownership to SUB-002.

No labels or external data are read. Scores are from the exact inference pass
that wrote the candidate and matching shards. Keep the original file intact.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


SOURCE_THRESHOLD = 0.55
MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
SCORES_HEADER = "source1_entity_id\ttarget_id\tscore\n"


def read_scores(paths: list[Path]):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
            if source.readline() != SCORES_HEADER:
                raise ValueError(f"Wrong score header: {path}")
            for line in source:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 3:
                    raise ValueError(f"Malformed score row: {path}")
                qid, tid, raw = fields
                score = float(raw)
                if not 0 <= score <= 1 or not math.isfinite(score):
                    raise ValueError(f"Invalid score: {path}")
                yield qid, tid, score


def postprocess(matching: Path, score_dir: Path, output: Path, expected: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    files = sorted(score_dir.glob("*-scores.tsv.gz"))
    if len(files) != 192:
        raise ValueError(f"Expected 192 score shards; got {len(files)}")
    order: list[str] = []
    predictions: dict[str, set[str]] = {}
    first_owner: dict[str, str] = {}
    conflicts: dict[str, list[str]] = {}
    missing_source: dict[str, str] = {}
    original_links = 0
    with matching.open("r", encoding="utf-8", newline="") as source:
        if source.readline() != MATCHING_HEADER:
            raise ValueError("Wrong matching header")
        for line in source:
            qid, sep, raw = line.rstrip("\n").partition("\t")
            if not sep or not qid.startswith("S1-") or qid in predictions:
                raise ValueError("Malformed or duplicate Source 1 row")
            links = raw.split(",") if raw else []
            if len(links) != len(set(links)) or any(not t.startswith(("S2-", "S3-")) for t in links):
                raise ValueError("Invalid matching ID list")
            order.append(qid)
            predictions[qid] = set(links)
            original_links += len(links)
            if links:
                sources = {tid[:2] for tid in links}
                if len(sources) == 1:
                    missing_source[qid] = "S3" if "S2" in sources else "S2"
            for tid in links:
                owner = first_owner.setdefault(tid, qid)
                if owner != qid:
                    conflicts.setdefault(tid, [owner]).append(qid)
    if len(order) != expected:
        raise ValueError(f"Wrong Source 1 count: {len(order)}")

    best: dict[str, tuple[float, str]] = {}
    pair_score: dict[tuple[str, str], float] = {}
    for qid, tid, score in read_scores(files):
        absent = missing_source.get(qid)
        if absent and tid.startswith(absent):
            current = best.get(qid)
            if current is None or (score, tid) > current:
                best[qid] = (score, tid)
        if tid in conflicts and qid in conflicts[tid]:
            pair_score[qid, tid] = score

    additions = 0
    for qid, (score, tid) in best.items():
        if score < SOURCE_THRESHOLD or tid in predictions[qid]:
            continue
        predictions[qid].add(tid)
        pair_score[qid, tid] = score
        additions += 1
        owner = first_owner.setdefault(tid, qid)
        if owner != qid:
            conflicts.setdefault(tid, [owner]).append(qid)

    needed = {(qid, tid) for tid, owners in conflicts.items() for qid in owners} - pair_score.keys()
    if needed:
        for qid, tid, score in read_scores(files):
            if (qid, tid) in needed:
                pair_score[qid, tid] = score
        if needed - pair_score.keys():
            raise ValueError("Missing model scores for predicted conflict links")
    removed = 0
    for tid, owners in conflicts.items():
        if len(owners) != len(set(owners)):
            raise ValueError("Duplicate target owner")
        winner = max(owners, key=lambda qid: (pair_score[qid, tid], qid))
        for qid in owners:
            if qid != winner:
                predictions[qid].remove(tid)
                removed += 1

    final_links = original_links + additions - removed
    with output.open("x", encoding="utf-8", newline="") as target:
        target.write(MATCHING_HEADER)
        written_links = 0
        for qid in order:
            ids = sorted(predictions[qid])
            written_links += len(ids)
            target.write(qid + "\t" + ",".join(ids) + "\n")
    if written_links != final_links:
        raise ValueError("Postprocess link accounting differs")
    report = {"rows": len(order), "source_threshold": SOURCE_THRESHOLD,
              "original_links": original_links, "single_source_queries": len(missing_source),
              "eligible_missing_source_candidates": len(best), "added_links_before_ownership": additions,
              "conflict_targets": len(conflicts), "owner_links_removed": removed,
              "final_links": final_links, "score_shards": len(files)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1_732_544)
    args = parser.parse_args()
    report = postprocess(args.matching, args.scores, args.output, args.expected)
    (args.output.parent / "postprocess_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
